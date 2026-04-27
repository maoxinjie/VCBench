import torch
import torch.nn as nn
from typing import Iterable, Optional, Sequence


class MLP(nn.Module):
    """Simple MLP block reused by modality encoders and the final fusion layer."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout: float = 0.0,
            activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        dims = [input_dim] + hidden_dims + [output_dim]

        layers: Sequence[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PertAggregator(nn.Module):
    """Aggregate multiple perturbation embeddings (e.g., multi-gene perturbations) into a single vector
    
    Supports two input formats:
    1. Padded tensor: (batch, max_n, dim) + lengths (batch,) - Recommended, suitable for multiprocessing
    2. List[List[Tensor]] - Compatible with old code
    """
    
    def __init__(
            self,
            emb_dim: int,
            output_dim: int,
            hidden_dims: Optional[Iterable[int]] = None,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.mlp = MLP(emb_dim, output_dim, hidden_dims, dropout)

    def forward(self, pert_data, lengths=None) -> torch.Tensor:
        """
        Args:
            pert_data: Tensor (batch, max_n, dim) or List[List[Tensor]]
            lengths: Tensor (batch,) - Only needed when pert_data is a padded tensor
        Returns:
            Aggregated embedding (batch_size, output_dim)
        """
        # Determine input format
        if isinstance(pert_data, torch.Tensor):
            return self._forward_padded(pert_data, lengths)
        else:
            return self._forward_list(pert_data)
    
    def _forward_padded(self, pert_tensor, lengths):
        """Process padded tensor format (fast path)"""
        # pert_tensor: (batch, max_n, dim)
        # lengths: (batch,)
        batch_size, max_n, dim = pert_tensor.shape
        device = pert_tensor.device
        
        if lengths is None or lengths.sum() == 0:
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # Create mask: (batch, max_n)
        idx = torch.arange(max_n, device=device).unsqueeze(0)  # (1, max_n)
        mask = idx < lengths.unsqueeze(1)  # (batch, max_n)
        
        # Flatten valid embeddings (use reshape instead of view to avoid memory discontinuity issues)
        flat_emb = pert_tensor.reshape(-1, dim)  # (batch * max_n, dim)
        flat_mask = mask.reshape(-1)  # (batch * max_n,)
        valid_emb = flat_emb[flat_mask]  # (total_valid, dim)
        
        if valid_emb.shape[0] == 0:
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        # MLP processing
        valid_emb = self.mlp(valid_emb)  # (total_valid, output_dim)
        
        # Build batch indices for scatter_add (using reshape)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_n)  # (batch, max_n)
        valid_batch_idx = batch_idx.reshape(-1)[flat_mask]  # (total_valid,)
        
        # scatter_add aggregation
        agged = torch.zeros(batch_size, self.output_dim, device=device)
        idx_expanded = valid_batch_idx.unsqueeze(1).expand(-1, self.output_dim)
        agged.scatter_add_(0, idx_expanded, valid_emb)
        
        return agged
    
    def _forward_list(self, pert_batch):
        """Process List[List[Tensor]] format (compatible with old code)"""
        batch_size = len(pert_batch)
        device = next(self.mlp.parameters()).device
        
        total_perts = sum(len(pert_list) for pert_list in pert_batch)
        
        if total_perts == 0:
            return torch.zeros(batch_size, self.output_dim, device=device)
        
        stack_pert_emb = []
        pos_in_batch = torch.empty(total_perts, dtype=torch.long, device=device)
        
        idx = 0
        for batch_idx, pert_emb_list in enumerate(pert_batch):
            n = len(pert_emb_list)
            if n > 0:
                pos_in_batch[idx:idx + n] = batch_idx
                stack_pert_emb.extend(pert_emb_list)
                idx += n
        
        stack_pert_emb = torch.stack(stack_pert_emb).to(device)
        stack_pert_emb = self.mlp(stack_pert_emb)
        
        agged_pert_emb = torch.zeros(batch_size, self.output_dim, device=device)
        pos_expanded = pos_in_batch.unsqueeze(1).expand(-1, self.output_dim)
        agged_pert_emb.scatter_add_(0, pos_expanded, stack_pert_emb)
        
        return agged_pert_emb


class MixedPerturbationEncoder(nn.Module):
    """Mixed perturbation encoder: supports gene, drug, environment, CRISPR multi-modal perturbations"""

    def __init__(
            self,
            gene_pert_dim: int,
            drug_pert_dim: int,
            env_pert_dim: int,
            crispr_pert_dim: int = 1,
            hidden_dims: Optional[Iterable[int]] = None,
            per_modality_embed_dim: int|None = None,
            final_embed_dim: int | None = None,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        assert per_modality_embed_dim is not None or final_embed_dim is not None,\
             "per_modality_embed_dim and final_embed_dim must be provided"
        if final_embed_dim is None:
            final_embed_dim = per_modality_embed_dim
        if per_modality_embed_dim is None:
            per_modality_embed_dim = final_embed_dim
        self.per_modality_embed_dim = per_modality_embed_dim
        self.final_embed_dim = final_embed_dim

        self.fusion_mlp = MLP(self.per_modality_embed_dim, final_embed_dim)
        # Conditionally create encoders
        self.gene_encoder = PertAggregator(gene_pert_dim, self.per_modality_embed_dim, hidden_dims, dropout=dropout)\
             if gene_pert_dim > 1 else None
        self.drug_encoder = PertAggregator(drug_pert_dim, self.per_modality_embed_dim, hidden_dims, dropout=dropout)\
             if drug_pert_dim > 1 else None
        self.env_encoder = MLP(env_pert_dim, self.per_modality_embed_dim, hidden_dims, dropout=dropout)\
             if env_pert_dim > 1 else None
        self.crispr_encoder = MLP(crispr_pert_dim, self.per_modality_embed_dim, hidden_dims, dropout=dropout)\
             if crispr_pert_dim > 1 else None

        # Cache encoder existence (avoid repeated checks in forward)
        self._has_gene = self.gene_encoder is not None
        self._has_drug = self.drug_encoder is not None
        self._has_env = self.env_encoder is not None
        self._has_crispr = self.crispr_encoder is not None

    def forward(self, batch) -> torch.Tensor:
        """
        Supports two input formats:
        1. Padded tensor: batch.gene_pert (B, max_n, dim) + batch.gene_pert_len (B,)
        2. List format: batch.gene_pert = List[List[Tensor]] (compatible with old code)
        """
        # gene embedding
        if self._has_gene and hasattr(batch, 'gene_pert'):
            gene_len = getattr(batch, 'gene_pert_len', None)
            gene_pert_emb = self.gene_encoder(batch.gene_pert, gene_len)
        else:
            gene_pert_emb = 0
        
        # drug embedding
        if self._has_drug and hasattr(batch, 'drug_pert'):
            drug_len = getattr(batch, 'drug_pert_len', None)
            drug_pert_emb = self.drug_encoder(batch.drug_pert, drug_len)
        else:
            drug_pert_emb = 0
        
        # env/crispr (single value, no aggregation needed)
        env_pert_emb = self.env_encoder(batch.env_pert) if self._has_env and hasattr(batch, 'env_pert') else 0
        crispr_pert_emb = self.crispr_encoder(batch.crispr_pert) if self._has_crispr and hasattr(batch, 'crispr_pert') else 0

        final_pert_emb = gene_pert_emb + drug_pert_emb + env_pert_emb + crispr_pert_emb

        return self.fusion_mlp(final_pert_emb)  # (B, final_embed_dim)