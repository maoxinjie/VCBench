"""
scLAMBDA Model for VCBench

Adapted from the original scLAMBDA implementation:
https://github.com/Bunnybeibei/scLAMBDA

scLAMBDA uses:
1. VAE for gene expression encoding (basal state z)
2. Deterministic encoder for perturbation embeddings (perturbation effect s)
3. Additive latent model: z + s
4. MINE (Mutual Information Neural Estimator) for disentanglement
5. Adversarial training on perturbation embeddings

Key differences from original:
- Integrated into PyTorch Lightning framework
- Uses VCBench's Batch data structure
- Handles gene embeddings from external sources (GenePT)
- Avoids deadlocks with proper DataLoader configuration
"""

import torch
from torch.optim.lr_scheduler import StepLR
import lightning as L
import numpy as np
import pickle
import logging
from pathlib import Path

from ..nn.sclambda_networks import scLAMBDANet
from ..nn import MixedPerturbationEncoder
from .base import PerturbationModel

log = logging.getLogger(__name__)


class scLAMBDA(PerturbationModel):
    """
    scLAMBDA: Single-Cell Latent AMBA (Additive Model with Biological Annotations)

    A VAE-based model that disentangles basal cell state from perturbation effects
    using mutual information minimization and adversarial training.
    """

    def __init__(
        self,
        gene_embedding_path: str | None = None,
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        use_covs: bool = False,  # Unified covariate usage parameter
        latent_dim: int = 30,
        hidden_dim: int = 512,
        lambda_MI: float = 200.0,
        eps: float = 0.001,
        lr: float = 5e-4,
        wd: float = 1e-4,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        lr_step_size: int = 30,
        lr_gamma: float = 0.2,
        # ============= [NEW] Unified scheduler params - for switching between onecycle/plateau/step
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        # ============= [NEW] End
        datamodule: L.LightningDataModule | None = None,
        perturbation_combination_delimiter: str = "+",
        perturbation_control_value: str = "control",
        seed: int | None = None,
            **kwargs
    ):
        """
        Initialize scLAMBDA model.

        Args:
            n_genes: Number of genes
            n_perts: Number of perturbations
            gene_embedding_path: Path to gene embedding pickle file (e.g., GenePT embeddings)
            latent_dim: Dimension of latent space (z and s)
            hidden_dim: Hidden layer dimension
            lambda_MI: Weight for mutual information loss
            eps: Epsilon for adversarial perturbation
            lr: Learning rate
            wd: Weight decay
            lr_step_size: Step size for learning rate scheduler
            lr_gamma: Gamma for learning rate scheduler
            datamodule: Lightning data module
            perturbation_combination_delimiter: Delimiter for combination perturbations
            perturbation_control_value: Control condition name
            seed: Random seed (handled by PyTorch Lightning, included for compatibility)
        """

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        self.use_covs = use_covs

        super(scLAMBDA, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            # =================== [NEW] Pass unified scheduler params to parent
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            # =================== [NEW] End
            use_mask=use_mask,  # Pass use_mask to base class
        )


        # Save hyperparameters (avoid saving datamodule to prevent serialization issues)
        self.save_hyperparameters(ignore=["datamodule"])

        # Store datamodule reference (needed for setup())
        self.datamodule = datamodule

        # Manual optimization for adversarial training
        self.automatic_optimization = False

        self.use_mix_pert = True
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lambda_MI = lambda_MI
        self.eps = eps
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.perturbation_combination_delimiter = perturbation_combination_delimiter
        self.perturbation_control_value = perturbation_control_value

        # Initialize perturbation encoder/embeddings
        if self.use_mix_pert:
            pert_embed_dim = self.latent_dim
            self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                crispr_pert_dim=self.crispr_pert_dim,
                final_embed_dim=pert_embed_dim,
            )
            self.p_dim = pert_embed_dim
            self.gene_emb = None
        else:
            self.pert_encoder = None
            # Load gene embeddings
            self.gene_embedding_path = gene_embedding_path
            self.gene_emb = self._load_gene_embeddings(gene_embedding_path)
            self.p_dim = self.gene_emb[list(self.gene_emb.keys())[0]].shape[0]

            # Add control embedding (zeros)
            self.gene_emb[perturbation_control_value] = np.zeros(self.p_dim)

        # Adjust p_dim if covariates are used (covariates get concatenated to p for encoder input)
        p_dim_encoder = self.p_dim
        if self.use_covs and hasattr(self, 'cov_dims') and self.cov_dims:
            p_dim_encoder += sum(self.cov_dims.values())

        # Store original p_dim for decoder output
        self.p_dim_original = self.p_dim

        # Initialize network
        self.Net = scLAMBDANet(
            input_dim=self.n_genes,
            output_dim=self.n_genes,
            p_dim=p_dim_encoder,  # Encoder input dimension (includes covariates)
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            p_dim_decoder=self.p_dim_original  # Decoder output dimension (original p_dim)
        )

        # Control statistics (will be computed in setup())
        self.register_buffer("ctrl_mean", torch.zeros(self.n_genes ))

    def _load_gene_embeddings(self, embedding_path: str | None) -> dict:
        """Load gene embeddings from pickle file."""
        if embedding_path is None:
            raise ValueError(
                "gene_embedding_path is not set. "
                "Please pass --model.gene_embedding_path=<path_to_GenePT_gene_embedding_ada_text.pickle>."
            )

        embedding_path = Path(embedding_path)
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Gene embedding file not found: {embedding_path}\n"
                f"Please provide the path to GenePT_gene_embedding_ada_text.pickle"
            )

        log.info(f"Loading gene embeddings from {embedding_path}")
        with open(embedding_path, "rb") as f:
            gene_emb = pickle.load(f)

        # Convert to numpy arrays if needed (GenePT embeddings are lists)
        first_gene = list(gene_emb.keys())[0]
        if not isinstance(gene_emb[first_gene], np.ndarray):
            log.info("Converting embeddings to numpy arrays")
            gene_emb = {k: np.array(v, dtype=np.float32) for k, v in gene_emb.items()}

        # Add control embedding (zero vector) if not present
        if self.perturbation_control_value not in gene_emb:
            emb_dim = gene_emb[first_gene].shape[0]
            gene_emb[self.perturbation_control_value] = np.zeros(emb_dim, dtype=np.float32)
            log.info(f"Added zero embedding for control: {self.perturbation_control_value}")

        log.info(f"Loaded embeddings for {len(gene_emb)} genes, "
                f"embedding dimension: {gene_emb[first_gene].shape[0]}")

        return gene_emb

    def _compute_perturbation_embedding(self, perturbation_names: list[str]|np.ndarray) -> torch.Tensor:
        """
        Compute perturbation embeddings from gene names.

        For combination perturbations (e.g., "GENE1+GENE2"), sum the individual embeddings.

        Args:
            perturbation_names: List of perturbation names

        Returns:
            Tensor of perturbation embeddings [batch_size, p_dim]
        """
        if self.gene_emb is None:
            raise RuntimeError("Gene embeddings are not initialized when using MixedPerturbationEncoder.")

        batch_size = len(perturbation_names)
        pert_emb_batch = np.zeros((batch_size, self.p_dim))

        for i, pert_name in enumerate(perturbation_names):
            if pert_name in [self.perturbation_control_value, "ctrl"]:
                # Control: zero embedding
                pert_emb_batch[i] = self.gene_emb[self.perturbation_control_value]
            else:
                # Split by delimiter for combination perturbations
                genes = pert_name.split(self.perturbation_combination_delimiter)
                pert_emb = np.zeros(self.p_dim)
                for gene in genes:
                    gene = gene.strip()
                    if gene in self.gene_emb:
                        pert_emb += self.gene_emb[gene]
                    else:
                        log.warning(f"Gene {gene} not found in embeddings, using zero vector")
                pert_emb_batch[i] = pert_emb

        return torch.from_numpy(pert_emb_batch).float().to(self.device)

    def setup(self, stage: str | None = None) -> None:
        """
        Compute control mean before training starts.

        This is called once per rank (GPU) by PyTorch Lightning.
        We compute the mean from the datamodule's control cells,
        ensuring it's consistent across all DDP ranks.
        """
        if stage in ("fit", "validate", None):
            if self.datamodule is None:
                log.warning("Datamodule is None. Cannot compute control mean.")
                return

            log.info("Computing control cell mean...")

            try:
                # Ensure datamodule is set up
                if not hasattr(self.datamodule, 'train_dataset'):
                    self.datamodule.setup(stage='fit')

                # Get control expression from dataset
                train_dataset = self.datamodule.train_dataset

                # Check if dataset has controls (SingleCellPerturbationWithControls)

                ctrl_x = train_dataset.control_adata.X
                ctrl_x_mean = ctrl_x.mean(axis=0)

                self.ctrl_mean = torch.tensor(ctrl_x_mean, dtype=torch.float32)
                log.info(f"Computed control mean from {ctrl_x.shape[0]} control cells")


            except Exception as e:
                log.error(f"Failed to compute control mean: {e}")
                log.warning("Using zero vector as control mean.")

    def unpack_batch(self, batch):
        """Unpack batch and prepare inputs for scLAMBDA."""
        # Get gene expression
        x = batch.pert_cell_counts
        if self.use_mix_pert:
            p = self.pert_encoder(batch)  # [batch_size, p_dim]
            pert_names = None
        else:
            pert_names = batch.perturbation
            p = self._compute_perturbation_embedding(pert_names)  # [batch_size, p_dim]

        # Get covariates if enabled
        covariates = None
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}

        return x, p, pert_names, covariates

    def forward(self, x: torch.Tensor, p: torch.Tensor, covariates: dict[str, torch.Tensor] | None = None):
        """
        Forward pass through scLAMBDA network.

        Args:
            x: Gene expression [batch_size, n_genes]
            p: Perturbation embeddings [batch_size, p_dim]
            covariates: Dictionary of covariate tensors (optional)

        Returns:
            Dictionary with network outputs
        """
        # Store original p for loss computation (before covariates are added)
        p_original = p.clone()  # Clone to ensure it's not affected by subsequent modifications

        # Handle covariates if enabled and provided
        if self.use_covs and covariates:
            # Concatenate all covariate embeddings to perturbation embeddings
            cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
            if cov_tensors:
                merged_covariates = torch.cat(cov_tensors, dim=-1)
                p = torch.cat([p, merged_covariates], dim=-1)

        # Center by control mean
        x_centered = x - self.ctrl_mean.unsqueeze(0)

        # Forward through network
        x_hat, p_hat, mean_z, log_var_z, s = self.Net(x_centered, p)

        return {
            "x_hat": x_hat,
            "p_hat": p_hat,
            "p_original": p_original,
            "mean_z": mean_z,
            "log_var_z": log_var_z,
            "s": s,
            "x_centered": x_centered,
        }

    def loss_function(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        p: torch.Tensor,
        p_hat: torch.Tensor,
        mean_z: torch.Tensor,
        log_var_z: torch.Tensor,
        s: torch.Tensor,
        s_marginal: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """Compute scLAMBDA loss: Reconstruction + KL + λ_MI * MI.

        Args:
            mask: Optional expression mask for masked loss calculation on x reconstruction.
        """
        # Reconstruction loss for x (MSE) - with optional mask
        if mask is not None:
            # Masked reconstruction loss for x - unified per-batch calculation
            mse_x = (x_hat - x) ** 2  # [batch_size, n_genes]
            mask = mask.to(mse_x.device)
            valid = mask.sum(dim=1)  # [batch_size] - per-batch valid gene count
            loss_per_batch = (mse_x * mask).sum(dim=1)  # [batch_size] - per-batch loss
            recon_loss_x = 0.5 * (loss_per_batch / valid).nanmean()
        else:
            # Standard reconstruction loss
            recon_loss_x = 0.5 * torch.mean(torch.sum((x_hat - x) ** 2, dim=1))

        # Reconstruction loss for p (perturbation reconstruction, no mask needed)
        recon_loss_p = 0.5 * torch.mean(torch.sum((p_hat - p) ** 2, dim=1))
        recon_loss = recon_loss_x + recon_loss_p

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1)
        )

        # Mutual information loss (MINE)
        T_joint = self.Net.MINE(mean_z, s.detach())
        T_marginal = self.Net.MINE(mean_z, s_marginal.detach())
        mi_loss = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))

        # Total loss
        total_loss = recon_loss + kl_loss + self.lambda_MI * mi_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "recon_loss_x": recon_loss_x,
            "recon_loss_p": recon_loss_p,
            "kl_loss": kl_loss,
            "mi_loss": mi_loss,
        }

    def loss_recon(self, x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor = None):
        """Reconstruction loss only (for adversarial training).

        Args:
            mask: Optional expression mask for masked loss calculation.
        """
        return self.auto_mse(x_hat, x, mask)*0.5
    def loss_MINE(
        self,
        mean_z: torch.Tensor,
        s: torch.Tensor,
        s_marginal: torch.Tensor
    ):
        """MINE loss (negative mutual information)."""
        T_joint = self.Net.MINE(mean_z, s)
        T_marginal = self.Net.MINE(mean_z, s_marginal)
        mi = torch.mean(T_joint) - torch.log(torch.mean(torch.exp(T_marginal)))
        return -mi

    def training_step(self, batch, batch_idx: int):
        """Training step with adversarial training and MINE optimization."""
        # Get optimizers
        optimizer_main, optimizer_MINE = self.optimizers()

        x, p, _, covariates = self.unpack_batch(batch)

        # Get expression mask if available - use unified method from base class
        mask = self._get_mask(batch)

        # Adversarial training on perturbation embeddings
        p_for_adv = p
        if p_for_adv.is_leaf and not p_for_adv.requires_grad:
            p_for_adv = p_for_adv.clone().detach().requires_grad_(True)
        self.Net.eval()

        with torch.enable_grad():
            outputs = self.forward(x, p_for_adv, covariates)
            recon_loss = self.loss_recon(outputs["x_centered"], outputs["x_hat"], mask=mask)
            grads = torch.autograd.grad(recon_loss, p_for_adv, create_graph=False, retain_graph=False)[0]
            p_ae = p_for_adv + self.eps * torch.norm(p_for_adv, dim=1, keepdim=True) * torch.sign(grads.data)

        if not self.use_mix_pert:
            p_ae = p_ae.detach()
        p_target = p.detach()

        # Forward pass with adversarial perturbations
        self.Net.train()
        outputs = self.forward(x, p_ae, covariates)

        # Sample marginal perturbations for MINE
        batch_size = x.shape[0]
        marginal_indices = torch.randperm(batch_size, device=self.device)
        s_marginal = outputs["s"][marginal_indices]

        # Train MINE
        for _ in range(1):
            optimizer_MINE.zero_grad()
            mine_loss = self.loss_MINE(
                outputs["mean_z"].detach(),
                outputs["s"].detach(),
                s_marginal.detach()
            )
            self.manual_backward(mine_loss)
            optimizer_MINE.step()

        # Train main network
        optimizer_main.zero_grad()
        losses = self.loss_function(
            x=outputs["x_centered"],
            x_hat=outputs["x_hat"],
            p=outputs["p_original"],
            p_hat=outputs["p_hat"],
            mean_z=outputs["mean_z"],
            log_var_z=outputs["log_var_z"],
            s=outputs["s"],
            s_marginal=s_marginal,
            mask=mask,
        )

        self.manual_backward(losses["total_loss"])
        optimizer_main.step()

        # =================== [FIX] Manual scheduler step for manual_optimization mode
        # Since automatic_optimization=False, Lightning won't call scheduler.step() automatically
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            # schedulers can be a single scheduler or a list
            if isinstance(schedulers, (list, tuple)):
                for sch in schedulers:
                    if sch is not None:
                        sch.step()
            else:
                schedulers.step()
        # =================== [FIX] End

        # Logging
        if self.training:
            for key, value in losses.items():
                self.log(
                    f"train_{key}",
                    value,
                    prog_bar=True,
                    logger=True,
                    batch_size=len(batch),
                    on_step=True,
                    on_epoch=True,
                )

        return losses["total_loss"]

    def validation_step(self,data_tuple, batch_idx: int):
        """Validation step."""
        # Unpack batch
        batch,_=data_tuple
        x, p, _, _ = self.unpack_batch(batch)

        # Get expression mask if available - use unified method from base class
        mask = self._get_mask(batch)

        # Get covariates if enabled
        covariates_val = None
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates_val = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}

        # Forward pass
        outputs = self.forward(x, p, covariates_val)

        # Sample marginal perturbations
        batch_size = x.shape[0]
        marginal_indices = torch.randperm(batch_size, device=self.device)
        s_marginal = outputs["s"][marginal_indices]

        # Compute losses
        losses = self.loss_function(
            x=outputs["x_centered"],
            x_hat=outputs["x_hat"],
            p=outputs["p_original"],
            p_hat=outputs["p_hat"],
            mean_z=outputs["mean_z"],
            log_var_z=outputs["log_var_z"],
            s=outputs["s"],
            s_marginal=s_marginal,
            mask=mask,
        )

        # Log validation loss
        self.log("val_loss", losses["total_loss"], prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return losses["total_loss"]


    def predict(self, batch) -> torch.Tensor:
        """
        Predict perturbed gene expression.

        Args:
            batch: Batch containing control expression and perturbations

        Returns:
            Predicted perturbed gene expression [batch_size, n_genes]
        """
        self.Net.eval()

        # Unpack batch
        x, p, _, covariates = self.unpack_batch(batch)

        # Handle covariates if enabled and provided (same as forward method)
        if self.use_covs and covariates:
            # Concatenate all covariate embeddings to perturbation embeddings
            cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
            if cov_tensors:
                merged_covariates = torch.cat(cov_tensors, dim=-1)
                p = torch.cat([p, merged_covariates], dim=-1)

        # Use control expression if available, otherwise use batch expression
        ctrl_values = getattr(batch, "controls", None)
        if ctrl_values is None and hasattr(batch, "control_cell_counts"):
            ctrl_values = batch.control_cell_counts

        if ctrl_values is not None:
            ctrl_x = ctrl_values.squeeze()
            # Center by control mean
            ctrl_x_centered = ctrl_x - self.ctrl_mean.unsqueeze(0)
        else:
            ctrl_x_centered = x - self.ctrl_mean.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            x_hat, _, _, _, _ = self.Net(ctrl_x_centered, p)

            # Add back control mean
            predictions = x_hat + self.ctrl_mean.unsqueeze(0)

        return predictions

    def configure_optimizers(self):
        """
        Configure optimizers for main network and MINE.
        Supports multiple scheduler modes: onecycle, plateau, or step (default).
        Both main and MINE optimizers use the same scheduler mode.

        Returns two optimizers:
        1. Main optimizer: for encoder and decoder
        2. MINE optimizer: for mutual information estimator
        """
        # Main network parameters (encoder + decoder)
        main_params = (
            list(self.Net.Encoder_x.parameters()) +
            list(self.Net.Encoder_p.parameters()) +
            list(self.Net.Decoder_x.parameters()) +
            list(self.Net.Decoder_p.parameters())
        )
        if getattr(self, "use_mix_pert", False) and getattr(self, "pert_encoder", None) is not None:
            main_params += list(self.pert_encoder.parameters())

        optimizer_main = torch.optim.Adam(
            main_params,
            lr=self.lr,
            weight_decay=self.wd
        )

        optimizer_MINE = torch.optim.Adam(
            self.Net.MINE.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        # ===================================[NEW] Unified scheduler selection - supports onecycle/plateau/step modes
        # Select scheduler based on lr_scheduler_mode
        if self.lr_scheduler_mode == "onecycle":
            # OneCycleLR scheduler
            total_steps = self.lr_scheduler_total_steps
            if total_steps is None:
                try:
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
                    total_steps = steps_per_epoch * self.trainer.max_epochs
                except Exception:
                    total_steps = 100 * 100  # fallback

            scheduler_main = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_main,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            scheduler_MINE = torch.optim.lr_scheduler.OneCycleLR(
                optimizer_MINE,
                max_lr=self.lr_scheduler_max_lr or self.lr,
                total_steps=total_steps,
            )
            lr_scheduler_main = {"scheduler": scheduler_main, "interval": "step"}
            lr_scheduler_MINE = {"scheduler": scheduler_MINE, "interval": "step"}

        elif self.lr_scheduler_mode == "plateau":
            # ReduceLROnPlateau scheduler
            scheduler_main = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_main,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            scheduler_MINE = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_MINE,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            lr_scheduler_main = {
                "scheduler": scheduler_main,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }
            lr_scheduler_MINE = {
                "scheduler": scheduler_MINE,
                "monitor": self.lr_monitor_key,
                "frequency": self.lr_scheduler_freq,
                "interval": self.lr_scheduler_interval,
            }

        else:
            # Default: StepLR (scLAMBDA original implementation with step_size=30, gamma=0.2)
            scheduler_main = StepLR(
                optimizer_main,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma
            )
            scheduler_MINE = StepLR(
                optimizer_MINE,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma
            )
            lr_scheduler_main = {"scheduler": scheduler_main}
            lr_scheduler_MINE = {"scheduler": scheduler_MINE}
        
        # ==============================================[NEW] End unified scheduler selection

        return [
            {"optimizer": optimizer_main, "lr_scheduler": lr_scheduler_main},
            {"optimizer": optimizer_MINE, "lr_scheduler": lr_scheduler_MINE},
        ]
