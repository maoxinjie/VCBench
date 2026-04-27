import torch
import torch.nn.functional as F
import lightning as L
import logging
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder
from ..nn.genepert_networks import GenePertMLP

log = logging.getLogger(__name__)


class GenePert(PerturbationModel):

    def __init__(
            self,
            hidden_size: int = 128,
            use_cell_emb: bool = False,
            use_mask: bool = False,  # Unified mask switch for training loss and evaluation
            use_covs: bool = False,  # Unified covariate usage parameter
            lr: float = 1e-3,
            wd: float = 1e-5,
            lr_scheduler_freq: int | None = None,
            lr_scheduler_interval: str | None = None,
            lr_scheduler_patience: int | None = None,
            lr_scheduler_factor: float | None = None,
            lr_scheduler_mode: str | None = None,
            lr_scheduler_max_lr: float | None = None,
            lr_scheduler_total_steps: int | None = None,
            datamodule: L.LightningDataModule | None = None,
            **kwargs
    ):
        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        super(GenePert, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            use_mask=use_mask,  # Pass use_mask to base class
        )


        self.hidden_size = hidden_size
        self.use_cell_emb = use_cell_emb
        self.use_covs = use_covs

        # Perturbation encoder / embedding dimension setup
        self.pert_encoder = None
        if getattr(self, "use_mix_pert", False):
            self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                crispr_pert_dim=self.crispr_pert_dim,
                final_embed_dim=self.hidden_size,
            )
            self.embedding_dim = self.hidden_size
        else:
            self.embedding_dim = self.datamodule.train_dataset.transform.embedding_dim

        # Calculate total input dimension including covariates
        total_input_dim = self.embedding_dim
        if self.use_covs and hasattr(self, 'cov_dims') and self.cov_dims:
            total_input_dim += sum(self.cov_dims.values())

        # Initialize MLP model
        self.mlp = GenePertMLP(
            input_dim=total_input_dim,
            output_dim=self.n_genes,
            hidden_size=self.hidden_size
        )
        # Note: No ctrl_mean buffers needed - we predict full expression values directly
        # This matches the official GenePert implementation

    def _encode_perturbation(self, batch) -> torch.Tensor:
        """Encode perturbation signals from the batch."""
        if self.pert_encoder is not None:
            return self.pert_encoder(batch)
        return batch["pert_emb"]

    def forward(self, perturbation_embeddings: torch.Tensor, covariates: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            perturbation_embeddings: Tensor of shape [batch_size, embedding_dim]
            covariates: Dictionary of covariate tensors (optional)

        Returns:
            Predicted gene expression values [batch_size, n_genes]
        """
        # Start with perturbation embeddings
        combined_input = perturbation_embeddings

        # Add covariates if enabled and provided
        if self.use_covs and covariates:
            # Concatenate all covariate embeddings
            cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
            if cov_tensors:
                merged_covariates = torch.cat(cov_tensors, dim=-1)
                combined_input = torch.cat([combined_input, merged_covariates], dim=-1)

        # MLP predicts the full expression values (like official GenePert)
        predicted_expression = self.mlp(combined_input)
        return predicted_expression

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Training loss
        """

        # Get observed expression (target is full expression values)
        observed_expression = batch['pert_cell_counts']

        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Get covariates if enabled
        covariates = None
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}

        # Forward pass: predict full expression values directly
        predicted_expression = self.forward(pert_embeddings, covariates)

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        loss=self.auto_mse(predicted_expression, observed_expression, mask)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, data_tuple, batch_idx):
        """
        Validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        batch, _ = data_tuple
        # Get observed expression (target is full expression values)
        observed_expression = batch['pert_cell_counts']

        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Get covariates if enabled
        covariates = None
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}

        # Forward pass: predict full expression values directly
        predicted_expression = self.forward(pert_embeddings, covariates)

        # Use expression mask for loss calculation
        mask = self._get_mask(batch)
        loss=self.auto_mse(predicted_expression, observed_expression, mask)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def predict(self, batch):
        """
        Predict gene expression for a batch.

        Args:
            batch: Input batch with perturbation info

        Returns:
            Predicted gene expression values [batch_size, n_genes]
        """
        # Get perturbation embeddings
        pert_embeddings = self._encode_perturbation(batch)

        # Get covariates if enabled
        covariates = None
        if self.use_covs and hasattr(self, 'cov_keys') and self.cov_keys:
            covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys if cov_key in batch}

        # Forward pass: predict full expression values directly
        predicted_expression = self.forward(pert_embeddings, covariates)

        return predicted_expression

