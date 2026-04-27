import torch
import torch.nn.functional as F
import lightning as L

from ..nn.mlp import MLP, MaskNet
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder


class LatentAdditive(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        use_cell_emb: bool = False,
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        n_layers: int = 2,
        encoder_width: int = 128,
        latent_dim: int = 32,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,
        lr_scheduler_step_size: int | None = None,
        lr_scheduler_gamma: float | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        sparse_additive_mechanism: bool = False,
        use_covs: bool = False,  # Unified covariate usage parameter
        datamodule: L.LightningDataModule | None = None,
    ) -> None:
        """
        The constructor for the LatentAdditive class.

        Args:
            n_genes: Number of genes to use for prediction
            n_perts: Number of perturbations in the dataset
                (not including controls)
            n_layers: Number of layers in the encoder/decoder
            encoder_width: Width of the hidden layers in the encoder/decoder
            latent_dim: Dimension of the latent space
            lr: Learning rate
            wd: Weight decay
            lr_scheduler_freq: How often the learning rate scheduler checks
                val_loss
            lr_scheduler_interval: Whether the learning rate scheduler checks
                every epoch or step
            lr_scheduler_patience: Learning rate scheduler patience
            lr_scheduler_factor: Factor by which to reduce learning rate when
                learning rate scheduler triggers
            lr_scheduler_mode: Learning rate scheduler mode ("plateau", "onecycle", "step")
            lr_scheduler_step_size: Step size for StepLR scheduler
            lr_scheduler_gamma: Gamma factor for StepLR scheduler
            lr_scheduler_max_lr: Maximum learning rate for OneCycleLR
            lr_scheduler_total_steps: Total training steps for OneCycleLR
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            use_covs: Whether to use covariates in both encoder and decoder
            datamodule: The datamodule used to train the model
        """
        super(LatentAdditive, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_step_size=lr_scheduler_step_size,
            lr_scheduler_gamma=lr_scheduler_gamma,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            use_mask=use_mask,  # Pass use_mask to base class
        )

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        self.use_covs = use_covs
        self.save_hyperparameters(ignore=["datamodule"])

        n_total_covariates = sum([dim for dim in datamodule.train_dataset.transform.cov_dims.values()]) if hasattr(datamodule.train_dataset.transform, 'cov_dims') else 0
        self.use_cell_emb = use_cell_emb

        self.n_input_features=self.embedding_dim if use_cell_emb else self.n_genes
        encoder_input_dim = (
            self.n_input_features + n_total_covariates
            if use_covs
            else self.n_input_features
        )
        decoder_input_dim = (
            latent_dim + n_total_covariates if use_covs else latent_dim
        )

        self.gene_encoder = MLP(
            encoder_input_dim, encoder_width, latent_dim, n_layers, dropout
        )
        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )
        self.pert_encoder = MixedPerturbationEncoder(gene_pert_dim=self.gene_pert_dim,
                                                     drug_pert_dim=self.drug_pert_dim,
                                                     env_pert_dim=self.env_pert_dim,
                                                     crispr_pert_dim=self.crispr_pert_dim,
                                                     hidden_dims=[latent_dim]*(n_layers-1) if n_layers>1 else [],
                                                     final_embed_dim=latent_dim)

        if sparse_additive_mechanism:
            self.mask_encoder = MaskNet(
                self.n_perts, encoder_width, latent_dim, n_layers
            )

        self.dropout = dropout
        self.softplus_output = softplus_output
        self.sparse_additive_mechanism = sparse_additive_mechanism

    def forward(
        self,
        batch
    ):
        control_input=batch.control_cell_emb if self.use_cell_emb else batch.control_cell_counts
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}
        if self.use_covs:
            merged_covariates = torch.cat(
                [cov for cov in covariates.values()], dim=1
            )
            control_input = torch.cat([control_input, merged_covariates], dim=1)

        latent_control = self.gene_encoder(control_input)
        latent_perturbation = self.pert_encoder(batch)


        latent_perturbed = latent_control + latent_perturbation

        if self.use_covs:
            latent_perturbed = torch.cat([latent_perturbed, merged_covariates], dim=1)
        predicted_perturbed_expression = self.decoder(latent_perturbed)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch, batch_idx: int):

        observed_perturbed_expression = batch["pert_cell_counts"]

        predicted_perturbed_expression = self.forward(
            batch
        )

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=observed_perturbed_expression.size(0))

        return loss

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple

        observed_perturbed_expression = batch["pert_cell_counts"]

        predicted_perturbed_expression = self.forward(
            batch
        )

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        val_loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=observed_perturbed_expression.size(0), on_step=True, on_epoch=True)

        return val_loss

    def predict(self, batch):

        predicted_perturbed_expression = self.forward(
            batch
        )
        return predicted_perturbed_expression

