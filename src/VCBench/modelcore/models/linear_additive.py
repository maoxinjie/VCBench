import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from VCBench.data.types import Batch
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder


class LinearAdditive(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        use_covs: bool = False,  # Unified covariate usage parameter
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        encoder_width: int = 512,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: int | None = None,
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        softplus_output: bool = True,
        datamodule: L.LightningDataModule | None = None,
        **kwargs
    ) -> None:
        """
        The constructor for the LinearAdditive class.

        Args:
            n_genes (int): Number of genes in the dataset
            n_perts (int): Number of perturbations in the dataset (not including controls)
            lr (float): Learning rate
            wd (float): Weight decay
            lr_scheduler_freq (int): How often the learning rate scheduler checks val_loss
            lr_scheduler_interval (str): Whether to check val_loss every epoch or every step
            lr_scheduler_patience (int): Learning rate scheduler patience
            lr_scheduler_factor (float): Factor by which to reduce learning rate when learning rate scheduler triggers
            lr_scheduler_mode (str): Learning rate scheduler mode ("plateau", "onecycle", "step")
            lr_scheduler_max_lr (float): Maximum learning rate for OneCycleLR
            lr_scheduler_total_steps (int): Total training steps for OneCycleLR
            use_covs: Whether to condition the linear layer on covariates
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule: The datamodule used to train the model
        """
        super(LinearAdditive, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_mode=lr_scheduler_mode,
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
        self.softplus_output = softplus_output

        n_total_covariates = datamodule.train_dataset.transform.n_total_covs
        self.pert_encoder = None
        pert_dim = encoder_width
        self.pert_encoder = MixedPerturbationEncoder(
            gene_pert_dim=self.gene_pert_dim,
            drug_pert_dim=self.drug_pert_dim,
            env_pert_dim=self.env_pert_dim,
            crispr_pert_dim=self.crispr_pert_dim,
            final_embed_dim=encoder_width,
        )

        if use_covs:
            self.fc_pert = nn.Linear(pert_dim + n_total_covariates, self.n_genes)
        else:
            self.fc_pert = nn.Linear(pert_dim, self.n_genes)

    def _encode_perturbation(self, batch: Batch) -> torch.Tensor:
        if self.pert_encoder is not None:
            return self.pert_encoder(batch)
        return batch[self.pert_key]

    def _get_control_expression(self, batch: Batch) -> torch.Tensor:
        return batch.control_cell_counts
    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict,
    ):
        if self.use_covs:
            merged_covariates = torch.cat(
                [cov for cov in covariates.values()], dim=1
            )
            perturbation = torch.cat([perturbation, merged_covariates], dim=1)

        predicted_perturbed_expression = control_expression + self.fc_pert(perturbation)
        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression

    def training_step(self, batch, batch_idx: int):

        observed_perturbed_expression = batch.pert_cell_counts
        control_expression = self._get_control_expression(batch)
        perturbation = self._encode_perturbation(batch)
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}

        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
        )

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple

        observed_perturbed_expression = batch.pert_cell_counts
        control_expression = self._get_control_expression(batch)
        perturbation = self._encode_perturbation(batch)
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}

        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
        )

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        val_loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return val_loss

    def predict(self, batch):
        control_expression = self._get_control_expression(batch)
        perturbation = self._encode_perturbation(batch)
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}
        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression
