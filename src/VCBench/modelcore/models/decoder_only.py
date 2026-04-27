"""
BSD 3-Clause License

Copyright (c) 2024, <anonymized authors of NeurIPS submission #1306>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch
import torch.nn.functional as F
import lightning as L

from ..nn.mlp import MLP
from .base import PerturbationModel
from VCBench.data.types import Batch
from ..nn import MixedPerturbationEncoder


class DecoderOnly(PerturbationModel):
    """
    A latent additive model for predicting perturbation effects
    """

    def __init__(
        self,
        n_layers=2,
        encoder_width=512,
        softplus_output=True,
        use_covs: bool = False,  # Unified covariate usage parameter
        use_perturbations=True,
        use_cell_emb: bool = False,
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        datamodule: L.LightningDataModule | None = None,
        **kwargs
    ) -> None:
        """
        The constructor for the DecoderOnly class.

        Args:
            n_genes (int): Number of genes to use for prediction
            n_perts (int): Number of perturbations in the dataset (not including controls)
            n_layers (int): Number of layers in the encoder/decoder
            lr (float): Learning rate
            wd (float): Weight decay
            lr_scheduler_freq (int): How often the learning rate scheduler checks val_loss
            lr_scheduler_interval (str): Whether the learning rate scheduler checks every epoch or step
            lr_scheduler_patience (int): Learning rate scheduler patience
            lr_scheduler_factor (float): Factor by which to reduce learning rate when learning rate scheduler triggers
            lr_scheduler_mode (str): Learning rate scheduler mode ("plateau", "onecycle", "step")
            lr_scheduler_max_lr (float): Maximum learning rate for OneCycleLR
            lr_scheduler_total_steps (int): Total training steps for OneCycleLR
            softplus_output (bool): Whether to apply a softplus activation to the output of the decoder to enforce non-negativity
        """

        super(DecoderOnly, self).__init__(
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
        self.save_hyperparameters(ignore=["datamodule"])

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate usage
            use_covs = True

        if not (use_covs or use_perturbations):
            raise ValueError(
                "'use_covs' and 'use_perturbations' can not both be false. Either covariates or perturbations have to be used."
            )


        n_total_covariates = datamodule.train_dataset.transform.n_total_covs

        self.pert_encoder = MixedPerturbationEncoder(
                gene_pert_dim=self.gene_pert_dim,
                drug_pert_dim=self.drug_pert_dim,
                env_pert_dim=self.env_pert_dim,
                crispr_pert_dim=self.crispr_pert_dim,
                final_embed_dim=encoder_width,
            )

        if use_covs and use_perturbations:
            decoder_input_dim = n_total_covariates + encoder_width
        elif use_covs:
            decoder_input_dim = n_total_covariates
        else:
            decoder_input_dim =encoder_width

        self.decoder = MLP(decoder_input_dim, encoder_width, self.n_genes, n_layers)
        self.softplus_output = softplus_output
        self.use_covs = use_covs
        self.use_perturbations = use_perturbations
        self.use_cell_emb = use_cell_emb

    def _get_control_expression(self, batch: Batch) -> torch.Tensor:
        if self.use_cell_emb:
            return batch.control_cell_emb
        else:
            return batch.control_cell_counts
    
    def _encode_perturbation(self, batch: Batch) -> torch.Tensor:
        if self.pert_encoder is not None:
            return self.pert_encoder(batch)
        return batch[self.pert_key]

    def forward(
        self,
        control_expression: torch.Tensor,
        perturbation: torch.Tensor,
        covariates: dict[str, torch.Tensor],
    ):
        if self.use_covs and self.use_perturbations:
            embedding = torch.cat([cov for cov in covariates.values()], dim=1)
            embedding = torch.cat([perturbation, embedding], dim=1)
        elif self.use_covs:
            embedding = torch.cat([cov for cov in covariates.values()], dim=1)
        elif self.use_perturbations:
            embedding = perturbation

        predicted_perturbed_expression = self.decoder(embedding)

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

        observed_perturbed_expression=batch.pert_cell_counts
        control_expression = self._get_control_expression(batch)
        perturbation=self._encode_perturbation(batch)
        covariates={cov_key:batch[cov_key] for cov_key in self.cov_keys}

        predicted_perturbed_expression = self.forward(
            control_expression, perturbation, covariates
        )

        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        val_loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return val_loss

    def predict(self, batch):
        # Get control expression and move it to the current device
        control_expression = self._get_control_expression(batch).to(self.device)
        perturbation = self._encode_perturbation(batch).to(self.device)
        covariates = {cov_key:batch[cov_key] for cov_key in self.cov_keys}

        predicted_perturbed_expression = self.forward(
            control_expression,
            perturbation,
            covariates,
        )
        return predicted_perturbed_expression


