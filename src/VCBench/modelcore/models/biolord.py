import torch
import torch.nn.functional as F
import lightning as L

from ..nn.mlp import MLP
from .base import PerturbationModel
from ..nn import MixedPerturbationEncoder


class BiolordStar(PerturbationModel):
    """
    A version of Biolord
    """

    def __init__(
        self,
        n_layers: int = 2,
        encoder_width: int = 128,
        latent_dim: int = 32,
        penalty_weight: float = 10000.0,
        noise: float = 0.1,
        use_cell_emb: bool=False,
        lr: float | None = None,
        wd: float | None = None,
        lr_scheduler_freq: int | None = None,
        lr_scheduler_interval: str | None = None,
        lr_scheduler_patience: int | None = None,
        lr_scheduler_factor: float | None = None,
        lr_scheduler_mode: str | None = None,
        lr_scheduler_max_lr: float | None = None,
        lr_scheduler_total_steps: int | None = None,
        dropout: float | None = None,
        softplus_output: bool = True,
        n_total_covariates: int | None = None,
        use_mask: bool = False,  # Unified mask switch for training loss and evaluation
        use_covs: bool = False,  # Unified covariate usage parameter
        datamodule: L.LightningDataModule | None = None,
        **kwargs,
    ):
        """
        The constructor for the BiolordStar class.

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
            lr_scheduler_max_lr: Maximum learning rate for OneCycleLR
            lr_scheduler_total_steps: Total training steps for OneCycleLR
            dropout: Dropout rate or None for no dropout.
            softplus_output: Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            datamodule: The datamodule used to train the model
        """
        super(BiolordStar, self).__init__(
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
        self.use_cell_emb=use_cell_emb
        self.save_hyperparameters(ignore=["datamodule"])

        n_total_covariates = datamodule.train_dataset.transform.n_total_covs
        self.n_perts = datamodule.train_dataset.transform.n_perts

        decoder_input_dim = 3 * latent_dim
        self.lord_embedding = torch.nn.Parameter(
            torch.randn(latent_dim, n_total_covariates)
        )
        self.gene_encoder = MLP(
            self.n_genes if not use_cell_emb else self.embedding_dim,
            encoder_width,
             latent_dim,
              n_layers, 
              dropout
        )
        self.decoder = MLP(
            decoder_input_dim, encoder_width, self.n_genes, n_layers, dropout
        )
        self.pert_encoder = MixedPerturbationEncoder(gene_pert_dim=self.gene_pert_dim,
                                                     drug_pert_dim=self.drug_pert_dim,
                                                     env_pert_dim=self.env_pert_dim,
                                                     crispr_pert_dim=self.crispr_pert_dim,
                                                     hidden_dims=[latent_dim]*(n_layers-1) if n_layers>1 else [],
                                                     per_modality_embed_dim=latent_dim,
                                                     final_embed_dim=latent_dim)

        self.penalty_weight = penalty_weight
        self.noise = noise
        self.dropout = dropout
        self.softplus_output = softplus_output

    def forward(
        self,
        input_expression: torch.Tensor,
        batch,
        add_noise: bool = True
    ):
        """
        Forward pass: predict perturbed expression from input expression + perturbation.
        
        Args:
            input_expression: Input expression (typically control expression during training/prediction,
                             or perturbed expression as fallback)
            batch: Batch containing perturbation and covariate information
            add_noise: Whether to add noise to latent representation (only for training, not validation/test)
        """
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}
        latent_input_expression = self.gene_encoder(
            input_expression
        )
        # Only add noise during training, not during validation/test
        if add_noise:
            latent_input_expression += self.noise * torch.randn_like(
                latent_input_expression
            )
        latent_perturbation = self.pert_encoder(batch)
        # # Fix: check whether covariates exist and are valid
        # if "cell_cluster" in covariates and len(covariates["cell_cluster"]) > 0:
        #     latent_covariates = torch.vstack(
        #         [self.lord_embedding[:, cov.bool()].T for cov in covariates["cell_cluster"]]
        #     )
        # Handle covariates: concatenate one-hot vectors of all covariates, then project with lord_embedding
        if self.use_covs and covariates and any(cov is not None and len(cov) > 0 for cov in covariates.values()):
            # Concatenate one-hot vectors for all covariates
            # covariates is a dictionary where each value is a tensor of shape [batch_size, onehot_dim]
            cov_tensors = [cov for cov in covariates.values() if cov is not None and len(cov) > 0]
            if cov_tensors:
                batch_size = cov_tensors[0].shape[0]
                # Concatenate all covariate one-hot vectors in order, resulting in [batch_size, total_onehot_dim]
                merged_cov = torch.cat(cov_tensors, dim=1)  # concatenate along feature dimension
                # Project to embedding vectors via lord_embedding
                # merged_cov is one-hot; locate the position of 1 in each sample
                # Use matrix multiplication: lord_embedding @ merged_cov.T, then transpose
                latent_covariates = (self.lord_embedding @ merged_cov.T).T  # [batch_size, latent_dim]
            else:
                batch_size = latent_input_expression.shape[0]
                latent_covariates = torch.zeros(
                    batch_size, 
                    self.lord_embedding.shape[0],
                    device=input_expression.device
                )
        else:
            # When no covariates exist, use an all-zero vector on the correct device
            batch_size = latent_input_expression.shape[0]
            latent_covariates = torch.zeros(
                batch_size, 
                self.lord_embedding.shape[0],
                device=input_expression.device  # ensure the same device is used
            )
    
        latent_perturbed_expression = torch.cat(
            [
                latent_input_expression,
                latent_perturbation,
                latent_covariates,
            ],
            dim=-1,
        )

        predicted_perturbed_expression = self.decoder(latent_perturbed_expression)

        if self.softplus_output:
            predicted_perturbed_expression = F.softplus(predicted_perturbed_expression)
        return predicted_perturbed_expression, (latent_covariates**2).sum()

    def training_step(self, batch, batch_idx: int):
        observed_perturbed_expression = batch.pert_cell_counts
        
        # Get control expression for training (consistent with prediction)
        # Must have control expression to avoid training contamination
        if not self.use_cell_emb:
            control_expression = batch.control_cell_counts
        else : control_expression = batch.control_cell_emb

        # Training: add noise for regularization
        predicted_perturbed_expression, penalty = self.forward(
            control_expression, batch, add_noise=True
        )
        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        recon_loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)

        # Total loss includes penalty (this is what we optimize)
        penalty_term = self.penalty_weight * penalty
        total_loss = recon_loss + penalty_term

        # Log both reconstruction loss and total loss (with penalty)
        self.log("train_recon_loss", recon_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_penalty", penalty, prog_bar=False, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_penalty_term", penalty_term, prog_bar=False, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        self.log("train_loss", total_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        return total_loss

    def validation_step(self, data_tuple, batch_idx: int):
        batch,_=data_tuple
        observed_perturbed_expression = batch.pert_cell_counts
        
        # Get control expression for validation (consistent with prediction)
        # Must have control expression to avoid validation contamination
        if not self.use_cell_emb:
            control_expression = batch.control_cell_counts
        else : control_expression = batch.control_cell_emb

        # Validation: no noise (deterministic evaluation)
        predicted_perturbed_expression, penalty = self.forward(
            control_expression, batch, add_noise=False
        )
        # Use expression mask for loss calculation - only compute loss on expressed genes
        mask = self._get_mask(batch)
        val_recon_loss=self.auto_mse(predicted_perturbed_expression, observed_perturbed_expression, mask)
        
        # Total loss includes penalty
        penalty_term = self.penalty_weight * penalty
        val_loss = val_recon_loss + penalty_term

        # Log both reconstruction loss and total loss (with penalty)
        self.log("val_recon_loss", val_recon_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_penalty", penalty, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_penalty_term", penalty_term, on_step=True, on_epoch=True, prog_bar=False, logger=True, batch_size=len(batch))
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch))
        return val_loss

    def predict(self, batch):
        # Get control expression - handle different batch formats
        if not self.use_cell_emb:
            control_expression = batch.control_cell_counts
        else : control_expression = batch.control_cell_emb
        
        control_expression = control_expression.to(self.device)

        # Prediction: no noise (deterministic)
        predicted_perturbed_expression, _ = self.forward(
            control_expression,
            batch,
            add_noise=False
        )
        return predicted_perturbed_expression


