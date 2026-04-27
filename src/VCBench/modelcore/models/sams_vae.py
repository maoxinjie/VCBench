from typing import Tuple
import torch
import torch.distributions as dist
import lightning as L

from ..nn.vae import BaseEncoder
from ..nn.mlp import gumbel_softmax_bernoulli
from .base import PerturbationModel
from ..nn.decoders import (
    DeepIsotropicGaussian,
)
from ..nn import MixedPerturbationEncoder


class SparseAdditiveVAE(PerturbationModel):
    """
    Sparse Additive Variational Autoencoder (VAE) model, following the model proposed in the paper:

    Bereket, Michael, and Theofanis Karaletsos.
    "Modelling Cellular Perturbations with the Sparse Additive Mechanism Shift Variational Autoencoder."
    Advances in Neural Information Processing Systems 36 (2024).

    Attributes:
        n_genes (int): Number of genes.
        n_perts (int): Number of perturbations.
        lr (int): Learning rate.
        wd (int): Weight decay.
        lr_scheduler_freq (int): Frequency of the learning rate scheduler.
        lr_scheduler_patience (int): Patience of the learning rate scheduler.
        lr_scheduler_factor (int): Factor of the learning rate scheduler.
        latent_dim (int): Latent dimension.
        sparse_additive_mechanism (bool): Whether to use sparse additive mechanism.
        mean_field_encoding (bool): Whether to use mean field encoding.
        use_covs (bool): Whether to inject covariates in both encoder and decoder.
        mask_prior_probability (float): The target probability for the masks.
        datamodule (L.LightningDataModule | None): LightningDataModule for data loading.

    Methods:
        reparameterize(mu, log_var): Reparametrizes the Gaussian distribution.
        training_step(batch, batch_idx): Performs a training step.
        validation_step(batch, batch_idx): Performs a validation step.
        configure_optimizers(): Configures the optimizers.

    """

    def __init__(
            self,
            pert_comb_delim: str ='+',
            use_mask: bool = False,  # Unified mask switch for training loss and evaluation
            n_layers_encoder_x: int = 2,
            n_layers_encoder_e: int = 2,
            n_layers_decoder: int = 3,
            hidden_dim_x: int = 850,
            hidden_dim_cond: int = 128,
            latent_dim: int = 40,
            dropout: float = 0.2,
            use_covs: bool = False,  # Unified covariate usage parameter
            use_cell_emb: bool = False,
            mask_prior_probability: float = 0.01,
            lr: int | None = None,
            wd: int | None = None,
            lr_scheduler_freq: int | None = None,
            lr_scheduler_interval: str | None = None,
            lr_scheduler_patience: int | None = None,
            lr_scheduler_factor: float | None = None,
            lr_scheduler_mode: str | None = None,
            lr_scheduler_max_lr: float | None = None,
            lr_scheduler_total_steps: int | None = None,
            softplus_output: bool = True,
            generative_counterfactual: bool = False,
            embedding_width: int | None = None,
            disable_sparsity: bool = False,
            disable_e_dist: bool = False,
            datamodule: L.LightningDataModule | None = None,
            **kwargs
    ) -> None:
        """
        Initializes the SparseAdditiveVAE model.

        Args:
            n_genes (int): Number of genes.
            n_perts (int): Number of perturbations.
            transform (Dispatch): Transform for the data.
            context (dict): Context for the data.
            evaluation (DictConfig): Evaluation configuration.
            n_layers_encoder_x (int): Number of layers in the encoder for x.
            n_layers_encoder_e (int): Number of layers in the encoder for e.
            n_layers_decoder (int): Number of layers in the decoder.
            hidden_dim_x (int): Hidden dimension for x.
            hidden_dim_cond (int): Hidden dimension for the conditional input.
            latent_dim (int): Latent dimension.
            lr (int): Learning rate.
            wd (int): Weight decay.
            lr_scheduler_freq (int): Frequency of the learning rate scheduler.
            lr_scheduler_patience (int): Patience of the learning rate scheduler.
            lr_scheduler_factor (int): Factor of the learning rate scheduler.
            lr_scheduler_mode (str): Learning rate scheduler mode ("plateau", "onecycle", "step").
            lr_scheduler_max_lr (float): Maximum learning rate for OneCycleLR.
            lr_scheduler_total_steps (int): Total training steps for OneCycleLR.
            use_covs (bool): Whether to inject covariates in both encoder and decoder.
            mask_prior_probability (float): The target probability for the masks.
            softplus_output (bool): Whether to apply a softplus activation to the
                output of the decoder to enforce non-negativity
            generative_counterfactual (bool): Whether to use the generative mode, i.e. sample from the prior distribution. Only used for inference.

        Returns:
            None
        """

        super(SparseAdditiveVAE, self).__init__(
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

        self.n_perts=datamodule.train_dataset.transform.n_perts

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        self.use_covs = use_covs
        self.use_cell_emb=use_cell_emb
        self.latent_dim = latent_dim
        self.latent_dim_pert = latent_dim * self.n_perts
        self.mask_prior_probability = mask_prior_probability
        self.softplus_output = softplus_output
        self.generative_counterfactual = generative_counterfactual

        self.perturbations_all_sum = None

        if self.use_covs:
            self.n_total_covariates = datamodule.train_dataset.transform.n_total_covs

        encoder_input_dim = self.embedding_dim  if self.use_cell_emb else self.n_genes
        encoder_input_dim += self.n_total_covariates if self.use_covs else 0
     
        decoder_input_dim = (
            latent_dim + self.n_total_covariates
            if self.use_covs
            else latent_dim
        )

        self.encoder_x = BaseEncoder(
            input_dim=encoder_input_dim + self.latent_dim,
            hidden_dim=hidden_dim_x,
            latent_dim=latent_dim,
            n_layers=n_layers_encoder_x,
        )

        self.disable_sparsity = disable_sparsity
        self.disable_e_dist = disable_e_dist
        self.encoder_e = BaseEncoder(
            input_dim=latent_dim + self.n_perts
            if not self.disable_sparsity
            else self.n_perts,
            hidden_dim=hidden_dim_x,
            latent_dim=latent_dim,
            n_layers=n_layers_encoder_e,
        )

        self.m_logits = torch.nn.Parameter(-torch.ones((self.n_perts, self.latent_dim)))

        self.decoder = DeepIsotropicGaussian(
            decoder_input_dim,
            hidden_dim_x,
            self.n_genes,
            n_layers_decoder,
            dropout,
            softplus_output,
        )

        self.pert_encoder = MixedPerturbationEncoder(
            gene_pert_dim=self.gene_pert_dim,
            drug_pert_dim=self.drug_pert_dim,
            env_pert_dim=self.env_pert_dim,
            crispr_pert_dim=self.crispr_pert_dim,
            hidden_dims=[latent_dim] * (n_layers_encoder_e - 1) if n_layers_encoder_e > 1 else [],
            final_embed_dim=latent_dim,
        )

    def forward(
            self,
            observed_perturbed_expression: torch.Tensor,
            perturbation: torch.Tensor,
            covariates: dict,
            batch,
            inference: bool = False,
            mask: torch.Tensor = None,
    ) -> Tuple:
        batch_size = observed_perturbed_expression.shape[0]
        # perturbations_per_cell = perturbation.sum(axis=1)

        if self.use_covs:
            merged_covariates = torch.cat(
                [cov for cov in covariates.values()], dim=1
            )

        if self.use_covs:
            observed_expression_with_covariates = torch.cat(
                [observed_perturbed_expression, merged_covariates.to(self.device)],
                dim=1,
            )
        else:
            observed_expression_with_covariates = observed_perturbed_expression

        if self.disable_sparsity:
            m = torch.ones_like(self.m_logits)
        else:
            m_probs = torch.sigmoid(self.m_logits)
            m = gumbel_softmax_bernoulli(m_probs)

        if self.pert_encoder:
            # mix_pert: directly obtain perturbation embedding with MixedPerturbationEncoder
            z_p = self.pert_encoder(batch)  # [B, latent_dim]
            e_mu = e_log_var = None
        else:
            # Original multi-hot perturbation workflow
            perturbations_per_cell = perturbation.sum(axis=1)
            # Get indices where perturbations are active (1s)
            z_p_index_batch, z_p_index_pert = torch.where(perturbation.bool())

            # Initialize z_p with zeros early
            z_p = torch.zeros((batch_size, self.latent_dim), device=self.device)
            # Initialize e_mu and e_log_var as None
            e_mu, e_log_var = None, None

            # Only process perturbations if there are any in the batch
            if z_p_index_batch.nelement() > 0:
                m_t = torch.cat(
                    [
                        m[perturbation[i].bool()]
                        for i in range(batch_size)
                        if perturbation[i].bool().any()
                    ]
                )
                perturbation_expanded = perturbation.repeat_interleave(
                    perturbations_per_cell.int(), dim=0
                )

                if self.disable_sparsity:
                    mask_and_perturbation = perturbation_expanded
                else:
                    mask_and_perturbation = torch.cat([m_t, perturbation_expanded], dim=-1)
                e_mu, e_log_var = self.encoder_e(mask_and_perturbation)

                if self.disable_e_dist:
                    e_t = e_mu
                else:
                    # Numerical stability: clamp log variance
                    e_log_var = torch.clamp(e_log_var, min=-10.0, max=10.0)
                    # Sample from q(e|x,p)
                    e_dist = dist.Normal(e_mu, torch.exp(0.5 * e_log_var).clip(min=1e-8))
                    e_t = e_dist.rsample()

                # Calculate element-wise product
                combined_effect = m_t * e_t

                # Use scatter_add_ to sum the effects for each batch sample
                z_p.index_add_(0, z_p_index_batch, combined_effect)

        observed_expression_with_covariates_and_z_p = torch.cat(
            [observed_expression_with_covariates, z_p], dim=-1
        )  # use torch.zeros_like(z_p) to mimic posterior inference
        z_mu_x, z_log_var_x = self.encoder_x(
            observed_expression_with_covariates_and_z_p
        )

        # Numerical stability: clamp log variance and handle NaN/inf values
        z_log_var_x = torch.clamp(z_log_var_x, min=-10.0, max=10.0)  # Prevent extreme values

        # Sample from q(z|x)
        q_z = dist.Normal(z_mu_x, torch.exp(0.5 * z_log_var_x).clip(min=1e-8))
        # only z_basal is sampled from the prior at inference time
        z_basal = q_z.rsample() if not inference else torch.randn_like(z_mu_x)

        z = z_basal + z_p

        if self.use_covs:
            z = torch.cat([z, merged_covariates], dim=1)

        predictions = self.decoder(z, library_size=None)

        # Compute log probabilities
        p_z = dist.Normal(torch.zeros_like(z_mu_x), torch.ones_like(z_mu_x))
        log_qz = q_z.log_prob(z_basal).sum(axis=-1)
        log_pz = p_z.log_prob(z_basal).sum(axis=-1)

        # Compute reconstruction loss first
        if mask is not None:
            # Masked reconstruction loss (MSE)
            import torch.nn.functional as F
            mse = F.mse_loss(predictions, observed_perturbed_expression, reduction='none')
            valid = mask.sum(dim=1) 
            reconstruction_loss_per_batch = (mse * mask).sum(dim=1)
            reconstruction_loss = (reconstruction_loss_per_batch / valid).nanmean()
        else:
            reconstruction_loss = self.decoder.reconstruction_loss(
                predictions, observed_perturbed_expression
            )

        # Then compute the KL branch
        if self.pert_encoder:
            log_qe_pe = torch.zeros(batch_size, device=self.device)
            log_qm_pm = torch.zeros(1, device=self.device)
        else:
            # Initialize log_qe and log_pe
            log_qe_pe = torch.zeros(batch_size, device=self.device)

            if not self.disable_e_dist:
                # Calculate log probabilities for perturbation effects if there are any
                if e_mu is not None:
                    # Numerical stability: clamp log variance
                    e_log_var_clamped = torch.clamp(e_log_var, min=-10.0, max=10.0)
                    q_e = dist.Normal(e_mu, torch.exp(0.5 * e_log_var_clamped).clip(min=1e-8))
                    p_e = dist.Normal(torch.zeros_like(e_mu), torch.ones_like(e_mu))
                    log_qe = q_e.log_prob(e_t).sum(axis=-1)
                    log_pe = p_e.log_prob(e_t).sum(axis=-1)

                    # Add log prob terms to the correct batch samples
                    log_qe_pe.index_add_(0, z_p_index_batch, log_qe - log_pe)

                # Apply adjustment factor
                adjustment_factor = 1 / (
                        perturbation @ self.perturbations_all_sum.to(self.device)
                )
                # avoid inf
                adjustment_factor[adjustment_factor.isinf()] = 0
                log_qe_pe = log_qe_pe * adjustment_factor

            if self.disable_sparsity:
                log_qm_pm = torch.zeros(
                    perturbation.shape[1],
                    device=reconstruction_loss.device,
                    dtype=reconstruction_loss.dtype,
                )
            else:
                # Compute mask prior log probabilities
                q_m = dist.Bernoulli(probs=torch.sigmoid(self.m_logits))
                p_m = dist.Bernoulli(
                    probs=self.mask_prior_probability * torch.ones_like(self.m_logits)
                )
                log_qm_pm = (q_m.log_prob(m) - p_m.log_prob(m)).sum(axis=-1)
                log_qm_pm = (
                        log_qm_pm
                        * perturbation.sum(axis=0)
                        / self.perturbations_all_sum.to(self.device)
                )

        # kld & elbo
        kld = (log_qz - log_pz).mean() + log_qe_pe.mean() + log_qm_pm.sum() / batch_size
        elbo = -(reconstruction_loss + kld)

        return predictions, reconstruction_loss, kld, elbo

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        observed_perturbed_expression = batch.pert_cell_counts
        if self.pert_encoder:
            perturbation = None
        else:
            perturbation = batch[self.pert_key]
        covariates = {cov_key:batch[cov_key] for cov_key in self.cov_keys}

        # Get expression mask if available
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(observed_perturbed_expression.device)

        predictions, recon_loss, kld, elbo = self(
            observed_perturbed_expression, perturbation, covariates, batch, mask=mask
        )
        loss = -elbo  # Minimize negative ELBO
        
        # Log train_loss (main loss for monitoring)
        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)
        
        self.log(
            "kld",
            kld,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "recon_loss",
            recon_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log(
            "elbo",
            elbo,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )

        return loss

    def validation_step(self, data_tuple, batch_idx: int) -> torch.Tensor:
        batch,_=data_tuple
        observed_perturbed_expression = batch.pert_cell_counts
        if self.pert_encoder:
            perturbation = None
        else:
            perturbation = batch[self.pert_key]
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}

        # Get expression mask if available
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(observed_perturbed_expression.device)

        predictions, recon_loss, kld, elbo = self(
            observed_perturbed_expression, perturbation, covariates, batch, mask=mask
        )
        val_loss = -elbo  # Minimize negative ELBO
        self.log(
            "val_kld",
            kld,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
        )
        self.log("val_loss", val_loss, prog_bar=True, logger=True, batch_size=len(batch), on_step=True, on_epoch=True)

        return val_loss

    def predict(self, batch) -> torch.Tensor:
        observed_perturbed_expression = batch.pert_cell_counts
        if self.pert_encoder:
            perturbation = None
        else:
            perturbation = batch[self.pert_key]
        covariates = {cov_key: batch[cov_key] for cov_key in self.cov_keys}

        if self.generative_counterfactual:
            x_sample, _, _, _ = self(
                observed_perturbed_expression, perturbation, covariates, batch, inference=True
            )
        else:
            x_sample, _, _, _ = self(
                observed_perturbed_expression, perturbation, covariates, batch, inference=False
            )
        return x_sample

    def reparameterize(
            self,
            mu: torch.Tensor,
            log_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparametrizes the Gaussian distribution so (stochastic) backpropagation can be applied.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std
