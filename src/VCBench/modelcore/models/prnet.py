from __future__ import annotations
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .base import PerturbationModel
from ..nn.mix_pert_encoder import MixedPerturbationEncoder


# --- Loss functions (aligned with framework style) ---
def autofocus_mse(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Weighted MSE emphasizing larger residuals when gamma>1"""
    diff = pred - target
    weights = diff.detach().abs().pow(gamma - 1.0)
    return ((diff.pow(2) * (1.0 + weights)).mean())


class _PAdaptor(nn.Module):
    """PRNet Adaptor: maps pooled perturbation token embeddings to comb_dim.
    Uses framework MLP with BatchNorm + LeakyReLU for PRNet-specific structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int, comb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        # Use MLP as backbone, but adapt for PRNet's BatchNorm+LeakyReLU requirement
        # Since MLP uses LayerNorm+ReLU, we build a custom structure here
        # but keep it minimal and aligned with framework patterns
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU(0.3))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.3))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, comb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PEncoder(nn.Module):
    """PRNet Encoder: maps [x_ctrl, c_emb] -> z.
    Uses framework MLP pattern but adapted for PRNet's BatchNorm+LeakyReLU structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, n_layers: int, dropout: float):
        super().__init__()
        # Build MLP-like structure with PRNet's BatchNorm+LeakyReLU (framework-aligned pattern)
        # Non-variational encoder like original PRNet
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.3))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers) if layers else None
        self.to_mean = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.backbone is not None:
            x = self.backbone(x)
        z = self.to_mean(x)
        # Non-variational: return deterministic latent like original PRNet
        return {"z": z, "mu": z, "logvar": None}


class _PDecoder(nn.Module):
    """PRNet Decoder: maps [z, c_emb, n_emb] -> [pred, logvar] (2*x_dim output).
    Uses framework MLP pattern, output split into pred (ReLU) and logvar.
    """

    def __init__(self, input_dim: int, hidden_dim: int, x_dim: int, n_layers: int, dropout: float):
        super().__init__()
        # Use MLP but adapt for PRNet's BatchNorm+LeakyReLU and 2*x_dim output
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU(0.3))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.3))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 2 * x_dim))
        self.net = nn.Sequential(*layers)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.net(x)
        dim = out.size(1) // 2
        # PRNet style: first half through ReLU (pred), second half as logvar
        recon = torch.cat((self.relu(out[:, :dim]), out[:, dim:]), dim=1)
        return {"recon": recon, "pred": recon[:, :dim], "logvar": recon[:, dim:]}


class PRNet(PerturbationModel):
    """PRNet model for perturbation prediction.

    Follows framework patterns using MLP-based components where possible.
    Adaptor, Encoder, and Decoder use BatchNorm+LeakyReLU for PRNet-specific structure.
    """

    def __init__(
            self,
            datamodule: L.LightningDataModule | None = None,
            use_cell_emb: bool = False,
            use_mask: bool = False,  # Unified mask switch for training loss and evaluation
            lr: float = 1e-3,
            wd: float = 1e-5,
            lr_scheduler_factor: float = 0.2,
            lr_scheduler_patience: int = 15,
            lr_scheduler_interval: str = "epoch",
            lr_scheduler_freq: int = 1,
            lr_scheduler_mode: str | None = None,
            lr_scheduler_max_lr: float | None = None,
            lr_scheduler_total_steps: int | None = None,
            lr_monitor_key: str = "val_loss",
            z_dimension: int = 10,
            comb_dimension: int = 50,
            adaptor_hidden_dim: int = 128,
            adaptor_n_layers: int = 1,
            encoder_hidden_dim: int = 128,
            encoder_n_layers: int = 2,
            decoder_hidden_dim: int = 128,
            decoder_n_layers: int = 2,
            dropout: float = 0.05,
            perturbation_combination_delimiter: str = "+",
            control_token: str = "ctrl",
            auto_focus_gamma: float = 2.0,
            use_uncertainty: bool = False,
            use_covs: bool = False,  # Unified covariate usage parameter
    ):
        super(PRNet, self).__init__(
            datamodule=datamodule,
            lr=lr,
            wd=wd,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_freq=lr_scheduler_freq,
            lr_scheduler_mode=lr_scheduler_mode,
            lr_scheduler_max_lr=lr_scheduler_max_lr,
            lr_scheduler_total_steps=lr_scheduler_total_steps,
            lr_monitor_key=lr_monitor_key,
            use_mask=use_mask,  # Pass use_mask to base class
        )

        self.save_hyperparameters(ignore=["datamodule"])

        self.combo_delim = perturbation_combination_delimiter
        self.control_token = control_token
        self.use_uncertainty = use_uncertainty
        self.auto_focus_gamma = auto_focus_gamma

        # Auto-configure covariate usage based on data transform's use_covs setting or parameter
        if hasattr(datamodule.train_dataset.transform, 'use_covs') and datamodule.train_dataset.transform.use_covs:
            # If data transform enables covariates, automatically enable covariate injection
            use_covs = True

        # Use standard Gaussian NLL Loss like original PRNet
        if self.use_uncertainty:
            self.gaussian_nll_loss = nn.GaussianNLLLoss()
        self.use_covs = use_covs
        self.use_cell_emb = use_cell_emb

        # Calculate total covariate dimensions if using covariates
        self.n_total_covariates = getattr(datamodule.train_dataset.transform, 'n_total_covs', 0)


        # Build PRNet components using framework-aligned structure
        self.adaptor = _PAdaptor(
            input_dim=comb_dimension,
            hidden_dim=adaptor_hidden_dim,
            comb_dim=comb_dimension,
            n_layers=adaptor_n_layers,
            dropout=dropout,
        )


        # Use MixedPerturbationEncoder instead of token embeddings
        self.mix_pert_encoder = MixedPerturbationEncoder(
            gene_pert_dim=self.gene_pert_dim,
            drug_pert_dim=self.drug_pert_dim,
            env_pert_dim=self.env_pert_dim,
            crispr_pert_dim=self.crispr_pert_dim,
            hidden_dims=[comb_dimension] * (encoder_n_layers - 1) if encoder_n_layers > 1 else [],
            per_modality_embed_dim=comb_dimension,
            final_embed_dim=comb_dimension,
            dropout=dropout,
        )

        # Encoder: [x_ctrl, c_emb, covariates?] -> z
        encoder_input_dim = self.n_genes + comb_dimension \
            if not self.use_cell_emb else self.embedding_dim+comb_dimension
        if self.use_covs:
            encoder_input_dim += self.n_total_covariates
        self.encoder = _PEncoder(
            input_dim=encoder_input_dim,
            hidden_dim=encoder_hidden_dim,
            z_dim=z_dimension,
            n_layers=encoder_n_layers,
            dropout=dropout,
        )

        # Decoder: [z, c_emb, n_emb, covariates?] -> [pred, logvar]
        decoder_input_dim = z_dimension + comb_dimension + 10
        if self.use_covs:
            decoder_input_dim += self.n_total_covariates
        self.decoder = _PDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=decoder_hidden_dim,
            x_dim=self.n_genes,
            n_layers=decoder_n_layers,
            dropout=dropout,
        )


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # Get control expression (use pert_cell_counts as fallback if control_cell_counts is None)
        x_ctrl = batch.control_cell_counts if not self.use_cell_emb else batch.control_cell_emb

        covariates = {}
        if self.use_covs:
            covariates = {cov_key: getattr(batch, cov_key, torch.tensor([])) for cov_key in self.cov_keys}

        device = x_ctrl.device
        # Use MixPertEncoder to get raw perturbation embedding, then adapt it
        c_raw = self.mix_pert_encoder(batch)
        c_emb = self.adaptor(c_raw)

        # Merge covariates if using them
        merged_covariates = None
        if self.use_covs:
            if covariates is None:
                raise RuntimeError(
                    "use_covs is True but batch.covariates is None"
                )
            # Use fixed covariate_keys order to ensure consistent concatenation
            vecs = []
            for k in self.cov_keys:
                cov = covariates[k]
                # Handle different tensor dimensions safely
                if cov.dim() == 1:
                    cov = cov.unsqueeze(1)  # [B] -> [B, 1]
                elif cov.dim() > 2:
                    cov = cov.view(cov.size(0), -1)  # Flatten to [B, -1]
                # If dim == 2, keep as is [B, d]
                vecs.append(cov)
            merged_covariates = torch.cat(vecs, dim=1).to(device)

        # Encoder input: [x_ctrl, c_emb, covariates?]
        enc_in = torch.cat([x_ctrl, c_emb], dim=1)
        if self.use_covs and merged_covariates is not None:
            enc_in = torch.cat([enc_in, merged_covariates], dim=1)
        enc_out = self.encoder(enc_in)
        z = enc_out["z"]

        # Decoder input: [z, c_emb, n_emb, covariates?]
        # Use random noise as in original PRNet (not deterministic control state)
        n_emb = torch.randn(x_ctrl.size(0), 10, device=device)
        dec_in = torch.cat([z, c_emb, n_emb], dim=1)
        if self.use_covs and merged_covariates is not None:
            dec_in = torch.cat([dec_in, merged_covariates], dim=1)
        dec_out = self.decoder(dec_in)
        return {
            "pred": dec_out["pred"],
            "logvar": dec_out["logvar"],
            "enc_mu": enc_out.get("mu"),
            "enc_logvar": enc_out.get("logvar"),
        }

    def _loss(
            self,
            pred: torch.Tensor,
            x_obs: torch.Tensor,
            x_ctrl: torch.Tensor,
            dec_logvar: Optional[torch.Tensor] = None,
            enc_mu: Optional[torch.Tensor] = None,
            enc_logvar: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        # Use Gaussian NLL as primary loss (like original PRNet)
        if self.use_uncertainty and dec_logvar is not None:
            # Apply softplus to logvar to get var, like original PRNet
            var = F.softplus(dec_logvar)
            # Clamp for numerical stability, use a reasonable minimum
            var = torch.clamp(var, min=1e-3, max=1e3)

            if mask is not None:
                # Masked Gaussian NLL loss
                nll = 0.5 * (torch.log(var) + (pred - x_obs) ** 2 / var)
                # This computes loss over valid genes per batch sample before averaging across the batch
                valid = mask.sum(dim=1)  # Specify batch dimension [batch]
                rec_per_batch = (nll * mask).sum(dim=1)  # [batch]
                rec = (rec_per_batch / valid).nanmean()
            else:
                rec = self.gaussian_nll_loss(pred, x_obs, var)
        else:
            # Fallback to MSE if uncertainty not available
            if mask is not None:
                # Masked MSE loss
                mse = (pred - x_obs) ** 2
                # This computes loss over valid genes per batch sample before averaging across the batch
                valid = mask.sum(dim=1)  # Specify batch dimension [batch]
                rec_per_batch = (mse * mask).sum(dim=1)  # [batch]
                rec = (rec_per_batch / valid).nanmean()
            else:
                rec = autofocus_mse(pred, x_obs, gamma=self.auto_focus_gamma)

        # No additional regularization losses like original PRNet
        total = rec
        return total, rec

    def training_step(self, batch, batch_idx: int):
        # Get target and control expressions directly from batch
        x_obs = batch.pert_cell_counts
        x_ctrl = batch.control_cell_counts

        # Get expression mask using unified method from base class
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(x_obs.device)

        out = self.forward(batch)
        total, rec = self._loss(
            out["pred"],
            x_obs,
            x_ctrl,
            out.get("logvar"),
            out.get("enc_mu"),
            out.get("enc_logvar"),
            mask=mask,
        )
        self.log_dict(
            {"train_loss": total, "train_rec": rec},
            prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True,
        )

        return total

    def validation_step(self,data_tuple, batch_idx: int):
        batch,_=data_tuple
        # Get target and control expressions directly from batch
        x_obs = batch.pert_cell_counts
        x_ctrl = batch.control_cell_counts
        if x_ctrl is None or (hasattr(x_ctrl, 'numel') and x_ctrl.numel() == 0):
            x_ctrl = x_obs

        # Get expression mask using unified method from base class
        mask = self._get_mask(batch)
        if mask is not None:
            mask = mask.to(x_obs.device)

        out = self.forward(batch)
        total, rec = self._loss(
            out["pred"],
            x_obs,
            x_ctrl,
            out.get("logvar"),
            out.get("enc_mu"),
            out.get("enc_logvar"),
            mask=mask,
        )
        self.log_dict(
            {"val_loss": total, "val_rec": rec},
            prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=len(batch), sync_dist=True,
        )

        return total

    def predict(self, batch) -> torch.Tensor:
        return self.forward(batch)["pred"]