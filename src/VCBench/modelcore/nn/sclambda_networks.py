import torch
import torch.nn as nn
class scLAMBDANet(nn.Module):
    """
    Main scLAMBDA network combining all components.

    Architecture:
    - Encoder_x: VAE encoder for gene expression -> latent z (basal state)
    - Encoder_p: Deterministic encoder for perturbation embeddings -> latent s (perturbation effect)
    - Decoder_x: Reconstructs gene expression from z+s
    - Decoder_p: Reconstructs perturbation embedding from s
    - MINE: Mutual information estimator for disentangling z and s
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p_dim: int,
        latent_dim: int = 30,
        hidden_dim: int = 512,
        p_dim_decoder: int = None
    ):
        super(scLAMBDANet, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.Encoder_x = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            VAE=True
        )
        self.Encoder_p = Encoder(
            input_dim=p_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            VAE=False
        )

        # Decoders
        self.Decoder_x = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        # Use p_dim_decoder for decoder output dimension, default to p_dim if not specified
        p_dim_decoder = p_dim_decoder if p_dim_decoder is not None else p_dim
        self.Decoder_p = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=p_dim_decoder
        )

        # MINE for mutual information estimation
        self.MINE = MINE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )

    def reparameterization(self, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        """
        Forward pass through scLAMBDA network.

        Args:
            x: Gene expression [batch_size, x_dim]
            p: Perturbation embeddings [batch_size, p_dim]

        Returns:
            x_hat: Reconstructed gene expression
            p_hat: Reconstructed perturbation embedding
            mean_z: Mean of latent z (basal state)
            log_var_z: Log variance of latent z
            s: Perturbation effect embedding
        """
        # Encode gene expression (VAE)
        mean_z, log_var_z = self.Encoder_x(x)
        z = self.reparameterization(mean_z, torch.exp(0.5 * log_var_z))

        # Encode perturbation (deterministic)
        s = self.Encoder_p(p)

        # Decode
        x_hat = self.Decoder_x(z + s)  # Additive model: basal + perturbation
        p_hat = self.Decoder_p(s)

        return x_hat, p_hat, mean_z, log_var_z, s


class Encoder(nn.Module):
    """
    Encoder network for scLAMBDA.

    Can be used as VAE encoder (returns mean and log_var) or
    deterministic encoder (returns only mean).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        VAE: bool = True
    ):
        super(Encoder, self).__init__()
        self.VAE = VAE

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)

        if self.VAE:
            self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):
        h = self.LeakyReLU(self.FC_input(x))
        h = self.LeakyReLU(self.FC_input2(h))
        mean = self.FC_mean(h)

        if self.VAE:
            log_var = self.FC_var(h)
            return mean, log_var
        else:
            return mean


class Decoder(nn.Module):
    """Decoder network for scLAMBDA."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super(Decoder, self).__init__()

        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))
        out = self.FC_output(h)
        return out


class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE).

    Used to estimate and maximize mutual information between
    basal state (z) and perturbation effect (s) for disentanglement.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int
    ):
        super(MINE, self).__init__()

        self.FC_hidden = nn.Linear(latent_dim * 2, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Compute MINE statistic T(z, s).

        Args:
            z: Basal state latent [batch_size, latent_dim]
            s: Perturbation effect latent [batch_size, latent_dim]

        Returns:
            T: MINE statistic [batch_size, 1]
        """
        h = torch.cat((z, s), dim=1)
        h = self.LeakyReLU(self.FC_hidden(h))
        h = self.LeakyReLU(self.FC_hidden2(h))
        T = self.FC_output(h)
        # Clamp to prevent numerical instability
        return torch.clamp(T, min=-50.0, max=50.0)
