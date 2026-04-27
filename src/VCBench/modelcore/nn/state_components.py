"""
State transition model components.

This module contains helper modules and utilities for the StateTransitionPerturbationModel:
- LatentToGeneDecoder: Decoder from latent space to gene expression
- CombinedLoss: Combined Sinkhorn + Energy loss
- ConfidenceToken: Learnable confidence token for loss prediction
- NBDecoder: Negative binomial decoder (scVI-style)
- Transformer backbone utilities (GPT2, LLaMA with bidirectional attention)
- MLP building utilities
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from geomloss import SamplesLoss
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel, PreTrainedModel

# LoRA / PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    TaskType = None  # type: ignore


# =============================================================================
# MLP Utilities
# =============================================================================

def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU,  # default to nn.ReLU class
) -> nn.Sequential:
    """
    Build an MLP of `n_layers` from `in_dim` to `out_dim`.
    """
    layers = []
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")

    if n_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation())  # instantiate the class
        layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())  # instantiate again
            layers.append(nn.Dropout(dropout))

        # Final layer
        layers.append(nn.Linear(hidden_dim, out_dim))

    return nn.Sequential(*layers)


def get_activation_class(name: str) -> nn.Module:
    """
    Given a string activation name, return the corresponding nn.Module class.

    Supported activation functions:
    - ReLU, LeakyReLU, ELU, SELU, GELU
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "elu":
        return nn.ELU
    elif name == "selu":
        return nn.SELU
    elif name == "gelu":
        return nn.GELU
    else:
        raise ValueError(f"Unsupported activation function: {name}")


def get_loss_fn(loss: Union[str, nn.Module]) -> nn.Module:
    """
    Given a string loss function name, return the corresponding nn.Module class.

    Supported loss functions:
    - MSELoss, L1Loss, SmoothL1Loss
    """
    if isinstance(loss, nn.Module):
        return loss

    loss = loss.lower()

    if loss == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


# =============================================================================
# Decoder Components
# =============================================================================

class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    This decoder is trained separately from the main perturbation model.

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        residual_decoder: If True, adds residual connections between every other layer block
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: list[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()

        self.residual_decoder = residual_decoder

        if residual_decoder:
            # Build individual blocks for residual connections
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            # Final output layer
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            # Original implementation without residual connections
            layers = []
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            # Final output layer
            layers.append(nn.Linear(input_dim, gene_dim))
            # Make sure outputs are non-negative
            layers.append(nn.ReLU())

            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]

        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        if self.residual_decoder:
            # Apply blocks with residual connections between every other block
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)

                # Add residual connection from every other previous block
                # Pattern: blocks 1, 3, 5, ... get residual from blocks 0, 2, 4, ...
                if i >= 1 and i % 2 == 1:  # Odd-indexed blocks (1, 3, 5, ...)
                    residual_idx = i - 1  # Previous even-indexed block
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
            return self.decoder(x)


class NBDecoder(nn.Module):
    """
    scVI-style decoder that maps a latent embedding (optionally with batch covariates)
    to the parameters of a negative-binomial (or ZINB) distribution over raw counts.

    Y_ig ~ NB(μ_ig, θ_g)         where
      μ_ig = l_i * softplus(W_g z_i + b_g)
      θ_g  = softplus(r_g)       (gene-specific inverse dispersion)

    Optionally, a zero-inflation gate π_ig can be produced (not shown here).
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims=[1024, 256, 256],
        dropout: float = 0.0,
        use_zero_inflation: bool = False,
    ):
        super().__init__()
        modules = []
        in_features = latent_dim
        for h in hidden_dims:
            modules += [
                nn.Linear(in_features, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_features = h
        self.encoder = nn.Sequential(*modules)

        self.skip = nn.Identity() if in_features == latent_dim else nn.Linear(latent_dim, in_features, bias=False)
        self.post_norm = nn.LayerNorm(in_features)

        # Mean parameter
        self.px_scale = nn.Linear(in_features, gene_dim)

        self.l_encoder = nn.Linear(in_features, 1)

        # Gene-specific inverse dispersion (log-space, broadcasted)
        self.log_theta = nn.Parameter(torch.randn(gene_dim))

        # Optional zero-inflation gate
        self.use_zero_inflation = use_zero_inflation
        if use_zero_inflation:
            self.px_dropout = nn.Linear(in_features, gene_dim)

    @property
    def theta(self):
        # softplus to keep positive
        return F.softplus(self.log_theta)

    def forward(self, z: torch.Tensor, log_library: torch.Tensor | None = None):
        """
        z:            [B, latent_dim]
        log_library:  [B, 1]           (optional – if None we predict it)
        returns μ, θ (and π if requested)
        """
        flat = False
        if z.dim() == 3:  # [B,S,D]  → flatten
            B, S, D = z.shape
            z = z.reshape(-1, D)
            flat = True

        h = self.encoder(z)  # [B* S, H]
        h = self.post_norm(h + self.skip(z))

        if log_library is None:
            log_library = self.l_encoder(h)  # [B* S, 1]
        px_scale = F.softplus(self.px_scale(h))  # [B* S, G]
        mu = torch.exp(log_library) * px_scale  # NB mean

        if self.use_zero_inflation:
            pi = torch.sigmoid(self.px_dropout(h))
            outs = (mu, self.theta, pi)
        else:
            outs = (mu, self.theta)

        if flat:  # reshape back to [B,S,*]
            mu = mu.reshape(B, S, -1)
            if self.use_zero_inflation:
                pi = pi.reshape(B, S, -1)
                return mu, self.theta, pi  # θ remains [G]
            else:
                return mu, self.theta
        return outs

    def gene_dim(self) -> int:
        return self.px_scale.out_features


def nb_nll(x, mu, theta, eps: float = 1e-6):
    """
    Negative-binomial negative log-likelihood.
        x, mu : [..., G]
        theta : [G] or [..., G]
    returns scalar
    """
    logits = (mu + eps).log() - (theta + eps).log()  # NB parameterisation
    dist = NegativeBinomial(total_count=theta, logits=logits)
    return -dist.log_prob(x).mean()


# =============================================================================
# Loss Components
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined Sinkhorn + Energy loss
    """

    def __init__(self, sinkhorn_weight=0.001, energy_weight=1.0, blur=0.05):
        super().__init__()
        self.sinkhorn_weight = sinkhorn_weight
        self.energy_weight = energy_weight
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", blur=blur)
        self.energy_loss = SamplesLoss(loss="energy", blur=blur)

    def forward(self, pred, target):
        sinkhorn_val = self.sinkhorn_loss(pred, target)
        energy_val = self.energy_loss(pred, target)
        return self.sinkhorn_weight * sinkhorn_val + self.energy_weight * energy_val


# =============================================================================
# Confidence Token
# =============================================================================

class ConfidenceToken(nn.Module):
    """
    Learnable confidence token that gets appended to the input sequence
    and learns to predict the expected loss value.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Learnable confidence token embedding
        self.confidence_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Projection head to map confidence token output to scalar loss prediction
        self.confidence_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.ReLU(),  # Ensure positive loss prediction
        )

    def append_confidence_token(self, seq_input: torch.Tensor) -> torch.Tensor:
        """
        Append confidence token to the sequence input.

        Args:
            seq_input: Input tensor of shape [B, S, E]

        Returns:
            Extended tensor of shape [B, S+1, E]
        """
        batch_size = seq_input.size(0)
        # Expand confidence token to batch size
        confidence_tokens = self.confidence_token.expand(batch_size, -1, -1)
        # Concatenate along sequence dimension
        return torch.cat([seq_input, confidence_tokens], dim=1)

    def extract_confidence_prediction(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract main output and confidence prediction from transformer output.

        Args:
            transformer_output: Output tensor of shape [B, S+1, E]

        Returns:
            main_output: Tensor of shape [B, S, E]
            confidence_pred: Tensor of shape [B, 1]
        """
        # Split the output
        main_output = transformer_output[:, :-1, :]  # [B, S, E]
        confidence_output = transformer_output[:, -1:, :]  # [B, 1, E]

        # Project confidence token output to scalar
        confidence_pred = self.confidence_projection(confidence_output).squeeze(-1)  # [B, 1]

        return main_output, confidence_pred


# =============================================================================
# Transformer Backbone Utilities
# =============================================================================

class NoRoPE(nn.Module):
    """
    A drop-in replacement for LlamaRotaryEmbedding that always returns:
      cos = all ones, sin = all zeros
    of shape (batch_size, seq_len, head_dim), so rotary has no effect.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.LongTensor):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _hidden_dim = hidden_states.shape

        # Create cos = ones, sin = zeros
        #   shape --> (batch_size, seq_len, head_dim)
        cos = hidden_states.new_ones(batch_size, seq_len, self.head_dim)
        sin = hidden_states.new_zeros(batch_size, seq_len, self.head_dim)
        return cos, sin


class LlamaBidirectionalModel(LlamaModel):
    """
    A drop-in replacement for LlamaModel with bidirectional attention.
    By overriding _update_causal_mask to return None, all tokens attend to each other.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.rotary_emb = NoRoPE(
            head_dim=config.head_dim,
        )
        
        # Explicitly disable causal attention
        self.config.is_causal = False
        # force every layer to be non-causal
        for layer in self.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False   # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values,
        output_attentions: bool = False,
    ):
        # By returning None, we disable any causal-(look-ahead) masking.
        # The only mask that remains is whatever "attention_mask" the user has passed
        # (e.g. padding-mask), which will be handled by Flash/SDPA internally as non-causal.
        return None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        cache_position: torch.LongTensor = None,
        **flash_attn_kwargs,
    ):
        flash_attn_kwargs["is_causal"] = False
        
        # If no attention_mask is provided, create an all-ones mask (no masking)
        # This ensures bidirectional attention with correct device/dtype
        if attention_mask is None:
            # Get batch size (B) and sequence length (S) from input_embeds if available, else from input_ids.
            # If neither is available, fall back to attention_mask=None and log a warning.
            B = None
            S = None
            if inputs_embeds is not None:
                B, S = inputs_embeds.size(0), inputs_embeds.size(1)
            if B and S:
                attention_mask = torch.ones((B, 1, S, S), dtype=torch.float, device=inputs_embeds.device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )


class GPT2BidirectionalModel(GPT2Model):
    """
    A thin wrapper around GPT2Model that disables the causal (unidirectional) mask,
    allowing full bidirectional attention—and prints the internal bias mask each forward pass.
    """

    def __init__(self, config: GPT2Config):
        # Mark as not-a-decoder (for downstream utilities).
        config.is_decoder = False
        super().__init__(config)

        # Overwrite each attention's bias so no triangular masking occurs.
        for block in self.h:
            # block.attn.bias is a bool-tensor of shape (1, 1, max_pos, max_pos).
            block.attn.bias.data.fill_(True)
            block.attn.is_causal = False

        def _no_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values,
            output_attentions: bool,
        ):
            return None

        self._update_causal_mask = _no_causal_mask.__get__(self, GPT2Model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Determine sequence length for printing the relevant slice of bias
        if input_ids is not None:
            seq_len = input_ids.size(1)
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.size(1)
        else:
            seq_len = None  # If neither is given, we can't infer seq_len

        if seq_len is not None:
            # Print the (1, 1, seq_len, seq_len) slice of the bias for the first block
            self.h[0].attn.bias[0, 0, :seq_len, :seq_len]

        # If a 2D attention_mask was provided, print its expanded 4D version:
        if attention_mask is not None:
            # Expand to (batch_size, 1, seq_len, seq_len)
            B, S = attention_mask.size()
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).expand(B, 1, S, S)
            # Convert to float mask (1→0.0, 0→-inf) just like GPT2 does internally
            neg_inf = torch.finfo(self.dtype).min
            (1.0 - expanded.to(self.dtype)) * neg_inf

        # Finally, call the parent forward method
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def get_transformer_backbone(key, kwargs) -> PreTrainedModel:
    kwargs = dict(kwargs or {})

    if key == "GPT2":
        config = GPT2Config(**kwargs)
        model = GPT2BidirectionalModel(config)

        # Zero out position embeddings and freeze them
        model.wpe.weight.requires_grad = False
        model.wte.weight.requires_grad = False
        model.wpe.weight.zero_()
        model.wte.weight.zero_()

        model_dim = config.n_embd
    elif key == "llama":
        bidirectional_attention = bool(kwargs.pop("bidirectional_attention", False))

        config = LlamaConfig(**kwargs)
        if bidirectional_attention:
            model = LlamaBidirectionalModel(config)
        else:
            model = LlamaModel(config)
        model_dim = config.hidden_size

        model.embed_tokens.weight.requires_grad = False
        model.embed_tokens.weight.zero_()
    else:
        raise ValueError(f"Unknown backbone key {key}")

    return model, model_dim


# =============================================================================
# LoRA Utilities
# =============================================================================

def _default_lora_targets(backbone_key: str, adapt_mlp: bool) -> list[str]:
    """
    Choose target module names for LoRA injection based on backbone type.
    """
    k = backbone_key.lower()
    if k == "llama":
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if adapt_mlp:
            targets += ["gate_proj", "up_proj", "down_proj"]
        return targets
    if k == "gpt2":
        targets = ["c_attn", "c_proj"]
        if adapt_mlp:
            targets += ["mlp.c_fc", "mlp.c_proj"]
        return targets
    raise ValueError(f"Unsupported backbone for LoRA: {backbone_key}")


def apply_lora(model: PreTrainedModel, backbone_key: str, lora_cfg: dict | None) -> PreTrainedModel:
    """
    Apply LoRA adapters to a HuggingFace transformer model when enabled.
    If PEFT is unavailable or config is disabled, returns the original model.
    """
    if not lora_cfg or not lora_cfg.get("enable", False):
        return model

    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed but `lora.enable` is True. Add `peft` to dependencies."
        )

    target = lora_cfg.get("target", "auto")
    adapt_mlp = bool(lora_cfg.get("adapt_mlp", False))
    target_modules = (
        lora_cfg.get("target_modules")
        if target != "auto"
        else _default_lora_targets(backbone_key, adapt_mlp)
    )

    # Build PEFT LoRA config
    task_type_key = lora_cfg.get("task_type", "FEATURE_EXTRACTION")
    task_type = TaskType[task_type_key] if isinstance(task_type_key, str) else task_type_key

    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=task_type,
    )

    peft_model = get_peft_model(model, config)

    # Optional: print trainable params summary if available
    try:
        peft_model.print_trainable_parameters()
    except Exception:
        pass

    return peft_model
