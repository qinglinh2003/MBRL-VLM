"""
Latent World Model for Model-Based VLM RL.

Predicts the next visual latent representation z_{t+1} from the LLM hidden
state at a special <|latent_token|> position.  The frozen VLM visual encoder
provides the regression target, preventing representation collapse without
contrastive losses or EMA networks.

Architecture (VL-JEPA-inspired):
    Input tokens: [h_t, q_1, ..., q_N] -- bidirectional self-attention
    Output: take q_1..q_N positions as predicted visual tokens z_hat_{t+1}
    Target: O_{t+1} -> frozen VLM Visual Encoder -> z_{t+1}
    Loss: per-token cosine alignment + InfoNCE contrastive + optional reward MSE
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENT_TOKEN = "<|latent_token|>"


@dataclass
class WorldModelConfig:
    """Configuration for the Latent World Model."""

    hidden_dim: int = 2048          # LLM hidden size (Qwen2.5-VL-3B)
    n_visual_tokens: int = 9        # visual tokens after PatchMerger (env-dependent)
    n_projector_layers: int = 2     # Transformer layers in projector
    n_heads: int = 8                # attention heads
    dropout: float = 0.0
    reward_loss_weight: float = 0.1 # lambda_reward
    cosine_loss_weight: float = 1.0 # lambda_cosine
    infonce_loss_weight: float = 1.0 # lambda_infonce
    infonce_temperature: float = 0.07 # temperature for InfoNCE


class VisualTokenProjector(nn.Module):
    """Maps a single LLM hidden state to N visual token embeddings.

    VL-JEPA-inspired: concatenate h_t with N learned query tokens, run
    bidirectional self-attention, take query positions as output.

    Input:  (batch, hidden_dim) -- hidden state at <|latent_token|>
    Output: (batch, n_visual_tokens, hidden_dim)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        d = config.hidden_dim
        n = config.n_visual_tokens

        # Learned query embeddings -- one per predicted visual token
        self.queries = nn.Parameter(torch.empty(1, n, d))
        nn.init.trunc_normal_(self.queries, std=1.0 / math.sqrt(d))

        # Bidirectional Transformer: all tokens attend to each other
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_projector_layers
        )

        self.out_norm = nn.LayerNorm(d)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, hidden_dim)
        Returns:
            predicted visual tokens: (batch, n_visual_tokens, hidden_dim)
        """
        batch = hidden_state.shape[0]
        h = hidden_state.unsqueeze(1)                       # (B, 1, D)
        queries = self.queries.expand(batch, -1, -1)         # (B, N, D)
        tokens = torch.cat([h, queries], dim=1)              # (B, 1+N, D)
        out = self.transformer(tokens)                       # (B, 1+N, D)
        query_out = out[:, 1:, :]                            # (B, N, D) -- skip h_t position
        return self.out_norm(query_out)


class RewardHead(nn.Module):
    """Predicts scalar reward from the LLM hidden state."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: (batch, hidden_dim)
        Returns:
            predicted reward: (batch,)
        """
        return self.net(hidden_state).squeeze(-1)


class LatentWorldModel(nn.Module):
    """Latent-space world model for Model-Based VLM RL.

    Components:
        1. VisualTokenProjector: hidden_state -> predicted visual tokens
        2. RewardHead: hidden_state -> predicted scalar reward

    Training target:
        Visual tokens from the frozen VLM visual encoder on the real next
        observation O_{t+1}.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.projector = VisualTokenProjector(config)
        self.reward_head = RewardHead(config.hidden_dim)

    def predict_next_latent(
        self, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict visual token embeddings for the next observation.

        Args:
            hidden_state: (batch, hidden_dim) -- at <|latent_token|> position

        Returns:
            z_hat: (batch, n_visual_tokens, hidden_dim)
        """
        return self.projector(hidden_state)

    def predict_reward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward.

        Args:
            hidden_state: (batch, hidden_dim)
        Returns:
            r_hat: (batch,)
        """
        return self.reward_head(hidden_state)

    def compute_loss(
        self,
        predicted_latent: torch.Tensor,
        target_latent: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        target_reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute world model training loss.

        Loss = lambda_cos * L_cosine + lambda_nce * L_infonce + lambda_r * L_reward

        Cosine loss is computed per-token then averaged.
        InfoNCE is computed on pooled representations for batch-level discrimination.

        Args:
            predicted_latent: (batch, n_vis_tokens, hidden_dim)
            target_latent:    (batch, n_vis_tokens, hidden_dim) -- from frozen ViT
            predicted_reward: (batch,) optional
            target_reward:    (batch,) optional

        Returns:
            total_loss, metrics_dict
        """
        cfg = self.config

        # Per-token cosine similarity, averaged over tokens and batch
        pred_norm = F.normalize(predicted_latent, dim=-1)
        tgt_norm = F.normalize(target_latent.detach(), dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)           # (B, N)
        cosine_loss = (1.0 - cos_sim).mean()

        total_loss = cfg.cosine_loss_weight * cosine_loss

        metrics = {
            "wm/cosine_loss": cosine_loss.item(),
            "wm/cosine_sim_mean": cos_sim.mean().item(),
        }

        # InfoNCE on pooled representations
        if cfg.infonce_loss_weight > 0 and predicted_latent.shape[0] > 1:
            pred_pooled = F.normalize(predicted_latent.mean(dim=1), dim=-1)  # (B, D)
            tgt_pooled = F.normalize(target_latent.detach().mean(dim=1), dim=-1)  # (B, D)

            # Similarity matrix: (B, B)
            logits = pred_pooled @ tgt_pooled.T / cfg.infonce_temperature
            labels = torch.arange(logits.shape[0], device=logits.device)

            # Bidirectional InfoNCE (pred→tgt + tgt→pred)
            infonce_loss = (F.cross_entropy(logits, labels)
                           + F.cross_entropy(logits.T, labels)) / 2

            total_loss = total_loss + cfg.infonce_loss_weight * infonce_loss

            # InfoNCE accuracy: how often is the correct target the top match
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == labels).float().mean()
            metrics["wm/infonce_loss"] = infonce_loss.item()
            metrics["wm/infonce_acc"] = acc.item()

        if predicted_reward is not None and target_reward is not None:
            reward_loss = F.mse_loss(predicted_reward, target_reward.detach())
            total_loss = total_loss + cfg.reward_loss_weight * reward_loss
            metrics["wm/reward_loss"] = reward_loss.item()

        metrics["wm/total_loss"] = total_loss.item()
        return total_loss, metrics

    def forward(
        self,
        hidden_state: torch.Tensor,
        target_latent: Optional[torch.Tensor] = None,
        target_reward: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, dict]]]:
        """Full forward pass: predict latent + reward, optionally compute loss.

        Args:
            hidden_state: (batch, hidden_dim)
            target_latent: (batch, n_vis_tokens, hidden_dim) -- for training
            target_reward: (batch,) -- for training

        Returns:
            z_hat: (batch, n_vis_tokens, hidden_dim)
            r_hat: (batch,)
            loss_tuple: (loss, metrics) if targets provided, else None
        """
        z_hat = self.predict_next_latent(hidden_state)
        r_hat = self.predict_reward(hidden_state)

        loss_tuple = None
        if target_latent is not None:
            loss_tuple = self.compute_loss(
                z_hat, target_latent, r_hat, target_reward
            )

        return z_hat, r_hat, loss_tuple


# ---------------------------------------------------------------------------
# Utility: extract vision target from frozen visual encoder
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_vision_target(
    visual_encoder: nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen VLM visual encoder to get target visual tokens.

    This calls the same `model.visual(pixel_values, grid_thw=...)` that
    Qwen2-VL uses internally, but returns the output directly as the
    regression target for world model training.

    Args:
        visual_encoder: Qwen2VisionTransformerPretrainedModel (frozen)
        pixel_values:   (total_patches, C*T*P*P) preprocessed by image_processor
        image_grid_thw: (num_images, 3) grid dimensions

    Returns:
        target_tokens: (total_merged_tokens, hidden_dim)
            After PatchMerger, in the LLM embedding space.
    """
    visual_encoder.eval()
    pixel_values = pixel_values.to(
        device=visual_encoder.get_device(),
        dtype=visual_encoder.get_dtype(),
    )
    image_grid_thw = image_grid_thw.to(device=visual_encoder.get_device())
    return visual_encoder(pixel_values, grid_thw=image_grid_thw)


def compute_n_visual_tokens(
    image_height: int,
    image_width: int,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    temporal_patch_size: int = 2,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> int:
    """Compute the number of visual tokens after Qwen2-VL processing.

    Reproduces the smart_resize + PatchMerger token count calculation.
    """
    import math as _math

    factor = patch_size * spatial_merge_size
    h_bar = round(image_height / factor) * factor
    w_bar = round(image_width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = _math.sqrt((image_height * image_width) / max_pixels)
        h_bar = _math.floor(image_height / beta / factor) * factor
        w_bar = _math.floor(image_width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = _math.sqrt(min_pixels / (image_height * image_width))
        h_bar = _math.ceil(image_height * beta / factor) * factor
        w_bar = _math.ceil(image_width * beta / factor) * factor

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    grid_t = 1  # single image

    n_merged = grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
    return n_merged
