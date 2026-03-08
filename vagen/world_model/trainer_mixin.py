"""
World Model Trainer Mixin — hooks into RayPPOTrainer for MBRL integration.

Add to the trainer via:
    class MyTrainer(WorldModelTrainerMixin, RayPPOTrainer):
        pass

Or call the hook methods directly from the training loop:
    trainer._wm_post_rollout(observations, actions, ...)
    trainer._wm_update(global_step)
    trainer._wm_augment_batch(batch)

Integration points in fit() loop:
    1. After rollout completes -> _wm_post_rollout()
    2. Before advantage computation -> _wm_augment_rewards()
    3. After advantage computation -> _wm_update()
    4. Metrics logging -> _wm_metrics()
"""

import json
import os
from typing import Dict, List, Optional

import torch

from vagen.world_model.latent_world_model import LatentWorldModel, WorldModelConfig, LATENT_TOKEN
from vagen.world_model.imagination import LatentImagination, ImaginationConfig
from vagen.world_model.rl_integration import (
    DynaAugmentor,
    DynaConfig,
    ImaginedValueEstimator,
    ImaginedValueConfig,
    WorldModelRLManager,
    WorldModelRLConfig,
)


def create_world_model_manager(
    vlm_model,
    processor,
    tokenizer,
    wm_checkpoint_dir: str,
    config: dict,
    device: str = "cuda:0",
) -> WorldModelRLManager:
    """Factory function to create WorldModelRLManager from config dict.

    Args:
        vlm_model: loaded Qwen2.5-VL model
        processor: HF processor
        tokenizer: HF tokenizer (with <|latent_token|> added)
        wm_checkpoint_dir: path to trained world model checkpoint
        config: dict with world_model config section
        device: torch device

    Returns:
        Configured WorldModelRLManager
    """
    # Load world model
    wm_config_path = os.path.join(wm_checkpoint_dir, "config.json")
    with open(wm_config_path) as f:
        wm_cfg = json.load(f)
    wm_config = WorldModelConfig(**wm_cfg)
    world_model = LatentWorldModel(wm_config).to(device)
    world_model.load_state_dict(
        torch.load(
            os.path.join(wm_checkpoint_dir, "best_model.pt"),
            map_location=device,
        )
    )

    # Ensure latent token exists in tokenizer
    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    if latent_token_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_TOKEN]})
        processor.tokenizer = tokenizer
        latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

    # Build RL config from dict
    rl_config = WorldModelRLConfig(
        enabled=config.get("enabled", False),
        wm_lr=config.get("wm_lr", 1e-4),
        wm_update_freq=config.get("wm_update_freq", 1),
        wm_batch_size=config.get("wm_batch_size", 64),
        imagination=ImaginationConfig(
            horizon=config.get("imagination_horizon", 5),
            discount=config.get("discount", 0.99),
        ),
        dyna=DynaConfig(
            enabled=config.get("dyna_enabled", False),
            n_imagined_actions=config.get("dyna_n_actions", 4),
            horizon=config.get("dyna_horizon", 3),
            reward_weight=config.get("dyna_reward_weight", 0.5),
        ),
        imagined_value=ImaginedValueConfig(
            enabled=config.get("imagined_value_enabled", False),
            horizon=config.get("value_horizon", 5),
            discount=config.get("discount", 0.99),
            value_weight=config.get("value_weight", 0.5),
        ),
    )

    manager = WorldModelRLManager(
        vlm=vlm_model,
        processor=processor,
        world_model=world_model,
        latent_token_id=latent_token_id,
        config=rl_config,
        device=device,
    )

    # Initialize LoRA if Level 3 enabled
    if config.get("lora_enabled", False):
        manager.setup_lora()

    return manager


class WorldModelTrainerMixin:
    """Mixin class that adds world model hooks to RayPPOTrainer.

    Expected attributes from the base trainer:
        self.config.world_model: dict with WM config
        self.wm_manager: WorldModelRLManager (initialized in fit())

    Hook points:
        _wm_post_rollout():  called after rollout, before advantage
        _wm_update():        called after policy update
        _wm_get_metrics():   returns WM metrics for logging
    """

    def _wm_init(self):
        """Initialize world model manager. Call once at start of fit()."""
        wm_config = self.config.get("world_model", {})
        if not wm_config.get("enabled", False):
            self.wm_manager = None
            return

        # The VLM for imagination needs to be a separate instance or
        # the same model used for rollout. For simplicity, we create
        # the manager with a reference to load it lazily.
        print("[WorldModel] Initializing world model manager...")
        # NOTE: In practice, the VLM model loading depends on the
        # specific trainer setup. This is a template — adapt to your
        # model loading pattern.
        self.wm_manager = None  # Placeholder
        self._wm_metrics_buffer = {}
        print("[WorldModel] Manager initialized (lazy — VLM set on first use)")

    def _wm_post_rollout(
        self,
        observations: Optional[list] = None,
        actions: Optional[List[str]] = None,
        next_observations: Optional[list] = None,
        rewards: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Called after real rollout completes.

        1. Stores real transitions in WM buffer (for online training)
        2. Runs Dyna imagination if enabled
        3. Returns metrics

        Args:
            observations: visited states (PIL images)
            actions: action names taken
            next_observations: resulting states
            rewards: observed rewards
        """
        if self.wm_manager is None:
            return {}

        metrics = {}

        # Store real transitions for online WM training
        if observations and actions and next_observations and rewards:
            self.wm_manager.collect_wm_data(
                observations, actions, next_observations, rewards,
            )

        # Dyna imagination
        if self.wm_manager.config.dyna.enabled and observations and actions:
            _, dyna_metrics = self.wm_manager.imagination_step(
                observations, actions,
            )
            metrics.update(dyna_metrics)

        self._wm_metrics_buffer.update(metrics)
        return metrics

    def _wm_update(self, global_step: int) -> Dict[str, float]:
        """Called after policy update. Trains world model online.

        Args:
            global_step: current training step

        Returns:
            WM training metrics
        """
        if self.wm_manager is None:
            return {}

        freq = self.wm_manager.config.wm_update_freq
        if global_step % freq != 0:
            return {}

        wm_metrics = self.wm_manager.update_world_model()
        self._wm_metrics_buffer.update(wm_metrics)
        return wm_metrics

    def _wm_get_metrics(self) -> Dict[str, float]:
        """Return accumulated WM metrics and clear buffer."""
        if not hasattr(self, "_wm_metrics_buffer"):
            return {}
        metrics = dict(self._wm_metrics_buffer)
        self._wm_metrics_buffer.clear()
        return metrics
