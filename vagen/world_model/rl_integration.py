"""
World Model RL Integration — three levels of integration.

Level 1 (DynaAugmentor):
    After real rollout, generate imagined transitions from visited states.
    Produces auxiliary reward signal and additional training data.

Level 2 (ImaginedValueEstimator):
    Use H-step imagined rollouts to compute value targets (Dreamer-style).
    Improves advantage estimation for policy gradient updates.

Level 3 (WorldModelRLManager):
    Full integration: shared VLM backbone with separate LoRA adapters for
    world model / actor / critic. Online WM training alongside RL.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vagen.world_model.imagination import (
    ImaginationConfig,
    ImaginedTrajectory,
    LatentImagination,
)
from vagen.world_model.latent_world_model import LatentWorldModel, WorldModelConfig


# =========================================================================
# Level 1: Dyna-style Data Augmentation
# =========================================================================


@dataclass
class DynaConfig:
    """Config for Dyna-style augmentation."""
    enabled: bool = False
    n_imagined_actions: int = 4       # actions to try per real state
    horizon: int = 3                  # imagined rollout depth
    reward_weight: float = 0.5        # weight of imagined rewards in training
    action_pool: List[str] = field(default_factory=lambda: [
        "move up", "move down", "move left", "move right",
    ])


class DynaAugmentor:
    """Level 1: Dyna-style data augmentation.

    After each real rollout, takes the visited observations and generates
    imagined transitions for counterfactual actions. The imagined rewards
    are added as auxiliary training signal.

    Usage in training loop:
        augmentor = DynaAugmentor(imagination_engine, config)
        imagined_data = augmentor.augment(real_observations, real_actions)
        # Mix imagined_data into training batch
    """

    def __init__(self, imagination: LatentImagination, config: DynaConfig):
        self.imagination = imagination
        self.config = config

    def generate_imagined_transitions(
        self,
        observations: list,
        real_actions: List[str],
    ) -> List[ImaginedTrajectory]:
        """Generate imagined rollouts from real visited states.

        Args:
            observations: list of PIL images from real rollout [O_0, ..., O_T]
            real_actions: list of action names taken in real rollout

        Returns:
            List of ImaginedTrajectory, one per (state, counterfactual_action) pair.
        """
        trajectories = []

        for t, obs in enumerate(observations):
            # Sample counterfactual actions (different from the real one)
            real_act = real_actions[t] if t < len(real_actions) else None
            candidate_actions = [
                a for a in self.config.action_pool if a != real_act
            ]
            sampled = random.sample(
                candidate_actions,
                min(self.config.n_imagined_actions, len(candidate_actions)),
            )

            for start_action in sampled:
                # Build full action sequence: start_action + random continuation
                action_seq = [start_action]
                for _ in range(self.config.horizon - 1):
                    action_seq.append(random.choice(self.config.action_pool))

                traj = self.imagination.imagine_from_image(obs, action_seq)
                trajectories.append(traj)

        return trajectories

    def compute_augmented_rewards(
        self,
        trajectories: List[ImaginedTrajectory],
    ) -> Dict[str, float]:
        """Aggregate imagined rewards into auxiliary training signal.

        Returns dict with summary metrics.
        """
        all_rewards = []
        all_returns = []

        for traj in trajectories:
            all_rewards.extend(traj.rewards)
            gamma = self.imagination.config.discount
            all_returns.append(traj.discounted_return(gamma))

        metrics = {}
        if all_rewards:
            metrics["dyna/mean_reward"] = sum(all_rewards) / len(all_rewards)
            metrics["dyna/mean_return"] = sum(all_returns) / len(all_returns)
            metrics["dyna/n_trajectories"] = len(trajectories)
            metrics["dyna/n_transitions"] = len(all_rewards)

        return metrics


# =========================================================================
# Level 2: Imagined Value Estimation
# =========================================================================


@dataclass
class ImaginedValueConfig:
    """Config for imagined value estimation."""
    enabled: bool = False
    horizon: int = 5                  # imagination depth for value targets
    discount: float = 0.99
    lambda_: float = 0.95             # TD(lambda) for imagined returns
    value_weight: float = 0.5         # blend real vs imagined value targets


class ImaginedValueEstimator:
    """Level 2: Dreamer-style imagined value targets.

    Uses the world model to do H-step imagination from each visited state,
    then computes lambda-returns as value targets. These can replace or
    blend with standard GAE advantage estimation.

    Usage in training loop:
        estimator = ImaginedValueEstimator(imagination_engine, config)
        imagined_values = estimator.compute_value_targets(
            observations, actions, real_values
        )
        # Use imagined_values for advantage computation
    """

    def __init__(self, imagination: LatentImagination, config: ImaginedValueConfig):
        self.imagination = imagination
        self.config = config

    def compute_imagined_returns(
        self,
        observation,
        action_sequence: List[str],
    ) -> Tuple[float, List[float]]:
        """Compute lambda-return from an H-step imagined rollout.

        Args:
            observation: PIL image (starting state)
            action_sequence: actions to take in imagination

        Returns:
            (lambda_return, per_step_rewards)
        """
        traj = self.imagination.imagine_from_image(observation, action_sequence)
        rewards = traj.rewards

        if not rewards:
            return 0.0, []

        # Compute TD(lambda) returns
        # G_t^lambda = r_t + gamma * [(1-lambda) * V(s_{t+1}) + lambda * G_{t+1}^lambda]
        # Since we don't have a separate value function in imagination,
        # use the predicted rewards as bootstrap:
        # G_H = r_H (terminal value = 0 or last predicted reward)
        H = len(rewards)
        gamma = self.config.discount
        lam = self.config.lambda_

        # Simple MC return (no separate value function for bootstrap)
        returns = [0.0] * H
        returns[-1] = rewards[-1]
        for t in reversed(range(H - 1)):
            returns[t] = rewards[t] + gamma * returns[t + 1]

        lambda_return = returns[0] if returns else 0.0
        return lambda_return, rewards

    def compute_value_targets(
        self,
        observations: list,
        action_sequences: List[List[str]],
        real_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute imagined value targets for a batch of states.

        Args:
            observations: list of PIL images
            action_sequences: list of action sequences (one per state)
            real_values: (N,) real value estimates from critic (for blending)

        Returns:
            (N,) value targets
        """
        N = len(observations)
        imagined_values = torch.zeros(N)

        for i, (obs, actions) in enumerate(zip(observations, action_sequences)):
            lam_return, _ = self.compute_imagined_returns(obs, actions)
            imagined_values[i] = lam_return

        # Blend with real values if available
        if real_values is not None:
            w = self.config.value_weight
            blended = w * imagined_values + (1 - w) * real_values.cpu()
            return blended

        return imagined_values


# =========================================================================
# Level 3: Full Integration with LoRA Adapter Switching
# =========================================================================


@dataclass
class WorldModelRLConfig:
    """Config for full world model RL integration."""
    enabled: bool = False

    # World model training
    wm_lr: float = 1e-4
    wm_update_freq: int = 1           # update WM every N RL steps
    wm_batch_size: int = 64
    wm_grad_clip: float = 1.0

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Imagination during RL
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    dyna: DynaConfig = field(default_factory=DynaConfig)
    imagined_value: ImaginedValueConfig = field(default_factory=ImaginedValueConfig)


class WorldModelRLManager:
    """Level 3: Full online MBRL with shared backbone and LoRA switching.

    Manages three LoRA adapters on a shared VLM backbone:
      - lora_wm:     world model (Projector training)
      - lora_actor:  policy (PPO/GRPO updates)
      - lora_critic: value function

    Training loop orchestration:
      1. Rollout with lora_actor -> collect real trajectories
      2. Switch to lora_wm -> extract hidden states, run imagination
      3. Update world model on real transitions (online)
      4. Switch to lora_actor -> PPO update with real + imagined data
      5. Switch to lora_critic -> value function update
    """

    def __init__(
        self,
        vlm,
        processor,
        world_model: LatentWorldModel,
        latent_token_id: int,
        config: WorldModelRLConfig,
        device: str = "cuda:0",
    ):
        self.vlm = vlm
        self.processor = processor
        self.world_model = world_model
        self.latent_token_id = latent_token_id
        self.config = config
        self.device = device

        # Sub-components
        self.imagination = LatentImagination(
            vlm, processor, world_model, latent_token_id,
            config.imagination, device,
        )
        self.dyna = DynaAugmentor(self.imagination, config.dyna)
        self.value_estimator = ImaginedValueEstimator(
            self.imagination, config.imagined_value,
        )

        # LoRA adapters (initialized on first call to setup_lora)
        self._lora_initialized = False
        self._active_adapter = None

        # WM optimizer
        self.wm_optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=config.wm_lr,
            weight_decay=0.01,
        )

        # Experience buffer for online WM training
        self.wm_buffer: List[dict] = []

    # ------------------------------------------------------------------
    # LoRA Adapter Management
    # ------------------------------------------------------------------

    def setup_lora(self):
        """Initialize LoRA adapters on the shared VLM backbone.

        Requires peft library. Creates three adapters:
        lora_wm, lora_actor, lora_critic.
        """
        try:
            from peft import LoraConfig, get_peft_model, PeftModel
        except ImportError:
            raise ImportError(
                "peft is required for Level 3 LoRA integration. "
                "Install with: pip install peft"
            )

        if self._lora_initialized:
            return

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Wrap VLM with peft and add named adapters
        self.vlm = get_peft_model(self.vlm, lora_config, adapter_name="lora_actor")
        self.vlm.add_adapter("lora_wm", lora_config)
        self.vlm.add_adapter("lora_critic", lora_config)

        # Start with actor adapter active
        self.vlm.set_adapter("lora_actor")
        self._active_adapter = "lora_actor"
        self._lora_initialized = True

    def switch_adapter(self, name: str):
        """Switch the active LoRA adapter.

        Args:
            name: one of "lora_actor", "lora_wm", "lora_critic"
        """
        if not self._lora_initialized:
            return
        if name != self._active_adapter:
            self.vlm.set_adapter(name)
            self._active_adapter = name

    # ------------------------------------------------------------------
    # Online World Model Training
    # ------------------------------------------------------------------

    def collect_wm_data(
        self,
        observations: list,
        actions: List[str],
        next_observations: list,
        rewards: List[float],
    ):
        """Store real transitions for online WM training.

        Args:
            observations: list of PIL images O_t
            actions: list of action names
            next_observations: list of PIL images O_{t+1}
            rewards: list of scalar rewards
        """
        for obs, act, next_obs, r in zip(
            observations, actions, next_observations, rewards
        ):
            self.wm_buffer.append({
                "obs": obs,
                "action": act,
                "next_obs": next_obs,
                "reward": r,
            })

    @torch.no_grad()
    def _extract_wm_features(self, batch: List[dict]):
        """Extract (hidden_state, vision_target, reward) from a mini-batch.

        Runs VLM forward pass to get h_t at <|latent_token|>, and frozen ViT
        to get target z_{t+1}.
        """
        hidden_states = []
        vision_targets = []
        rewards = []

        for item in batch:
            h_t = self.imagination.real_step(item["obs"], item["action"])
            z_target = self.imagination.get_vision_target(item["next_obs"])
            hidden_states.append(h_t)
            vision_targets.append(z_target)
            rewards.append(item["reward"])

        hidden_states = torch.stack(hidden_states)     # (B, D)
        vision_targets = torch.stack(vision_targets)   # (B, n_vis, D)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        return hidden_states, vision_targets, rewards

    def update_world_model(self, max_samples: Optional[int] = None) -> Dict[str, float]:
        """Update world model on buffered real transitions.

        Returns training metrics dict.
        """
        if not self.wm_buffer:
            return {}

        bs = self.config.wm_batch_size
        n = max_samples or len(self.wm_buffer)
        samples = random.sample(self.wm_buffer, min(n, len(self.wm_buffer)))

        total_loss = 0.0
        total_cos = 0.0
        n_batches = 0

        # Switch to WM adapter for feature extraction
        self.switch_adapter("lora_wm")

        for i in range(0, len(samples), bs):
            batch = samples[i:i + bs]
            h, z_tgt, r = self._extract_wm_features(batch)
            h = h.to(self.device)
            z_tgt = z_tgt.to(self.device)
            r = r.to(self.device)

            # World model forward + loss
            _, _, (loss, metrics) = self.world_model(h, z_tgt, r)

            self.wm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.world_model.parameters(), self.config.wm_grad_clip,
            )
            self.wm_optimizer.step()

            total_loss += metrics["wm/total_loss"]
            total_cos += metrics["wm/cosine_sim_mean"]
            n_batches += 1

        # Switch back to actor
        self.switch_adapter("lora_actor")

        if n_batches == 0:
            return {}

        return {
            "wm_online/loss": total_loss / n_batches,
            "wm_online/cosine_sim": total_cos / n_batches,
            "wm_online/buffer_size": len(self.wm_buffer),
        }

    # ------------------------------------------------------------------
    # Full step: imagination + augmentation + value estimation
    # ------------------------------------------------------------------

    def imagination_step(
        self,
        observations: list,
        actions: List[str],
        action_pool: Optional[List[str]] = None,
    ) -> Tuple[List[ImaginedTrajectory], Dict[str, float]]:
        """Run imagination from real states, return trajectories + metrics.

        This is the main entry point called from the RL training loop after
        real rollout completes.
        """
        self.switch_adapter("lora_wm")

        # Dyna augmentation
        dyna_trajs = []
        dyna_metrics = {}
        if self.config.dyna.enabled:
            dyna_trajs = self.dyna.generate_imagined_transitions(
                observations, actions,
            )
            dyna_metrics = self.dyna.compute_augmented_rewards(dyna_trajs)

        self.switch_adapter("lora_actor")
        return dyna_trajs, dyna_metrics

    def compute_imagined_value_targets(
        self,
        observations: list,
        action_sequences: List[List[str]],
        real_values: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute imagined value targets if Level 2 is enabled."""
        if not self.config.imagined_value.enabled:
            return None

        self.switch_adapter("lora_wm")
        targets = self.value_estimator.compute_value_targets(
            observations, action_sequences, real_values,
        )
        self.switch_adapter("lora_actor")
        return targets
