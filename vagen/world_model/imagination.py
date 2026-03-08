"""
Latent Imagination Engine for Model-Based VLM RL.

Runs multi-step imagined rollouts by predicting next visual tokens with the
world model Projector, then injecting them back into the VLM via inputs_embeds.

Used by all three integration levels:
  - Level 1 (Dyna): generate extra training data from imagined transitions
  - Level 2 (Value): compute imagined returns for better value targets
  - Level 3 (LoRA): full online imagination with adapter switching
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from vagen.world_model.latent_world_model import LatentWorldModel, LATENT_TOKEN


SYSTEM_PROMPT = (
    "You are a world model for the Sokoban puzzle game.\n\n"
    "Rules:\n"
    "- The player can move in four directions: up, down, left, right.\n"
    "- If a box is in the moving direction and the space behind it is free, "
    "the player pushes the box forward.\n"
    "- If a wall or another box blocks the way, the move has no effect.\n"
    "- The goal is to push all boxes onto the target locations.\n\n"
    "Your task: Given the current board image and the agent's action, "
    "predict the resulting next board state."
)


@dataclass
class ImaginationConfig:
    """Configuration for the imagination engine."""
    horizon: int = 5                    # max imagination steps
    image_size: Tuple[int, int] = (96, 96)  # env render resolution
    system_prompt: str = SYSTEM_PROMPT
    discount: float = 0.99             # gamma for imagined returns


@dataclass
class ImaginedStep:
    """One step of an imagined trajectory."""
    step: int
    action_name: str
    h_t: torch.Tensor           # (D,) hidden state at <|latent_token|>
    z_hat: torch.Tensor         # (n_vis, D) predicted visual tokens
    predicted_reward: float     # scalar reward from RewardHead
    is_real: bool               # True for the grounded first step


@dataclass
class ImaginedTrajectory:
    """A full imagined trajectory."""
    steps: List[ImaginedStep] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [s.predicted_reward for s in self.steps]

    @property
    def length(self) -> int:
        return len(self.steps)

    def discounted_return(self, gamma: float = 0.99) -> float:
        G = 0.0
        for s in reversed(self.steps):
            G = s.predicted_reward + gamma * G
        return G


class LatentImagination:
    """Multi-step latent imagination engine.

    Given a starting observation (real image) or latent state (z_hat),
    runs H-step imagined rollouts using the world model Projector to
    predict visual tokens and the VLM backbone as the recurrence.
    """

    def __init__(
        self,
        vlm,
        processor,
        world_model: LatentWorldModel,
        latent_token_id: int,
        config: ImaginationConfig,
        device: str = "cuda:0",
    ):
        self.vlm = vlm
        self.processor = processor
        self.world_model = world_model
        self.latent_token_id = latent_token_id
        self.config = config
        self.device = device

        # Cache the image token id from model config
        self.image_token_id = vlm.config.image_token_id

        # Pre-compute dummy image template for imagined steps
        self._dummy_image = Image.new(
            "RGB", config.image_size, color=(0, 0, 0)
        )

    def _build_prompt_messages(self, action_name: str):
        return [
            {"role": "system", "content": [{"type": "text", "text": self.config.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Action taken: {action_name}\n{LATENT_TOKEN}"},
                ],
            },
        ]

    def _get_prompt_inputs(self, image, action_name: str):
        """Process image + action into model inputs."""
        messages = self._build_prompt_messages(action_name)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(self.device)
        return inputs

    def _extract_h_at_latent_token(self, outputs, input_ids):
        """Extract hidden state at <|latent_token|> position."""
        last_hidden = outputs.hidden_states[-1]  # (1, seq, D)
        pos = (input_ids[0] == self.latent_token_id).nonzero(as_tuple=True)[0][-1].item()
        return last_hidden[0, pos]

    # ------------------------------------------------------------------
    # Core forward passes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def real_step(self, image: Image.Image, action_name: str) -> torch.Tensor:
        """Grounded step: real image through full VLM pipeline.

        Returns h_t at <|latent_token|>, shape (D,).
        """
        inputs = self._get_prompt_inputs(image, action_name)
        outputs = self.vlm(**inputs, output_hidden_states=True, return_dict=True)
        h_t = self._extract_h_at_latent_token(outputs, inputs["input_ids"])
        return h_t.float()

    @torch.no_grad()
    def imagined_step(self, z_hat: torch.Tensor, action_name: str) -> torch.Tensor:
        """Imagined step: inject z_hat as visual tokens via inputs_embeds.

        Args:
            z_hat: (n_vis, D) predicted visual tokens from Projector

        Returns h_t at <|latent_token|>, shape (D,).
        """
        # Use dummy image for correct tokenization template
        inputs = self._get_prompt_inputs(self._dummy_image, action_name)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_grid_thw = inputs["image_grid_thw"]

        # Build inputs_embeds: text embeddings with z_hat at image positions
        inputs_embeds = self.vlm.model.embed_tokens(input_ids)

        image_mask = (input_ids == self.image_token_id)
        n_img = image_mask.sum().item()
        assert n_img == z_hat.shape[0], (
            f"Token count mismatch: {n_img} placeholders vs {z_hat.shape[0]} z_hat tokens"
        )

        # masked_scatter replaces image placeholder embeddings with z_hat
        z_hat_cast = z_hat.to(device=self.device, dtype=inputs_embeds.dtype)
        mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask_3d, z_hat_cast)

        # Forward: input_ids for position_ids (3D mRoPE), inputs_embeds for computation
        outputs = self.vlm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        h_t = self._extract_h_at_latent_token(outputs, input_ids)
        return h_t.float()

    @torch.no_grad()
    def get_vision_target(self, image: Image.Image) -> torch.Tensor:
        """Extract frozen ViT output for a real image -> (n_vis, D)."""
        img_inputs = self.processor.image_processor(
            [image], return_tensors="pt",
        ).to(self.device)
        pixel_values = img_inputs["pixel_values"].to(dtype=self.vlm.visual.dtype)
        grid_thw = img_inputs["image_grid_thw"]
        out = self.vlm.visual(pixel_values, grid_thw=grid_thw)
        return out.float()

    # ------------------------------------------------------------------
    # Multi-step imagination
    # ------------------------------------------------------------------

    @torch.no_grad()
    def imagine_from_hidden(
        self,
        h_t: torch.Tensor,
        action_names: List[str],
    ) -> ImaginedTrajectory:
        """Run H-step imagination from a hidden state.

        Args:
            h_t: (D,) hidden state (e.g., from a real step)
            action_names: list of action strings for each imagined step

        Returns:
            ImaginedTrajectory with H steps.
        """
        traj = ImaginedTrajectory()
        H = min(len(action_names), self.config.horizon)

        for step in range(H):
            # Projector: h_t -> z_hat, r_hat
            z_hat = self.world_model.predict_next_latent(
                h_t.unsqueeze(0).to(self.device)
            ).squeeze(0).float()  # (n_vis, D)

            r_hat = self.world_model.predict_reward(
                h_t.unsqueeze(0).to(self.device)
            ).item()

            traj.steps.append(ImaginedStep(
                step=step,
                action_name=action_names[step],
                h_t=h_t.cpu(),
                z_hat=z_hat.cpu(),
                predicted_reward=r_hat,
                is_real=False,
            ))

            if step < H - 1:
                # Next step: inject z_hat into VLM
                h_t = self.imagined_step(z_hat, action_names[step + 1])

        return traj

    @torch.no_grad()
    def imagine_from_image(
        self,
        image: Image.Image,
        action_names: List[str],
    ) -> ImaginedTrajectory:
        """Full rollout: real first step, then imagined continuation.

        Args:
            image: starting observation O_0
            action_names: [a_0, a_1, ..., a_{H-1}] actions for each step

        Returns:
            ImaginedTrajectory where step 0 is grounded, rest imagined.
        """
        if len(action_names) == 0:
            return ImaginedTrajectory()

        # Step 0: grounded on real image
        h_t = self.real_step(image, action_names[0])

        z_hat = self.world_model.predict_next_latent(
            h_t.unsqueeze(0).to(self.device)
        ).squeeze(0).float()

        r_hat = self.world_model.predict_reward(
            h_t.unsqueeze(0).to(self.device)
        ).item()

        traj = ImaginedTrajectory()
        traj.steps.append(ImaginedStep(
            step=0,
            action_name=action_names[0],
            h_t=h_t.cpu(),
            z_hat=z_hat.cpu(),
            predicted_reward=r_hat,
            is_real=True,
        ))

        # Steps 1+: imagined
        if len(action_names) > 1:
            h_t = self.imagined_step(z_hat, action_names[1])
            rest = self.imagine_from_hidden(h_t, action_names[1:])
            traj.steps.extend(rest.steps)

        return traj

    @torch.no_grad()
    def imagine_batch(
        self,
        images: List[Image.Image],
        action_sequences: List[List[str]],
    ) -> List[ImaginedTrajectory]:
        """Imagine trajectories for a batch of starting states.

        Note: processes sequentially (batch=1 VLM forward). For higher
        throughput, see imagine_batch_parallel which batches VLM calls.
        """
        trajectories = []
        for img, actions in zip(images, action_sequences):
            traj = self.imagine_from_image(img, actions)
            trajectories.append(traj)
        return trajectories
