"""
Multi-step latent rollout validation.

Tests the core hypothesis: z_hat from the world model can be injected back
into the VLM via inputs_embeds, enabling multi-step imagination without real
images.

For each rollout of length K:
  Step 0 (grounded): real O_0 + a_0 -> VLM -> h_0 -> Projector -> z_hat_1
  Step k (imagined): z_hat_k + a_k -> VLM(inputs_embeds) -> h_k -> Projector -> z_hat_{k+1}

At each step, compare z_hat with the real ViT output for ground-truth O_{t+1}.
"""

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vagen.world_model.latent_world_model import (
    LatentWorldModel,
    WorldModelConfig,
    LATENT_TOKEN,
)


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

ACTION_NAMES = {
    0: "no operation",
    1: "push up", 2: "push down", 3: "push left", 4: "push right",
    5: "move up", 6: "move down", 7: "move left", 8: "move right",
}


def build_prompt_messages(action_name: str):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Action taken: {action_name}\n{LATENT_TOKEN}"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_vlm(model_path, device):
    print(f"Loading VLM from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [LATENT_TOKEN]}
    )
    if num_added > 0:
        processor.tokenizer = tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    print(f"  latent_token_id={latent_token_id}, image_token_id={model.config.image_token_id}")
    return model, tokenizer, processor, latent_token_id


def setup_world_model(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    config = WorldModelConfig(**cfg)
    wm = LatentWorldModel(config).to(device)
    wm.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    )
    wm.eval()
    print(f"  World model loaded ({sum(p.numel() for p in wm.parameters())/1e6:.1f}M params)")
    return wm


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

def get_vision_target(vlm, processor, image, device):
    """Run frozen ViT + PatchMerger on a real image -> (n_vis, D)."""
    image_inputs = processor.image_processor([image], return_tensors="pt").to(device)
    pixel_values = image_inputs["pixel_values"].to(dtype=vlm.visual.dtype)
    image_grid_thw = image_inputs["image_grid_thw"]
    with torch.no_grad():
        out = vlm.visual(pixel_values, grid_thw=image_grid_thw)
    return out.cpu().float()  # (n_vis, D)


def real_forward_step(vlm, processor, image, action_name, latent_token_id, device):
    """Real observation step: full VLM pipeline with an actual image.

    Returns h_t at <|latent_token|> position, shape (D,).
    """
    messages = build_prompt_messages(action_name)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)

    input_ids = inputs["input_ids"]
    last_hidden = outputs.hidden_states[-1]  # (1, seq, D)
    pos = (input_ids[0] == latent_token_id).nonzero(as_tuple=True)[0][-1].item()
    return last_hidden[0, pos].cpu().float()


def imagined_forward_step(vlm, processor, z_hat, action_name, latent_token_id, device):
    """Imagined step: inject z_hat as visual tokens via inputs_embeds.

    Uses a dummy image to obtain the correct tokenization template (input_ids
    with image_token_id placeholders and correct image_grid_thw for 3D mRoPE),
    then replaces image token embeddings with z_hat.

    Args:
        z_hat: (n_vis, D) predicted visual tokens from Projector

    Returns:
        h_t at <|latent_token|> position, shape (D,).
    """
    messages = build_prompt_messages(action_name)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    # Dummy image with same resolution -> correct tokenization + grid_thw
    dummy_image = Image.new("RGB", (96, 96), color=(0, 0, 0))
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]           # (1, seq_len)
    attention_mask = inputs["attention_mask"]  # (1, seq_len)
    image_grid_thw = inputs["image_grid_thw"] # (1, 3)

    # Build inputs_embeds: text embeddings with z_hat at image positions
    inputs_embeds = vlm.model.embed_tokens(input_ids)  # (1, seq_len, D)

    image_token_id = vlm.config.image_token_id
    image_mask = (input_ids == image_token_id)  # (1, seq_len)
    n_img_tokens = image_mask.sum().item()
    assert n_img_tokens == z_hat.shape[0], (
        f"Token count mismatch: {n_img_tokens} image placeholders vs "
        f"{z_hat.shape[0]} z_hat tokens"
    )

    # Replace image placeholder embeddings with z_hat (masked_scatter)
    z_hat_bf16 = z_hat.to(device=device, dtype=inputs_embeds.dtype)
    image_mask_3d = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask_3d, z_hat_bf16)

    # Forward pass:
    #   input_ids   -> used ONLY for position_ids (3D mRoPE) computation
    #   inputs_embeds -> used for the actual forward (skips embed_tokens + ViT)
    #   image_grid_thw -> needed for 2D spatial coords in mRoPE
    #   pixel_values NOT passed -> ViT never runs
    with torch.no_grad():
        outputs = vlm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

    last_hidden = outputs.hidden_states[-1]
    pos = (input_ids[0] == latent_token_id).nonzero(as_tuple=True)[0][-1].item()
    return last_hidden[0, pos].cpu().float()


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_single_rollout(vlm, processor, world_model, observations, actions,
                       latent_token_id, device):
    """Run a K-step rollout comparing imagined vs ground-truth.

    Args:
        observations: list of K+1 PIL images [O_0, O_1, ..., O_K]
        actions:      list of K ints [a_0, ..., a_{K-1}]

    Returns list of per-step dicts with cosine sim metrics.
    """
    K = len(actions)
    assert len(observations) == K + 1

    results = []
    z_hat = None

    for step in range(K):
        action_name = ACTION_NAMES[actions[step]]

        # Ground-truth ViT output for O_{t+1}
        z_real = get_vision_target(vlm, processor, observations[step + 1], device)

        # VLM forward: real image at step 0, imagined thereafter
        if step == 0:
            h_t = real_forward_step(
                vlm, processor, observations[0], action_name,
                latent_token_id, device,
            )
        else:
            h_t = imagined_forward_step(
                vlm, processor, z_hat, action_name,
                latent_token_id, device,
            )

        # Projector: h_t -> z_hat_{t+1}
        with torch.no_grad():
            z_hat = world_model.predict_next_latent(
                h_t.unsqueeze(0).to(device)
            ).squeeze(0).cpu().float()  # (n_vis, D)

        # Per-token cosine similarity
        cos_per_token = F.cosine_similarity(z_hat, z_real, dim=-1)  # (n_vis,)
        cos_mean = cos_per_token.mean().item()

        # Pooled cosine similarity
        z_hat_p = F.normalize(z_hat.mean(0, keepdim=True), dim=-1)
        z_real_p = F.normalize(z_real.mean(0, keepdim=True), dim=-1)
        cos_pooled = (z_hat_p * z_real_p).sum().item()

        step_type = "real" if step == 0 else f"imagined-{step}"
        results.append({
            "step": step,
            "type": step_type,
            "action": action_name,
            "cos_sim_token": cos_mean,
            "cos_sim_pooled": cos_pooled,
        })

        print(f"  Step {step} ({step_type:>12s}) | {action_name:<15s} | "
              f"token_cos={cos_mean:.4f} | pooled_cos={cos_pooled:.4f}")

    return results


# ---------------------------------------------------------------------------
# Trajectory collection from Sokoban env
# ---------------------------------------------------------------------------

def collect_trajectory(n_steps, dim_room=(6, 6), num_boxes=1):
    """Run a random episode, return observations and actions."""
    from gym_sokoban.envs import SokobanEnv

    env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes)
    env.reset()

    observations = [Image.fromarray(env.render("rgb_array"))]
    actions = []

    for _ in range(n_steps):
        action = random.choice([5, 6, 7, 8])  # move only (safer, always valid)
        _, _, done, _ = env.step(action)
        observations.append(Image.fromarray(env.render("rgb_array")))
        actions.append(action)
        if done:
            break

    return observations, actions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_rollout_results(all_results, output_path):
    """Plot per-step cosine similarity across multiple rollouts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Aggregate by step
    max_steps = max(len(r) for r in all_results)
    step_token_cos = {s: [] for s in range(max_steps)}
    step_pooled_cos = {s: [] for s in range(max_steps)}

    for rollout in all_results:
        for res in rollout:
            s = res["step"]
            step_token_cos[s].append(res["cos_sim_token"])
            step_pooled_cos[s].append(res["cos_sim_pooled"])

    steps = sorted(step_token_cos.keys())
    token_means = [np.mean(step_token_cos[s]) for s in steps]
    token_stds = [np.std(step_token_cos[s]) for s in steps]
    pooled_means = [np.mean(step_pooled_cos[s]) for s in steps]
    pooled_stds = [np.std(step_pooled_cos[s]) for s in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(steps, token_means, yerr=token_stds, marker="o",
                label="Per-token cosine sim", capsize=4)
    ax.errorbar(steps, pooled_means, yerr=pooled_stds, marker="s",
                label="Pooled cosine sim", capsize=4)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(0.25, ax.get_ylim()[0] + 0.02, "real", ha="center", fontsize=9, color="gray")
    ax.text(1.5, ax.get_ylim()[0] + 0.02, "imagined", ha="center", fontsize=9, color="gray")

    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Cosine Similarity with Ground Truth")
    ax.set_title("Multi-Step Latent Rollout: Error Compounding")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}\n({'real' if s == 0 else 'img'})" for s in steps])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved rollout plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-step rollout validation")
    parser.add_argument("--model_path", default="/workspace/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--wm_dir", default="/workspace/VAGEN/checkpoints/world_model/sokoban")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--rollout_len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir",
                        default="/workspace/VAGEN/checkpoints/world_model/sokoban/rollout_eval")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    vlm, tokenizer, processor, latent_token_id = setup_vlm(args.model_path, args.device)
    wm = setup_world_model(args.wm_dir, args.device)

    all_results = []
    for i in range(args.n_rollouts):
        print(f"\n{'='*60}")
        print(f"Rollout {i+1}/{args.n_rollouts}")
        print(f"{'='*60}")

        observations, actions = collect_trajectory(
            args.rollout_len, dim_room=(6, 6), num_boxes=1,
        )
        actual_len = len(actions)
        if actual_len == 0:
            print("  Empty trajectory, skipping")
            continue

        results = run_single_rollout(
            vlm, processor, wm, observations, actions,
            latent_token_id, args.device,
        )
        all_results.append(results)

    # Aggregate summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    max_steps = max(len(r) for r in all_results)
    for s in range(max_steps):
        vals = [r[s]["cos_sim_token"] for r in all_results if s < len(r)]
        pvals = [r[s]["cos_sim_pooled"] for r in all_results if s < len(r)]
        if vals:
            label = "real    " if s == 0 else f"img-{s:>3d} "
            print(f"  Step {s} ({label}): "
                  f"token_cos={np.mean(vals):.4f}+-{np.std(vals):.4f}  "
                  f"pooled_cos={np.mean(pvals):.4f}+-{np.std(pvals):.4f}  "
                  f"(n={len(vals)})")

    # Save results
    results_path = os.path.join(args.output_dir, "rollout_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot
    plot_rollout_results(all_results, os.path.join(args.output_dir, "rollout_cosine.png"))


if __name__ == "__main__":
    main()
