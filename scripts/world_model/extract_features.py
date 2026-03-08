"""
Extract features for world model training from collected Sokoban transitions.

Two passes:
1. Extract LLM hidden states at <|latent_token|> position for each (O_t, a_t)
2. Extract frozen ViT target latents for each O_{t+1}

Saves pre-computed tensors to disk so Projector training doesn't need the VLM.
"""

import argparse
import os
import pickle
import math

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

# Qwen2.5-VL uses this model class
from transformers import Qwen2_5_VLForConditionalGeneration

from vagen.world_model.latent_world_model import LATENT_TOKEN, compute_n_visual_tokens


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


def build_prompt_messages(action_name: str):
    """Build chat messages for the world model prompt."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": f"Action taken: {action_name}\n{LATENT_TOKEN}",
                },
            ],
        },
    ]


def setup_model_and_processor(model_path, device="cuda:0"):
    """Load model, tokenizer, processor. Add <|latent_token|> to vocab."""
    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    # Add the special latent token
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_TOKEN]})
    if num_added > 0:
        print(f"Added {num_added} special token(s): {LATENT_TOKEN}")
        # Also update the processor's tokenizer
        processor.tokenizer = tokenizer

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    # Resize embeddings if new tokens were added
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    latent_token_id = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)
    print(f"Latent token ID: {latent_token_id}")

    return model, tokenizer, processor, latent_token_id


def extract_hidden_states_batch(
    model, processor, tokenizer, latent_token_id,
    images, action_names, device="cuda:0",
):
    """Extract LLM hidden states at <|latent_token|> position for a batch.

    Args:
        images: list of PIL.Image (O_t)
        action_names: list of str

    Returns:
        hidden_states: (batch, hidden_dim) tensor
    """
    all_texts = []
    all_images = []
    for img, act in zip(images, action_names):
        messages = build_prompt_messages(act)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        all_texts.append(text)
        all_images.append(img)

    inputs = processor(
        text=all_texts,
        images=all_images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Get last layer hidden states
    last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

    # Find <|latent_token|> positions in each sequence
    input_ids = inputs["input_ids"]  # (batch, seq_len)
    batch_size = input_ids.shape[0]
    hidden_dim = last_hidden.shape[-1]
    result = torch.zeros(batch_size, hidden_dim, device=device, dtype=last_hidden.dtype)

    for i in range(batch_size):
        positions = (input_ids[i] == latent_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError(
                f"No <|latent_token|> found in sequence {i}. "
                f"Token ID {latent_token_id} not in input_ids."
            )
        # Take the last occurrence
        pos = positions[-1].item()
        result[i] = last_hidden[i, pos]

    return result.cpu().float()


def extract_vision_targets_batch(model, processor, images, device="cuda:0"):
    """Extract frozen ViT + PatchMerger output for a batch of images.

    Args:
        images: list of PIL.Image (O_{t+1})

    Returns:
        vision_targets: (batch, n_visual_tokens, hidden_dim) tensor
    """
    # Process images using the image processor
    image_inputs = processor.image_processor(images, return_tensors="pt").to(device)
    pixel_values = image_inputs["pixel_values"].to(dtype=model.visual.dtype)
    image_grid_thw = image_inputs["image_grid_thw"]

    with torch.no_grad():
        vision_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        # vision_output: (total_merged_tokens, hidden_dim)

    # Split per image (each image produces n_visual_tokens)
    tokens_per_image = []
    for thw in image_grid_thw:
        t, h, w = thw.tolist()
        merge = model.visual.spatial_merge_size
        n = t * (h // merge) * (w // merge)
        tokens_per_image.append(n)

    targets = vision_output.split(tokens_per_image, dim=0)
    # Stack into (batch, n_visual_tokens, hidden_dim)
    targets = torch.stack(targets, dim=0)

    return targets.cpu().float()


def main():
    parser = argparse.ArgumentParser(description="Extract features for world model")
    parser.add_argument("--model_path", type=str,
                        default="/workspace/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_path", type=str,
                        default="/workspace/VAGEN/data/world_model/sokoban/transitions.pkl")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/VAGEN/data/world_model/sokoban/features")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    args = parser.parse_args()

    # Load transitions
    print(f"Loading transitions from {args.data_path} ...")
    with open(args.data_path, "rb") as f:
        transitions = pickle.load(f)

    if args.max_samples > 0:
        transitions = transitions[:args.max_samples]
    print(f"Processing {len(transitions)} transitions")

    # Setup model
    model, tokenizer, processor, latent_token_id = setup_model_and_processor(
        args.model_path, args.device
    )

    os.makedirs(args.output_dir, exist_ok=True)
    n = len(transitions)
    bs = args.batch_size

    # --- Pass 1: Extract LLM hidden states ---
    print("\n=== Pass 1: Extracting LLM hidden states ===")
    all_hidden = []
    all_rewards = []
    all_dones = []

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = transitions[start:end]

        images = [t["obs_image"] for t in batch]
        actions = [t["action_name"] for t in batch]

        hidden = extract_hidden_states_batch(
            model, processor, tokenizer, latent_token_id,
            images, actions, device=args.device,
        )
        all_hidden.append(hidden)
        all_rewards.extend([t["reward"] for t in batch])
        all_dones.extend([t["done"] for t in batch])

        if (start // bs + 1) % 20 == 0 or end == n:
            print(f"  Hidden states: {end}/{n}")

    all_hidden = torch.cat(all_hidden, dim=0)  # (N, hidden_dim)
    all_rewards = torch.tensor(all_rewards, dtype=torch.float32)
    all_dones = torch.tensor(all_dones, dtype=torch.bool)

    # --- Pass 2: Extract ViT targets ---
    print("\n=== Pass 2: Extracting ViT target latents ===")
    all_targets = []

    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = transitions[start:end]

        next_images = [t["next_obs_image"] for t in batch]
        targets = extract_vision_targets_batch(
            model, processor, next_images, device=args.device,
        )
        all_targets.append(targets)

        if (start // bs + 1) % 20 == 0 or end == n:
            print(f"  ViT targets: {end}/{n}")

    all_targets = torch.cat(all_targets, dim=0)  # (N, n_vis_tokens, hidden_dim)

    # --- Save ---
    print(f"\nSaving features to {args.output_dir} ...")
    torch.save(all_hidden, os.path.join(args.output_dir, "hidden_states.pt"))
    torch.save(all_targets, os.path.join(args.output_dir, "vision_targets.pt"))
    torch.save(all_rewards, os.path.join(args.output_dir, "rewards.pt"))
    torch.save(all_dones, os.path.join(args.output_dir, "dones.pt"))

    # Save action info for evaluation
    action_info = [{"action": t["action"], "action_name": t["action_name"]} for t in transitions]
    with open(os.path.join(args.output_dir, "action_info.pkl"), "wb") as f:
        pickle.dump(action_info, f)

    # Save obs/next_obs images as a separate file for NN retrieval eval
    obs_images = [t["obs_image"] for t in transitions]
    next_obs_images = [t["next_obs_image"] for t in transitions]
    with open(os.path.join(args.output_dir, "images.pkl"), "wb") as f:
        pickle.dump({"obs": obs_images, "next_obs": next_obs_images}, f)

    print(f"Done! Shapes:")
    print(f"  hidden_states: {all_hidden.shape}")
    print(f"  vision_targets: {all_targets.shape}")
    print(f"  rewards: {all_rewards.shape}")


if __name__ == "__main__":
    main()
