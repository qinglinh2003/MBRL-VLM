"""
Evaluate trained world model with:
1. Training curve plot (cosine similarity over epochs)
2. Nearest-neighbor retrieval visualization
3. Action discrimination test
"""

import argparse
import json
import os
import pickle
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from vagen.world_model.latent_world_model import LatentWorldModel, WorldModelConfig


def plot_training_curves(history_path, output_path):
    """Plot training and validation cosine similarity over epochs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_cos = [h["train/cosine_sim"] for h in history]
    val_cos = [h["val/cosine_sim"] for h in history]
    train_loss = [h["train/loss"] for h in history]
    val_loss = [h["val/loss"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_cos, label="Train", linewidth=2)
    axes[0].plot(epochs, val_cos, label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Cosine Similarity (higher = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="random baseline")

    axes[1].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[1].plot(epochs, val_loss, label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Total Loss (lower = better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def nn_retrieval_eval(model, features_dir, output_dir, device, n_examples=20):
    """Nearest-neighbor retrieval: predict z_hat, find closest real z in dataset."""
    hidden_states = torch.load(os.path.join(features_dir, "hidden_states.pt"), map_location="cpu")
    vision_targets = torch.load(os.path.join(features_dir, "vision_targets.pt"), map_location="cpu")

    with open(os.path.join(features_dir, "images.pkl"), "rb") as f:
        images_data = pickle.load(f)
    with open(os.path.join(features_dir, "action_info.pkl"), "rb") as f:
        action_info = pickle.load(f)

    obs_images = images_data["obs"]
    next_obs_images = images_data["next_obs"]

    # Build retrieval database: pool each target to a single vector
    # target shape: (N, n_vis_tokens, hidden_dim) -> pool to (N, hidden_dim)
    target_pooled = vision_targets.mean(dim=1)  # (N, hidden_dim)
    target_normed = F.normalize(target_pooled, dim=-1)

    # Predict for a random subset
    n = len(hidden_states)
    indices = random.sample(range(n), min(n_examples, n))

    model.eval()
    results = []

    for idx in indices:
        h = hidden_states[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            z_hat = model.predict_next_latent(h)  # (1, n_vis, hidden_dim)

        # Pool predicted latent
        z_hat_pooled = z_hat.mean(dim=1).cpu()  # (1, hidden_dim)
        z_hat_normed = F.normalize(z_hat_pooled, dim=-1)

        # Find nearest neighbor
        sims = (z_hat_normed @ target_normed.T).squeeze(0)  # (N,)
        nn_idx = sims.argmax().item()
        nn_sim = sims[nn_idx].item()

        # Also compute similarity with the actual target
        actual_target_pooled = target_pooled[idx].unsqueeze(0)
        actual_target_normed = F.normalize(actual_target_pooled, dim=-1)
        actual_sim = (z_hat_normed @ actual_target_normed.T).item()

        results.append({
            "query_idx": idx,
            "nn_idx": nn_idx,
            "nn_sim": nn_sim,
            "actual_sim": actual_sim,
            "action": action_info[idx]["action_name"],
            "nn_is_correct": nn_idx == idx,
        })

    # Visualize
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_show = min(10, len(results))
        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        for i, res in enumerate(results[:n_show]):
            idx = res["query_idx"]
            nn_idx = res["nn_idx"]

            axes[i, 0].imshow(obs_images[idx])
            axes[i, 0].set_title(f"O_t (action: {res['action']})", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(next_obs_images[idx])
            axes[i, 1].set_title(f"O_{{t+1}} (actual)", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(next_obs_images[nn_idx])
            axes[i, 2].set_title(
                f"NN retrieval (sim={res['nn_sim']:.3f})",
                fontsize=9,
                color="green" if res["nn_is_correct"] else "red",
            )
            axes[i, 2].axis("off")

        plt.suptitle("World Model NN Retrieval: O_t -> action -> predicted O_{t+1}", fontsize=12)
        plt.tight_layout()
        output_path = os.path.join(output_dir, "nn_retrieval.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved NN retrieval visualization to {output_path}")
    except ImportError:
        print("matplotlib not available, skipping visualization")

    # Print summary
    mean_actual_sim = np.mean([r["actual_sim"] for r in results])
    mean_nn_sim = np.mean([r["nn_sim"] for r in results])
    correct_rate = np.mean([r["nn_is_correct"] for r in results])
    print(f"\nNN Retrieval Summary ({len(results)} samples):")
    print(f"  Mean cosine sim with actual target: {mean_actual_sim:.4f}")
    print(f"  Mean cosine sim with NN: {mean_nn_sim:.4f}")
    print(f"  Exact match rate: {correct_rate:.2%}")

    return results


def action_discrimination_test(model, features_dir, device, n_states=100):
    """Test if same state + different actions produce distinguishable predictions."""
    hidden_states = torch.load(os.path.join(features_dir, "hidden_states.pt"), map_location="cpu")
    vision_targets = torch.load(os.path.join(features_dir, "vision_targets.pt"), map_location="cpu")

    with open(os.path.join(features_dir, "action_info.pkl"), "rb") as f:
        action_info = pickle.load(f)

    # Group transitions by (episode, step) -- same state, different actions
    # Each state has 9 consecutive transitions (all actions)
    n_total = len(hidden_states)
    n_states_available = n_total // 9
    state_indices = random.sample(range(n_states_available), min(n_states, n_states_available))

    model.eval()
    intra_state_sims = []   # similarity between predictions from same state, different actions
    inter_state_sims = []   # similarity between predictions from different states
    correct_vs_wrong = []   # sim(pred, correct_target) vs sim(pred, wrong_target)

    for si in state_indices:
        base = si * 9
        hs = hidden_states[base:base+9].to(device)  # (9, hidden_dim)
        targets = vision_targets[base:base+9]         # (9, n_vis, hidden_dim)

        with torch.no_grad():
            preds = model.predict_next_latent(hs)  # (9, n_vis, hidden_dim)

        # Pool to (9, hidden_dim) for comparison
        preds_pooled = F.normalize(preds.mean(dim=1).cpu(), dim=-1)
        targets_pooled = F.normalize(targets.mean(dim=1), dim=-1)

        # Intra-state: cosine between all pairs of predictions from same state
        sim_matrix = preds_pooled @ preds_pooled.T  # (9, 9)
        mask = ~torch.eye(9, dtype=torch.bool)
        intra_state_sims.extend(sim_matrix[mask].tolist())

        # Correct vs wrong: for each action, sim with correct vs random wrong target
        for a in range(9):
            correct_sim = (preds_pooled[a] @ targets_pooled[a]).item()
            wrong_idx = random.choice([j for j in range(9) if j != a])
            wrong_sim = (preds_pooled[a] @ targets_pooled[wrong_idx]).item()
            correct_vs_wrong.append(correct_sim - wrong_sim)

    # Inter-state: random pairs from different states
    for _ in range(len(intra_state_sims)):
        s1, s2 = random.sample(range(n_states_available), 2)
        a1, a2 = random.randint(0, 8), random.randint(0, 8)
        i1, i2 = s1 * 9 + a1, s2 * 9 + a2
        if i1 >= n_total or i2 >= n_total:
            continue
        h1 = hidden_states[i1].unsqueeze(0).to(device)
        h2 = hidden_states[i2].unsqueeze(0).to(device)
        with torch.no_grad():
            p1 = model.predict_next_latent(h1).mean(dim=1).cpu()
            p2 = model.predict_next_latent(h2).mean(dim=1).cpu()
        sim = F.cosine_similarity(p1, p2).item()
        inter_state_sims.append(sim)

    print(f"\nAction Discrimination Test ({len(state_indices)} states):")
    print(f"  Intra-state sim (same state, diff actions): {np.mean(intra_state_sims):.4f}")
    print(f"  Inter-state sim (diff states): {np.mean(inter_state_sims):.4f}")
    print(f"  Mean (correct_sim - wrong_sim): {np.mean(correct_vs_wrong):.4f}")
    print(f"  Correct > Wrong rate: {np.mean([x > 0 for x in correct_vs_wrong]):.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--checkpoint", type=str,
                        default="/workspace/VAGEN/checkpoints/world_model/sokoban/best_model.pt")
    parser.add_argument("--features_dir", type=str,
                        default="/workspace/VAGEN/data/world_model/sokoban/features")
    parser.add_argument("--history_path", type=str,
                        default="/workspace/VAGEN/checkpoints/world_model/sokoban/training_history.json")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/VAGEN/checkpoints/world_model/sokoban/eval")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config_path = os.path.dirname(args.checkpoint)
    config_file = os.path.join(config_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            cfg = json.load(f)
        config = WorldModelConfig(**cfg)
    else:
        config = WorldModelConfig()

    # Load model
    model = LatentWorldModel(config).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # 1. Training curves
    if os.path.exists(args.history_path):
        plot_training_curves(args.history_path, os.path.join(args.output_dir, "training_curves.png"))

    # 2. NN retrieval
    nn_retrieval_eval(model, args.features_dir, args.output_dir, args.device, n_examples=50)

    # 3. Action discrimination
    action_discrimination_test(model, args.features_dir, args.device, n_states=200)


if __name__ == "__main__":
    main()
