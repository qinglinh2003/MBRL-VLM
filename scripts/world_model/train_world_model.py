"""
Train the Latent World Model (Projector + RewardHead) on pre-extracted features.

No VLM needed at training time -- operates purely on saved hidden states and
ViT target latents.
"""

import argparse
import os
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vagen.world_model.latent_world_model import LatentWorldModel, WorldModelConfig


class WorldModelDataset(Dataset):
    """Dataset of (hidden_state, target_latent, reward) tuples."""

    def __init__(self, features_dir):
        self.hidden_states = torch.load(
            os.path.join(features_dir, "hidden_states.pt"), map_location="cpu"
        )
        self.vision_targets = torch.load(
            os.path.join(features_dir, "vision_targets.pt"), map_location="cpu"
        )
        self.rewards = torch.load(
            os.path.join(features_dir, "rewards.pt"), map_location="cpu"
        )
        assert len(self.hidden_states) == len(self.vision_targets) == len(self.rewards)

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            "hidden_state": self.hidden_states[idx],
            "target_latent": self.vision_targets[idx],
            "reward": self.rewards[idx],
        }


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_cos_sim = 0
    total_infonce_acc = 0
    n_batches = 0

    for batch in loader:
        h = batch["hidden_state"].to(device)
        z_target = batch["target_latent"].to(device)
        r_target = batch["reward"].to(device)

        _, _, (loss, metrics) = model(h, z_target, r_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += metrics["wm/total_loss"]
        total_cos_sim += metrics["wm/cosine_sim_mean"]
        total_infonce_acc += metrics.get("wm/infonce_acc", 0)
        n_batches += 1

    return {
        "train/loss": total_loss / n_batches,
        "train/cosine_sim": total_cos_sim / n_batches,
        "train/infonce_acc": total_infonce_acc / n_batches,
    }


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_cos_sim = 0
    total_reward_loss = 0
    total_infonce_acc = 0
    n_batches = 0

    for batch in loader:
        h = batch["hidden_state"].to(device)
        z_target = batch["target_latent"].to(device)
        r_target = batch["reward"].to(device)

        _, _, (loss, metrics) = model(h, z_target, r_target)

        total_loss += metrics["wm/total_loss"]
        total_cos_sim += metrics["wm/cosine_sim_mean"]
        total_reward_loss += metrics.get("wm/reward_loss", 0)
        total_infonce_acc += metrics.get("wm/infonce_acc", 0)
        n_batches += 1

    return {
        "val/loss": total_loss / n_batches,
        "val/cosine_sim": total_cos_sim / n_batches,
        "val/reward_loss": total_reward_loss / n_batches,
        "val/infonce_acc": total_infonce_acc / n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train world model Projector")
    parser.add_argument("--features_dir", type=str,
                        default="/workspace/VAGEN/data/world_model/sokoban/features")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/VAGEN/checkpoints/world_model/sokoban")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--n_visual_tokens", type=int, default=9)
    parser.add_argument("--n_projector_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--reward_loss_weight", type=float, default=0.1)
    parser.add_argument("--infonce_loss_weight", type=float, default=1.0)
    parser.add_argument("--infonce_temperature", type=float, default=0.07)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--wandb_project", type=str, default="world-model-sokoban",
                        help="wandb project name (set to empty string to disable)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name (auto-generated if not set)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load dataset
    print(f"Loading features from {args.features_dir} ...")
    dataset = WorldModelDataset(args.features_dir)
    print(f"Dataset size: {len(dataset)}")

    # Train/val split
    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Create model
    config = WorldModelConfig(
        hidden_dim=args.hidden_dim,
        n_visual_tokens=args.n_visual_tokens,
        n_projector_layers=args.n_projector_layers,
        n_heads=args.n_heads,
        reward_loss_weight=args.reward_loss_weight,
        infonce_loss_weight=args.infonce_loss_weight,
        infonce_temperature=args.infonce_temperature,
    )
    model = LatentWorldModel(config).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Initialize wandb
    use_wandb = HAS_WANDB and args.wandb_project
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **vars(config),
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "n_params": n_params,
            },
        )
        print("wandb logging enabled")
    elif not HAS_WANDB and args.wandb_project:
        print("wandb not installed, logging disabled")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_val_cos = -1.0

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device)
        val_metrics = eval_epoch(model, val_loader, args.device)
        scheduler.step()

        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        history.append(metrics)

        if use_wandb:
            wandb.log(metrics, step=epoch)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss={metrics['train/loss']:.4f} "
                f"train_cos={metrics['train/cosine_sim']:.4f} "
                f"train_nce_acc={metrics['train/infonce_acc']:.2%} | "
                f"val_loss={metrics['val/loss']:.4f} "
                f"val_cos={metrics['val/cosine_sim']:.4f} "
                f"val_nce_acc={metrics['val/infonce_acc']:.2%}"
            )

        # Save best model
        if val_metrics["val/cosine_sim"] > best_val_cos:
            best_val_cos = val_metrics["val/cosine_sim"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

        # Periodic save
        if epoch % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch}.pt"))

    # Final save
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))

    # Save history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    print(f"\nDone! Best val cosine_sim: {best_val_cos:.4f}")
    print(f"Checkpoints saved to {args.output_dir}")

    if use_wandb:
        wandb.summary["best_val_cosine_sim"] = best_val_cos
        wandb.finish()


if __name__ == "__main__":
    main()
