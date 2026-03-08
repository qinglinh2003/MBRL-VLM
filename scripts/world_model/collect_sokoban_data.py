"""
Collect Sokoban transition data for world model training.

For each visited state, tries all 9 actions to get complete transition coverage,
including failures (hitting walls, invalid pushes).

Output: a directory of .pkl files, each containing:
    {
        "obs_image": PIL.Image (96x96),
        "action": int (0-8),
        "action_name": str,
        "next_obs_image": PIL.Image (96x96),
        "reward": float,
        "done": bool,
    }
"""

import argparse
import os
import pickle
import copy
import random
from collections import defaultdict

import numpy as np
from PIL import Image


def make_env(dim_room=(6, 6), num_boxes=1):
    from gym_sokoban.envs import SokobanEnv
    env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes)
    return env


ACTION_LOOKUP = {
    0: "no operation",
    1: "push up",
    2: "push down",
    3: "push left",
    4: "push right",
    5: "move up",
    6: "move down",
    7: "move left",
    8: "move right",
}


def render_pil(env):
    """Render environment state as a PIL Image."""
    rgb = env.render("rgb_array")
    return Image.fromarray(rgb)


def collect_episodes(n_episodes, dim_room=(6, 6), num_boxes=1, max_steps=20, seed=0):
    """Collect transitions by running random episodes and trying all actions per state."""
    random.seed(seed)
    np.random.seed(seed)

    transitions = []
    total_states = 0

    for ep in range(n_episodes):
        env = make_env(dim_room=dim_room, num_boxes=num_boxes)
        env.reset()

        for step in range(max_steps):
            obs_img = render_pil(env)
            saved_state = copy.deepcopy(env)

            # Try all 9 actions from this state
            for action in range(9):
                env_copy = copy.deepcopy(saved_state)
                _, reward, done, info = env_copy.step(action)
                next_obs_img = render_pil(env_copy)

                transitions.append({
                    "obs_image": obs_img,
                    "action": action,
                    "action_name": ACTION_LOOKUP[action],
                    "next_obs_image": next_obs_img,
                    "reward": float(reward),
                    "done": bool(done),
                    "episode": ep,
                    "step": step,
                })

            total_states += 1

            # Take a random action to advance the episode
            random_action = random.randint(0, 8)
            _, _, done, _ = env.step(random_action)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(f"  Episodes: {ep+1}/{n_episodes}, states: {total_states}, transitions: {len(transitions)}")

    return transitions


def save_transitions(transitions, output_dir):
    """Save transitions to disk."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transitions.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(transitions, f)
    print(f"Saved {len(transitions)} transitions to {output_path}")

    # Save a small summary
    n_done = sum(1 for t in transitions if t["done"])
    n_rewarded = sum(1 for t in transitions if t["reward"] != 0)
    action_counts = defaultdict(int)
    for t in transitions:
        action_counts[t["action_name"]] += 1

    print(f"  Done transitions: {n_done}")
    print(f"  Non-zero reward: {n_rewarded}")
    print(f"  Action distribution:")
    for name, count in sorted(action_counts.items()):
        print(f"    {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Collect Sokoban transitions")
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument("--dim_room", type=int, nargs=2, default=[6, 6])
    parser.add_argument("--num_boxes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/VAGEN/data/world_model/sokoban")
    args = parser.parse_args()

    print(f"Collecting {args.n_episodes} episodes, dim_room={args.dim_room}, "
          f"num_boxes={args.num_boxes}, max_steps={args.max_steps}")

    transitions = collect_episodes(
        n_episodes=args.n_episodes,
        dim_room=tuple(args.dim_room),
        num_boxes=args.num_boxes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    save_transitions(transitions, args.output_dir)


if __name__ == "__main__":
    main()
