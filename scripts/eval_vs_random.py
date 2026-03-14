"""
Quick win-rate evaluation using the local poke-engine environment.

Plays N games: our model (side_one) vs random opponent (side_two).
No network connection required.

Usage:
    python scripts/eval_vs_random.py --checkpoint checkpoints/bc_init.pt --n_games 200
    python scripts/eval_vs_random.py --checkpoint checkpoints/latest.pt --n_games 200
"""

from __future__ import annotations

import argparse
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from pokebot.env.poke_engine_env import PokeEngineEnv
from pokebot.model.poke_transformer import PokeTransformer


def random_opponent(obs_dict: dict) -> int:
    """Pick a uniformly random legal action."""
    legal = obs_dict.get("legal_actions", list(range(10)))
    return random.choice(legal)


def eval_checkpoint(checkpoint_path: str, n_games: int, device_str: str,
                    deterministic: bool) -> dict:
    device = torch.device(device_str)
    model = PokeTransformer().to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    env = PokeEngineEnv(opponent_policy=random_opponent)

    wins = losses = ties = 0
    total_turns = 0

    for game in range(n_games):
        obs, _ = env.reset()
        done = False
        turns = 0
        while not done:
            int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(device)
            float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(device)
            legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(device)
            with torch.no_grad():
                log_probs, _, _ = model(int_ids, float_f, legal_m)
            if deterministic:
                action = int(log_probs.argmax(dim=-1).item())
            else:
                action = int(torch.distributions.Categorical(logits=log_probs).sample().item())
            obs, reward, done, _, info = env.step(action)
            turns += 1

        total_turns += turns
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            ties += 1

        if (game + 1) % 20 == 0:
            wr = wins / (game + 1) * 100
            print(f"  game {game+1:4d}/{n_games}  W/L/T: {wins}/{losses}/{ties}  WR: {wr:.1f}%")

    total = wins + losses + ties
    return {
        "wins": wins, "losses": losses, "ties": ties, "total": total,
        "win_rate": wins / max(total, 1),
        "avg_turns": total_turns / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_init.pt")
    parser.add_argument("--n_games",    type=int, default=200)
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: {args.checkpoint} not found")
        sys.exit(1)

    print(f"Evaluating: {args.checkpoint}")
    print(f"Games: {args.n_games}  |  Device: {args.device}  |  "
          f"Mode: {'deterministic' if args.deterministic else 'sampling'}")
    print(f"Opponent: RandomPlayer (uniform random legal action)")
    print()

    results = eval_checkpoint(args.checkpoint, args.n_games, args.device, args.deterministic)

    print(f"\n{'='*50}")
    print(f"Results: {results['wins']}/{results['losses']}/{results['ties']} "
          f"(W/L/T)  WR: {results['win_rate']*100:.1f}%  "
          f"avg_turns: {results['avg_turns']:.1f}")


if __name__ == "__main__":
    main()
