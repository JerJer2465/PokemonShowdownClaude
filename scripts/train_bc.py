"""
Phase 1: Behavioral Cloning from a teacher policy.

Supports two teacher modes:
  --teacher mcts     : MCTS plays side_one (slow but strong with high ms)
  --teacher heuristic: smart_heuristic_opponent plays side_one (fast, ~86% vs random)

The teacher acts as side_one. side_two uses a random opponent so the model
learns to beat random. We imitate the teacher's action choices.

Usage:
    # Fast: heuristic teacher, 1M steps
    python scripts/train_bc.py --teacher heuristic --steps 1000000

    # MCTS teacher with 50ms budget
    python scripts/train_bc.py --teacher mcts --mcts_ms 50 --steps 500000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MODEL_CONFIG
from config.training_config import TRAINING_CONFIG
from pokebot.env.poke_engine_env import (
    PokeEngineEnv, simple_heuristic_opponent, smart_heuristic_opponent,
    mcts_side_one_action, _random_opponent,
)
from pokebot.env.obs_builder import ObsBuilder, FLOAT_DIM_PER_POKEMON
from pokebot.model.poke_transformer import PokeTransformer

import random


def _random_opp(obs_dict):
    legal = obs_dict.get("legal_actions", list(range(10)))
    return random.choice(legal)


def collect_bc_batch(
    envs: list[PokeEngineEnv],
    obs_list: list[dict],
    batch_size: int,
    teacher: str,
    mcts_ms: int,
):
    """
    Collect (obs, teacher_action) pairs.
    teacher='heuristic': smart_heuristic_opponent picks action for side_one
    teacher='mcts': MCTS picks action for side_one
    """
    n_envs = len(envs)
    n_tokens = MODEL_CONFIG["n_tokens"]
    F_dim = FLOAT_DIM_PER_POKEMON
    n_actions = MODEL_CONFIG["n_actions"]

    per_env = (batch_size + n_envs - 1) // n_envs
    total = per_env * n_envs

    int_ids_buf = np.zeros((total, n_tokens, 8),    dtype=np.int64)
    float_buf   = np.zeros((total, n_tokens, F_dim), dtype=np.float32)
    legal_buf   = np.zeros((total, n_actions),       dtype=np.float32)
    action_buf  = np.zeros(total,                    dtype=np.int64)

    idx = 0
    for _ in range(per_env):
        for ei, (env, obs) in enumerate(zip(envs, obs_list)):
            if teacher == "mcts":
                action = mcts_side_one_action(env._state, mcts_ms)
            else:
                # Heuristic teacher: build raw obs_dict for side_one
                raw_obs = env._build_obs_dict(perspective="side_one")
                action = smart_heuristic_opponent(raw_obs)

            # Record (encoded_obs, teacher_action)
            int_ids_buf[idx] = obs["int_ids"]
            float_buf[idx]   = obs["float_feats"]
            legal_buf[idx]   = obs["legal_mask"]
            action_buf[idx]  = action
            idx += 1

            next_obs, _, done, _, _ = env.step(action)
            if done:
                next_obs, _ = env.reset()
            obs_list[ei] = next_obs

    return int_ids_buf[:idx], float_buf[:idx], legal_buf[:idx], action_buf[:idx]


def evaluate_vs_random(model, device, n_games=100, deterministic=True) -> float:
    """Quick win-rate eval: model as side_one vs random side_two."""
    model.eval()
    env = PokeEngineEnv(opponent_policy=_random_opp)
    wins = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
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
            obs, reward, done, _, _ = env.step(action)
        if reward > 0:
            wins += 1
    return wins / n_games


def main():
    parser = argparse.ArgumentParser(description="BC from teacher demonstrations")
    parser.add_argument("--teacher",   type=str,   default="heuristic",
                        choices=["heuristic", "mcts"],
                        help="Teacher policy: 'heuristic' (fast) or 'mcts' (slow)")
    parser.add_argument("--steps",     type=int,   default=1_000_000)
    parser.add_argument("--envs",      type=int,   default=8)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--batch",     type=int,   default=1024)
    parser.add_argument("--log_every", type=int,   default=25)
    parser.add_argument("--eval_every", type=int,  default=100,
                        help="Evaluate WR vs random every N updates")
    parser.add_argument("--eval_games", type=int,  default=100)
    parser.add_argument("--mcts_ms",   type=int,   default=50,
                        help="MCTS time budget per teacher move (ms)")
    parser.add_argument("--out",       type=str,   default="checkpoints/bc_smart.pt")
    parser.add_argument("--device",    type=str,   default=TRAINING_CONFIG["device"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"BC Training  |  teacher={args.teacher}  device={device}  "
          f"steps={args.steps:,}  envs={args.envs}  batch={args.batch}")
    if args.teacher == "mcts":
        est_min = args.steps * args.mcts_ms / 1000 / 60
        print(f"MCTS time budget: {args.mcts_ms}ms  (~{est_min:.0f} min)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Model
    model = PokeTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps // args.batch, eta_min=1e-5
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    # Environments — side_two = random so model learns to beat random
    envs: list[PokeEngineEnv] = []
    obs_list: list[dict] = []
    for _ in range(args.envs):
        env = PokeEngineEnv(opponent_policy=_random_opp)
        obs, _ = env.reset()
        envs.append(env)
        obs_list.append(obs)

    steps_collected = 0
    update = 0
    running_loss = 0.0
    best_wr = 0.0

    print("\nTraining...")
    while steps_collected < args.steps:
        int_ids_np, float_np, legal_np, actions_np = collect_bc_batch(
            envs, obs_list, args.batch, args.teacher, args.mcts_ms,
        )

        int_ids = torch.from_numpy(int_ids_np).to(device)
        float_f = torch.from_numpy(float_np).to(device)
        legal_m = torch.from_numpy(legal_np).to(device)
        targets = torch.from_numpy(actions_np).to(device)

        model.train()
        log_probs, _, _ = model(int_ids, float_f, legal_m)
        loss = F.nll_loss(log_probs, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        steps_collected += len(actions_np)
        update += 1
        running_loss += loss.item()

        if update % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            running_loss = 0.0
            pct = steps_collected / args.steps * 100
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"update={update:5d}  steps={steps_collected:8,}/{args.steps:,} ({pct:.0f}%)  "
                f"loss={avg_loss:.4f}  lr={lr:.1e}"
            )

        if update % args.eval_every == 0:
            wr = evaluate_vs_random(model, device, n_games=args.eval_games)
            print(f"  >>> Eval vs Random: {wr*100:.1f}% WR ({args.eval_games} games)")
            if wr > best_wr:
                best_wr = wr
                torch.save({"model_state": model.state_dict(), "update": update,
                            "win_rate": wr}, args.out)
                print(f"  >>> New best! Saved → {args.out}")

    # Final eval
    wr = evaluate_vs_random(model, device, n_games=200)
    print(f"\nFinal eval: {wr*100:.1f}% WR vs random (200 games)")
    if wr > best_wr:
        best_wr = wr
        torch.save({"model_state": model.state_dict(), "update": update,
                    "win_rate": wr}, args.out)
        print(f"New best! Saved → {args.out}")
    else:
        # Save final model separately, don't overwrite best
        final_path = args.out.replace(".pt", "_final.pt")
        torch.save({"model_state": model.state_dict(), "update": update,
                    "win_rate": wr}, final_path)
        print(f"Best WR was {best_wr*100:.1f}% (kept). Final model → {final_path}")


if __name__ == "__main__":
    main()
