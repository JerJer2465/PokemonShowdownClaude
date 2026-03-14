"""
Collect a large offline dataset of (obs, teacher_action, category) for BC training.

Runs smart_heuristic_opponent as teacher on CPU, saves encoded observations
and action categories to .npz for fast GPU-only training.

Usage:
    python scripts/collect_bc_dataset.py --steps 2000000 --envs 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random

from config.model_config import MODEL_CONFIG
from pokebot.env.poke_engine_env import (
    PokeEngineEnv, smart_heuristic_opponent, _random_opponent,
)
from pokebot.env.obs_builder import FLOAT_DIM_PER_POKEMON

# ── Action categories ────────────────────────────────────────────────────────
# 0 = damaging move, 1 = switch, 2 = hazard, 3 = setup, 4 = status/other
HAZARD_IDS = {"stealthrock", "spikes", "toxicspikes"}
SETUP_IDS = {
    "swordsdance", "calmmind", "dragondance", "nastyplot", "agility",
    "rockpolish", "bulkup", "curse", "quiverdance", "coil", "shellsmash",
    "bellydrum", "growth", "workup", "howl", "tailglow",
}
STATUS_IDS = {
    "thunderwave", "toxic", "willowisp", "spore", "sleeppowder",
    "stunspore", "hypnosis", "yawn", "glare", "lovelykiss",
    "grasswhistle", "sing", "darkvoid", "poisonpowder",
    "reflect", "lightscreen", "taunt", "substitute", "protect",
    "recover", "softboiled", "roost", "slackoff", "moonlight",
    "morningsun", "synthesis", "wish", "milkdrink",
}


def classify_action(action: int, obs_dict: dict) -> int:
    """Classify teacher action into category."""
    if action >= 4:
        return 1  # switch

    active = obs_dict.get("side_one", {}).get("active", {})
    moves = active.get("moves", [])
    if action >= len(moves):
        return 0  # fallback

    move_id = moves[action].get("id", "").lower()
    bp = moves[action].get("basePower", 0)

    if move_id in HAZARD_IDS:
        return 2  # hazard
    if move_id in SETUP_IDS:
        return 3  # setup
    if bp == 0 or move_id in STATUS_IDS:
        return 4  # status/other
    return 0  # damaging move


def _random_opp(obs_dict):
    legal = obs_dict.get("legal_actions", list(range(10)))
    return random.choice(legal)


def main():
    parser = argparse.ArgumentParser(description="Collect BC dataset from teacher")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Total (obs, action) pairs to collect")
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--out", type=str, default="data/bc_dataset_2M.npz")
    parser.add_argument("--log_every", type=int, default=50000)
    args = parser.parse_args()

    n_tokens = MODEL_CONFIG["n_tokens"]
    F_dim = FLOAT_DIM_PER_POKEMON
    n_actions = MODEL_CONFIG["n_actions"]

    # Pre-allocate buffers
    N = args.steps
    int_ids_buf = np.zeros((N, n_tokens, 8), dtype=np.int64)
    float_buf = np.zeros((N, n_tokens, F_dim), dtype=np.float32)
    legal_buf = np.zeros((N, n_actions), dtype=np.float32)
    action_buf = np.zeros(N, dtype=np.int64)
    category_buf = np.zeros(N, dtype=np.int8)

    # Create environments
    envs = []
    obs_list = []
    for _ in range(args.envs):
        env = PokeEngineEnv(opponent_policy=_random_opp)
        obs, _ = env.reset()
        envs.append(env)
        obs_list.append(obs)

    print(f"Collecting {N:,} teacher demonstrations with {args.envs} envs...")
    t0 = time.time()
    idx = 0
    cat_counts = [0, 0, 0, 0, 0]

    while idx < N:
        for ei, (env, obs) in enumerate(zip(envs, obs_list)):
            if idx >= N:
                break

            # Teacher picks action
            raw_obs = env._build_obs_dict(perspective="side_one")
            action = smart_heuristic_opponent(raw_obs)
            category = classify_action(action, raw_obs)

            # Store
            int_ids_buf[idx] = obs["int_ids"]
            float_buf[idx] = obs["float_feats"]
            legal_buf[idx] = obs["legal_mask"]
            action_buf[idx] = action
            category_buf[idx] = category
            cat_counts[category] += 1
            idx += 1

            # Step environment
            next_obs, _, done, _, _ = env.step(action)
            if done:
                next_obs, _ = env.reset()
            obs_list[ei] = next_obs

        if idx % args.log_every < args.envs:
            elapsed = time.time() - t0
            rate = idx / elapsed
            eta = (N - idx) / rate
            print(f"  {idx:>10,}/{N:,} ({idx/N*100:.0f}%)  "
                  f"{rate:.0f} steps/s  ETA {eta:.0f}s  "
                  f"cats=[move:{cat_counts[0]}, sw:{cat_counts[1]}, "
                  f"haz:{cat_counts[2]}, setup:{cat_counts[3]}, status:{cat_counts[4]}]")

    elapsed = time.time() - t0
    print(f"\nCollection done in {elapsed:.1f}s ({idx/elapsed:.0f} steps/s)")
    print(f"Category distribution:")
    cat_names = ["damaging", "switch", "hazard", "setup", "status"]
    for i, (name, count) in enumerate(zip(cat_names, cat_counts)):
        print(f"  {name:10s}: {count:>8,} ({count/idx*100:.1f}%)")

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(
        args.out,
        int_ids=int_ids_buf[:idx],
        float_feats=float_buf[:idx],
        legal_mask=legal_buf[:idx],
        actions=action_buf[:idx],
        categories=category_buf[:idx],
    )
    fsize = os.path.getsize(args.out) / 1e9
    print(f"Saved → {args.out} ({fsize:.2f} GB)")


if __name__ == "__main__":
    main()
