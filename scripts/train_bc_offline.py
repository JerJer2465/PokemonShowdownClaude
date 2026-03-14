"""
Fast GPU-only BC training from pre-collected dataset.

Loads .npz dataset, applies class-weighted NLL loss, trains with per-category
diagnostic logging. No CPU env bottleneck — pure GPU training.

Usage:
    python scripts/train_bc_offline.py --dataset data/bc_dataset_2M.npz --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from config.model_config import MODEL_CONFIG
from config.training_config import TRAINING_CONFIG
from pokebot.model.poke_transformer import PokeTransformer
from pokebot.env.poke_engine_env import PokeEngineEnv

# Category names for logging
CAT_NAMES = ["damaging", "switch", "hazard", "setup", "status"]

# Default per-category loss weights
DEFAULT_WEIGHTS = {
    0: 1.0,    # damaging move
    1: 8.0,    # switch
    2: 10.0,   # hazard
    3: 8.0,    # setup
    4: 4.0,    # status/other
}


def _random_opp(obs_dict):
    legal = obs_dict.get("legal_actions", list(range(10)))
    return random.choice(legal)


def evaluate_vs_random(model, device, n_games=100) -> float:
    """Quick win-rate eval: model vs random."""
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
            action = int(log_probs.argmax(dim=-1).item())
            obs, reward, done, _, _ = env.step(action)
        if reward > 0:
            wins += 1
    return wins / n_games


def compute_category_accuracy(log_probs, targets, categories):
    """Compute per-category top-1 accuracy."""
    preds = log_probs.argmax(dim=-1)
    correct = (preds == targets)

    results = {}
    for cat_id, cat_name in enumerate(CAT_NAMES):
        mask = (categories == cat_id)
        n = mask.sum().item()
        if n > 0:
            acc = correct[mask].float().mean().item()
            results[cat_name] = (acc, n)
    return results


def main():
    parser = argparse.ArgumentParser(description="Offline BC training with weighted loss")
    parser.add_argument("--dataset", type=str, default="data/bc_dataset_2M.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=3e-5)
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N updates")
    parser.add_argument("--eval_every", type=int, default=2,
                        help="Evaluate WR every N epochs")
    parser.add_argument("--eval_games", type=int, default=200)
    parser.add_argument("--out", type=str, default="checkpoints/bc_smart_v5.pt")
    parser.add_argument("--device", type=str, default=TRAINING_CONFIG["device"])
    # Category weights
    parser.add_argument("--w_move", type=float, default=DEFAULT_WEIGHTS[0])
    parser.add_argument("--w_switch", type=float, default=DEFAULT_WEIGHTS[1])
    parser.add_argument("--w_hazard", type=float, default=DEFAULT_WEIGHTS[2])
    parser.add_argument("--w_setup", type=float, default=DEFAULT_WEIGHTS[3])
    parser.add_argument("--w_status", type=float, default=DEFAULT_WEIGHTS[4])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load dataset ──
    print(f"Loading dataset: {args.dataset}")
    data = np.load(args.dataset)
    int_ids_all = data["int_ids"]       # (N, 15, 8)
    float_all = data["float_feats"]     # (N, 15, F)
    legal_all = data["legal_mask"]      # (N, 10)
    actions_all = data["actions"]       # (N,)
    categories_all = data["categories"] # (N,)
    N = len(actions_all)

    # Category stats
    cat_weights_map = {
        0: args.w_move, 1: args.w_switch, 2: args.w_hazard,
        3: args.w_setup, 4: args.w_status,
    }
    print(f"Dataset: {N:,} samples")
    for cat_id, cat_name in enumerate(CAT_NAMES):
        count = (categories_all == cat_id).sum()
        print(f"  {cat_name:10s}: {count:>8,} ({count/N*100:.1f}%)  weight={cat_weights_map[cat_id]:.1f}")

    # Pre-compute per-sample weights
    sample_weights = np.array([cat_weights_map[c] for c in categories_all], dtype=np.float32)

    # ── Model ──
    model = PokeTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params  device={device}")

    updates_per_epoch = (N + args.batch - 1) // args.batch
    total_updates = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=args.lr_min,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    best_wr = 0.0
    global_step = 0

    print(f"\nTraining: {args.epochs} epochs × {updates_per_epoch} updates = {total_updates:,} total updates")
    print(f"Batch size: {args.batch}  LR: {args.lr} → {args.lr_min}\n")

    for epoch in range(args.epochs):
        t0 = time.time()
        perm = np.random.permutation(N)

        model.train()
        running_loss = 0.0
        running_cat_acc = {name: [0.0, 0] for name in CAT_NAMES}
        updates_this_epoch = 0

        for batch_start in range(0, N, args.batch):
            idx = perm[batch_start:batch_start + args.batch]
            if len(idx) < 32:
                continue  # skip tiny last batch

            int_ids = torch.from_numpy(int_ids_all[idx]).to(device)
            float_f = torch.from_numpy(float_all[idx]).to(device)
            legal_m = torch.from_numpy(legal_all[idx]).to(device)
            targets = torch.from_numpy(actions_all[idx].astype(np.int64)).to(device)
            cats = torch.from_numpy(categories_all[idx].astype(np.int64)).to(device)
            weights = torch.from_numpy(sample_weights[idx]).to(device)

            log_probs, _, _ = model(int_ids, float_f, legal_m)

            # Weighted NLL loss
            nll = F.nll_loss(log_probs, targets, reduction='none')
            loss = (nll * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            updates_this_epoch += 1
            global_step += 1

            # Per-category accuracy tracking
            with torch.no_grad():
                cat_acc = compute_category_accuracy(log_probs, targets, cats)
                for name, (acc, n) in cat_acc.items():
                    running_cat_acc[name][0] += acc * n
                    running_cat_acc[name][1] += n

            if updates_this_epoch % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                lr = optimizer.param_groups[0]["lr"]
                cat_str = "  ".join(
                    f"{name}={running_cat_acc[name][0]/max(running_cat_acc[name][1],1)*100:.0f}%"
                    for name in CAT_NAMES if running_cat_acc[name][1] > 0
                )
                print(f"  ep={epoch+1:2d} upd={global_step:5d}  loss={avg_loss:.4f}  lr={lr:.1e}  {cat_str}")
                running_cat_acc = {name: [0.0, 0] for name in CAT_NAMES}

        epoch_time = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs} done in {epoch_time:.1f}s")

        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            wr = evaluate_vs_random(model, device, n_games=args.eval_games)
            print(f"  >>> Eval vs Random: {wr*100:.1f}% WR ({args.eval_games} games)")
            if wr > best_wr:
                best_wr = wr
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "win_rate": wr,
                }, args.out)
                print(f"  >>> New best! Saved → {args.out}")

    # Final eval
    wr = evaluate_vs_random(model, device, n_games=300)
    print(f"\nFinal eval: {wr*100:.1f}% WR vs random (300 games)")
    if wr > best_wr:
        torch.save({
            "model_state": model.state_dict(),
            "epoch": args.epochs,
            "global_step": global_step,
            "win_rate": wr,
        }, args.out)
        print(f"New best! Saved → {args.out}")
    else:
        final_path = args.out.replace(".pt", "_final.pt")
        torch.save({
            "model_state": model.state_dict(),
            "epoch": args.epochs,
            "global_step": global_step,
            "win_rate": wr,
        }, final_path)
        print(f"Best was {best_wr*100:.1f}% (kept). Final → {final_path}")


if __name__ == "__main__":
    main()
