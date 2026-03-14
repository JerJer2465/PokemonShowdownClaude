"""
Quick evaluation of a checkpoint against built-in opponents (no PS server needed).

Usage:
    python scripts/eval_checkpoint.py [--checkpoint checkpoints/bc_init.pt] [--n_games 200]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokebot.evaluation.eval_engine import run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_init.pt")
    parser.add_argument("--n_games", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opponents", nargs="+", default=["random", "heuristic"])
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Evaluating: {args.checkpoint}")
    results = run_eval(
        args.checkpoint,
        n_games=args.n_games,
        opponents=args.opponents,
        device=args.device,
    )

    print("\n===== SUMMARY =====")
    for opp, stats in results.items():
        print(f"  {stats}")


if __name__ == "__main__":
    main()
