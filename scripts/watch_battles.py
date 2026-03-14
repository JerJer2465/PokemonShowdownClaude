"""
Watch live battles between our bot and a heuristic opponent on the local PS server.

Open http://localhost:8000 in your browser → click a username → spectate.

Usage:
    # Bot vs MaxBasePower (default)
    python scripts/watch_battles.py --n_games 5

    # Bot vs SimpleHeuristics, save replays
    python scripts/watch_battles.py --opponent heuristic --n_games 10 --save_replays

    # Use a specific checkpoint
    python scripts/watch_battles.py --checkpoint checkpoints/latest.pt --n_games 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poke_env import LocalhostServerConfiguration
from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer, RandomPlayer

try:
    from poke_env import AccountConfiguration
except ImportError:
    from poke_env.player import AccountConfiguration

from pokebot.players.showdown_player import PokeTransformerPlayer


def _rand_suffix(n=4):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, default="checkpoints/latest.pt")
    p.add_argument("--n_games",      type=int, default=5)
    p.add_argument("--format",       type=str, default="gen4randombattle")
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--opponent",     type=str, default="maxpower",
                   choices=["maxpower", "heuristic", "random", "mirror"],
                   help="Opponent type: maxpower | heuristic | random | mirror")
    p.add_argument("--checkpoint2",  type=str, default=None,
                   help="Opponent checkpoint for mirror matches (default: same as --checkpoint)")
    p.add_argument("--save_replays", action="store_true",
                   help="Save HTML replays to ./replays/")
    p.add_argument("--deterministic", action="store_true",
                   help="Greedy action selection (default: sampling)")
    return p.parse_args()


async def main(args):
    server_cfg = LocalhostServerConfiguration

    suffix = _rand_suffix()
    # --- Build our bot ---
    bot = PokeTransformerPlayer.from_checkpoint(
        args.checkpoint,
        device=args.device,
        deterministic=args.deterministic,
        battle_format=args.format,
        server_configuration=server_cfg,
        save_replays=bool(args.save_replays),
        account_configuration=AccountConfiguration(f"PokeBot{suffix}", None),
    )

    # --- Build opponent ---
    opp_kwargs = dict(
        battle_format=args.format,
        server_configuration=server_cfg,
        account_configuration=AccountConfiguration(f"Heuristic{suffix}", None),
    )
    if args.opponent == "mirror":
        opp_ckpt = args.checkpoint2 or args.checkpoint
        opponent = PokeTransformerPlayer.from_checkpoint(
            opp_ckpt,
            device=args.device,
            deterministic=args.deterministic,
            battle_format=args.format,
            server_configuration=server_cfg,
            save_replays=bool(args.save_replays),
            account_configuration=AccountConfiguration(f"Mirror{suffix}", None),
        )
        opp_label = f"Mirror ({opp_ckpt})"
    elif args.opponent == "maxpower":
        opponent = MaxBasePowerPlayer(**opp_kwargs)
        opp_label = "MaxBasePowerPlayer"
    elif args.opponent == "heuristic":
        opponent = SimpleHeuristicsPlayer(**opp_kwargs)
        opp_label = "SimpleHeuristicsPlayer"
    else:
        opponent = RandomPlayer(**opp_kwargs)
        opp_label = "RandomPlayer"

    n_params = sum(p.numel() for p in bot.model.parameters())
    print(f"\n{'='*60}")
    print(f"  Bot:      {args.checkpoint}  ({n_params/1e6:.1f}M params)")
    print(f"  Opponent: {opp_label}")
    print(f"  Format:   {args.format}")
    print(f"  Games:    {args.n_games}")
    print(f"\n  Watching: http://localhost:8000")
    print(f"  → Open in browser, click a username to spectate a live battle")
    print(f"{'='*60}\n")

    await bot.battle_against(opponent, n_battles=args.n_games)

    total = bot.n_won_battles + bot.n_lost_battles + bot.n_tied_battles
    wr = bot.n_won_battles / max(total, 1) * 100
    print(f"\n{'='*60}")
    print(f"  Final: {bot.n_won_battles}W / {bot.n_lost_battles}L / {bot.n_tied_battles}T  "
          f"WR: {wr:.1f}%")
    if args.save_replays:
        print(f"  Replays saved to: ./replays/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
