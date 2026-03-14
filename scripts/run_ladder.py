"""
Phase 3: Online Ladder Testing on Pokemon Showdown.

Connects to the actual Pokemon Showdown server (or localhost) and plays
gen4randombattle games, reporting win rate and Glicko estimate.

Usage:
    # Play 20 ladder games against real humans (requires PS account)
    python scripts/run_ladder.py --username BotName --password YourPass --n_games 20

    # Evaluate against the built-in RandomPlayer (no account needed for localhost)
    python scripts/run_ladder.py --local --vs_random --n_games 50

    # Use best checkpoint
    python scripts/run_ladder.py --checkpoint checkpoints/latest.pt --n_games 20 --username BotName

    # Deterministic greedy play
    python scripts/run_ladder.py --deterministic --checkpoint checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Pokemon Showdown ladder evaluation")
    parser.add_argument("--checkpoint",   type=str, default="checkpoints/bc_init.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--username",     type=str, default=None,
                        help="PS username (required for real ladder)")
    parser.add_argument("--password",     type=str, default=None,
                        help="PS password (leave blank for registered guest)")
    parser.add_argument("--n_games",      type=int, default=20,
                        help="Number of games to play")
    parser.add_argument("--format",       type=str, default="gen4randombattle",
                        help="Battle format")
    parser.add_argument("--device",       type=str, default="cpu",
                        help="Inference device (cpu or cuda)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy (argmax) action selection")
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Softmax temperature for sampling")
    parser.add_argument("--local",        action="store_true",
                        help="Use local PS server (localhost:8000)")
    parser.add_argument("--vs_random",    action="store_true",
                        help="Play against built-in RandomPlayer instead of ladder")
    parser.add_argument("--concurrency",  type=int, default=1,
                        help="Number of concurrent battles (ladder only)")
    parser.add_argument("--verbose",      action="store_true",
                        help="Show detailed per-turn info")
    return parser.parse_args()


async def run_vs_random(player, n_games: int, battle_format: str, server_cfg):
    """Play N games against RandomPlayer and report results."""
    from poke_env.player import RandomPlayer

    rand_player = RandomPlayer(
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    print(f"Playing {n_games} games against RandomPlayer...")
    t0 = time.time()
    await player.battle_against(rand_player, n_battles=n_games)
    elapsed = time.time() - t0

    wins = player.n_won_battles
    total = player.n_finished_battles
    wr = wins / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"Results vs RandomPlayer: {wins}/{total} won ({wr:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(total,1):.1f}s/game)")
    return wins, total


async def run_ladder(player, n_games: int):
    """Play N ladder games on PS."""
    print(f"Playing {n_games} ladder games on {player.battle_format}...")
    t0 = time.time()
    await player.ladder(n_games)
    elapsed = time.time() - t0

    wins = player.n_won_battles
    total = player.n_finished_battles
    wr = wins / max(total, 1) * 100
    print(f"\n{'='*50}")
    print(f"Ladder results: {wins}/{total} won ({wr:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(total,1):.1f}s/game)")
    return wins, total


async def main(args):
    from pokebot.players.showdown_player import PokeTransformerPlayer

    # Pick server configuration
    if args.local:
        from poke_env import LocalhostServerConfiguration
        server_cfg = LocalhostServerConfiguration
    else:
        from poke_env import ShowdownServerConfiguration
        server_cfg = ShowdownServerConfiguration

    # Player configuration — poke-env 0.8+ uses AccountConfiguration
    player_cfg = None
    if args.username:
        try:
            from poke_env import AccountConfiguration
            player_cfg = AccountConfiguration(args.username, args.password)
        except ImportError:
            try:
                from poke_env import PlayerConfiguration
                player_cfg = PlayerConfiguration(args.username, args.password)
            except ImportError:
                pass

    # Build kwargs
    kwargs = {
        "battle_format": args.format,
        "device": args.device,
        "deterministic": args.deterministic,
        "temperature": args.temperature,
        "server_configuration": server_cfg,
    }
    if player_cfg is not None:
        # poke-env 0.8+ uses account_configuration; older uses player_configuration
        try:
            from poke_env import AccountConfiguration
            kwargs["account_configuration"] = player_cfg
        except ImportError:
            kwargs["player_configuration"] = player_cfg

    print(f"Loading model from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        for f in os.listdir("checkpoints"):
            print(f"  checkpoints/{f}")
        sys.exit(1)

    player = PokeTransformerPlayer.from_checkpoint(
        args.checkpoint,
        **kwargs,
    )

    n_params = sum(p.numel() for p in player.model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params | device={args.device}")
    print(f"Mode: {'deterministic' if args.deterministic else f'sampling (T={args.temperature})'}")
    print(f"Format: {args.format}")

    if args.vs_random:
        wins, total = await run_vs_random(player, args.n_games, args.format, server_cfg)
    else:
        wins, total = await run_ladder(player, args.n_games)

    print(f"\nFinal win rate: {wins/max(total,1)*100:.1f}% ({wins}/{total})")
    return wins / max(total, 1)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
