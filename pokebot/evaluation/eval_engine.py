"""
Evaluation utilities: win-rate tracking, Glicko-2, and self-play benchmark.

Usage:
    evaluator = Evaluator(model, device="cuda")
    stats = evaluator.run(n_games=200, opponent="random")
    print(stats)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from pokebot.env.poke_engine_env import (
    PokeEngineEnv,
    simple_heuristic_opponent,
    _random_opponent,
)
from pokebot.env.obs_builder import ObsBuilder
from config.model_config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Glicko-2 implementation
# ---------------------------------------------------------------------------

class Glicko2:
    """
    Glicko-2 rating system for tracking bot ELO over time.

    Default initial values mirror the Pokemon Showdown ladder:
      r=1500, RD=350, sigma=0.06
    """

    def __init__(self, r: float = 1500.0, rd: float = 350.0, sigma: float = 0.06):
        self.r = r          # Rating
        self.rd = rd        # Rating deviation
        self.sigma = sigma  # Volatility

    def update(self, opponent_r: float, opponent_rd: float, score: float):
        """
        Update Glicko-2 rating after a single game.
        score: 1.0 = win, 0.5 = draw, 0.0 = loss
        """
        TAU = 0.5  # system constant (constrains volatility)
        EPS = 1e-6

        # Step 1: Convert to Glicko-2 scale
        mu = (self.r - 1500) / 173.7178
        phi = self.rd / 173.7178
        mu_j = (opponent_r - 1500) / 173.7178
        phi_j = opponent_rd / 173.7178

        # Step 2: Compute g, E
        g_phi_j = 1.0 / math.sqrt(1 + 3 * phi_j ** 2 / math.pi ** 2)
        E_mu = 1.0 / (1.0 + math.exp(-g_phi_j * (mu - mu_j)))

        # Step 3: Compute v
        v = 1.0 / (g_phi_j ** 2 * E_mu * (1 - E_mu))

        # Step 4: Compute delta
        delta = v * g_phi_j * (score - E_mu)

        # Step 5: Update sigma via Illinois algorithm
        a = math.log(self.sigma ** 2)
        f = lambda x: (
            math.exp(x) * (delta ** 2 - phi ** 2 - v - math.exp(x))
            / (2 * (phi ** 2 + v + math.exp(x)) ** 2)
            - (x - a) / TAU ** 2
        )
        A = a
        if delta ** 2 > phi ** 2 + v:
            B = math.log(delta ** 2 - phi ** 2 - v)
        else:
            k = 1
            while f(a - k * TAU) < 0:
                k += 1
            B = a - k * TAU
        fA, fB = f(A), f(B)
        for _ in range(100):
            C = A + (A - B) * fA / (fB - fA + EPS)
            fC = f(C)
            if fB * fC < 0:
                A, fA = B, fB
            else:
                fA /= 2
            B, fB = C, fC
            if abs(B - A) < 1e-6:
                break
        sigma_prime = math.exp(A / 2)

        # Step 6: Update phi*, mu*, phi_prime
        phi_star = math.sqrt(phi ** 2 + sigma_prime ** 2)
        phi_prime = 1.0 / math.sqrt(1.0 / phi_star ** 2 + 1.0 / v)
        mu_prime = mu + phi_prime ** 2 * g_phi_j * (score - E_mu)

        # Step 7: Convert back
        self.r = 173.7178 * mu_prime + 1500
        self.rd = 173.7178 * phi_prime
        self.sigma = sigma_prime

    def __repr__(self):
        return f"Glicko2(r={self.r:.1f}, rd={self.rd:.1f}, sigma={self.sigma:.4f})"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalStats:
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_turns: int = 0
    total_reward: float = 0.0
    elapsed: float = 0.0
    opponent: str = "unknown"
    glicko: Optional[Glicko2] = None

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total, 1)

    @property
    def avg_turns(self) -> float:
        return self.total_turns / max(self.total, 1)

    def __str__(self):
        wr = self.win_rate * 100
        g = f" | Glicko: {self.glicko.r:.0f}±{self.glicko.rd:.0f}" if self.glicko else ""
        return (
            f"[vs {self.opponent}] "
            f"W/L/T: {self.wins}/{self.losses}/{self.ties} "
            f"({wr:.1f}% WR) "
            f"avg_turns={self.avg_turns:.1f} "
            f"total_reward={self.total_reward:.2f} "
            f"elapsed={self.elapsed:.1f}s"
            f"{g}"
        )


class Evaluator:
    """
    Runs benchmark evaluations of a model against built-in opponents
    using the poke-engine environment (no PS server needed).

    Opponents:
      - "random": random action selection
      - "heuristic": max base-power heuristic (simple_heuristic_opponent)
      - "self": play against current model weights (self-play eval)
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.obs_builder = ObsBuilder()

    def _make_policy(self, model_to_use=None):
        """Create a callable policy fn from a model."""
        m = (model_to_use or self.model).eval()
        obs_builder = ObsBuilder()
        device = self.device

        @torch.no_grad()
        def policy(obs_dict: dict) -> int:
            obs = obs_builder.encode(obs_dict)
            int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(device)
            float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(device)
            legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(device)
            log_probs, _, _ = m(int_ids, float_f, legal_m)
            return int(log_probs.argmax(dim=-1).item())

        return policy

    def run(
        self,
        n_games: int = 200,
        opponent: str = "heuristic",
        glicko: Optional[Glicko2] = None,
    ) -> EvalStats:
        """
        Evaluate the model against the specified opponent for n_games.

        Returns EvalStats.
        """
        if opponent == "random":
            opp_policy = _random_opponent
            opp_r, opp_rd = 1000.0, 350.0  # Rough random player Glicko
        elif opponent == "heuristic":
            opp_policy = simple_heuristic_opponent
            opp_r, opp_rd = 1200.0, 150.0  # Rough heuristic player Glicko
        elif opponent == "self":
            opp_policy = self._make_policy()
            opp_r, opp_rd = glicko.r if glicko else 1500.0, 200.0
        else:
            raise ValueError(f"Unknown opponent: {opponent}")

        model_policy = self._make_policy()
        env = PokeEngineEnv(opponent_policy=opp_policy)

        stats = EvalStats(opponent=opponent, glicko=glicko or Glicko2())
        t0 = time.time()

        for game_idx in range(n_games):
            obs, _ = env.reset()
            done = False
            game_reward = 0.0
            turns = 0

            while not done:
                # Agent uses model policy
                int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(self.device)
                float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(self.device)
                legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    log_probs, _, _ = self.model(int_ids, float_f, legal_m)
                action = int(log_probs.argmax(dim=-1).item())

                obs, reward, done, _, info = env.step(action)
                game_reward += reward
                turns += 1

            result = info.get("result")
            if result == "win":
                stats.wins += 1
                score = 1.0
            elif result == "loss":
                stats.losses += 1
                score = 0.0
            else:
                stats.ties += 1
                score = 0.5

            stats.total_turns += turns
            stats.total_reward += game_reward

            # Update Glicko
            stats.glicko.update(opp_r, opp_rd, score)

            if (game_idx + 1) % 50 == 0:
                wr = stats.wins / (game_idx + 1) * 100
                print(
                    f"  [{game_idx+1}/{n_games}] WR: {wr:.1f}%  "
                    f"Glicko: {stats.glicko.r:.0f}±{stats.glicko.rd:.0f}"
                )

        stats.elapsed = time.time() - t0
        return stats


def run_eval(
    checkpoint_path: str,
    n_games: int = 200,
    opponents: list = None,
    device: str = "cpu",
):
    """
    Convenience function: load checkpoint and run full evaluation suite.

    Returns dict of {opponent_name: EvalStats}
    """
    from pokebot.model.poke_transformer import PokeTransformer

    if opponents is None:
        opponents = ["random", "heuristic"]

    model = PokeTransformer()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    evaluator = Evaluator(model, device=device)
    results = {}
    for opp in opponents:
        print(f"\n--- Evaluating vs {opp} ({n_games} games) ---")
        stats = evaluator.run(n_games=n_games, opponent=opp)
        print(stats)
        results[opp] = stats

    return results
