"""
DUCT (Decoupled UCT) MCTS for simultaneous-action Pokemon battles.

Architecture:
  - Simultaneous moves: both players choose independently each turn
  - DUCT separates the UCB computation per player (no joint action table)
  - Hidden info determinization: sample a plausible opponent team from priors,
    then run tree search; average over K determinizations
  - Leaf evaluation: PokeTransformer value head
  - Prior policy: PokeTransformer policy head (for UCB exploration bonus)

Reference: "Monte-Carlo Tree Search for the Game of Pokemon" (Jett Wang 2024 MIT)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import numpy as np
import torch

from pokebot.env.poke_engine_env import (
    PokeEngineEnv,
    _build_legal_mask_from_state,
    _action_to_move_str,
    _sample_and_apply,
    _is_terminal,
    _state_to_obs_dict,
)
from pokebot.env.obs_builder import ObsBuilder
from config.model_config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# DUCT Node
# ---------------------------------------------------------------------------

class DUCTNode:
    """
    A node in the DUCT MCTS tree.

    Each node stores:
    - visit counts N_a[player][action] per player per action
    - total value Q_a[player][action] per player per action
    - children indexed by (s1_action, s2_action)
    """

    __slots__ = [
        "state", "s1_meta", "s2_meta", "turn",
        "n_legal_s1", "n_legal_s2",
        "N1", "N2",       # (n_actions,) visit counts for each player
        "Q1", "Q2",       # (n_actions,) total value for each player
        "children",       # dict: (a1, a2) → DUCTNode
        "prior1", "prior2",  # (n_actions,) policy priors
        "visit_count",    # total visits
        "is_terminal",
        "terminal_value", # scalar result (+1 win for s1, -1 loss)
        "_legal1", "_legal2",  # list of legal action indices
    ]

    def __init__(self, state, s1_meta, s2_meta, turn):
        self.state = state
        self.s1_meta = s1_meta
        self.s2_meta = s2_meta
        self.turn = turn
        self.children: dict = {}

        legal1 = _build_legal_mask_from_state(state.side_one)
        legal2 = _build_legal_mask_from_state(state.side_two)
        self._legal1 = legal1
        self._legal2 = legal2
        n1 = MODEL_CONFIG["n_actions"]
        n2 = MODEL_CONFIG["n_actions"]

        self.N1 = np.zeros(n1, dtype=np.float32)
        self.N2 = np.zeros(n2, dtype=np.float32)
        self.Q1 = np.zeros(n1, dtype=np.float32)
        self.Q2 = np.zeros(n2, dtype=np.float32)
        self.prior1 = np.ones(n1, dtype=np.float32) / max(len(legal1), 1)
        self.prior2 = np.ones(n2, dtype=np.float32) / max(len(legal2), 1)
        self.visit_count = 0

        result = _is_terminal(state)
        self.is_terminal = result is not None
        self.terminal_value = {
            "win": 1.0, "loss": -1.0, "tie": 0.0, None: 0.0
        }[result]

    def n1_total(self) -> float:
        return float(self.N1[self._legal1].sum())

    def n2_total(self) -> float:
        return float(self.N2[self._legal2].sum())

    def ucb_action_s1(self, c_puct: float) -> int:
        """DUCT UCB for side 1: argmax over legal actions."""
        n_total = self.n1_total() + 1.0
        best_a, best_score = self._legal1[0], -1e9
        for a in self._legal1:
            q = self.Q1[a] / max(self.N1[a], 1)
            u = c_puct * self.prior1[a] * math.sqrt(n_total) / (1 + self.N1[a])
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a

    def ucb_action_s2(self, c_puct: float) -> int:
        """DUCT UCB for side 2 (adversarial: negate Q)."""
        n_total = self.n2_total() + 1.0
        best_a, best_score = self._legal2[0], -1e9
        for a in self._legal2:
            # Side 2 maximizes its own value = minimizes side 1 value
            q = -self.Q2[a] / max(self.N2[a], 1)
            u = c_puct * self.prior2[a] * math.sqrt(n_total) / (1 + self.N2[a])
            score = q + u
            if score > best_score:
                best_score = score
                best_a = a
        return best_a


# ---------------------------------------------------------------------------
# DUCT MCTS
# ---------------------------------------------------------------------------

class DUCTMCTS:
    """
    Decoupled UCT Monte Carlo Tree Search.

    At inference time (real battle), we:
    1. Determinize the hidden info (opponent's unrevealed mons)
    2. Run N simulations from the root
    3. Return action with highest visit count for side 1 (the agent)

    During self-play training (poke-engine), both sides are fully observable.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        n_simulations: int = 200,
        c_puct: float = 1.5,
        max_depth: int = 50,
        n_determinizations: int = 1,  # For training; set >1 for real games
        temperature: float = 1.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.eval()
        self.obs_builder = ObsBuilder()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.n_determinizations = n_determinizations
        self.temperature = temperature

    @torch.no_grad()
    def _eval_state(self, state, s1_meta, s2_meta, turn, legal_s1):
        """
        Evaluate state with the model.
        Returns (prior_s1: np.ndarray, prior_s2: np.ndarray, value: float)
        where value is from side 1's perspective.
        """
        legal_s2 = _build_legal_mask_from_state(state.side_two)
        obs_s1 = self._build_obs(state, s1_meta, s2_meta, turn, legal_s1, flip=False)
        obs_s2 = self._build_obs(state, s1_meta, s2_meta, turn, legal_s2, flip=True)

        int1 = torch.from_numpy(obs_s1["int_ids"]).unsqueeze(0).to(self.device)
        flt1 = torch.from_numpy(obs_s1["float_feats"]).unsqueeze(0).to(self.device)
        lm1  = torch.from_numpy(obs_s1["legal_mask"]).unsqueeze(0).to(self.device)

        int2 = torch.from_numpy(obs_s2["int_ids"]).unsqueeze(0).to(self.device)
        flt2 = torch.from_numpy(obs_s2["float_feats"]).unsqueeze(0).to(self.device)
        lm2  = torch.from_numpy(obs_s2["legal_mask"]).unsqueeze(0).to(self.device)

        lp1, _, v1 = self.model(int1, flt1, lm1)
        lp2, _, _  = self.model(int2, flt2, lm2)

        prior1 = torch.softmax(lp1[0], dim=0).cpu().numpy()
        prior2 = torch.softmax(lp2[0], dim=0).cpu().numpy()
        value  = float(v1.item())

        return prior1, prior2, value

    def _build_obs(self, state, s1_meta, s2_meta, turn, legal_actions, flip=False):
        """Build obs dict for one side (flip=True swaps sides)."""
        if not flip:
            obs_dict = {
                "side_one": self._side_to_dict(state.side_one, s1_meta, is_own=True, turn=turn),
                "side_two": self._side_to_dict(state.side_two, s2_meta, is_own=False, turn=turn),
            }
        else:
            obs_dict = {
                "side_one": self._side_to_dict(state.side_two, s2_meta, is_own=True, turn=turn),
                "side_two": self._side_to_dict(state.side_one, s1_meta, is_own=False, turn=turn),
            }

        from pokebot.env.poke_engine_env import _WEATHER_MAP, _TERRAIN_MAP
        obs_dict.update({
            "weather": _WEATHER_MAP.get(str(state.weather).lower(), ""),
            "weather_turns": state.weather_turns_remaining,
            "terrain": _TERRAIN_MAP.get(str(state.terrain).lower(), ""),
            "terrain_turns": state.terrain_turns_remaining,
            "trick_room": state.trick_room,
            "trick_room_turns": state.trick_room_turns_remaining,
            "turn": turn,
            "legal_actions": legal_actions,
        })
        return self.obs_builder.encode(obs_dict)

    def _side_to_dict(self, side, meta, is_own, turn):
        from pokebot.env.poke_engine_env import _pe_side_to_dict
        return _pe_side_to_dict(side, meta, is_own=is_own, turn=turn)

    # ------------------------------------------------------------------

    def _simulate(self, root: DUCTNode, depth: int = 0) -> float:
        """
        One MCTS simulation from root. Returns value from side 1 perspective.
        """
        if root.is_terminal:
            return root.terminal_value
        if depth >= self.max_depth:
            # Leaf eval without expansion
            legal1 = root._legal1
            _, _, value = self._eval_state(
                root.state, root.s1_meta, root.s2_meta, root.turn, legal1
            )
            return value

        # Select actions via DUCT UCB
        a1 = root.ucb_action_s1(self.c_puct)
        a2 = root.ucb_action_s2(self.c_puct)

        key = (a1, a2)
        if key not in root.children:
            # Expand: create new child node
            s1_mv = _action_to_move_str(a1, root.state.side_one)
            s2_mv = _action_to_move_str(a2, root.state.side_two)
            new_state = _sample_and_apply(root.state, s1_mv, s2_mv)
            child = DUCTNode(new_state, root.s1_meta, root.s2_meta, root.turn + 1)

            # Initialize priors from model
            if not child.is_terminal:
                p1, p2, value = self._eval_state(
                    new_state, root.s1_meta, root.s2_meta, root.turn + 1,
                    child._legal1,
                )
                child.prior1 = p1
                child.prior2 = p2
            else:
                value = child.terminal_value

            root.children[key] = child

            # Backprop
            root.N1[a1] += 1
            root.N2[a2] += 1
            root.Q1[a1] += value
            root.Q2[a2] += value
            root.visit_count += 1
            return value

        # Recurse
        child = root.children[key]
        value = self._simulate(child, depth + 1)

        root.N1[a1] += 1
        root.N2[a2] += 1
        root.Q1[a1] += value
        root.Q2[a2] += value
        root.visit_count += 1
        return value

    def search(self, state, s1_meta, s2_meta, turn: int) -> np.ndarray:
        """
        Run MCTS from the given state.
        Returns action visit count distribution for side 1 (shape: n_actions).
        """
        root = DUCTNode(state, s1_meta, s2_meta, turn)

        if root.is_terminal:
            legal = root._legal1
            pi = np.zeros(MODEL_CONFIG["n_actions"], dtype=np.float32)
            if legal:
                pi[legal[0]] = 1.0
            return pi

        # Initialize priors
        p1, p2, _ = self._eval_state(
            state, s1_meta, s2_meta, turn, root._legal1
        )
        root.prior1 = p1
        root.prior2 = p2

        for _ in range(self.n_simulations):
            self._simulate(root)

        # Build policy from visit counts with temperature
        N = root.N1.copy()
        # Zero out illegal actions
        mask = np.zeros_like(N)
        for a in root._legal1:
            mask[a] = 1.0
        N = N * mask

        if self.temperature <= 0 or self.temperature < 1e-3:
            # Argmax
            pi = np.zeros_like(N)
            pi[N.argmax()] = 1.0
        else:
            N_temp = N ** (1.0 / self.temperature)
            total = N_temp.sum()
            pi = N_temp / max(total, 1e-8)

        return pi

    def select_action(self, state, s1_meta, s2_meta, turn: int) -> int:
        """Run search and sample/argmax action for side 1."""
        pi = self.search(state, s1_meta, s2_meta, turn)
        if self.temperature <= 1e-3:
            return int(pi.argmax())
        return int(np.random.choice(len(pi), p=pi / pi.sum()))


# ---------------------------------------------------------------------------
# MCTS-augmented player (for poke-env real battles)
# ---------------------------------------------------------------------------

class MCTSPlayer:
    """
    Wraps a PokeTransformerPlayer with MCTS lookahead.

    During real PS battles, we can't directly use poke-engine for tree search
    since we'd need the actual game state. This player:
    1. Builds a determinized poke-engine state from the poke-env battle
    2. Runs DUCT MCTS on it
    3. Returns the most visited action

    This is an approximation since we're determinizing hidden info.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        n_simulations: int = 100,
        c_puct: float = 1.5,
    ):
        self.mcts = DUCTMCTS(
            model=model,
            device=device,
            n_simulations=n_simulations,
            c_puct=c_puct,
        )
