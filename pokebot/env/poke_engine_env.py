"""
Gymnasium environment wrapping poke-engine for Gen 8 Random Battle training.

poke-engine exposes a Python API (via PyO3 Rust bindings).
The API we rely on (from poke-engine docs):

  from poke_engine import State, MoveChoice, SwitchChoice
  from poke_engine import state_from_string, generate_instructions, ...

Each step:
  1. Both sides pick a move simultaneously (DUCT during MCTS; one agent during training).
  2. poke-engine generates instructions and applies them.
  3. We observe the new state.

For RL training we control side_one; the opponent policy is passed as a callable.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym

from pokebot.env.obs_builder import ObsBuilder
from pokebot.env.reward_shaper import compute_reward

try:
    from poke_engine import State, generate_instructions
    from poke_engine.helpers import state_from_teams  # hypothetical helper
    POKE_ENGINE_AVAILABLE = True
except ImportError:
    POKE_ENGINE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Random team generator (placeholder until poke-engine team loading works)
# ---------------------------------------------------------------------------

def _random_gen8_state() -> Any:
    """
    Create a randomized Gen 8 Random Battle state.
    In production this uses poke-engine's built-in team generator.
    During development on Mac (no poke-engine), returns a mock dict.
    """
    if not POKE_ENGINE_AVAILABLE:
        return _mock_state()
    # poke-engine's generate_random_state or equivalent
    try:
        from poke_engine import generate_random_state
        return generate_random_state(format="gen8randombattle")
    except Exception:
        return _mock_state()


def _mock_state() -> dict:
    """Minimal mock battle state for testing obs_builder without poke-engine."""
    def _mon(species: str, slot: int, is_active: bool = False) -> dict:
        return {
            "species": species,
            "hp": 300,
            "maxhp": 300,
            "status": None,
            "boosts": {},
            "moves": [
                {"id": "tackle", "basePower": 40, "type": "Normal",
                 "category": "physical", "priority": 0, "pp": 35, "maxpp": 35, "is_known": True},
                {"id": "flamethrower", "basePower": 90, "type": "Fire",
                 "category": "special", "priority": 0, "pp": 15, "maxpp": 15, "is_known": True},
            ],
            "ability": "blaze",
            "item": "choiceband",
            "types": ["Fire"],
            "base_stats": {"hp": 78, "attack": 84, "defense": 78,
                           "special-attack": 109, "special-defense": 85, "speed": 100},
            "volatile_statuses": [],
            "is_fainted": False,
            "is_dynamaxed": False,
            "can_dynamax": True,
            "dynamax_turns_remaining": 0,
            "is_active": is_active,
        }

    def _side(prefix: str) -> dict:
        return {
            "active": _mon(f"{prefix}charizard", slot=0, is_active=True),
            "reserve": [_mon(f"{prefix}blastoise", slot=i+1) for i in range(5)],
            "hazards": {},
            "screens": {},
        }

    return {
        "side_one": _side("own_"),
        "side_two": _side("opp_"),
        "weather": None,
        "weather_turns": 0,
        "terrain": None,
        "terrain_turns": 0,
        "trick_room": False,
        "trick_room_turns": 0,
        "turn": 1,
        "legal_actions": list(range(10)),
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PokeEngineEnv(gym.Env):
    """
    Single-agent wrapper around poke-engine.

    Observation space: dict with "int_ids", "float_feats", "legal_mask"
    Action space: Discrete(10)  — 4 moves + 6 switches

    The opponent policy is a callable: state_dict → action_int.
    During self-play this is updated to point at the latest (or a past) checkpoint.
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent_policy: Optional[Callable] = None):
        super().__init__()
        self.obs_builder = ObsBuilder()
        self.opponent_policy: Callable = opponent_policy or _random_opponent

        from pokebot.env.obs_builder import FLOAT_DIM_PER_POKEMON, N_TOKENS
        from config.model_config import MODEL_CONFIG
        n_tokens = N_TOKENS
        float_dim = FLOAT_DIM_PER_POKEMON
        n_actions = MODEL_CONFIG["n_actions"]

        self.observation_space = gym.spaces.Dict({
            "int_ids": gym.spaces.Box(0, 2**31 - 1, shape=(n_tokens, 7), dtype=np.int64),
            "float_feats": gym.spaces.Box(-1e6, 1e6, shape=(n_tokens, float_dim), dtype=np.float32),
            "legal_mask": gym.spaces.Box(0, 1, shape=(n_actions,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(n_actions)

        self._state: Optional[dict] = None
        self._prev_state: Optional[dict] = None
        self._done: bool = True

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state = _random_gen8_state()
        self._prev_state = None
        self._done = False
        obs = self.obs_builder.encode(self._state)
        return obs, {}

    def step(self, action: int):
        assert not self._done, "Call reset() before step()"

        # Opponent picks its action
        opp_action = self.opponent_policy(self._state)
        opp_action = int(opp_action)

        # Apply both actions
        next_state, result = self._apply_actions(self._state, int(action), opp_action)

        reward = compute_reward(self._state, next_state, result)
        terminated = result is not None
        truncated = False

        self._prev_state = self._state
        self._state = next_state
        self._done = terminated

        obs = self.obs_builder.encode(next_state)
        info = {"result": result, "turn": next_state.get("turn", 0)}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _apply_actions(self, state: dict, own_action: int, opp_action: int) -> tuple[dict, Optional[str]]:
        """
        Apply both players' actions and return (next_state, result).
        result is "win"/"loss"/"tie" if terminal, else None.
        """
        if POKE_ENGINE_AVAILABLE:
            return self._apply_poke_engine(state, own_action, opp_action)
        else:
            return self._apply_mock(state, own_action, opp_action)

    def _apply_poke_engine(self, state: dict, own_action: int, opp_action: int):
        """Real poke-engine step. To be implemented when poke-engine is installed."""
        # poke-engine API (placeholder — adapt to actual API once installed):
        #   instructions = generate_instructions(state_obj, side1_choice, side2_choice)
        #   state_obj.apply_instructions(instructions)
        #   check state_obj.is_over()
        raise NotImplementedError("Install poke-engine on Windows for full training")

    def _apply_mock(self, state: dict, own_action: int, opp_action: int) -> tuple[dict, Optional[str]]:
        """
        Mock step for testing on Mac without poke-engine.
        Randomly ends the game after ~30 turns.
        """
        import copy
        next_state = copy.deepcopy(state)
        next_state["turn"] = state.get("turn", 1) + 1

        # Mock: random game over after 20-40 turns
        if next_state["turn"] > random.randint(20, 40):
            result = random.choice(["win", "loss"])
        else:
            result = None

        # Update legal actions (stay the same in mock)
        next_state["legal_actions"] = state.get("legal_actions", list(range(10)))
        return next_state, result

    def set_opponent(self, policy: Callable):
        self.opponent_policy = policy


def _random_opponent(state: dict) -> int:
    """Baseline: pick a random legal action."""
    legal = state.get("legal_actions", list(range(4)))
    return random.choice(legal)
