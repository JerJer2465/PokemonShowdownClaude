"""
Gymnasium environment wrapping poke-engine for Gen 8 Random Battle training.

Each step:
  1. Both sides pick a move simultaneously.
  2. generate_instructions() resolves the turn stochastically.
  3. The chosen outcome is applied to the state in-place.
  4. Forced-switch frames are handled inline (no reward, same timestep from PPO perspective).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import gymnasium as gym

from poke_engine import (
    State, Side, Pokemon, Move, SideConditions,
    PokemonIndex, Weather, generate_instructions,
)

# Optional imports that may not exist in Gen 4 poke-engine builds
try:
    from poke_engine import Terrain as _Terrain
    _TERRAIN_NONE = _Terrain.NONE
    _HAS_TERRAIN = True
except (ImportError, AttributeError):
    _Terrain = None  # type: ignore
    _TERRAIN_NONE = None
    _HAS_TERRAIN = False

try:
    from poke_engine import VolatileStatusDurations  # noqa: F401
except ImportError:
    pass

from pokebot.env.obs_builder import ObsBuilder, FLOAT_DIM_PER_POKEMON, N_TOKENS
from pokebot.env.reward_shaper import compute_reward
from config.model_config import MODEL_CONFIG

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# ---------------------------------------------------------------------------
# Nature stat multipliers
# ---------------------------------------------------------------------------
_NATURE_MODS: dict[str, tuple[int, int]] = {
    # (boosted_stat_idx, reduced_stat_idx) — using [hp,atk,def,spa,spd,spe]
    "hardy": (0, 0), "docile": (1, 1), "serious": (2, 2),
    "bashful": (3, 3), "quirky": (4, 4),
    "lonely": (1, 2), "brave": (1, 5), "adamant": (1, 3), "naughty": (1, 4),
    "bold": (2, 1), "relaxed": (2, 5), "impish": (2, 3), "lax": (2, 4),
    "timid": (5, 1), "hasty": (5, 2), "jolly": (5, 3), "naive": (5, 4),
    "modest": (3, 1), "mild": (3, 2), "quiet": (3, 5), "rash": (3, 4),
    "calm": (4, 1), "gentle": (4, 2), "sassy": (4, 5), "careful": (4, 3),
}

def _calc_stat(base: int, ev: int, level: int, nature_idx: int, stat_slot: int) -> int:
    """Standard Gen 8 stat formula. stat_slot: 0=hp,1=atk,2=def,3=spa,4=spd,5=spe."""
    iv = 31
    if stat_slot == 0:  # HP
        return math.floor((2 * base + iv + ev // 4) * level / 100) + level + 10
    raw = math.floor((2 * base + iv + ev // 4) * level / 100) + 5
    # nature modifier
    return math.floor(raw * _nature_multiplier(nature_idx, stat_slot))

def _nature_multiplier(nature_idx: int, stat_slot: int) -> float:
    name = list(_NATURE_MODS.keys())[nature_idx] if nature_idx < len(_NATURE_MODS) else "hardy"
    boost, reduce = _NATURE_MODS.get(name, (0, 0))
    if boost == reduce:
        return 1.0
    if stat_slot == boost:
        return 1.1
    if stat_slot == reduce:
        return 0.9
    return 1.0


# ---------------------------------------------------------------------------
# Randbats team generator
# ---------------------------------------------------------------------------

class RandbatsGenerator:
    """Generates random Gen 4 teams from the pkmn/randbats data."""

    _instance: Optional["RandbatsGenerator"] = None

    def __init__(self):
        path = DATA_DIR / "gen4randombattle.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python scripts/build_vocab.py"
            )
        with open(path) as f:
            self._data: dict = json.load(f)
        self._species_list = list(self._data.keys())

        # Base stats lookup from data (we embed them in mon metadata for obs_builder)
        # We store them as a separate dict keyed by normalized species name
        self._base_stats: dict[str, dict] = {}
        self._load_base_stats()

        # Move metadata lookup for obs features
        self._move_data: dict[str, dict] = {}
        self._load_move_data()

    def _load_base_stats(self):
        """Load base stats from gen4_base_stats.json if available."""
        bst_path = DATA_DIR / "gen4_base_stats.json"
        if bst_path.exists():
            with open(bst_path) as f:
                self._base_stats = {k.lower(): v for k, v in json.load(f).items()}

    def _load_move_data(self):
        """Load move metadata (basePower, type, category, priority) from gen4_move_data.json."""
        md_path = DATA_DIR / "gen4_move_data.json"
        if md_path.exists():
            with open(md_path) as f:
                self._move_data = json.load(f)

    @classmethod
    def get(cls) -> "RandbatsGenerator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _normalize(self, name: str) -> str:
        return "".join(c for c in name.lower() if c.isalnum())

    def _norm_move_id(self, name: str) -> str:
        """Normalize a move name for poke-engine. Strips type suffix from Hidden Power."""
        norm = self._normalize(name)
        # poke-engine uses "hiddenpower" not "hiddenpowerice", "hiddenpowerfire", etc.
        if norm.startswith("hiddenpower") and len(norm) > len("hiddenpower"):
            return "hiddenpower"
        return norm

    def sample_team(self) -> tuple[list[Pokemon], list[dict]]:
        """
        Returns (poke_engine_team, meta_list).
        meta_list[i] has the extra info needed by obs_builder (base_stats, move_data).
        """
        species_sample = random.sample(self._species_list, 6)
        mons: list[Pokemon] = []
        metas: list[dict] = []

        for species in species_sample:
            entry = self._data[species]
            mon, meta = self._build_mon(species, entry)
            mons.append(mon)
            metas.append(meta)

        return mons, metas

    def _build_mon(self, species: str, entry: dict) -> tuple[Pokemon, dict]:
        norm_species = self._normalize(species)
        level = entry.get("level", 100)

        # pkmn/randbats format: moves/abilities/items may be nested under "roles"
        if "roles" in entry:
            roles = entry["roles"]
            role = roles[random.choice(list(roles.keys()))]
            move_pool = role.get("moves", [])
            ability_pool = role.get("abilities", entry.get("abilities", ["none"]))
            item_pool = role.get("items", entry.get("items", ["none"]))
            evs_entry = role.get("evs", entry.get("evs", {}))
        else:
            move_pool = entry.get("moves", ["tackle"])
            ability_pool = entry.get("abilities", ["none"])
            item_pool = entry.get("items", ["none"])
            evs_entry = entry.get("evs", {})

        n_moves = min(4, len(move_pool))
        chosen_moves = random.sample(move_pool, n_moves)
        chosen_ability = random.choice(ability_pool)
        chosen_item = random.choice(item_pool) if item_pool else "none"

        # EVs: randbats provides partial EV spread
        ev_hp  = evs_entry.get("hp", 0)
        ev_atk = evs_entry.get("atk", 0)
        ev_def = evs_entry.get("def", 0)
        ev_spa = evs_entry.get("spa", 0)
        ev_spd = evs_entry.get("spd", 0)
        ev_spe = evs_entry.get("spe", 0)

        # Base stats
        bst = self._base_stats.get(norm_species, {})
        b_hp  = bst.get("hp", 80)
        b_atk = bst.get("attack", 80)
        b_def = bst.get("defense", 80)
        b_spa = bst.get("special-attack", 80)
        b_spd = bst.get("special-defense", 80)
        b_spe = bst.get("speed", 80)

        # Random nature
        nature_name = random.choice(list(_NATURE_MODS.keys()))
        nature_idx = list(_NATURE_MODS.keys()).index(nature_name)

        # Compute stats
        hp  = _calc_stat(b_hp,  ev_hp,  level, nature_idx, 0)
        atk = _calc_stat(b_atk, ev_atk, level, nature_idx, 1)
        def_ = _calc_stat(b_def, ev_def, level, nature_idx, 2)
        spa = _calc_stat(b_spa, ev_spa, level, nature_idx, 3)
        spd = _calc_stat(b_spd, ev_spd, level, nature_idx, 4)
        spe = _calc_stat(b_spe, ev_spe, level, nature_idx, 5)

        # Types from base stats table (fallback: normal/typeless)
        types_raw = bst.get("types", ["Normal"])
        if len(types_raw) == 1:
            type_tuple = (types_raw[0].lower(), "typeless")
        else:
            type_tuple = (types_raw[0].lower(), types_raw[1].lower())

        pe_moves = [
            Move(id=self._norm_move_id(m), pp=16)
            for m in chosen_moves
        ]
        # Pad to 4 moves
        while len(pe_moves) < 4:
            pe_moves.append(Move(id="none", pp=1))

        mon = Pokemon(
            id=norm_species,
            level=level,
            types=type_tuple,
            base_types=type_tuple,
            hp=hp,
            maxhp=hp,
            ability=self._normalize(chosen_ability),
            base_ability=self._normalize(chosen_ability),
            item=self._normalize(chosen_item),
            nature=nature_name,
            evs=(ev_hp, ev_atk, ev_def, ev_spa, ev_spd, ev_spe),
            attack=atk,
            defense=def_,
            special_attack=spa,
            special_defense=spd,
            speed=spe,
            moves=pe_moves,
        )

        # Meta for obs_builder (base stats + move metadata for obs encoding)
        move_names_norm = [self._normalize(m) for m in chosen_moves]
        meta = {
            "base_stats": {
                "hp": b_hp, "attack": b_atk, "defense": b_def,
                "special-attack": b_spa, "special-defense": b_spd, "speed": b_spe,
            },
            "types": list(types_raw),
            "move_names": move_names_norm,
            "move_meta": [self._move_data.get(mn, {}) for mn in move_names_norm],
        }
        return mon, meta


def _make_side(mons: list[Pokemon]) -> Side:
    return Side(
        pokemon=mons,
        side_conditions=SideConditions(),
        active_index=PokemonIndex.P0,
    )


def _random_gen4_state() -> tuple[State, list[dict], list[dict]]:
    """
    Returns (state, s1_meta, s2_meta).
    meta[i] = {base_stats, types, move_names, move_meta} for slot i.
    """
    gen = RandbatsGenerator.get()
    s1_mons, s1_meta = gen.sample_team()
    s2_mons, s2_meta = gen.sample_team()
    state_kwargs: dict = {
        "side_one": _make_side(s1_mons),
        "side_two": _make_side(s2_mons),
        "weather": Weather.NONE,
    }
    if _HAS_TERRAIN:
        state_kwargs["terrain"] = _TERRAIN_NONE
    state = State(**state_kwargs)
    return state, s1_meta, s2_meta


# ---------------------------------------------------------------------------
# State → obs dict converter
# ---------------------------------------------------------------------------

_WEATHER_MAP = {
    Weather.NONE: "", "none": "",
    Weather.SUN: "sunnyday", "sun": "sunnyday",
    Weather.RAIN: "raindance", "rain": "raindance",
    Weather.SAND: "sandstorm", "sand": "sandstorm",
    Weather.HAIL: "hail", "hail": "hail",
    # String fallbacks for any weather string poke-engine may produce
    "snow": "hail",           # Gen 9 rename of hail — map to hail for Gen 4 compat
    "snowscape": "hail",
    "harshsun": "sunnyday",
    "heavyrain": "raindance",
}

# Terrain doesn't exist in Gen 4 — only string keys needed as fallback
_TERRAIN_MAP = {
    "": "", "none": "",
    "electricterrain": "electricterrain",
    "grassyterrain": "grassyterrain",
    "mistyterrain": "mistyterrain",
    "psychicterrain": "psychicterrain",
}

_STATUS_MAP = {
    "none": None, "": None,
    "burn": "brn", "brn": "brn",
    "poison": "psn", "psn": "psn",
    "toxic": "tox", "tox": "tox",
    "sleep": "slp", "slp": "slp",
    "freeze": "frz", "frz": "frz",
    "paralysis": "par", "par": "par",
}


def _pe_side_to_dict(
    side: Side,
    meta: list[dict],
    is_own: bool,
    turn: int,
) -> dict:
    """Convert a poke_engine Side + metadata → obs_builder dict format."""
    active_idx = int(str(side.active_index))  # PokemonIndex "0".."5" → int

    def _mon_to_dict(mon: Pokemon, slot: int, is_active: bool, mon_meta: dict) -> dict:
        # Boosts (poke-engine stores on Side for active mon, zeros for reserve)
        if is_active:
            boosts = {
                "atk": side.attack_boost,
                "def": side.defense_boost,
                "spa": side.special_attack_boost,
                "spd": side.special_defense_boost,
                "spe": side.speed_boost,
                "accuracy": side.accuracy_boost,
                "evasion": side.evasion_boost,
            }
            volatile = set(side.volatile_statuses)
        else:
            boosts = {}
            volatile = set()

        # Move metadata lookup: prefer move_meta from RandbatsGenerator if available
        move_meta_list = mon_meta.get("move_meta", [])

        # Moves
        moves_out = []
        for i, pe_move in enumerate(mon.moves):
            if pe_move.id in ("none", ""):
                continue
            meta_entry = move_meta_list[i] if i < len(move_meta_list) else {}
            moves_out.append({
                "id": pe_move.id,
                "basePower": meta_entry.get("basePower", 0),
                "accuracy": meta_entry.get("accuracy", 100),
                "type": meta_entry.get("type", "Normal"),
                "category": meta_entry.get("category", "physical"),
                "priority": meta_entry.get("priority", 0),
                "pp": pe_move.pp,
                "maxpp": 16,
                "is_known": is_own or is_active,
                "disabled": pe_move.disabled,
            })

        status_raw = str(mon.status).lower() if mon.status else ""
        status = _STATUS_MAP.get(status_raw)

        types = list(mon.types)
        if types[1] == "typeless":
            types = [types[0]]

        return {
            "species": mon.id,
            "level": mon.level,
            "hp": mon.hp,
            "maxhp": mon.maxhp,
            "status": status,
            "boosts": boosts,
            "moves": moves_out,
            "ability": mon.ability,   # poke-engine: always available in simulation
            "item": mon.item,         # poke-env: None until revealed, then actual value
            "types": types,
            "base_stats": mon_meta.get("base_stats", {}),
            "volatile_statuses": list(volatile),
            "is_fainted": mon.hp <= 0,
            "is_active": is_active,
            # Per-mon counters (relevant for active mon, 0 for reserve)
            "sleep_turns": mon.sleep_turns,
            "rest_turns": mon.rest_turns,
        }

    # Build ordered list: active first, then reserve
    mons_in_order: list[tuple[Pokemon, int, bool, dict]] = []
    # active slot
    active_mon = side.pokemon[active_idx]
    active_meta = meta[active_idx] if active_idx < len(meta) else {}
    mons_in_order.append((active_mon, active_idx, True, active_meta))
    # reserve (all non-active, in slot order)
    for i, mon in enumerate(side.pokemon):
        if i == active_idx:
            continue
        m = meta[i] if i < len(meta) else {}
        mons_in_order.append((mon, i, False, m))

    active_dict = _mon_to_dict(*mons_in_order[0])
    reserve_dicts = [_mon_to_dict(*t) for t in mons_in_order[1:]]

    sc = side.side_conditions
    vsd = side.volatile_status_durations
    return {
        "active": active_dict,
        "reserve": reserve_dicts,
        "hazards": {
            "stealth_rock": bool(sc.stealth_rock),
            "spikes": sc.spikes,
            "toxic_spikes": sc.toxic_spikes,
            "sticky_web": bool(getattr(sc, "sticky_web", 0)),  # Gen 6+
        },
        "screens": {
            "light_screen": sc.light_screen,
            "reflect": sc.reflect,
            "aurora_veil": int(getattr(sc, "aurora_veil", 0)),  # Gen 7+
        },
        # Side-level state for encoding
        "last_used_move": str(side.last_used_move).replace("move:", ""),
        "substitute_health": side.substitute_health,
        "force_trapped": bool(side.force_trapped),
        "wish": side.wish,               # (turns, hp) tuple
        "toxic_count": sc.toxic_count,
        "tailwind": sc.tailwind,
        "safeguard": sc.safeguard,
        "protect_count": sc.protect,
        "locked_move": bool(vsd.lockedmove > 0),
        "perish_count": getattr(side, "perish_count", 0),  # TODO: verify poke-engine field name
        "mist": getattr(sc, "mist", 0),                    # TODO: verify poke-engine field name
        "lucky_chant": getattr(sc, "lucky_chant", 0),      # TODO: verify poke-engine field name
        "volatile_durations": {
            "confusion": vsd.confusion,
            "taunt": vsd.taunt,
            "encore": vsd.encore,
            "yawn": vsd.yawn,
            "lockedmove": vsd.lockedmove,
        },
    }


def _state_to_obs_dict(
    state: State,
    s1_meta: list[dict],
    s2_meta: list[dict],
    turn: int,
    legal_actions: list[int],
) -> dict:
    terrain_str = ""
    terrain_turns = 0
    if _HAS_TERRAIN and hasattr(state, "terrain"):
        terrain_str = _TERRAIN_MAP.get(str(state.terrain).lower(), "")
        terrain_turns = getattr(state, "terrain_turns_remaining", 0)
    return {
        "side_one": _pe_side_to_dict(state.side_one, s1_meta, is_own=True, turn=turn),
        "side_two": _pe_side_to_dict(state.side_two, s2_meta, is_own=False, turn=turn),
        "weather": _WEATHER_MAP.get(str(state.weather).lower(), ""),
        "weather_turns": state.weather_turns_remaining,
        "terrain": terrain_str,
        "terrain_turns": terrain_turns,
        "trick_room": state.trick_room,
        "trick_room_turns": state.trick_room_turns_remaining,
        "gravity": getattr(state, "gravity", False),
        "gravity_turns": getattr(state, "gravity_turns_remaining", 0),
        "turn": turn,
        "legal_actions": legal_actions,
    }


# ---------------------------------------------------------------------------
# Action mapping
# ---------------------------------------------------------------------------

def _build_legal_mask_from_state(side: Side) -> list[int]:
    """
    Returns list of legal action indices (0-9).
    0-3 = moves, 4-9 = switches.
    If force_switch: only switches legal.
    """
    active_idx = int(str(side.active_index))
    active = side.pokemon[active_idx]
    legal: list[int] = []

    # Treat fainted active as force_switch (poke-engine doesn't auto-set it on faint)
    active_fainted = active.hp <= 0
    if not side.force_switch and not active_fainted:
        for i, mv in enumerate(active.moves[:4]):
            if mv.id not in ("none", "") and not mv.disabled and mv.pp > 0:
                legal.append(i)

    # Switches: dense indexing over alive non-active mons only
    # action 4 → first alive non-active, 5 → second, etc.
    # Must match _action_to_move_str's candidate ordering exactly.
    switch_slot = 4
    for i, mon in enumerate(side.pokemon):
        if i == active_idx:
            continue
        if switch_slot > 9:
            break
        if mon.hp > 0:
            legal.append(switch_slot)
            switch_slot += 1  # only advance for alive mons

    if not legal:
        # Struggle: if all moves are disabled/out of PP, allow move 0
        legal = [0]

    return legal


def _action_to_move_str(action: int, side: Side) -> str:
    """
    Convert action index (0-9) to poke-engine move string.

    poke-engine's generate_instructions resolves the string by checking:
      1. Is it a move id on the active pokemon? → use as move
      2. Is it a reserve pokemon's id?          → treat as switch
    So switches are encoded as just the pokemon's id (no "switch " prefix).
    """
    active_idx = int(str(side.active_index))
    active = side.pokemon[active_idx]

    if action < 4:
        if action < len(active.moves):
            mv_id = active.moves[action].id
            if mv_id in ("none", ""):
                return "struggle"
            return mv_id
        return "struggle"
    else:
        # Switch: collect alive non-active slots in order, map offset 0→first, 1→second, ...
        switch_offset = action - 4
        candidates = [
            mon for i, mon in enumerate(side.pokemon)
            if i != active_idx and mon.hp > 0
        ]
        if switch_offset < len(candidates):
            return candidates[switch_offset].id
        # Fallback to first alive non-active
        for i, mon in enumerate(side.pokemon):
            if i != active_idx and mon.hp > 0:
                return mon.id
        return "struggle"


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

def _is_terminal(state: State) -> Optional[str]:
    """Return 'win', 'loss', 'tie', or None."""
    s1_alive = any(m.hp > 0 for m in state.side_one.pokemon)
    s2_alive = any(m.hp > 0 for m in state.side_two.pokemon)
    if not s1_alive and not s2_alive:
        return "tie"
    if not s1_alive:
        return "loss"
    if not s2_alive:
        return "win"
    return None


# ---------------------------------------------------------------------------
# Stochastic transition
# ---------------------------------------------------------------------------

def _first_alive_non_active(side: Side) -> str:
    """Return the ID of the first alive non-active Pokemon (for faint-switch handling)."""
    active_idx = int(str(side.active_index))
    for i, mon in enumerate(side.pokemon):
        if i != active_idx and mon.hp > 0:
            return mon.id
    return "struggle"


def _first_valid_move(side: Side) -> str:
    """Return the first non-empty move id for the active pokemon (used as dummy for force_switch turns)."""
    active_idx = int(str(side.active_index))
    active = side.pokemon[active_idx]
    for mv in active.moves:
        if mv.id not in ("none", "") and mv.pp > 0:
            return mv.id
    return "tackle"  # absolute fallback


def _sample_and_apply(state: State, s1_move: str, s2_move: str) -> State:
    """
    Stochastically sample one outcome from generate_instructions and apply it.
    apply_instructions() returns a NEW state (does not mutate in place).
    """
    outcomes = generate_instructions(state, s1_move, s2_move)
    if not outcomes:
        return state
    weights = [o.percentage for o in outcomes]
    total = sum(weights)
    if total <= 0:
        chosen = outcomes[0]
    else:
        r = random.random() * total
        cumul = 0.0
        chosen = outcomes[-1]
        for o in outcomes:
            cumul += o.percentage
            if r <= cumul:
                chosen = o
                break
    return state.apply_instructions(chosen)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PokeEngineEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper around poke-engine.

    Observation space: dict {"int_ids", "float_feats", "legal_mask"}
    Action space: Discrete(10)  — 4 moves + 6 switches

    The opponent policy is a callable: (obs_dict) -> action_int.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_policy: Optional[Callable] = None,
        mcts_opponent_ms: int = 0,
        mcts_opponent_prob: float = 0.5,
    ):
        """
        opponent_policy: callable(obs_dict) -> action_int  (fallback / heuristic)
        mcts_opponent_ms: if > 0, use MCTS for the opponent with this time budget (ms)
        mcts_opponent_prob: probability per episode of using MCTS vs heuristic (curriculum)
                            1.0 = always MCTS, 0.0 = always heuristic
        """
        super().__init__()
        self.obs_builder = ObsBuilder()
        self.opponent_policy: Callable = opponent_policy or _random_opponent
        self.mcts_opponent_ms: int = mcts_opponent_ms
        self.mcts_opponent_prob: float = mcts_opponent_prob
        self._episode_use_mcts: bool = False  # set each reset()

        n_actions = MODEL_CONFIG["n_actions"]
        self.observation_space = gym.spaces.Dict({
            "int_ids":    gym.spaces.Box(0, 2**31-1, shape=(N_TOKENS, 8), dtype=np.int64),
            "float_feats": gym.spaces.Box(-1e6, 1e6, shape=(N_TOKENS, FLOAT_DIM_PER_POKEMON), dtype=np.float32),
            "legal_mask": gym.spaces.Box(0, 1, shape=(n_actions,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(n_actions)

        self._state: Optional[State] = None
        self._s1_meta: list[dict] = []
        self._s2_meta: list[dict] = []
        self._turn: int = 1
        self._done: bool = True

    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._state, self._s1_meta, self._s2_meta = _random_gen4_state()
        self._turn = 1
        self._done = False
        # Curriculum: decide per-episode whether to use MCTS opponent
        if self.mcts_opponent_ms > 0:
            self._episode_use_mcts = random.random() < self.mcts_opponent_prob
        else:
            self._episode_use_mcts = False
        obs_dict = self._build_obs_dict()
        self._last_obs_dict = obs_dict
        obs = self.obs_builder.encode(obs_dict)
        return obs, {}

    def _opp_action(self, perspective: str = "side_two") -> int:
        """Return opponent's chosen action, using MCTS or obs-dict policy."""
        if self._episode_use_mcts:
            # MCTS always works from side_two on the current state
            return mcts_opponent_action(self._state, self.mcts_opponent_ms)
        obs_dict = self._build_obs_dict(perspective=perspective)
        return int(self.opponent_policy(obs_dict))

    def step(self, action: int):
        assert not self._done, "Call reset() before step()"

        opp_action = self._opp_action()

        # Save prev obs dict for reward shaping
        prev_obs_dict = self._build_obs_dict()

        # Convert actions to move strings
        s1_move = _action_to_move_str(int(action), self._state.side_one)
        s2_move = _action_to_move_str(opp_action, self._state.side_two)

        # Apply turn
        self._state = _sample_and_apply(self._state, s1_move, s2_move)
        self._turn += 1

        # Handle forced switches inline (keep stepping until no force_switch)
        max_forced = 6
        for _ in range(max_forced):
            s1_forced = self._state.side_one.force_switch
            s2_forced = self._state.side_two.force_switch
            if not s1_forced and not s2_forced:
                break

            if s1_forced and s2_forced:
                obs_dict_tmp = self._build_obs_dict()
                fs1 = int(self.opponent_policy(obs_dict_tmp))
                fs2 = self._opp_action()
                fsm1 = _action_to_move_str(fs1, self._state.side_one)
                fsm2 = _action_to_move_str(fs2, self._state.side_two)
                self._state = _sample_and_apply(self._state, fsm1, fsm2)
            elif s1_forced:
                obs_dict_tmp = self._build_obs_dict()
                fs1 = int(self.opponent_policy(obs_dict_tmp))
                fsm1 = _action_to_move_str(fs1, self._state.side_one)
                dummy2 = _first_valid_move(self._state.side_two)
                self._state = _sample_and_apply(self._state, fsm1, dummy2)
            else:
                fs2 = self._opp_action()
                fsm2 = _action_to_move_str(fs2, self._state.side_two)
                dummy1 = _first_valid_move(self._state.side_one)
                self._state = _sample_and_apply(self._state, dummy1, fsm2)

        # Handle faint-switches: poke-engine doesn't auto-set force_switch when active faints
        for _ in range(6):
            s1_act_idx = int(str(self._state.side_one.active_index))
            s2_act_idx = int(str(self._state.side_two.active_index))
            s1_fainted = self._state.side_one.pokemon[s1_act_idx].hp <= 0
            s2_fainted = self._state.side_two.pokemon[s2_act_idx].hp <= 0
            if not s1_fainted and not s2_fainted:
                break
            # Check if any alive non-active exist; if not, terminal will be detected below
            s1_has_alive = any(m.hp > 0 for i, m in enumerate(self._state.side_one.pokemon) if i != s1_act_idx)
            s2_has_alive = any(m.hp > 0 for i, m in enumerate(self._state.side_two.pokemon) if i != s2_act_idx)
            if s1_fainted and not s1_has_alive:
                break  # all s1 fainted → terminal
            if s2_fainted and not s2_has_alive:
                break  # all s2 fainted → terminal
            if s1_fainted:
                sw1_obs = self._build_obs_dict()
                sw_a1 = int(self.opponent_policy(sw1_obs))
                sw1 = _action_to_move_str(sw_a1, self._state.side_one)
                if sw1 == "struggle":
                    sw1 = _first_alive_non_active(self._state.side_one)
            else:
                sw1 = _first_valid_move(self._state.side_one)
            if s2_fainted:
                sw2_obs = self._build_obs_dict(perspective="side_two")
                sw_a2 = int(self.opponent_policy(sw2_obs))
                sw2 = _action_to_move_str(sw_a2, self._state.side_two)
                if sw2 == "struggle":
                    sw2 = _first_alive_non_active(self._state.side_two)
            else:
                sw2 = _first_valid_move(self._state.side_two)
            self._state = _sample_and_apply(self._state, sw1, sw2)

        # Check terminal (also enforce max_turns to prevent stalling battles)
        result = _is_terminal(self._state)
        if result is None and self._turn >= 300:
            result = "tie"
        terminated = result is not None
        self._done = terminated

        curr_obs_dict = self._build_obs_dict()
        self._last_obs_dict = curr_obs_dict
        reward = compute_reward(prev_obs_dict, curr_obs_dict, result)

        obs = self.obs_builder.encode(curr_obs_dict)
        info = {"result": result, "turn": self._turn}
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------

    def _build_obs_dict(self, perspective: str = "side_one") -> dict:
        """Build obs dict. perspective='side_one' = agent's view, 'side_two' = opponent's view."""
        if perspective == "side_one":
            s1, s1m, s2, s2m = (
                self._state.side_one, self._s1_meta,
                self._state.side_two, self._s2_meta,
            )
        else:
            # Flip sides so caller always looks from side_one slot
            s1, s1m, s2, s2m = (
                self._state.side_two, self._s2_meta,
                self._state.side_one, self._s1_meta,
            )

        legal = _build_legal_mask_from_state(s1)
        terrain_str = ""
        terrain_turns = 0
        if _HAS_TERRAIN and hasattr(self._state, "terrain"):
            terrain_str = _TERRAIN_MAP.get(str(self._state.terrain).lower(), "")
            terrain_turns = getattr(self._state, "terrain_turns_remaining", 0)

        return {
            "side_one": _pe_side_to_dict(s1, s1m, is_own=True, turn=self._turn),
            "side_two": _pe_side_to_dict(s2, s2m, is_own=False, turn=self._turn),
            "weather": _WEATHER_MAP.get(str(self._state.weather).lower(), ""),
            "weather_turns": self._state.weather_turns_remaining,
            "terrain": terrain_str,
            "terrain_turns": terrain_turns,
            "trick_room": self._state.trick_room,
            "trick_room_turns": self._state.trick_room_turns_remaining,
            "turn": self._turn,
            "legal_actions": legal,
        }

    def get_opponent_obs_encoded(self) -> dict:
        """Get encoded observation from opponent's (side_two) perspective.

        Returns the same dict format as reset()/step(): {int_ids, float_feats, legal_mask}.
        Used by GPU-batched self-play to route opponent inference through the GPU server.
        """
        opp_obs_dict = self._build_obs_dict(perspective="side_two")
        return self.obs_builder.encode(opp_obs_dict)

    def step_dual(self, agent_action: int, opp_action: int):
        """Step with both agent and opponent actions provided externally.

        Same as step() but skips internal _opp_action() call.
        Forced/faint switches use smart_heuristic_opponent (fast CPU fallback).
        """
        assert not self._done, "Call reset() before step()"

        prev_obs_dict = self._build_obs_dict()

        s1_move = _action_to_move_str(int(agent_action), self._state.side_one)
        s2_move = _action_to_move_str(int(opp_action), self._state.side_two)

        self._state = _sample_and_apply(self._state, s1_move, s2_move)
        self._turn += 1

        # Forced switches — use heuristic (fast, no GPU round-trip needed)
        for _ in range(6):
            s1_forced = self._state.side_one.force_switch
            s2_forced = self._state.side_two.force_switch
            if not s1_forced and not s2_forced:
                break

            if s1_forced:
                obs_s1 = self._build_obs_dict()
                fs1 = int(smart_heuristic_opponent(obs_s1))
                fsm1 = _action_to_move_str(fs1, self._state.side_one)
            else:
                fsm1 = _first_valid_move(self._state.side_one)

            if s2_forced:
                obs_s2 = self._build_obs_dict(perspective="side_two")
                fs2 = int(smart_heuristic_opponent(obs_s2))
                fsm2 = _action_to_move_str(fs2, self._state.side_two)
            else:
                fsm2 = _first_valid_move(self._state.side_two)

            self._state = _sample_and_apply(self._state, fsm1, fsm2)

        # Faint-switches — use heuristic
        for _ in range(6):
            s1_act_idx = int(str(self._state.side_one.active_index))
            s2_act_idx = int(str(self._state.side_two.active_index))
            s1_fainted = self._state.side_one.pokemon[s1_act_idx].hp <= 0
            s2_fainted = self._state.side_two.pokemon[s2_act_idx].hp <= 0
            if not s1_fainted and not s2_fainted:
                break
            s1_has_alive = any(m.hp > 0 for i, m in enumerate(self._state.side_one.pokemon) if i != s1_act_idx)
            s2_has_alive = any(m.hp > 0 for i, m in enumerate(self._state.side_two.pokemon) if i != s2_act_idx)
            if s1_fainted and not s1_has_alive:
                break
            if s2_fainted and not s2_has_alive:
                break
            if s1_fainted:
                sw1_obs = self._build_obs_dict()
                sw_a1 = int(smart_heuristic_opponent(sw1_obs))
                sw1 = _action_to_move_str(sw_a1, self._state.side_one)
                if sw1 == "struggle":
                    sw1 = _first_alive_non_active(self._state.side_one)
            else:
                sw1 = _first_valid_move(self._state.side_one)
            if s2_fainted:
                sw2_obs = self._build_obs_dict(perspective="side_two")
                sw_a2 = int(smart_heuristic_opponent(sw2_obs))
                sw2 = _action_to_move_str(sw_a2, self._state.side_two)
                if sw2 == "struggle":
                    sw2 = _first_alive_non_active(self._state.side_two)
            else:
                sw2 = _first_valid_move(self._state.side_two)
            self._state = _sample_and_apply(self._state, sw1, sw2)

        # Terminal check
        result = _is_terminal(self._state)
        if result is None and self._turn >= 300:
            result = "tie"
        terminated = result is not None
        self._done = terminated

        curr_obs_dict = self._build_obs_dict()
        self._last_obs_dict = curr_obs_dict
        reward = compute_reward(prev_obs_dict, curr_obs_dict, result)

        obs = self.obs_builder.encode(curr_obs_dict)
        info = {"result": result, "turn": self._turn}
        return obs, reward, terminated, False, info

    def set_opponent(self, policy: Callable):
        self.opponent_policy = policy


# ---------------------------------------------------------------------------
# Default opponent policies
# ---------------------------------------------------------------------------

def _mcts_move_str_to_action(move_str: str, side: Side) -> int:
    """
    Reverse of _action_to_move_str: convert a poke-engine move string back
    to our action int (0-9).  Used to decode MCTS results.

    move_str is either:
      - a move id  (e.g. "surf", "flamethrower")    → action 0-3 (slot index)
      - a pokemon id (e.g. "vaporeon", "garchomp")   → action 4-9 (switch index)
    """
    active_idx = int(str(side.active_index))
    active = side.pokemon[active_idx]

    # Try moves first
    for slot, mv in enumerate(active.moves[:4]):
        if mv.id == move_str:
            return slot

    # Try switches (dense ordering: alive non-active in team order)
    switch_slot = 4
    for i, mon in enumerate(side.pokemon):
        if i == active_idx:
            continue
        if mon.hp > 0:
            if mon.id == move_str:
                return switch_slot
            switch_slot += 1

    # Fallback: first legal action
    legal = _build_legal_mask_from_state(side)
    return legal[0] if legal else 0


def mcts_opponent_action(state: "State", duration_ms: int = 10) -> int:
    """
    Run MCTS on the current state and return side_two's best action int.
    duration_ms controls search time per move (10ms ≈ 100-200 iterations).
    """
    try:
        from poke_engine import monte_carlo_tree_search
        result = monte_carlo_tree_search(state, duration_ms)
        if result.side_two:
            # Pick move with highest visit count
            best = max(result.side_two, key=lambda r: r.visits)
            return _mcts_move_str_to_action(best.move_choice, state.side_two)
    except Exception:
        pass
    # Fallback to heuristic on any error
    legal = _build_legal_mask_from_state(state.side_two)
    return legal[0] if legal else 0


def mcts_side_one_action(state: "State", duration_ms: int = 10) -> int:
    """
    Run MCTS on the current state and return side_one's best action int.
    Used for BC data collection: MCTS plays side_one, we imitate it.
    """
    try:
        from poke_engine import monte_carlo_tree_search
        result = monte_carlo_tree_search(state, duration_ms)
        if result.side_one:
            best = max(result.side_one, key=lambda r: r.visits)
            return _mcts_move_str_to_action(best.move_choice, state.side_one)
    except Exception:
        pass
    legal = _build_legal_mask_from_state(state.side_one)
    return legal[0] if legal else 0


def _random_opponent(obs_dict: dict) -> int:
    legal = obs_dict.get("legal_actions", list(range(4)))
    return random.choice(legal)


def simple_heuristic_opponent(obs_dict: dict) -> int:
    """
    Pick the highest base-power move among legal actions.
    Falls back to switching to the healthiest pokemon.
    """
    legal = obs_dict.get("legal_actions", [0])
    move_actions = [a for a in legal if a < 4]
    switch_actions = [a for a in legal if a >= 4]

    if move_actions:
        active = obs_dict["side_one"].get("active", {})
        moves = active.get("moves", [])
        best_a, best_bp = move_actions[0], -1
        for a in move_actions:
            if a < len(moves):
                bp = moves[a].get("basePower", 0)
                if bp > best_bp:
                    best_bp = bp
                    best_a = a
        return best_a

    if switch_actions:
        # Switch to healthiest reserve mon
        reserve = obs_dict["side_one"].get("reserve", [])
        best_a, best_hp = switch_actions[0], -1.0
        for i, a in enumerate(switch_actions):
            if i < len(reserve):
                mon = reserve[i]
                hp_frac = mon.get("hp", 0) / max(mon.get("maxhp", 1), 1)
                if hp_frac > best_hp:
                    best_hp = hp_frac
                    best_a = a
        return best_a

    return legal[0]


# ---------------------------------------------------------------------------
# Gen 4 type effectiveness chart
# ---------------------------------------------------------------------------

_TYPE_CHART: dict[str, dict[str, float]] = {
    "Normal":   {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2,
                 "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0,
                 "Flying": 2, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5,
                 "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2,
                 "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5,
                 "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5,
                 "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0,
                 "Dark": 2, "Steel": 2},
    "Poison":   {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
                 "Ghost": 0.5, "Steel": 0},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2,
                 "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2,
                 "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5,
                 "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5,
                 "Flying": 2, "Bug": 2, "Steel": 0.5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5},
    "Dragon":   {"Dragon": 2, "Steel": 0.5},
    "Dark":     {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2,
                 "Rock": 2, "Steel": 0.5},
    "Fairy":    {},  # Gen 4 has no Fairy; included for safety
}


def _type_effectiveness(atk_type: str, def_types: list[str]) -> float:
    """Compute type effectiveness multiplier: atk_type vs defender's types."""
    mult = 1.0
    # _TYPE_CHART uses capitalized keys; env types are lowercase
    atk_key = atk_type.capitalize() if atk_type else "Normal"
    chart = _TYPE_CHART.get(atk_key, {})
    for dt in def_types:
        dt_key = dt.capitalize() if dt else "Normal"
        mult *= chart.get(dt_key, 1.0)
    return mult


def _estimate_damage(
    move: dict,
    attacker: dict,
    defender: dict,
    *,
    attacker_types: list[str] | None = None,
) -> float:
    """
    Rough damage estimate for choosing moves.  Not exact but captures the
    key factors: base-power, STAB, type effectiveness, category, accuracy.
    """
    bp = move.get("basePower", 0)
    move_id = move.get("id", "").lower()
    # Fixed-damage moves
    if move_id in ("seismictoss", "nightshade"):
        # Does 100 fixed damage at level 100 (immune if Normal vs Ghost or vice versa)
        def_types = defender.get("types", ["Normal"])
        move_type = move.get("type", "Normal")
        eff = _type_effectiveness(move_type, def_types)
        return 100.0 if eff > 0 else 0.0
    if bp <= 0:
        return 0.0

    atk_types = attacker_types or attacker.get("types", ["Normal"])
    move_type = move.get("type", "Normal")
    category = move.get("category", "physical").lower()
    def_types = defender.get("types", ["Normal"])
    accuracy = move.get("accuracy", 100)
    if isinstance(accuracy, bool) or accuracy > 100:
        accuracy = 100  # never-miss moves

    # STAB bonus
    stab = 1.5 if move_type in atk_types else 1.0

    # Type effectiveness
    eff = _type_effectiveness(move_type, def_types)

    # Very rough attack / defense ratio using base stats
    bst_a = attacker.get("base_stats", {})
    bst_d = defender.get("base_stats", {})
    if category == "physical":
        atk_stat = bst_a.get("attack", 80)
        def_stat = bst_d.get("defense", 80)
    else:
        atk_stat = bst_a.get("special-attack", 80)
        def_stat = bst_d.get("special-defense", 80)

    # Simplified damage formula: (bp * atk/def * STAB * eff * acc/100)
    score = bp * (atk_stat / max(def_stat, 1)) * stab * eff * (accuracy / 100.0)
    return score


def _score_status_move(move: dict, own_active: dict, opp_active: dict,
                       own_side: dict, turn: int) -> float:
    """
    Score a status move (bp=0). Returns a positive score if the move is
    strategically valuable, else 0.0.
    """
    move_id = move.get("id", "").lower()
    opp_status = opp_active.get("status")
    opp_hp_frac = opp_active.get("hp", 0) / max(opp_active.get("maxhp", 1), 1)
    own_hp_frac = own_active.get("hp", 0) / max(own_active.get("maxhp", 1), 1)

    # Stealth Rock: very high value on early turns if not already set
    hazards = own_side.get("hazards", {}) if isinstance(own_side, dict) else {}
    # Note: own_side hazards = opponent's side hazards (hazards on the side we're attacking)
    opp_hazards = own_side.get("hazards", {})

    # --- Entry hazards ---
    if move_id == "stealthrock":
        if not opp_hazards.get("stealth_rock"):
            return 250.0 if turn <= 3 else 150.0  # Very high early game
        return 0.0  # Already set
    if move_id == "spikes":
        layers = opp_hazards.get("spikes", 0)
        if layers < 3:
            return 180.0 if turn <= 4 else 100.0
        return 0.0
    if move_id == "toxicspikes":
        layers = opp_hazards.get("toxic_spikes", 0)
        if layers < 2:
            return 160.0 if turn <= 4 else 80.0
        return 0.0

    # --- Status-inflicting moves ---
    _STATUS_INFLICT = {
        "sleeppowder", "spore", "hypnosis", "lovelykiss", "grasswhistle",
        "sing", "darkvoid", "thunderwave", "stunspore", "glare",
        "toxic", "willowisp", "poisonpowder",
    }
    if move_id in _STATUS_INFLICT:
        # Don't use status moves on already-statused opponents
        if opp_status is not None and opp_status != "":
            return 0.0
        if move_id in ("sleeppowder", "spore", "hypnosis", "lovelykiss",
                        "grasswhistle", "sing", "darkvoid"):
            return 300.0
        if move_id in ("thunderwave", "stunspore", "glare"):
            opp_speed = opp_active.get("base_stats", {}).get("speed", 80)
            return 200.0 if opp_speed > 90 else 120.0
        if move_id == "toxic":
            return 180.0 if opp_hp_frac > 0.7 else 100.0
        if move_id in ("willowisp", "poisonpowder"):
            opp_atk = opp_active.get("base_stats", {}).get("attack", 80)
            return 180.0 if opp_atk > 90 else 100.0
        return 100.0  # Other status-inflicting moves

    # --- Setup moves ---
    own_boosts = own_active.get("boosts", {})
    if move_id in ("swordsdance", "dragondance", "nastyplot", "calmmind",
                    "bulkup", "agility", "rockpolish", "curse", "quiverdance"):
        # Only set up if we haven't already boosted much and have decent HP
        atk_boost = own_boosts.get("atk", 0) + own_boosts.get("spa", 0)
        if atk_boost < 2 and own_hp_frac > 0.6:
            return 200.0 if turn <= 5 else 120.0
        return 0.0

    # --- Recovery moves ---
    if move_id in ("recover", "softboiled", "roost", "slackoff", "moonlight",
                    "morningsun", "synthesis", "wish", "milkdrink"):
        if own_hp_frac < 0.5:
            return 150.0  # Heal when low
        return 0.0

    # --- Protect/Substitute ---
    if move_id == "substitute":
        if own_hp_frac > 0.3:
            return 80.0
        return 0.0
    if move_id == "protect":
        return 30.0  # Low value in singles, but not zero

    # --- Screens ---
    if move_id in ("reflect", "lightscreen"):
        return 120.0 if turn <= 3 else 60.0

    # --- Taunt (prevent opponent setup) ---
    if move_id == "taunt":
        return 100.0

    # Default: small positive so status moves aren't completely ignored
    return 20.0


def smart_heuristic_opponent(obs_dict: dict) -> int:
    """
    Type-aware heuristic that considers STAB, type effectiveness, accuracy,
    status moves, setup moves, and switches to better matchups.
    Designed as a strong BC teacher.
    """
    legal = obs_dict.get("legal_actions", [0])
    move_actions = [a for a in legal if a < 4]
    switch_actions = [a for a in legal if a >= 4]

    own = obs_dict.get("side_one", {})
    opp = obs_dict.get("side_two", {})
    active = own.get("active", {})
    opp_active = opp.get("active", {})
    turn = obs_dict.get("turn", 1)

    if not active or not opp_active:
        return legal[0]

    # --- Score each legal move (damaging + status) ---
    best_move_a = None
    best_move_score = -1.0
    for a in move_actions:
        moves = active.get("moves", [])
        if a >= len(moves):
            continue
        m = moves[a]
        bp = m.get("basePower", 0)
        if bp > 0:
            score = _estimate_damage(m, active, opp_active)
        else:
            # Status move — use strategic scoring
            # Pass opponent's side for hazard checking
            score = _score_status_move(m, active, opp_active, opp, turn)
        if score > best_move_score:
            best_move_score = score
            best_move_a = a

    # --- Score switching options ---
    best_switch_a = None
    best_switch_score = -1.0
    reserve = own.get("reserve", [])
    for i, a in enumerate(switch_actions):
        if i >= len(reserve):
            continue
        mon = reserve[i]
        if mon.get("is_fainted") or mon.get("hp", 0) <= 0:
            continue
        hp_frac = mon.get("hp", 0) / max(mon.get("maxhp", 1), 1)
        if hp_frac < 0.15:
            continue

        mon_moves = mon.get("moves", [])
        mon_types = mon.get("types", ["Normal"])
        best_dmg = 0.0
        for m in mon_moves:
            dmg = _estimate_damage(m, mon, opp_active, attacker_types=mon_types)
            best_dmg = max(best_dmg, dmg)

        opp_moves = opp_active.get("moves", [])
        worst_incoming = 0.0
        for m in opp_moves:
            incoming = _estimate_damage(m, opp_active, mon)
            worst_incoming = max(worst_incoming, incoming)

        switch_score = (best_dmg * 1.2 - worst_incoming * 0.3) * hp_frac
        if switch_score > best_switch_score:
            best_switch_score = switch_score
            best_switch_a = a

    # --- Decision: move vs switch ---
    should_switch = False
    if best_switch_a is not None and best_move_a is not None:
        if best_move_score <= 0:
            should_switch = True
        elif best_switch_score > best_move_score * 2.0:
            should_switch = True

    if should_switch and best_switch_a is not None:
        return best_switch_a
    if best_move_a is not None:
        return best_move_a
    if switch_actions:
        return switch_actions[0]
    return legal[0]
