"""
Battle state → tensor encoder.

Produces a (15, raw_dim) token sequence:
  token 0:     FIELD (weather, terrain, hazards, screens, turn)
  tokens 1–6:  OWN_TEAM[0–5]  (slot 0 = active)
  tokens 7–12: OPP_TEAM[0–5]  (slot 0 = active; unrevealed = UNKNOWN/EMPTY)
  token 13:    ACTOR query (all zeros — filled by learned embedding in model)
  token 14:    CRITIC query (all zeros — filled by learned embedding in model)

The raw per-pokemon token vector is assembled here and projected to d_model inside
the Transformer. All integer IDs (species, moves, ability, item) are returned as
separate index arrays so the model can look them up in embedding tables.

Performance: encode() uses pre-allocated buffers and direct index writes to avoid
allocating hundreds of small numpy arrays per call (3.2ms → <0.5ms).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TYPES = 18
N_STAT_BOOST_LEVELS = 13   # -6 … +6
N_STATUS = 7               # healthy, BRN, PSN, TOX, SLP, FRZ, PAR
N_VOLATILE = 27            # see VOLATILE_STATUS_LIST below
N_MOVES_PER_MON = 4
N_TEAM_SLOTS = 6

# UNKNOWN / EMPTY sentinel indices (must match vocab built by build_vocab.py)
UNKNOWN_SPECIES_IDX = 0
UNKNOWN_MOVE_IDX = 0
UNKNOWN_ABILITY_IDX = 0
UNKNOWN_ITEM_IDX = 0
EMPTY_MOVE_IDX = 1   # slot genuinely has no move (e.g. <4 moves)

TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]
TYPE_TO_IDX = {t.lower(): i for i, t in enumerate(TYPES)}

STATUS_TO_IDX = {
    None: 0, "": 0,
    "brn": 1, "psn": 2, "tox": 3,
    "slp": 4, "frz": 5, "par": 6,
}

VOLATILE_STATUS_LIST = [
    "confusion", "infatuation", "leechseed", "curse",
    "aquaring", "ingrain", "taunt", "encore", "flinch",
    "embargo", "healblock", "magnetrise", "partiallytrapped",
    "perishsong", "powertrick", "substitute", "yawn",
    "focusenergy", "charge", "stockpile",
    # Added: Gen 4+ volatiles that were missing
    "torment", "nightmare", "imprison",
    "mustrecharge",     # Hyper Beam, Giga Impact, etc.
    "twoturnmove",      # Fly/Dig/Dive/Bounce semi-invulnerable
    "destinybond",
    "grudge",
]
VOLATILE_TO_IDX = {v: i for i, v in enumerate(VOLATILE_STATUS_LIST)}

WEATHER_TO_IDX = {
    None: 0, "": 0,
    "sunnyday": 1, "raindance": 2, "sandstorm": 3, "hail": 4, "snowscape": 4,
}
TERRAIN_TO_IDX = {
    None: 0, "": 0,
    "electricterrain": 1, "grassyterrain": 2, "mistyterrain": 3, "psychicterrain": 4,
}
# Note: Gen 4 has no terrain — these always map to 0 during training

MOVE_CATEGORY_TO_IDX = {"physical": 0, "special": 1, "status": 2}

# Map priority integer → one-hot index  (-3 → 0, … +4 → 7)
PRIORITY_MIN = -3

N_TOKENS = 15   # 1 field + 6 own + 6 opp + 1 actor + 1 critic

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Vocab loader
# ---------------------------------------------------------------------------

class Vocab:
    """Lazy-loaded vocabulary mappings from data/*.json."""

    _instance: Optional["Vocab"] = None

    def __init__(self):
        self.species: dict[str, int] = self._load("species_index.json")
        self.moves: dict[str, int] = self._load("move_index.json")
        self.abilities: dict[str, int] = self._load("ability_index.json")
        self.items: dict[str, int] = self._load("item_index.json")

    @staticmethod
    def _load(filename: str) -> dict[str, int]:
        path = DATA_DIR / filename
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    @classmethod
    def get(cls) -> "Vocab":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def species_idx(self, name: str) -> int:
        return self.species.get(str(name).lower(), UNKNOWN_SPECIES_IDX)

    def move_idx(self, name: Optional[str]) -> int:
        if name is None or name == "":
            return EMPTY_MOVE_IDX
        return self.moves.get(str(name).lower(), UNKNOWN_MOVE_IDX)

    def ability_idx(self, name: Optional[str]) -> int:
        return self.abilities.get(str(name).lower() if name else "", UNKNOWN_ABILITY_IDX)

    def item_idx(self, name: Optional[str]) -> int:
        return self.items.get(str(name).lower() if name else "", UNKNOWN_ITEM_IDX)


# ---------------------------------------------------------------------------
# Observation spec
# ---------------------------------------------------------------------------

@dataclass
class MoveFeatures:
    """Float features for one move slot (excluding the embedding index)."""
    base_power_bin: np.ndarray   # (8,)  one-hot
    accuracy_bin: np.ndarray     # (6,)  one-hot  [0, 50, 70, 80, 90, 100+]
    type_onehot: np.ndarray      # (18,) one-hot
    category_onehot: np.ndarray  # (3,)  one-hot
    priority_onehot: np.ndarray  # (8,)  one-hot  (-3…+4)
    pp_fraction: float           # scalar [0, 1]
    is_known: float              # 1.0 if move is known, 0.0 if hidden

    @property
    def dim(self) -> int:
        return 8 + 6 + 18 + 3 + 8 + 1 + 1  # = 45

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.base_power_bin,
            self.accuracy_bin,
            self.type_onehot,
            self.category_onehot,
            self.priority_onehot,
            [self.pp_fraction],
            [self.is_known],
        ]).astype(np.float32)


@dataclass
class PokemonToken:
    """
    All features for one Pokemon slot.

    Integer IDs (species, moves, ability, item) are kept separate for embedding
    table lookup. Float features are concatenated directly.
    """
    # Embedding indices
    species_idx: int
    move_idxs: list[int]         # length 4, EMPTY_MOVE_IDX for missing moves
    ability_idx: int
    item_idx: int

    # Float features
    hp_fraction: float
    hp_bin: np.ndarray           # (10,) one-hot
    base_stats: np.ndarray       # (6,)  HP/Atk/Def/SpA/SpD/Spe, each /255
    stat_boosts: np.ndarray      # (91,) 7 stats × 13 one-hot levels
    status_onehot: np.ndarray    # (7,)
    volatile_multihot: np.ndarray  # (20,)
    type1_onehot: np.ndarray     # (18,)
    type2_onehot: np.ndarray     # (18,)  all-zero if single type
    is_fainted: float
    is_active: float
    slot_onehot: np.ndarray      # (6,)
    is_own: float
    move_features: list[MoveFeatures]  # 4 MoveFeatures


FLOAT_DIM_PER_POKEMON = (
    1    # hp_fraction
    + 10  # hp_bin
    + 6   # base_stats
    + 91  # stat_boosts
    + 7   # status
    + N_VOLATILE  # volatile multi-hot (27)
    + 18  # type1
    + 18  # type2
    + 1   # is_fainted
    + 1   # is_active
    + 6   # slot
    + 1   # is_own
    + 4 * 45  # move features (4 × 45: bp_bin(8) + acc_bin(6) + type(18) + cat(3) + pri(8) + pp(1) + known(1))
    # --- per-pokemon extended features ---
    + 4   # sleep_turns bin [0, 1, 2, 3+]
    + 3   # rest_turns bin [0, 1, 2]
    + 1   # substitute_health (fraction of maxhp)
    + 1   # force_trapped flag
    + 4   # move_disabled flags (one per move slot)
    + 4   # confusion_turns bin [0, 1, 2, 3+]
    + 1   # taunt flag
    + 1   # encore flag
    + 1   # yawn flag (critical: sleep NEXT turn!)
    + 1   # level_normalized (level / 100.0)
    + 4   # perish_count_bin [0, 1, 2, 3] (0 = no perish song)
    + 1   # protect_count_normalized (consecutive protect uses / 4.0)
    + 1   # locked_move flag (Choice lock / Outrage etc.)
)
# = 1+10+6+91+7+27+18+18+1+1+6+1+180+4+3+1+1+4+4+1+1+1+1+4+1+1 = 394


FIELD_DIM = (
    5    # weather one-hot
    + 8  # weather turns bin
    # terrain removed — Gen 4 has no terrain mechanics
    + 5  # pseudo-weather multi-hot (trick room, gravity, wonder room, etc.)
    + 4  # trick room turns
    + 7  # hazards own (sr=1, spikes=3, tspikes=2, web=1)
    + 7  # hazards opp
    + 6  # screens own (light_screen flag+turns, reflect flag+turns) — aurora veil removed (Gen 7+)
    + 6  # screens opp
    + 10  # turn number bin
    + 2   # total fainted (own / opp) normalized
    # --- side-level fields ---
    + 5  # toxic_count own: 5-bin [0, 1, 2, 3-4, 5+]
    + 5  # toxic_count opp: 5-bin
    + 2  # tailwind own + opp flags
    + 2  # wish own + opp flags
    + 2  # safeguard own + opp flags
    + 2  # mist own + opp flags
    + 2  # lucky_chant own + opp flags
    + 4  # gravity turns bin
)
# FIELD_DIM = 76 + 2+2+4 = 84


# ---------------------------------------------------------------------------
# Precomputed float_feats offsets (derived programmatically for safety)
# ---------------------------------------------------------------------------

_MON_OFFSET_SPEC = [
    ("hp_frac",       1),
    ("hp_bin",       10),
    ("base_stats",    6),
    ("boosts",       91),
    ("status",    N_STATUS),
    ("volatile",  N_VOLATILE),
    ("type1",     N_TYPES),
    ("type2",     N_TYPES),
    ("is_fainted",    1),
    ("is_active",     1),
    ("slot",      N_TEAM_SLOTS),
    ("is_own",        1),
    ("moves",     N_MOVES_PER_MON * 45),
    ("sleep_bin",     4),
    ("rest_bin",      3),
    ("sub_frac",      1),
    ("force_trapped", 1),
    ("move_disabled", N_MOVES_PER_MON),
    ("confusion_bin", 4),
    ("taunt",         1),
    ("encore",        1),
    ("yawn",          1),
    ("level",         1),
    ("perish_bin",    4),
    ("protect",       1),
    ("locked_move",   1),
]

_OFF = {}
_pos = 0
for _name, _size in _MON_OFFSET_SPEC:
    _OFF[_name] = _pos
    _pos += _size
assert _pos == FLOAT_DIM_PER_POKEMON, f"Mon offset mismatch: {_pos} != {FLOAT_DIM_PER_POKEMON}"

# Offset to slot one-hot within float_feats (used by _pad_team)
_SLOT_ONEHOT_OFFSET = _OFF["slot"]

# Per-move sub-offsets within the 45-dim move feature block
_MOVE_SUB_SPEC = [
    ("bp_bin",    8),
    ("acc_bin",   6),
    ("type",     18),
    ("cat",       3),
    ("pri",       8),
    ("pp",        1),
    ("known",     1),
]
_MOVE_OFF = {}
_mpos = 0
for _name, _size in _MOVE_SUB_SPEC:
    _MOVE_OFF[_name] = _mpos
    _mpos += _size
assert _mpos == 45, f"Move sub-offset mismatch: {_mpos} != 45"

# Field encoding offsets (84-dim)
_FIELD_SPEC = [
    ("weather",       5),
    ("weather_turns", 8),
    ("pseudo",        5),
    ("tr_turns",      4),
    ("hazards_own",   7),
    ("hazards_opp",   7),
    ("screens_own",   6),
    ("screens_opp",   6),
    ("turn_bin",     10),
    ("fainted",       2),
    ("toxic_own",     5),
    ("toxic_opp",     5),
    ("tailwind",      2),
    ("wish",          2),
    ("safeguard",     2),
    ("mist",          2),
    ("lucky_chant",   2),
    ("gravity_turns", 4),
]
_FIELD_OFF = {}
_fpos = 0
for _name, _size in _FIELD_SPEC:
    _FIELD_OFF[_name] = _fpos
    _fpos += _size
assert _fpos == FIELD_DIM, f"Field offset mismatch: {_fpos} != {FIELD_DIM}"

# Precomputed bin thresholds as tuples (faster than list for small searches)
_HP_BINS = (0.0, 0.1, 0.2, 0.33, 0.5, 0.66, 0.75, 0.875, 1.0)
_BP_THRESHOLDS = (0, 1, 41, 61, 81, 101, 121, 151)
_ACC_THRESHOLDS = (0, 50, 70, 80, 90, 100)
_TURN_THRESHOLDS = (1, 2, 4, 6, 9, 13, 18, 25, 35)


# ---------------------------------------------------------------------------
# Fast bin-index functions (return int, no array allocation)
# ---------------------------------------------------------------------------

def _bin_hp_idx(hp_frac: float) -> int:
    for i, b in enumerate(_HP_BINS):
        if hp_frac <= b:
            return i
    return 9


def _bin_bp_idx(bp: int) -> int:
    for i in range(7, -1, -1):
        if bp >= _BP_THRESHOLDS[i]:
            return i
    return 0


def _bin_acc_idx(acc: int) -> int:
    if acc <= 0 or acc > 100:
        return 0
    for i in range(5, -1, -1):
        if acc >= _ACC_THRESHOLDS[i]:
            return i
    return 0


def _bin_turn_idx(turn: int) -> int:
    for i in range(8, -1, -1):
        if turn >= _TURN_THRESHOLDS[i]:
            return i + 1
    return 0


def _toxic_bin_idx(count: int) -> int:
    if count <= 0:
        return 0
    if count <= 2:
        return count
    if count <= 4:
        return 3
    return 4


# ---------------------------------------------------------------------------
# Helper functions (legacy, used by old encode path and tests)
# ---------------------------------------------------------------------------

def _one_hot(idx: int, n: int) -> np.ndarray:
    arr = np.zeros(n, dtype=np.float32)
    if 0 <= idx < n:
        arr[idx] = 1.0
    return arr


def _bin_hp(hp_fraction: float) -> np.ndarray:
    """10-bin one-hot for HP fraction."""
    bins = [0.0, 0.1, 0.2, 0.33, 0.5, 0.66, 0.75, 0.875, 1.0]
    idx = np.searchsorted(bins, hp_fraction, side="right") - 1
    idx = int(np.clip(idx, 0, 9))
    return _one_hot(idx, 10)


def _bin_base_power(bp: int) -> np.ndarray:
    """8-bin one-hot for move base power."""
    thresholds = [0, 1, 41, 61, 81, 101, 121, 151]
    idx = np.searchsorted(thresholds, bp, side="right") - 1
    idx = int(np.clip(idx, 0, 7))
    return _one_hot(idx, 8)


def _bin_accuracy(acc: int) -> np.ndarray:
    """6-bin one-hot for move accuracy: [0(status/never-miss), 50, 70, 80, 90, 100+]."""
    if acc <= 0 or acc > 100:
        return _one_hot(0, 6)  # bypass accuracy check / never miss / status
    thresholds = [0, 50, 70, 80, 90, 100]
    idx = np.searchsorted(thresholds, acc, side="right") - 1
    idx = int(np.clip(idx, 0, 5))
    return _one_hot(idx, 6)


def _bin_turns(turns: int, max_bin: int = 8) -> np.ndarray:
    """One-hot encode remaining turns (0 = expired/none, 1…max_bin-1 = turns)."""
    idx = int(np.clip(turns, 0, max_bin - 1))
    return _one_hot(idx, max_bin)


def _bin_turn_number(turn: int) -> np.ndarray:
    """10-bin one-hot for turn number."""
    thresholds = [1, 2, 4, 6, 9, 13, 18, 25, 35]
    idx = np.searchsorted(thresholds, turn, side="right")
    idx = int(np.clip(idx, 0, 9))
    return _one_hot(idx, 10)


def _encode_stat_boosts(boosts: dict[str, int]) -> np.ndarray:
    """
    Encode 7 stat boost values (each -6…+6) as 7 × 13 one-hot = 91 features.
    Stats order: atk, def, spa, spd, spe, accuracy, evasion
    """
    stat_keys = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
    parts = []
    for key in stat_keys:
        val = boosts.get(key, 0) if boosts else 0
        idx = int(val) + 6   # shift -6…+6 → 0…12
        parts.append(_one_hot(idx, 13))
    return np.concatenate(parts)


def _encode_volatile_status(volatile: set[str]) -> np.ndarray:
    arr = np.zeros(N_VOLATILE, dtype=np.float32)
    for v in (volatile or set()):
        idx = VOLATILE_TO_IDX.get(v.lower())
        if idx is not None:
            arr[idx] = 1.0
    return arr


def _encode_move(move_data: Optional[dict], is_known: bool, vocab: Vocab) -> tuple[int, MoveFeatures]:
    """
    Encode one move slot.
    Returns (move_idx, MoveFeatures).
    move_data: dict with keys id, basePower, type, category, priority, pp, maxpp
               or None if unknown/empty.
    """
    if move_data is None or not is_known:
        move_idx = UNKNOWN_MOVE_IDX if is_known is False else EMPTY_MOVE_IDX
        feat = MoveFeatures(
            base_power_bin=np.zeros(8, dtype=np.float32),
            accuracy_bin=np.zeros(6, dtype=np.float32),
            type_onehot=np.zeros(N_TYPES, dtype=np.float32),
            category_onehot=np.zeros(3, dtype=np.float32),
            priority_onehot=np.zeros(8, dtype=np.float32),
            pp_fraction=1.0,
            is_known=0.0,
        )
        return move_idx, feat

    move_idx = vocab.move_idx(move_data.get("id", ""))
    bp = move_data.get("basePower", 0)
    accuracy = move_data.get("accuracy", 100)
    move_type = move_data.get("type", "Normal").lower()
    category = move_data.get("category", "physical").lower()
    priority = move_data.get("priority", 0)
    pp = move_data.get("pp", 0)
    max_pp = move_data.get("maxpp", 1)

    priority_idx = int(np.clip(priority - PRIORITY_MIN, 0, 7))
    type_idx = TYPE_TO_IDX.get(move_type, 0)

    feat = MoveFeatures(
        base_power_bin=_bin_base_power(bp),
        accuracy_bin=_bin_accuracy(accuracy),
        type_onehot=_one_hot(type_idx, N_TYPES),
        category_onehot=_one_hot(MOVE_CATEGORY_TO_IDX.get(category, 0), 3),
        priority_onehot=_one_hot(priority_idx, 8),
        pp_fraction=float(pp) / max(max_pp, 1),
        is_known=1.0,
    )
    return move_idx, feat


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

_N_INT_IDS = 8
_STAT_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_BST_KEYS = ("hp", "attack", "defense", "special-attack", "special-defense", "speed")


class ObsBuilder:
    """
    Converts a battle state dict (as produced by poke_engine_env) into
    the token arrays expected by PokeTransformer.

    Uses pre-allocated buffers and direct index writes for performance.
    """

    def __init__(self):
        self.vocab = Vocab.get()
        # Pre-allocate output buffers (reused across calls, copies returned)
        self._int_buf = np.zeros((N_TOKENS, _N_INT_IDS), dtype=np.int64)
        self._float_buf = np.zeros((N_TOKENS, FLOAT_DIM_PER_POKEMON), dtype=np.float32)
        self._legal_buf = np.zeros(10, dtype=np.float32)

    def encode(self, state: dict) -> dict:
        """
        Returns a dict with:
          "int_ids"    : np.ndarray (15, 8)
          "float_feats": np.ndarray (15, 394)
          "legal_mask" : np.ndarray (10,)
        """
        ib = self._int_buf
        fb = self._float_buf

        # Zero buffers (tokens 13, 14 = query tokens stay zero)
        ib[:] = 0
        fb[:] = 0.0

        # Token 0: field
        self._fill_field(state, fb[0])

        # Tokens 1-6: own team, Tokens 7-12: opp team
        side_one = state["side_one"]
        side_two = state["side_two"]
        self._fill_team(side_one, is_own=True, ib=ib, fb=fb, token_offset=1)
        self._fill_team(side_two, is_own=False, ib=ib, fb=fb, token_offset=7)

        # Legal mask
        self._fill_legal_mask(state)

        return {
            "int_ids": ib.copy(),
            "float_feats": fb.copy(),
            "legal_mask": self._legal_buf.copy(),
        }

    # ------------------------------------------------------------------
    # Field token (writes into fb[0, :FIELD_DIM])
    # ------------------------------------------------------------------

    def _fill_field(self, state: dict, buf: np.ndarray):
        weather = state.get("weather", "") or ""
        weather_turns = state.get("weather_turns", 0) or 0
        trick_room = state.get("trick_room", False)
        trick_room_turns = state.get("trick_room_turns", 0) or 0
        turn = state.get("turn", 1) or 1

        side_one = state["side_one"]
        side_two = state["side_two"]

        # Weather one-hot (5)
        o = _FIELD_OFF["weather"]
        w_idx = WEATHER_TO_IDX.get(weather.lower(), 0)
        if 0 <= w_idx < 5:
            buf[o + w_idx] = 1.0

        # Weather turns bin (8)
        o = _FIELD_OFF["weather_turns"]
        buf[o + min(weather_turns, 7)] = 1.0

        # Pseudo-weather (5)
        o = _FIELD_OFF["pseudo"]
        if trick_room:
            buf[o] = 1.0
        if state.get("gravity"):
            buf[o + 1] = 1.0
        if state.get("wonder_room"):
            buf[o + 2] = 1.0

        # Trick room turns (4)
        o = _FIELD_OFF["tr_turns"]
        buf[o + min(trick_room_turns, 3)] = 1.0

        # Hazards
        self._fill_hazards(side_one.get("hazards", {}), buf, _FIELD_OFF["hazards_own"])
        self._fill_hazards(side_two.get("hazards", {}), buf, _FIELD_OFF["hazards_opp"])

        # Screens
        self._fill_screens(side_one.get("screens", {}), buf, _FIELD_OFF["screens_own"])
        self._fill_screens(side_two.get("screens", {}), buf, _FIELD_OFF["screens_opp"])

        # Turn number bin (10)
        o = _FIELD_OFF["turn_bin"]
        buf[o + _bin_turn_idx(turn)] = 1.0

        # Fainted counts (2)
        o = _FIELD_OFF["fainted"]
        own_fainted = sum(1 for m in side_one.get("reserve", []) if m.get("is_fainted"))
        opp_fainted = sum(1 for m in side_two.get("reserve", []) if m.get("is_fainted"))
        if side_one.get("active", {}).get("is_fainted"):
            own_fainted += 1
        if side_two.get("active", {}).get("is_fainted"):
            opp_fainted += 1
        buf[o] = own_fainted / 6.0
        buf[o + 1] = opp_fainted / 6.0

        # Toxic count bins (5 each)
        o = _FIELD_OFF["toxic_own"]
        buf[o + _toxic_bin_idx(side_one.get("toxic_count", 0))] = 1.0
        o = _FIELD_OFF["toxic_opp"]
        buf[o + _toxic_bin_idx(side_two.get("toxic_count", 0))] = 1.0

        # Side flags (2 each)
        o = _FIELD_OFF["tailwind"]
        buf[o] = float(side_one.get("tailwind", 0) > 0)
        buf[o + 1] = float(side_two.get("tailwind", 0) > 0)

        o = _FIELD_OFF["wish"]
        buf[o] = float(side_one.get("wish", (0, 0))[0] > 0)
        buf[o + 1] = float(side_two.get("wish", (0, 0))[0] > 0)

        o = _FIELD_OFF["safeguard"]
        buf[o] = float(side_one.get("safeguard", 0) > 0)
        buf[o + 1] = float(side_two.get("safeguard", 0) > 0)

        o = _FIELD_OFF["mist"]
        buf[o] = float(side_one.get("mist", 0) > 0)
        buf[o + 1] = float(side_two.get("mist", 0) > 0)

        o = _FIELD_OFF["lucky_chant"]
        buf[o] = float(side_one.get("lucky_chant", 0) > 0)
        buf[o + 1] = float(side_two.get("lucky_chant", 0) > 0)

        # Gravity turns (4)
        o = _FIELD_OFF["gravity_turns"]
        gravity_turns = state.get("gravity_turns", 0) or 0
        buf[o + min(gravity_turns, 3)] = 1.0

    def _fill_hazards(self, hazards: dict, buf: np.ndarray, offset: int):
        """7-dim: stealth_rock(1) + spikes(3) + tspikes(2) + web(1)."""
        buf[offset] = float(hazards.get("stealth_rock", False))
        spikes = int(hazards.get("spikes", 0))
        if spikes >= 1: buf[offset + 1] = 1.0
        if spikes >= 2: buf[offset + 2] = 1.0
        if spikes >= 3: buf[offset + 3] = 1.0
        tspikes = int(hazards.get("toxic_spikes", 0))
        if tspikes >= 1: buf[offset + 4] = 1.0
        if tspikes >= 2: buf[offset + 5] = 1.0
        buf[offset + 6] = float(hazards.get("sticky_web", False))

    def _fill_screens(self, screens: dict, buf: np.ndarray, offset: int):
        """6-dim: light_screen (flag+turns) + reflect (flag+turns) + 2 padding."""
        ls = int(screens.get("light_screen", 0))
        ref = int(screens.get("reflect", 0))
        buf[offset] = float(ls > 0)
        buf[offset + 1] = float(ls) / 5.0
        buf[offset + 2] = float(ref > 0)
        buf[offset + 3] = float(ref) / 5.0
        # offset+4, offset+5 = 0.0 (padding, already zeroed)

    # ------------------------------------------------------------------
    # Team tokens
    # ------------------------------------------------------------------

    def _fill_team(self, side: dict, is_own: bool,
                   ib: np.ndarray, fb: np.ndarray, token_offset: int):
        """Fill tokens [token_offset : token_offset+6] for one side."""
        slot = 0
        active = side.get("active")
        if active:
            # Inject side-level state into active mon
            active_aug = dict(active)
            active_aug["substitute_health"] = side.get("substitute_health", 0)
            active_aug["force_trapped"] = side.get("force_trapped", False)
            active_aug["volatile_durations"] = side.get("volatile_durations", {})
            active_aug["last_used_move"] = side.get("last_used_move", "none")
            active_aug["protect_count"] = side.get("protect_count", 0)
            active_aug["locked_move"] = side.get("locked_move", False)
            active_aug["perish_count"] = side.get("perish_count", 0)
            self._fill_mon(active_aug, slot=0, is_own=is_own, is_active=True,
                           int_row=ib[token_offset], float_row=fb[token_offset])
            slot = 1

        for mon in side.get("reserve", []):
            if slot >= N_TEAM_SLOTS:
                break
            self._fill_mon(mon, slot=slot, is_own=is_own, is_active=False,
                           int_row=ib[token_offset + slot], float_row=fb[token_offset + slot])
            slot += 1

        # Pad remaining slots (buffers already zeroed; just set slot + is_own)
        while slot < N_TEAM_SLOTS:
            # int_ids stay 0 (= UNKNOWN for all)
            fb[token_offset + slot, _OFF["slot"] + slot] = 1.0
            fb[token_offset + slot, _OFF["is_own"]] = float(is_own)
            slot += 1

    def _fill_mon(self, mon: dict, slot: int, is_own: bool, is_active: bool,
                  int_row: np.ndarray, float_row: np.ndarray):
        """Fill one pokemon's int_ids and float_feats rows in-place."""
        vocab = self.vocab

        # --- Integer IDs ---
        int_row[0] = vocab.species_idx(mon.get("species", ""))

        moves = mon.get("moves", [])
        for i in range(N_MOVES_PER_MON):
            if i < len(moves):
                m = moves[i]
                is_known = m.get("is_known", is_own)
                if is_known:
                    int_row[1 + i] = vocab.move_idx(m.get("id", ""))
                else:
                    int_row[1 + i] = UNKNOWN_MOVE_IDX
            else:
                int_row[1 + i] = EMPTY_MOVE_IDX

        int_row[5] = vocab.ability_idx(mon.get("ability"))
        int_row[6] = vocab.item_idx(mon.get("item"))

        last_move = mon.get("last_used_move", "none")
        int_row[7] = vocab.move_idx(last_move if last_move and last_move != "none" else None)

        # --- Float features (direct index writes) ---
        hp = float(mon.get("hp", 0))
        max_hp = float(mon.get("maxhp", 1))
        hp_frac = hp / max(max_hp, 1)

        # hp_fraction (1)
        float_row[_OFF["hp_frac"]] = hp_frac

        # hp_bin (10)
        float_row[_OFF["hp_bin"] + _bin_hp_idx(hp_frac)] = 1.0

        # base_stats (6)
        o = _OFF["base_stats"]
        bst = mon.get("base_stats", {})
        for i, k in enumerate(_BST_KEYS):
            float_row[o + i] = bst.get(k, 0) / 255.0

        # stat_boosts (91 = 7 stats × 13 levels)
        o = _OFF["boosts"]
        boosts = mon.get("boosts", {})
        for i, key in enumerate(_STAT_KEYS):
            val = boosts.get(key, 0) if boosts else 0
            idx = int(val) + 6  # shift -6…+6 → 0…12
            if 0 <= idx < 13:
                float_row[o + i * 13 + idx] = 1.0

        # status (7)
        o = _OFF["status"]
        status = (mon.get("status") or "").lower()
        float_row[o + STATUS_TO_IDX.get(status, 0)] = 1.0

        # volatile (27)
        o = _OFF["volatile"]
        for v in (mon.get("volatile_statuses", []) or []):
            idx = VOLATILE_TO_IDX.get(v.lower() if isinstance(v, str) else v)
            if idx is not None:
                float_row[o + idx] = 1.0

        # types (18 + 18)
        types = mon.get("types", ["Normal"])
        o = _OFF["type1"]
        float_row[o + TYPE_TO_IDX.get((types[0] if types else "Normal").lower(), 0)] = 1.0
        if len(types) > 1:
            o = _OFF["type2"]
            float_row[o + TYPE_TO_IDX.get(types[1].lower(), 0)] = 1.0

        # is_fainted (1), is_active (1)
        float_row[_OFF["is_fainted"]] = float(mon.get("is_fainted", False) or hp <= 0)
        float_row[_OFF["is_active"]] = float(is_active)

        # slot (6)
        float_row[_OFF["slot"] + slot] = 1.0

        # is_own (1)
        float_row[_OFF["is_own"]] = float(is_own)

        # moves (4 × 45)
        moves_base = _OFF["moves"]
        for i in range(N_MOVES_PER_MON):
            move_off = moves_base + i * 45
            if i < len(moves):
                m = moves[i]
                is_known = m.get("is_known", is_own)
                if is_known:
                    self._fill_move(m, float_row, move_off)
                else:
                    # Unknown move: pp_fraction=1.0, is_known=0.0, rest stays zero
                    float_row[move_off + _MOVE_OFF["pp"]] = 1.0
            else:
                # Empty move slot: pp_fraction=1.0, is_known=0.0
                float_row[move_off + _MOVE_OFF["pp"]] = 1.0

        # --- Extended per-pokemon features ---

        # sleep_turns bin (4)
        o = _OFF["sleep_bin"]
        float_row[o + min(int(mon.get("sleep_turns", 0)), 3)] = 1.0

        # rest_turns bin (3)
        o = _OFF["rest_bin"]
        float_row[o + min(int(mon.get("rest_turns", 0)), 2)] = 1.0

        # substitute health fraction (1)
        sub_hp = float(mon.get("substitute_health", 0))
        float_row[_OFF["sub_frac"]] = sub_hp / max(max_hp, 1) if sub_hp > 0 else 0.0

        # force_trapped (1)
        float_row[_OFF["force_trapped"]] = float(mon.get("force_trapped", False))

        # move_disabled (4)
        o = _OFF["move_disabled"]
        for i in range(min(N_MOVES_PER_MON, len(moves))):
            if moves[i].get("disabled", False):
                float_row[o + i] = 1.0

        # Volatile durations
        vd = mon.get("volatile_durations", {})

        # confusion_bin (4)
        o = _OFF["confusion_bin"]
        float_row[o + min(int(vd.get("confusion", 0)), 3)] = 1.0

        # taunt, encore, yawn flags
        float_row[_OFF["taunt"]] = float(int(vd.get("taunt", 0)) > 0)
        float_row[_OFF["encore"]] = float(int(vd.get("encore", 0)) > 0)
        float_row[_OFF["yawn"]] = float(int(vd.get("yawn", 0)) > 0)

        # level_normalized (1)
        float_row[_OFF["level"]] = float(mon.get("level", 100)) / 100.0

        # perish_count bin (4)
        o = _OFF["perish_bin"]
        float_row[o + min(int(mon.get("perish_count", 0)), 3)] = 1.0

        # protect_count_normalized (1)
        float_row[_OFF["protect"]] = min(int(mon.get("protect_count", 0)), 4) / 4.0

        # locked_move (1)
        float_row[_OFF["locked_move"]] = float(mon.get("locked_move", False))

    def _fill_move(self, move_data: dict, buf: np.ndarray, offset: int):
        """Fill 45-dim move features into buf[offset:offset+45]."""
        # bp_bin (8)
        bp = move_data.get("basePower", 0)
        buf[offset + _MOVE_OFF["bp_bin"] + _bin_bp_idx(bp)] = 1.0

        # acc_bin (6)
        acc = move_data.get("accuracy", 100)
        buf[offset + _MOVE_OFF["acc_bin"] + _bin_acc_idx(acc)] = 1.0

        # type (18)
        move_type = move_data.get("type", "Normal").lower()
        buf[offset + _MOVE_OFF["type"] + TYPE_TO_IDX.get(move_type, 0)] = 1.0

        # category (3)
        cat = move_data.get("category", "physical").lower()
        buf[offset + _MOVE_OFF["cat"] + MOVE_CATEGORY_TO_IDX.get(cat, 0)] = 1.0

        # priority (8)
        pri = move_data.get("priority", 0)
        pri_idx = max(0, min(7, pri - PRIORITY_MIN))
        buf[offset + _MOVE_OFF["pri"] + pri_idx] = 1.0

        # pp_fraction (1)
        pp = move_data.get("pp", 0)
        max_pp = move_data.get("maxpp", 1)
        buf[offset + _MOVE_OFF["pp"]] = float(pp) / max(max_pp, 1)

        # is_known (1)
        buf[offset + _MOVE_OFF["known"]] = 1.0

    # ------------------------------------------------------------------
    # Legal action mask
    # ------------------------------------------------------------------

    def _fill_legal_mask(self, state: dict):
        legal = state.get("legal_actions")
        if legal is None:
            self._legal_buf[:] = 1.0
        else:
            self._legal_buf[:] = 0.0
            for a in legal:
                if 0 <= a < 10:
                    self._legal_buf[a] = 1.0

    # ------------------------------------------------------------------
    # Legacy API (kept for backward compatibility with tests/scripts)
    # ------------------------------------------------------------------

    def _encode_field(self, state: dict) -> np.ndarray:
        buf = np.zeros(FIELD_DIM, dtype=np.float32)
        self._fill_field(state, buf)
        return buf

    def _encode_team(self, side: dict, is_own: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        tokens = []
        active = side.get("active")
        if active:
            active_aug = dict(active)
            active_aug["substitute_health"] = side.get("substitute_health", 0)
            active_aug["force_trapped"] = side.get("force_trapped", False)
            active_aug["volatile_durations"] = side.get("volatile_durations", {})
            active_aug["last_used_move"] = side.get("last_used_move", "none")
            active_aug["protect_count"] = side.get("protect_count", 0)
            active_aug["locked_move"] = side.get("locked_move", False)
            active_aug["perish_count"] = side.get("perish_count", 0)
            tokens.append(self._encode_mon_legacy(active_aug, slot=0, is_own=is_own, is_active=True))
        for i, mon in enumerate(side.get("reserve", [])):
            tokens.append(self._encode_mon_legacy(mon, slot=len(tokens), is_own=is_own, is_active=False))
        return tokens

    def _encode_mon(self, mon: dict, slot: int, is_own: bool, is_active: bool
                    ) -> tuple[np.ndarray, np.ndarray]:
        return self._encode_mon_legacy(mon, slot, is_own, is_active)

    def _encode_mon_legacy(self, mon: dict, slot: int, is_own: bool, is_active: bool
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy encode_mon that allocates arrays (kept for tests)."""
        int_row = np.zeros(_N_INT_IDS, dtype=np.int64)
        float_row = np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32)
        self._fill_mon(mon, slot, is_own, is_active, int_row, float_row)
        return int_row, float_row

    def _pad_team(self, tokens: list[tuple[np.ndarray, np.ndarray]], is_own: bool
                  ) -> list[tuple[np.ndarray, np.ndarray]]:
        while len(tokens) < N_TEAM_SLOTS:
            slot = len(tokens)
            int_ids = np.zeros(_N_INT_IDS, dtype=np.int64)
            float_feats = np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32)
            float_feats[_OFF["slot"] + slot] = 1.0
            float_feats[_OFF["is_own"]] = float(is_own)
            tokens.append((int_ids, float_feats))
        return tokens[:N_TEAM_SLOTS]

    def _build_legal_mask(self, state: dict) -> np.ndarray:
        legal = state.get("legal_actions")
        if legal is None:
            return np.ones(10, dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        for a in legal:
            if 0 <= a < 10:
                mask[a] = 1.0
        return mask

    def _encode_hazards(self, hazards: dict) -> np.ndarray:
        buf = np.zeros(7, dtype=np.float32)
        self._fill_hazards(hazards, buf, 0)
        return buf

    def _encode_screens(self, screens: dict) -> np.ndarray:
        buf = np.zeros(6, dtype=np.float32)
        self._fill_screens(screens, buf, 0)
        return buf

    def _encode_toxic_count(self, count: int) -> np.ndarray:
        return _one_hot(_toxic_bin_idx(count), 5)
