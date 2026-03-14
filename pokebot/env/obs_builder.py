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
"""

from __future__ import annotations

import json
import math
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

# Offset to slot one-hot within float_feats (used by _pad_team)
_SLOT_ONEHOT_OFFSET = 1 + 10 + 6 + 91 + 7 + N_VOLATILE + 18 + 18 + 1 + 1

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
# Helper functions
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

class ObsBuilder:
    """
    Converts a battle state dict (as produced by poke_engine_env) into
    the token arrays expected by PokeTransformer.

    The poke-engine battle state is a Python dict with structure:
      {
        "side_one": { "active": {...mon...}, "reserve": [{...mon...}, ...] },
        "side_two": { ... same ... },
        "weather": str,
        "weather_turns": int,
        "terrain": str,
        "terrain_turns": int,
        "trick_room": bool,
        "trick_room_turns": int,
        "turn": int,
      }
    Each mon dict has: species, hp, maxhp, status, boosts, moves, ability, item,
                       types, base_stats, volatile_statuses, is_fainted
    """

    def __init__(self):
        self.vocab = Vocab.get()

    def encode(self, state: dict) -> dict:
        """
        Returns a dict with:
          "int_ids"    : np.ndarray (15, 8)  — [species, m0..m3, ability, item, last_used_move]
                         (query tokens use UNKNOWN for all int IDs)
          "float_feats": np.ndarray (15, float_dim)  — per-token float features
                         (query tokens are all zeros)
          "legal_mask" : np.ndarray (n_actions,)  — 1.0 where action is legal
        """
        field_vec = self._encode_field(state)

        side_one = state["side_one"]
        side_two = state["side_two"]

        own_tokens = self._encode_team(side_one, is_own=True)
        opp_tokens = self._encode_team(side_two, is_own=False)

        # Pad to 6 slots each
        own_tokens = self._pad_team(own_tokens, is_own=True)
        opp_tokens = self._pad_team(opp_tokens, is_own=False)

        # field_dim float features for token 0; UNKNOWN int IDs
        # int_ids shape per token: (8,) — [species, m0, m1, m2, m3, ability, item, last_used_move]
        N_INT_IDS = 8
        field_int = np.zeros(N_INT_IDS, dtype=np.int64)
        field_float = np.pad(field_vec, (0, FLOAT_DIM_PER_POKEMON - len(field_vec))).astype(np.float32)

        # Collect all 15 tokens
        all_int_ids = np.stack(
            [field_int] + [t[0] for t in own_tokens] + [t[0] for t in opp_tokens]
            + [np.zeros(N_INT_IDS, dtype=np.int64), np.zeros(N_INT_IDS, dtype=np.int64)],
            axis=0,
        )  # (15, 8)

        all_float = np.stack(
            [field_float] + [t[1] for t in own_tokens] + [t[1] for t in opp_tokens]
            + [np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32),
               np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32)],
            axis=0,
        )  # (15, FLOAT_DIM_PER_POKEMON)

        legal_mask = self._build_legal_mask(state)

        return {
            "int_ids": all_int_ids,
            "float_feats": all_float,
            "legal_mask": legal_mask,
        }

    # ------------------------------------------------------------------
    # Field token
    # ------------------------------------------------------------------

    def _encode_field(self, state: dict) -> np.ndarray:
        weather = state.get("weather", "") or ""
        weather_turns = state.get("weather_turns", 0) or 0
        trick_room = state.get("trick_room", False)
        trick_room_turns = state.get("trick_room_turns", 0) or 0
        turn = state.get("turn", 1) or 1

        side_one = state["side_one"]
        side_two = state["side_two"]

        pseudo = np.zeros(5, dtype=np.float32)
        if trick_room:
            pseudo[0] = 1.0
        if state.get("gravity"):
            pseudo[1] = 1.0
        if state.get("wonder_room"):
            pseudo[2] = 1.0

        # Hazards — side_one = own
        hazards_own = self._encode_hazards(side_one.get("hazards", {}))
        hazards_opp = self._encode_hazards(side_two.get("hazards", {}))

        # Screens — stored as turn counts in state
        screens_own = self._encode_screens(side_one.get("screens", {}))
        screens_opp = self._encode_screens(side_two.get("screens", {}))

        # Fainted counts
        own_fainted = sum(1 for m in side_one.get("reserve", []) if m.get("is_fainted"))
        opp_fainted = sum(1 for m in side_two.get("reserve", []) if m.get("is_fainted"))
        if side_one.get("active", {}).get("is_fainted"):
            own_fainted += 1
        if side_two.get("active", {}).get("is_fainted"):
            opp_fainted += 1

        # NEW: side-level conditions
        toxic_own = self._encode_toxic_count(side_one.get("toxic_count", 0))
        toxic_opp = self._encode_toxic_count(side_two.get("toxic_count", 0))
        tailwind_own = float(side_one.get("tailwind", 0) > 0)
        tailwind_opp = float(side_two.get("tailwind", 0) > 0)
        wish_own = float(side_one.get("wish", (0, 0))[0] > 0)
        wish_opp = float(side_two.get("wish", (0, 0))[0] > 0)
        safeguard_own = float(side_one.get("safeguard", 0) > 0)
        safeguard_opp = float(side_two.get("safeguard", 0) > 0)

        # Mist and Lucky Chant (Gen 4+)
        mist_own = float(side_one.get("mist", 0) > 0)
        mist_opp = float(side_two.get("mist", 0) > 0)
        lucky_chant_own = float(side_one.get("lucky_chant", 0) > 0)
        lucky_chant_opp = float(side_two.get("lucky_chant", 0) > 0)

        # Gravity turns (separate from pseudo-weather flag)
        gravity_turns = state.get("gravity_turns", 0) or 0

        return np.concatenate([
            _one_hot(WEATHER_TO_IDX.get(weather.lower(), 0), 5),
            _bin_turns(weather_turns),
            # Terrain omitted — Gen 4 has no terrain mechanics
            pseudo,
            _bin_turns(trick_room_turns, 4),
            hazards_own,
            hazards_opp,
            screens_own,
            screens_opp,
            _bin_turn_number(turn),
            [own_fainted / 6.0, opp_fainted / 6.0],
            # side-level fields
            toxic_own,
            toxic_opp,
            [tailwind_own, tailwind_opp],
            [wish_own, wish_opp],
            [safeguard_own, safeguard_opp],
            [mist_own, mist_opp],
            [lucky_chant_own, lucky_chant_opp],
            _bin_turns(gravity_turns, 4),
        ]).astype(np.float32)

    def _encode_hazards(self, hazards: dict) -> np.ndarray:
        """7-dim: stealth_rock(1) + spikes(3 one-hot layers) + tspikes(2) + web(1)."""
        sr = float(hazards.get("stealth_rock", False))
        spikes = int(hazards.get("spikes", 0))
        tspikes = int(hazards.get("toxic_spikes", 0))
        web = float(hazards.get("sticky_web", False))
        # Spikes: 0-3 layers → 3 binary flags
        s_feats = np.array([float(spikes >= 1), float(spikes >= 2), float(spikes >= 3)], dtype=np.float32)
        t_feats = np.array([float(tspikes >= 1), float(tspikes >= 2)], dtype=np.float32)
        return np.array([sr, *s_feats, *t_feats, web], dtype=np.float32)  # (7,)

    def _encode_screens(self, screens: dict) -> np.ndarray:
        """6-dim: light_screen (flag+turns) + reflect (flag+turns). Aurora Veil removed (Gen 7+)."""
        ls = int(screens.get("light_screen", 0))
        ref = int(screens.get("reflect", 0))
        return np.array([
            float(ls > 0), float(ls) / 5.0,
            float(ref > 0), float(ref) / 5.0,
            0.0, 0.0,  # reserved padding
        ], dtype=np.float32)

    def _encode_toxic_count(self, count: int) -> np.ndarray:
        """5-bin one-hot for toxic counter: [0, 1, 2, 3-4, 5+]."""
        if count <= 0:
            return _one_hot(0, 5)
        elif count == 1:
            return _one_hot(1, 5)
        elif count == 2:
            return _one_hot(2, 5)
        elif count <= 4:
            return _one_hot(3, 5)
        else:
            return _one_hot(4, 5)

    # ------------------------------------------------------------------
    # Team tokens
    # ------------------------------------------------------------------

    def _encode_team(self, side: dict, is_own: bool) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns list of (int_ids, float_feats) per mon.
        Active mon is always first (slot index 0).
        """
        tokens = []
        active = side.get("active")
        if active:
            # Inject side-level state into active mon for encoding
            active_aug = dict(active)
            active_aug["substitute_health"] = side.get("substitute_health", 0)
            active_aug["force_trapped"] = side.get("force_trapped", False)
            active_aug["volatile_durations"] = side.get("volatile_durations", {})
            active_aug["last_used_move"] = side.get("last_used_move", "none")
            active_aug["protect_count"] = side.get("protect_count", 0)
            active_aug["locked_move"] = side.get("locked_move", False)
            active_aug["perish_count"] = side.get("perish_count", 0)
            tokens.append(self._encode_mon(active_aug, slot=0, is_own=is_own, is_active=True))
        for i, mon in enumerate(side.get("reserve", [])):
            tokens.append(self._encode_mon(mon, slot=len(tokens), is_own=is_own, is_active=False))
        return tokens

    def _encode_mon(
        self,
        mon: dict,
        slot: int,
        is_own: bool,
        is_active: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        vocab = self.vocab

        # --- Integer IDs ---
        species_idx = vocab.species_idx(mon.get("species", ""))
        moves = mon.get("moves", [])
        move_idxs = []
        move_feats = []
        for i in range(N_MOVES_PER_MON):
            if i < len(moves):
                m = moves[i]
                is_known = m.get("is_known", is_own)  # own moves always known
                midx, mfeat = _encode_move(m if is_known else None, is_known=is_known, vocab=vocab)
            else:
                midx, mfeat = EMPTY_MOVE_IDX, _encode_move(None, is_known=True, vocab=vocab)[1]
            move_idxs.append(midx)
            move_feats.append(mfeat)

        ability_idx = vocab.ability_idx(mon.get("ability"))  # None maps to UNKNOWN_IDX
        item_idx = vocab.item_idx(mon.get("item"))            # None maps to UNKNOWN_IDX

        # last_used_move: 8th int ID (reuses move vocab)
        last_move = mon.get("last_used_move", "none")
        last_move_idx = vocab.move_idx(last_move if last_move and last_move != "none" else None)

        # (8,): species, m0, m1, m2, m3, ability, item, last_used_move
        int_ids = np.array([species_idx, *move_idxs, ability_idx, item_idx, last_move_idx], dtype=np.int64)

        # --- Float features ---
        hp = float(mon.get("hp", 0))
        max_hp = float(mon.get("maxhp", 1))
        hp_frac = hp / max(max_hp, 1)

        # Base stats (from mon dict or zeros if unknown)
        bst = mon.get("base_stats", {})
        base_stats = np.array([
            bst.get("hp", 0), bst.get("attack", 0), bst.get("defense", 0),
            bst.get("special-attack", 0), bst.get("special-defense", 0), bst.get("speed", 0),
        ], dtype=np.float32) / 255.0

        boosts = mon.get("boosts", {})
        status = (mon.get("status") or "").lower()
        volatile = set(mon.get("volatile_statuses", []) or [])

        types = mon.get("types", ["Normal"])
        type1 = _one_hot(TYPE_TO_IDX.get((types[0] if types else "Normal").lower(), 0), N_TYPES)
        type2 = (_one_hot(TYPE_TO_IDX.get(types[1].lower(), 0), N_TYPES)
                 if len(types) > 1 else np.zeros(N_TYPES, dtype=np.float32))

        is_fainted = float(mon.get("is_fainted", False) or hp <= 0)

        # --- NEW: per-pokemon features ---
        sleep_turns = int(mon.get("sleep_turns", 0))
        sleep_bin = _one_hot(min(sleep_turns, 3), 4)  # [0, 1, 2, 3+]

        rest_turns = int(mon.get("rest_turns", 0))
        rest_bin = _one_hot(min(rest_turns, 2), 3)  # [0, 1, 2]

        # Substitute health as fraction of maxhp (0 if no sub)
        sub_hp = float(mon.get("substitute_health", 0))
        sub_frac = sub_hp / max(max_hp, 1) if sub_hp > 0 else 0.0

        force_trapped = float(mon.get("force_trapped", False))

        # Move disabled flags
        move_disabled = np.zeros(N_MOVES_PER_MON, dtype=np.float32)
        for i in range(min(N_MOVES_PER_MON, len(moves))):
            if moves[i].get("disabled", False):
                move_disabled[i] = 1.0

        # Volatile durations (from side-level data, only active mon has these)
        vd = mon.get("volatile_durations", {})
        confusion_bin = _one_hot(min(int(vd.get("confusion", 0)), 3), 4)
        taunt_flag = float(int(vd.get("taunt", 0)) > 0)
        encore_flag = float(int(vd.get("encore", 0)) > 0)
        yawn_flag = float(int(vd.get("yawn", 0)) > 0)

        # Level (normalized; important for Gen 4 randbats where levels vary)
        level = float(mon.get("level", 100))
        level_norm = level / 100.0

        # Perish Song counter (0 = not active, 1-3 = turns remaining)
        perish_count = int(mon.get("perish_count", 0))
        perish_bin = _one_hot(min(perish_count, 3), 4)  # [0, 1, 2, 3]

        # Protect counter (consecutive uses, affects success rate)
        protect_count = int(mon.get("protect_count", 0))
        protect_norm = min(protect_count, 4) / 4.0

        # Locked move flag (Choice item lock or multi-turn move like Outrage)
        locked_move = float(mon.get("locked_move", False))

        float_feats = np.concatenate([
            [hp_frac],
            _bin_hp(hp_frac),
            base_stats,
            _encode_stat_boosts(boosts),
            _one_hot(STATUS_TO_IDX.get(status, 0), N_STATUS),
            _encode_volatile_status(volatile),
            type1,
            type2,
            [float(is_fainted), float(is_active)],
            _one_hot(slot, N_TEAM_SLOTS),
            [float(is_own)],
            *[mf.to_array() for mf in move_feats],
            # Extended per-pokemon features
            sleep_bin,
            rest_bin,
            [sub_frac],
            [force_trapped],
            move_disabled,
            confusion_bin,
            [taunt_flag],
            [encore_flag],
            [yawn_flag],
            [level_norm],
            perish_bin,
            [protect_norm],
            [locked_move],
        ]).astype(np.float32)

        return int_ids, float_feats

    def _pad_team(
        self, tokens: list[tuple[np.ndarray, np.ndarray]], is_own: bool
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pad team to exactly 6 slots with UNKNOWN tokens."""
        while len(tokens) < N_TEAM_SLOTS:
            slot = len(tokens)
            int_ids = np.array(
                [UNKNOWN_SPECIES_IDX, UNKNOWN_MOVE_IDX, UNKNOWN_MOVE_IDX,
                 UNKNOWN_MOVE_IDX, UNKNOWN_MOVE_IDX, UNKNOWN_ABILITY_IDX, UNKNOWN_ITEM_IDX,
                 UNKNOWN_MOVE_IDX],  # last_used_move
                dtype=np.int64,
            )
            float_feats = np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32)
            # Mark slot position
            slot_start = _SLOT_ONEHOT_OFFSET
            if slot_start + slot < FLOAT_DIM_PER_POKEMON:
                float_feats[slot_start + slot] = 1.0
            # is_own flag
            is_own_offset = slot_start + N_TEAM_SLOTS
            if is_own_offset < FLOAT_DIM_PER_POKEMON:
                float_feats[is_own_offset] = float(is_own)
            tokens.append((int_ids, float_feats))
        return tokens[:N_TEAM_SLOTS]

    # ------------------------------------------------------------------
    # Legal action mask
    # ------------------------------------------------------------------

    def _build_legal_mask(self, state: dict) -> np.ndarray:
        """
        10-dim mask (4 moves + 6 switches), 1.0 = legal.
        Based on state["legal_actions"] if provided, else all ones.
        """
        legal = state.get("legal_actions")
        if legal is None:
            return np.ones(10, dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)
        for a in legal:
            if 0 <= a < 10:
                mask[a] = 1.0
        return mask
