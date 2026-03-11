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
N_VOLATILE = 20            # see VOLATILE_STATUS_LIST below
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
    type_onehot: np.ndarray      # (18,) one-hot
    category_onehot: np.ndarray  # (3,)  one-hot
    priority_onehot: np.ndarray  # (8,)  one-hot  (-3…+4)
    pp_fraction: float           # scalar [0, 1]
    is_known: float              # 1.0 if move is known, 0.0 if hidden

    @property
    def dim(self) -> int:
        return 8 + 18 + 3 + 8 + 1 + 1  # = 39

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.base_power_bin,
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
    is_dynamaxed: float
    can_dynamax: float
    dynamax_turns: np.ndarray    # (4,)  one-hot (0,1,2,3 turns remaining)
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
    + 20  # volatile
    + 18  # type1
    + 18  # type2
    + 1   # is_dynamaxed
    + 1   # can_dynamax
    + 4   # dynamax_turns
    + 1   # is_fainted
    + 1   # is_active
    + 6   # slot
    + 1   # is_own
    + 4 * 39  # move features (4 × 39)
)
# = 1+10+6+91+7+20+18+18+1+1+4+1+1+6+1+156 = 342


FIELD_DIM = (
    5    # weather one-hot
    + 8  # weather turns bin
    + 5  # terrain one-hot
    + 8  # terrain turns bin
    + 5  # pseudo-weather multi-hot (trick room, gravity, wonder room, etc.)
    + 4  # trick room turns
    + 7  # hazards own (sr=1, spikes=3, tspikes=2, web=1)
    + 7  # hazards opp
    + 8  # screens own (ls turns 0-5 + reflect turns + aurora veil turns) — simplified to 3 flags + 5 turn bins each
    + 8  # screens opp
    + 10  # turn number bin
    + 2   # total fainted (own / opp) normalized
)
# simplified FIELD_DIM = 77 (exact value computed from to_array below)


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
            type_onehot=np.zeros(N_TYPES, dtype=np.float32),
            category_onehot=np.zeros(3, dtype=np.float32),
            priority_onehot=np.zeros(8, dtype=np.float32),
            pp_fraction=1.0,
            is_known=0.0,
        )
        return move_idx, feat

    move_idx = vocab.move_idx(move_data.get("id", ""))
    bp = move_data.get("basePower", 0)
    move_type = move_data.get("type", "Normal").lower()
    category = move_data.get("category", "physical").lower()
    priority = move_data.get("priority", 0)
    pp = move_data.get("pp", 0)
    max_pp = move_data.get("maxpp", 1)

    priority_idx = int(np.clip(priority - PRIORITY_MIN, 0, 7))
    type_idx = TYPE_TO_IDX.get(move_type, 0)

    feat = MoveFeatures(
        base_power_bin=_bin_base_power(bp),
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
          "int_ids"    : np.ndarray (15, 6)  — [species, m0, m1, m2, m3, ability, item]
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
        # int_ids shape per token: (7,) — [species, m0, m1, m2, m3, ability, item]
        field_int = np.zeros(7, dtype=np.int64)
        field_float = np.pad(field_vec, (0, FLOAT_DIM_PER_POKEMON - len(field_vec))).astype(np.float32)

        # Collect all 15 tokens
        all_int_ids = np.stack(
            [field_int] + [t[0] for t in own_tokens] + [t[0] for t in opp_tokens]
            + [np.zeros(7, dtype=np.int64), np.zeros(7, dtype=np.int64)],
            axis=0,
        )  # (15, 7)

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
        terrain = state.get("terrain", "") or ""
        terrain_turns = state.get("terrain_turns", 0) or 0
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

        return np.concatenate([
            _one_hot(WEATHER_TO_IDX.get(weather.lower(), 0), 5),
            _bin_turns(weather_turns),
            _one_hot(TERRAIN_TO_IDX.get(terrain.lower(), 0), 5),
            _bin_turns(terrain_turns),
            pseudo,
            _bin_turns(trick_room_turns, 4),
            hazards_own,
            hazards_opp,
            screens_own,
            screens_opp,
            _bin_turn_number(turn),
            [own_fainted / 6.0, opp_fainted / 6.0],
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
        """8-dim: light_screen_turns(0-5) + reflect_turns(0-5) ... simplified to 3 bool + turns."""
        ls = int(screens.get("light_screen", 0))
        ref = int(screens.get("reflect", 0))
        av = int(screens.get("aurora_veil", 0))
        # Encode each as present-flag + turns-remaining (capped at 5)
        return np.array([
            float(ls > 0), float(ls) / 5.0,
            float(ref > 0), float(ref) / 5.0,
            float(av > 0), float(av) / 5.0,
            0.0, 0.0,  # padding to 8
        ], dtype=np.float32)

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
            tokens.append(self._encode_mon(active, slot=0, is_own=is_own, is_active=True))
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

        ability_idx = vocab.ability_idx(mon.get("ability") if is_own else None)
        item_idx = vocab.item_idx(mon.get("item") if is_own else None)

        int_ids = np.array([species_idx, *move_idxs, ability_idx, item_idx], dtype=np.int64)  # (7,)
        # Pad to 6 for consistent shape: we use 6 cols (species + 4 moves + ability + item = 7, so store as (7,))
        # Actually reshape to fit our 6-col schema: species, m0, m1, m2, m3, ability (drop item → embed separately)
        # For simplicity, store all 7 IDs; model will handle indexing.
        int_ids = np.array([species_idx, *move_idxs, ability_idx, item_idx], dtype=np.int64)

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
        type2 = _one_hot(TYPE_TO_IDX.get((types[1].lower() if len(types) > 1 else ""), 0), N_TYPES)

        is_dynamaxed = float(mon.get("is_dynamaxed", False))
        can_dynamax = float(mon.get("can_dynamax", False))
        dmax_turns = int(mon.get("dynamax_turns_remaining", 0))

        is_fainted = float(mon.get("is_fainted", False) or hp <= 0)

        float_feats = np.concatenate([
            [hp_frac],
            _bin_hp(hp_frac),
            base_stats,
            _encode_stat_boosts(boosts),
            _one_hot(STATUS_TO_IDX.get(status, 0), N_STATUS),
            _encode_volatile_status(volatile),
            type1,
            type2,
            [is_dynamaxed, can_dynamax],
            _bin_turns(dmax_turns, 4),
            [float(is_fainted), float(is_active)],
            _one_hot(slot, N_TEAM_SLOTS),
            [float(is_own)],
            *[mf.to_array() for mf in move_feats],
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
                 UNKNOWN_MOVE_IDX, UNKNOWN_MOVE_IDX, UNKNOWN_ABILITY_IDX, UNKNOWN_ITEM_IDX],
                dtype=np.int64,
            )
            float_feats = np.zeros(FLOAT_DIM_PER_POKEMON, dtype=np.float32)
            # Mark slot position
            slot_start = (
                1 + 10 + 6 + 91 + 7 + 20 + 18 + 18 + 1 + 1 + 4 + 1 + 1
            )  # offset to slot one-hot
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
