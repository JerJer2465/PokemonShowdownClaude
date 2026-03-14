"""
poke-env Player wrapper for the PokeTransformer model.

Converts poke-env's Battle object into the obs dict format expected by
ObsBuilder, runs the model, and maps the predicted action back to a
poke-env BattleOrder.

Compatible with poke-env >= 0.8.1 and Gen 4 Random Battle.
"""

from __future__ import annotations

import os
import sys
import asyncio
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from poke_env.player import Player
from poke_env.environment import (
    AbstractBattle,
    Pokemon,
    Move,
    Weather,
    Field,
    SideCondition,
    PokemonType,
    Status,
)

# DamageClass was called MoveCategory in older poke-env versions
try:
    from poke_env.environment import DamageClass as MoveCategory
except ImportError:
    try:
        from poke_env.environment import MoveCategory
    except ImportError:
        MoveCategory = None  # type: ignore

# Effect enum (volatile statuses)
try:
    from poke_env.environment import Effect
except ImportError:
    try:
        from poke_env.environment.effect import Effect
    except ImportError:
        Effect = None  # type: ignore

from pokebot.env.obs_builder import ObsBuilder
from pokebot.model.poke_transformer import PokeTransformer
from config.model_config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Enum → string mappings
# ---------------------------------------------------------------------------

_WEATHER_TO_STR: Dict = {
    Weather.SUNNYDAY: "sunnyday",
    Weather.RAINDANCE: "raindance",
    Weather.SANDSTORM: "sandstorm",
    Weather.HAIL: "hail",
}
# Add Gen8+ variants safely if they exist in this poke-env version
for _attr, _val in [("SNOW", "hail"), ("DESOLATELAND", "sunnyday"),
                    ("PRIMORDIALSEA", "raindance"), ("DELTASTREAM", "")]:
    try:
        _WEATHER_TO_STR[getattr(Weather, _attr)] = _val
    except AttributeError:
        pass
del _attr, _val

_STATUS_TO_STR: Dict = {
    Status.BRN: "brn",
    Status.PSN: "psn",
    Status.TOX: "tox",
    Status.SLP: "slp",
    Status.FRZ: "frz",
    Status.PAR: "par",
    None: None,
}

_TYPE_TO_STR: Dict = {
    PokemonType.NORMAL: "Normal", PokemonType.FIRE: "Fire",
    PokemonType.WATER: "Water", PokemonType.ELECTRIC: "Electric",
    PokemonType.GRASS: "Grass", PokemonType.ICE: "Ice",
    PokemonType.FIGHTING: "Fighting", PokemonType.POISON: "Poison",
    PokemonType.GROUND: "Ground", PokemonType.FLYING: "Flying",
    PokemonType.PSYCHIC: "Psychic", PokemonType.BUG: "Bug",
    PokemonType.ROCK: "Rock", PokemonType.GHOST: "Ghost",
    PokemonType.DRAGON: "Dragon", PokemonType.DARK: "Dark",
    PokemonType.STEEL: "Steel", PokemonType.FAIRY: "Fairy",
}

def _build_move_cat_map() -> Dict:
    if MoveCategory is None:
        return {None: "physical"}
    try:
        return {
            MoveCategory.PHYSICAL: "physical",
            MoveCategory.SPECIAL: "special",
            MoveCategory.STATUS: "status",
            None: "physical",
        }
    except AttributeError:
        return {None: "physical"}

_MOVE_CAT_TO_STR: Dict = _build_move_cat_map()


def _build_effect_map() -> Dict:
    """Build volatile status mapping from poke-env Effect enum."""
    if Effect is None:
        return {}
    _names = {
        "CONFUSION": "confusion", "ATTRACT": "infatuation",
        "LEECH_SEED": "leechseed", "CURSE": "curse",
        "AQUA_RING": "aquaring", "INGRAIN": "ingrain",
        "TAUNT": "taunt", "ENCORE": "encore",
        "EMBARGO": "embargo", "YAWN": "yawn",
        "FLINCH": "flinch", "MAGNET_RISE": "magnetrise",
        "FOCUS_ENERGY": "focusenergy", "SUBSTITUTE": "substitute",
        "PERISHSONG": "perishsong",
    }
    result = {}
    for attr, val in _names.items():
        try:
            result[getattr(Effect, attr)] = val
        except AttributeError:
            pass
    return result

# poke-env Effect → our volatile status string
_EFFECT_TO_VOLATILE: Dict = _build_effect_map()

_FIELD_TO_TERRAIN: Dict = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
}


# ---------------------------------------------------------------------------
# Helper: normalize species/move names to our vocab format
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalnum())


def _base_stats_from_pe(pe_stats: dict) -> dict:
    """Remap poke-env base stat keys to our obs_builder keys."""
    return {
        "hp": pe_stats.get("hp", 80),
        "attack": pe_stats.get("atk", 80),
        "defense": pe_stats.get("def", 80),
        "special-attack": pe_stats.get("spa", 80),
        "special-defense": pe_stats.get("spd", 80),
        "speed": pe_stats.get("spe", 80),
    }


# ---------------------------------------------------------------------------
# Battle → obs dict conversion
# ---------------------------------------------------------------------------

def _move_to_dict(move: Move, is_known: bool = True, pp_override: Optional[int] = None) -> dict:
    """Convert a poke-env Move to obs_builder move dict."""
    if move is None or not is_known:
        return {
            "id": "unknown",
            "basePower": 0,
            "type": "Normal",
            "category": "physical",
            "priority": 0,
            "pp": 16,
            "maxpp": 16,
            "is_known": False,
        }
    pp = pp_override if pp_override is not None else (move.current_pp or 1)
    max_pp = move.max_pp or 16
    return {
        "id": _normalize(move.id),
        "basePower": getattr(move, "base_power", 0) or 0,
        "type": _TYPE_TO_STR.get(getattr(move, "type", None), "Normal"),
        "category": _MOVE_CAT_TO_STR.get(getattr(move, "category", None), "physical"),
        "priority": getattr(move, "priority", 0) or 0,
        "pp": pp,
        "maxpp": max_pp,
        "is_known": True,
        "disabled": False,
    }


def _pokemon_to_dict(
    pokemon: Pokemon,
    is_own: bool,
    is_active: bool,
    available_moves: Optional[List[Move]] = None,
    available_switches: Optional[List[Pokemon]] = None,
) -> dict:
    """
    Convert a poke-env Pokemon to obs_builder mon dict.

    For the active pokemon on our side, available_moves is used for the move
    list so the ordering matches our action indices (0-3).
    For all other mons, the pokemon.moves dict is used.
    """
    # HP
    if is_own:
        hp = pokemon.current_hp or 0
        maxhp = pokemon.max_hp or max(hp, 1)
    else:
        # Opponent: we have hp_fraction; estimate maxhp from base stats
        bst = getattr(pokemon, "base_stats", {}) or {}
        base_hp = bst.get("hp", 80)
        # Rough max HP at level 100: floor(2*base + 31) + 110
        maxhp = 2 * base_hp + 141
        hp = int(pokemon.current_hp_fraction * maxhp)

    # Status
    status_enum = pokemon.status
    status_str = _STATUS_TO_STR.get(status_enum, None)

    # Types
    types_raw = pokemon.types  # Tuple[PokemonType, Optional[PokemonType]]
    type1 = _TYPE_TO_STR.get(types_raw[0], "Normal") if types_raw else "Normal"
    type2 = _TYPE_TO_STR.get(types_raw[1], None) if (types_raw and len(types_raw) > 1 and types_raw[1]) else None
    types_list = [type1] + ([type2] if type2 else [])

    # Boosts
    boosts = dict(pokemon.boosts) if pokemon.boosts else {}

    # Volatile statuses
    effects = getattr(pokemon, "effects", {}) or {}
    volatile = []
    for eff, _ in effects.items():
        vs = _EFFECT_TO_VOLATILE.get(eff)
        if vs:
            volatile.append(vs)

    # Base stats
    pe_bst = getattr(pokemon, "base_stats", {}) or {}
    base_stats = _base_stats_from_pe(pe_bst)

    # Moves — for our active mon, use available_moves to match action ordering
    if is_active and is_own and available_moves is not None:
        # Use available moves as slots 0-3 (these match action indices directly)
        moves_list = []
        for i in range(4):
            if i < len(available_moves):
                moves_list.append(_move_to_dict(available_moves[i], is_known=True))
            else:
                moves_list.append(_move_to_dict(None, is_known=True))
    else:
        # Reserve own / all opp: use pokemon.moves dict
        pm = list(pokemon.moves.values()) if pokemon.moves else []
        moves_list = []
        for i in range(4):
            if i < len(pm):
                m = pm[i]
                moves_list.append(_move_to_dict(m, is_known=is_own))
            else:
                moves_list.append(_move_to_dict(None, is_known=True))

    return {
        "species": _normalize(pokemon.species),
        "hp": hp,
        "maxhp": maxhp,
        "status": status_str,
        "boosts": boosts,
        "moves": moves_list,
        "ability": _normalize(pokemon.ability) if pokemon.ability else None,  # revealed via battle events
        "item": _normalize(pokemon.item) if pokemon.item else None,          # revealed via battle events
        "types": types_list,
        "base_stats": base_stats,
        "volatile_statuses": volatile,
        "is_fainted": pokemon.fainted,
        "is_active": is_active,
    }


def _encode_side_conditions(
    side_conds: Dict[SideCondition, int]
) -> Tuple[dict, dict]:
    """Returns (hazards_dict, screens_dict) from poke-env side conditions."""
    hazards = {
        "stealth_rock": SideCondition.STEALTH_ROCK in side_conds,
        "spikes": side_conds.get(SideCondition.SPIKES, 0),
        "toxic_spikes": side_conds.get(SideCondition.TOXIC_SPIKES, 0),
        "sticky_web": SideCondition.STICKY_WEB in side_conds,
    }
    screens = {
        "light_screen": side_conds.get(SideCondition.LIGHT_SCREEN, 0),
        "reflect": side_conds.get(SideCondition.REFLECT, 0),
        "aurora_veil": 0,  # Gen 7+ only; always 0 in Gen 4
    }
    return hazards, screens


def battle_to_obs_dict(
    battle: AbstractBattle,
    available_moves: List[Move],
    available_switches: List[Pokemon],
) -> dict:
    """
    Convert a poke-env AbstractBattle to the obs dict format expected by ObsBuilder.

    available_moves: battle.available_moves (in order, up to 4)
    available_switches: battle.available_switches (in order, up to 6)

    Legal action mask:
      - Actions 0..(len(available_moves)-1): use those move indices
      - Actions 4..(4+len(available_switches)-1): use those switch indices
    """
    # ---- Our side ----
    own_active = battle.active_pokemon
    own_reserve = [p for p in battle.team.values() if p != own_active and not p.fainted]
    # Include fainted mons at the end (for team awareness)
    own_fainted = [p for p in battle.team.values() if p != own_active and p.fainted]
    own_reserve_all = own_reserve + own_fainted

    own_active_dict = _pokemon_to_dict(
        own_active, is_own=True, is_active=True,
        available_moves=available_moves,
    )
    own_reserve_dicts = [
        _pokemon_to_dict(p, is_own=True, is_active=False)
        for p in own_reserve_all
    ]

    own_hazards, own_screens = _encode_side_conditions(battle.side_conditions)

    side_one = {
        "active": own_active_dict,
        "reserve": own_reserve_dicts,
        "hazards": own_hazards,
        "screens": own_screens,
    }

    # ---- Opponent side ----
    opp_active = battle.opponent_active_pokemon
    opp_reserve = []
    if opp_active and battle.opponent_team:
        opp_reserve = [
            p for p in battle.opponent_team.values()
            if p != opp_active
        ]

    if opp_active:
        opp_active_dict = _pokemon_to_dict(opp_active, is_own=False, is_active=True)
    else:
        # Fallback if we somehow don't know the opponent's active mon
        opp_active_dict = {
            "species": "unknown", "hp": 100, "maxhp": 100, "status": None,
            "boosts": {}, "moves": [], "ability": None, "item": None,
            "types": ["Normal"], "base_stats": {}, "volatile_statuses": [],
            "is_fainted": False, "is_active": True,
        }

    opp_reserve_dicts = [
        _pokemon_to_dict(p, is_own=False, is_active=False)
        for p in opp_reserve
    ]

    opp_hazards, opp_screens = _encode_side_conditions(battle.opponent_side_conditions)

    side_two = {
        "active": opp_active_dict,
        "reserve": opp_reserve_dicts,
        "hazards": opp_hazards,
        "screens": opp_screens,
    }

    # ---- Field ----
    weather_str = ""
    weather_turns = 0
    for w, turns in (battle.weather or {}).items():
        weather_str = _WEATHER_TO_STR.get(w, "")
        weather_turns = turns
        break

    terrain_str = ""
    terrain_turns = 0
    trick_room = False
    trick_room_turns = 0
    for f, turns in (battle.fields or {}).items():
        t = _FIELD_TO_TERRAIN.get(f)
        if t:
            terrain_str = t
            terrain_turns = turns
        if f == Field.TRICK_ROOM:
            trick_room = True
            trick_room_turns = turns

    # ---- Legal actions ----
    legal_actions = list(range(len(available_moves)))
    for i in range(len(available_switches)):
        legal_actions.append(4 + i)
    if not legal_actions:
        legal_actions = [0]

    return {
        "side_one": side_one,
        "side_two": side_two,
        "weather": weather_str,
        "weather_turns": weather_turns,
        "terrain": terrain_str,
        "terrain_turns": terrain_turns,
        "trick_room": trick_room,
        "trick_room_turns": trick_room_turns,
        "turn": battle.turn,
        "legal_actions": legal_actions,
    }


# ---------------------------------------------------------------------------
# Player class
# ---------------------------------------------------------------------------

class PokeTransformerPlayer(Player):
    """
    Pokemon Showdown bot player using the PokeTransformer model.

    Usage:
        player = PokeTransformerPlayer.from_checkpoint(
            "checkpoints/bc_init.pt",
            player_configuration=PlayerConfiguration("BotName", None),
            battle_format="gen4randombattle",
        )
        await player.ladder(20)
    """

    def __init__(
        self,
        model: PokeTransformer,
        device: str = "cpu",
        deterministic: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.obs_builder = ObsBuilder()
        self.deterministic = deterministic
        self.temperature = temperature

        # Stats tracking
        self._win_count = 0
        self._loss_count = 0
        self._tie_count = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        **kwargs,
    ) -> "PokeTransformerPlayer":
        """Load model from a .pt checkpoint file and create the player."""
        model = PokeTransformer()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded model from {checkpoint_path}")
        return cls(model=model, device=device, **kwargs)

    def choose_move(self, battle: AbstractBattle):
        """Select the best move using the trained model."""
        available_moves = list(battle.available_moves)
        available_switches = list(battle.available_switches)

        # If forced switch with no moves, handle gracefully
        if not available_moves and not available_switches:
            return self.choose_random_move(battle)

        # Build obs dict
        obs_dict = battle_to_obs_dict(battle, available_moves, available_switches)

        # Encode
        obs = self.obs_builder.encode(obs_dict)

        # Inference
        int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(self.device)
        float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(self.device)
        legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_probs, _, value = self.model(int_ids, float_f, legal_m)

        # Scale by temperature for exploration
        if self.temperature != 1.0:
            log_probs = log_probs / self.temperature

        # Select action
        if self.deterministic:
            action = int(log_probs.argmax(dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(probs=log_probs.exp()).sample().item())

        # Map model action (slot-based) to poke-env BattleOrder.
        #
        # Model action 0-3 = move *slot* on active pokemon (fixed positions).
        # poke-env available_moves is a *dense* list of currently-legal moves,
        # which may skip disabled/PP-empty slots. We must match by move ID.
        if action < 4:
            # Build slot → available move mapping by matching move IDs
            all_moves = list(battle.active_pokemon.moves.values())  # up to 4 slots
            slot_to_move = {}
            for avail_move in available_moves:
                for slot_idx, slot_move in enumerate(all_moves):
                    if slot_move.id == avail_move.id:
                        slot_to_move[slot_idx] = avail_move
                        break
            if action in slot_to_move:
                return self.create_order(slot_to_move[action])
            # Chosen slot is disabled/PP-empty — fall through to first available
            if available_moves:
                return self.create_order(available_moves[0])

        elif action >= 4:
            switch_idx = action - 4
            if switch_idx < len(available_switches):
                return self.create_order(available_switches[switch_idx])

        # Fallback: any legal move
        if available_moves:
            return self.create_order(available_moves[0])
        elif available_switches:
            return self.create_order(available_switches[0])
        else:
            return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle: AbstractBattle):
        """Track win/loss statistics."""
        if battle.won:
            self._win_count += 1
        elif battle.lost:
            self._loss_count += 1
        else:
            self._tie_count += 1
        total = self._win_count + self._loss_count + self._tie_count
        wr = self._win_count / max(total, 1) * 100
        print(
            f"Battle {total}: {'WIN' if battle.won else 'LOSS' if battle.lost else 'TIE'}  "
            f"| W/L/T: {self._win_count}/{self._loss_count}/{self._tie_count}  "
            f"| WR: {wr:.1f}%  "
            f"| turns: {battle.turn}"
        )

    @property
    def win_rate(self) -> float:
        total = self._win_count + self._loss_count + self._tie_count
        return self._win_count / max(total, 1)
