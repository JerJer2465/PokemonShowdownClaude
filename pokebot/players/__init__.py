"""
poke-env ladder player for the PokeTransformerBot.

BattleStateAdapter  — converts a poke-env Battle into the obs dict format
                      consumed by ObsBuilder.
PokeTransformerPlayer — poke-env Player subclass that uses PokeTransformer
                        model weights to pick actions.

Usage (after training):
    python scripts/run_ladder.py --username MyBot --password secret --n_games 50
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional

try:
    from poke_env.player import Player
    from poke_env.environment.abstract_battle import AbstractBattle
    from poke_env.environment.weather import Weather as PEWeather
    from poke_env.environment.field import Field as PEField
    from poke_env.environment.side_condition import SideCondition as PESC
    from poke_env.environment.status import Status as PEStatus
    from poke_env.environment.damage_class import DamageClass as PEDamageClass

    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False

    class Player:  # type: ignore[no-redef]
        """Stub so module imports without poke-env installed."""
        pass

    class AbstractBattle:  # type: ignore[no-redef]
        pass

from pokebot.env.obs_builder import ObsBuilder
from config.model_config import MODEL_CONFIG


# ---------------------------------------------------------------------------
# Enum → string helpers
# ---------------------------------------------------------------------------

def _status_str(status) -> Optional[str]:
    """Convert poke-env Status enum → short string ('brn', 'psn', …) or None."""
    if status is None:
        return None
    _map = {
        "brn": "brn", "psn": "psn", "tox": "tox",
        "slp": "slp", "frz": "frz", "par": "par",
        # poke-env name variants
        "burn": "brn", "poison": "psn", "toxic": "tox",
        "sleep": "slp", "freeze": "frz", "paralysis": "par",
    }
    raw = ""
    try:
        raw = str(status.value).lower()
    except AttributeError:
        raw = str(status).lower()
    return _map.get(raw) or _map.get(raw.split(".")[-1])


def _weather_info(battle) -> tuple[str, int]:
    """Return (weather_id, turns_remaining)."""
    _map = {
        "SUNNYDAY":    "sunnyday",  "RAINDANCE":   "raindance",
        "SANDSTORM":   "sandstorm", "HAIL":        "hail",
        "SNOWSCAPE":   "hail",
        "DESOLATELAND": "sunnyday", "PRIMORDIALSEA": "raindance",
        "EXTREMELYHARSHSUNLIGHT": "sunnyday",
        "HEAVYRAIN": "raindance",
    }
    for w, turns in (battle.weather or {}).items():
        key = (w.name if hasattr(w, "name") else str(w)).upper().replace(" ", "")
        name = _map.get(key, "")
        if name:
            return name, int(turns)
    return "", 0


def _terrain_info(battle) -> tuple[str, int, bool, int]:
    """Return (terrain_id, terrain_turns, trick_room, trick_room_turns)."""
    _tmap = {
        "ELECTRICTERRAIN":  "electricterrain",
        "ELECTRIC_TERRAIN": "electricterrain",
        "GRASSYTERRAIN":    "grassyterrain",
        "GRASSY_TERRAIN":   "grassyterrain",
        "MISTYTERRAIN":     "mistyterrain",
        "MISTY_TERRAIN":    "mistyterrain",
        "PSYCHICTERRAIN":   "psychicterrain",
        "PSYCHIC_TERRAIN":  "psychicterrain",
    }
    terrain, terrain_turns = "", 0
    trick_room, trick_room_turns = False, 0

    for f, turns in (battle.fields or {}).items():
        key = (f.name if hasattr(f, "name") else str(f)).upper().replace(" ", "")
        t = _tmap.get(key, "")
        if t:
            terrain, terrain_turns = t, int(turns)
        if "TRICKROOM" in key or "TRICK_ROOM" in key:
            trick_room, trick_room_turns = True, int(turns)

    return terrain, terrain_turns, trick_room, trick_room_turns


def _side_conditions(conds: dict) -> tuple[dict, dict]:
    """Convert poke-env side_conditions dict → (hazards dict, screens dict)."""
    hazards = {"stealth_rock": False, "spikes": 0, "toxic_spikes": 0, "sticky_web": False}
    screens = {"light_screen": 0, "reflect": 0, "aurora_veil": 0}
    for sc, val in (conds or {}).items():
        key = (sc.name if hasattr(sc, "name") else str(sc)).upper().replace(" ", "_")
        if "STEALTH_ROCK" in key:
            hazards["stealth_rock"] = True
        elif "TOXIC_SPIKES" in key:
            hazards["toxic_spikes"] = int(val)
        elif "SPIKES" in key:
            hazards["spikes"] = int(val)
        elif "STICKY_WEB" in key:
            hazards["sticky_web"] = True
        elif "LIGHT_SCREEN" in key:
            screens["light_screen"] = int(val)
        elif "REFLECT" in key:
            screens["reflect"] = int(val)
        elif "AURORA_VEIL" in key:
            screens["aurora_veil"] = int(val)
    return hazards, screens


def _type_names(pokemon) -> list[str]:
    types = []
    for t in (pokemon.types or []):
        if t is not None:
            name = t.name if hasattr(t, "name") else str(t)
            types.append(name.capitalize())
    return types or ["Normal"]


def _category_str(move) -> str:
    dc = getattr(move, "damage_class", None)
    if dc is None:
        return "physical"
    name = (dc.name if hasattr(dc, "name") else str(dc)).upper()
    if "PHYSICAL" in name:
        return "physical"
    if "SPECIAL" in name:
        return "special"
    return "status"


# ---------------------------------------------------------------------------
# Pokemon dict builder
# ---------------------------------------------------------------------------

def _mon_to_dict(pokemon, is_own: bool, is_active: bool, slot: int) -> dict:
    """Convert a poke-env Pokemon → obs dict mon entry."""
    hp_frac = pokemon.current_hp_fraction if not pokemon.fainted else 0.0
    hp = int(hp_frac * 1000)

    status = _status_str(pokemon.status)

    # Stat boosts — only active mon has meaningful boosts
    boosts: dict = {}
    if is_active:
        raw = getattr(pokemon, "boosts", {}) or {}
        boosts = {k: int(v) for k, v in raw.items()}

    # Moves (known only)
    moves_out = []
    for move in pokemon.moves.values():
        pp = getattr(move, "current_pp", None)
        max_pp = getattr(move, "max_pp", 16) or 16
        if pp is None:
            pp = max_pp
        moves_out.append({
            "id":        move.id,
            "basePower": getattr(move, "base_power", 0) or 0,
            "type":      move.type.name.capitalize() if hasattr(move.type, "name") else "Normal",
            "category":  _category_str(move),
            "priority":  getattr(move, "priority", 0) or 0,
            "pp":        int(pp),
            "maxpp":     int(max_pp),
            "is_known":  True,
            "disabled":  False,
        })

    # Base stats (poke-env uses short keys: hp, atk, def, spa, spd, spe)
    bst = getattr(pokemon, "base_stats", {}) or {}
    base_stats = {
        "hp":              bst.get("hp",  bst.get("HP",  0)),
        "attack":          bst.get("atk", bst.get("attack",  0)),
        "defense":         bst.get("def", bst.get("defense", 0)),
        "special-attack":  bst.get("spa", bst.get("special-attack", 0)),
        "special-defense": bst.get("spd", bst.get("special-defense", 0)),
        "speed":           bst.get("spe", bst.get("speed", 0)),
    }

    types = _type_names(pokemon)

    # Volatile statuses (poke-env Effect enums)
    volatile = []
    for eff in (getattr(pokemon, "effects", None) or {}).keys():
        name = (eff.name if hasattr(eff, "name") else str(eff)).lower()
        volatile.append(name)

    return {
        "species":                pokemon.species,
        "hp":                     hp,
        "maxhp":                  1000,
        "status":                 status,
        "boosts":                 boosts,
        "moves":                  moves_out,
        "ability":                pokemon.ability,   # poke-env: None until revealed via battle events
        "item":                   pokemon.item,     # poke-env: None until revealed via battle events
        "types":                  types,
        "base_stats":             base_stats,
        "volatile_statuses":      volatile,
        "is_fainted":             pokemon.fainted,
        "is_dynamaxed":           getattr(pokemon, "is_dynamaxed", False),
        "can_dynamax":            False,
        "dynamax_turns_remaining": 0,
        "is_active":              is_active,
    }


def _build_side_dict(active, team: dict, side_conds: dict, is_own: bool) -> dict:
    """Build side dict (active + reserve list) for ObsBuilder."""
    active_dict = _mon_to_dict(active, is_own=is_own, is_active=True, slot=0)

    reserve = []
    slot = 1
    for mon in team.values():
        if mon is active:
            continue
        reserve.append(_mon_to_dict(mon, is_own=is_own, is_active=False, slot=slot))
        slot += 1

    hazards, screens = _side_conditions(side_conds)
    return {
        "active":  active_dict,
        "reserve": reserve,
        "hazards": hazards,
        "screens": screens,
    }


# ---------------------------------------------------------------------------
# Legal action helpers
# ---------------------------------------------------------------------------

def _ordered_moves(active):
    """Active Pokemon's moves as an ordered 4-slot list (None = empty slot)."""
    mvs = list(active.moves.values())
    while len(mvs) < 4:
        mvs.append(None)
    return mvs[:4]


def _ordered_switches(active, team: dict):
    """Non-active alive mons in team-dict order, padded to 6 slots."""
    switches = [m for m in team.values() if m is not active and not m.fainted]
    while len(switches) < 6:
        switches.append(None)
    return switches[:6]


def _build_legal_actions(battle, o_moves, o_switches) -> list[int]:
    avail_ids     = {m.id      for m in battle.available_moves}
    avail_species = {p.species for p in battle.available_switches}

    legal: list[int] = []
    for i, mv in enumerate(o_moves):
        if mv is not None and mv.id in avail_ids:
            legal.append(i)
    for i, mon in enumerate(o_switches):
        if mon is not None and mon.species in avail_species:
            legal.append(4 + i)

    if not legal:
        legal = [0]  # struggle fallback
    return legal


# ---------------------------------------------------------------------------
# BattleStateAdapter
# ---------------------------------------------------------------------------

class BattleStateAdapter:
    """Converts a poke-env Battle into the obs dict format for ObsBuilder."""

    def convert(self, battle) -> tuple[dict, list, list]:
        """
        Returns:
          obs_dict      — ready for ObsBuilder.encode()
          ordered_moves — 4-slot list of poke-env Move (or None)
          ordered_sws   — 6-slot list of poke-env Pokemon (or None)
        """
        weather, w_turns = _weather_info(battle)
        terrain, t_turns, trick_room, tr_turns = _terrain_info(battle)

        own_active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        side_one = _build_side_dict(
            own_active, battle.team, battle.side_conditions, is_own=True
        )
        side_two = _build_side_dict(
            opp_active, battle.opponent_team, battle.opponent_side_conditions, is_own=False
        )

        o_moves    = _ordered_moves(own_active)
        o_switches = _ordered_switches(own_active, battle.team)
        legal      = _build_legal_actions(battle, o_moves, o_switches)

        obs_dict = {
            "side_one":         side_one,
            "side_two":         side_two,
            "weather":          weather,
            "weather_turns":    w_turns,
            "terrain":          terrain,
            "terrain_turns":    t_turns,
            "trick_room":       trick_room,
            "trick_room_turns": tr_turns,
            "turn":             battle.turn,
            "legal_actions":    legal,
        }
        return obs_dict, o_moves, o_switches


# ---------------------------------------------------------------------------
# PokeTransformerPlayer
# ---------------------------------------------------------------------------

class PokeTransformerPlayer(Player):
    """
    poke-env Player that uses the trained PokeTransformer model.

    Example:
        from poke_env import AccountConfiguration, ShowdownServerConfiguration
        player = PokeTransformerPlayer(
            model=model,
            device="cuda",
            account_configuration=AccountConfiguration("MyBot", ""),
            server_configuration=ShowdownServerConfiguration,
            battle_format="gen8randombattle",
        )
        asyncio.run(player.ladder(50))
    """

    def __init__(self, model, device: str = "cpu", **kwargs):
        if not POKE_ENV_AVAILABLE:
            raise ImportError(
                "poke-env is required for online play. "
                "Install with: pip install 'poke-env>=0.8.1'"
            )
        super().__init__(**kwargs)
        self._model = model
        self._device = torch.device(device)
        self._obs_builder = ObsBuilder()
        self._adapter = BattleStateAdapter()
        self._model.eval()
        self._results: list[dict] = []

    # ------------------------------------------------------------------

    def choose_move(self, battle):
        try:
            obs_dict, o_moves, o_switches = self._adapter.convert(battle)
        except Exception as exc:
            print(f"[PokeTransformerPlayer] obs error turn={battle.turn}: {exc}")
            return self.choose_random_move(battle)

        obs = self._obs_builder.encode(obs_dict)

        with torch.no_grad():
            int_ids    = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(self._device)
            float_f    = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(self._device)
            legal_mask = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(self._device)

            log_probs, _, _ = self._model(int_ids, float_f, legal_mask)
            action = int(log_probs.argmax(dim=-1).item())

        return self._action_to_order(action, battle, o_moves, o_switches)

    def _action_to_order(self, action: int, battle, o_moves, o_switches):
        """Map model action index 0-9 → poke-env BattleOrder."""
        avail_move_map   = {m.id: m      for m in battle.available_moves}
        avail_switch_map = {p.species: p for p in battle.available_switches}

        if action < 4:
            mv = o_moves[action] if action < len(o_moves) else None
            if mv is not None and mv.id in avail_move_map:
                return self.create_order(avail_move_map[mv.id])
        elif action < 10:
            idx = action - 4
            mon = o_switches[idx] if idx < len(o_switches) else None
            if mon is not None and mon.species in avail_switch_map:
                return self.create_order(avail_switch_map[mon.species])

        # Fallback to first available legal action
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])
        if battle.available_switches:
            return self.create_order(battle.available_switches[0])
        return self.choose_random_move(battle)

    def _battle_finished_callback(self, battle):
        """Track results after each finished battle."""
        if battle.won:
            result = "win"
        elif battle.lost:
            result = "loss"
        else:
            result = "tie"

        self._results.append({
            "id":     battle.battle_tag,
            "result": result,
            "turn":   battle.turn,
        })
        wins  = sum(1 for r in self._results if r["result"] == "win")
        total = len(self._results)
        print(
            f"[Battle {total:3d}] {result:4s} in {battle.turn:3d} turns | "
            f"W/L: {wins}/{total - wins} "
            f"({100 * wins / total:.0f}% win rate)"
        )
