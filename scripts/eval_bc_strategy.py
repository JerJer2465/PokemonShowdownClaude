"""
BC Strategy Audit: Does the model understand key Pokemon battle concepts?

Plays 500+ games, intercepting every decision to record what the model chose
and whether it made the "right" call in identifiable strategic situations.

Tests:
  1. Type effectiveness — does it pick super-effective moves over neutral?
  2. STAB preference — does it prefer same-type-attack-bonus moves?
  3. Physical vs Special — does it pick the right category for the matchup?
  4. Status moves — does it use Thunder Wave, Toxic, Will-O-Wisp, Stealth Rock?
  5. Switching — does it switch when at a type disadvantage?
  6. Low HP behavior — does it switch or use priority when low?
  7. Setup moves — does it use Swords Dance, Calm Mind, etc.?
  8. Priority moves — does it use priority to finish low-HP opponents?
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

from config.model_config import MODEL_CONFIG
from pokebot.env.poke_engine_env import (
    PokeEngineEnv, _action_to_move_str, _build_legal_mask_from_state,
)
from pokebot.model.poke_transformer import PokeTransformer

# ── Type chart (Gen 4) ──────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
with open(DATA_DIR / "gen4_move_data.json") as f:
    MOVE_DATA = json.load(f)

TYPE_CHART = {
    # effectiveness[attacking_type][defending_type] = multiplier
    "Normal":   {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire":     {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass":    {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
    "Ice":      {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2},
    "Poison":   {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": 0.5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
    "Bug":      {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Psychic": 2, "Ghost": 0.5, "Dark": 2, "Steel": 0.5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5},
    "Dragon":   {"Dragon": 2, "Steel": 0.5},
    "Dark":     {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Steel": 0.5},
    "Steel":    {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5},
}


def _type_effectiveness(move_type: str, def_types: list[str]) -> float:
    """Compute type effectiveness multiplier. Case-insensitive."""
    mult = 1.0
    # Normalize to title case for TYPE_CHART lookup
    mt = move_type.strip().title()
    chart = TYPE_CHART.get(mt, {})
    for dt in def_types:
        dt_title = dt.strip().title()
        mult *= chart.get(dt_title, 1.0)
    return mult


def _get_move_info(move_id: str) -> dict:
    return MOVE_DATA.get(move_id, {})


class StrategyTracker:
    """Tracks strategic decision quality across many games."""

    def __init__(self):
        # Type effectiveness
        self.se_available = 0       # times a SE move was available
        self.se_chosen = 0          # times model picked a SE move
        self.se_best_chosen = 0     # times model picked the BEST SE move
        self.immune_avoided = 0     # times model avoided immune moves
        self.immune_available = 0   # times immune move existed among options

        # STAB
        self.stab_available = 0
        self.stab_chosen = 0

        # Physical vs Special (exploit lower defense)
        self.phys_correct = 0
        self.spec_correct = 0
        self.category_total = 0

        # Status moves
        self.status_used = defaultdict(int)       # move_id → count
        self.status_available = defaultdict(int)   # move_id → count

        # Hazards
        self.hazard_set_available = 0  # times SR/Spikes available and not yet up
        self.hazard_set_chosen = 0

        # Setup moves (swords dance, calm mind, dragon dance, nasty plot, agility, etc.)
        self.setup_available = 0
        self.setup_chosen = 0
        self.setup_at_full_hp = 0
        self.setup_chosen_at_full = 0

        # Switching
        self.switch_when_disadvantaged = 0  # switched when at type disadvantage
        self.disadvantaged_total = 0
        self.switch_when_low_hp = 0
        self.low_hp_total = 0

        # Priority
        self.priority_available_vs_low = 0  # priority move available AND opp is low
        self.priority_chosen_vs_low = 0

        # Overall
        self.total_decisions = 0
        self.move_decisions = 0
        self.switch_decisions = 0

    def report(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("BC STRATEGY AUDIT REPORT")
        lines.append("=" * 70)

        def pct(n, d):
            return f"{n}/{d} ({n/d*100:.1f}%)" if d > 0 else "N/A"

        lines.append(f"\nTotal decisions: {self.total_decisions}")
        lines.append(f"  Move decisions:   {self.move_decisions}")
        lines.append(f"  Switch decisions: {self.switch_decisions}")

        lines.append(f"\n{'─'*50}")
        lines.append("1. TYPE EFFECTIVENESS")
        lines.append(f"  Picked SE move when available:      {pct(self.se_chosen, self.se_available)}")
        lines.append(f"  Picked BEST SE move:                {pct(self.se_best_chosen, self.se_available)}")
        lines.append(f"  Avoided immune moves:               {pct(self.immune_avoided, self.immune_available)}")

        lines.append(f"\n{'─'*50}")
        lines.append("2. STAB (Same-Type Attack Bonus)")
        lines.append(f"  Picked STAB move when available:    {pct(self.stab_chosen, self.stab_available)}")

        lines.append(f"\n{'─'*50}")
        lines.append("3. PHYSICAL vs SPECIAL (exploit weaker def)")
        lines.append(f"  Correct category chosen:            {pct(self.phys_correct + self.spec_correct, self.category_total)}")

        lines.append(f"\n{'─'*50}")
        lines.append("4. STATUS MOVES")
        for move_id in sorted(self.status_available, key=lambda m: -self.status_available[m]):
            used = self.status_used[move_id]
            avail = self.status_available[move_id]
            lines.append(f"  {move_id:20s} used {pct(used, avail)}")

        lines.append(f"\n{'─'*50}")
        lines.append("5. ENTRY HAZARDS (SR/Spikes when not up)")
        lines.append(f"  Set hazards when available:          {pct(self.hazard_set_chosen, self.hazard_set_available)}")

        lines.append(f"\n{'─'*50}")
        lines.append("6. SETUP MOVES (SD, CM, DD, NP, Agility, etc.)")
        lines.append(f"  Used setup when available:           {pct(self.setup_chosen, self.setup_available)}")
        lines.append(f"  Used setup at full HP:               {pct(self.setup_chosen_at_full, self.setup_at_full_hp)}")

        lines.append(f"\n{'─'*50}")
        lines.append("7. SWITCHING BEHAVIOR")
        lines.append(f"  Switched when type-disadvantaged:    {pct(self.switch_when_disadvantaged, self.disadvantaged_total)}")
        lines.append(f"  Switched when low HP (<25%):         {pct(self.switch_when_low_hp, self.low_hp_total)}")

        lines.append(f"\n{'─'*50}")
        lines.append("8. PRIORITY MOVES (vs low-HP opponent)")
        lines.append(f"  Used priority to finish low opp:     {pct(self.priority_chosen_vs_low, self.priority_available_vs_low)}")

        lines.append(f"\n{'='*70}")
        return "\n".join(lines)


SETUP_MOVES = {
    "swordsdance", "calmmind", "dragondance", "nastyplot", "agility",
    "rockpolish", "bulkup", "curse", "workup", "howl", "bellydrum",
    "growth", "tailglow", "quiverdance", "coil", "shellsmash",
}

HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes"}

STATUS_MOVES_TRACK = {
    "thunderwave", "toxic", "willowisp", "spore", "sleeppowder",
    "stunspore", "hypnosis", "yawn", "glare", "stealthrock", "spikes",
    "toxicspikes", "reflect", "lightscreen",
}

PRIORITY_MOVES = {
    "extremespeed", "machpunch", "bulletpunch", "iceshard", "aquajet",
    "suckerpunch", "quickattack", "shadowsneak", "fakeout", "vacuumwave",
}


def analyze_decision(
    tracker: StrategyTracker,
    obs_dict: dict,
    action: int,
    state,
    probs: np.ndarray,
):
    """Analyze a single model decision for strategic quality."""
    tracker.total_decisions += 1

    side = state.side_one
    active_idx = int(str(side.active_index))
    my_active = obs_dict["side_one"]["active"]
    opp_active = obs_dict["side_two"]["active"]

    my_types = my_active.get("types", [])
    opp_types = opp_active.get("types", [])
    my_hp_frac = my_active["hp"] / max(my_active["maxhp"], 1)
    opp_hp_frac = opp_active["hp"] / max(opp_active["maxhp"], 1)

    # Get available moves with metadata
    legal = _build_legal_mask_from_state(side)
    legal_moves = [a for a in legal if a < 4]
    legal_switches = [a for a in legal if a >= 4]

    moves_info = []
    for i in legal_moves:
        if i < len(my_active["moves"]):
            m = my_active["moves"][i]
            mi = _get_move_info(m["id"])
            move_type = mi.get("type", m.get("type", "Normal"))
            eff = _type_effectiveness(move_type, opp_types)
            is_stab = move_type.lower() in [t.lower() for t in my_types]
            bp = mi.get("basePower", m.get("basePower", 0))
            cat = mi.get("category", m.get("category", "physical"))
            pri = mi.get("priority", m.get("priority", 0))
            moves_info.append({
                "idx": i, "id": m["id"], "type": move_type, "bp": bp,
                "cat": cat, "eff": eff, "stab": is_stab, "priority": pri,
            })

    is_move = action < 4
    is_switch = action >= 4

    if is_move:
        tracker.move_decisions += 1
    else:
        tracker.switch_decisions += 1

    chosen_move = None
    if is_move:
        for mi in moves_info:
            if mi["idx"] == action:
                chosen_move = mi
                break

    # ── 1. Type Effectiveness ──
    se_moves = [m for m in moves_info if m["eff"] > 1.0 and m["bp"] > 0]
    if se_moves:
        tracker.se_available += 1
        if chosen_move and chosen_move["eff"] > 1.0:
            tracker.se_chosen += 1
        best_eff = max(m["eff"] for m in se_moves)
        if chosen_move and chosen_move["eff"] == best_eff:
            tracker.se_best_chosen += 1

    immune_moves = [m for m in moves_info if m["eff"] == 0 and m["bp"] > 0]
    if immune_moves:
        tracker.immune_available += 1
        if not chosen_move or chosen_move["eff"] != 0:
            tracker.immune_avoided += 1

    # ── 2. STAB ──
    stab_moves = [m for m in moves_info if m["stab"] and m["bp"] > 0]
    non_stab_moves = [m for m in moves_info if not m["stab"] and m["bp"] > 0]
    if stab_moves and non_stab_moves:  # only count when there's a real choice
        tracker.stab_available += 1
        if chosen_move and chosen_move["stab"]:
            tracker.stab_chosen += 1

    # ── 3. Physical vs Special ──
    opp_def = opp_active.get("base_stats", {}).get("defense", 80)
    opp_spd = opp_active.get("base_stats", {}).get("special-defense", 80)
    if chosen_move and chosen_move["bp"] > 0:
        tracker.category_total += 1
        if chosen_move["cat"] == "physical" and opp_def <= opp_spd:
            tracker.phys_correct += 1
        elif chosen_move["cat"] == "special" and opp_spd <= opp_def:
            tracker.spec_correct += 1
        elif opp_def == opp_spd:
            # Equal defenses — either is fine
            tracker.phys_correct += 1

    # ── 4. Status Moves ──
    for mi in moves_info:
        if mi["id"] in STATUS_MOVES_TRACK:
            tracker.status_available[mi["id"]] += 1
    if chosen_move and chosen_move["id"] in STATUS_MOVES_TRACK:
        tracker.status_used[chosen_move["id"]] += 1

    # ── 5. Hazards ──
    hazards_own = obs_dict["side_two"].get("hazards", {})  # on opponent's side
    opp_has_sr = hazards_own.get("stealth_rock", False)
    opp_spikes = hazards_own.get("spikes", 0)
    hazard_available = [m for m in moves_info if m["id"] in HAZARD_MOVES]
    unset_hazards = []
    for m in hazard_available:
        if m["id"] == "stealthrock" and not opp_has_sr:
            unset_hazards.append(m)
        elif m["id"] == "spikes" and opp_spikes < 3:
            unset_hazards.append(m)
        elif m["id"] == "toxicspikes":
            unset_hazards.append(m)
    if unset_hazards:
        tracker.hazard_set_available += 1
        if chosen_move and chosen_move["id"] in HAZARD_MOVES:
            tracker.hazard_set_chosen += 1

    # ── 6. Setup Moves ──
    setup_available = [m for m in moves_info if m["id"] in SETUP_MOVES]
    if setup_available:
        tracker.setup_available += 1
        if chosen_move and chosen_move["id"] in SETUP_MOVES:
            tracker.setup_chosen += 1
        if my_hp_frac > 0.95:
            tracker.setup_at_full_hp += 1
            if chosen_move and chosen_move["id"] in SETUP_MOVES:
                tracker.setup_chosen_at_full += 1

    # ── 7. Switching ──
    # Type disadvantage: opponent has SE move types against us
    # Simplified: check if opp's types are SE against our types
    opp_se_against_us = any(
        _type_effectiveness(ot, my_types) > 1.0 for ot in opp_types
    )
    my_se_against_opp = any(
        _type_effectiveness(mt, opp_types) > 1.0 for mt in my_types
    )
    type_disadvantaged = opp_se_against_us and not my_se_against_opp
    if type_disadvantaged and legal_switches:
        tracker.disadvantaged_total += 1
        if is_switch:
            tracker.switch_when_disadvantaged += 1

    if my_hp_frac < 0.25 and my_hp_frac > 0 and legal_switches:
        tracker.low_hp_total += 1
        if is_switch:
            tracker.switch_when_low_hp += 1

    # ── 8. Priority ──
    priority_moves = [m for m in moves_info if m["priority"] > 0 and m["bp"] > 0]
    if priority_moves and opp_hp_frac < 0.20 and opp_hp_frac > 0:
        tracker.priority_available_vs_low += 1
        if chosen_move and chosen_move["priority"] > 0:
            tracker.priority_chosen_vs_low += 1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc_smart_v4.pt")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = PokeTransformer().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint}  (update={ckpt.get('update','?')}, wr={ckpt.get('win_rate','?')})")

    def _random_opp(obs_dict):
        legal = obs_dict.get("legal_actions", list(range(10)))
        return random.choice(legal)

    tracker = StrategyTracker()
    env = PokeEngineEnv(opponent_policy=_random_opp)
    wins = 0

    for game in range(args.games):
        obs, _ = env.reset()
        done = False
        while not done:
            int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(device)
            float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(device)
            legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(device)

            with torch.no_grad():
                log_probs, _, _ = model(int_ids, float_f, legal_m)

            probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
            action = int(log_probs.argmax(dim=-1).item())

            # Analyze this decision
            obs_dict = env._build_obs_dict()
            analyze_decision(tracker, obs_dict, action, env._state, probs)

            obs, reward, done, _, _ = env.step(action)

        if reward > 0:
            wins += 1

        if (game + 1) % 100 == 0:
            print(f"  {game+1}/{args.games} games  WR={wins/(game+1)*100:.1f}%")

    print(f"\nOverall: {wins}/{args.games} = {wins/args.games*100:.1f}% WR vs random\n")
    print(tracker.report())


if __name__ == "__main__":
    main()
