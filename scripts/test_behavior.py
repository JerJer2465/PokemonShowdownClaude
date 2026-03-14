"""
Behavioral verification: does the trained model make correct decisions
in constructed scenarios where the right answer is obvious?

Tests:
1. Type effectiveness: Use super-effective move over neutral move
2. Status moves: Does it ever use Stealth Rock, Sleep Powder, etc?
3. Switching: Switch away from a terrible type matchup
4. Immunity avoidance: Don't use Electric on Ground type
5. STAB preference: Prefer STAB move over non-STAB of same power
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pokebot.env.obs_builder import ObsBuilder, FLOAT_DIM_PER_POKEMON
from pokebot.model.poke_transformer import PokeTransformer

# Load the best BC checkpoint
ckpt_path = "checkpoints/bc_smart_v3.pt"
if not os.path.exists(ckpt_path):
    ckpt_path = "checkpoints/bc_smart_v2.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PokeTransformer().to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded {ckpt_path} (update {ckpt.get('update', '?')}, WR {ckpt.get('win_rate', '?')})")

builder = ObsBuilder()


def make_mon(species, types, moves, hp=300, maxhp=300, status=None,
             boosts=None, volatile=None, is_active=True, is_own=True,
             base_stats=None):
    """Helper to create a pokemon dict."""
    if base_stats is None:
        base_stats = {"hp": 80, "attack": 80, "defense": 80,
                      "special-attack": 80, "special-defense": 80, "speed": 80}
    return {
        "species": species, "hp": hp, "maxhp": maxhp,
        "status": status, "boosts": boosts or {},
        "volatile_statuses": list(volatile or set()),
        "moves": moves,
        "types": types,
        "base_stats": base_stats,
        "ability": "overgrow" if is_own else None,
        "item": "leftovers" if is_own else None,
        "is_fainted": hp <= 0,
        "is_active": is_active,
    }


def make_move(name, bp, accuracy, mtype, category, priority=0, pp=15):
    return {"id": name, "basePower": bp, "accuracy": accuracy,
            "type": mtype, "category": category, "priority": priority,
            "pp": pp, "maxpp": pp, "is_known": True}


def make_state(own_active, opp_active, own_reserve=None, opp_reserve=None,
               legal_actions=None, weather="", trick_room=False):
    if own_reserve is None:
        own_reserve = [make_mon(f"reserve{i}", ["normal"],
                                [make_move("tackle", 40, 100, "Normal", "physical")]*4,
                                is_active=False)
                       for i in range(5)]
    if opp_reserve is None:
        opp_reserve = [make_mon(f"oppreserve{i}", ["normal"],
                                [make_move("tackle", 40, 100, "Normal", "physical")]*4,
                                is_active=False, is_own=False)
                       for i in range(5)]
    if legal_actions is None:
        n_moves = len([m for m in own_active["moves"] if m["basePower"] >= 0])
        legal_actions = list(range(n_moves)) + [4, 5, 6, 7, 8]

    return {
        "side_one": {"active": own_active, "reserve": own_reserve,
                     "hazards": {}, "screens": {}},
        "side_two": {"active": opp_active, "reserve": opp_reserve,
                     "hazards": {}, "screens": {}},
        "weather": weather, "weather_turns": 0,
        "terrain": "", "terrain_turns": 0,
        "trick_room": trick_room, "trick_room_turns": 0,
        "turn": 1,
        "legal_actions": legal_actions,
    }


def get_model_action(state, deterministic=True):
    enc = builder.encode(state)
    int_ids = torch.from_numpy(enc["int_ids"]).unsqueeze(0).to(device)
    float_f = torch.from_numpy(enc["float_feats"]).unsqueeze(0).to(device)
    legal_m = torch.from_numpy(enc["legal_mask"]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = model(int_ids, float_f, legal_m)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if deterministic:
        action = int(logits.argmax(dim=-1).item())
    else:
        action = int(torch.distributions.Categorical(logits=logits).sample().item())
    return action, probs


def run_test(name, state, expected_actions, n_trials=100):
    """Run the model n_trials times and report action distribution."""
    action_counts = np.zeros(10)
    for _ in range(n_trials):
        action, probs = get_model_action(state, deterministic=False)
        action_counts[action] += 1

    # Also get deterministic action
    det_action, probs = get_model_action(state, deterministic=True)

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Expected: actions {expected_actions}")
    print(f"Deterministic choice: action {det_action}")

    # Show move names for context
    moves = state["side_one"]["active"]["moves"]
    for i, m in enumerate(moves):
        pct = action_counts[i] / n_trials * 100
        prob = probs[i] * 100
        marker = " <<<" if i in expected_actions else ""
        print(f"  Move {i}: {m['id']:15s} ({m['type']:10s} bp={m['basePower']:3d}) "
              f"  prob={prob:5.1f}%  chosen={pct:5.1f}%{marker}")
    for i in range(4, 10):
        if action_counts[i] > 0:
            pct = action_counts[i] / n_trials * 100
            prob = probs[i] * 100
            marker = " <<<" if i in expected_actions else ""
            print(f"  Switch {i-4}:               "
                  f"  prob={prob:5.1f}%  chosen={pct:5.1f}%{marker}")

    correct = sum(action_counts[a] for a in expected_actions)
    pct_correct = correct / n_trials * 100
    passed = det_action in expected_actions
    status = "PASS" if passed else "FAIL"
    print(f"  Result: [{status}] correct={pct_correct:.0f}% "
          f"(det={'correct' if passed else 'WRONG'})")
    return passed


# ===== Test 1: Type Effectiveness (basic) =====
# Pikachu (Electric) vs Gyarados (Water/Flying)
# Thunderbolt is 4x effective, Surf is neutral
own = make_mon("pikachu", ["electric"], [
    make_move("thunderbolt", 90, 100, "Electric", "special"),
    make_move("surf", 90, 100, "Water", "special"),
    make_move("icebeam", 90, 100, "Ice", "special"),
    make_move("focusblast", 120, 70, "Fighting", "special"),
])
opp = make_mon("gyarados", ["water", "flying"], [
    make_move("waterfall", 80, 100, "Water", "physical"),
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("icefang", 65, 95, "Ice", "physical"),
    make_move("dragondance", 0, 0, "Dragon", "status"),
], is_own=False)
state = make_state(own, opp, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t1 = run_test("Type effectiveness: Thunderbolt (4x) vs Gyarados",
              state, expected_actions=[0])  # Thunderbolt

# ===== Test 2: Immunity Avoidance =====
# Pikachu vs Steelix (Steel/Ground) — Electric is immune
own2 = make_mon("pikachu", ["electric"], [
    make_move("thunderbolt", 90, 100, "Electric", "special"),
    make_move("surf", 90, 100, "Water", "special"),
    make_move("hiddenpowerice", 70, 100, "Ice", "special"),
    make_move("grassknot", 80, 100, "Grass", "special"),
])
opp2 = make_mon("steelix", ["steel", "ground"], [
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("stoneedge", 100, 80, "Rock", "physical"),
    make_move("gyroball", 0, 100, "Steel", "physical"),
    make_move("stealthrock", 0, 0, "Rock", "status"),
], is_own=False)
state2 = make_state(own2, opp2, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t2 = run_test("Immunity avoidance: Don't use Thunderbolt vs Ground",
              state2, expected_actions=[1])  # Surf (super effective on Ground)

# ===== Test 3: STAB Preference =====
# Charizard (Fire/Flying) has Fire Blast (STAB) vs Flamethrower (STAB) vs Dragon Pulse (no STAB)
# vs a neutral target. Should prefer highest effective power.
own3 = make_mon("charizard", ["fire", "flying"], [
    make_move("flamethrower", 90, 100, "Fire", "special"),    # STAB → 135 eff
    make_move("dragonpulse", 90, 100, "Dragon", "special"),   # no STAB → 90 eff
    make_move("airslash", 75, 95, "Flying", "special"),       # STAB → 112.5 eff
    make_move("hiddenpowergrass", 70, 100, "Grass", "special"), # no STAB → 70 eff
])
opp3 = make_mon("machamp", ["fighting"], [
    make_move("dynamicpunch", 100, 50, "Fighting", "physical"),
    make_move("payback", 50, 100, "Dark", "physical"),
    make_move("bulletpunch", 40, 100, "Steel", "physical"),
    make_move("stoneedge", 100, 80, "Rock", "physical"),
], is_own=False)
state3 = make_state(own3, opp3, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t3 = run_test("STAB preference: Flamethrower (STAB) vs neutral target",
              state3, expected_actions=[0])  # Flamethrower (STAB Fire vs Fighting)

# ===== Test 4: Status Move Usage =====
# Does the model ever pick Stealth Rock?
own4 = make_mon("swampert", ["water", "ground"], [
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("waterfall", 80, 100, "Water", "physical"),
    make_move("stealthrock", 0, 0, "Rock", "status"),    # SR!
    make_move("icepunch", 75, 100, "Ice", "physical"),
])
opp4 = make_mon("blissey", ["normal"], [  # Special wall, hard to kill
    make_move("seismictoss", 0, 100, "Normal", "physical"),
    make_move("toxic", 0, 90, "Poison", "status"),
    make_move("softboiled", 0, 0, "Normal", "status"),
    make_move("thunderwave", 0, 100, "Electric", "status"),
], is_own=False,
    base_stats={"hp": 255, "attack": 10, "defense": 10,
                "special-attack": 75, "special-defense": 105, "speed": 55})
state4 = make_state(own4, opp4, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t4 = run_test("Status move usage: Does it ever use Stealth Rock?",
              state4, expected_actions=[2])  # SR is correct here

# ===== Test 5: Sleep Move =====
own5 = make_mon("venusaur", ["grass", "poison"], [
    make_move("sleeppowder", 0, 75, "Grass", "status"),
    make_move("sludgebomb", 90, 100, "Poison", "special"),
    make_move("energyball", 80, 100, "Grass", "special"),
    make_move("synthesis", 0, 0, "Grass", "status"),
])
opp5 = make_mon("tyranitar", ["rock", "dark"], [
    make_move("crunch", 80, 100, "Dark", "physical"),
    make_move("stoneedge", 100, 80, "Rock", "physical"),
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("pursuit", 40, 100, "Dark", "physical"),
], is_own=False)
state5 = make_state(own5, opp5, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t5 = run_test("Sleep: Does it use Sleep Powder (strongest opening)?",
              state5, expected_actions=[0])

# ===== Test 6: Setup Move (Swords Dance/Dragon Dance) =====
own6 = make_mon("salamence", ["dragon", "flying"], [
    make_move("dragondance", 0, 0, "Dragon", "status"),    # Setup!
    make_move("outrage", 120, 100, "Dragon", "physical"),
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("firefang", 65, 95, "Fire", "physical"),
])
opp6 = make_mon("skarmory", ["steel", "flying"], [  # Physical wall
    make_move("bravebird", 120, 100, "Flying", "physical"),
    make_move("spikes", 0, 0, "Ground", "status"),
    make_move("roost", 0, 0, "Flying", "status"),
    make_move("whirlwind", 0, 0, "Normal", "status"),
], is_own=False,
    base_stats={"hp": 65, "attack": 80, "defense": 140,
                "special-attack": 40, "special-defense": 70, "speed": 70})
state6 = make_state(own6, opp6, legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t6 = run_test("Setup: Does it use Dragon Dance vs a wall?",
              state6, expected_actions=[0])

# ===== Test 7: Switching when type-disadvantaged =====
# Own active is Grass, opponent is Fire. Should switch to Water type.
water_mon = make_mon("vaporeon", ["water"], [
    make_move("surf", 90, 100, "Water", "special"),
    make_move("icebeam", 90, 100, "Ice", "special"),
    make_move("hiddenpowergrass", 70, 100, "Grass", "special"),
    make_move("wish", 0, 0, "Normal", "status"),
], is_active=False)

own7 = make_mon("tangela", ["grass"], [
    make_move("energyball", 80, 100, "Grass", "special"),
    make_move("hiddenpowerfire", 70, 100, "Fire", "special"),
    make_move("sleeppowder", 0, 75, "Grass", "status"),
    make_move("leechseed", 0, 90, "Grass", "status"),
])
opp7 = make_mon("arcanine", ["fire"], [
    make_move("flareblitz", 120, 100, "Fire", "physical"),
    make_move("extremespeed", 80, 100, "Normal", "physical"),
    make_move("wildcharge", 90, 100, "Electric", "physical"),
    make_move("closecombat", 120, 100, "Fighting", "physical"),
], is_own=False)

own_reserve7 = [water_mon] + [
    make_mon(f"reserve{i}", ["normal"],
             [make_move("tackle", 40, 100, "Normal", "physical")]*4,
             is_active=False)
    for i in range(4)]

state7 = make_state(own7, opp7, own_reserve=own_reserve7,
                    legal_actions=[0, 1, 2, 3, 4, 5, 6, 7, 8])
t7 = run_test("Switch to Water vs Fire opponent (or sleep it)",
              state7, expected_actions=[2, 4])  # Sleep Powder or switch to Vaporeon

# ===== Summary =====
tests = [t1, t2, t3, t4, t5, t6, t7]
names = [
    "Type effectiveness (4x)", "Immunity avoidance", "STAB preference",
    "Stealth Rock usage", "Sleep Powder usage", "Dragon Dance setup",
    "Switch to counter",
]
print("\n" + "=" * 60)
print("BEHAVIORAL TEST SUMMARY")
print("=" * 60)
for name, passed in zip(names, tests):
    print(f"  {'PASS' if passed else 'FAIL'}: {name}")
print(f"\n{sum(tests)}/{len(tests)} tests passed")

if sum(tests) < 3:
    print("\n!!! WARNING: Model fails most behavioral tests!")
    print("The BC teacher (smart_heuristic) never demonstrates:")
    print("  - Status moves (bp=0 → _estimate_damage returns 0)")
    print("  - Setup moves (Dragon Dance, Swords Dance, etc.)")
    print("  - Stealth Rock / hazard setup")
    print("PPO must learn these from scratch through exploration.")
elif sum(tests) < 5:
    print("\nModel shows partial understanding. PPO should improve further.")
else:
    print("\nModel demonstrates strong strategic play!")
