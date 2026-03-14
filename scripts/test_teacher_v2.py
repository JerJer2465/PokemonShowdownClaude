"""Test upgraded smart_heuristic_opponent with status move support."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from pokebot.env.poke_engine_env import (
    PokeEngineEnv, smart_heuristic_opponent, _random_opponent, _estimate_damage
)

def make_move(name, bp, accuracy, mtype, category, priority=0, pp=15):
    return {"id": name, "basePower": bp, "accuracy": accuracy,
            "type": mtype, "category": category, "priority": priority,
            "pp": pp, "maxpp": pp, "is_known": True}

def make_mon(species, types, moves, hp=300, maxhp=300, base_stats=None, status=None):
    if base_stats is None:
        base_stats = {"hp": 80, "attack": 80, "defense": 80,
                      "special-attack": 80, "special-defense": 80, "speed": 80}
    return {"species": species, "hp": hp, "maxhp": maxhp,
            "status": status, "boosts": {}, "volatile_statuses": [],
            "moves": moves, "types": types, "base_stats": base_stats,
            "ability": "overgrow", "item": "leftovers",
            "is_fainted": False, "is_active": True}

# Test 1: Should use Stealth Rock early
print("=== Test 1: Swampert vs Blissey turn 1 ===")
swampert = make_mon("swampert", ["water", "ground"], [
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("waterfall", 80, 100, "Water", "physical"),
    make_move("stealthrock", 0, 0, "Rock", "status"),
    make_move("icepunch", 75, 100, "Ice", "physical"),
])
blissey = make_mon("blissey", ["normal"], [
    make_move("seismictoss", 0, 100, "Normal", "physical"),
]*4, base_stats={"hp": 255, "attack": 10, "defense": 10,
    "special-attack": 75, "special-defense": 105, "speed": 55})
obs = {"side_one": {"active": swampert, "reserve": [], "hazards": {}},
       "side_two": {"active": blissey, "reserve": [], "hazards": {}},
       "legal_actions": [0, 1, 2, 3], "turn": 1}
action = smart_heuristic_opponent(obs)
print(f"  Picks: {swampert['moves'][action]['id']} "
      f"({'CORRECT: SR!' if action == 2 else 'WRONG: should SR'})")

# Test 2: Should use Sleep Powder
print("\n=== Test 2: Venusaur vs Tyranitar ===")
venus = make_mon("venusaur", ["grass", "poison"], [
    make_move("sleeppowder", 0, 75, "Grass", "status"),
    make_move("sludgebomb", 90, 100, "Poison", "special"),
    make_move("energyball", 80, 100, "Grass", "special"),
    make_move("synthesis", 0, 0, "Grass", "status"),
])
ttar = make_mon("tyranitar", ["rock", "dark"], [
    make_move("crunch", 80, 100, "Dark", "physical"),
]*4, base_stats={"hp": 100, "attack": 134, "defense": 110,
    "special-attack": 95, "special-defense": 100, "speed": 61})
obs2 = {"side_one": {"active": venus, "reserve": [], "hazards": {}},
        "side_two": {"active": ttar, "reserve": [], "hazards": {}},
        "legal_actions": [0, 1, 2, 3], "turn": 1}
action2 = smart_heuristic_opponent(obs2)
print(f"  Picks: {venus['moves'][action2]['id']} "
      f"({'CORRECT: Sleep Powder!' if action2 == 0 else 'WRONG'})")

# Test 3: Should Dragon Dance vs wall
print("\n=== Test 3: Salamence vs Skarmory ===")
sala = make_mon("salamence", ["dragon", "flying"], [
    make_move("dragondance", 0, 0, "Dragon", "status"),
    make_move("outrage", 120, 100, "Dragon", "physical"),
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("firefang", 65, 95, "Fire", "physical"),
], base_stats={"hp": 95, "attack": 135, "defense": 80,
    "special-attack": 110, "special-defense": 80, "speed": 100})
skarm = make_mon("skarmory", ["steel", "flying"], [
    make_move("bravebird", 120, 100, "Flying", "physical"),
]*4, base_stats={"hp": 65, "attack": 80, "defense": 140,
    "special-attack": 40, "special-defense": 70, "speed": 70})
obs3 = {"side_one": {"active": sala, "reserve": [], "hazards": {}},
        "side_two": {"active": skarm, "reserve": [], "hazards": {}},
        "legal_actions": [0, 1, 2, 3], "turn": 2}
action3 = smart_heuristic_opponent(obs3)
# Fire Fang is 2x effective (Fire vs Steel), EQ hits Steel 2x but Flying immune
# Dragon Dance scores 200 at turn 2. Fire Fang: 65 * (135/140) * 1.0 * 2.0 * 0.95 = ~119
# Outrage: 120 * (135/80) * 1.5 * 0.5 = ~152 (Dragon vs Steel=0.5)
# EQ: 100 * (135/140) * 1.0 * 0.0 = 0 (Flying immune to Ground!)
print(f"  Picks: {sala['moves'][action3]['id']} "
      f"({'CORRECT: DD!' if action3 == 0 else sala['moves'][action3]['id']})")

# Test 4: Thunder Wave vs fast threat
print("\n=== Test 4: Blissey vs Starmie (fast special attacker) ===")
blissey2 = make_mon("blissey", ["normal"], [
    make_move("thunderwave", 0, 100, "Electric", "status"),
    make_move("seismictoss", 0, 100, "Normal", "physical"),
    make_move("softboiled", 0, 0, "Normal", "status"),
    make_move("toxic", 0, 90, "Poison", "status"),
], base_stats={"hp": 255, "attack": 10, "defense": 10,
    "special-attack": 75, "special-defense": 105, "speed": 55})
starmie = make_mon("starmie", ["water", "psychic"], [
    make_move("surf", 90, 100, "Water", "special"),
]*4, base_stats={"hp": 60, "attack": 75, "defense": 85,
    "special-attack": 100, "special-defense": 85, "speed": 115})
obs4 = {"side_one": {"active": blissey2, "reserve": [], "hazards": {}},
        "side_two": {"active": starmie, "reserve": [], "hazards": {}},
        "legal_actions": [0, 1, 2, 3], "turn": 1}
action4 = smart_heuristic_opponent(obs4)
print(f"  Picks: {blissey2['moves'][action4]['id']} "
      f"(TWave or Toxic both good: {'OK' if action4 in [0, 3] else 'WRONG'})")

# Test 5: Don't use status on already-statused opponent
print("\n=== Test 5: Blissey vs already-paralyzed Starmie ===")
starmie_par = make_mon("starmie", ["water", "psychic"], [
    make_move("surf", 90, 100, "Water", "special"),
]*4, status="par", base_stats={"hp": 60, "attack": 75, "defense": 85,
    "special-attack": 100, "special-defense": 85, "speed": 115})
obs5 = {"side_one": {"active": blissey2, "reserve": [], "hazards": {}},
        "side_two": {"active": starmie_par, "reserve": [], "hazards": {}},
        "legal_actions": [0, 1, 2, 3], "turn": 5}
action5 = smart_heuristic_opponent(obs5)
print(f"  Picks: {blissey2['moves'][action5]['id']} "
      f"(Should NOT Thunder Wave: {'OK' if action5 != 0 else 'WRONG: redundant TWave'})")

# Test 6: WR of improved heuristic vs random
print("\n=== Win Rate Test: Smart Heuristic v2 vs Random (200 games) ===")
env = PokeEngineEnv(opponent_policy=_random_opponent)
wins = 0
for g in range(200):
    obs_dict, _ = env.reset()
    done = False
    while not done:
        legal = obs_dict["legal_mask"]
        legal_actions = [i for i in range(10) if legal[i] > 0.5]
        # Build raw obs_dict for smart_heuristic
        raw = env._build_obs_dict(perspective="side_one")
        action = smart_heuristic_opponent(raw)
        obs_dict, reward, done, _, info = env.step(action)
    if reward > 0:
        wins += 1
wr = wins / 200
print(f"  Win rate: {wr*100:.1f}% vs random (200 games)")
print(f"  (Previous v1 heuristic was ~91%)")
