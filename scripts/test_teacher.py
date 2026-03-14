"""Test what the smart_heuristic_opponent chooses in key scenarios."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pokebot.env.poke_engine_env import smart_heuristic_opponent, _estimate_damage, _type_effectiveness

def make_move(name, bp, accuracy, mtype, category, priority=0, pp=15):
    return {"id": name, "basePower": bp, "accuracy": accuracy,
            "type": mtype, "category": category, "priority": priority,
            "pp": pp, "maxpp": pp, "is_known": True}

def make_mon(species, types, moves, hp=300, maxhp=300, base_stats=None):
    if base_stats is None:
        base_stats = {"hp": 80, "attack": 80, "defense": 80,
                      "special-attack": 80, "special-defense": 80, "speed": 80}
    return {"species": species, "hp": hp, "maxhp": maxhp,
            "status": None, "boosts": {}, "volatile_statuses": [],
            "moves": moves, "types": types, "base_stats": base_stats,
            "ability": "overgrow", "item": "leftovers",
            "is_fainted": False, "is_active": True}

# Test 1: Pikachu vs Gyarados
print("=== Test 1: Pikachu (Electric) vs Gyarados (Water/Flying) ===")
pikachu = make_mon("pikachu", ["electric"], [
    make_move("thunderbolt", 90, 100, "Electric", "special"),
    make_move("surf", 90, 100, "Water", "special"),
    make_move("icebeam", 90, 100, "Ice", "special"),
    make_move("focusblast", 120, 70, "Fighting", "special"),
])
gyarados = make_mon("gyarados", ["water", "flying"], [
    make_move("waterfall", 80, 100, "Water", "physical"),
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("icefang", 65, 95, "Ice", "physical"),
    make_move("dragondance", 0, 0, "Dragon", "status"),
])

# Show damage estimates for each move
for i, m in enumerate(pikachu["moves"]):
    dmg = _estimate_damage(m, pikachu, gyarados)
    eff = _type_effectiveness(m["type"], gyarados["types"])
    print(f"  Move {i}: {m['id']:15s} type={m['type']:10s} bp={m['basePower']:3d} "
          f"eff={eff:.1f}x  est_damage={dmg:.1f}")

obs_dict = {
    "side_one": {"active": pikachu, "reserve": []},
    "side_two": {"active": gyarados, "reserve": []},
    "legal_actions": [0, 1, 2, 3],
}
action = smart_heuristic_opponent(obs_dict)
print(f"  Teacher picks: action {action} ({pikachu['moves'][action]['id']})")

# Test 2: Pikachu vs Steelix (Ground immunity)
print("\n=== Test 2: Pikachu vs Steelix (Steel/Ground) ===")
steelix = make_mon("steelix", ["steel", "ground"], [
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("stoneedge", 100, 80, "Rock", "physical"),
    make_move("gyroball", 0, 100, "Steel", "physical"),
    make_move("stealthrock", 0, 0, "Rock", "status"),
])
pikachu2 = make_mon("pikachu", ["electric"], [
    make_move("thunderbolt", 90, 100, "Electric", "special"),
    make_move("surf", 90, 100, "Water", "special"),
    make_move("hiddenpowerice", 70, 100, "Ice", "special"),
    make_move("grassknot", 80, 100, "Grass", "special"),
])
for i, m in enumerate(pikachu2["moves"]):
    dmg = _estimate_damage(m, pikachu2, steelix)
    eff = _type_effectiveness(m["type"], steelix["types"])
    print(f"  Move {i}: {m['id']:15s} type={m['type']:10s} bp={m['basePower']:3d} "
          f"eff={eff:.1f}x  est_damage={dmg:.1f}")

obs_dict2 = {
    "side_one": {"active": pikachu2, "reserve": []},
    "side_two": {"active": steelix, "reserve": []},
    "legal_actions": [0, 1, 2, 3],
}
action2 = smart_heuristic_opponent(obs_dict2)
print(f"  Teacher picks: action {action2} ({pikachu2['moves'][action2]['id']})")

# Test 3: Status moves
print("\n=== Test 3: Swampert vs Blissey (status move scenario) ===")
swampert = make_mon("swampert", ["water", "ground"], [
    make_move("earthquake", 100, 100, "Ground", "physical"),
    make_move("waterfall", 80, 100, "Water", "physical"),
    make_move("stealthrock", 0, 0, "Rock", "status"),
    make_move("icepunch", 75, 100, "Ice", "physical"),
])
blissey = make_mon("blissey", ["normal"], [
    make_move("seismictoss", 0, 100, "Normal", "physical"),
    make_move("toxic", 0, 90, "Poison", "status"),
    make_move("softboiled", 0, 0, "Normal", "status"),
    make_move("thunderwave", 0, 100, "Electric", "status"),
], base_stats={"hp": 255, "attack": 10, "defense": 10,
               "special-attack": 75, "special-defense": 105, "speed": 55})
for i, m in enumerate(swampert["moves"]):
    dmg = _estimate_damage(m, swampert, blissey)
    print(f"  Move {i}: {m['id']:15s} bp={m['basePower']:3d} est_damage={dmg:.1f}")

obs_dict3 = {
    "side_one": {"active": swampert, "reserve": []},
    "side_two": {"active": blissey, "reserve": []},
    "legal_actions": [0, 1, 2, 3],
}
action3 = smart_heuristic_opponent(obs_dict3)
print(f"  Teacher picks: action {action3} ({swampert['moves'][action3]['id']})")
print(f"  (Stealth Rock would be optimal but has bp=0, so teacher always ignores it)")

# Test type effectiveness in STAB case
print("\n=== Type effectiveness checks ===")
cases = [
    ("Electric", ["water", "flying"], "Pikachu Tbolt vs Gyarados"),
    ("Electric", ["steel", "ground"], "Pikachu Tbolt vs Steelix"),
    ("Water", ["steel", "ground"], "Pikachu Surf vs Steelix"),
    ("Ice", ["dragon", "flying"], "Ice vs Dragonite"),
    ("Grass", ["water", "ground"], "Grass vs Swampert"),
    ("Fire", ["grass"], "Fire vs Grass"),
]
for atk_type, def_types, desc in cases:
    eff = _type_effectiveness(atk_type, def_types)
    print(f"  {desc}: {atk_type} vs {def_types} = {eff}x")
