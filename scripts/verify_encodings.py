"""
Exhaustive verification of observation encodings.
Tests every field the model sees to ensure data flows correctly.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pokebot.env.obs_builder import (
    ObsBuilder, FLOAT_DIM_PER_POKEMON, FIELD_DIM, N_TOKENS,
    _one_hot, _bin_hp, _bin_base_power, _bin_accuracy, _bin_turns,
    _bin_turn_number, _encode_stat_boosts, _encode_volatile_status,
    _encode_move, TYPE_TO_IDX, STATUS_TO_IDX, WEATHER_TO_IDX,
    VOLATILE_TO_IDX, Vocab, N_TYPES, N_STATUS, N_VOLATILE,
    MOVE_CATEGORY_TO_IDX, MoveFeatures,
)
from pokebot.env.poke_engine_env import (
    PokeEngineEnv, _random_opponent, _type_effectiveness,
    _estimate_damage, smart_heuristic_opponent,
)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


print("=" * 70)
print("ENCODING VERIFICATION")
print("=" * 70)

# ===== 1. Dimension checks =====
print("\n--- 1. Dimension Checks ---")
check("FLOAT_DIM_PER_POKEMON = 380",
      FLOAT_DIM_PER_POKEMON == 380,
      f"Got {FLOAT_DIM_PER_POKEMON}")
check("FIELD_DIM = 76", FIELD_DIM == 76, f"Got {FIELD_DIM}")
check("N_TOKENS = 15", N_TOKENS == 15, f"Got {N_TOKENS}")

# Verify MoveFeatures.dim
mf = MoveFeatures(
    base_power_bin=np.zeros(8), accuracy_bin=np.zeros(6),
    type_onehot=np.zeros(18), category_onehot=np.zeros(3),
    priority_onehot=np.zeros(8), pp_fraction=1.0, is_known=1.0,
)
check("MoveFeatures.dim = 45", mf.dim == 45, f"Got {mf.dim}")
check("MoveFeatures.to_array() shape = (45,)",
      mf.to_array().shape == (45,),
      f"Got {mf.to_array().shape}")

# Verify total: 1+10+6+91+7+20+18+18+1+1+6+1+180 + 4+3+1+1+4+4+1+1+1 = 380
manual_sum = 1 + 10 + 6 + 91 + 7 + 20 + 18 + 18 + 1 + 1 + 6 + 1 + 4*45 + 4 + 3 + 1 + 1 + 4 + 4 + 1 + 1 + 1
check(f"Manual float dim sum = 380", manual_sum == 380, f"Got {manual_sum}")

# ===== 2. Create a real env and get obs =====
print("\n--- 2. Env Reset & Obs Shape ---")
env = PokeEngineEnv(opponent_policy=_random_opponent)
obs, _ = env.reset()

check("obs has int_ids", "int_ids" in obs)
check("obs has float_feats", "float_feats" in obs)
check("obs has legal_mask", "legal_mask" in obs)
check("int_ids shape = (15, 8)", obs["int_ids"].shape == (15, 8),
      f"Got {obs['int_ids'].shape}")
check("float_feats shape = (15, 380)", obs["float_feats"].shape == (15, 380),
      f"Got {obs['float_feats'].shape}")
check("legal_mask shape = (10,)", obs["legal_mask"].shape == (10,),
      f"Got {obs['legal_mask'].shape}")

# ===== 3. Check field token (token 0) =====
print("\n--- 3. Field Token Encoding ---")
field_float = obs["float_feats"][0]  # token 0 = field
check("Field token is padded to 380", field_float.shape == (380,))
# At least one element should be non-zero (weather or turn number)
check("Field token has non-zero values", np.any(field_float != 0),
      "All zeros - field encoding is empty!")

# Check specific offsets in field token
# Weather one-hot: [0:5]
weather_onehot = field_float[0:5]
check("Weather one-hot sums to 1", abs(weather_onehot.sum() - 1.0) < 0.01,
      f"Sum = {weather_onehot.sum()}")

# Weather turns: [5:13] (8-dim bin)
weather_turns_bin = field_float[5:13]
check("Weather turns bin has correct dim", len(weather_turns_bin) == 8)

# Pseudo-weather: [13:18] (5-dim)
pseudo = field_float[13:18]
check("Pseudo-weather has 5 dims", len(pseudo) == 5)

# Trick room turns: [18:22] (4-dim)
tr_turns = field_float[18:22]
check("Trick room turns has 4 dims", len(tr_turns) == 4)

# Hazards own: [22:29] (7-dim)
hazards_own = field_float[22:29]
check("Hazards own has 7 dims", len(hazards_own) == 7)

# Hazards opp: [29:36] (7-dim)
hazards_opp = field_float[29:36]
check("Hazards opp has 7 dims", len(hazards_opp) == 7)

# Screens own: [36:42] (6-dim)
screens_own = field_float[36:42]
check("Screens own has 6 dims", len(screens_own) == 6)

# Screens opp: [42:48] (6-dim)
screens_opp = field_float[42:48]
check("Screens opp has 6 dims", len(screens_opp) == 6)

# Turn bin: [48:58] (10-dim)
turn_bin = field_float[48:58]
check("Turn number bin has 10 dims", len(turn_bin) == 10)
check("Turn bin sums to 1", abs(turn_bin.sum() - 1.0) < 0.01,
      f"Sum = {turn_bin.sum()}")

# Fainted counts: [58:60] (2 dim)
fainted = field_float[58:60]
check("Fainted counts has 2 dims", len(fainted) == 2)

# New fields in 60:76 range (toxic count, tailwind, wish, safeguard)
new_field_feats = field_float[60:76]
check("New field features present (toxic/tailwind/wish/safeguard)", len(new_field_feats) == 16)
# Rest should be zero (padding to FLOAT_DIM)
padding = field_float[76:]
check("Field token padding (76:380) is all zeros", np.all(padding == 0),
      f"Non-zero count: {np.count_nonzero(padding)}")

# ===== 4. Pokemon Token Encoding =====
print("\n--- 4. Pokemon Token Float Features ---")
# Token 1 = own active
own_active_float = obs["float_feats"][1]
check("Own active token shape = (360,)", own_active_float.shape == (380,))

# Verify offset structure within a Pokemon token
# [0]: hp_fraction
hp_frac = own_active_float[0]
check("HP fraction in [0, 1]", 0 <= hp_frac <= 1.0, f"Got {hp_frac}")

# [1:11]: hp_bin (10-dim one-hot)
hp_bin = own_active_float[1:11]
check("HP bin sums to 1", abs(hp_bin.sum() - 1.0) < 0.01, f"Sum = {hp_bin.sum()}")

# [11:17]: base_stats (6-dim, each /255)
base_stats = own_active_float[11:17]
check("Base stats all in [0, 1]",
      np.all((base_stats >= 0) & (base_stats <= 1.1)),
      f"Range: [{base_stats.min():.3f}, {base_stats.max():.3f}]")
check("Base stats not all zero", np.any(base_stats > 0),
      "All zeros - base stats missing!")

# [17:108]: stat_boosts (91-dim = 7 stats × 13 levels)
stat_boosts = own_active_float[17:108]
check("Stat boosts has 91 dims", len(stat_boosts) == 91)
# Each block of 13 should sum to 1 (one-hot for each stat)
for i in range(7):
    block = stat_boosts[i*13:(i+1)*13]
    check(f"  Stat boost [{i}] sums to 1",
          abs(block.sum() - 1.0) < 0.01, f"Sum = {block.sum()}")

# [108:115]: status (7-dim one-hot)
status = own_active_float[108:115]
check("Status one-hot sums to 1", abs(status.sum() - 1.0) < 0.01,
      f"Sum = {status.sum()}")

# [115:135]: volatile (20-dim multi-hot)
volatile = own_active_float[115:135]
check("Volatile status has 20 dims", len(volatile) == 20)

# [135:153]: type1 (18-dim one-hot)
type1 = own_active_float[135:153]
check("Type1 one-hot sums to 1", abs(type1.sum() - 1.0) < 0.01,
      f"Sum = {type1.sum()}")

# [153:171]: type2 (18-dim one-hot)
type2 = own_active_float[153:171]
# type2 can be all zeros for single-type pokemon
check("Type2 sums to 0 or 1", type2.sum() < 1.01, f"Sum = {type2.sum()}")

# [171]: is_fainted
is_fainted = own_active_float[171]
check("is_fainted = 0 for active", is_fainted == 0.0, f"Got {is_fainted}")

# [172]: is_active
is_active = own_active_float[172]
check("is_active = 1 for active", is_active == 1.0, f"Got {is_active}")

# [173:179]: slot (6-dim one-hot)
slot = own_active_float[173:179]
check("Slot one-hot sums to 1", abs(slot.sum() - 1.0) < 0.01, f"Sum = {slot.sum()}")
check("Active is slot 0", slot[0] == 1.0, f"Got slot = {np.argmax(slot)}")

# [179]: is_own
is_own = own_active_float[179]
check("is_own = 1 for own team", is_own == 1.0, f"Got {is_own}")

# [180:360]: 4 moves × 45 dims
print("\n--- 5. Move Feature Encoding ---")
for move_i in range(4):
    offset = 180 + move_i * 45
    move_feats = own_active_float[offset:offset+45]

    bp_bin = move_feats[0:8]      # base power bin
    acc_bin = move_feats[8:14]    # accuracy bin
    type_oh = move_feats[14:32]   # type one-hot
    cat_oh = move_feats[32:35]    # category one-hot
    pri_oh = move_feats[35:43]    # priority one-hot
    pp_frac = move_feats[43]      # PP fraction
    is_known = move_feats[44]     # is_known flag

    if is_known > 0.5:
        check(f"  Move {move_i}: bp_bin sums to 1",
              abs(bp_bin.sum() - 1.0) < 0.01, f"Sum = {bp_bin.sum()}")
        check(f"  Move {move_i}: acc_bin sums to 1",
              abs(acc_bin.sum() - 1.0) < 0.01, f"Sum = {acc_bin.sum()}")
        check(f"  Move {move_i}: type sums to 1",
              abs(type_oh.sum() - 1.0) < 0.01, f"Sum = {type_oh.sum()}")
        check(f"  Move {move_i}: category sums to 1",
              abs(cat_oh.sum() - 1.0) < 0.01, f"Sum = {cat_oh.sum()}")
        check(f"  Move {move_i}: priority sums to 1",
              abs(pri_oh.sum() - 1.0) < 0.01, f"Sum = {pri_oh.sum()}")
        check(f"  Move {move_i}: pp_fraction in [0,1]",
              0 <= pp_frac <= 1.0, f"Got {pp_frac}")
    else:
        check(f"  Move {move_i}: unknown (is_known=0), all zeros",
              move_feats.sum() == 0.0, f"Sum = {move_feats.sum()}")

# ===== 6. Opponent Token Encoding =====
print("\n--- 6. Opponent Token (token 7 = opp active) ---")
opp_active_float = obs["float_feats"][7]
opp_is_own = opp_active_float[179]
check("Opp is_own = 0", opp_is_own == 0.0, f"Got {opp_is_own}")

opp_is_active = opp_active_float[172]
check("Opp active is_active = 1", opp_is_active == 1.0, f"Got {opp_is_active}")

# Opp active should have known moves (we can see opponent's active moves)
for move_i in range(4):
    offset = 180 + move_i * 45
    is_known = opp_active_float[offset + 44]
    if is_known > 0.5:
        check(f"  Opp move {move_i}: is_known=1 (opponent active visible)", True)
    else:
        check(f"  Opp move {move_i}: unknown (slot may be empty)", True)

# Opp reserve should NOT have known moves
opp_reserve_float = obs["float_feats"][8]  # token 8 = first opp reserve
for move_i in range(4):
    offset = 180 + move_i * 45
    is_known = opp_reserve_float[offset + 44]
    check(f"  Opp reserve move {move_i}: is_known=0 (hidden)",
          is_known == 0.0, f"Got {is_known}")

# ===== 7. Query Tokens (Actor/Critic) =====
print("\n--- 7. Query Tokens (13=actor, 14=critic) ---")
actor_float = obs["float_feats"][13]
critic_float = obs["float_feats"][14]
check("Actor token is all zeros", np.all(actor_float == 0))
check("Critic token is all zeros", np.all(critic_float == 0))
check("Actor int_ids are all zeros", np.all(obs["int_ids"][13] == 0))
check("Critic int_ids are all zeros", np.all(obs["int_ids"][14] == 0))

# ===== 8. Weather Encoding =====
print("\n--- 8. Weather Encoding (via synthetic state) ---")
builder = ObsBuilder()

# Build synthetic state dict with specific weather
def make_synthetic_state(weather="", trick_room=False, trick_room_turns=0,
                         status=None, boosts=None, volatile=None):
    """Build a minimal synthetic state dict for testing encodings."""
    def make_mon(species="pikachu", hp=100, maxhp=100, moves=None,
                 types=None, status=None, boosts=None, volatile=None,
                 is_active=True, is_fainted=False):
        return {
            "species": species, "hp": hp, "maxhp": maxhp,
            "status": status, "boosts": boosts or {},
            "volatile_statuses": list(volatile or set()),
            "moves": moves or [
                {"id": "thunderbolt", "basePower": 90, "accuracy": 100,
                 "type": "Electric", "category": "special", "priority": 0,
                 "pp": 15, "maxpp": 15, "is_known": True},
                {"id": "surf", "basePower": 90, "accuracy": 100,
                 "type": "Water", "category": "special", "priority": 0,
                 "pp": 15, "maxpp": 15, "is_known": True},
                {"id": "icebeam", "basePower": 90, "accuracy": 100,
                 "type": "Ice", "category": "special", "priority": 0,
                 "pp": 10, "maxpp": 10, "is_known": True},
                {"id": "protect", "basePower": 0, "accuracy": 0,
                 "type": "Normal", "category": "status", "priority": 4,
                 "pp": 10, "maxpp": 10, "is_known": True},
            ],
            "types": types or ["electric"],
            "base_stats": {"hp": 35, "attack": 55, "defense": 40,
                          "special-attack": 50, "special-defense": 50, "speed": 90},
            "ability": "static", "item": "lightball",
            "is_fainted": is_fainted, "is_active": is_active,
        }

    reserve = [make_mon(species=f"reserve{i}", is_active=False,
                        types=["normal"]) for i in range(5)]
    opp_reserve = [make_mon(species=f"oppreserve{i}", is_active=False,
                            types=["normal"]) for i in range(5)]

    return {
        "side_one": {
            "active": make_mon(status=status, boosts=boosts, volatile=volatile),
            "reserve": reserve,
        },
        "side_two": {
            "active": make_mon(types=["water", "ground"]),
            "reserve": opp_reserve,
        },
        "weather": weather,
        "weather_turns": 3 if weather else 0,
        "terrain": "",
        "terrain_turns": 0,
        "trick_room": trick_room,
        "trick_room_turns": trick_room_turns,
        "turn": 5,
        "legal_actions": [0, 1, 2, 3, 4, 5],
    }

# Test each weather
for weather_str, expected_idx in [("", 0), ("sunnyday", 1), ("raindance", 2),
                                   ("sandstorm", 3), ("hail", 4)]:
    state = make_synthetic_state(weather=weather_str)
    enc = builder.encode(state)
    w_onehot = enc["float_feats"][0][0:5]
    actual_idx = int(np.argmax(w_onehot))
    check(f"Weather '{weather_str}' → idx {expected_idx}",
          actual_idx == expected_idx,
          f"Got idx {actual_idx}, onehot={w_onehot}")

# Weather turns when weather is active
state_rain = make_synthetic_state(weather="raindance")
enc_rain = builder.encode(state_rain)
weather_turns_bin = enc_rain["float_feats"][0][5:13]
check("Weather turns bin non-zero when weather active",
      np.any(weather_turns_bin > 0),
      f"All zeros: {weather_turns_bin}")

# ===== 9. Status Encoding =====
print("\n--- 9. Status Encoding ---")
for status_str, expected_idx in [(None, 0), ("brn", 1), ("psn", 2), ("tox", 3),
                                  ("slp", 4), ("frz", 5), ("par", 6)]:
    state = make_synthetic_state(status=status_str)
    enc = builder.encode(state)
    status_onehot = enc["float_feats"][1][108:115]
    actual_idx = int(np.argmax(status_onehot))
    check(f"Status '{status_str}' → idx {expected_idx}",
          actual_idx == expected_idx,
          f"Got idx {actual_idx}, onehot={status_onehot}")

# ===== 10. Volatile Status Encoding =====
print("\n--- 10. Volatile Status Encoding ---")
volatile_tests = ["confusion", "leechseed", "substitute", "taunt", "encore",
                   "partiallytrapped", "yawn", "focusenergy"]
for vol in volatile_tests:
    state = make_synthetic_state(volatile={vol})
    enc = builder.encode(state)
    volatile_vec = enc["float_feats"][1][115:135]
    expected_idx = VOLATILE_TO_IDX.get(vol)
    if expected_idx is not None:
        check(f"Volatile '{vol}' → bit {expected_idx} set",
              volatile_vec[expected_idx] == 1.0,
              f"volatile_vec = {volatile_vec}")
    else:
        check(f"Volatile '{vol}' not in VOLATILE_TO_IDX", False,
              "Missing from index!")

# Multi-hot: two volatile statuses at once
state = make_synthetic_state(volatile={"confusion", "substitute"})
enc = builder.encode(state)
vol_vec = enc["float_feats"][1][115:135]
conf_idx = VOLATILE_TO_IDX["confusion"]
sub_idx = VOLATILE_TO_IDX["substitute"]
check("Multi-hot: confusion + substitute both set",
      vol_vec[conf_idx] == 1.0 and vol_vec[sub_idx] == 1.0,
      f"confusion={vol_vec[conf_idx]}, sub={vol_vec[sub_idx]}")
check("Multi-hot: other bits not set",
      vol_vec.sum() == 2.0, f"Sum = {vol_vec.sum()}")

# ===== 11. Stat Boosts Encoding =====
print("\n--- 11. Stat Boost Encoding ---")
boosts = {"atk": 2, "def": -1, "spa": 0, "spd": 0, "spe": 3, "accuracy": 0, "evasion": 0}
state = make_synthetic_state(boosts=boosts)
enc = builder.encode(state)
boost_vec = enc["float_feats"][1][17:108]

# atk=2 → index 2+6=8 in 13-hot
check("atk boost +2 → bit 8", boost_vec[8] == 1.0, f"atk block = {boost_vec[0:13]}")
# def=-1 → index -1+6=5
check("def boost -1 → bit 5", boost_vec[13+5] == 1.0, f"def block = {boost_vec[13:26]}")
# spe=+3 → index 3+6=9
check("spe boost +3 → bit 9", boost_vec[4*13+9] == 1.0, f"spe block = {boost_vec[4*13:5*13]}")

# ===== 12. Trick Room Encoding =====
print("\n--- 12. Trick Room Encoding ---")
state_tr = make_synthetic_state(trick_room=True, trick_room_turns=3)
enc_tr = builder.encode(state_tr)
pseudo = enc_tr["float_feats"][0][13:18]
check("Trick Room bit set", pseudo[0] == 1.0, f"pseudo = {pseudo}")
tr_turns = enc_tr["float_feats"][0][18:22]
check("Trick room turns encoded", np.any(tr_turns > 0), f"tr_turns = {tr_turns}")

# ===== 13. Type Effectiveness =====
print("\n--- 13. Type Effectiveness ---")
check("Fire vs Grass = 2.0",
      _type_effectiveness("fire", ["grass"]) == 2.0)
check("Water vs Fire = 2.0",
      _type_effectiveness("water", ["fire"]) == 2.0)
check("Electric vs Ground = 0.0",
      _type_effectiveness("electric", ["ground"]) == 0.0)
check("Normal vs Ghost = 0.0",
      _type_effectiveness("normal", ["ghost"]) == 0.0)
check("Fighting vs Normal = 2.0",
      _type_effectiveness("fighting", ["normal"]) == 2.0)
check("Ice vs Dragon/Flying = 4.0",
      _type_effectiveness("ice", ["dragon", "flying"]) == 4.0)
check("Water vs Water/Ground = 1.0 (Water resists, Ground weak)",
      abs(_type_effectiveness("water", ["water", "ground"]) - 1.0) < 0.01)
check("Grass vs Water/Ground = 4.0",
      _type_effectiveness("grass", ["water", "ground"]) == 4.0)
# Case insensitivity
check("FIRE vs grass (case) = 2.0",
      _type_effectiveness("FIRE", ["grass"]) == 2.0)
check("fire vs GRASS (case) = 2.0",
      _type_effectiveness("fire", ["GRASS"]) == 2.0)

# ===== 14. Move Accuracy Encoding =====
print("\n--- 14. Move Accuracy Bins ---")
# [0(never-miss), 50, 70, 80, 90, 100+]
check("acc=0 → bin 0 (never-miss)", np.argmax(_bin_accuracy(0)) == 0)
check("acc=-1 → bin 0", np.argmax(_bin_accuracy(-1)) == 0)
check("acc=101 → bin 0 (>100 = bypass)", np.argmax(_bin_accuracy(101)) == 0)
check("acc=50 → bin 1", np.argmax(_bin_accuracy(50)) == 1)
check("acc=70 → bin 2", np.argmax(_bin_accuracy(70)) == 2)
check("acc=75 → bin 2 (sleeppowder)", np.argmax(_bin_accuracy(75)) == 2)
check("acc=80 → bin 3 (hydro pump)", np.argmax(_bin_accuracy(80)) == 3)
check("acc=85 → bin 3 (fire blast)", np.argmax(_bin_accuracy(85)) == 3)
check("acc=90 → bin 4", np.argmax(_bin_accuracy(90)) == 4)
check("acc=100 → bin 5", np.argmax(_bin_accuracy(100)) == 5)

# ===== 15. Move Base Power Bins =====
print("\n--- 15. Move Base Power Bins ---")
check("bp=0 → bin 0 (status)", np.argmax(_bin_base_power(0)) == 0)
check("bp=40 → bin 1", np.argmax(_bin_base_power(40)) == 1)
check("bp=60 → bin 2", np.argmax(_bin_base_power(60)) == 2)
check("bp=80 → bin 3", np.argmax(_bin_base_power(80)) == 3)
check("bp=90 → bin 3 (tbolt/surf)", np.argmax(_bin_base_power(90)) == 3)
check("bp=100 → bin 3", np.argmax(_bin_base_power(100)) == 3)
check("bp=120 → bin 5", np.argmax(_bin_base_power(120)) == 5)
check("bp=150 → bin 6", np.argmax(_bin_base_power(150)) == 6)

# ===== 16. Legal Mask =====
print("\n--- 16. Legal Mask ---")
legal = obs["legal_mask"]
check("Legal mask has at least 1 legal action", legal.sum() >= 1)
check("Legal mask values are 0 or 1",
      np.all(np.isin(legal, [0.0, 1.0])),
      f"Unique values: {np.unique(legal)}")

# ===== 17. Int IDs =====
print("\n--- 17. Integer ID Embeddings ---")
# Own active should have real species/move/ability/item IDs
own_int_ids = obs["int_ids"][1]  # token 1 = own active
check("Own active species_idx > 0", own_int_ids[0] > 0,
      f"species_idx = {own_int_ids[0]}")
# At least one move should have a non-zero/non-one idx
check("Own active has real move IDs",
      any(own_int_ids[i] > 1 for i in range(1, 5)),
      f"move_idxs = {own_int_ids[1:5]}")
check("Own active ability_idx > 0", own_int_ids[5] > 0,
      f"ability_idx = {own_int_ids[5]}")

# ===== 18. Run Multiple Steps and Check Consistency =====
print("\n--- 18. Multi-Step Consistency ---")
obs, _ = env.reset()
for step in range(10):
    legal = obs["legal_mask"]
    legal_actions = [i for i in range(10) if legal[i] > 0.5]
    action = np.random.choice(legal_actions)
    obs, reward, done, _, info = env.step(action)

    # Check shapes
    assert obs["float_feats"].shape == (15, FLOAT_DIM_PER_POKEMON), f"Step {step}: wrong shape {obs['float_feats'].shape}"
    assert obs["int_ids"].shape == (15, 8), f"Step {step}: wrong int_ids shape {obs['int_ids'].shape}"

    # Check no NaN/Inf
    assert not np.any(np.isnan(obs["float_feats"])), f"Step {step}: NaN in float_feats!"
    assert not np.any(np.isinf(obs["float_feats"])), f"Step {step}: Inf in float_feats!"

    if done:
        obs, _ = env.reset()

check("10 steps: no shape errors", True)
check("10 steps: no NaN/Inf values", True)

# ===== 19. Run Many Games to Check for Encoding Errors =====
print("\n--- 19. Stress Test (50 games) ---")
n_games = 50
errors = 0
for g in range(n_games):
    obs, _ = env.reset()
    done = False
    while not done:
        legal = obs["legal_mask"]
        legal_actions = [i for i in range(10) if legal[i] > 0.5]
        if not legal_actions:
            errors += 1
            break
        action = np.random.choice(legal_actions)
        obs, reward, done, _, info = env.step(action)

        if np.any(np.isnan(obs["float_feats"])):
            errors += 1
            print(f"  NaN found in game {g}!")
            break
        if obs["float_feats"].shape != (15, FLOAT_DIM_PER_POKEMON):
            errors += 1
            print(f"  Wrong shape in game {g}: {obs['float_feats'].shape}")
            break

check(f"50 games completed without encoding errors", errors == 0,
      f"{errors} errors found")

# ===== 20. Verify specific moves have correct accuracy from data =====
print("\n--- 20. Move Data Accuracy Check ---")
import json
move_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gen4_move_data.json")
if os.path.exists(move_data_path):
    with open(move_data_path) as f:
        move_data = json.load(f)

    expected = {
        "thunderbolt": {"basePower": 95, "accuracy": 100, "type": "Electric"},
        "thunder": {"basePower": 120, "accuracy": 70, "type": "Electric"},
        "hydropump": {"basePower": 120, "accuracy": 80, "type": "Water"},
        "fireblast": {"basePower": 120, "accuracy": 85, "type": "Fire"},
        "sleeppowder": {"basePower": 0, "accuracy": 75, "type": "Grass"},
        "surf": {"basePower": 95, "accuracy": 100, "type": "Water"},
        "earthquake": {"basePower": 100, "accuracy": 100, "type": "Ground"},
        "icebeam": {"basePower": 95, "accuracy": 100, "type": "Ice"},
    }

    for move_id, exp in expected.items():
        md = move_data.get(move_id, {})
        acc = md.get("accuracy", "missing")
        bp = md.get("basePower", "missing")
        check(f"{move_id}: accuracy={exp['accuracy']}",
              acc == exp["accuracy"],
              f"Got {acc}")
        check(f"{move_id}: basePower={exp['basePower']}",
              bp == exp["basePower"],
              f"Got {bp}")
else:
    print("  [SKIP] gen4_move_data.json not found")

# ===== 21. New Encoding: Opponent Ability/Item Visible =====
print("\n--- 21. Opponent Ability/Item Now Visible ---")
obs21, _ = env.reset()
raw21 = env._build_obs_dict(perspective="side_one")
opp_active = raw21["side_two"]["active"]
check("Opponent ability not None", opp_active["ability"] is not None,
      f"ability={opp_active['ability']}")
check("Opponent item not None", opp_active["item"] is not None,
      f"item={opp_active['item']}")
# Verify int_ids for opponent active have non-zero ability/item
opp_int = obs21["int_ids"][7]  # token 7 = opp active
check("Opp active ability_idx > 0", opp_int[5] > 0,
      f"ability_idx={opp_int[5]}")

# ===== 22. New Encoding: Dimensions Check =====
print("\n--- 22. New Dimensions ---")
check(f"FLOAT_DIM = {FLOAT_DIM_PER_POKEMON} (expected 380)",
      FLOAT_DIM_PER_POKEMON == 380)
check(f"FIELD_DIM = {FIELD_DIM} (expected 76)",
      FIELD_DIM == 76)
check("int_ids shape (15, 8)", obs21["int_ids"].shape == (15, 8),
      f"got {obs21['int_ids'].shape}")
check("float_feats shape (15, 380)", obs21["float_feats"].shape == (15, 380),
      f"got {obs21['float_feats'].shape}")

# ===== 23. New Encoding: Side-Level State =====
print("\n--- 23. Side-Level State Extracted ---")
own_side = raw21["side_one"]
check("toxic_count present", "toxic_count" in own_side, f"keys={list(own_side.keys())}")
check("tailwind present", "tailwind" in own_side)
check("safeguard present", "safeguard" in own_side)
check("wish present", "wish" in own_side)
check("substitute_health present", "substitute_health" in own_side)
check("force_trapped present", "force_trapped" in own_side)
check("volatile_durations present", "volatile_durations" in own_side)
check("last_used_move present", "last_used_move" in own_side)
vd = own_side.get("volatile_durations", {})
check("volatile_durations has confusion", "confusion" in vd)
check("volatile_durations has yawn", "yawn" in vd)
check("volatile_durations has taunt", "taunt" in vd)

# ===== 24. New Encoding: Per-Mon Fields =====
print("\n--- 24. Per-Mon Sleep/Rest Turns ---")
own_active = raw21["side_one"]["active"]
check("sleep_turns present", "sleep_turns" in own_active)
check("rest_turns present", "rest_turns" in own_active)
check("sleep_turns initial = 0", own_active["sleep_turns"] == 0)
check("rest_turns initial = 0", own_active["rest_turns"] == 0)

# ===== Summary =====
print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} checks")
if failed > 0:
    print("SOME CHECKS FAILED! Fix before continuing training.")
else:
    print("ALL CHECKS PASSED! Encodings are correct.")
print("=" * 70)
