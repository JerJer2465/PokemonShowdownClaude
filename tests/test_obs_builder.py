"""Unit tests for obs_builder.py — can run on Mac without poke-engine."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pokebot.env.obs_builder import (
    ObsBuilder, FLOAT_DIM_PER_POKEMON, N_TOKENS,
    _one_hot, _bin_hp, _bin_base_power, _encode_stat_boosts, _encode_volatile_status,
    TYPE_TO_IDX, STATUS_TO_IDX, VOLATILE_STATUS_LIST,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mon(species="charizard", hp=300, maxhp=300, status=None, slot=0,
              is_active=False, is_fainted=False, types=None):
    return {
        "species": species,
        "hp": hp,
        "maxhp": maxhp,
        "status": status,
        "boosts": {"atk": 1, "def": -1},
        "moves": [
            {"id": "flamethrower", "basePower": 90, "type": "Fire",
             "category": "special", "priority": 0, "pp": 15, "maxpp": 15, "is_known": True},
            {"id": "tackle", "basePower": 40, "type": "Normal",
             "category": "physical", "priority": 0, "pp": 35, "maxpp": 35, "is_known": True},
        ],
        "ability": "blaze",
        "item": "choiceband",
        "types": types or ["Fire", "Flying"],
        "base_stats": {"hp": 78, "attack": 84, "defense": 78,
                       "special-attack": 109, "special-defense": 85, "speed": 100},
        "volatile_statuses": ["confusion"],
        "is_fainted": is_fainted,
        "is_dynamaxed": False,
        "can_dynamax": True,
        "dynamax_turns_remaining": 0,
        "is_active": is_active,
    }


def _make_state():
    return {
        "side_one": {
            "active": _make_mon("charizard", slot=0, is_active=True),
            "reserve": [_make_mon(f"mon{i}", slot=i+1) for i in range(5)],
            "hazards": {"stealth_rock": True, "spikes": 2},
            "screens": {"light_screen": 3},
        },
        "side_two": {
            "active": _make_mon("blastoise", slot=0, is_active=True),
            "reserve": [_make_mon(f"opp_mon{i}", slot=i+1) for i in range(5)],
            "hazards": {},
            "screens": {},
        },
        "weather": "raindance",
        "weather_turns": 3,
        "terrain": None,
        "terrain_turns": 0,
        "trick_room": False,
        "trick_room_turns": 0,
        "turn": 5,
        "legal_actions": [0, 1, 5, 6, 7],
    }


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

def test_one_hot_basic():
    arr = _one_hot(2, 5)
    assert arr.shape == (5,)
    assert arr[2] == 1.0
    assert arr.sum() == 1.0


def test_one_hot_out_of_bounds():
    arr = _one_hot(10, 5)
    assert arr.sum() == 0.0


def test_bin_hp():
    assert _bin_hp(0.0)[0] == 1.0
    assert _bin_hp(1.0).sum() == 1.0
    assert _bin_hp(0.5).sum() == 1.0
    assert _bin_hp(0.5).shape == (10,)


def test_bin_base_power():
    arr = _bin_base_power(0)
    assert arr.shape == (8,)
    assert arr.sum() == 1.0
    arr90 = _bin_base_power(90)
    assert arr90.sum() == 1.0
    # Different bins for 0 vs 90
    assert not np.array_equal(arr, arr90)


def test_stat_boosts_shape():
    boosts = {"atk": 1, "def": -2, "spe": 6}
    arr = _encode_stat_boosts(boosts)
    assert arr.shape == (91,)
    assert arr.sum() == 7.0  # 7 stats, each one-hot


def test_stat_boosts_zero():
    arr = _encode_stat_boosts({})
    # All stats at 0 → index 6 in each 13-dim block
    assert arr.shape == (91,)
    assert arr.sum() == 7.0


def test_volatile_status():
    arr = _encode_volatile_status({"confusion", "taunt"})
    assert arr.shape == (20,)
    assert arr.sum() == 2.0
    assert arr[0] == 1.0   # confusion is index 0
    assert arr[6] == 1.0   # taunt is index 6


# ---------------------------------------------------------------------------
# ObsBuilder integration tests
# ---------------------------------------------------------------------------

def test_obs_builder_output_shape():
    builder = ObsBuilder()
    state = _make_state()
    obs = builder.encode(state)

    assert "int_ids" in obs
    assert "float_feats" in obs
    assert "legal_mask" in obs

    assert obs["int_ids"].shape == (N_TOKENS, 7)
    assert obs["float_feats"].shape == (N_TOKENS, FLOAT_DIM_PER_POKEMON)
    assert obs["legal_mask"].shape == (10,)


def test_obs_builder_dtypes():
    builder = ObsBuilder()
    state = _make_state()
    obs = builder.encode(state)
    assert obs["int_ids"].dtype == np.int64
    assert obs["float_feats"].dtype == np.float32
    assert obs["legal_mask"].dtype == np.float32


def test_legal_mask_correct():
    builder = ObsBuilder()
    state = _make_state()
    obs = builder.encode(state)
    expected_legal = [0, 1, 5, 6, 7]
    for i in range(10):
        if i in expected_legal:
            assert obs["legal_mask"][i] == 1.0, f"Action {i} should be legal"
        else:
            assert obs["legal_mask"][i] == 0.0, f"Action {i} should be illegal"


def test_obs_no_nan():
    builder = ObsBuilder()
    state = _make_state()
    obs = builder.encode(state)
    assert not np.any(np.isnan(obs["float_feats"])), "NaN in float_feats"
    assert not np.any(np.isinf(obs["float_feats"])), "Inf in float_feats"


def test_query_tokens_are_zero():
    """Tokens 13 (ACTOR) and 14 (CRITIC) float_feats should be all zeros."""
    builder = ObsBuilder()
    state = _make_state()
    obs = builder.encode(state)
    assert np.allclose(obs["float_feats"][13], 0.0), "ACTOR token float_feats should be zero"
    assert np.allclose(obs["float_feats"][14], 0.0), "CRITIC token float_feats should be zero"


def test_fainted_mon_zero_hp():
    builder = ObsBuilder()
    state = _make_state()
    state["side_one"]["reserve"][0]["is_fainted"] = True
    state["side_one"]["reserve"][0]["hp"] = 0
    obs = builder.encode(state)
    # Should not crash
    assert obs["float_feats"].shape == (N_TOKENS, FLOAT_DIM_PER_POKEMON)


def test_unknown_opp_slots():
    """Opponent slots beyond revealed count should use UNKNOWN tokens."""
    builder = ObsBuilder()
    state = _make_state()
    # Only reveal 2 opponent mons
    state["side_two"]["reserve"] = [_make_mon(f"opp_mon{i}", slot=i+1) for i in range(1)]
    obs = builder.encode(state)
    # Should still produce 15 tokens
    assert obs["int_ids"].shape[0] == N_TOKENS


def test_weather_encoded():
    builder = ObsBuilder()
    state = _make_state()
    obs_rain = builder.encode(state)

    state2 = _make_state()
    state2["weather"] = "sunnyday"
    obs_sun = builder.encode(state2)

    # Field tokens should differ
    assert not np.allclose(obs_rain["float_feats"][0], obs_sun["float_feats"][0])


def test_reproducible():
    """Same state → same obs."""
    builder = ObsBuilder()
    state = _make_state()
    obs1 = builder.encode(state)
    obs2 = builder.encode(state)
    assert np.allclose(obs1["float_feats"], obs2["float_feats"])
    assert np.array_equal(obs1["int_ids"], obs2["int_ids"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
