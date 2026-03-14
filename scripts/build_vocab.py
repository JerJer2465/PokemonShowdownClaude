"""
Build vocabulary index files from Pokemon Showdown data.

Downloads Gen 4 Random Battle set data from pkmn/randbats and
generates species_index.json, move_index.json, ability_index.json, item_index.json.

Usage:
    python scripts/build_vocab.py

Outputs to data/ directory.
"""

import json
import os
import sys
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

RANDBATS_URL = (
    "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen4randombattle.json"
)

# Additional moves/species/abilities/items that might appear in Gen 4 battles
# but aren't in randbats (e.g., moves used only by certain mons via level-up)
EXTRA_MOVES = [
    "struggle", "recharge",  # special cases
]


def fetch_randbats(url: str) -> dict:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode())
    print(f"  Got {len(data)} species entries.")
    return data


def normalize(name: str) -> str:
    """Normalize name to lowercase alphanumeric (PS convention)."""
    return "".join(c for c in name.lower() if c.isalnum())


def build_vocabs(randbats: dict) -> tuple[dict, dict, dict, dict]:
    species_set: set[str] = set()
    move_set: set[str] = set()
    ability_set: set[str] = set()
    item_set: set[str] = set()

    for species, data in randbats.items():
        species_set.add(normalize(species))
        # Gen 8 format: flat lists directly on species entry
        # Gen 9 format: nested under "roles" key
        if "roles" in data:
            sources = list(data["roles"].values())
        else:
            sources = [data]
        for entry in sources:
            for m in entry.get("moves", []):
                move_set.add(normalize(m))
            for a in entry.get("abilities", []):
                ability_set.add(normalize(a))
            for it in entry.get("items", []):
                item_set.add(normalize(it))

    # Add extras
    for m in EXTRA_MOVES:
        move_set.add(normalize(m))

    # Sort for determinism; index 0 = UNKNOWN, index 1 = EMPTY (moves only)
    def _make_index(names: set, extra_reserved: int = 1) -> dict[str, int]:
        sorted_names = sorted(names)
        # Reserve indices 0..extra_reserved-1 for special tokens
        return {name: i + extra_reserved for i, name in enumerate(sorted_names)}

    species_idx = _make_index(species_set, extra_reserved=1)   # 0 = UNKNOWN
    ability_idx = _make_index(ability_set, extra_reserved=1)
    item_idx = _make_index(item_set, extra_reserved=1)
    move_idx = _make_index(move_set, extra_reserved=2)          # 0 = UNKNOWN, 1 = EMPTY

    return species_idx, move_idx, ability_idx, item_idx


def save_json(data: dict, filename: str):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"  Saved {len(data)} entries → {path}")


def main():
    # Fetch randbats data
    randbats_path = os.path.join(DATA_DIR, "gen4randombattle.json")
    if os.path.exists(randbats_path):
        print(f"Using cached {randbats_path}")
        with open(randbats_path) as f:
            randbats = json.load(f)
    else:
        randbats = fetch_randbats(RANDBATS_URL)
        save_json(randbats, "gen4randombattle.json")

    print("Building vocabulary indices...")
    species_idx, move_idx, ability_idx, item_idx = build_vocabs(randbats)

    save_json(species_idx, "species_index.json")
    save_json(move_idx, "move_index.json")
    save_json(ability_idx, "ability_index.json")
    save_json(item_idx, "item_index.json")

    print("\nVocabulary sizes (including special tokens):")
    print(f"  Species:   {max(species_idx.values()) + 1} (0=UNKNOWN)")
    print(f"  Moves:     {max(move_idx.values()) + 1} (0=UNKNOWN, 1=EMPTY)")
    print(f"  Abilities: {max(ability_idx.values()) + 1} (0=UNKNOWN)")
    print(f"  Items:     {max(item_idx.values()) + 1} (0=UNKNOWN)")
    print("\nUpdate config/model_config.py with these sizes!")


if __name__ == "__main__":
    main()
