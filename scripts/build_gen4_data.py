"""
Build Gen 4 data files needed for training:
  - data/gen4randombattle.json  (randbats team sets)
  - data/gen4_base_stats.json   (species base stats + types)
  - data/gen4_move_data.json    (move basePower, type, category, priority)

Data sources: Pokemon Showdown GitHub (raw JSON/TS data).

Usage:
    python scripts/build_gen4_data.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# URL constants
# ---------------------------------------------------------------------------

RANDBATS_URL = (
    "https://raw.githubusercontent.com/pkmn/randbats/main/data/gen4randombattle.json"
)

# Showdown's compiled learnset/pokedex JSON endpoints
POKEDEX_URL = (
    "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/pokedex.ts"
)
MOVES_URL = (
    "https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/moves.ts"
)

# Alternative: use the pre-parsed JSON from @pkmn/data on npm CDN
PKMN_DEXDATA_URL = (
    "https://raw.githubusercontent.com/pkmn/ps/main/dex/data/gens.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> str:
    print(f"  Fetching {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}")
        return ""


def normalize(name: str) -> str:
    return "".join(c for c in str(name).lower() if c.isalnum())


def save_json(data: dict | list, filename: str):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  Saved {len(data)} entries → {path}")


# ---------------------------------------------------------------------------
# Gen 4 Randbats data
# ---------------------------------------------------------------------------

def fetch_randbats() -> dict:
    path = os.path.join(DATA_DIR, "gen4randombattle.json")
    if os.path.exists(path):
        print(f"  Using cached gen4randombattle.json")
        with open(path) as f:
            return json.load(f)
    raw = fetch_url(RANDBATS_URL)
    if not raw:
        raise RuntimeError("Failed to download gen4randombattle.json")
    data = json.loads(raw)
    save_json(data, "gen4randombattle.json")
    print(f"  Downloaded {len(data)} Gen 4 randbats species.")
    return data


# ---------------------------------------------------------------------------
# Gen 4 base stats — parse from Showdown pokedex.ts
# ---------------------------------------------------------------------------

def parse_pokedex_ts(ts_text: str) -> dict[str, dict]:
    """
    Extract base stats and types from Showdown's pokedex.ts using line-by-line scan.
    Returns {normalized_species: {hp,attack,defense,special-attack,special-defense,speed,types}}
    """
    result: dict[str, dict] = {}

    # Showdown pokedex.ts format (TAB-indented):
    #   \tbulbasaur: {
    #   \t\tnum: 1,
    #   \t\ttypes: ["Grass", "Poison"],
    #   \t\tbaseStats: { hp: 45, atk: 49, def: 49, spa: 65, spd: 65, spe: 45 },
    #   \t},

    # entry_start: lines like "\tbulbasaur: {" (1 tab, lowercase key)
    entry_start = re.compile(r"^\t['\"]?([\w\-\.\' ]+?)['\"]?\s*:\s*\{")
    stats_line  = re.compile(
        r'baseStats:\s*\{[^}]*hp:\s*(\d+)[^}]*atk:\s*(\d+)[^}]*def:\s*(\d+)'
        r'[^}]*spa:\s*(\d+)[^}]*spd:\s*(\d+)[^}]*spe:\s*(\d+)'
    )
    types_line  = re.compile(r'types:\s*\[([^\]]+)\]')
    num_line    = re.compile(r'^\t\tnum:\s*(-?\d+)')
    entry_end   = re.compile(r'^\t\},?')  # closing brace at indent=1 tab

    current_name: str | None = None
    current_num: int | None = None
    current_stats: dict | None = None
    current_types: list | None = None

    def _save():
        if current_name and current_stats and current_num and current_num > 0:
            result[normalize(current_name)] = {**current_stats,
                                               "types": current_types or []}

    for line in ts_text.splitlines():
        m_start = entry_start.match(line)
        if m_start and '{' in line and line.count('{') > line.count('}'):
            _save()
            current_name  = m_start.group(1).strip()
            current_num   = None
            current_stats = None
            current_types = None
            continue

        if current_name is None:
            continue

        m_num = num_line.match(line)
        if m_num:
            current_num = int(m_num.group(1))
            continue

        m_stats = stats_line.search(line)
        if m_stats:
            current_stats = {
                "hp":              int(m_stats.group(1)),
                "attack":          int(m_stats.group(2)),
                "defense":         int(m_stats.group(3)),
                "special-attack":  int(m_stats.group(4)),
                "special-defense": int(m_stats.group(5)),
                "speed":           int(m_stats.group(6)),
            }
            continue

        m_types = types_line.search(line)
        if m_types and current_types is None:
            current_types = [t.strip().strip('"').strip("'")
                             for t in m_types.group(1).split(",") if t.strip()]
            continue

        if entry_end.match(line) and len(line.strip()) <= 2:
            _save()
            current_name  = None
            current_num   = None
            current_stats = None
            current_types = None

    _save()  # save last entry
    return result


def fetch_base_stats() -> dict[str, dict]:
    path = os.path.join(DATA_DIR, "gen4_base_stats.json")
    if os.path.exists(path):
        print(f"  Using cached gen4_base_stats.json ({_count_entries(path)} entries)")
        with open(path) as f:
            return json.load(f)

    ts_text = fetch_url(POKEDEX_URL)
    if not ts_text:
        print("  WARNING: Could not fetch pokedex.ts; using empty base stats table.")
        return {}

    data = parse_pokedex_ts(ts_text)
    print(f"  Parsed {len(data)} species from pokedex.ts")
    save_json(data, "gen4_base_stats.json")
    return data


def _count_entries(path: str) -> int:
    try:
        with open(path) as f:
            return len(json.load(f))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Gen 4 move data — parse from Showdown moves.ts
# ---------------------------------------------------------------------------

# Gen 4 type chart (Physical types pre-split would be wrong, but
# Showdown's moves.ts already has per-move category since it targets all gens)
# Gen 4 introduced the physical/special split so all we need is from moves.ts.

def parse_moves_ts(ts_text: str) -> dict[str, dict]:
    """
    Extract move metadata from Showdown's moves.ts using line-by-line scan.
    Returns {normalized_id: {basePower, type, category, priority, pp}}
    """
    result: dict[str, dict] = {}

    # Showdown moves.ts format:
    #   Thunderbolt: {
    #     num: 85,
    #     accuracy: 100,
    #     basePower: 90,
    #     category: "Special",
    #     type: "Electric",
    #     pp: 15,
    #   },

    entry_start = re.compile(r"^\t['\"]?([\w\-\.\' ]+?)['\"]?\s*:\s*\{")
    entry_end   = re.compile(r"^\t\},?")
    num_line    = re.compile(r'^\t\tnum:\s*(-?\d+)')
    bp_line     = re.compile(r'^\t\tbasePower:\s*(\d+)')
    type_line   = re.compile(r'^\t\ttype:\s*["\']([A-Za-z]+)["\']')
    cat_line    = re.compile(r'^\t\tcategory:\s*["\']([A-Za-z]+)["\']')
    pri_line    = re.compile(r'^\t\tpriority:\s*(-?\d+)')
    pp_line     = re.compile(r'^\t\tpp:\s*(\d+)')

    cur_name: str | None = None
    cur_num: int | None = None
    cur: dict = {}

    def _save():
        if cur_name and cur_num is not None and cur_num > 0:
            result[normalize(cur_name)] = {
                "basePower": cur.get("basePower", 0),
                "type":      cur.get("type", "Normal"),
                "category":  cur.get("category", "physical"),
                "priority":  cur.get("priority", 0),
                "pp":        cur.get("pp", 10),
            }

    for line in ts_text.splitlines():
        m_start = entry_start.match(line)
        if m_start and '{' in line and line.count('{') > line.count('}'):
            _save()
            cur_name = m_start.group(1).strip()
            cur_num  = None
            cur      = {}
            continue

        if cur_name is None:
            continue

        m_num = num_line.match(line)
        if m_num:
            cur_num = int(m_num.group(1))
        elif bp_line.match(line):
            cur["basePower"] = int(bp_line.match(line).group(1))
        elif type_line.match(line):
            cur["type"] = type_line.match(line).group(1).capitalize()
        elif cat_line.match(line):
            cur["category"] = cat_line.match(line).group(1).lower()
        elif pri_line.match(line):
            cur["priority"] = int(pri_line.match(line).group(1))
        elif pp_line.match(line):
            cur["pp"] = int(pp_line.match(line).group(1))
        elif entry_end.match(line) and len(line.strip()) <= 2:
            _save()
            cur_name = None
            cur_num  = None
            cur      = {}

    _save()
    return result


def fetch_move_data() -> dict[str, dict]:
    path = os.path.join(DATA_DIR, "gen4_move_data.json")
    if os.path.exists(path):
        print(f"  Using cached gen4_move_data.json ({_count_entries(path)} entries)")
        with open(path) as f:
            return json.load(f)

    ts_text = fetch_url(MOVES_URL)
    if not ts_text:
        print("  WARNING: Could not fetch moves.ts; using empty move data table.")
        return {}

    data = parse_moves_ts(ts_text)
    print(f"  Parsed {len(data)} moves from moves.ts")
    save_json(data, "gen4_move_data.json")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Building Gen 4 data files ===")

    print("\n[1/3] Gen 4 Randbats sets:")
    randbats = fetch_randbats()
    print(f"  {len(randbats)} species in gen4randombattle.json")

    print("\n[2/3] Gen 4 base stats:")
    base_stats = fetch_base_stats()
    if not base_stats:
        print("  WARNING: No base stats loaded. Pokemon stats will default to 80.")
    else:
        # Quick sanity check
        pikachu = base_stats.get("pikachu", {})
        print(f"  Pikachu base stats: {pikachu}")

    print("\n[3/3] Gen 4 move data:")
    move_data = fetch_move_data()
    if not move_data:
        print("  WARNING: No move data loaded. Move features will default to base_power=0.")
    else:
        # Quick sanity check
        tb = move_data.get("thunderbolt", {})
        print(f"  Thunderbolt: {tb}")

    print("\n=== Done. Run scripts/build_vocab.py next. ===")


if __name__ == "__main__":
    main()
