"""Generation-specific configuration — single source of truth for all gen-dependent settings.

To add a new generation:
1. Add an entry to GEN_REGISTRY below
2. Create data files (genXrandombattle.json, genX_base_stats.json, genX_move_data.json)
3. Build vocab indices (python scripts/build_vocab.py --gen X)
4. Train a new model (same architecture, different vocab sizes and feature flags)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    """All generation-specific parameters in one place."""

    gen: int                    # 4, 5, 6, 7, 8, 9
    battle_format: str          # "gen4randombattle"

    # Feature flags — which mechanics exist in this gen?
    has_fairy: bool             # Gen 6+ (Fairy type)
    has_terrain: bool           # Gen 5+ (Grassy/Electric/Psychic/Misty)
    has_aurora_veil: bool       # Gen 7+ (Aurora Veil screen)
    has_sticky_web: bool        # Gen 6+ (Sticky Web hazard)
    has_wonder_room: bool       # Gen 5+ (Wonder Room pseudo-weather)
    has_dynamax: bool           # Gen 8 only
    has_terastallize: bool      # Gen 9+
    has_mega: bool              # Gen 6-7
    has_zmoves: bool            # Gen 7

    # Vocabulary sizes (+1 for UNKNOWN token already included)
    n_species: int
    n_moves: int
    n_abilities: int
    n_items: int

    # Data file paths (relative to project root)
    randbats_file: str          # "data/gen4randombattle.json"
    base_stats_file: str        # "data/gen4_base_stats.json"
    move_data_file: str         # "data/gen4_move_data.json"


GEN_REGISTRY: dict[int, GenConfig] = {
    4: GenConfig(
        gen=4,
        battle_format="gen4randombattle",
        has_fairy=False,
        has_terrain=False,
        has_aurora_veil=False,
        has_sticky_web=False,
        has_wonder_room=False,
        has_dynamax=False,
        has_terastallize=False,
        has_mega=False,
        has_zmoves=False,
        n_species=297,
        n_moves=188,
        n_abilities=102,
        n_items=39,
        randbats_file="data/gen4randombattle.json",
        base_stats_file="data/gen4_base_stats.json",
        move_data_file="data/gen4_move_data.json",
    ),
    # Future generations:
    # 5: GenConfig(gen=5, ...),
    # 6: GenConfig(gen=6, has_fairy=True, has_terrain=True, ...),
    # 7: GenConfig(gen=7, has_fairy=True, has_terrain=True, has_aurora_veil=True, has_mega=True, has_zmoves=True, ...),
    # 8: GenConfig(gen=8, has_fairy=True, has_terrain=True, has_aurora_veil=True, has_dynamax=True, ...),
    # 9: GenConfig(gen=9, has_fairy=True, has_terrain=True, has_aurora_veil=True, has_terastallize=True, ...),
}


def get_gen_config(gen: int | None = None, battle_format: str | None = None) -> GenConfig:
    """Get generation config by gen number or battle format string.

    Examples:
        get_gen_config(gen=4)
        get_gen_config(battle_format="gen4randombattle")
    """
    if gen is not None:
        if gen not in GEN_REGISTRY:
            raise ValueError(f"Generation {gen} not registered. Available: {list(GEN_REGISTRY.keys())}")
        return GEN_REGISTRY[gen]

    if battle_format is not None:
        # Extract gen number from format string like "gen4randombattle"
        for g, cfg in GEN_REGISTRY.items():
            if cfg.battle_format == battle_format:
                return cfg
        # Try parsing gen number from format
        import re
        m = re.match(r"gen(\d+)", battle_format)
        if m:
            g = int(m.group(1))
            if g in GEN_REGISTRY:
                return GEN_REGISTRY[g]
        raise ValueError(f"No config for format '{battle_format}'. Available: {[c.battle_format for c in GEN_REGISTRY.values()]}")

    raise ValueError("Must specify gen or battle_format")
