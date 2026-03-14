"""Neural network architecture hyperparameters.

Vocab sizes (n_species, n_moves, etc.) are pulled from GenConfig at runtime.
This file contains only architecture hyperparameters.
"""

from config.gen_config import get_gen_config
from config.training_config import TRAINING_CONFIG

# Pull generation-specific vocab sizes from GenConfig
_gen_config = get_gen_config(battle_format=TRAINING_CONFIG["battle_format"])

MODEL_CONFIG = {
    # Transformer
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,
    "dropout": 0.1,

    # Embedding dimensions
    "species_embed_dim": 128,
    "move_embed_dim": 64,
    "ability_embed_dim": 32,
    "item_embed_dim": 32,

    # Vocabulary sizes — pulled from GenConfig (generation-specific)
    "n_species": _gen_config.n_species,
    "n_moves": _gen_config.n_moves,
    "n_abilities": _gen_config.n_abilities,
    "n_items": _gen_config.n_items,

    # Action space: 4 moves + 6 switches = 10 (no Dynamax for simplicity)
    # Switch to 14 if Dynamax is added later
    "n_actions": 10,

    # Distributional value head (C51)
    "v_min": -1.5,
    "v_max": 1.5,
    "n_atoms": 51,

    # Token sequence length: 1 field + 6 own + 6 opp + 1 actor + 1 critic = 15
    "n_tokens": 15,
}
