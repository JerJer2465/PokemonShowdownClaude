"""Neural network architecture hyperparameters."""

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

    # Vocabulary sizes (set after build_vocab.py runs)
    # +1 for UNKNOWN token in each
    "n_species": 900,
    "n_moves": 850,
    "n_abilities": 280,
    "n_items": 220,

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
