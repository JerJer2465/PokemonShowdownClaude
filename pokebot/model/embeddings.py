"""Embedding tables for discrete Pokemon entities."""

import torch
import torch.nn as nn

from config.model_config import MODEL_CONFIG


class PokemonEmbeddings(nn.Module):
    """
    Embedding tables for species, moves, abilities, and items.
    Each token has IDs at positions [species, m0, m1, m2, m3, ability, item].
    """

    def __init__(self, cfg: dict = MODEL_CONFIG):
        super().__init__()
        self.species_emb = nn.Embedding(cfg["n_species"], cfg["species_embed_dim"], padding_idx=0)
        self.move_emb = nn.Embedding(cfg["n_moves"], cfg["move_embed_dim"], padding_idx=0)
        self.ability_emb = nn.Embedding(cfg["n_abilities"], cfg["ability_embed_dim"], padding_idx=0)
        self.item_emb = nn.Embedding(cfg["n_items"], cfg["item_embed_dim"], padding_idx=0)

        self.out_dim = (
            cfg["species_embed_dim"]
            + 4 * cfg["move_embed_dim"]
            + cfg["ability_embed_dim"]
            + cfg["item_embed_dim"]
        )  # 128 + 256 + 32 + 32 = 448

    def forward(self, int_ids: torch.Tensor) -> torch.Tensor:
        """
        int_ids: (batch, n_tokens, 7) — [species, m0, m1, m2, m3, ability, item]
        Returns: (batch, n_tokens, out_dim)
        """
        species = self.species_emb(int_ids[..., 0])          # (..., 128)
        moves = torch.cat([
            self.move_emb(int_ids[..., 1]),
            self.move_emb(int_ids[..., 2]),
            self.move_emb(int_ids[..., 3]),
            self.move_emb(int_ids[..., 4]),
        ], dim=-1)                                            # (..., 256)
        ability = self.ability_emb(int_ids[..., 5])           # (..., 32)
        item = self.item_emb(int_ids[..., 6])                 # (..., 32)
        return torch.cat([species, moves, ability, item], dim=-1)  # (..., 448)
