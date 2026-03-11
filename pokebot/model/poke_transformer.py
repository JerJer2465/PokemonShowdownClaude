"""
PokeTransformer: Actor-Critic Transformer for Pokemon battle decisions.

Architecture:
  1. Per-token MLP projects (embed_dim + float_dim) → d_model
  2. Learned positional embeddings for all 15 token slots
  3. 6-layer Pre-LN Transformer with Poke-Mask:
       - ACTOR token (13) cannot see CRITIC token (14) and vice-versa
       - State tokens (0-12) see each other freely
  4. Policy head reads from ACTOR output → log-probs over actions
  5. Distributional value head reads from CRITIC output → C51 distribution
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MODEL_CONFIG
from pokebot.model.embeddings import PokemonEmbeddings
from pokebot.model.heads import PolicyHead, DistributionalValueHead
from pokebot.env.obs_builder import FLOAT_DIM_PER_POKEMON


N_TOKENS = MODEL_CONFIG["n_tokens"]   # 15
ACTOR_IDX = 13
CRITIC_IDX = 14


def _build_poke_mask(n_tokens: int = N_TOKENS) -> torch.Tensor:
    """
    Attention mask: True = position is BLOCKED (masked out).
    ACTOR token cannot attend to CRITIC token, and vice-versa.
    All state tokens attend to each other freely.
    Shape: (n_tokens, n_tokens)
    """
    mask = torch.zeros(n_tokens, n_tokens, dtype=torch.bool)
    # Actor cannot see Critic
    mask[ACTOR_IDX, CRITIC_IDX] = True
    # Critic cannot see Actor
    mask[CRITIC_IDX, ACTOR_IDX] = True
    return mask


class TokenProjection(nn.Module):
    """
    Projects (embed_out_dim + float_dim) → d_model for each token.
    Shared weights across all 13 state tokens; query tokens (13,14) use
    a separate learned embedding initialized to zero + added after projection.
    """

    def __init__(self, embed_dim: int, float_dim: int, d_model: int):
        super().__init__()
        in_dim = embed_dim + float_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_in) → (batch, d_model)"""
        return self.mlp(x)


class PokeTransformer(nn.Module):
    """
    Full actor-critic Transformer for Pokemon battle decisions.
    """

    def __init__(self, cfg: dict = MODEL_CONFIG):
        super().__init__()
        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        n_layers = cfg["n_layers"]
        d_ff = cfg["d_ff"]
        dropout = cfg["dropout"]

        # Embeddings
        self.embeddings = PokemonEmbeddings(cfg)
        embed_dim = self.embeddings.out_dim  # 448

        # Token projection: embed + float → d_model
        self.token_proj = TokenProjection(embed_dim, FLOAT_DIM_PER_POKEMON, d_model)

        # Positional embeddings (one per slot, learned)
        self.pos_emb = nn.Embedding(N_TOKENS, d_model)

        # Query tokens for ACTOR (13) and CRITIC (14)
        self.actor_query = nn.Parameter(torch.zeros(d_model))
        self.critic_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.actor_query, std=0.02)
        nn.init.normal_(self.critic_query, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Poke-Mask (register as buffer so it moves with .to(device))
        self.register_buffer("poke_mask", _build_poke_mask(N_TOKENS))

        # Heads
        self.policy_head = PolicyHead(d_model, cfg["n_actions"])
        self.value_head = DistributionalValueHead(
            d_model, cfg["n_atoms"], cfg["v_min"], cfg["v_max"]
        )

    def forward(
        self,
        int_ids: torch.Tensor,
        float_feats: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        int_ids    : (batch, n_tokens, 7)    — integer IDs for embedding lookup
        float_feats: (batch, n_tokens, F)    — float features per token
        legal_mask : (batch, n_actions)      — 1.0 = legal

        Returns:
          log_probs : (batch, n_actions)
          value_probs: (batch, n_atoms)      — distributional value
          value     : (batch,)              — expected value (scalar)
        """
        batch = int_ids.shape[0]

        # --- Embed + project all 15 tokens ---
        embed_out = self.embeddings(int_ids)           # (B, 15, 448)
        token_in = torch.cat([embed_out, float_feats], dim=-1)  # (B, 15, 448+F)

        # Flatten, project, unflatten
        B, T, D = token_in.shape
        token_proj = self.token_proj(token_in.view(B * T, D)).view(B, T, -1)  # (B, 15, d_model)

        # Add positional embeddings
        pos_ids = torch.arange(T, device=int_ids.device)
        token_proj = token_proj + self.pos_emb(pos_ids).unsqueeze(0)

        # Replace query token projections with learned query vectors
        token_proj[:, ACTOR_IDX] = token_proj[:, ACTOR_IDX] * 0 + self.actor_query
        token_proj[:, CRITIC_IDX] = token_proj[:, CRITIC_IDX] * 0 + self.critic_query

        # --- Transformer ---
        out = self.transformer(token_proj, mask=self.poke_mask)  # (B, 15, d_model)

        # --- Heads ---
        actor_out = out[:, ACTOR_IDX]      # (B, d_model)
        critic_out = out[:, CRITIC_IDX]    # (B, d_model)

        log_probs = self.policy_head(actor_out, legal_mask)
        value_probs, value = self.value_head(critic_out)

        return log_probs, value_probs, value

    @torch.no_grad()
    def act(
        self,
        int_ids: torch.Tensor,
        float_feats: torch.Tensor,
        legal_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or argmax action. Used during rollout collection.
        Returns: action (B,), log_prob (B,), value (B,)
        """
        log_probs, _, value = self.forward(int_ids, float_feats, legal_mask)
        if deterministic:
            action = log_probs.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=log_probs).sample()
        log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
        return action, log_prob, value
