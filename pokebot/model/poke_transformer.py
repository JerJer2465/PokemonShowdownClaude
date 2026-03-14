"""
PokeTransformer: Actor-Critic Transformer for Pokemon battle decisions.

Architecture:
  1. Per-token MLP projects (embed_dim + float_dim) → d_model
  2. Learned positional embeddings for all 15 token slots
  3. 6-layer Pre-LN Transformer with Poke-Mask:
       - ACTOR token (13) cannot see CRITIC token (14) and vice-versa
       - State tokens (0-12) see each other freely
       - Mask is a float additive bias (-inf at blocked pairs) so SDPA
         can use the efficient/flash backend (bool mask forces slow math backend)
  4. Policy head reads from ACTOR output → log-probs over actions
  5. Distributional value head reads from CRITIC output → C51 distribution
"""

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


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with fused QKV and F.scaled_dot_product_attention.

    Uses SDPA directly instead of nn.MultiheadAttention so that:
    - Float additive attention bias is supported (enables flash/efficient backends)
    - No data-dependent control flow (CUDA-graph compatible)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.wqkv = nn.Linear(d_model, 3 * d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, D)
        attn_bias: (1, 1, T, T) float additive mask, -inf at blocked positions
        """
        # Pre-LN self-attention
        h = self.ln1(x)
        B, T, D = h.shape
        head_dim = D // self.n_heads
        qkv = self.wqkv(h).view(B, T, 3, self.n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(attn)

        # Pre-LN FFN
        x = x + self.ff(self.ln2(x))
        return x


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

        # Transformer layers (manual SDPA for fast attention + CUDA graph compat)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Attention bias: float additive mask, -inf at Actor↔Critic
        attn_bias = torch.zeros(1, 1, N_TOKENS, N_TOKENS)
        attn_bias[0, 0, ACTOR_IDX, CRITIC_IDX] = float("-inf")
        attn_bias[0, 0, CRITIC_IDX, ACTOR_IDX] = float("-inf")
        self.register_buffer("attn_bias", attn_bias)

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
        int_ids    : (batch, n_tokens, 8)    — integer IDs for embedding lookup
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
        for layer in self.layers:
            token_proj = layer(token_proj, attn_bias=self.attn_bias)

        # --- Heads ---
        actor_out = token_proj[:, ACTOR_IDX]      # (B, d_model)
        critic_out = token_proj[:, CRITIC_IDX]    # (B, d_model)

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
