"""Policy head and distributional value head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MODEL_CONFIG


class PolicyHead(nn.Module):
    """Linear policy head with legal action masking."""

    def __init__(self, d_model: int = MODEL_CONFIG["d_model"], n_actions: int = MODEL_CONFIG["n_actions"]):
        super().__init__()
        self.linear = nn.Linear(d_model, n_actions)

    def forward(self, actor_token: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """
        actor_token : (batch, d_model)
        legal_mask  : (batch, n_actions)  — 1.0 legal, 0.0 illegal
        Returns     : (batch, n_actions)  — log-probabilities (log-softmax)
        """
        logits = self.linear(actor_token)
        logits = logits.masked_fill(legal_mask == 0, float("-inf"))
        return F.log_softmax(logits, dim=-1)


class DistributionalValueHead(nn.Module):
    """
    C51 distributional value head.

    Models the return distribution as a categorical distribution over
    n_atoms support points in [v_min, v_max]. This handles the high
    variance of Random Battle outcomes better than scalar regression.
    """

    def __init__(
        self,
        d_model: int = MODEL_CONFIG["d_model"],
        n_atoms: int = MODEL_CONFIG["n_atoms"],
        v_min: float = MODEL_CONFIG["v_min"],
        v_max: float = MODEL_CONFIG["v_max"],
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.linear = nn.Linear(d_model, n_atoms)

        # Support vector (not a parameter, just a buffer)
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer("support", support)

    def forward(self, critic_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        critic_token : (batch, d_model)
        Returns:
          probs  : (batch, n_atoms)  — categorical probabilities
          value  : (batch,)          — expected value = sum(probs * support)
        """
        logits = self.linear(critic_token)
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.support).sum(dim=-1)
        return probs, value

    def distributional_loss(
        self,
        probs: torch.Tensor,
        target_returns: torch.Tensor,
        gamma: float,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute C51 cross-entropy loss.

        Projects the Bellman target distribution onto the support atoms
        and computes cross-entropy against predicted probs.

        probs          : (batch, n_atoms)  — predicted distribution
        target_returns : (batch,)          — scalar bootstrap targets (GAE returns)
        gamma          : discount factor
        done           : (batch,)          — 1.0 if terminal

        Note: For simplicity in PPO we use a scalar regression fallback.
        Full distributional Bellman update requires next-state distributions.
        """
        # Clamp targets to [v_min, v_max]
        targets = target_returns.clamp(self.v_min, self.v_max)
        # Project to atom indices
        b = (targets - self.v_min) / (self.v_max - self.v_min) * (self.n_atoms - 1)
        lower = b.floor().long().clamp(0, self.n_atoms - 1)
        upper = b.ceil().long().clamp(0, self.n_atoms - 1)

        # Build target distribution
        target_dist = torch.zeros_like(probs)
        frac_upper = b - lower.float()
        frac_lower = 1.0 - frac_upper
        target_dist.scatter_add_(1, lower.unsqueeze(1), frac_lower.unsqueeze(1))
        target_dist.scatter_add_(1, upper.unsqueeze(1), frac_upper.unsqueeze(1))

        # Cross-entropy loss
        log_probs = torch.log(probs.clamp(min=1e-8))
        loss = -(target_dist * log_probs).sum(dim=-1)
        return loss.mean()
