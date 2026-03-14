"""
PPO update loop with C51 distributional value loss.

Mixed precision: forward passes run in bf16 on CUDA (Ampere tensor cores),
gradients and optimizer state remain in fp32.  C51 projection (scatter_add)
stays in fp32 throughout to avoid precision issues.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MODEL_CONFIG
from config.training_config import TRAINING_CONFIG
from pokebot.training.replay_buffer import RolloutBuffer


class PPOTrainer:
    """
    Performs PPO updates on a PokeTransformer actor-critic model.

    C51 distributional value loss:
      - Atom support z is fixed: v_min..v_max with n_atoms atoms.
      - Target distribution: project lambda-returns onto the atom grid.
      - Loss: cross-entropy(projected_target, model_value_probs).
    """

    def __init__(self, model: nn.Module, cfg: dict = TRAINING_CONFIG):
        self.model = model
        self.cfg = cfg

        mc = MODEL_CONFIG
        self.v_min = mc["v_min"]
        self.v_max = mc["v_max"]
        self.n_atoms = mc["n_atoms"]
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        # Atom support — registered on model device later
        self.z_support = torch.linspace(self.v_min, self.v_max, self.n_atoms)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["lr_start"],
            eps=1e-5,
        )

        # LR scheduler: linear decay from lr_start → lr_end over total_steps
        total_updates = (
            cfg["total_steps"]
            // (cfg["num_parallel_envs"] * cfg["rollout_steps"])
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=cfg["lr_end"] / cfg["lr_start"],
            total_iters=total_updates,
        )

        # Mixed precision: GradScaler for bf16 on CUDA (no-op on CPU)
        # bf16 doesn't need dynamic loss scaling (no inf/nan issues), but
        # GradScaler is kept for fp16 fallback compatibility.
        _use_amp = (next(model.parameters()).device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=_use_amp)
        self._use_amp = _use_amp

        self._update_count = 0

    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _entropy_coeff(self) -> float:
        cfg = self.cfg
        progress = min(
            self._update_count * cfg["num_parallel_envs"] * cfg["rollout_steps"]
            / cfg["entropy_anneal_steps"],
            1.0,
        )
        return cfg["entropy_coeff_start"] + progress * (
            cfg["entropy_coeff_end"] - cfg["entropy_coeff_start"]
        )

    # ------------------------------------------------------------------

    def update(self, buf: RolloutBuffer) -> dict:
        """
        Run ppo_epochs passes of PPO on the concatenated rollout buffer.
        Returns a dict of scalar loss metrics.
        """
        device = self.device
        T = buf.T

        # Move data to GPU once (fp32 — kept as-is for advantages/returns)
        int_ids    = torch.from_numpy(buf.obs_int_ids).to(device)   # (T, 15, 8)
        float_f    = torch.from_numpy(buf.obs_float).to(device)     # (T, 15, F)
        legal_m    = torch.from_numpy(buf.legal_masks).to(device)   # (T, A)
        actions    = torch.from_numpy(buf.actions).to(device)       # (T,)
        old_lp     = torch.from_numpy(buf.log_probs).to(device)     # (T,)
        advantages = torch.from_numpy(buf.advantages).to(device)    # (T,)
        returns    = torch.from_numpy(buf.returns).to(device)       # (T,)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Precompute C51 projected targets from lambda-returns (stays fp32)
        z = self.z_support.to(device)                               # (n_atoms,)
        proj_targets = self._project_returns(returns, z)            # (T, n_atoms)

        # Accumulate losses
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates = 0

        _amp_dtype = torch.bfloat16 if self._use_amp else torch.float32

        for _ in range(self.cfg["ppo_epochs"]):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, self.cfg["minibatch_size"]):
                idx = perm[start : start + self.cfg["minibatch_size"]]
                if len(idx) < 2:
                    continue

                # Forward pass under bf16 autocast
                with torch.autocast(device_type=device.type, dtype=_amp_dtype,
                                    enabled=self._use_amp):
                    log_probs_new, value_probs_new, _ = self.model(
                        int_ids[idx], float_f[idx], legal_m[idx]
                    )

                    # --- Policy loss (clipped PPO) ---
                    lp_new = log_probs_new.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                    ratio  = torch.exp(lp_new - old_lp[idx])
                    adv    = advantages[idx]
                    clip_e = self.cfg["clip_epsilon"]
                    L_clip = torch.min(
                        ratio * adv,
                        torch.clamp(ratio, 1 - clip_e, 1 + clip_e) * adv,
                    ).mean()

                    # --- Entropy bonus ---
                    entropy = torch.distributions.Categorical(logits=log_probs_new).entropy().mean()

                    # --- C51 value loss (cast value_probs to fp32 for numerical stability) ---
                    tgt = proj_targets[idx]                             # (B, n_atoms) fp32
                    vp  = value_probs_new.float()                       # fp32 for cross-entropy
                    v_loss = -(tgt * torch.log(vp.clamp(min=1e-8))).sum(-1).mean()

                    loss = (
                        -L_clip
                        + self.cfg["value_coeff"] * v_loss
                        - self._entropy_coeff() * entropy
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                # Unscale before grad clip so the clip threshold is in the original scale
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["max_grad_norm"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += L_clip.item()
                total_value_loss  += v_loss.item()
                total_entropy     += entropy.item()
                n_updates         += 1

        self.scheduler.step()
        self._update_count += 1

        denom = max(n_updates, 1)
        return {
            "policy_loss": -total_policy_loss / denom,
            "value_loss":   total_value_loss  / denom,
            "entropy":      total_entropy     / denom,
            "entropy_coeff": self._entropy_coeff(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    # ------------------------------------------------------------------

    def _project_returns(self, returns: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Project scalar lambda-returns onto the C51 atom grid.

        For each return G, distribute its mass to the two neighbouring atoms
        using linear interpolation (Bellemare et al., 2017).

        returns: (T,)
        z:       (n_atoms,)
        → (T, n_atoms) float32 probability distribution
        """
        T = returns.shape[0]
        n = self.n_atoms
        proj = torch.zeros(T, n, device=returns.device, dtype=torch.float32)

        G = returns.clamp(self.v_min, self.v_max)          # (T,)
        b = (G - self.v_min) / self.delta_z                # (T,) float index
        lo = b.long().clamp(0, n - 2)                      # lower atom index
        hi = (lo + 1).clamp(0, n - 1)                      # upper atom index
        up_frac = b - lo.float()                           # upper fraction
        lo_frac = 1.0 - up_frac

        proj.scatter_add_(1, lo.unsqueeze(1), lo_frac.unsqueeze(1))
        proj.scatter_add_(1, hi.unsqueeze(1), up_frac.unsqueeze(1))
        return proj
