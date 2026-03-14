"""
Rollout buffer for PPO: stores one rollout from one env worker and computes
Generalized Advantage Estimates (GAE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pokebot.env.obs_builder import FLOAT_DIM_PER_POKEMON, N_TOKENS
from config.model_config import MODEL_CONFIG


N_ACTIONS = MODEL_CONFIG["n_actions"]


@dataclass
class RolloutBuffer:
    """Stores T transitions from one episode/rollout."""

    # Observations
    obs_int_ids:  np.ndarray   # (T, 15, 8)   int64
    obs_float:    np.ndarray   # (T, 15, F)   float32
    legal_masks:  np.ndarray   # (T, A)       float32

    # Actions & log-probs from the policy that collected the data
    actions:      np.ndarray   # (T,)         int64
    log_probs:    np.ndarray   # (T,)         float32

    # Value estimates from the critic at collection time
    values:       np.ndarray   # (T,)         float32

    # Environment signals
    rewards:      np.ndarray   # (T,)         float32
    dones:        np.ndarray   # (T,)         bool

    # GAE output — filled by compute_gae()
    advantages:   np.ndarray = field(default_factory=lambda: np.empty(0))
    returns:      np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def T(self) -> int:
        return len(self.actions)


def make_empty_buffer(T: int) -> RolloutBuffer:
    """Allocate a buffer for T timesteps."""
    F = FLOAT_DIM_PER_POKEMON
    return RolloutBuffer(
        obs_int_ids=np.zeros((T, N_TOKENS, 8),  dtype=np.int64),
        obs_float  =np.zeros((T, N_TOKENS, F),  dtype=np.float32),
        legal_masks=np.zeros((T, N_ACTIONS),    dtype=np.float32),
        actions    =np.zeros(T,                 dtype=np.int64),
        log_probs  =np.zeros(T,                 dtype=np.float32),
        values     =np.zeros(T,                 dtype=np.float32),
        rewards    =np.zeros(T,                 dtype=np.float32),
        dones      =np.zeros(T,                 dtype=bool),
    )


def compute_gae(
    buf: RolloutBuffer,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> RolloutBuffer:
    """
    Compute GAE advantages and lambda-returns in-place and return buf.

    last_value: V(s_{T+1}) bootstrapped from the critic at the end of the rollout.
                Should be 0.0 if the last step was terminal.
    """
    T = buf.T
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value * (1.0 - float(buf.dones[t]))
        else:
            next_value = buf.values[t + 1] * (1.0 - float(buf.dones[t]))
        delta = buf.rewards[t] + gamma * next_value - buf.values[t]
        last_gae = delta + gamma * lam * (1.0 - float(buf.dones[t])) * last_gae
        advantages[t] = last_gae

    buf.advantages = advantages
    buf.returns = advantages + buf.values
    return buf


def buffer_from_shm_views(views: tuple) -> RolloutBuffer:
    """
    Build a RolloutBuffer by COPYING from rollout SharedMemory views.
    views = (obs_int_ids, obs_float, legal_masks, actions, log_probs,
             values, rewards, dones, advantages, returns)
    The copy is necessary so the worker can safely overwrite shm for the next rollout.
    """
    (obs_int_ids, obs_float, legal_masks, actions, log_probs,
     values, rewards, dones, advantages, returns) = views
    return RolloutBuffer(
        obs_int_ids=obs_int_ids.copy(),
        obs_float  =obs_float.copy(),
        legal_masks=legal_masks.copy(),
        actions    =actions.astype(np.int64),
        log_probs  =log_probs.copy(),
        values     =values.copy(),
        rewards    =rewards.copy(),
        dones      =dones.astype(bool),
        advantages =advantages.copy(),
        returns    =returns.copy(),
    )


def concatenate_buffers(buffers: list[RolloutBuffer]) -> RolloutBuffer:
    """
    Merge a list of RolloutBuffers (from multiple workers) into one flat buffer.
    Each buffer must have had compute_gae() called first.
    """
    return RolloutBuffer(
        obs_int_ids=np.concatenate([b.obs_int_ids for b in buffers], axis=0),
        obs_float  =np.concatenate([b.obs_float   for b in buffers], axis=0),
        legal_masks=np.concatenate([b.legal_masks for b in buffers], axis=0),
        actions    =np.concatenate([b.actions     for b in buffers], axis=0),
        log_probs  =np.concatenate([b.log_probs   for b in buffers], axis=0),
        values     =np.concatenate([b.values      for b in buffers], axis=0),
        rewards    =np.concatenate([b.rewards     for b in buffers], axis=0),
        dones      =np.concatenate([b.dones       for b in buffers], axis=0),
        advantages =np.concatenate([b.advantages  for b in buffers], axis=0),
        returns    =np.concatenate([b.returns     for b in buffers], axis=0),
    )
