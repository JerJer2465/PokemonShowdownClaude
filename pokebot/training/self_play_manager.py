"""
Opponent pool for self-play: maintains a FIFO buffer of past checkpoints and
samples opponents for each rollout worker.
"""

from __future__ import annotations

import io
import random
from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class SelfPlayManager:
    """
    Manages a pool of past model checkpoints for self-play training.

    Sampling policy:
      - With probability `self_play_prob` (default 0.8): return current weights.
      - Otherwise: sample uniformly from the pool of past checkpoints.
    """

    def __init__(
        self,
        pool_size: int = 20,
        self_play_prob: float = 0.8,
    ):
        self.pool_size = pool_size
        self.self_play_prob = self_play_prob
        # Each entry: (update_n, weights_bytes)
        self._pool: deque[tuple[int, bytes]] = deque()

    def add_checkpoint(self, model: nn.Module, update_n: int) -> None:
        """Serialize model weights and add to pool (FIFO eviction)."""
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        weights_bytes = buf.getvalue()

        self._pool.append((update_n, weights_bytes))
        if len(self._pool) > self.pool_size:
            self._pool.popleft()

    def sample_opponent(self, current_weights: bytes) -> bytes:
        """
        Return weights bytes for an opponent.
        80% of the time: current weights (self-play).
        20% of the time: random past checkpoint.
        Falls back to current if pool is empty.
        """
        if not self._pool or random.random() < self.self_play_prob:
            return current_weights
        _, weights_bytes = random.choice(list(self._pool))
        return weights_bytes

    def pool_size_current(self) -> int:
        return len(self._pool)
