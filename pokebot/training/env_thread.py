"""
Lightweight env worker thread for GPU-server-based PPO rollout collection.

Each EnvThread owns one PokeEngineEnv and collects T rollout steps by:
  1. Encoding obs locally (fast NumPy ops, GIL released for most of it)
  2. Submitting to InferenceServer for GPU action selection (blocks briefly)
  3. Stepping the env (poke-engine Rust, GIL released)
  4. Writing transitions to a local RolloutBuffer

The opponent uses simple_heuristic_opponent (CPU, no inference needed).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import torch

from pokebot.env.poke_engine_env import PokeEngineEnv, simple_heuristic_opponent
from pokebot.env.obs_builder import ObsBuilder
from pokebot.training.replay_buffer import RolloutBuffer, make_empty_buffer, compute_gae

if TYPE_CHECKING:
    from pokebot.training.inference_server import InferenceServer


class EnvThread(threading.Thread):
    """
    Worker thread that collects one rollout of T steps using the GPU inference server.

    Call start_rollout() to begin a collection cycle.
    The thread signals done_event when T steps have been collected and GAE computed.
    Retrieve the result via get_buffer().
    """

    def __init__(
        self,
        thread_id: int,
        server: "InferenceServer",
        rollout_steps: int,
        gamma: float,
        gae_lambda: float,
    ):
        super().__init__(daemon=True, name=f"EnvThread-{thread_id}")
        self.thread_id = thread_id
        self.server = server
        self.T = rollout_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Each thread has its own env + obs builder
        self.env = PokeEngineEnv(opponent_policy=simple_heuristic_opponent)
        self.obs_builder = ObsBuilder()

        # Synchronization: main thread sets this event to trigger a rollout
        self._start_event = threading.Event()
        # Thread sets this event when the rollout buffer is ready
        self._done_event = threading.Event()

        self._buffer: RolloutBuffer | None = None
        self._stop = False
        self._obs, _ = self.env.reset()

    # ------------------------------------------------------------------
    # Control interface (called from main thread)
    # ------------------------------------------------------------------

    def start_rollout(self):
        """Signal this thread to begin collecting a new rollout."""
        self._done_event.clear()
        self._start_event.set()

    def wait_done(self):
        """Block until this thread's rollout is complete."""
        self._done_event.wait()

    def get_buffer(self) -> RolloutBuffer:
        """Return the completed rollout buffer (call after wait_done)."""
        assert self._buffer is not None
        return self._buffer

    def stop(self):
        """Ask the thread to exit after the current rollout (or immediately)."""
        self._stop = True
        self._start_event.set()

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            # Wait for the next rollout signal
            self._start_event.wait()
            self._start_event.clear()
            if self._stop:
                break

            self._buffer = self._collect_rollout()
            self._done_event.set()

    def _collect_rollout(self) -> RolloutBuffer:
        buf = make_empty_buffer(self.T)
        obs = self._obs

        for t in range(self.T):
            int_ids    = obs["int_ids"]     # (15, 7)
            float_feats = obs["float_feats"] # (15, F)
            legal_mask = obs["legal_mask"]  # (10,)

            # GPU inference (blocks until batch is processed)
            action, log_prob, value = self.server.infer(int_ids, float_feats, legal_mask)

            next_obs, reward, done, _, _ = self.env.step(action)

            buf.obs_int_ids[t] = int_ids
            buf.obs_float[t]   = float_feats
            buf.legal_masks[t] = legal_mask
            buf.actions[t]     = action
            buf.log_probs[t]   = log_prob
            buf.values[t]      = value
            buf.rewards[t]     = reward
            buf.dones[t]       = done

            if done:
                next_obs, _ = self.env.reset()

            obs = next_obs

        self._obs = obs

        # Bootstrap last value for GAE
        if buf.dones[-1]:
            last_value = 0.0
        else:
            _, _, last_value = self.server.infer(
                obs["int_ids"], obs["float_feats"], obs["legal_mask"]
            )

        compute_gae(buf, last_value, self.gamma, self.gae_lambda)
        return buf
