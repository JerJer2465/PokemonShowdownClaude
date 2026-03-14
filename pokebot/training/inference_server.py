"""
Centralized GPU inference server for PPO rollout collection (shared-memory + Pipe version).

Workers write obs to their pre-allocated SharedMemory segment, then signal via
their per-worker obs_pipe (sending only a tiny b'\\x00' byte). The server uses
multiprocessing.connection.wait() to multiplex all worker obs pipes simultaneously
(O(1) system call instead of sequential Queue.gets), reads obs from shared memory,
batches them, runs a single GPU forward pass, writes results back to per-worker
SharedMemory, then signals each worker via their res_pipe.

IPC protocol:
    worker  →  obs_pipes[i]:  b'\\x00'  (1-byte signal, tiny)
    server  →  res_pipes[i]:  b'\\x00'  (1-byte signal, tiny)
    obs data: worker → obs_shm[i]  (SharedMemory, no serialization)
    results:  server → res_shm[i]  (SharedMemory, no serialization)

Pipe vs Queue latency (Windows):
    mp.Queue one-way: ~0.7ms  → 64 signals = 45ms
    mp.Pipe  one-way: ~0.14ms → 64 signals =  9ms  (5x speedup)
"""

from __future__ import annotations

import multiprocessing.connection
import threading
import time
from typing import NamedTuple

import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn

from pokebot.training.shm_layout import make_obs_views, make_res_views

_SIGNAL = b"\x00"


class _Request(NamedTuple):
    worker_id: int


class InferenceServer:
    """
    Batched GPU inference server using shared memory + Pipe IPC.

    Usage:
        server = InferenceServer(model, device, obs_pipe_ends, res_pipe_ends,
                                 obs_shms, res_shms)
        server.start()
        # workers run in separate processes
        server.stop()
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        obs_pipe_ends: list,      # list[Connection] — server-side read ends of obs pipes
        res_pipe_ends: list,      # list[Connection] — server-side write ends of res pipes
        obs_shms: list,           # list[SharedMemory], one per worker (obs input)
        res_shms: list,           # list[SharedMemory], one per worker (result output)
        max_batch: int = 64,
        timeout_ms: float = 0.5,
    ):
        self.model = model
        self.device = device
        self.obs_pipe_ends = obs_pipe_ends   # server reads from these
        self.res_pipe_ends = res_pipe_ends   # server writes to these
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0

        # Map Connection → worker_id for fast lookup after connection.wait()
        self._conn_to_id = {conn: i for i, conn in enumerate(obs_pipe_ends)}

        # Build numpy views into each worker's shared memory (created once, reused)
        self._obs_views = [make_obs_views(shm) for shm in obs_shms]
        self._res_views = [make_res_views(shm) for shm in res_shms]

        self._running = False
        self._thread: threading.Thread | None = None
        self._weight_lock = threading.Lock()

        # Dedicated high-priority CUDA stream so inference forward passes can
        # interleave with PPO backprop (which uses the default stream).
        # Priority -1 = highest on consumer NVIDIA GPUs (Ampere/Turing).
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device, priority=-1)
        else:
            self._stream = None

    # ------------------------------------------------------------------

    def start(self):
        """Start the background inference loop thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="InferenceServer"
        )
        self._thread.start()

    def stop(self):
        """Signal the server to stop and wait for it to exit."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------

    def _run_loop(self):
        """Background thread: collect requests and dispatch batched GPU calls."""
        while self._running:
            batch = self._collect_batch()
            if batch:
                self._dispatch(batch)

    def _collect_batch(self) -> list[_Request]:
        """
        Use connection.wait() to multiplex all worker obs pipes simultaneously.
        Block until at least one worker signals, then drain up to max_batch.
        Returns empty list only if server is stopping.

        Windows WaitForMultipleObjects limit: 63 handles per call.
        We chunk the connections and merge results.
        """
        _WIN_LIMIT = 63  # Windows WaitForMultipleObjects max handles

        def _wait_chunked(conns, timeout):
            """Call connection.wait() in chunks of _WIN_LIMIT, return all ready."""
            ready = []
            for start in range(0, len(conns), _WIN_LIMIT):
                chunk = conns[start : start + _WIN_LIMIT]
                ready.extend(multiprocessing.connection.wait(chunk, timeout=timeout))
            return ready

        while self._running:
            ready = _wait_chunked(self.obs_pipe_ends, self.timeout)
            if not ready:
                continue

            # Drain signals from all ready connections (they sent b'\x00')
            batch: list[_Request] = []
            for conn in ready:
                conn.recv_bytes()  # consume the signal byte
                batch.append(_Request(self._conn_to_id[conn]))
                if len(batch) >= self.max_batch:
                    break

            # If we have room, do one more wait(0) pass to grab stragglers
            if len(batch) < self.max_batch:
                extra = _wait_chunked(self.obs_pipe_ends, 0)
                for conn in extra:
                    if len(batch) >= self.max_batch:
                        break
                    conn.recv_bytes()
                    batch.append(_Request(self._conn_to_id[conn]))

            return batch

        return []

    @torch.no_grad()
    def _dispatch(self, batch: list[_Request]):
        """Read obs from shared memory, run GPU forward, write results back."""
        n = len(batch)

        with self._weight_lock:
            # Stack obs from shared memory (zero-copy reads)
            int_ids_list    = [self._obs_views[r.worker_id][0].copy() for r in batch]
            float_feats_list = [self._obs_views[r.worker_id][1].copy() for r in batch]
            legal_mask_list = [self._obs_views[r.worker_id][2].copy() for r in batch]

            int_ids_np     = np.stack(int_ids_list)     # (n, 15, 7)
            float_feats_np = np.stack(float_feats_list) # (n, 15, F)
            legal_mask_np  = np.stack(legal_mask_list)  # (n, 10)

            if self._stream is not None:
                # Use dedicated high-priority stream to interleave with PPO's
                # default stream; synchronize before reading results back to CPU
                with torch.cuda.stream(self._stream):
                    int_ids_t     = torch.from_numpy(int_ids_np).to(self.device)
                    float_feats_t = torch.from_numpy(float_feats_np).to(self.device)
                    legal_mask_t  = torch.from_numpy(legal_mask_np).to(self.device)
                    log_probs, _, values = self.model(int_ids_t, float_feats_t, legal_mask_t)
                    actions       = torch.distributions.Categorical(logits=log_probs).sample()
                    log_prob_vals = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                self._stream.synchronize()
            else:
                int_ids_t     = torch.from_numpy(int_ids_np).to(self.device)
                float_feats_t = torch.from_numpy(float_feats_np).to(self.device)
                legal_mask_t  = torch.from_numpy(legal_mask_np).to(self.device)
                log_probs, _, values = self.model(int_ids_t, float_feats_t, legal_mask_t)
                actions       = torch.distributions.Categorical(logits=log_probs).sample()
                log_prob_vals = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            actions_cpu    = actions.cpu().numpy()
            log_probs_cpu  = log_prob_vals.cpu().numpy()
            values_cpu     = values.cpu().numpy()

        # Write results to shared memory and signal workers via Pipe
        for i, req in enumerate(batch):
            action_v, logprob_v, value_v = self._res_views[req.worker_id]
            action_v[0]  = actions_cpu[i]
            logprob_v[0] = log_probs_cpu[i]
            value_v[0]   = values_cpu[i]
            self.res_pipe_ends[req.worker_id].send_bytes(_SIGNAL)

    def update_weights(self, state_dict: dict):
        """Hot-swap model weights (call from main thread after PPO update)."""
        with self._weight_lock:
            self.model.load_state_dict(state_dict)
