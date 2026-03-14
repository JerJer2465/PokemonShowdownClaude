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

CUDA Graphs:
    After a warmup phase, the server captures a CUDA graph for the full
    max_batch forward pass. Subsequent dispatches replay the graph with
    zero-padded static buffers, avoiding kernel launch overhead (~2.4× speedup).
"""

from __future__ import annotations

import logging
import multiprocessing.connection
import threading
import time
from typing import NamedTuple

import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn

from pokebot.training.shm_layout import make_obs_views, make_res_views

log = logging.getLogger(__name__)

_SIGNAL = b"\x00"
_WARMUP_BATCHES = 10


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
        use_cuda_graph: bool = True,
    ):
        self.model = model
        self.device = device
        self.obs_pipe_ends = obs_pipe_ends   # server reads from these
        self.res_pipe_ends = res_pipe_ends   # server writes to these
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0
        self._use_cuda_graph = use_cuda_graph and device.type == "cuda"

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
        if device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device, priority=-1)
        else:
            self._stream = None

        # Batch size tracking
        self._batch_sizes: list[int] = []
        self._graph_dispatches = 0
        self._total_dispatches = 0

        # CUDA graph state
        self._graph: torch.cuda.CUDAGraph | None = None
        self._warmup_count = 0
        self._static_int: torch.Tensor | None = None
        self._static_float: torch.Tensor | None = None
        self._static_legal: torch.Tensor | None = None
        # Output buffers (filled by graph replay)
        self._static_log_probs: torch.Tensor | None = None
        self._static_values: torch.Tensor | None = None

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
        Block until at least one worker signals, then drain all ready workers.

        Relies on "natural batching": while _dispatch processes the previous
        batch (~3ms GPU), workers accumulate their signals. The next
        _collect_batch immediately finds them, giving batch sizes of ~15-20
        with 32 workers.

        Windows WaitForMultipleObjects limit: 63 handles per call.
        """
        _WIN_LIMIT = 63

        def _wait_chunked(conns, timeout):
            ready = []
            for start in range(0, len(conns), _WIN_LIMIT):
                chunk = conns[start : start + _WIN_LIMIT]
                ready.extend(multiprocessing.connection.wait(chunk, timeout=timeout))
            return ready

        while self._running:
            ready = _wait_chunked(self.obs_pipe_ends, self.timeout)
            if not ready:
                continue

            batch: list[_Request] = []
            seen = set()
            for conn in ready:
                conn.recv_bytes()
                wid = self._conn_to_id[conn]
                if wid not in seen:
                    batch.append(_Request(wid))
                    seen.add(wid)
                if len(batch) >= self.max_batch:
                    break

            # Quick non-blocking drain to grab stragglers
            if len(batch) < self.max_batch:
                extra = _wait_chunked(self.obs_pipe_ends, 0)
                for conn in extra:
                    if len(batch) >= self.max_batch:
                        break
                    conn.recv_bytes()
                    wid = self._conn_to_id[conn]
                    if wid not in seen:
                        batch.append(_Request(wid))
                        seen.add(wid)

            return batch

        return []

    @torch.no_grad()
    def _dispatch(self, batch: list[_Request]):
        """Read obs from shared memory, run GPU forward, write results back."""
        n = len(batch)
        self._batch_sizes.append(n)
        self._total_dispatches += 1

        with self._weight_lock:
            # Stack obs from shared memory (workers are blocked, shm safe to read)
            int_ids_np     = np.stack([self._obs_views[r.worker_id][0] for r in batch])
            float_feats_np = np.stack([self._obs_views[r.worker_id][1] for r in batch])
            legal_mask_np  = np.stack([self._obs_views[r.worker_id][2] for r in batch])

            if self._use_cuda_graph and self._graph is not None:
                self._graph_dispatches += 1
                log_probs, values = self._graph_replay(
                    int_ids_np, float_feats_np, legal_mask_np, n
                )
            elif self._stream is not None:
                with torch.cuda.stream(self._stream):
                    log_probs, values = self._eager_forward(
                        int_ids_np, float_feats_np, legal_mask_np
                    )
                self._stream.synchronize()
            else:
                log_probs, values = self._eager_forward(
                    int_ids_np, float_feats_np, legal_mask_np
                )

            # Sample actions from log_probs
            actions       = torch.distributions.Categorical(logits=log_probs[:n]).sample()
            log_prob_vals = log_probs[:n].gather(1, actions.unsqueeze(1)).squeeze(1)

            actions_cpu   = actions.cpu().numpy()
            log_probs_cpu = log_prob_vals.cpu().numpy()
            values_cpu    = values[:n].cpu().numpy()

            # Try to capture CUDA graph after warmup
            if (self._use_cuda_graph and self._graph is None
                    and self._warmup_count < _WARMUP_BATCHES):
                self._warmup_count += 1
                if self._warmup_count >= _WARMUP_BATCHES:
                    self._try_capture_graph()

        # Write results to shared memory and signal workers via Pipe
        for i, req in enumerate(batch):
            action_v, logprob_v, value_v = self._res_views[req.worker_id]
            action_v[0]  = actions_cpu[i]
            logprob_v[0] = log_probs_cpu[i]
            value_v[0]   = values_cpu[i]
            self.res_pipe_ends[req.worker_id].send_bytes(_SIGNAL)

    def _eager_forward(self, int_ids_np, float_feats_np, legal_mask_np):
        """Eager (non-graph) GPU forward pass. Returns (log_probs, values) on GPU."""
        int_ids_t     = torch.from_numpy(int_ids_np).to(self.device)
        float_feats_t = torch.from_numpy(float_feats_np).to(self.device)
        legal_mask_t  = torch.from_numpy(legal_mask_np).to(self.device)
        log_probs, _, values = self.model(int_ids_t, float_feats_t, legal_mask_t)
        return log_probs, values

    # ------------------------------------------------------------------
    # CUDA Graph capture and replay
    # ------------------------------------------------------------------

    def _try_capture_graph(self):
        """Capture a CUDA graph for max_batch forward pass."""
        M = self.max_batch
        dev = self.device

        # Get tensor shapes from obs views
        int_shape = self._obs_views[0][0].shape     # (15, 8)
        float_shape = self._obs_views[0][1].shape   # (15, F)
        legal_shape = self._obs_views[0][2].shape   # (10,)

        try:
            # Allocate static input buffers
            self._static_int   = torch.zeros(M, *int_shape, dtype=torch.long, device=dev)
            self._static_float = torch.zeros(M, *float_shape, dtype=torch.float32, device=dev)
            self._static_legal = torch.ones(M, *legal_shape, dtype=torch.float32, device=dev)

            # Warmup run on the dedicated stream to populate autograd caches
            with torch.cuda.stream(self._stream):
                for _ in range(3):
                    self.model(self._static_int, self._static_float, self._static_legal)
            self._stream.synchronize()

            # Capture
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self._stream):
                with torch.cuda.graph(self._graph, stream=self._stream):
                    lp, _, v = self.model(
                        self._static_int, self._static_float, self._static_legal
                    )
                    self._static_log_probs = lp
                    self._static_values = v
            self._stream.synchronize()
            log.info("CUDA graph captured successfully (max_batch=%d)", M)

        except Exception as e:
            log.warning("CUDA graph capture failed, falling back to eager: %s", e)
            self._graph = None
            self._static_int = None
            self._static_float = None
            self._static_legal = None
            self._static_log_probs = None
            self._static_values = None
            self._use_cuda_graph = False

    def _graph_replay(self, int_ids_np, float_feats_np, legal_mask_np, n):
        """Copy data into static buffers, replay CUDA graph, return results."""
        with torch.cuda.stream(self._stream):
            # Zero-fill padding slots only (not entire buffer)
            if n < self.max_batch:
                self._static_int[n:].zero_()
                self._static_float[n:].zero_()
                self._static_legal[n:].fill_(1.0)

            # Copy actual data into the first n slots
            self._static_int[:n].copy_(
                torch.from_numpy(int_ids_np).to(self.device, non_blocking=True)
            )
            self._static_float[:n].copy_(
                torch.from_numpy(float_feats_np).to(self.device, non_blocking=True)
            )
            self._static_legal[:n].copy_(
                torch.from_numpy(legal_mask_np).to(self.device, non_blocking=True)
            )

            # Replay the captured graph
            self._graph.replay()

        self._stream.synchronize()

        # Return the static output tensors (caller reads first n entries)
        return self._static_log_probs, self._static_values

    # ------------------------------------------------------------------

    def get_batch_stats(self) -> dict:
        """Return and reset batch size statistics."""
        sizes = self._batch_sizes
        graph_d = self._graph_dispatches
        total_d = self._total_dispatches
        self._batch_sizes = []
        self._graph_dispatches = 0
        self._total_dispatches = 0
        if not sizes:
            return {"avg_batch": 0, "n_dispatches": 0, "graph_pct": 0}
        return {
            "avg_batch": float(np.mean(sizes)),
            "n_dispatches": total_d,
            "graph_pct": graph_d / max(total_d, 1) * 100,
        }

    def update_weights(self, state_dict: dict):
        """Hot-swap model weights (call from main thread after PPO update)."""
        with self._weight_lock:
            self.model.load_state_dict(state_dict)
            # Invalidate CUDA graph — will be re-captured on next dispatch cycle
            if self._graph is not None:
                self._graph = None
                self._warmup_count = 0
                log.info("CUDA graph invalidated after weight update, will re-capture")
