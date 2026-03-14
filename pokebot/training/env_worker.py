"""
Multiprocessing env worker for GPU-server-based PPO rollout collection.

Each worker is a separate process (own Python interpreter, own GIL).
Workers run poke-engine + obs encoding in true parallel across CPU cores.

IPC protocol (zero-copy everywhere):
    Worker writes obs        → obs_shm[worker_id]     (per-step obs input)
    Worker  →  obs_pipe:       b'\\x00'  (signal server: obs ready)
    Server  →  res_pipe:       b'\\x00'  (signal worker: result ready)
    Server writes result     → res_shm[worker_id]     (per-step result output)
    Worker writes transition → rollout_shm[worker_id] (per-step, during rollout)
    Worker  →  done_pipe:      b'\\x00'  (signal main: rollout complete)
    Main    →  ack_pipe:       b'\\x00'  (signal worker: safe to overwrite rollout_shm)

Large numpy arrays NEVER go through queues. done_pipe carries only 1 byte per rollout.

Self-play mode (GPU-batched):
    Both agent AND opponent inference go through the same GPU inference server.
    The worker reuses its obs_shm/pipe channel for opponent obs (sequential: send
    agent obs, get result, send opponent obs, get result).  No CPU model at all.
    Forced/faint switches use smart_heuristic_opponent (fast CPU fallback).
"""

from __future__ import annotations

import multiprocessing as mp


def run_worker(
    worker_id: int,
    obs_shm_name: str,
    res_shm_name: str,
    rollout_shm_name: str,
    obs_pipe,         # Connection — worker writes signal to server
    res_pipe,         # Connection — worker reads signal from server
    done_pipe,        # Connection — worker writes "rollout done" to main
    ack_pipe,         # Connection — worker reads "ok to start next rollout" from main
    stop_event: mp.Event,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    mcts_opponent_ms: int = 0,
    mcts_opponent_prob: float = 0.0,
    # Self-play league params (Stage 3)
    selfplay_pool_dir: str = "",    # Path to checkpoints dir; "" = no self-play
    selfplay_heuristic_prob: float = 0.2,  # Fraction of episodes vs smart heuristic anchor
    selfplay_latest_prob: float = 0.2, # Fraction of episodes vs latest self
    # Remaining fraction = random pool checkpoint
):
    """
    Worker process main loop.  Runs forever until stop_event is set.

    Each iteration: collect T steps into rollout_shm → compute GAE in-place
    → signal done → wait for ack → repeat.
    """
    # Imports inside the worker process (required for Windows spawn)
    # NOTE: All heavy imports are here (not at module level) so that the
    # spawned processes don't simultaneously load large DLLs at module import time.
    import ctypes
    import os
    import random
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory
    from pokebot.env.poke_engine_env import PokeEngineEnv, simple_heuristic_opponent, smart_heuristic_opponent, _random_opponent
    from pokebot.training.shm_layout import make_obs_views, make_res_views, make_rollout_views

    # Reduce Windows timer resolution to 1ms for faster process wakeup after I/O
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    _SIGNAL = b"\x00"

    # ---- Self-play setup -----------------------------------------------
    selfplay_mode = bool(selfplay_pool_dir)

    # Per-episode flag: True = use GPU for opponent, False = use heuristic
    _use_gpu_opponent = False

    if selfplay_mode:
        def _pick_opponent_type():
            """Choose opponent type for this episode. Returns True if GPU self-play."""
            nonlocal _use_gpu_opponent
            r = random.random()
            if r < selfplay_heuristic_prob:
                _use_gpu_opponent = False
            else:
                _use_gpu_opponent = True

    # ---- Attach to shared memory ---------------------------------------
    obs_shm     = SharedMemory(name=obs_shm_name)
    res_shm     = SharedMemory(name=res_shm_name)
    rollout_shm = SharedMemory(name=rollout_shm_name)

    int_ids_view, float_feats_view, legal_mask_view = make_obs_views(obs_shm)
    action_view, logprob_view, value_view = make_res_views(res_shm)

    T = rollout_steps
    (r_int_ids, r_float, r_legal, r_actions, r_logprobs,
     r_values, r_rewards, r_dones, r_adv, r_ret) = make_rollout_views(rollout_shm, T)

    # ---- Build environment ---------------------------------------------
    # Pick opponent: 50% random + 50% smart heuristic for diverse training
    def _mixed_opponent(obs_dict):
        if random.random() < 0.5:
            return _random_opponent(obs_dict)
        return smart_heuristic_opponent(obs_dict)

    env = PokeEngineEnv(
        opponent_policy=_mixed_opponent,
        mcts_opponent_ms=mcts_opponent_ms if not selfplay_mode else 0,
        mcts_opponent_prob=mcts_opponent_prob if not selfplay_mode else 0.0,
    )
    obs, _ = env.reset()

    # Apply initial self-play opponent selection
    if selfplay_mode:
        _pick_opponent_type()

    # ---- Main rollout loop ---------------------------------------------
    while not stop_event.is_set():
        for t in range(T):
            int_ids     = obs["int_ids"]
            float_feats = obs["float_feats"]
            legal_mask  = obs["legal_mask"]

            # Write obs to per-step shared memory
            np.copyto(int_ids_view,     int_ids)
            np.copyto(float_feats_view, float_feats)
            np.copyto(legal_mask_view,  legal_mask)

            # Signal inference server and wait for agent action
            obs_pipe.send_bytes(_SIGNAL)
            res_pipe.recv_bytes()

            action   = int(action_view[0])
            log_prob = float(logprob_view[0])
            value    = float(value_view[0])

            # Step environment
            if selfplay_mode and _use_gpu_opponent:
                # GPU-batched self-play: get opponent action via same GPU server
                opp_obs = env.get_opponent_obs_encoded()
                np.copyto(int_ids_view,     opp_obs["int_ids"])
                np.copyto(float_feats_view, opp_obs["float_feats"])
                np.copyto(legal_mask_view,  opp_obs["legal_mask"])
                obs_pipe.send_bytes(_SIGNAL)
                res_pipe.recv_bytes()
                opp_action = int(action_view[0])

                next_obs, reward, done, _, _ = env.step_dual(action, opp_action)
            else:
                # Heuristic opponent (or non-selfplay mode)
                next_obs, reward, done, _, _ = env.step(action)

            # Write transition directly to rollout SharedMemory (zero-copy)
            r_int_ids[t]  = int_ids
            r_float[t]    = float_feats
            r_legal[t]    = legal_mask
            r_actions[t]  = action
            r_logprobs[t] = log_prob
            r_values[t]   = value
            r_rewards[t]  = reward
            r_dones[t]    = int(done)

            if done:
                next_obs, _ = env.reset()
                if selfplay_mode:
                    _pick_opponent_type()
            obs = next_obs

        # Bootstrap last value for GAE
        np.copyto(int_ids_view,     obs["int_ids"])
        np.copyto(float_feats_view, obs["float_feats"])
        np.copyto(legal_mask_view,  obs["legal_mask"])
        obs_pipe.send_bytes(_SIGNAL)
        res_pipe.recv_bytes()
        last_value = 0.0 if r_dones[T - 1] else float(value_view[0])

        # Compute GAE in-place directly in rollout SharedMemory
        _compute_gae_inplace(r_values, r_rewards, r_dones, r_adv, r_ret,
                              last_value, gamma, gae_lambda, T)

        # Signal main: rollout is ready in rollout_shm
        done_pipe.send_bytes(_SIGNAL)

        # Wait for main to finish reading before we overwrite rollout_shm
        ack_pipe.recv_bytes()

    obs_shm.close()
    res_shm.close()
    rollout_shm.close()


def _compute_gae_inplace(values, rewards, dones, advantages, returns,
                          last_value, gamma, lam, T):
    """Compute GAE advantages and lambda-returns in-place in SharedMemory arrays."""
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        if t == T - 1:
            next_value = last_value * mask
        else:
            next_value = float(values[t + 1]) * mask
        delta = float(rewards[t]) + gamma * next_value - float(values[t])
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    for t in range(T):
        returns[t] = advantages[t] + values[t]
