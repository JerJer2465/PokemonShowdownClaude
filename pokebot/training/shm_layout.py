"""
Shared memory layout constants and view helpers for zero-copy IPC.

All shapes and byte sizes are computed from obs_builder constants (FLOAT_DIM_PER_POKEMON).
See make_obs_views / make_rollout_views for the actual layout.

obs_shm layout (per worker, per-step):
  [0           : INT_IDS_BYTES           )  int_ids     (15, 8)           int64
  [INT_IDS_BYTES : +FLOAT_FEATS_BYTES    )  float_feats (15, FLOAT_DIM)  float32
  [... : +LEGAL_MASK_BYTES               )  legal_mask  (10,)            float32

res_shm layout (per worker, per-step):
  [0:4)  action   int32
  [4:8)  log_prob float32
  [8:12) value    float32

rollout_shm: T steps of obs + actions/logprobs/values/rewards/dones + GAE output.
Byte sizes computed dynamically by rollout_shm_bytes(T).
"""

from __future__ import annotations

import numpy as np
from pokebot.env.obs_builder import FLOAT_DIM_PER_POKEMON

# ------------------------------------------------------------------ per-step obs/res

# Observation shapes
INT_IDS_SHAPE    = (15, 8)
FLOAT_FEATS_SHAPE = (15, FLOAT_DIM_PER_POKEMON)
LEGAL_MASK_SHAPE = (10,)

# Byte sizes
INT_IDS_BYTES     = 15 * 8 * 8
FLOAT_FEATS_BYTES = 15 * FLOAT_DIM_PER_POKEMON * 4
LEGAL_MASK_BYTES  = 10 * 4
OBS_BYTES         = INT_IDS_BYTES + FLOAT_FEATS_BYTES + LEGAL_MASK_BYTES  # 21040
RES_BYTES         = 4 + 4 + 4           # 12 (action int32, log_prob f32, value f32)


def make_obs_views(shm):
    """Return (int_ids, float_feats, legal_mask) numpy views into obs SharedMemory."""
    int_ids    = np.ndarray(INT_IDS_SHAPE,    dtype=np.int64,   buffer=shm.buf, offset=0)
    float_feats = np.ndarray(FLOAT_FEATS_SHAPE, dtype=np.float32, buffer=shm.buf, offset=INT_IDS_BYTES)
    legal_mask = np.ndarray(LEGAL_MASK_SHAPE, dtype=np.float32, buffer=shm.buf, offset=INT_IDS_BYTES + FLOAT_FEATS_BYTES)
    return int_ids, float_feats, legal_mask


def make_res_views(shm):
    """Return (action, log_prob, value) numpy views into res SharedMemory."""
    action   = np.ndarray((1,), dtype=np.int32,   buffer=shm.buf, offset=0)
    log_prob = np.ndarray((1,), dtype=np.float32, buffer=shm.buf, offset=4)
    value    = np.ndarray((1,), dtype=np.float32, buffer=shm.buf, offset=8)
    return action, log_prob, value


# ------------------------------------------------------------------ per-rollout buffer

def rollout_shm_bytes(T: int) -> int:
    """Compute byte size of rollout SharedMemory for T steps."""
    # obs_int_ids: T * 840 (int64)
    obs_int_bytes   = T * INT_IDS_BYTES
    # obs_float: T * 20160 (float32)
    obs_float_bytes = T * FLOAT_FEATS_BYTES
    # legal_masks: T * 40 (float32)
    legal_bytes     = T * LEGAL_MASK_BYTES
    # actions: T * 8 (int64)
    actions_bytes   = T * 8
    # log_probs: T * 4 (float32)
    logp_bytes      = T * 4
    # values: T * 4 (float32)
    val_bytes       = T * 4
    # rewards: T * 4 (float32)
    rew_bytes       = T * 4
    # dones: T * 1 (uint8) + 3 padding for alignment
    done_bytes      = T * 1
    done_pad        = (4 - done_bytes % 4) % 4  # pad to 4-byte boundary
    # advantages: T * 4 (float32)
    adv_bytes       = T * 4
    # returns: T * 4 (float32)
    ret_bytes       = T * 4

    return (obs_int_bytes + obs_float_bytes + legal_bytes
            + actions_bytes + logp_bytes + val_bytes + rew_bytes
            + done_bytes + done_pad + adv_bytes + ret_bytes)


def make_rollout_views(shm, T: int):
    """
    Return numpy views into a rollout SharedMemory block.

    Returns:
        (obs_int_ids, obs_float, legal_masks, actions, log_probs,
         values, rewards, dones, advantages, returns)
    """
    obs_int_bytes   = T * INT_IDS_BYTES
    obs_float_bytes = T * FLOAT_FEATS_BYTES
    legal_bytes     = T * LEGAL_MASK_BYTES
    actions_bytes   = T * 8
    logp_bytes      = T * 4
    val_bytes       = T * 4
    rew_bytes       = T * 4
    done_bytes      = T * 1
    done_pad        = (4 - done_bytes % 4) % 4

    off_int   = 0
    off_float = off_int   + obs_int_bytes
    off_legal = off_float + obs_float_bytes
    off_act   = off_legal + legal_bytes
    off_logp  = off_act   + actions_bytes
    off_val   = off_logp  + logp_bytes
    off_rew   = off_val   + val_bytes
    off_done  = off_rew   + rew_bytes
    off_adv   = off_done  + done_bytes + done_pad
    off_ret   = off_adv   + T * 4

    obs_int_ids  = np.ndarray((T, 15, 8),                    dtype=np.int64,   buffer=shm.buf, offset=off_int)
    obs_float    = np.ndarray((T, 15, FLOAT_DIM_PER_POKEMON), dtype=np.float32, buffer=shm.buf, offset=off_float)
    legal_masks  = np.ndarray((T, 10),      dtype=np.float32, buffer=shm.buf, offset=off_legal)
    actions      = np.ndarray((T,),         dtype=np.int64,   buffer=shm.buf, offset=off_act)
    log_probs    = np.ndarray((T,),         dtype=np.float32, buffer=shm.buf, offset=off_logp)
    values       = np.ndarray((T,),         dtype=np.float32, buffer=shm.buf, offset=off_val)
    rewards      = np.ndarray((T,),         dtype=np.float32, buffer=shm.buf, offset=off_rew)
    dones        = np.ndarray((T,),         dtype=np.uint8,   buffer=shm.buf, offset=off_done)
    advantages   = np.ndarray((T,),         dtype=np.float32, buffer=shm.buf, offset=off_adv)
    returns      = np.ndarray((T,),         dtype=np.float32, buffer=shm.buf, offset=off_ret)

    return (obs_int_ids, obs_float, legal_masks, actions, log_probs,
            values, rewards, dones, advantages, returns)
