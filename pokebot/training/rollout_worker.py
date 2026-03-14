"""
Parallel rollout collection via Ray remote workers.

Each RolloutWorker owns one PokeEngineEnv and collects `rollout_steps`
transitions, then returns a RolloutBuffer to the learner.
"""

from __future__ import annotations

import io
import numpy as np
import torch

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from config.model_config import MODEL_CONFIG
from config.training_config import TRAINING_CONFIG
from pokebot.env.poke_engine_env import PokeEngineEnv
from pokebot.training.replay_buffer import RolloutBuffer, make_empty_buffer, compute_gae


def _weights_to_bytes(model: torch.nn.Module) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _bytes_to_weights(model: torch.nn.Module, data: bytes) -> None:
    buf = io.BytesIO(data)
    state = torch.load(buf, map_location="cpu", weights_only=True)
    model.load_state_dict(state)


def _build_model() -> torch.nn.Module:
    from pokebot.model.poke_transformer import PokeTransformer
    return PokeTransformer()


def _make_policy_fn(model: torch.nn.Module, device: torch.device):
    """Returns a callable (obs_dict) → action_int using greedy sampling."""
    from pokebot.env.obs_builder import ObsBuilder
    obs_builder = ObsBuilder()

    @torch.no_grad()
    def policy(obs_dict: dict) -> int:
        obs = obs_builder.encode(obs_dict)
        int_ids    = torch.from_numpy(obs["int_ids"]).unsqueeze(0).to(device)
        float_feats= torch.from_numpy(obs["float_feats"]).unsqueeze(0).to(device)
        legal_mask = torch.from_numpy(obs["legal_mask"]).unsqueeze(0).to(device)
        log_probs, _, _ = model(int_ids, float_feats, legal_mask)
        action = torch.distributions.Categorical(logits=log_probs).sample()
        return int(action.item())

    return policy


# ---------------------------------------------------------------------------
# Ray worker
# ---------------------------------------------------------------------------

if RAY_AVAILABLE:
    @ray.remote(num_cpus=1)
    class RolloutWorker:
        """
        Remote Ray actor that holds a PokeEngineEnv and collects rollouts.

        On each collect_rollout() call:
          1. Load fresh model weights from the learner.
          2. Set the opponent to a (possibly older) checkpoint.
          3. Run rollout_steps steps.
          4. Compute GAE.
          5. Return RolloutBuffer (CPU numpy).
        """

        def __init__(self, worker_id: int, cfg: dict = TRAINING_CONFIG):
            self.worker_id = worker_id
            self.cfg = cfg
            self.device = torch.device("cpu")   # workers run on CPU

            # Limit PyTorch to 1 OMP thread per worker to prevent thread-count explosion
            # (N workers × default_threads would create N×16 threads for 16-core machines)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

            # Build model (weights loaded before each rollout)
            self.model = _build_model().to(self.device)
            self.model.eval()

            # Opponent model (may lag behind)
            self.opp_model = _build_model().to(self.device)
            self.opp_model.eval()

            opp_policy = _make_policy_fn(self.opp_model, self.device)
            self.env = PokeEngineEnv(opponent_policy=opp_policy)

            self._obs, _ = self.env.reset()
            self._ep_done = False

        def collect_rollout(
            self,
            model_weights: bytes,
            opp_weights: bytes,
        ) -> RolloutBuffer:
            """
            Collect `rollout_steps` transitions with the given weights.
            Returns a RolloutBuffer with GAE computed.
            """
            T = self.cfg["rollout_steps"]
            gamma = self.cfg["gamma"]
            lam   = self.cfg["gae_lambda"]

            # Load weights
            _bytes_to_weights(self.model, model_weights)
            _bytes_to_weights(self.opp_model, opp_weights)

            policy_fn = _make_policy_fn(self.model, self.device)

            buf = make_empty_buffer(T)

            obs = self._obs
            for t in range(T):
                # Encode obs for the learner policy
                int_ids_t    = torch.from_numpy(obs["int_ids"]).unsqueeze(0)
                float_feats_t= torch.from_numpy(obs["float_feats"]).unsqueeze(0)
                legal_mask_t = torch.from_numpy(obs["legal_mask"]).unsqueeze(0)

                with torch.no_grad():
                    log_probs, _, value = self.model(
                        int_ids_t, float_feats_t, legal_mask_t
                    )
                    action = torch.distributions.Categorical(logits=log_probs).sample()
                    log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze()

                a = int(action.item())
                next_obs, reward, done, _, _ = self.env.step(a)

                buf.obs_int_ids[t] = obs["int_ids"]
                buf.obs_float[t]   = obs["float_feats"]
                buf.legal_masks[t] = obs["legal_mask"]
                buf.actions[t]     = a
                buf.log_probs[t]   = float(log_prob.item())
                buf.values[t]      = float(value.item())
                buf.rewards[t]     = float(reward)
                buf.dones[t]       = done

                if done:
                    next_obs, _ = self.env.reset()

                obs = next_obs

            self._obs = obs

            # Bootstrap last value
            if buf.dones[-1]:
                last_value = 0.0
            else:
                int_ids_last    = torch.from_numpy(obs["int_ids"]).unsqueeze(0)
                float_feats_last= torch.from_numpy(obs["float_feats"]).unsqueeze(0)
                legal_mask_last = torch.from_numpy(obs["legal_mask"]).unsqueeze(0)
                with torch.no_grad():
                    _, _, last_v = self.model(
                        int_ids_last, float_feats_last, legal_mask_last
                    )
                last_value = float(last_v.item())

            compute_gae(buf, last_value, gamma, lam)
            return buf

        def ping(self) -> int:
            return self.worker_id

else:
    # Stub for environments without Ray (testing / single-process mode)
    class RolloutWorker:  # type: ignore[no-redef]
        def __init__(self, worker_id: int, cfg: dict = TRAINING_CONFIG):
            self.worker_id = worker_id
            self.cfg = cfg
            self.device = torch.device("cpu")
            self.model = _build_model()
            self.opp_model = _build_model()
            opp_policy = _make_policy_fn(self.opp_model, self.device)
            self.env = PokeEngineEnv(opponent_policy=opp_policy)
            self._obs, _ = self.env.reset()

        def collect_rollout(
            self,
            model_weights: bytes,
            opp_weights: bytes,
        ) -> RolloutBuffer:
            _bytes_to_weights(self.model, model_weights)
            _bytes_to_weights(self.opp_model, opp_weights)
            self.model.eval()
            self.opp_model.eval()

            T = self.cfg["rollout_steps"]
            buf = make_empty_buffer(T)
            obs = self._obs

            for t in range(T):
                int_ids_t    = torch.from_numpy(obs["int_ids"]).unsqueeze(0)
                float_feats_t= torch.from_numpy(obs["float_feats"]).unsqueeze(0)
                legal_mask_t = torch.from_numpy(obs["legal_mask"]).unsqueeze(0)

                with torch.no_grad():
                    log_probs, _, value = self.model(
                        int_ids_t, float_feats_t, legal_mask_t
                    )
                    action = torch.distributions.Categorical(logits=log_probs).sample()
                    log_prob = log_probs.gather(1, action.unsqueeze(1)).squeeze()

                a = int(action.item())
                next_obs, reward, done, _, _ = self.env.step(a)

                buf.obs_int_ids[t] = obs["int_ids"]
                buf.obs_float[t]   = obs["float_feats"]
                buf.legal_masks[t] = obs["legal_mask"]
                buf.actions[t]     = a
                buf.log_probs[t]   = float(log_prob.item())
                buf.values[t]      = float(value.item())
                buf.rewards[t]     = float(reward)
                buf.dones[t]       = done

                if done:
                    next_obs, _ = self.env.reset()
                obs = next_obs

            self._obs = obs
            if buf.dones[-1]:
                last_value = 0.0
            else:
                with torch.no_grad():
                    _, _, lv = self.model(
                        torch.from_numpy(obs["int_ids"]).unsqueeze(0),
                        torch.from_numpy(obs["float_feats"]).unsqueeze(0),
                        torch.from_numpy(obs["legal_mask"]).unsqueeze(0),
                    )
                last_value = float(lv.item())

            compute_gae(buf, last_value, self.cfg["gamma"], self.cfg["gae_lambda"])
            return buf
