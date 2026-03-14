"""
Phase 2: PPO Self-Play Training.

Uses a centralized GPU inference server + multiprocessing env workers for maximum
GPU utilization. Workers run poke-engine + obs encoding in true parallel across CPU
cores (separate GILs). All model inference is batched on the GPU.

IPC design (zero-copy everywhere):
    per-step:  obs/res through SharedMemory; 1-byte Pipe signals to/from server
    per-rollout: transitions written directly to rollout SharedMemory;
                 done_pipe carries only 1 byte (not 5MB pickled RolloutBuffer!)
                 main ack_pipe signals worker to start next rollout

Architecture:
    64 Workers ──obs_shm──► InferenceServer (GPU, batched) ──res_shm──► Workers
               ──obs_pipe────────────────────────────────── res_pipe──►
               ──rollout_shm───────────────────────────────────────────► Main
               ──done_pipe─────────────────────────────────────────────► Main
               ◄─ack_pipe──────────────────────────────────────────────── Main

Usage:
    python scripts/train_ppo.py [--steps 50000000] [--envs 32] [--bc_init checkpoints/bc_init.pt]
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: Heavy imports (torch, model, trainer) are inside main() so that
# Windows-spawned child processes do NOT load CUDA DLLs when re-importing
# this module. Each worker process only uses the stdlib imports above.


def save_checkpoint(model, update: int, out_dir: str) -> str:
    import torch
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"checkpoint_{update:06d}.pt")
    torch.save({"model_state": model.state_dict(), "update": update}, path)
    latest = os.path.join(out_dir, "latest.pt")
    torch.save({"model_state": model.state_dict(), "update": update}, latest)
    return path


def load_bc_init(model, path: str):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded BC init from {path} (update {ckpt.get('update', '?')})", flush=True)


def _quick_eval(model_state_dict, n_games: int = 100) -> dict:
    """
    Evaluate current weights against a random opponent using the local env.
    Runs in-process (no subprocess) — call only when workers are idle or
    from a thread.  Returns {win_rate, avg_turns}.
    """
    import random
    import torch
    from pokebot.model.poke_transformer import PokeTransformer
    from pokebot.env.poke_engine_env import PokeEngineEnv

    device = torch.device("cpu")
    model = PokeTransformer().to(device)
    model.load_state_dict({k: v.cpu() for k, v in model_state_dict.items()})
    model.eval()

    def _rand_opp(obs):
        legal = obs.get("legal_actions", list(range(10)))
        return random.choice(legal)

    env = PokeEngineEnv(opponent_policy=_rand_opp)
    wins = total_turns = 0
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        turns = 0
        while not done:
            int_ids = torch.from_numpy(obs["int_ids"]).unsqueeze(0)
            float_f = torch.from_numpy(obs["float_feats"]).unsqueeze(0)
            legal_m = torch.from_numpy(obs["legal_mask"]).unsqueeze(0)
            with torch.no_grad():
                log_probs, _, _ = model(int_ids, float_f, legal_m)
            action = int(torch.distributions.Categorical(logits=log_probs).sample().item())
            obs, reward, done, _, _ = env.step(action)
            turns += 1
        wins += int(reward > 0)
        total_turns += turns
    return {"win_rate": wins / n_games, "avg_turns": total_turns / n_games}


def main():
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from config.training_config import TRAINING_CONFIG
    from pokebot.model.poke_transformer import PokeTransformer
    from pokebot.training.ppo_trainer import PPOTrainer
    from pokebot.training.inference_server import InferenceServer
    from pokebot.training.env_worker import run_worker
    from pokebot.training.replay_buffer import concatenate_buffers, buffer_from_shm_views
    from pokebot.training.shm_layout import OBS_BYTES, RES_BYTES, rollout_shm_bytes, make_rollout_views

    # Reduce Windows timer resolution to 1ms for lower scheduling latency
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int, default=TRAINING_CONFIG["total_steps"])
    parser.add_argument("--envs",    type=int, default=TRAINING_CONFIG["num_parallel_envs"])
    parser.add_argument("--bc_init", type=str, default=None)
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--device",  type=str, default=TRAINING_CONFIG["device"])
    parser.add_argument("--batch",   type=int, default=TRAINING_CONFIG["inference_batch_size"],
                        help="Max inference batch size for the GPU server")
    parser.add_argument("--logdir",  type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--eval_games", type=int, default=100,
                        help="Games per periodic eval vs random (0 to disable)")
    parser.add_argument("--selfplay", action="store_true",
                        help="Stage 3: self-play league (pool + latest + heuristic anchor)")
    parser.add_argument("--selfplay_heuristic_prob", type=float, default=0.2,
                        help="Fraction of self-play episodes vs smart heuristic anchor")
    parser.add_argument("--selfplay_latest_prob", type=float, default=0.2,
                        help="Fraction of self-play episodes vs latest checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = dict(TRAINING_CONFIG)
    cfg["num_parallel_envs"] = args.envs
    cfg["total_steps"] = args.steps
    cfg["device"] = str(device)

    T = cfg["rollout_steps"]
    steps_per_update = args.envs * T
    total_updates = args.steps // steps_per_update

    print(f"PPO (zero-copy SharedMemory): {args.envs} workers x {T} steps "
          f"= {steps_per_update:,} transitions/update", flush=True)
    print(f"Device: {device}  |  inference batch: {args.batch}  |  total updates: {total_updates:,}", flush=True)

    # ------------------------------------------------------------------ model
    model = PokeTransformer().to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params", flush=True)

    # TensorBoard writer — run_name includes envs + device + mode for easy comparison
    mode_tag = "selfplay" if args.selfplay else "fixed"
    run_name = f"ppo_{mode_tag}_e{args.envs}_{device.type}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))

    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        start_update = ckpt.get("update", 0)
        print(f"Resumed from {args.resume} (update {start_update})", flush=True)
    elif args.bc_init:
        load_bc_init(model, args.bc_init)

    trainer = PPOTrainer(model, cfg)

    # ------------------------------------------------------------------ shared memory
    obs_shms     = [SharedMemory(create=True, size=OBS_BYTES) for _ in range(args.envs)]
    res_shms     = [SharedMemory(create=True, size=RES_BYTES) for _ in range(args.envs)]
    r_bytes      = rollout_shm_bytes(T)
    rollout_shms = [SharedMemory(create=True, size=r_bytes) for _ in range(args.envs)]

    # Build rollout views for main process to read from (zero-copy)
    rollout_views = [make_rollout_views(shm, T) for shm in rollout_shms]

    print(f"SharedMemory: obs={OBS_BYTES}B, res={RES_BYTES}B, rollout={r_bytes//1024}KB per worker "
          f"({args.envs * r_bytes // (1024*1024)}MB total)", flush=True)

    # ------------------------------------------------------------------ Pipe IPC (1-byte signals)
    # Per-worker Pipes for inference server (obs/result signals)
    obs_pipe_pairs  = [mp.Pipe(duplex=False) for _ in range(args.envs)]
    res_pipe_pairs  = [mp.Pipe(duplex=False) for _ in range(args.envs)]
    # Per-worker Pipes for rollout sync (done/ack signals)
    done_pipe_pairs = [mp.Pipe(duplex=False) for _ in range(args.envs)]
    ack_pipe_pairs  = [mp.Pipe(duplex=False) for _ in range(args.envs)]

    server_obs_ends  = [p[0] for p in obs_pipe_pairs]    # server reads
    server_res_ends  = [p[1] for p in res_pipe_pairs]    # server writes
    worker_obs_ends  = [p[1] for p in obs_pipe_pairs]    # workers write
    worker_res_ends  = [p[0] for p in res_pipe_pairs]    # workers read
    main_done_ends   = [p[0] for p in done_pipe_pairs]   # main reads "rollout done"
    worker_done_ends = [p[1] for p in done_pipe_pairs]   # workers write "rollout done"
    main_ack_ends    = [p[1] for p in ack_pipe_pairs]    # main writes "ack"
    worker_ack_ends  = [p[0] for p in ack_pipe_pairs]    # workers read "ack"

    stop_event = mp.Event()

    # ------------------------------------------------------------------ GPU inference server
    inference_model = PokeTransformer().to(device)
    inference_model.load_state_dict(model.state_dict())
    inference_model.eval()

    server = InferenceServer(
        inference_model, device,
        server_obs_ends,
        server_res_ends,
        obs_shms=obs_shms,
        res_shms=res_shms,
        max_batch=args.batch,
    )
    server.start()

    # ------------------------------------------------------------------ env worker processes
    processes = [
        mp.Process(
            target=run_worker,
            args=(
                i,
                obs_shms[i].name,
                res_shms[i].name,
                rollout_shms[i].name,
                worker_obs_ends[i],
                worker_res_ends[i],
                worker_done_ends[i],
                worker_ack_ends[i],
                stop_event,
                T,
                cfg["gamma"],
                cfg["gae_lambda"],
                cfg.get("opponent_mcts_ms", 0),
                cfg.get("opponent_mcts_prob", 0.0),
                # Self-play league (Stage 3)
                args.out_dir if args.selfplay else "",
                args.selfplay_heuristic_prob,
                args.selfplay_latest_prob,
            ),
            daemon=True,
            name=f"EnvWorker-{i}",
        )
        for i in range(args.envs)
    ]
    for p in processes:
        p.start()

    print(f"Started {args.envs} env worker processes", flush=True)

    _SIGNAL = b"\x00"
    _WIN_LIMIT = 63
    _done_conn_to_id = {conn: i for i, conn in enumerate(main_done_ends)}

    def _wait_chunked(conns, timeout):
        """Wait on connections in chunks of 63 (Windows limit)."""
        ready = []
        for start in range(0, len(conns), _WIN_LIMIT):
            chunk = conns[start : start + _WIN_LIMIT]
            ready.extend(mp.connection.wait(chunk, timeout=timeout))
        return ready

    t0 = time.time()

    _t_rollout_total = 0.0
    _t_read_total    = 0.0
    _t_ppo_total     = 0.0
    _n_logged        = 0

    try:
        for update in range(start_update, total_updates):
            _t0 = time.perf_counter()

            # ---- wait for all workers to finish their rollout --------
            # Collect 1-byte done signals; workers wrote transitions to rollout_shm
            pending = list(main_done_ends)
            while pending:
                ready = _wait_chunked(pending, timeout=1.0)
                for conn in ready:
                    conn.recv_bytes()
                    pending.remove(conn)

            _t1 = time.perf_counter()

            # ---- read rollout data from shared memory (zero-copy) ----
            buffers = [buffer_from_shm_views(rollout_views[i]) for i in range(args.envs)]

            # ---- ack all workers: safe to start next rollout ---------
            for i in range(args.envs):
                main_ack_ends[i].send_bytes(_SIGNAL)

            _t2 = time.perf_counter()

            # ---- PPO update on GPU ----------------------------------
            model.train()
            merged = concatenate_buffers(buffers)
            stats  = trainer.update(merged)
            model.eval()

            _t3 = time.perf_counter()

            _t_rollout_total += _t1 - _t0
            _t_read_total    += _t2 - _t1
            _t_ppo_total     += _t3 - _t2
            _n_logged        += 1

            # ---- push updated weights to inference server -----------
            server.update_weights(model.state_dict())

            # ---- checkpoint -----------------------------------------
            if (update + 1) % cfg["checkpoint_every"] == 0:
                path = save_checkpoint(model, update + 1, args.out_dir)
                print(f"Saved checkpoint -> {path}", flush=True)

            # ---- logging --------------------------------------------
            if (update + 1) % 10 == 0:
                elapsed = time.time() - t0
                total_steps = (update + 1 - start_update) * steps_per_update
                global_steps = (update + 1) * steps_per_update
                sps = total_steps / max(elapsed, 1)
                avg_rollout = _t_rollout_total / _n_logged * 1000
                avg_read    = _t_read_total    / _n_logged * 1000
                avg_ppo     = _t_ppo_total     / _n_logged * 1000
                batch_stats = server.get_batch_stats()
                print(
                    f"update={update+1:6d}/{total_updates}  "
                    f"steps={total_steps/1e6:.2f}M  "
                    f"sps={sps:,.0f}  "
                    f"pi_loss={stats['policy_loss']:.4f}  "
                    f"v_loss={stats['value_loss']:.4f}  "
                    f"H={stats['entropy']:.3f}  "
                    f"lr={stats['lr']:.2e}  "
                    f"[rollout={avg_rollout:.0f}ms read={avg_read:.0f}ms ppo={avg_ppo:.0f}ms "
                    f"batch={batch_stats['avg_batch']:.1f} "
                    f"graph={batch_stats.get('graph_pct', 0):.0f}%]",
                    flush=True,
                )
                # TensorBoard scalars
                writer.add_scalar("losses/policy_loss",  stats['policy_loss'],  global_steps)
                writer.add_scalar("losses/value_loss",   stats['value_loss'],   global_steps)
                writer.add_scalar("losses/entropy",      stats['entropy'],       global_steps)
                writer.add_scalar("losses/entropy_coeff",stats['entropy_coeff'],global_steps)
                writer.add_scalar("train/sps",           sps,                   global_steps)
                writer.add_scalar("train/lr",            stats['lr'],           global_steps)
                writer.add_scalar("perf/rollout_ms",     avg_rollout,           global_steps)
                writer.add_scalar("perf/ppo_ms",         avg_ppo,               global_steps)
                _t_rollout_total = _t_read_total = _t_ppo_total = 0.0
                _n_logged = 0

            # ---- periodic eval vs random ----------------------------
            if args.eval_games > 0 and (update + 1) % cfg["eval_every"] == 0:
                global_steps = (update + 1) * steps_per_update
                print(f"Running eval ({args.eval_games} games)...", flush=True)
                eval_res = _quick_eval(
                    {k: v.cpu() for k, v in model.state_dict().items()},
                    n_games=args.eval_games,
                )
                print(
                    f"  vs_random  WR={eval_res['win_rate']*100:.1f}%  "
                    f"avg_turns={eval_res['avg_turns']:.1f}",
                    flush=True,
                )
                writer.add_scalar("eval/win_rate_vs_random", eval_res['win_rate'], global_steps)
                writer.add_scalar("eval/avg_turns",          eval_res['avg_turns'], global_steps)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...", flush=True)

    finally:
        # ---- shutdown -----------------------------------------------
        stop_event.set()
        for p in processes:
            p.join(timeout=10.0)
        server.stop()

        for shm in obs_shms + res_shms + rollout_shms:
            shm.close()
            shm.unlink()

        all_conns = (server_obs_ends + server_res_ends
                     + worker_obs_ends + worker_res_ends
                     + main_done_ends + worker_done_ends
                     + main_ack_ends + worker_ack_ends)
        for conn in all_conns:
            conn.close()

        path = save_checkpoint(model, total_updates, args.out_dir)
        print(f"Training complete. Final checkpoint -> {path}", flush=True)

        writer.close()

        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
