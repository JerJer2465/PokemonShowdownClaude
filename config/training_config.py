"""PPO training hyperparameters."""

TRAINING_CONFIG = {
    "total_steps": 250_000_000,
    "num_parallel_envs": 32,
    "rollout_steps": 256,           # Steps per env per rollout → 64 * 256 = 16,384 per cycle
    "inference_batch_size": 32,     # GPU inference server: max obs per forward pass
    "ppo_epochs": 4,
    "minibatch_size": 2048,
    "clip_epsilon": 0.2,
    "entropy_coeff_start": 0.01,
    "entropy_coeff_end": 0.005,
    "entropy_anneal_steps": 100_000_000,
    "value_coeff": 0.5,
    "max_grad_norm": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "lr_start": 3e-4,
    "lr_end": 1e-5,
    "opponent_pool_size": 20,
    "self_play_prob": 0.8,       # Fraction of games vs current self (rest vs pool)
    "checkpoint_every": 100,     # Updates between checkpoints
    "eval_every": 200,           # Updates between win-rate evaluations
    "opponent_mcts_ms": 0,       # MCTS search time per move (ms) — 0 = disabled (use heuristic/random mix)
    "opponent_mcts_prob": 0.0,   # Fraction of episodes using MCTS opponent (0 = all heuristic/random)
    # Stage 3 self-play league (passed via --selfplay flag to train_ppo.py)
    "selfplay_heuristic_prob": 0.2,  # 20% smart heuristic anchor episodes
    "selfplay_latest_prob": 0.2,     # 20% vs latest-self episodes
    # remaining 60% = random pool checkpoint
    "battle_format": "gen4randombattle",
    "device": "cuda",            # "cuda" on Windows with 3080, "cpu" on Mac for testing
}
