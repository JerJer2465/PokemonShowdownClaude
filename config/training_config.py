"""PPO training hyperparameters."""

TRAINING_CONFIG = {
    "total_steps": 250_000_000,
    "num_parallel_envs": 64,
    "rollout_steps": 256,       # Steps per env per rollout → 64 * 256 = 16,384 per cycle
    "ppo_epochs": 4,
    "minibatch_size": 512,
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
    "checkpoint_every": 500,     # Updates between checkpoints
    "eval_every": 1000,          # Updates between win-rate evaluations
    "battle_format": "gen8randombattle",
    "device": "cuda",            # "cuda" on Windows with 3080, "cpu" on Mac for testing
}
