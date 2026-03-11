"""MCTS hyperparameters (used at evaluation / ladder time)."""

MCTS_CONFIG = {
    "n_simulations": 400,
    "n_determinizations": 20,
    "sims_per_det": 20,
    "c_puct": 1.5,
    "max_depth": 10,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "temperature": 0.1,         # Near-greedy action selection
    "time_budget_sec": 10.0,    # Hard limit per move (PS timer allows ~10s safely)
}
