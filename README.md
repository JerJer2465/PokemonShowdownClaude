# Pokemon Showdown Bot — Superhuman Gen 8 Random Battle

A state-of-the-art Pokemon Showdown bot using hybrid **PPO reinforcement learning + MCTS**
targeting Gen 8 Random Battle. Architecture based on:
- [Jett Wang 2024 MIT thesis](https://dspace.mit.edu/bitstream/handle/1721.1/153888/wang-jett-meng-eecs-2024-thesis.pdf) — RL+MCTS achieving 1756 Glicko (target to beat)
- [Nebraskinator/ps-ppo](https://github.com/Nebraskinator/ps-ppo) — Transformer RL reference
- [pmariglia/foul-play](https://github.com/pmariglia/foul-play) — MCTS reference using poke-engine

## Setup

### 1. Create conda environment (Mac for development, Windows for training)

```bash
conda create -n pokebot python=3.11 -y
conda activate pokebot
pip install pytest numpy gymnasium
```

### 2. Install battle simulator (Windows only — requires Rust)

```bash
# Install Rust first: https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install poke-engine for Gen 8
pip install poke-engine --config-settings="build-args=--features poke-engine/gen8 --no-default-features"
```

### 3. Install ML dependencies (Windows — GPU training)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ray poke-env tqdm pyyaml
```

### 4. Build vocabulary indices

```bash
python scripts/build_vocab.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Training (Windows with RTX 3080)

```bash
# Phase 1: Behavioral cloning from heuristic player
python scripts/train_bc.py

# Phase 2: PPO self-play
python scripts/train_ppo.py

# Phase 3: Evaluate on PS ladder
python scripts/run_ladder.py
```

## Project Structure

```
pokebot/
  env/           # Gymnasium env + observation encoder + reward
  model/         # PokeTransformer actor-critic
  training/      # PPO trainer, rollout workers, self-play
  mcts/          # DUCT MCTS + determinizer for hidden info
  players/       # poke-env ladder players
  evaluation/    # Elo tracking, baseline evaluation
config/          # Hyperparameters
data/            # Vocabulary indices + randbats set data
scripts/         # Training and evaluation entry points
tests/           # Unit tests (run on Mac without poke-engine)
```

## Key Design Decisions

- **poke-engine (Rust, Gen 8)** for fast training: 10,000–50,000 steps/sec vs 500 with PS server
- **PokeTransformer**: 6-layer Pre-LN Transformer, d_model=256, separate ACTOR/CRITIC read tokens
- **C51 distributional value head**: handles high variance of Random Battle outcomes
- **DUCT MCTS**: decoupled UCT for simultaneous Pokemon moves + determinization for hidden info
- Training target: **1800+ Glicko** on gen8randombattle ladder
