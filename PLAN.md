# Pokemon Showdown Bot — Gen 4 Random Battle Plan

## Why Gen 4?

- **Simplest competitive gen**: No Dynamax, no Megas, no Z-moves, no Terastallization, no Terrain
- **Proven target**: Jett Wang's MIT thesis achieved 1756 Glicko on Gen 4 Random with RL+MCTS
- **poke-engine default**: The Python bindings default to gen4, and gen4 is the most mature/tested
- **Smaller scope**: ~250 species in Gen 4 randbats vs ~457 in Gen 8, fewer moves/abilities/items
- **Action space stays at 10**: 4 moves + 6 switches (no Dynamax means no extra actions)

## What Changes From Gen 8 → Gen 4

| Aspect | Gen 8 (current code) | Gen 4 (target) |
|--------|---------------------|----------------|
| poke-engine install | `--features poke-engine/gen8` | default (gen4) or `--features poke-engine/gen4` |
| Randbats data | `gen8randombattle.json` (457 species) | Need `gen4randombattle.json` (~250 species) |
| Dynamax | Exists (unused) | Doesn't exist — remove all Dynamax obs features |
| Terrain | 4 terrain types | Doesn't exist — remove terrain obs features |
| Weather | 5 types + harsh/heavy | 4 types only (Sun, Rain, Sand, Hail) |
| Physical/Special split | Per-move | Per-**type** (Gen 1-3) — wait, Gen 4 introduced the split, so per-move is correct |
| Abilities | ~250 | ~120 |
| Items | ~200 | ~80 |
| Moves | ~800 | ~460 |
| Battle format | `gen8randombattle` | `gen4randombattle` |
| Volatile statuses | 20+ | Fewer (no mega-related ones) |
| Screens | Light Screen, Reflect, Aurora Veil | Light Screen, Reflect only (no Aurora Veil) |

## Current State Assessment

The project scaffold is complete but has **never been trained or tested**:
- PokeTransformer actor-critic (d_model=256, 6 layers)
- poke-engine Gymnasium env with obs builder and reward shaper
- PPO trainer with C51 distributional value head
- DUCT MCTS for inference-time search
- poke-env ShowdownPlayer for real ladder play
- All training scripts (BC, PPO, eval, ladder)

**Critical bugs and gaps exist that block training** (detailed in Phase 0).

---

## Phase 0: Critical Fixes & Gen 4 Migration

### 0.1 — Reinstall poke-engine for Gen 4
The current poke-engine is compiled with gen8 features. Reinstall with gen4:
```bash
conda activate pokeEnv
pip uninstall poke-engine -y
pip install poke-engine  # default is gen4
# OR if default doesn't work:
pip install poke-engine --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
```
**Verify**: After install, the Weather enum should NOT have `SNOW`, `HARSH_SUN`, `HEAVY_RAIN`. Terrain should still exist but won't be used.

### 0.2 — Install missing dependencies
```bash
conda activate pokeEnv
pip install ray>=2.9.0 tqdm pyyaml requests
```

### 0.3 — Download Gen 4 randbats data
Get `gen4randombattle.json` from the Showdown data repo:
```
https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/random-battles/gen4/sets.json
```
Save to `data/gen4randombattle.json`. Update all references from `gen8randombattle.json`.

### 0.4 — Create `gen4_base_stats.json`
The `RandbatsGenerator` loads base stats from a JSON but the file doesn't exist. Without it, all Pokemon base stats default to 80, making them **indistinguishable**.

**Action**: Write `scripts/build_gen4_data.py` that:
1. Fetches Gen 4 Pokemon base stats + types from Showdown's `pokedex.ts` data or a bundled JSON
2. Saves to `data/gen4_base_stats.json` with format: `{"bulbasaur": {"hp": 45, "attack": 49, ..., "types": ["Grass", "Poison"]}, ...}`
3. Also regenerate vocab indices (`species_index.json`, `move_index.json`, etc.) for Gen 4 only

### 0.5 — Create move metadata lookup table
In `poke_engine_env.py:309-316`, all moves have `basePower=0, type="Normal", category="physical"` — move features are useless during training.

**Action**: Build `data/gen4_move_data.json` with `{move_id: {basePower, type, category, priority, pp}}`. Modify `_pe_side_to_dict()` to look up move metadata from this table when building obs.

### 0.6 — Simplify obs_builder for Gen 4
Remove Gen 8-specific features that don't exist in Gen 4:
- Remove **Dynamax** features: `is_dynamaxed`, `can_dynamax`, `dynamax_turns` (saves 6 floats per Pokemon)
- Remove **Terrain** from field encoding (no terrains in Gen 4; saves 13 floats)
- Remove **Aurora Veil** from screens (Gen 7+ only)
- Remove **Snow** weather variant
- Keep everything else (weather, hazards, screens, status, volatile, etc. all exist in Gen 4)

This reduces FLOAT_DIM_PER_POKEMON and FIELD_DIM, which is good — smaller input = faster training.

### 0.7 — Update config files for Gen 4
- `model_config.py`: Update vocab sizes (`n_species`, `n_moves`, `n_abilities`, `n_items`) after rebuilding vocabs. Keep `n_actions=10` (unchanged).
- `training_config.py`: Change `battle_format` to `"gen4randombattle"`.
- `mcts_config.py`: No changes needed.

### 0.8 — Fix `train_bc.py` cross-entropy bug
`train_bc.py:147` calls `F.cross_entropy(log_probs, targets)` but `log_probs` is already log-softmaxed. `F.cross_entropy` expects raw logits.

**Fix**: Change to `F.nll_loss(log_probs, targets)`.

### 0.9 — Fix PPO C51 value loss bug
`ppo_trainer.py:137` calls `F.cross_entropy(value_probs_new, tgt)` where `value_probs_new` is already softmaxed. `F.cross_entropy` expects logits.

**Fix**: Have the distributional value head also return logits. Use `F.cross_entropy(value_logits, tgt)` in the PPO trainer.

### 0.10 — Update RandbatsGenerator for Gen 4
- Point to `gen4randombattle.json` instead of `gen8randombattle.json`
- Point to `gen4_base_stats.json` instead of `gen8_base_stats.json`
- Remove Dynamax-related team generation code
- Gen 4 randbats data format may differ slightly from Gen 8 — verify JSON structure

### 0.11 — Update ShowdownPlayer for Gen 4
- Change default `battle_format` to `gen4randombattle`
- Remove Dynamax handling
- Remove terrain handling from `battle_to_obs_dict()`

### 0.12 — Verify `bc_init.pt`
Check if the existing checkpoint is trained or a placeholder. If untrained (or trained on Gen 8 data), delete it and retrain in Phase 1.

---

## Phase 1: Behavioral Cloning (~1-2 hours GPU)

### 1.1 — Smoke test the pipeline
Before training, verify the end-to-end pipeline works:
```bash
python -c "
from pokebot.env.poke_engine_env import PokeEngineEnv
env = PokeEngineEnv()
obs, _ = env.reset()
print('int_ids shape:', obs['int_ids'].shape)
print('float_feats shape:', obs['float_feats'].shape)
print('legal_mask shape:', obs['legal_mask'].shape)
for i in range(10):
    obs, r, done, _, info = env.step(env.action_space.sample())
    if done: obs, _ = env.reset()
print('Pipeline OK')
"
```

### 1.2 — Run behavioral cloning
```bash
python scripts/train_bc.py --steps 500000 --envs 4 --device cuda
```
**Target**: >60% action agreement with SimpleHeuristic. If <50%, there's a bug.

### 1.3 — Evaluate BC checkpoint
```bash
python scripts/eval_checkpoint.py --checkpoint checkpoints/bc_init.pt --n_games 200
```
Should beat RandomPlayer >80% and have some wins vs Heuristic.

---

## Phase 2: PPO Self-Play Training (~8h to 3 days GPU)

### 2.1 — Short validation run (50M steps)
```bash
python scripts/train_ppo.py --bc_init checkpoints/bc_init.pt --envs 64 --device cuda --steps 50000000
```
Validate that loss metrics look reasonable before committing to a long run.

### 2.2 — Full training run (250M steps)
```bash
python scripts/train_ppo.py --resume checkpoints/latest.pt --envs 64 --device cuda --steps 250000000
```

### 2.3 — Monitor metrics
- **Policy loss**: Small and stable (0.01-0.1)
- **Value loss**: Decreasing over time
- **Entropy**: Gradual decrease from ~0.01 to ~0.005
- **Win rate vs heuristic**: Check periodically via eval_checkpoint.py

### 2.4 — Training improvements to add
- **TensorBoard logging** for live monitoring
- **Periodic evaluation** vs heuristic/random during training loop
- **Win-rate-gated curriculum**: save checkpoint only when win rate improves

---

## Phase 3: Strengthen the Bot

### 3.1 — Scale up the model
Current: d_model=256, 6 layers (~2-3M params). Scale to:
- d_model=512, 8 layers (~10-15M params) — still fits on 3080
- Research (Metamon) shows larger models significantly outperform smaller ones

### 3.2 — Improve reward function
Current: faint differential + HP advantage. Add:
- **Hazard advantage** (Stealth Rock, Spikes setup/clear)
- **Status advantage** (inflicting Burns/Para/Toxic)
- **Stat boost momentum** (Swords Dance, Calm Mind, etc.)
- Keep coefficients small (0.02-0.05) to avoid reward hacking

### 3.3 — Enhance observation encoding
- **Type effectiveness** features (move type vs opponent types → expected multiplier)
- **Speed tier** feature (who outspeeds given current stats + boosts)
- **Threat assessment** (can opponent OHKO/2HKO our active?)
- **Team preview info** (if applicable to the format)

### 3.4 — Improve self-play diversity
- Add **diverse team sampling** during self-play (not just random — vary compositions)
- Add **league of historical agents** with Elo-based matchmaking
- Train against **multiple heuristic opponents** (not just SimpleHeuristic)

---

## Phase 4: MCTS at Inference Time

This is the key differentiator that pushed Jett Wang to 1756 Glicko.

### 4.1 — Implement proper determinization
For real battles, the opponent's team is hidden. Implement:
- Sample K plausible opponent teams from Gen 4 randbats distribution
- Constrain samples using revealed info (species seen, moves used)
- Run MCTS on each determinization, average the visit counts

### 4.2 — Optimize MCTS speed
- **Batch leaf evaluation**: Accumulate leaf nodes, evaluate in one GPU batch
- **`torch.compile()`** for model inference
- **Root parallelism**: Run determinizations in parallel (multiprocessing)
- Target: 200+ simulations within the PS timer (~10 seconds)

### 4.3 — Tune MCTS hyperparameters
- `n_simulations`: Start at 200, scale up with time budget
- `c_puct`: Tune between 1.0-3.0
- `n_determinizations`: 10-20 for real games
- `temperature`: Near-zero for competitive play (0.1)

---

## Phase 5: Online Ladder Testing

### 5.1 — Create Pokemon Showdown account
Register a bot account on PS for `gen4randombattle`.

### 5.2 — Local testing first
```bash
# Start local PS server
npx pokemon-showdown
# Test vs random
python scripts/run_ladder.py --local --vs_random --n_games 50 --checkpoint checkpoints/latest.pt --format gen4randombattle
```

### 5.3 — Ladder games
```bash
python scripts/run_ladder.py --username BotName --password Pass --n_games 50 --checkpoint checkpoints/latest.pt --format gen4randombattle --deterministic
```

### 5.4 — MCTS-augmented ladder play
For maximum strength, run with MCTS search enabled at decision time.

**Target**: 1600+ Glicko initially, 1800+ after full training and MCTS.

---

## Priority Order

**Do first (blockers)**:
1. Reinstall poke-engine for gen4 (0.1)
2. Install ray + other deps (0.2)
3. Download gen4 randbats data (0.3)
4. Build gen4_base_stats.json + gen4_move_data.json (0.4, 0.5)
5. Simplify obs_builder for gen4 (0.6)
6. Fix training bugs (0.8, 0.9)
7. Update all configs and references (0.7, 0.10, 0.11)

**Then train**:
8. Smoke test pipeline (1.1)
9. BC training (1.2)
10. PPO training (2.1-2.2)

**Then improve**:
11. Scale model (3.1)
12. Better rewards/obs (3.2-3.3)
13. MCTS at inference (4.1-4.3)
14. Online ladder (5.1-5.4)

---

## Expected Timeline

| Phase | Task | Est. Time |
|-------|------|-----------|
| 0 | Gen 4 migration + critical fixes | 3-5 hours coding |
| 1 | BC training | 1-2 hours GPU |
| 2 | PPO training (50M steps) | 8-12 hours GPU |
| 2+ | PPO training (250M steps) | 2-3 days GPU |
| 3 | Architecture/reward improvements | 4-8 hours coding |
| 4 | MCTS implementation | 4-8 hours coding |
| 5 | Online ladder testing | 2-4 hours |

**Milestone targets**:
- After Phase 1: >80% vs RandomPlayer
- After Phase 2 (50M): >70% vs Heuristic
- After Phase 2 (250M): >85% vs Heuristic
- After Phase 4 (MCTS): 1600+ Glicko on ladder
- Full optimization: 1800+ Glicko (beat Wang's 1756)
