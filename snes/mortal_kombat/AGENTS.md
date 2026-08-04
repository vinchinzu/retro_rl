# Agent Instructions — Mortal Kombat SNES Speedrun

## North-Star Goal

**~90% full tournament clear** on **normal mode**, LiuKang ladder (12 fights).
Algorithm-agnostic — multimodel chaining is the current best path, not one PPO policy.

## Research Docs (read first)

| Doc | Purpose |
|-----|---------|
| [docs/autoresearch_meta.md](docs/autoresearch_meta.md) | Why single-model failed, math for 90%, eval protocol |
| [docs/experiments.md](docs/experiments.md) | Numbered experiment backlog (P0/P1/P2) |
| [docs/ram_policy_plan.md](docs/ram_policy_plan.md) | RAM-vector MLP stack, discovery sprint, E017 pivot |
| [SPEEDRUN_PLAN.md](SPEEDRUN_PLAN.md) | Tournament structure, state inventory, multimodel design |
| [experiments/README.md](experiments/README.md) | Experiment harness stub |

## How Agents Run the Next Experiment

1. Open `docs/experiments.md` — pick highest-priority `open` experiment (usually P0).
2. If no recent baseline, run:
   ```bash
   cd mortal_kombat && ./experiments/run_experiment.sh baseline
   ```
3. Execute the experiment's training command (typically `train_speedrun.py`).
4. Eval with fixed protocol:
   ```bash
   uv run python speedrun_multimodel.py \
     --general <candidate>.zip \
     --attempts 20 \
     --tournament 100
   ```
5. Log result: `./experiments/run_experiment.sh eval models/<candidate>.zip E00N`
6. **Keep** if `full_clear_rate` improves; **discard** otherwise.
7. Update experiment status in `docs/experiments.md`.
8. On keep: update `STAGE_MODELS` in `speedrun_multimodel.py` + `model_registry.py`.

### Autoresearch-style loop

```
propose (experiments.md) → run (train) → measure (N=100 tournament) → keep/discard
```

Pattern adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Do **not** clone autoresearch into this repo — use `experiments/` harness or external clone.

## Benchmark Commands

```bash
cd mortal_kombat

# Primary metric: full clear rate (N=100 tournament sims)
uv run python speedrun_multimodel.py \
  --general mk1_fresh_ppo_final.zip \
  --tournament 100 \
  --attempts 20

# Per-stage bottleneck analysis (single model)
uv run python speedrun_test.py \
  --model models/mk1_speedrun_ppo_final.zip \
  --char LiuKang \
  --attempts 20

# Training (curriculum presets fix tier-mix failures)
uv run python train_speedrun.py --curriculum ladder --steps 8000000 --fresh
uv run python train_speedrun.py --curriculum boss --steps 10000000
uv run python train_speedrun.py --curriculum endurance --steps 8000000

# RAM-vector MLP (E017 — see docs/ram_policy_plan.md)
uv run python train_ram_ppo.py --state Fight_LiuKang --steps 1000000
uv run python speedrun_test.py --model models/mk1_ram_ppo_final.zip --ram --attempts 20

# Model registry
uv run python model_registry.py list
uv run python model_registry.py benchmark models/X.zip --level N
```

## Known Results (do not re-discover)

| Run | Eval | Notes |
|-----|------|-------|
| 8M `train_speedrun.py --fresh` | ~8% overall | M1 30%, M4 10%, M7 50%; M2–M6/E/G/S 0% |
| Training log | ~15% WR | Inflated vs eval — always benchmark |
| Default tier mix | 38% boss+endurance | Starved M2–M6 |
| `mk1_shangtsung_ppo_final` | ~60% Shang | Boss specialist works |
| `mk1_goro_ppo_final` | weak | Needs more steps |
| `mk1_fresh_ppo_final` | ~40% M1 | Good ladder base for fine-tune |

## Do-Not-Repeat

- Single model with default `full` curriculum expecting mid-ladder learning
- Trusting training win rate without `speedrun_test` / `--tournament 100`
- Fine-tuning from narrow boss specialist as general model (catastrophic forgetting)
- One-off training scripts — use `train_speedrun.py` flags
- `mk1_ppo_*` naming (deprecated)

## Script Discipline

- **Training:** `train_speedrun.py` only (`--curriculum`, `--fresh`, `--steps`)
- **Eval:** `speedrun_test.py` (per-stage), `speedrun_multimodel.py` (tournament)
- **States:** `cheat_extractor.py`, `match_manager.py`, `validate_states.py`
- **Watch:** `watch.py` (visual debug)

## Tournament Structure (SNES MK1)

```
M1-M6 → M7 (mirror) → E1 → E1B → E2 → Goro → Shang Tsung
```

12 fights, 10 stage prefixes. Goro = Endurance2B alias.

## RAM Addresses

| Variable | Address | Hex |
|----------|---------|-----|
| health (P1) | 1209 | 0x04B9 |
| enemy_health (P2) | 1211 | 0x04BB |
| timer | 290 | 0x0122 |
| continue_timer | 999 | 0x03E7 |
| p1_character | 6514 | 0x1972 |
| p1_x | 218 | 0x00DA |
| p1_y | 219 | 0x00DB |
| p2_x | 372 | 0x0174 |

Max health: 161. Win = `rounds_won >= 2 AND rounds_won > rounds_lost`.

## Multimodel Map

`speedrun_multimodel.py` → `STAGE_MODELS` overrides per stage.
Default specialists: Goro, Shang. Ladder model should cover M1–M7 + Endurance after E003/E004.
