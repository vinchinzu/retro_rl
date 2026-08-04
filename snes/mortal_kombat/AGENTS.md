# Agent Instructions — Mortal Kombat (SNES)

North star: **~90% full tournament clear** on normal, LiuKang ladder
(12 fights). Multimodel chaining > single PPO. Docs:
`docs/autoresearch_meta.md`, `docs/experiments.md`, `docs/ram_policy_plan.md`,
`SPEEDRUN_PLAN.md`.

## Commands

```bash
cd mortal_kombat   # or: snes/mortal_kombat from monorepo root

# Next experiment: open docs/experiments.md → highest-priority `open` (P0)
./experiments/run_experiment.sh baseline
uv run python train_speedrun.py --curriculum ladder --steps 8000000 --fresh

# Eval (keep only if full_clear_rate improves)
uv run python speedrun_multimodel.py \
  --general <candidate>.zip --attempts 20 --tournament 100
./experiments/run_experiment.sh eval models/<candidate>.zip E00N

uv run python speedrun_test.py --model models/X.zip --char LiuKang --attempts 20
uv run python model_registry.py list
```

Loop: `propose → train → measure N=100 tournament → keep/discard`. Do **not**
clone karpathy/autoresearch into this repo.

## Do-not-repeat

- Single model + default `full` curriculum expecting mid-ladder learning
- Trust training WR without `speedrun_test` / `--tournament 100`
- Fine-tune general from narrow boss specialist (forgetting)
- One-off training scripts — use `train_speedrun.py` flags only
- Deprecated `mk1_ppo_*` naming

## Traps / structure

- Tournament: `M1–M6 → M7 → E1 → E1B → E2 → Goro → Shang` (12 fights).
- `STAGE_MODELS` in `speedrun_multimodel.py` holds per-stage specialists.
- Max health 161. Win = `rounds_won >= 2 AND rounds_won > rounds_lost`.
- Known baselines / RAM table: `docs/STATUS.md` (do not re-discover in AGENTS).
