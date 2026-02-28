# Super Mario Bros Full-Game Pipeline

Unified full-route workflow for SMB any% (warp route), including:
- `8-3`
- full `8-4` progression via segments `8-4_1..8-4_5`

Entry module: `super_mario_bros.fullgame`

Wrapper script: `super_mario_bros/run_fullgame.sh`

Hybrid optimizer entry module: `super_mario_bros.hybrid_pipeline`

## Architecture

The pipeline runs in two contexts from one manifest:

1. `standalone`
- Evaluates each segment from its own start state.
- Optimized for segment-level completion/speed checks.

2. `chained`
- Evaluates end-to-end on one emulator session (`chain-live` behavior).
- Uses chained start states (`Chained_*`) when available.

The training/profile stage scores candidate recordings per segment for both contexts, then writes:

- `super_mario_bros/optimizer/fullgame_manifest.json`

The eval stage replays exactly the selected recordings and writes:

- `super_mario_bros/optimizer/fullgame_eval_report.json`

## Implemented Optimizations

1. Context-aware recording selection
- Different artifacts are selected for standalone vs chained contexts.
- Fixes common mismatch where a recording works in one context and fails in the other.

2. Candidate ranking by completion/speed/progress
- Completed candidates ranked by fastest frame count.
- Incomplete candidates ranked by fitness/progress instead of shortest failure.

3. Chained-state-aware scoring
- Chained context scoring uses real `Chained_*` save states, not only default level states.
- Reduces false positives from unrealistic start conditions.

4. Optional parallel scoring
- `--workers N` evaluates segments in parallel processes to cut profiling overhead.

## Measured Impact (Current Assets)

Using `train` + `eval` with current recordings:

- Standalone baseline: `10/12` segments completed.
- Standalone selected: `12/12` segments completed.
- Chained baseline: `5/12` segments completed.
- Chained selected: `6/12` segments completed.

Primary chained gain came from selecting `smb_8_2/recording_001.json` instead of `recording_000.json`.

## Command Recipes

From repo root (`retro_rl/`):

```bash
# 1) Train/profile and write manifest
./super_mario_bros/run_fullgame.sh train \
  --route smb_any_percent \
  --max-candidates 8 \
  --workers 1 \
  --manifest super_mario_bros/optimizer/fullgame_manifest.json
```

```bash
# 2) Evaluate selected route (both standalone + chained)
./super_mario_bros/run_fullgame.sh eval \
  --mode both \
  --manifest super_mario_bros/optimizer/fullgame_manifest.json \
  --report super_mario_bros/optimizer/fullgame_eval_report.json
```

```bash
# 3) One-shot train+eval
./super_mario_bros/run_fullgame.sh run \
  --route smb_any_percent \
  --max-candidates 8 \
  --workers 1 \
  --manifest super_mario_bros/optimizer/fullgame_manifest.json \
  --report super_mario_bros/optimizer/fullgame_eval_report.json
```

## Hybrid Recipes (Non-Destructive)

The hybrid pipeline keeps an immutable registry of selected artifacts and
does not overwrite canonical `hillclimb_best_final.json` files.

`--selection-context` controls how candidates are scored:
- `standalone`: evaluate from each level’s default start state.
- `chained`: evaluate from `Chained_*` state overrides when available (preferred for real full-route behavior).

```bash
# Analyze existing artifacts + update registry + route best-times report
uv run python -m super_mario_bros.hybrid_pipeline analyze \
  --route smb_any_percent \
  --selection-context chained \
  --force-eval \
  --report super_mario_bros/optimizer/hybrid_report.json \
  --registry super_mario_bros/optimizer/model_registry.json
```

```bash
# Improve weakest segments with composable sub-agents
# (replay + optional ga_raw + hillclimb(_raw) + optional neuro/ppo)
uv run python -m super_mario_bros.hybrid_pipeline run \
  --route smb_any_percent \
  --selection-context chained \
  --force-eval \
  --weak-top-k 3 \
  --ga-generations 6 --ga-population 20 \
  --hill-iterations 600 --hill-raw-iterations 600 \
  --report super_mario_bros/optimizer/hybrid_report.json \
  --registry super_mario_bros/optimizer/model_registry.json
```

### Background Launcher

Use the bundled launcher to kick off heavier chained-context jobs in the background:

```bash
./super_mario_bros/run_hybrid_bg.sh
```

It launches:
- `analyze` (chained context, force-eval)
- `run` jobs for `smb_8_1`, `smb_8_2`, `smb_8_4`

Artifacts:
- logs + PID table under `super_mario_bros/optimizer/logs/hybrid_bg_<timestamp>/`
- per-job reports under `super_mario_bros/optimizer/`

Optional tuning via env vars:
- `GA_GENERATIONS`, `GA_POPULATION`, `HILL_ITERS`, `HILL_RAW_ITERS`, `MAX_CANDIDATES`

## Notes

- `chain-live` currently breaks at `8-3` with existing assets; this is the main blocker before unattended overnight full-run.
- To generate/update chained states while evaluating:
  - add `--save-states` to `eval`/`run`.

## Current Best (2026-02-27)

- Best validated full chained any% route: `25779` frames (`429.65s`)
  - Source: `super_mario_bros/optimizer/current_chain_eval.json`
- Faster standalone winners for `1-1/1-2/4-1/4-2` are folded into:
  - `super_mario_bros/optimizer/folded_selection.json`
- Chained compatibility sweep for those swaps (16 combos):
  - `super_mario_bros/optimizer/folded_combo_sweep.json`
  - Result: no fully completed chained run yet with those replacements.
