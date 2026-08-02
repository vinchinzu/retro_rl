# TASK SM-CLEAN-STAB: Stabilize Clean Bomb Torizo tip (×2)

## Recipe step
stabilize

## Model
Planner-serial (or Luna under planner)

## Wave type
stabilize

## Own files only
- residual / notes only; no geometry unless recheck RED
- `docs/tasks/SM-CLEAN-STAB-residual.md`

Depends: `SM-CLEAN-BOMBS` GREEN once.

## Do
1. Two matching clean `--to bombs` runs (no-video OK).
2. Compare frames loosely; integrity must match (zero resource writes both).
3. Residual → `SM-CLEAN-STATUS`.

## Acceptance
- [ ] Two clean integrity-green reports
- [ ] Assisted defaults untouched

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py --to bombs --clean --no-video
uv run python super_metroid/scripts/record/continuous.py --to bombs --clean --no-video \
  --report super_metroid/recordings/start_to_bomb_torizo_clean_reverify.json
```
