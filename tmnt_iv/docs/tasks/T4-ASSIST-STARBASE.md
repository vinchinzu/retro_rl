# TASK T4-ASSIST-STARBASE: Cut Starbase damage (assisted)

## Recipe step
policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — Starbase hover / launch / Foot knobs only (**one** group)
- residual: `docs/tasks/T4-ASSIST-STARBASE-residual.md`

## Context
- Baseline stage damage **749** (16.0%).
- Launch guard prevents enemyless stall; hover Foot need jump-slash.
- Prefer continuous-faithful Raph Starbase mids over Leo states.

## Read first
- `docs/BASELINE_METRICS.md`
- `docs/STATUS.md` (Starbase notes)
- `scripts/run_stage9_segment.py`

## Do
1. Probe Starbase entry under emergency assist.
2. One knob (hover cadence / range / stall guard).
3. Residual → `T4-ASSIST-DRYRUN` if improved.

## Acceptance
- [ ] One-knob residual with metrics
- [ ] Launch soft-lock still prevented
- [ ] No STATUS self-apply

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage9_segment --help
```
