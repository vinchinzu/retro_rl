# TASK T4-ASSIST-WK: Cut Wounded Knee damage (assisted)

## Recipe step
policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — WK / Leatherhead knobs only (**one** group)
- residual: `docs/tasks/T4-ASSIST-WK-residual.md`

## Context
- Baseline stage damage **579** after Raph cadence (was 1,159 Leo-era).
- Stall Y-quantize + elevated `0xb0` B+Y escape already happy-path-neutral.

## Read first
- `docs/BASELINE_METRICS.md`
- `scripts/run_stage7_segment.py`

## Do
1. Probe WK continuous-faithful under emergency assist.
2. One knob; residual → dry-run if improved.

## Acceptance
- [ ] One-knob residual
- [ ] No STATUS self-apply

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.run_stage7_segment --help
```
