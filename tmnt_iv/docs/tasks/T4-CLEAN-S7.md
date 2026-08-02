# TASK T4-CLEAN-S7: Wounded Knee Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage7_clean.py` (create if missing)
- `policy.py` — one WK-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S7-residual.md`

## Context
- Stage byte **6**. Raph cadence; stacked `0xb0` jump-slash; stall Y-quantize.
- Assisted bucket **579** damage after Raph route.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `scripts/run_stage7_segment.py`

## Do
1. Scaffold heal=none multi-entry suite (prefer Raph continuous-faithful).
2. 0 e-heals, 0 lives lost through Leatherhead clear.
3. One knob residual if RED.

## Acceptance
- [ ] Suite green or residual with next card
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage7_clean --suite
```
