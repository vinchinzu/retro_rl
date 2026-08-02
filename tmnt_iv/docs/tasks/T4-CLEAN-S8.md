# TASK T4-CLEAN-S8: Neon Night Riders Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage8_clean.py` (create if missing)
- `policy.py` — one Neon-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S8-residual.md`

## Context
- Stage byte **7**. Mode-7: fight near-band only `y≥140`; filter props.
- Krang left-flank poke.

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `scripts/run_stage8_segment.py`

## Do
1. Scaffold heal=none multi-entry suite.
2. 0 e-heals, 0 lives lost; no vanishing-point chase.
3. Residual one knob if RED.

## Acceptance
- [ ] Suite green or residual with next card
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage8_clean --suite
```
