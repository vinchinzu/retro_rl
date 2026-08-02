# TASK T4-CLEAN-S6: Skull & Crossbones Clean multi-entry suite

## Recipe step
probe suite

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/probe_stage6_clean.py` (create if missing)
- `policy.py` — one Skull-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S6-residual.md`

## Context
- Stage byte **5**. Duo Bebop/Rocksteady left-flank.
- **Never enable global pizza seek** (soft-locks this stage).

## Read first
- `docs/CLEAN_PLAYBOOK.md`
- `scripts/run_stage6_segment.py`

## Do
1. Scaffold heal=none multi-entry suite.
2. Scope pizza seek by stage allowlist only.
3. 0 e-heals, 0 lives lost through stage_advance.

## Acceptance
- [ ] Suite green or residual with one next knob
- [ ] Global pizza seek remains disabled
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage6_clean --suite
```
