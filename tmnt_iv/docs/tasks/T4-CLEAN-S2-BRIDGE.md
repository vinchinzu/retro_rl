# TASK T4-CLEAN-S2-BRIDGE: Continuous-faithful Alleycat entry (Clean)

## Recipe step
probe / policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — one Alleycat-local knob only if RED
- residual: `docs/tasks/T4-CLEAN-S2-residual.md`

## Context
- Entry: `--from-stage1-clear` (Stage1_Clear → Alleycat live).
- Clean continuous-faithful is **harder** than checkpoint; soft-lock/timeout
  counts as RED (not “almost clear”).
- Checkpoint green does **not** imply bridge green (path RNG / wave timing).

## Read first
- `docs/tasks/CLEAN_LADDER.md`
- `docs/CLEAN_PLAYBOOK.md`
- residual hits / timeout reasons from last suite

## Do
1. Run bridge probe.
2. GREEN: `stage_advance`, 0 e-heals, no life_loss, no timeout soft-lock.
3. RED: residual names failure (life_loss vs timeout/stall); one next knob.
4. Do not claim SUITE until CKPT+BRIDGE both green.

## Do not
- STATUS “bridge CLEAR” without this probe’s JSON
- Mid-wave far pizza chase

## Acceptance
- [ ] stage_advance **or** RED residual with mode + one next card
- [ ] JSON evidence path cited

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --from-stage1-clear
```
