# TASK T4-CLEAN-S2-CKPT: Full Stage2 checkpoint → stage_advance (Clean)

## Recipe step
probe / policy knob

## Model
Luna

## Wave type
implement

## Own files only
- `policy.py` — one Alleycat-local knob **only if** still RED after REACH wins
- residual: `docs/tasks/T4-CLEAN-S2-residual.md`

## Context
- Harder than assisted Stage2. GREEN only if pizza-only **stage_advance** from
  fight-ready `Stage2` with 0 e-heals and no life_loss.
- Prefer finishing REACH/EDGE first so this is a verify card, not a thrash card.

## Read first
- `docs/tasks/CLEAN_LADDER.md`
- `docs/tasks/T4-CLEAN-S2-residual.md`
- `docs/CLEAN_PLAYBOOK.md`

## Do
1. Run `--state Stage2` heal=none.
2. If stage_advance: residual GREEN for CKPT; Next = `T4-CLEAN-S2-BRIDGE`.
3. If life_loss: do **not** stack knobs — residual Next = REACH/EDGE with one
   failure window from hits[].

## Acceptance
- [ ] `outcome=stage_advance` **or** RED residual with one next thin card
- [ ] Metrics from JSON
- [ ] No STATUS claim of full suite

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2
```
