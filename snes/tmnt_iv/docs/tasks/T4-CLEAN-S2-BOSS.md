# TASK T4-CLEAN-S2-BOSS: Metalhead pizza-only clear

## Recipe step
probe suite (single entry)

## Model
Flash / Gemini

## Wave type
implement

## Own files only
- residual note only if re-verify RED: `docs/tasks/T4-CLEAN-S2-residual.md`

## Context
- **Already GREEN** historically (`Boss2` stage_advance, pizza-only).
- Re-run only after Alleycat policy KEEP landings (stabilize child).

## Read first
- `docs/tasks/CLEAN_LADDER.md`

## Do
1. Run Boss2-only probe (or suite and read Boss2 row).
2. GREEN if `outcome=stage_advance`, 0 e-heals, lives not decreased by death.
3. If RED: residual one boss-local knob only; do not thrash wave knobs.

## Acceptance
- [ ] Boss2 stage_advance pizza-only **or** residual with one next knob
- [ ] No STATUS edit

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Boss2
```

## Status
**done** (keep re-checking after S2 knobs via STAB).
