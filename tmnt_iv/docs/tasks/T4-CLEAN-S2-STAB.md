# TASK T4-CLEAN-S2-STAB: Alleycat stabilize (suite + assisted dry-run)

## Recipe step
stabilize

## Model
Flash / Gemini

## Wave type
stabilize

## Own files only
- residual verify paste only (optional `T4-CLEAN-S2-residual.md`)

## Context
- After any Alleycat **KEEP** knob: re-verify Clean suite + assisted continuous
  dry-run. **No new knobs.**
- Assisted dry-run must stay 0 life_losses; report time/damage/heal **deltas**
  vs STATUS baseline (00:57:19 / 4,667 / 65). Regression is a planner decision
  (revert vs accept).

## Do
1. Run stage2 Clean suite.
2. Run assisted dry-run (long).
3. Residual: suite pass count + dry-run metrics; Next = PLANNER-GATE.

## Do not
- Change policy
- Update BASELINE_METRICS / STATUS (planner)

## Acceptance
- [ ] Suite JSON + dry-run JSON paths + key metrics pasted
- [ ] Zero policy diff this session

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
```
