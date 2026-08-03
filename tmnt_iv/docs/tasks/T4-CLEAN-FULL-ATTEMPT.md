# TASK T4-CLEAN-FULL-ATTEMPT: Clean continuous dry-run (expect RED)

## Recipe step
continuous

## Model
Flash / Gemini / Luna

## Wave type
implement

## Own files only
- residual: `docs/tasks/T4-CLEAN-FULL-residual.md`

## Context
- **Measurement card.** Until stages are Clean, expect fail — that is OK.
- GREEN only if hard credits + 0 e-heals + 0 iframe + 0 life_losses.
- RED residual must name **first failure stage** + route to that stage’s thin
  card (e.g. `T4-CLEAN-S2-CKPT`), not a multi-stage rewrite.

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/tasks/CLEAN_LADDER.md`

## Do
1. Run clean dry-run (long).
2. Paste integrity + stage-of-death / metrics from JSON.
3. Next card ID = one stage thin rung or PLANNER-GATE.
4. Do not edit policy unless residual already named one knob **and** card is
   upgraded by planner — default is probe-only.

## Do not
- Overwrite assisted baselines
- STATUS promote
- Stack policy knobs to “push through”

## Acceptance
- [ ] Clean report path written (`*_clean*`)
- [ ] Residual: first failure stage + one next thin card
- [ ] Assisted baseline files untouched

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run
```
