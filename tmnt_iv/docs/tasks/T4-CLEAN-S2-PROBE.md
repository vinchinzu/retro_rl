# TASK T4-CLEAN-S2-PROBE: Alleycat Clean suite baseline (no policy)

## Recipe step
probe suite

## Model
Flash / Gemini

## Wave type
implement

## Own files only
- residual: `docs/tasks/T4-CLEAN-S2-residual.md` (Suite table only)

## Context
- **No code.** Quote numbers only from JSON under `recordings/stage2_clean_track/`.
- Clean pizza-only is hard; this card only measures.

## Read first
- `docs/tasks/CLEAN_LADDER.md`
- `docs/tasks/T4-CLEAN-S2-residual.md`

## Do
1. Run suite command below.
2. Update residual **Suite** table from `clean_suite.json` (`passed`/`failed`/per-entry).
3. Set residual Outcome RED or PARTIAL from JSON only.
4. **Next card ID:** prefer `T4-CLEAN-S2-REACH` or `T4-CLEAN-S2-EDGE` if Stage2 still life_loss.

## Do not
- Edit `policy.py`, `STATUS.md`, QUEUE gates
- Invent frames/damage not in JSON
- Change more than the residual Suite + Outcome + Next action blocks

## Acceptance
- [ ] `clean_suite.json` regenerated this session
- [ ] Residual Suite table matches JSON field-for-field
- [ ] No policy / STATUS churn

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite
```

## Done when
Residual Suite table is truthful. Planner/reviewer may schedule REACH/EDGE.
