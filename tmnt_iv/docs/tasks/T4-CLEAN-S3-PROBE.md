# TASK T4-CLEAN-S3-PROBE: Sewer Clean suite baseline (no policy)

## Recipe step
probe suite

## Model
Flash / Gemini

## Wave type
implement

## Own files only
- residual: `docs/tasks/T4-CLEAN-S3-residual.md` (create if missing)

## Context
- **No code.** Prefer LiveHard entries. Quote JSON only.
- Clean pizza-only ≫ assisted.

## Do
1. Run suite below.
2. Write residual Suite table from `stage3_clean_track` JSON.
3. Next card: `T4-CLEAN-S3-REACH` or `T4-CLEAN-S3-BOSS` from failure mode.

## Do not
- Edit policy / STATUS
- Gate on last-life Boss3 fade

## Acceptance
- [ ] Residual matches suite JSON
- [ ] No policy churn

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite
```
