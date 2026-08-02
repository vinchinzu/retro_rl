# TASK SM-CLEAN-MORPH: Continuous power-on → Morph (Clean)

## Recipe step
compose + record (prefix baseline)

## Model
Luna

## Wave type
implement

## Own files only
- optional thin notes under `docs/routes/` if needed
- residual: `docs/tasks/SM-CLEAN-MORPH-residual.md`
- **do not** change morph controller geometry unless clean dies and residual
  names one knob; then serialize + re-verify assisted morph

Depends: `SM-CLEAN-ARTIFACTS`, `SM-CLEAN-CLI`, `SM-CLEAN-INTEGRITY` (or
manual `--report` path + flags if infra partial).

## Context
- Morph has no ammo unlock → clean should nearly match assisted (ammo assist
  already writes 0 on this prefix).
- Validates artifact isolation + clean integrity before Bomb Torizo.

## Read first
- `docs/CLEAN_TRACK.md`
- `docs/routes/START_TO_MORPH.md`
- `routes/continuous.py` (`run_start_to_morph`)

## Do
1. Record clean morph with isolated report path / `--clean`.
2. Assert integrity: success, 0 loads, 0 progression/capacity, 0 resource writes.
3. Residual → `SM-CLEAN-BOMBS` if green; economy card if RED.

## Acceptance
- [x] `recordings/start_to_morph_clean.json` (or equivalent) green
- [x] Assisted `start_to_morph` artifacts untouched
- [x] Clean resource writes all zero

**GREEN 2026-08-02** — 27,074f, `morph_ball_acquired`, residual
[`SM-CLEAN-MORPH-residual.md`](SM-CLEAN-MORPH-residual.md).

## Verify commands
```bash
uv run python super_metroid/scripts/record/continuous.py \
  --to morph --clean --no-video
# or explicit --report path if --clean not landed
```
