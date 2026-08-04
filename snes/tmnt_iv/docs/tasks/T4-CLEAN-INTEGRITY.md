# TASK T4-CLEAN-INTEGRITY: Zero-assist asserts for Clean continuous

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `scripts/record_full_hard_run.py` — integrity checks + report fields
- `tests/` — assert behavior without full game run where possible
- `docs/tasks/T4-CLEAN-INTEGRITY-residual.md`

Depends: CLI flags from `T4-CLEAN-CLI` (or land together).

## Context
- Successful Clean continuous must show:
  - `health_guard_interventions == 0`
  - `final_boss_iframe_guard_frames == 0`
  - `life_losses == 0`
  - no state loads / stage writes
- Report must label intervention class **Clean** when clean mode.
- Super Metroid analog: `SM-CLEAN-INTEGRITY` (`require_clean_resources`).

## Read first
- `docs/CLEAN_TRACK.md`
- `scripts/record_full_hard_run.py` (metrics + report construction)

## Do
1. When clean mode, after run (success or fail): assert e-heals == 0 and
   iframe frames == 0; non-zero is integrity RED (exit non-zero).
2. Manifest fields: `assisted: false`, `intervention_class: "Clean"`,
   explicit zero counts.
3. Assisted mode keeps current telemetry shape.
4. Do not claim continuous green in this card — infra only.

## Acceptance
- [ ] Clean mode fails closed if any assist counter > 0
- [ ] Assisted mode unchanged
- [ ] Report fields distinguish Clean vs assisted
- [ ] Tests green; residual written

## Verify commands
```bash
uv run pytest tmnt_iv/tests/ -q -k "clean or integrity or assist"
```
