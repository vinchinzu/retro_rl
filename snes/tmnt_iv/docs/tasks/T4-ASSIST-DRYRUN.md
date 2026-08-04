# TASK T4-ASSIST-DRYRUN: Re-record assisted continuous baseline

## Recipe step
continuous + status proposal

## Model
Luna run → **planner** apply STATUS / BASELINE

## Wave type
stabilize

## Own files only
- residual: `docs/tasks/T4-ASSIST-DRYRUN-residual.md`
- STATUS / BASELINE only if card explicitly hands planner-apply (else propose)

## Context
- After one or more `T4-ASSIST-*` knobs land.
- Defaults remain assisted. Do **not** use `--clean`.
- Report: prefer overwrite dry-run path only when planner approves new baseline
  (`tmnt_iv_full_hard_dry_run.json`).

## Do
1. Dry-run full hard continuous with current production policy.
2. Compare vs BASELINE_METRICS (time, damage, e-heals, iframe, lives).
3. If improved or neutral (0 lives lost held): propose BASELINE + STATUS
   metric updates.
4. If regressed: residual → revert knob or next fix card; do not promote.

## Acceptance
- [ ] Dry-run report path recorded
- [ ] 0 lives lost held
- [ ] Promotion only via planner
- [ ] Residual PROCESS fields

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --dry-run
```
