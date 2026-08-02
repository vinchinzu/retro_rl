# TASK T4-CLEAN-STAB: Dual re-verify Clean continuous

## Recipe step
stabilize

## Model
Luna → planner

## Wave type
stabilize

## Own files only
- residual: `docs/tasks/T4-CLEAN-STAB-residual.md`
- no policy edits

## Context
- After first `T4-CLEAN-FULL` green.
- Dual integrity: two independent Clean dry-runs (or dry-run + video encode)
  with matching success + zero assists.
- Super Metroid analog: `SM-CLEAN-STAB`.

## Do
1. Re-run Clean continuous dry-run with distinct report path.
2. Confirm metrics: credits, 0 e-heals, 0 iframe, 0 lives lost.
3. Optionally compare stage damage tables for fluke spikes.
4. Residual → `T4-CLEAN-STATUS` if dual green.

## Acceptance
- [ ] Two Clean continuous greens (or explicit single + planner exception)
- [ ] Assisted baselines untouched
- [ ] No new policy knobs

## Verify commands
```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run \
  --report tmnt_iv/recordings/tmnt_iv_full_hard_clean_stab1.json
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m tmnt_iv.scripts.record_full_hard_run --clean --dry-run \
  --report tmnt_iv/recordings/tmnt_iv_full_hard_clean_stab2.json
```
