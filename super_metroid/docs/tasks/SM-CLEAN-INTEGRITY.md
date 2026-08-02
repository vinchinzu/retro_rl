# TASK SM-CLEAN-INTEGRITY: Clean zero-resource integrity asserts

## Recipe step
infra

## Model
Luna

## Wave type
implement

## Own files only
- `routes/runtime.py` — extend `assist_integrity` / finish helpers for clean mode
- `scripts/verify/start_to_bombs.py` (and morph if easy) — optional clean checks
- tests under `super_metroid/tests/`
- residual: `docs/tasks/SM-CLEAN-INTEGRITY-residual.md`

Do **not** make assisted runs require zero energy/ammo writes.

## Context
- Assisted integrity: progression/capacity/state loads / deaths.
- Clean integrity **adds**: energy writes/restored == 0; each ammo type
  writes/restored == 0; `unlimited_*_enabled` false in report when clean.
- Spec: `docs/CLEAN_TRACK.md`.

## Read first
- `routes/runtime.py` (`assist_integrity`, report finish)
- `assist.py` (`AssistTelemetry`, `report()`)
- `docs/CLEAN_TRACK.md`

## Do
1. Add `require_clean_resources: bool = False` (or equivalent) to integrity
   helpers; when True, assert zero energy and ammo resource writes.
2. Ensure clean continuous finish path sets the flag when assists disabled.
3. Verify script: if report shows assists off, enforce clean resource zeros
   (or `--require-clean` flag).
4. Unit tests with synthetic telemetry.

## Acceptance
- [ ] Assisted finish path unchanged (still allows resource restores)
- [ ] Clean path fails integrity if any ammo/energy write counted
- [ ] Tests cover both modes

## Verify commands
```bash
uv run pytest super_metroid/tests/ -q -k "assist_integrity or clean"
```
