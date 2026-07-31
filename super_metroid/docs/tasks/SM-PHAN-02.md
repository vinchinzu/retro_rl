# TASK SM-PHAN-02: Expand Phantoon unit coverage + dev-only fight smoke docs

## Recipe step
boss pipeline (unit + docs — continuous deferred)

## Model
Luna

## Own files only
- `tests/test_phantoon_combat.py` (**extend**)
- optional: `combat/phantoon.py` (**tunable defaults only if tests need**)
- `docs/tasks/SM-PHAN-02-note.md` (**create**)

Do **not** edit continuous, STATUS, kpdr routes.

## Context
SM-PHAN-01 scaffolded strategy. Expand pure unit tests (phase transitions,
weapon select, timeout labels, evidence dict). Optional: document dev anchor
path if present. No natural ship entry claim.

## Read first
- combat/phantoon.py, protocol wrap
- tests/test_phantoon_combat.py, test_kraid_combat.py style
- docs/BOSS_PIPELINE.md

## Do
1. Add ≥4 meaningful unit tests (no emu or emu-optional).
2. Note: continuous deferred; natural entry still missing.
3. pytest green.

## Acceptance
- [ ] Tests green
- [ ] Note with non-claims

## Verify commands
```bash
uv run pytest super_metroid/tests/test_phantoon_combat.py super_metroid/tests/test_boss_pipeline.py -q
```
