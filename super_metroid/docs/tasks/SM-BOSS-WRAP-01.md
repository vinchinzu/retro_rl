# TASK SM-BOSS-WRAP-01: Wire boss wraps after parallel scaffolds (serialize)

## Recipe step
boss pipeline (protocol registration — run **after** scaffold cards exit)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/protocol.py` (add `wrap_*` only for bosses that have modules)
- `combat/__init__.py` (exports only)
- `tests/test_boss_pipeline.py` (**extend** registration checks)

## Context
- Wave 8 scaffolds create ridley / mother_brain / crocomire / golden_torizo
  modules without touching protocol (parallel safety).
- This card **serializes** wrap registration once modules exist.
- Skip bosses whose modules are missing (residual list).

## Read first
- Existing `wrap_phantoon` / `wrap_botwoon` / `wrap_draygon` patterns
- New combat modules if present
- `tests/test_boss_pipeline.py`

## Do
1. Add wrap helpers mirroring existing CallableBossStrategy pattern.
2. Export from `combat/__init__.py`.
3. Unit tests: each wrap returns strategy with matching catalog room.
4. Do not claim continuous for any deferred boss.

## Do not
- continuous / STATUS
- Geometry controllers

## Acceptance
- [ ] pytest boss pipeline + new wraps green
- [ ] Residual lists still-unwrapped bosses

## Verify
```bash
uv run pytest super_metroid/tests/test_boss_pipeline.py -q
```

## Dispatch note
**Blocked until** SM-RIDLEY-01 / SM-MB-01 / SM-CROC-01 / SM-GT-01 (as available)
have EXIT:0. Do not parallel with those cards.
