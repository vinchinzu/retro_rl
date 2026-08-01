# TASK SM-DRAY-02: Draygon gunk/phase refine epic (dev-only)

## Recipe step
boss pipeline (strategy refine — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/draygon.py`
- `tests/test_draygon_combat.py` (**extend**)

No Space Jump collect. No continuous / STATUS. No protocol unless wrap already
missing — then residual only (do not race protocol with other boss cards).

## Context
- SM-DRAY-01 + SM-WRAP-DRAY landed shell. Next hardness: gunk-clear timing,
  turret awareness if features exist, richer evidence.
- Dev: `dev_route_anchor_draygon.state` optional.

## Read first
- `combat/draygon.py`, `combat/features.py` (`draygon_catalog`)
- `tests/test_draygon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. Add one phase- or gunk-related helper with unit coverage.
2. Keep max_fight_frames bounded; no infinite loops.
3. ≥3 new tests.
4. Residual: natural Maridia entry still blocks continuous.

## Acceptance
- [ ] Tests green
- [ ] Residual PROCESS schema

## Verify
```bash
uv run pytest super_metroid/tests/test_draygon_combat.py -q
```
