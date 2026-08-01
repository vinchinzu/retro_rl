# TASK SM-GT-01: Golden Torizo BossStrategy scaffold (optional epic)

## Recipe step
boss pipeline (optional side — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/golden_torizo.py` (**create**)
- `tests/test_golden_torizo_combat.py` (**create**)

No protocol/__init__. Not on default KPDR; still catalog-complete practice.

## Context
- Catalog: `golden_torizo_catalog()` room `0xB283`, HP 13500, optional path.
- Mirror bomb_torizo / phantoon shell; label continuous_status optional.

## Read first
- `combat/features.py` (`golden_torizo_catalog`)
- `combat/bomb_torizo.py`, `combat/phantoon.py`
- `tests/test_bomb_torizo_strategy.py`

## Do
1. Scaffold strategy + evidence + pure action + play loop.
2. ≥4 unit tests importing from module path.
3. Residual: not KPDR continuous; wrap later.

## Verify
```bash
uv run pytest super_metroid/tests/test_golden_torizo_combat.py -q
```
