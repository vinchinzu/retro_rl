# TASK SM-BOSS-UNIT-MATRIX: Boss catalog × strategy unit matrix (Flash+Luna)

## Recipe step
boss pipeline (tests only — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `tests/test_boss_catalog_matrix.py` (**create**)

No combat module geometry rewrites. No continuous / STATUS.

## Context
- Wave 8 landed ridley/mb/croc/gt/escape + refined phan/botw/dray.
- Need one matrix test file: every catalog entry has room_id; every strategy
  module that exists imports cleanly; wrap presence is optional assertion.

## Read first
- `combat/features.py` BOSS_CATALOG
- `tests/test_boss_pipeline.py`
- New combat modules

## Do
1. Parametrized tests over catalog ids.
2. Soft-check wrap_* if present after SM-BOSS-WRAP-01.
3. Residual lists bosses still without strategy modules.

## Verify
```bash
uv run pytest super_metroid/tests/test_boss_catalog_matrix.py super_metroid/tests/test_boss_pipeline.py -q
```
