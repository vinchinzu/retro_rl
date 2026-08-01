# TASK SM-PRIM-01C: Migrate `settle_hold` in warehouse.py only

## Recipe step
primitive promote

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/warehouse.py` **only**
- optional residual note listing remaining settle files

## Context
- SM-PRIM-01B migrated `big_pink_shaft.py`.
- Next non-hot file: warehouse settle holds → `settle_hold`.
- Do **not** change settle frame lengths.

## Read first
- `routes/controller_common.py` (`settle_hold`)
- `routes/kpdr/big_pink_shaft.py` (migration style)
- `routes/kpdr/warehouse.py`

## Do
1. Replace obvious settle `hold`/`_hold` with `settle_hold` (import as needed).
2. Keep reasons.
3. No pure probe required (not climb file) unless cheap.

## Do not
- business_climb / hijump_return / varia_return this card
- continuous / STATUS

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
