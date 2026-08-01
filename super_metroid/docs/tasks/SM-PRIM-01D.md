# TASK SM-PRIM-01D: Migrate `settle_hold` in red_tower.py only

## Recipe step
primitive promote

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/red_tower.py` **only**

## Context
- 01B big_pink_shaft, 01C warehouse done. Next non-hot: red_tower settles.
- Do not change settle frame lengths.

## Read first
- `routes/controller_common.py` (`settle_hold`)
- `routes/kpdr/warehouse.py` (migration style)
- `routes/kpdr/red_tower.py`

## Do
1. Replace obvious settle holds with `settle_hold`.
2. Residual lists remaining files (kraid_approach, hijump_*, …).

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
