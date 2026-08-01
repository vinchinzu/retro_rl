# TASK SM-PRIM-01E: Migrate `settle_hold` in kraid_approach.py only

## Recipe step
primitive promote

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/kraid_approach.py` **only**

## Context
Disjoint from PRIM-01D (red_tower). No frame length changes. No geometry retune.

## Do
1. Migrate settle-style holds to `settle_hold`.
2. Residual next file.

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
