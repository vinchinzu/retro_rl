# TASK SM-PRIM-02B: Migrate `short_hop` in green_hill.py only

## Recipe step
primitive promote

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/green_hill.py` **only**
- optional residual

## Context
- `short_hop` extracted (SM-PRIM-02).
- green_hill has 24f jump holds that may map to short_hop — migrate **only**
  patterns that match the primitive semantics; do not retune hop length.
- Disjoint from PRIM-01C (warehouse).

## Read first
- `routes/controller_common.py` (`short_hop`)
- `routes/kpdr/green_hill.py`
- SM-PRIM-02 residual if present

## Do
1. One file; call-site migrate only.
2. Residual → next file if more sites remain.

## Do not
- Geometry retune
- continuous / STATUS
- business_climb

## Verify
```bash
uv run pytest super_metroid/tests/test_controller_common.py -q
```
