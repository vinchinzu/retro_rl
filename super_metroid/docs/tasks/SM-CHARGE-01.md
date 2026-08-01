# TASK SM-CHARGE-01: Charge Beam conventional return pure scaffold

## Recipe step
1 pure controller scaffold (optional K1 side — not continuous gate)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/charge_return.py` (**create**)
- `tests/test_charge_return_scaffold.py` (**create**)

## Context
- ROUTE_KPDR: Charge return is **optional** K1 side trip; continuous K1 already
  done via direct Big Pink→Red path.
- Scaffold only; pure green deferred. Dual-track value for completeness.

## Read first
- `routes/kpdr/big_pink.py`, `big_pink_shaft.py` (style)
- ROUTE_KPDR K1 parked note

## Do
1. Stub collect/return hop helpers with room constants for Charge room if known.
2. Unit tests + residual for source capture card.

## Verify
```bash
uv run pytest super_metroid/tests/test_charge_return_scaffold.py -q
```
