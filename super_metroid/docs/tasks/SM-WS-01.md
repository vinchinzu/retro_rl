# TASK SM-WS-01: Wrecked Ship approach scaffold (ship epic, dev-only)

## Recipe step
1 pure controller scaffold (geometry green not required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/wrecked_ship.py` (**create**)
- `tests/test_wrecked_ship_scaffold.py` (**create**)

No continuous / STATUS. Do not claim Phantoon natural entry.

## Context
- K6: Moat → West Ocean → WS → Phantoon. Moat scaffold landed (SM-MOAT-01).
- Dev: `dev_phantoon_entry.state`, `dev_route_phantoon.state` optional smoke only.
- Scaffold multi-stub: ocean approach / attic / basement door placeholders.

## Read first
- `routes/kpdr/moat.py`
- `combat/phantoon.py` (dev-only tone)
- `docs/routes/ROUTE_KPDR.md` K6
- `docs/archive/routes/ROUTE_SUPERS_TO_PHANTOON.md` if present (historical only)

## Do
1. Room constants for Moat/West Ocean/WS main as known.
2. Stub play functions with timeouts and developmentOnly docs.
3. Unit tests importable.
4. Residual: pure after Alpha PB continuous tip (planner gate chain).

## Verify
```bash
uv run pytest super_metroid/tests/test_wrecked_ship_scaffold.py -q
```
