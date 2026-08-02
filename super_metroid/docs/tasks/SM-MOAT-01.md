# TASK SM-MOAT-01: Moat pure controller scaffold (ship approach epic)

## Recipe step
1 pure controller scaffold (geometry green not required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/moat.py` (**create**)
- `tests/test_moat_scaffold.py` (**create**)
- optional residual

No continuous / STATUS / progression promote. No registry race — residual
registration card if pure CLI needs a name later.

## Context
- KPDR K6: Moat `0x95FF` after Alpha PB / elev — **not** continuous yet.
- With Speed + Hi-Jump, Moat is shinespark or platform jumps (ROUTE_KPDR).
- Dev states may exist under `dev_power_bombs_collected` or Crateria anchors —
  optional. Pure green deferred until natural PB+Speed loadout source.
- Hard geometry epic: scaffold + unit structure so parallel work continues.

## Read first
- `docs/routes/ROUTE_KPDR.md` K6
- `docs/archive/routes/ROUTE_SUPERS_TO_PHANTOON.md` if present (historical only)
- `routes/kpdr/red_tower.py` (style)
- `combat/phantoon.py` only for “dev-only” docstring tone

## Do
1. `play_moat_cross` scaffold: require Moat room, bounded platform/spark
   attempt placeholders, timeout residual-friendly.
2. Constants: `ROOM_MOAT = 0x95FF`, exit toward West Ocean if known.
3. Unit tests: import + callable + room constant.
4. Residual: needs capture source with Speed/PB; next pure card ID.

## Acceptance
- [ ] Scaffold + tests green
- [ ] Non-claims: not continuous ship path

## Verify
```bash
uv run pytest super_metroid/tests/test_moat_scaffold.py -q
```
