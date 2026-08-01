# TASK SM-ALPHA-PB-01: Alpha PB room pure scaffold (K5 prep epic)

## Recipe step
1 pure controller scaffold (geometry green not required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/alpha_pb.py` (**create**)
- `tests/test_alpha_pb_scaffold.py` (**create**)

No continuous / STATUS / progression promote. No registry race required.

## Context
- KPDR K5: Alpha Power Bomb Room `0xA3AE` preferred first PB (not Pink PB).
- Continuous still far; scaffold module + constants + stub play for later pure.
- Queue practice problem exists without teleport fixture for item room.

## Read first
- `docs/routes/ROUTE_KPDR.md` K5
- `routes/kpdr/moat.py` or `k4_norfair.py` (scaffold style)
- `routes/kpdr/pb_door.py` (PB interaction patterns — hint only)

## Do
1. `ROOM_ALPHA_PB = 0xA3AE`, stub `play_alpha_pb_collect` with require_room +
   bounded collect attempt + timeout.
2. Unit tests: importable, room constant, callable.
3. Residual: needs source with Ice/Speed loadout after K4; next pure card id.

## Acceptance
- [ ] Scaffold + tests green
- [ ] Non-claims: not continuous PB capacity evidence

## Verify
```bash
uv run pytest super_metroid/tests/test_alpha_pb_scaffold.py -q
```
