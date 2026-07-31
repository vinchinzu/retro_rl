# TASK SM-DRAY-01: Draygon BossStrategy scaffold (module + unit tests)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Own files only
- `combat/draygon.py` (**create**)
- `tests/test_draygon_combat.py` (**create**)

**Do not edit** `combat/protocol.py` or `combat/__init__.py` in this card
(parallel ownership: SM-BOTW-01 owns protocol/__init__ wraps). Residual may
list “planner or follow-up card adds wrap_draygon_as_boss_strategy”.

Do **not** edit continuous.py, STATUS, kpdr, botwoon.py, phantoon.py.

## Context
- Catalog: `draygon_catalog()` in `combat/features.py`
  (room `0xDA60`, HP 6000, supers primary, phases body, continuous deferred)
- Mirror structure of `combat/phantoon.py` / (if present) `combat/botwoon.py`
- Dev anchor may exist: `dev_route_anchor_draygon.state` (optional only)
- Space Jump closeout is **out of scope** for this scaffold (fight shell only)

## Read first (all)
- `combat/phantoon.py`
- `combat/features.py` (`draygon_catalog`)
- `combat/primitives.py`
- `tests/test_phantoon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do (thorough)
1. Create `combat/draygon.py` with:
   - `ROOM_DRAYGON = 0xDA60`
   - `DraygonStrategy` (fire period, max frames, weapon supers, optional jump period)
   - `DraygonEvidence` + `to_dict`
   - `fight_draygon_action(state, frame_index, strategy=...) -> tuple[str, ...]` pure
   - `play_draygon_fight(session) -> DraygonEvidence` bounded hold loop
   - Docstring: developmentOnly; no continuous claim; no Space Jump collect in this card
2. Unit tests import **directly** from `super_metroid.combat.draygon` (not package re-export):
   - catalog facts match `draygon_catalog()` room/HP
   - active enemy → includes face + X fire at least sometimes
   - defeated HP0 → empty tuple
   - strategy fire_period respected
3. Optional short dev probe only if trivial; else residual

## Residual required (super-clean)
- Protocol wrap + `__init__` export not done (owned by parallel card / follow-up)
- Natural entry + Space Jump closeout still deferred
- pytest paste + files changed

## Do not
- Edit protocol.py / __init__.py / STATUS / continuous
- Claim continuous Draygon or forge boss bits
- Implement full grab-phase perfect strategy — naive spray is fine

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_draygon_combat.py -q` green
- [ ] Module importable without package export

## Verify commands
```bash
uv run pytest super_metroid/tests/test_draygon_combat.py -q
uv run python -c "from super_metroid.combat.draygon import fight_draygon_action, ROOM_DRAYGON; print(ROOM_DRAYGON)"
```
