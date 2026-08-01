# TASK SM-RIDLEY-01: Ridley BossStrategy scaffold (dev-only epic)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Wave type
implement

## Own files only
- `combat/ridley.py` (**create**)
- `tests/test_ridley_combat.py` (**create**)

**Do not edit** `combat/protocol.py` or `combat/__init__.py` (wrap is a
follow-up card). Do **not** touch continuous, STATUS, kpdr routes.

## Context
- Catalog already: `ridley_catalog()` room `0xB32E`, HP 18000, supers primary,
  continuous_status deferred, major boss bit Norfair.
- Dev anchors (optional smoke only, never continuous evidence):
  `dev_route_anchor_ridley.state`, `dev_route_ridley_entry.state`
- Mirror `combat/phantoon.py` / `combat/draygon.py` shape.
- Natural LN entry does **not** exist on continuous chain — developmentOnly.

## Read first
- `combat/phantoon.py`, `combat/draygon.py`
- `combat/features.py` (`ridley_catalog`)
- `combat/primitives.py`
- `tests/test_phantoon_combat.py`, `tests/test_draygon_combat.py`
- `docs/BOSS_PIPELINE.md`

## Do
1. Create `combat/ridley.py`:
   - `ROOM_RIDLEY = 0xB32E`
   - `RidleyStrategy` (fire period, jump period, max frames, weapon supers/missiles)
   - `RidleyEvidence` + `to_dict` (body HP track, boss bit, outcome)
   - `fight_ridley_action(state, frame_index, strategy=...) -> tuple[str, ...]` pure
   - `play_ridley_fight(session) -> RidleyEvidence` bounded hold loop
   - Docstring: **developmentOnly**; no continuous / natural-entry claim
2. Unit tests import **directly** from `super_metroid.combat.ridley`:
   - catalog room/HP match
   - active enemy → fire/face sometimes
   - HP0 / defeated → empty actions
   - evidence dict keys stable
3. Optional: note path to dev anchor in residual; do **not** require emu green.

## Do not
- Write boss/event RAM
- continuous.py / STATUS
- protocol wrap (next: SM-BOSS-WRAP-01)

## Acceptance
- [ ] Module + ≥4 unit tests green
- [ ] Residual lists wrap + natural-entry blockers
- [ ] Explicit non-claims (not continuous)

## Verify
```bash
uv run pytest super_metroid/tests/test_ridley_combat.py -q
```

## Done when
Scaffold importable + tests green, or residual with missing catalog mismatch.
