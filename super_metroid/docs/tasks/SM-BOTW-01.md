# TASK SM-BOTW-01: Botwoon BossStrategy scaffold (dev only, not continuous)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Own files only
- `combat/botwoon.py` (**create**)
- `combat/protocol.py` (**add `wrap_botwoon_as_boss_strategy` only** — do not reformat whole file)
- `combat/__init__.py` (**Botwoon exports only**)
- `tests/test_botwoon_combat.py` (**create**)

Do **not** edit continuous.py, STATUS, kpdr routes, draygon files, phantoon.py.
Do **not** claim Maridia natural entry.

## Context
- Catalog already exists: `botwoon_catalog()` in `combat/features.py`
  (room `0xD95E`, HP 1500, supers primary, continuous_status deferred)
- Templates: `combat/phantoon.py`, `combat/bomb_torizo.py`, `protocol.wrap_*`
- Dev anchor may exist: `custom_integrations/SuperMetroid-Snes/dev_route_anchor_botwoon.state`
  (optional smoke only; do not STATUS-promote)
- BOSS_PIPELINE: deferred until natural entry on continuous chain

## Read first (all — multi-tool)
- `combat/phantoon.py` (scaffold shape to mirror)
- `combat/bomb_torizo.py`
- `combat/protocol.py` (CallableBossStrategy + wrap_phantoon)
- `combat/features.py` (`botwoon_catalog`, features helpers)
- `combat/primitives.py` (range_kite / spray helpers)
- `combat/__init__.py` (export pattern)
- `tests/test_phantoon_combat.py` (unit style)
- `docs/BOSS_PIPELINE.md` (critical deferred rule)

## Do (thorough)
1. Create `combat/botwoon.py` with:
   - `ROOM_BOTWOON = 0xD95E`
   - `BotwoonStrategy` dataclass (tunable fire/period/max frames, weapon supers)
   - `BotwoonEvidence` + `to_dict`
   - `fight_botwoon_action(state, frame_index, strategy=...) -> tuple[str, ...]` pure:
     face toward enemy0, spray supers/missiles via primitives; empty when HP≤0
   - `play_botwoon_fight(session) -> BotwoonEvidence` hold loop until HP0 / boss bit / timeout
   - Module docstring: **developmentOnly** until natural Maridia entry
2. Add `wrap_botwoon_as_boss_strategy()` in `protocol.py` mirroring phantoon
3. Export public symbols from `combat/__init__.py` (strategy, play, wrap, ROOM if others export rooms)
4. Unit tests (no emu):
   - catalog facts via wrap
   - active enemy → fire + face
   - defeated → empty actions
   - wrap boss_id `botwoon` + entry room
5. Optional: if a short probe path exists and finishes quickly, run ≤200f and report;
   otherwise skip and note residual

## Residual required (super-clean)
- Natural entry still missing on KPDR (Maridia after Phantoon path)
- Optional probe result or "not run"
- Files changed + pytest paste
- Do **not** touch STATUS

## Do not
- continuous / STATUS / forge boss bits for green claim
- Multi-hour fight tuning — scaffold + units is the bar
- Edit `draygon.py` / `phantoon.py`

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_botwoon_combat.py super_metroid/tests/test_boss_pipeline.py -q` green
  (if boss_pipeline imports all wraps, keep it green; if too broad, botwoon tests alone + import check)
- [ ] Module importable; wrap returns BossStrategy-shaped object

## Verify commands
```bash
uv run pytest super_metroid/tests/test_botwoon_combat.py -q
uv run pytest super_metroid/tests/test_boss_pipeline.py -q || true
# prefer both green when possible
```
