# TASK SM-PHAN-01: Phantoon BossStrategy scaffold (dev fight, not continuous)

## Recipe step
boss pipeline (strategy shell + unit tests — continuous deferred)

## Model
Luna

## Own files only
- `combat/phantoon.py` (**create**)
- `combat/protocol.py` (add `wrap_phantoon_as_boss_strategy` only)
- `combat/__init__.py` (exports only)
- `tests/test_phantoon_combat.py` (**create**)

Do **not** edit continuous.py, STATUS, kpdr routes, features.py catalog facts
unless a missing export blocks import (prefer reading existing `phantoon_catalog`).

## Context
- Catalog: `phantoon_catalog()` in `combat/features.py` (room 0xCD13, HP 2500)
- Templates: `combat/kraid.py`, `combat/bomb_torizo.py`, `protocol.wrap_*`
- Dev entry: `custom_integrations/SuperMetroid-Snes/dev_phantoon_entry.state`
- Probe: `scripts/probe/phantoon.py` (dev only — do not claim continuous)
- BOSS_PIPELINE: Phantoon deferred until natural ship access after Alpha PB

## Read first
- `combat/bomb_torizo.py` (strategy + evidence shape)
- `combat/kraid.py` (BossStrategy-style play loop)
- `combat/protocol.py` (CallableBossStrategy wraps)
- `combat/features.py` (`phantoon_catalog`, phase specs if any)
- `combat/primitives.py` (lane/spray helpers)
- `tests/test_kraid_combat.py` (unit style)
- `docs/BOSS_PIPELINE.md` (critical rule — deferred)

## Do (thorough — use many tools)
1. Create `combat/phantoon.py` with:
   - Room constants + strategy dataclass (tunable fire/period/max frames)
   - `fight_phantoon_action(state, frame_index) -> tuple[str, ...]` pure
     (no emu): face toward enemy, spray missiles/supers pattern using
     `features_from_state` + primitives — even if naive
   - `play_phantoon_fight(session) -> PhantoonEvidence` using hold loops
     until HP 0 / boss bit / timeout; evidence dataclass with to_dict
   - Docstring: **developmentOnly** until natural entry on continuous chain
2. Add `wrap_phantoon_as_boss_strategy()` in protocol.py mirroring bomb_torizo
3. Export public symbols from `combat/__init__.py`
4. Unit tests (no emu) covering:
   - catalog facts via strategy
   - action chooses fire/move on active enemy HP>0
   - defeated enemy returns idle/empty actions
   - wrap_phantoon_as_boss_strategy has boss_id phantoon + entry room
5. Optional: run `uv run python super_metroid/scripts/probe/phantoon.py fight --frames 200`
   **only if** it finishes quickly; report result; do not STATUS-promote

## Do not
- Claim continuous Phantoon or write boss bits in pure strategy for green claim
- Edit STATUS.md / continuous.py
- Multi-hour fight tuning sessions — scaffold + unit tests are the bar

## Acceptance
- [ ] `uv run pytest super_metroid/tests/test_phantoon_combat.py super_metroid/tests/test_boss_pipeline.py -q` green
- [ ] Module importable; wrap returns BossStrategy-shaped object
- [ ] Diff + residual (natural entry still missing on KPDR)

## Verify commands
```bash
uv run pytest super_metroid/tests/test_phantoon_combat.py super_metroid/tests/test_boss_pipeline.py -q
```
