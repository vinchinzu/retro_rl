# TASK SM-K4-NORF-01: K4 Norfair pure module scaffold (Bubble path epic)

## Recipe step
1 pure controller scaffold (geometry green **not** required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/k4_norfair.py` (**create**)
- `tests/test_k4_norfair_scaffold.py` (**create**)
- optional residual: `docs/tasks/SM-K4-NORF-01-residual.md`

**Do not** edit `continuous.py`, `STATUS.md`, `progression.py` verification
strings (edges stay `unverified`). **Do not** edit `business_climb.py` /
`kraid_return.py` / `varia_return.py`.

Registry / pure CLI wiring may be residual next card (`SM-K4-NORF-REG`) to
avoid parallel races — prefer self-contained module + unit import tests.

## Context
- Continuous tip still Varia-only; reverse pure not yet to Business.
- Graph already has `START_TO_SPEED_GRAPH` edges (all `unverified`):
  business→frog_save→speedway→farm→bubble→… (see `test_k4_speed_branches.py`).
- This card **scaffolds play stubs + room constants + docstrings** so later
  pure cards have a home file. Pure green is **bonus**, not acceptance.
- Dev business anchor: `dev_kpdr_business.state` (topology only; not continuous).

## Read first
- `tests/test_k4_speed_branches.py`
- `progression.py` (`START_TO_SPEED_GRAPH` edge ids only — read)
- `routes/kpdr/warehouse.py` or `green_hill.py` (controller style)
- `routes/kpdr/rooms.py` (room constants pattern)
- `docs/routes/ROUTE_KPDR.md` K4 section

## Do
1. Create `routes/kpdr/k4_norfair.py` with:
   - Room constants used by Bubble path (Business `0xA7DE`, Bubble `0xACB3`,
     Speed `0xAD1B` — use existing rooms.py if present, else local constants)
   - Stub `play_business_to_frog_save`, `play_frog_save_to_speedway`,
     `play_speedway_to_farm`, `play_farm_to_bubble` (or one composed
     `play_business_to_bubble_scaffold`) that:
     - `require_room` on entry
     - bounded holds / TODO geometry comments
     - raise or return with clear label on timeout (no infinite)
   - Module docstring: scaffold only; not continuous; pure green deferred
2. Unit tests: functions importable, segment callables exist, room constants
   match graph edge endpoints (no emu required).
3. Residual lists first real pure card after reverse spine reaches Business:
   `SM-K4-BUBBLE-PURE` + needed source capture.

## Do not
- Claim pure green without named continuous-like source + success
- Promote graph verification
- continuous compose

## Acceptance
- [ ] Module + unit tests green
- [ ] Residual PROCESS schema (pure green optional)
- [ ] Non-claims explicit

## Verify
```bash
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py super_metroid/tests/test_k4_speed_branches.py -q
```
