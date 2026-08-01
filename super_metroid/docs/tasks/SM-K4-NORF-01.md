# TASK SM-K4-NORF-01: K4 Norfair pure module scaffold (Bubble path epic)

## Recipe step
1 pure controller + continuous tip extension (**complete**)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/k4_norfair.py` (**create**)
- `tests/test_k4_norfair_scaffold.py` (**create**)
- optional residual: `docs/tasks/SM-K4-NORF-01-residual.md`

The completed implementation promoted only after two integrity-green power-on
Frog Save runs. `business_climb.py`, `kraid_return.py`, and `varia_return.py`
remain untouched.

Registry / pure CLI wiring may be residual next card (`SM-K4-NORF-REG`) to
avoid parallel races — prefer self-contained module + unit import tests.

## Context
- `business-to-frog-save` was pure green from `post_business_continuous`
  (**1,190f**) then composed twice from power-on to `0xB167` (**114,923f**).
- `START_TO_SPEED_GRAPH` marks Business→Frog continuous; the first open K4
  edge is now Frog Save→Speedway.
- Source-backed next anchor: `scratch/post_frog_continuous.state`.

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
- [x] Module + unit tests green
- [x] Pure green from the accepted Business source
- [x] Two integrity-green continuous runs; graph/tracker promoted

## Verify
```bash
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py super_metroid/tests/test_k4_speed_branches.py -q
```

## Result

**GREEN.** The controller waits for the incoming Business elevator, snakes to
the floor band, selects beam, and shoots through the closed Frog door. The
next bounded card is `SM-K4-SPEEDWAY-PURE`: Frog Save→Speedway from
`scratch/post_frog_continuous.state`.
