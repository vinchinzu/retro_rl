# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Instrumentated day planner + pinned morning/sleep fixtures; continuous boot→day-2 not yet ROM-verified |
| Last verification | 2026-07-28 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **M2 instrumented** — RAM catalog, scene classifier, day planner, multi-day shell |
| Integration | `HarvestMoon-Snes` |
| ROM | `roms/Harvest Moon.sfc` (local) / monorepo `roms/` candidates via `retro_setup` |
| Start contract | Named morning state (`Y1_Inside_House` / `Y1_Spring_Day01_06h*`) or multi-day from latest |
| Completion contract | Campaign (multi-year farm / marriage / ending) — TBD in ASSIST_CONTRACT when assists appear |
| Evidence | Unit suite under `tests/`; long-run logs under `logs/long_runs/`; pinned states in `custom_integrations/HarvestMoon-Snes/` |

## Done

- **M1-ish integration**: custom integration path, ROM symlink repair, editor + bot launchers
- **M2 instrumentation**: `harvest/core/ram_catalog.py`, scene classifier, world snapshot, NPC catalog
- Day planner with dynamic phases, failure policies, multi-day return-home/sleep loop
- Coop/cow/crop/harvest tasks with unit coverage
- **2026-07-28 route work**:
  - `GoToSleepTask` always finds the house via `ReturnHomeTask` before bed nav
  - Nav/multi-nav dismiss dialogue **and** menus (scene classifier)
  - `town_explore` route + `READY_TO_GO_HOME` flag phases
  - `day1` / `boot_to_day2` sequences chain macros → town → return → sleep
  - Specs layout: `docs/STATUS.md`, `docs/plan.md`, `docs/ram_map.md`, `scripts/`

## Next acceptance (M3)

Continuous single-day clear into next morning:

```bash
# Unit (no ROM)
uv run python -m unittest tests.test_day_plan_sequences tests.test_day_phase_registry -v

# ROM-backed: morning → sleep → day advanced (headless multi-day days=1)
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Inside_House --day-plan boot_to_day2 --days 1

# Or multi-day auto plan for one overnight
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Inside_House --days 1
```

Success predicate: calendar `day` advances, morning scene is house/farm at 06:00, no state load mid-run.

## Traps

- Viewport BFS: only ~16×14 tiles load; long routes need intermediate waypoints
- Sleep must stand at bed pixel (base `(70,86)`, L2 wife bed `(294,102)`), face **up**, plain A
- Doors reject carried items — return-home clears hands first
- Mutable `latest.state` is for manual diagnosis only; pin fixtures before regressions
- Multi-day planner sets `include_end_day=False` on the day task and owns return/sleep itself

## Key states

| State | Role |
|-------|------|
| `Y1_Inside_House` | Morning house, day≈2 06:00 |
| `Y1_Front_House` | Outdoor house front |
| `Y1_After_Sleep` | Post-sleep morning house |
| `Y1_Spring_Day01_06h` | Named early spring fixture (verify hour/day in RAM) |
| `day1` / `day1_end` | Human day1 recording endpoints |
