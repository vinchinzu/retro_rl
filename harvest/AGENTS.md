# Harvest Agent Notes

Primary repo-wide instructions live in [../AGENTS.md](../AGENTS.md).

## Commands

All commands require `uv run` (stable-retro is not in base Python).

```bash
# Run bot
./run_bot.sh play --autoplay --state latest

# Boot/morning fixture probe (M1/M2)
uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House

# One overnight toward day+1 (M3 target; ROM required)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --state Y1_Inside_House
# multi-day shell also works:
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Inside_House --days 1

# Tests (narrow — pick the modules you changed)
uv run python -m unittest tests.test_day_plan_sequences tests.test_day_phase_registry tests.test_task_progress -v
uv run python -m unittest tests.test_coop_task -v

# Record a new task (F5 saves)
uv run python -m harvest.runtime.harvest_bot play --state latest --record <name> --no-day-plan

# Editor
./kickoff.sh  # latest/current + autostart
./startup.sh
./startup.sh --state Y1_After_Buy_Potato
./startup.sh --state latest --autostart
PYTHONPATH=.. uv run --project .. python -m retro_harness.editor_launcher harvest -- --state latest
uv run python -m harvest.tools.editor_app --state Y1_After_Buy_Potato --export-dir debug_alignment/editor_exports
```

## File Organization

- Specs: `docs/STATUS.md` (gate facts), `docs/plan.md` (future), `docs/ram_map.md`
- Scripts: `harvest/scripts/boot_probe.py`, `harvest/scripts/run_to_day2.py`
- Save states: `custom_integrations/HarvestMoon-Snes/`
- Recordings: `tasks/<name>.json` + `tasks/<name>_end.state`
- Editor artifacts: `debug_alignment/` or `maps/`
- Tests: `tests/` (unit tests need no ROM; integration tests need ROM + states)
- ROM/state setup: use `harvest.runtime.retro_setup.register_harvest_integration(retro)` before `retro.make(...)`. It registers the custom integration with an absolute path and repairs the ignored `rom.sfc` symlink from known local ROM directories, so do not hand-roll `retro.data.Integrations.add_custom_path(...)` in new scripts. Recording flows should also call `retro_setup.backup_mutable_start_state(...)` so tasks do not point at drifting `latest` / `current` states.

## Day1 / boot→day2 route

Named sequences in `PHASE_SEQUENCES`:

- `day1` — exit farm, clear slice, buy seeds, plant/water, **town_explore**, **ready_to_go_home**, return home, sleep
- `boot_to_day2` — same plus optional recorded tool macros (`get_hammer` / `get_axe` / `get_sickle`)

`GoToSleepTask` always runs `ReturnHomeTask` first when not already in the house.
Town explore / buy-seeds success sets the planner go-home flag and ensures end-day phases exist.

## Adding New Autonomous Tasks

Follow the `CoopChoresTask` / `HarvestTask` pattern:

1. **Discover RAM addresses** — diff save states before/after the action.
2. **Discover walkable tiles** — replay a recording and collect tiles the player stands on. Do NOT trust static save-state tile IDs (SNES re-renders them as viewport scrolls, e.g. `0xA1` → `0x79`).
3. **Register tiles and landmarks** — update `harvest/core/tile_catalog.py` for tile IDs/walkability and `harvest/maps/map_config.py` for map exits, landmarks, and named routes. Do not add new walkable constants directly to task modules.
4. **Build the task** as a dataclass implementing `Task` protocol (reset/can_start/step). Use phase-based state machine with `_navigate_to_tile` + `_queue_press_a` + verify loops.
5. **Add phase spec** in `harvest/planner/day_phase_catalog.py`, register a builder in `day_phase_registry.py` (`PhaseKind` + `PHASE_TASK_BUILDERS`), add it to `build_day_phases()`.
6. **Add multi_nav route** in `harvest/maps/map_config.py` if navigation crosses long distances. Use intermediate waypoints every ~15 tiles to keep BFS within viewport range.
7. **Write unit tests** with fake RAM (no ROM needed). Write integration tests that replay recordings against the real emulator.
8. **For talk/gift tasks** inspect `harvest/core/npc_catalog.py` / `python -m harvest.runtime.harvest_bot npc` first. Dynamic positions come from the WRAM game-object table; dialogue/status data comes from the decomp text table and named flag banks.

## Best Practices Learned

- **Viewport-limited BFS**: The SNES only loads ~16x14 tiles around the player. BFS beyond this sees stale `0x72`/`0xFF`. Use hop targets clamped to 7 tiles.
- **Long-distance nav needs intermediate waypoints**: Multi-map routes over ~15 tiles apart need intermediate `Waypoint` entries or BFS will fail and the bot oscillates.
- **Avoid circular imports**: task modules cannot import from `harvest/planner/day_plan.py` or `day_plan_orchestrator.py`. Put shared RAM fields in `harvest/core/ram_catalog.py` and shared map/tile facts in `harvest/core/tile_catalog.py` / `harvest/maps/map_config.py`.
- **Planner layout**: orchestrators in `day_plan_orchestrator.py`; `PhaseKind` + `day_phase_registry.py` build sub-tasks; dynamic lists in `day_plan_phases.py`; `day_plan.py` is a compatibility barrel. Autoplay stall detection uses `harvest/core/task_progress.py` (`progress_snapshot()` on tasks).
- **Verify with recordings**: Always record a human playthrough first, replay it to discover tile IDs and RAM changes, then build the autonomous version.
- **Backup states**: Before recording, back up `latest.state` as `latest_backup_<description>.state`.
- **NPCs are dynamic objects**: use `WorldSnapshot.entities` or `harvest/core/npc_catalog.py` instead of hard-coding temporary NPC positions in task modules.

## Way Forward

- [ ] Extract reusable facts from `tasks/spring_festival.json`: Spring 23 festival route, NPC/dialogue/status changes, any girl question responses, and the 1304-frame coop trace; preserve start backup `latest_backup_spring_festival_20260427_155856.state`, end state `spring_festival_end.state`, and post-recording backup `latest_backup_post_spring_festival_20260427_160317.state` for replay/debug.
- [ ] Extract the ideal rainy-day routine from `tasks/fix_rainy_day.json`: Y1 Spring 24 route where cows were fed and milked, chicken eggs were shipped, the shed route avoided wasted tiles, crops were harvested, and the town social loop talked to people. Use it to improve rainy-day `build_day_phases()` sequencing, barn/coop/crop/town task ordering, and route efficiency. Preserve start backup `latest_backup_fix_rainy_day_20260427_202555.state`, end state `fix_rainy_day_end.state`, mirrored end state `custom_integrations/HarvestMoon-Snes/fix_rainy_day_end.state`, and the 1193-frame coop trace for replay/debug.
- [ ] Fix `CoopChoresTask` for the Spring 22 two-adult/two-egg coop state: feed adults in separate feed slots, treat visible egg object tiles as dynamic no-go/collision tiles, and add a regression replay before restoring it to the daily plan.
- [ ] Improve barn chores from the `cow_chores_fix` recording: keep the verified multi-cow feed loop, then add brushing/milking and stronger per-cow/per-slot verification before making it routine.
- [ ] Record and test barn chores (cow feeding/milking) — same pattern as coop
- [ ] Add gift delivery task (carry egg to NPC, needs town navigation)
- [ ] Promote candidate NPC sprite IDs to named NPC schedules and dialogue handlers
- [ ] Finish ROM-backed mountain walkable tiles and stump/forage landmarks for berry route autonomy
- [ ] Add cow milking/brushing to daily plan when cows owned
- [ ] Expand `build_day_phases()` for summer/fall crop rotations
