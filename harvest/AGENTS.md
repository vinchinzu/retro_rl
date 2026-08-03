# Harvest Agent Notes

Primary repo-wide instructions live in [../AGENTS.md](../AGENTS.md).

## Commands

All commands require `uv run` (stable-retro is not in base Python).

```bash
# Run bot
./run_bot.sh play --autoplay --state latest

# Boot/morning probes (M1/M2)
uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House
# Clean power-on → title → new diary → Spring D1 07:00 town (no state load)
HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on \
  --out recordings/power_on_boot_probe.json

# D1 town recon (six talks → truck → shed grass+can → sleep → D2)
uv run python -m harvest.scripts.town_day1_recon checklist
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon capture-entry
./scripts/record_town_day1_recon.sh              # interactive; F5 saves task
./scripts/record_town_day1_recon.sh --power-on   # clean boot + record
# Human rest-of-mask capture (Ann|Eve → full 0x3F → D2): tasks/town_day1_rest.json
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \
  --task town_day1_rest --state Y1_Spring_D1_AnnEve --require-day2
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_AnnEve --out recordings/town_day1_rest_auto.json
# Docs: docs/town_day1_recon.md

# One overnight toward day+1 (M3; ROM required)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --state Y1_Inside_House
# Two overnights: Spring D2 → D4 morning (verified 2026-07-28)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --days 2 --until-day 4 \
  --out recordings/run_to_day4.json
# Full spring calendar → Summer D1 (verified 2026-07-28 calendar; crop income open)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --end-of-spring \
  --out recordings/run_spring_month.json \
  --save-end-state Y1_Summer_D1_Morning
# multi-day shell also works:
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Inside_House --end-of-spring

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

`PowerOnStartTask` now establishes the real clean entry: power-on → new diary
→ Spring D1 07:00 at the town gate. The handoff is ROM-verified, but its
town-gate → farm return has not yet closed an overnight; do not describe the
D2→Summer soak as a D1 replay.

Named sequences in `PHASE_SEQUENCES`:

- `day1` — exit farm, clear slice, buy seeds, plant/water, **town_explore**, **ready_to_go_home**, return home, sleep
- `boot_to_day2` — same plus optional recorded tool macros (`get_hammer` / `get_axe` / `get_sickle`)

`GoToSleepTask` always runs `ReturnHomeTask` first when not already in the house.
Town explore / buy-seeds success sets the planner go-home flag and ensures end-day phases exist.

## Adding New Autonomous Tasks

Prefer **skill composition** over new 50–100 KB phase machines. See
[docs/PLANNING_STACK.md](docs/PLANNING_STACK.md).

1. **Discover RAM addresses** — diff save states before/after the action.
2. **Discover walkable tiles** — replay a recording and collect tiles the player stands on. Do NOT trust static save-state tile IDs (SNES re-renders them as viewport scrolls, e.g. `0xA1` → `0x79`).
3. **Register tiles and landmarks** — update `harvest/core/tile_catalog.py` for tile IDs/walkability and `harvest/maps/map_config.py` for map exits, landmarks, and named routes. Do not add new walkable constants directly to task modules.
4. **Compose skills first** — build from `harvest/tasks/primitives.py` and
   `harvest/tasks/skills.py` (`TaskSequence`, `PressAndVerifyTask`,
   `RamCondition`, `NavSkill`, interact helpers). Only fall back to a custom
   phase machine when a skill does not exist yet; then extract the skill for
   reuse. Implement `progress_snapshot()` with a `child` when wrapping sub-tasks.
5. **Add phase spec** in `harvest/planner/day_phase_catalog.py` (optional
   contract fields: `required_ram`, `required_maps`, `estimated_frames`,
   `failure_modes`). Register a builder in `day_phase_registry.py`
   (`PhaseKind` + `PHASE_TASK_BUILDERS` via `TaskBuildContext`), add it to
   `build_day_phases()`.
6. **Add multi_nav route** in `harvest/maps/map_config.py` if navigation crosses
   long distances. Use intermediate waypoints every ~15 tiles (or
   `densify_waypoints(..., max_hop_tiles=7)`) to keep BFS within viewport range.
7. **Write unit tests** with fake RAM (no ROM needed). Write integration tests that replay recordings against the real emulator.
8. **For talk/gift tasks** inspect `harvest/core/npc_catalog.py` / `python -m harvest.runtime.harvest_bot npc` first. Dynamic positions come from the WRAM game-object table; dialogue/status data comes from the decomp text table and named flag banks.

## Best Practices Learned

- **Viewport-limited BFS**: The SNES only loads ~16x14 tiles around the player. BFS beyond this sees stale `0x72`/`0xFF`. Use hop targets clamped to 7 tiles.
- **Long-distance nav needs intermediate waypoints**: Multi-map routes over ~15 tiles apart need intermediate `Waypoint` entries or BFS will fail and the bot oscillates. Prefer `densify_waypoints` for same-map long hops.
- **Avoid circular imports**: task modules cannot import from `harvest/planner/day_plan.py` or `day_plan_orchestrator.py`. Put shared RAM fields in `harvest/core/ram_catalog.py` and shared map/tile facts in `harvest/core/tile_catalog.py` / `harvest/maps/map_config.py`.
- **Planner layout**: orchestrators in `day_plan_orchestrator.py`; `PhaseKind` + `day_phase_registry.py` build sub-tasks; dynamic lists in `day_plan_phases.py`; `day_plan.py` is a compatibility barrel. Autoplay stall detection uses `harvest/core/task_progress.py` (`progress_snapshot()` on tasks). Optional plan advisor: `local_llm.py` (notes/defer by default; set `HARVEST_PLAN_LLM_APPLY=1` only for schema-validated optional phase rewrites).
- **Observation caching**: prefer `WorldContext` / `WorldSnapshot` over re-reading the same RAM fields every sub-step inside a phase.
- **Verify with recordings**: Always record a human playthrough first, replay it to discover tile IDs and RAM changes, then build the autonomous version.
- **Backup states**: Before recording, back up `latest.state` as `latest_backup_<description>.state`.
- **NPCs are dynamic objects**: use `WorldSnapshot.entities` or `harvest/core/npc_catalog.py` instead of hard-coding temporary NPC positions in task modules.
- **Do not grow mono task files**: coop/cow-sized FSMs are debt; extract skills before adding behavior.

## Way Forward

Architecture / planning spine: [docs/PLANNING_STACK.md](docs/PLANNING_STACK.md),
[docs/plan.md](docs/plan.md). Contracts on crop/coop/sleep phases + skill
boundary factories are in; next is crop income close-loop then coop skill split.

- [ ] Close crop loop: natural can refill → multi-day growth → harvest → ship;
      assert money > $100 after 5pm (see STATUS next acceptance).
- [ ] Power-on → full D1 → D2 with shed grass+can on `house_size=0` (no AnnEve fixture).
- [ ] Extract CoopChores feed/collect/ship into `tasks/skills.py` composers;
      fix Spring 22 multi-adult / dynamic egg tiles + replay before daily plan restore.
- [x] Soft contract preflight in day-plan probe (`preflight_phase_contract` /
      `contract_preflight` JSONL events; soft notes only).
- [ ] Extract reusable facts from `tasks/spring_festival.json`: Spring 23 festival route, NPC/dialogue/status changes, any girl question responses, and the 1304-frame coop trace; preserve start backup `latest_backup_spring_festival_20260427_155856.state`, end state `spring_festival_end.state`, and post-recording backup `latest_backup_post_spring_festival_20260427_160317.state` for replay/debug.
- [ ] Extract the ideal rainy-day routine from `tasks/fix_rainy_day.json`: Y1 Spring 24 route where cows were fed and milked, chicken eggs were shipped, the shed route avoided wasted tiles, crops were harvested, and the town social loop talked to people. Use it to improve rainy-day `build_day_phases()` sequencing, barn/coop/crop/town task ordering, and route efficiency. Preserve start backup `latest_backup_fix_rainy_day_20260427_202555.state`, end state `fix_rainy_day_end.state`, mirrored end state `custom_integrations/HarvestMoon-Snes/fix_rainy_day_end.state`, and the 1193-frame coop trace for replay/debug.
- [ ] Improve barn chores from the `cow_chores_fix` recording: keep the verified multi-cow feed loop, then add brushing/milking and stronger per-cow/per-slot verification before making it routine.
- [ ] Add gift delivery task (carry egg to NPC, needs town navigation)
- [ ] Promote candidate NPC sprite IDs to named NPC schedules and dialogue handlers
- [ ] Expand `build_day_phases()` for summer/fall crop rotations
