# Harvest Moon SNES Bot

Autonomous daily-life bot for Harvest Moon (SNES) using stable-retro.
Handles farm clearing, crop planting/watering/harvesting, chicken coop
chores, berry collection, and shop trips — all driven by a dynamic day
planner that inspects the save state and assembles tasks by priority.

## Quick Start

```bash
# Install dependencies (requires uv — all commands use `uv run`)
uv sync

# Autoplay from latest save (dynamic day plan)
./run_bot.sh play --autoplay --state latest

# Human mode (manual play, L+R+SELECT hot-swaps to bot)
./run_bot.sh play --state latest

# Force a specific day plan instead of auto-detection
./run_bot.sh play --autoplay --state latest --day-plan harvest

# Record a new task (F5 to save)
uv run python -m harvest.runtime.harvest_bot play --state latest --record coop_chores --no-day-plan

# Headless (no display)
HEADLESS=1 ./run_bot.sh play --autoplay --state latest
```

## Map Editor

```bash
# Open the exact-pixel map editor on a snapshot
uv run python -m harvest.tools.editor_app --state Y1_After_Buy_Potato

# Export the current map canvas and exit
uv run python -m harvest.tools.editor_app --state Y1_After_Buy_Potato --export-dir debug_alignment/editor_exports
```

The editor now renders exact emulator-observed pixels for the current snapshot/session and exposes useful overlay layers:

- `Doors / transitions`: known cross-map exits from `harvest/maps/map_config.py` plus door-like threshold tiles.
- `Collision / blocked tiles`: non-walkable tiles using the map-specific walkable sets.
- `Sprite clamp bounds`: ROM scene object clamp rectangle when present.
- `Sprite delta (live only)`: highlights pixels in the live emulator frame that differ from the current base render.
- `Live viewport overlay`: draws the current emulator viewport on top of the map canvas.
- `Player marker`: current player position.

Bot/world exports now include dynamic WRAM game-object positions through
`harvest/core/npc_catalog.py`, but the editor sprite layer is still a live-frame delta
overlay. Unseen map regions stay explicitly unknown instead of being
synthesized.

See [docs/editor_layers.md](docs/editor_layers.md) for the current overlay model and commands.

## Reference Map Tools

```bash
# Export the town reference image as PNG
uv run python -m harvest.maps.extract_tiles --export-reference-png town --export-reference-output debug_alignment/reference_exports/town.png

# Compare a snapshot render against the town reference
uv run python -m harvest.maps.extract_tiles --compare-reference town --compare-state TMP_Town_From_GoToShop --compare-dir debug_alignment/town_reference_compare
```

## Controls

| Key/Button | Action |
|------------|--------|
| L+R+SELECT | Toggle Human/Bot mode (hot-swap) |
| TAB | Fast forward |
| `[` / `]` | Speed down/up |
| F5 | Save recording when `--record`, otherwise save state |
| F9 | Load last save |
| P | Mark current tile as no-go |
| ESC | Exit |

## Day Plan System

`build_day_phases()` in `harvest/planner/day_plan_phases.py` inspects the save state and assembles
the day's task list dynamically.  Priority order:

1. **Exit building** (always)
2. **Chicken coop** — feed adults, collect egg, incubate or ship (`harvest/tasks/coop_task.py`)
3. **Harvest** ripe crops → ship to bin (`harvest/tasks/harvest_task.py`)
4. **Water** crops — ensure watering can from shed, BFS to field (`harvest/tasks/crop_planter.py`)
5. **Berry run** — walk to mountain, pick berries, ship (if before 15:00)

Named sequences (`--day-plan day1`, `sunday`, `harvest`, etc.) still work as
manual overrides.

## Program specs

- Status / maturity gate: [docs/STATUS.md](docs/STATUS.md)
- Plan (future work): [docs/plan.md](docs/plan.md)
- Planning stack (skills, contracts, advisor): [docs/PLANNING_STACK.md](docs/PLANNING_STACK.md)
- RAM map: [docs/ram_map.md](docs/ram_map.md)
- Morning fixture probe: `uv run python -m harvest.scripts.boot_probe`
- Clean power-on bootstrap (title → new diary → Spring D1):
  `HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on`
- Overnight target: `HEADLESS=1 uv run python -m harvest.scripts.run_to_day2`

Day sequences: `--day-plan day1` or `--day-plan boot_to_day2` (macros + town explore
go-home flag + return home + sleep that always finds the house).

## Architecture

```
harvest/runtime/harvest_bot.py   - Entry point, pygame display, hot-swap, recording
harvest/planner/day_plan_orchestrator.py - DayPlanTask, MultiDayPlannerTask
harvest/planner/day_phase_types.py   - PhaseKind enum, PhaseSpec
harvest/planner/day_phase_registry.py - PhaseKind → task builders
harvest/planner/day_plan_phases.py - Dynamic build_day_phases()
harvest/planner/day_plan.py      - Compatibility re-exports (tests/tools)
harvest/planner/tasks/home.py    - ReturnHomeTask, GoToSleepTask (find house first)
harvest/core/task_progress.py    - ProgressSnapshot (autoplay stall watchdog)
harvest/tasks/coop_task.py       - CoopChoresTask (feed/egg/incubate/ship, scales to 12)
harvest/tasks/harvest_task.py    - HarvestTask (pick ripe crops + ship)
harvest/tasks/crop_planter.py    - CropWaterTask (plant seeds + water)
harvest/tasks/farm_clearer.py    - FarmClearer + pathfinding (BFS, Navigator, Pathfinder)
harvest/maps/map_config.py       - Walkable tiles, map registry, named routes
harvest/core/harvest_state.py    - HarvestStateDocument (persistent tile layer)
harvest/runtime/rom_tools.py     - Save state parsing, VRAM/ROM inspection
harvest/scripts/    - boot_probe, run_to_day2
tasks/              - Recorded action sequences (JSON + end states)
custom_integrations/  # stable-retro game data + save states
tests/              - Unit + integration tests (see Testing below)
```

## Testing

```bash
# All harvest tests (fast, no ROM needed for unit tests)
uv run python -m unittest discover -s tests -v

# Specific test modules
uv run python -m unittest tests.test_day_plan_sequences -v
uv run python -m unittest tests.test_coop_task -v

# ROM-backed checks are covered by targeted test modules when local ROM/states exist.
```

## RAM Discovery

When adding new features, diff save states to find RAM addresses:

```python
from harvest.runtime.rom_tools import parse_save_state, STATES_DIR
s = parse_save_state(STATES_DIR / "latest.state")
ram = s.ram
# Compare before/after to find changed addresses
```

Key RAM regions:
- `0x0900–0x09FF` — inventory, tools, items
- `0x11F00–0x12000` — farm stats (money, hay, chicken count, time, etc.)
- `0xC200–0xC400` — livestock slots (chickens at 0xC286, cows at 0xC1C6)

## Configuration

Environment variables:
- `HEADLESS=1` — run without display
- `SKIP_HAMMER=1` — skip startup task to get hammer from shed
- `NO_GO_TILES="x,y;x,y"` — mark tiles as impassable

## Controls

| Key/Button | Action |
|------------|--------|
| L+R+SELECT | Toggle Human/Bot mode (hot-swap) |
| TAB | Fast forward |
| `[` / `]` | Speed down/up |
| F5 | Save recording when `--record`, otherwise save state |
| F9 | Load last save |
| P | Mark current tile as no-go |
| ESC | Exit |
