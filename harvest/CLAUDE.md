# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Farm clearing bot for Harvest Moon (SNES) using stable-retro emulation. The bot automates debris clearing (weeds, stones, rocks, stumps) using BFS pathfinding, tool management, and lift+toss mechanics.

## Commands

```bash
# Install dependencies
uv sync

# Run bot (interactive play with human/bot hot-swap)
./run_bot.sh play --state Y1_Spring_Day01_06h00m

# Autoplay mode (bot clears farm automatically)
./run_bot.sh play --autoplay --state Y1_Spring_Day01_06h00m

# Priority clearing (focus on specific debris types)
./run_bot.sh play --autoplay --priority "rock,stump" --state Y1_Spring_Day01_06h00m

# Headless mode (no display)
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Spring_Day01_06h00m

# Run tests
uv run python tests/run_tests.py
```

## Architecture

### Module Structure

| Module | Purpose |
|--------|---------|
| `harvest_bot.py` | Entry point, pygame display, game state parsing, `PlaySession` |
| `farm_clearer.py` | All clearing logic: `TileScanner`, `Pathfinder`, `Navigator`, `ToolManager`, `FarmClearer` |
| `controls.py` | Input handling: keyboard mapping, controller support, hot-swap chord |
| `task_recorder.py` | Record/replay action sequences as JSON |
| `tasks/` | Pre-recorded action sequences (get_hammer.json, etc.) |
| `custom_integrations/` | stable-retro game data + save states |

### Key Classes (farm_clearer.py)

- **TileScanner**: Scans RAM for debris tiles (tile IDs → DebrisType mapping)
- **Pathfinder**: BFS pathfinding with obstacle avoidance, temp blocking for stuck recovery
- **Navigator**: Path following, position tracking from RAM, stasis detection
- **ToolManager**: Tracks equipped tool, cycles through inventory to find tools
- **FarmClearer**: State machine with phases: `scanning` → `navigating` → `clearing` → `tool_switch`

### Data Types

```python
from farm_clearer import DebrisType, Tool, Point, Target

DebrisType.WEED    # Clear with sickle or lift
DebrisType.STONE   # Small stones - hammer or lift
DebrisType.ROCK    # Big rocks - hammer only
DebrisType.STUMP   # Tree stumps - axe only

Tool.SICKLE, Tool.HOE, Tool.HAMMER, Tool.AXE, Tool.WATERING_CAN
```

### RAM Addresses

Key addresses are defined in `farm_clearer.py`:
- `ADDR_X`, `ADDR_Y` (0x00D6, 0x00D8) - Player position
- `ADDR_TOOL` (0x0921) - Current equipped tool
- `ADDR_MAP` (0x09B6) - Tile map data
- `ADDR_TILEMAP` (0x0022) - Current map ID
- `ADDR_INPUT_LOCK` (0x019A) - Dialog/menu lock state

## Tests

```bash
uv run python tests/run_tests.py
```

Tests organized by level (L1-L7):
- **L1**: Deterministic task replay (ship_berry, get_hammer)
- **L2**: Navigation and tool acquisition (go_to_barn, get_hoe, buy_potato_seeds, nav pathfinding)
- **L3**: Target detection (TileScanner)
- **L4**: Tooling (use_tool action generation)
- **L6**: Multi-objective clearing
- **L7**: Robustness (dialog dismissal, stuck recovery)

## Issue Tracking (br/beads_rust)

```bash
br ready              # Find available work
br show <id>          # View issue details
br review             # Review issues with AI assistance
br update <id> --status in_progress  # Claim work
br close <id>         # Complete work
br sync --flush-only  # Export beads data (no git)
git add .beads/ && git commit -m "sync beads"
```

## Environment Variables

- `SKIP_HAMMER=1` - Skip startup task to get hammer from shed
- `NO_GO_TILES="x,y;x,y"` - Mark tiles as impassable
- `HEADLESS=1` - Run without display

## Session Completion

Before ending a session, always push to remote:
```bash
git pull --rebase
br sync --flush-only
git add .beads/
git commit -m "sync beads"
git push
```
