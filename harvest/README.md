# Harvest Moon SNES Bot

Farm clearing bot for Harvest Moon (SNES) using stable-retro.

## Quick Start

```bash
# Install dependencies
uv sync

# Human mode (manual play with hot-swap to bot)
./run_bot.sh play --state Y1_Spring_Day01_06h00m

# Autoplay (bot clears farm automatically)
./run_bot.sh play --autoplay --state Y1_Spring_Day01_06h00m

# Priority clearing (focus on specific debris types)
./run_bot.sh play --autoplay --priority "rock,stump" --state Y1_Spring_Day01_06h00m

# Headless (no display)
HEADLESS=1 ./run_bot.sh play --autoplay --state Y1_Spring_Day01_06h00m
```

## Controls

| Key/Button | Action |
|------------|--------|
| L+R+SELECT | Toggle Human/Bot mode (hot-swap) |
| TAB | Fast forward |
| `[` / `]` | Speed down/up |
| F5 | Save state |
| F9 | Load last save |
| P | Mark current tile as no-go |
| ESC | Exit |

## Architecture

```
harvest_bot.py    - Entry point, pygame display, game state
farm_clearer.py   - Clearing logic, pathfinding, tool management
controls.py       - Input handling (keyboard + controller)
tasks/            - Recorded action sequences (JSON)
```

## Clearing Phases

The bot clears debris in priority order (default: weed -> stone -> rock -> stump):

| Debris | Tool | Can Lift? |
|--------|------|-----------|
| Weed/Bush | Sickle | Yes |
| Small Stone | Hammer | Yes |
| Big Rock | Hammer | No |
| Tree Stump | Axe | No |

## Configuration

Environment variables:
- `SKIP_HAMMER=1` - Skip startup task to get hammer from shed
- `NO_GO_TILES="x,y;x,y"` - Mark tiles as impassable
- `HEADLESS=1` - Run without display

## File Structure

```
harvest_bot.py          # Main bot + pygame session
farm_clearer.py         # FarmClearer state machine
controls.py             # Input handling
task_recorder.py        # Record/replay action sequences
tasks/                  # Recorded tasks (get_hammer.json, etc.)
custom_integrations/    # stable-retro game data + save states
tests/run_tests.py      # Test suite (partially broken after refactor)
```
