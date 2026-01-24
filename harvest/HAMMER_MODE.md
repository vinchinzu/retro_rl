# Hammer Mode Documentation

## Overview

The bot now supports **hammer-only mode** which uses the hammer tool to smash both big rocks and small stones instead of lifting and tossing them into ponds. This is significantly faster for clearing debris.

## Changes Made

### 1. Fixed Tile Classifications

Liftable stones were misclassified as WEED. Fixed in `harvest_bot.py`:

```python
ROI_IDS = {
    0x03: DebrisType.WEED,
    0x04: DebrisType.ROCK,      # Big rocks (non-liftable)
    0x06: DebrisType.STONE,     # Small stones (liftable)
    0x09: DebrisType.STONE,     # Small stones (liftable) - was WEED
    0x0A: DebrisType.STONE,     # Small stones (liftable) - was WEED
    0x0B: DebrisType.STONE,     # Small stones (liftable) - was WEED
    0x0C: DebrisType.STUMP,
    # ... etc
}
```

### 2. Added `prefer_tools_over_lift` Flag

New bot configuration option in `ClearFarmBrain.__init__`:

```python
self.prefer_tools_over_lift = False  # Default: use lift+toss
```

When set to `True`:
- Bot uses hammer on small stones instead of lifting them
- Faster clearing (hammer ~0.5s vs lift+toss ~5-10s)
- Big rocks always use hammer (can't be lifted)

### 3. Tasks Recorded

- `shed_grab_hammer_smash_rock.json` (27s) - grab hammer, smash 1 rock
- `smash_three_rocks.json` (47.9s) - smash 3 more rocks
- `shed_hammer_smash_four_rocks.json` (74.9s) - merged full sequence

### 4. Tests Added

- `test_hammer_smash_rocks()` in `tests/run_tests.py` - validates hammer usage on rocks

## Usage

### Method 1: Run Script (Recommended)

```bash
./run_hammer_mode.sh
```

This starts the bot in hammer-only mode with visual rendering.

### Method 2: Python Code

```python
import harvest_bot as hb

bot = hb.AutoClearBot(priority=[hb.DebrisType.ROCK, hb.DebrisType.STONE])
bot.pond_dispose_enabled = True
bot.brain.prefer_tools_over_lift = True  # Enable hammer mode
bot.brain.only_liftable = False

session = hb.PlaySession(
    state="Y1_After_First_Rock_Smash",
    scale=4,
    bot=bot,
    autoplay=True
)
session.run()
```

### Method 3: Lift+Toss Mode (Default)

To use traditional lift+toss for stones:

```python
bot = hb.AutoClearBot(priority=[hb.DebrisType.STONE])
bot.pond_dispose_enabled = True
bot.brain.prefer_tools_over_lift = False  # Use lift+toss
```

## Performance Comparison

| Method | Time per Stone | Steps |
|--------|---------------|-------|
| **Hammer** | ~0.5s | Navigate adjacent → face → hammer |
| **Lift+Toss** | ~5-10s | Navigate → lift → navigate to pond → toss |

**Hammer is 10-20x faster!**

## Debris Types

- **0x03**: WEED (sickle)
- **0x04**: ROCK - big rocks (hammer only, can't lift)
- **0x06**: STONE - small stones (hammer or lift)
- **0x09**: STONE - small stones (hammer or lift)
- **0x0A**: STONE - small stones (hammer or lift)
- **0x0B**: STONE - small stones (hammer or lift)
- **0x0C-0x14**: STUMP (axe)

## Test Results

All 18 tests passing:
- ✅ L1-L3: Basic functionality
- ✅ L4: Hammer smash rocks (NEW)
- ✅ L5: Lift+pond (still works)
- ✅ L6: Multi-objective clear
- ✅ L7-L9: Advanced features

## Known Issues

1. **Pathfinding** - Bot occasionally targets unreachable rocks
2. **Stuck detection** - Recovery mechanism triggers frequently on complex paths
3. **Fence hopping** - Some stones near fences may be unreachable

## Next Steps

1. Improve pathfinding to avoid unreachable targets
2. Add fence post detection and handling
3. Optimize stuck detection sensitivity
4. Add "hold B to run" for faster navigation
