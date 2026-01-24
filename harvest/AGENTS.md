# Agent Instructions

This project uses **br** (beads_rust) for issue tracking. Run `br onboard` to get started.

## Architecture

The bot has two main modules:

- **harvest_bot.py** - Game integration, pygame display, state loading, input handling
- **farm_clearer.py** - All clearing logic: scanning, pathfinding, navigation, tool management

### Key Classes

| Module | Class | Purpose |
|--------|-------|---------|
| `harvest_bot` | `GameState` | Parse game state from RAM (date, time, money, stamina) |
| `harvest_bot` | `AutoClearBot` | Main bot wrapper, delegates to FarmClearer |
| `harvest_bot` | `PlaySession` | Interactive pygame session with human/bot modes |
| `farm_clearer` | `TileScanner` | Scan RAM for debris tiles |
| `farm_clearer` | `Pathfinder` | BFS pathfinding with obstacle avoidance |
| `farm_clearer` | `Navigator` | Path following and position tracking |
| `farm_clearer` | `ToolManager` | Track current tool, cycle through inventory |
| `farm_clearer` | `FarmClearer` | State machine for phase-based clearing |

### Data Types

```python
from farm_clearer import DebrisType, Tool, Point, Target

DebrisType.WEED    # Bushes, clear with sickle or lift
DebrisType.STONE   # Small stones, hammer or lift
DebrisType.ROCK    # Big rocks, hammer only
DebrisType.STUMP   # Tree stumps, axe only

Tool.SICKLE, Tool.HOE, Tool.HAMMER, Tool.AXE, Tool.WATERING_CAN
```

### Breaking Changes (Refactor 2025-01)

**Removed entirely:**
- `MapTransitionManager` - building navigation
- `DayPlanner` - daily task planning
- `PlotWorkerBot` - planting/watering automation
- `find_3x3_plot()`, `is_tilled_tile()`, `is_watered_tile()`, etc.
- `ITEM_ID_POTATO_SEEDS`

**Moved to farm_clearer module:**
- `Navigator`, `ToolManager`, `use_tool()`, `ADDR_INPUT_LOCK`
- `TargetFinder` renamed to `TileScanner`

**Changed API:**
- `AutoClearBot.brain` → `AutoClearBot.clearer`
- `AutoClearBot.startup_steps` → `AutoClearBot.clearer.startup_tasks`
- `AutoClearBot.nav` → `AutoClearBot.clearer.navigator`

**Tests:** Many tests in `tests/run_tests.py` are broken and need updating to import from `farm_clearer`.

## Quick Reference

```bash
br ready              # Find available work
br show <id>          # View issue details
br update <id> --status in_progress  # Claim work
br close <id>         # Complete work
br sync --flush-only  # Export beads data (no git)
git add .beads/
git commit -m "sync beads"
```

## Environment + run_bot

Use `uv` to provision the venv that `run_bot.sh` expects at `.venv/`.

```bash
uv sync
uv run python tests/run_tests.py

# Run the bot (script sources .venv/bin/activate internally)
uv run ./run_bot.sh

# Headless mode
HEADLESS=1 uv run ./run_bot.sh
```

## Landing the Plane (Session Completion)


**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
   ```bash
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND 
7. **Hand off** - Provide context for next session



<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

**Note:** `br` is non-invasive and never executes git commands. After `br sync --flush-only`, you must manually run `git add .beads/ && git commit`.

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
br ready              # Show issues ready to work (no blockers)
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once
br sync --flush-only  # Export beads data (no git)
git add .beads/
git commit -m "sync beads"
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run:
   ```bash
   br sync --flush-only
   git add .beads/
   git commit -m "sync beads"
   ```

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads data (no git)
git add .beads/
git commit -m "sync beads"
git commit -m "..."     # Commit code
br sync --flush-only    # Export beads data (no git)
git add .beads/
git commit -m "sync beads"
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `br sync --flush-only` (plus git add/commit) before ending session

<!-- end-bv-agent-instructions -->
