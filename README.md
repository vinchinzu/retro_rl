# Retro RL

Reinforcement learning and automation projects for classic SNES games using stable-retro.

## Quick Start

```bash
# Set up shared environment
./setup.sh

# Run a game
cd donkey_kong_country && ./run_bot.sh
cd harvest && ./run_bot.sh play --state Y1_Spring_Day01_06h00m
```

## Projects

### [Donkey Kong Country](./donkey_kong_country)
Interactive player for DKC with keyboard/controller support.

### [Harvest Moon Bot](./harvest)
Farm clearing automation using BFS pathfinding, tool management, and lift+toss mechanics.

**Features:**
- Automated debris clearing (weeds, stones, rocks, stumps)
- Human/bot hot-swap for interactive play
- Task recording and replay system

### [Super Metroid RL](./super_metroid_rl)
RL agents trained to speedrun Super Metroid using PPO and behavioral cloning.

**Current Status:**
- Ceres Station escape sequence
- Zebes navigation and item acquisition
- Custom reward functions for non-standard movement

## Shared Harness

The `retro_harness/` package provides common infrastructure:

- **Controls**: Keyboard/controller input handling with SNES button mapping
- **Env**: Environment setup with custom integrations support
- **Protocol**: Task/Skill/Plan interfaces for composable bot behaviors

## Adding New Games

See [ADDING_GAMES.md](./ADDING_GAMES.md) for step-by-step instructions.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Emulation | stable-retro 0.9.8 |
| RL | stable-baselines3, PyTorch |
| Automation | Custom pathfinding, state machines |
| Display | pygame |
| Env Mgmt | uv |

## Directory Structure

```
retro_rl/
├── setup.sh              # Creates shared .venv
├── pyproject.toml        # Root dependencies
├── retro_harness/        # Shared Python package
│   ├── controls.py       # Input handling
│   ├── env.py            # Environment utilities
│   └── protocol.py       # Task/Skill/Plan interfaces
├── roms/                 # Shared ROMs (git-ignored)
├── donkey_kong_country/  # DKC interactive player
├── harvest/              # Harvest Moon bot
└── super_metroid_rl/     # Super Metroid RL
```

## License

Projects are for educational and research purposes. ROMs are not included and must be provided by the user.

Note: `donkey_kong_country/README.md` documents the standardized replay + split recovery flow.
