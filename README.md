# Retro RL

Reinforcement learning and automation projects for classic SNES games using stable-retro.

## Projects

### [Harvest Moon Bot](./harvest)
Farm clearing automation for Harvest Moon (SNES). The bot uses BFS pathfinding, tool management, and lift+toss mechanics to automatically clear debris (weeds, stones, rocks, stumps) from your farm.

**Features:**
- Automated farm clearing with priority-based targeting
- Human/bot hot-swap for interactive play
- Task recording and replay system
- Comprehensive test suite

[View Harvest Moon Bot →](./harvest)

### [Super Metroid RL](./super_metroid_rl)
Reinforcement learning agents trained to speedrun Super Metroid (SNES) using stable-baselines3 PPO and behavioral cloning.

**Current Status:**
- Phase 1: Ceres Station escape sequence
- Phase 2: Zebes navigation and item acquisition
- Behavioral cloning with heuristic-enhanced navigation
- Custom reward functions for non-standard movement patterns

[View Super Metroid RL →](./super_metroid_rl)

## Technology Stack

- **Emulation**: stable-retro (OpenAI Retro fork with SNES support)
- **RL Framework**: stable-baselines3 (PPO, behavioral cloning)
- **Automation**: Custom pathfinding, state machines, task recording
- **Display**: pygame for visual debugging and human interaction

## Quick Start

Each project has its own dependencies and virtual environment:

```bash
# Harvest Moon Bot
cd harvest
uv sync
./run_bot.sh play --state Y1_Spring_Day01_06h00m

# Super Metroid RL
cd super_metroid_rl
# See project README for setup
```

## License

Projects are for educational and research purposes. ROMs are not included and must be provided by the user.
