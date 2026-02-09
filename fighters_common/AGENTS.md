# Agent Instructions — fighters_common

Shared library for fighting game RL training. Used by all fighting game directories.

## Module Map

| File | Purpose |
|------|---------|
| `fighting_env.py` | Gym wrappers: `DirectRAMReader`, `FrameSkip`, `GrayscaleResize`, `FightingEnv`, `DiscreteAction`, `FrameStack`, `make_fighting_env()` |
| `game_configs.py` | Per-game configs: RAM addresses, health ranges, menu sequences, `GAME_REGISTRY` |
| `menu_nav.py` | `MenuNavigator`, `navigate_to_fight()`, `create_fight_state()` |
| `train_ppo.py` | PPO trainer: `FighterCNN`, `EntropySchedule`, `FightMetricsCallback`, train/eval/create-state modes |
| `create_all_states.py` | Batch create fight-ready save states for all games |
| `train_all.sh` | Shell script to train all games sequentially |

## Commands

```bash
# Run all tests (20 tests)
PYTHONPATH=.. python -m pytest tests/ -v

# Train a game
python train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --steps 500000
python train_ppo.py --game mk1 --state Fight_MortalKombat --steps 500000 --n-envs 4

# Evaluate with rendering
python train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --eval --load ../street_fighter_ii/models/sf2_ppo_final.zip

# Create fight states
python create_all_states.py                # All games
python create_all_states.py --game sf2     # Single game

# Train all games
./train_all.sh 500000                      # Custom step count
```

## Key Classes

### FightingEnv (fighting_env.py)
Reward shaping wrapper. Tracks health deltas, round wins/losses, match termination.

- `REWARD_DAMAGE_DEALT = 1.0` per health point (normalized by max_health)
- `REWARD_DAMAGE_TAKEN = -0.5` per health point
- `REWARD_ROUND_WIN = 50.0`, `REWARD_ROUND_LOSS = -50.0`
- `REWARD_MATCH_WIN = 200.0` (terminates episode)
- `REWARD_TIME_PENALTY = -0.001` per step

### DirectRAMReader (fighting_env.py)
For games where health is in high WRAM (>= 0x2000). Reads `env.get_ram()` directly and injects values into info dict. Configured via `FightingGameConfig.ram_overrides`.

### FighterCNN (train_ppo.py)
```
Input: (4, 84, 84) → Conv2d(4→32, 8×8, stride=4) → Conv2d(32→64, 4×4, stride=2)
→ Conv2d(64→64, 3×3, stride=1) → Flatten → Linear(→512) → ReLU
Policy head: [256, 128] → 32 actions
Value head: [256, 128] → 1
```

### FIGHTING_ACTIONS (fighting_env.py)
32 discrete actions covering: no-op, 4-directional movement, 6 attack buttons, 8 directional attacks, 4 jump attacks, 2 blocks, 2 throws, 4 multi-button presses, 1 fireball motion.

## Adding a New Game

1. Add a `FightingGameConfig` to `game_configs.py` with game_id, health ranges, RAM addresses
2. Add aliases to `GAME_REGISTRY`
3. If health addresses are >= 0x2000, add them to `ram_overrides` instead of `data.json`
4. Test: `python -m pytest tests/test_env_load.py -v`

## Known Issues

- **stable-retro SNES mapping**: Only first 8KB of WRAM is mapped for `data.json`. Use `DirectRAMReader` for high addresses.
- **Menu navigation**: Button timing is approximate. MK2 required especially long waits (15s boot logos + START mashing through story screens). If states don't reach fights, save frames at each step to debug visually.
- **MK2 health addresses**: Originally documented as 0x2EFC/0x2EFE (high WRAM), but actual addresses are 0x020A/0x020E (low WRAM). Found by scanning for value pairs matching 161 during active fights.
