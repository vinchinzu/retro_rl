# Agent Instructions — retro_rl

Unified framework for training RL agents on classic SNES games using stable-retro emulation and PPO.

## Project Layout

```
retro_rl/
├── retro_harness/          # Shared library: controls, env setup, protocols, recording
├── fighters_common/        # Fighting game shared code: wrappers, PPO trainer, configs
│   ├── fighting_env.py     # Gym wrappers (FrameSkip, GrayscaleResize, FightingEnv, etc.)
│   ├── train_ppo.py        # Unified PPO trainer for all fighting games
│   ├── game_configs.py     # Per-game configs (RAM addrs, health ranges, menu sequences)
│   ├── menu_nav.py         # Automated menu navigation + save state creation
│   └── tests/              # 20 tests covering env load, wrappers, menu nav, PPO smoke
├── street_fighter_ii/      # SF2 Turbo — Hyper Fighting (SNES)
├── super_street_fighter_ii/ # Super SF2 — The New Challengers (SNES)
├── mortal_kombat/          # Mortal Kombat (SNES)
├── mortal_kombat_ii/       # Mortal Kombat II (SNES)
├── donkey_kong_country/    # DKC interactive player + autosplit
├── harvest/                # Harvest Moon farm automation bot
├── super_metroid_rl/       # Super Metroid RL training (PPO + BC)
├── roms/                   # Shared ROM directory (git-ignored)
├── setup.sh                # Create venv + install deps
└── pyproject.toml          # Root deps: numpy, pygame, stable-retro, [ml] extras
```

## Quick Start

```bash
./setup.sh                              # Create .venv, install deps
source .venv/bin/activate
uv pip install stable-baselines3 torch gymnasium opencv-python tensorboard  # ML deps

# Run tests
PYTHONPATH=. python -m pytest fighters_common/tests/ -v

# Train a fighting game agent
python fighters_common/train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --steps 500000

# Evaluate trained model
python fighters_common/train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --eval --load street_fighter_ii/models/sf2_ppo_final.zip

# Play interactively
./street_fighter_ii/run_bot.sh play --state Fight_StreetFighterIITurbo
```

## Fighting Game System

### Supported Games

| Alias | Game | Fight State | Health | Status |
|-------|------|-------------|--------|--------|
| `sf2` | SF2 Turbo | `Fight_StreetFighterIITurbo` | 176/176 | Trained (500K, 100% WR) |
| `ssf2` | Super SF2 | `Fight_SuperStreetFighterII` | 176/176 | Training (500K) |
| `mk1` | Mortal Kombat | `Fight_MortalKombat` | 161/161 | Trained (500K, 100% WR) |
| `mk2` | Mortal Kombat II | `Fight_MortalKombatII` | 161/161 | Training (500K) |

### Architecture

```
RetroEnv → [DirectRAMReader if needed] → FrameSkip(4) → GrayscaleResize(84×84)
  → FightingEnv (health-delta rewards) → DiscreteAction (32 moves) → FrameStack(4) → Monitor
```

- **FightingEnv**: Reward = damage_dealt - 0.5×damage_taken + round_win/loss bonus - time_penalty
- **DiscreteAction**: 32 fighting moves (movement, attacks, directional attacks, blocks, throws, combos)
- **FighterCNN**: Conv2d(4→32→64→64) → Linear(512) → pi[256,128] / vf[256,128]
- **DirectRAMReader**: For games with health in high WRAM (>0x2000) beyond stable-retro's mapping

### SNES RAM Mapping Limitation

stable-retro only maps the first 8KB (0x0000–0x1FFF) of SNES WRAM in `data.json`. Games with RAM addresses ≥ 0x2000 need `DirectRAMReader` which uses `env.get_ram()` directly. Configure via `ram_overrides` in `game_configs.py`. All 4 current games have health in low WRAM.

## Adding a New Game

1. Create `new_game/custom_integrations/GameName-Snes/` with `rom.sfc` symlink, `rom.sha`, `data.json`, `metadata.json`, `scenario.json`
2. Add config to `fighters_common/game_configs.py`
3. Create `run_bot.sh` + `run_bot.py` (copy from existing game)
4. Create fight state: `python fighters_common/create_all_states.py --game alias` or play interactively + F5
5. Train: `python fighters_common/train_ppo.py --game alias --state StateName --steps 500000`

## Next Steps / Roadmap

### Phase 1: Get Agents Winning (Current)
- [x] **SF2 Turbo**: Trained 500K steps, 100% win rate (41W/0L)
- [x] **Mortal Kombat**: Trained 500K steps, 100% win rate (136W/0L)
- [x] **Fix MK2**: Found correct health addresses (0x020A/0x020E), created fight state
- [ ] **SSF2**: Training in progress (500K steps)
- [ ] **MK2**: Training in progress (500K steps)
- [ ] **Evaluate all**: Visual eval with `--eval` flag, compare win rates
- [ ] **Hyperparameter sweep**: Try different learning rates (1e-4, 3e-4, 1e-3), entropy coefficients, reward scales

### Phase 2: Improve Agent Quality
- [ ] **Self-play**: Train P1 vs trained P2 (2-player env) for better generalization
- [ ] **Curriculum learning**: Start vs easy CPU, gradually increase difficulty
- [ ] **Reward engineering**: Add combo detection, special move bonuses, blocking rewards
- [ ] **Frame skip tuning**: Try 2 or 3 instead of 4 for faster reaction time
- [ ] **Action space refinement**: Add game-specific special move inputs (QCF, DPM, charge)

### Phase 3: Scale to More Games
- [ ] **Mortal Kombat 3** (ROM already in roms/)
- [ ] **Tekken** / **Killer Instinct** / other fighting games
- [ ] **Cross-game transfer**: Pre-train on one fighter, fine-tune on another
- [ ] **Unified evaluation**: Standardized win-rate benchmarks across all games

### Phase 4: Advanced Techniques
- [ ] **Imitation learning**: Record human gameplay demos, pre-train with behavioral cloning
- [ ] **LSTM/Transformer policy**: Add memory for combo tracking and opponent prediction
- [ ] **Opponent modeling**: Detect and exploit CPU patterns
- [ ] **Multi-agent tournament**: Round-robin between agents from different games

## Environment

- Python 3.12+, managed by `uv`
- GPU: CUDA (torch auto-detects)
- TensorBoard logs: `<game>/logs/`
- Model checkpoints: `<game>/models/` (every 25K steps)
- Headless mode: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` or `HEADLESS=1`
