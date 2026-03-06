# Multi-Game Training Infrastructure Setup

Successfully applied the MK1 multi-character training learnings to three additional fighting games:
- **Mortal Kombat II** (12 characters)
- **Street Fighter II Turbo** (12 World Warriors)
- **Super Street Fighter II** (16 characters)

## What Was Set Up

### For Each Game:

#### Python Scripts
- `manual_state_creator.py` - Manual state creation with TAB turbo mode
- `train_multi_character.py` - Multi-character training (Phase 1)
- `validate_states.py` - Validate all save states (symlinked from MK1)
- `validate_single_state.py` - Validate individual states (symlinked from MK1)
- `watch.py` - Watch trained agent play (symlinked from MK1)

#### Shell Scripts (All Executable)
- `train_multichar.sh` - Start fresh multi-character training
- `validate_states.sh` - Validate all states for the game
- `watch.sh` - Watch the trained model play
- `create_character_states.sh` - Create character starting states
- `run_bot.sh` - Run existing bot (already existed)

#### Documentation
- `TRAINING_GUIDE.md` - Complete training guide with 3-phase curriculum
- `STATE_CREATION_GUIDE.md` - State creation tools and procedures

## Game-Specific Details

### Mortal Kombat II
- **Characters:** 12 (Liu Kang, Kung Lao, Johnny Cage, Reptile, Sub-Zero, Shang Tsung, Kitana, Jax, Mileena, Baraka, Scorpion, Raiden)
- **Action Space:** MK_FIGHTING_ACTIONS (same as MK1)
- **RAM:** Uses ram_overrides for health values
- **Model Prefix:** `mk2_multichar_ppo_*`
- **Game Alias:** `mk2`

### Street Fighter II Turbo
- **Characters:** 12 World Warriors (Ryu, E.Honda, Blanka, Guile, Ken, Chun-Li, Zangief, Dhalsim, Balrog, Vega, Sagat, M.Bison)
- **Action Space:** FIGHTING_ACTIONS (default SF2-style)
- **RAM:** Standard health values
- **Model Prefix:** `sf2_multichar_ppo_*`
- **Game Alias:** `sf2`

### Super Street Fighter II
- **Characters:** 16 total (12 original + Cammy, Fei Long, Dee Jay, T.Hawk)
- **Action Space:** FIGHTING_ACTIONS (default SF2-style)
- **RAM:** Standard health values
- **Model Prefix:** `ssf2_multichar_ppo_*`
- **Game Alias:** `ssf2`
- **Note:** Character state creation uses batch system (12 + 4)

## Quick Start for Each Game

### Step 1: Create Character States

You need to create starting states for all characters before training.

**Mortal Kombat II:**
```bash
cd mortal_kombat_ii
./create_character_states.sh
# Create states for all 12 characters (F1-F12)
```

**Street Fighter II Turbo:**
```bash
cd street_fighter_ii
./create_character_states.sh
# Create states for all 12 World Warriors (F1-F12)
```

**Super Street Fighter II:**
```bash
cd super_street_fighter_ii
./create_character_states.sh 1  # Original 12 (F1-F12)
./create_character_states.sh 2  # New Challengers (F1-F4)
```

### Step 2: Validate States

After creating states, validate them:

```bash
./validate_states.sh
```

This will show each state for 3 seconds. Press SPACE to advance, Q to quit.

### Step 3: Start Training

Once you have character states:

```bash
./train_multichar.sh
```

This will:
- Train for 2M steps (~8-12 hours on GPU)
- Randomly select characters each episode
- Save models as `models/{game}_multichar_ppo_*.zip`
- Create checkpoints every 25k steps
- Log to TensorBoard

### Step 4: Monitor Training

**TensorBoard:**
```bash
cd models
tensorboard --logdir ppo_{game}_tensorboard/
# Open http://localhost:6006
```

**Watch Agent Train:**
```bash
./watch.sh  # In another terminal
```

## Training Philosophy (From MK1 Learnings)

### Why Multi-Character Training?

1. **Generic Strategies:** Learns universal fighting skills that work across characters
2. **Robustness:** More resilient to opponent variety
3. **Better Generalization:** Avoids overfitting to single character quirks
4. **Easier Scaling:** Can fine-tune specific characters after generic training

### 3-Phase Curriculum

**Phase 1: Character Rotation** (2M steps)
- This is what `./train_multichar.sh` does
- Randomly select character each episode
- Learn fundamental fighting skills
- Expected: 60-80% win rate vs first opponent

**Phase 2: Multi-Opponent** (1M more steps)
- Mix character states + opponent progression states
- Learn to handle different matchups
- Expected: 50-70% win rate across all opponents

**Phase 3: Fine-Tuning** (500k steps, optional)
- Focus on specific difficult matchups
- Near-perfect play against early opponents
- Competitive against later opponents

## Key Learnings Applied

From the MK1 experience, we incorporated:

### 1. Turbo Mode (TAB)
All manual state creators support holding TAB for 10x speed - crucial for skipping long intro sequences.

### 2. Generic Opponent Naming
Opponent states named `Fight_vs_Opponent2`, `Fight_vs_Opponent3`, etc. instead of character names, since opponent order depends on your character choice.

### 3. Proper State Management
- No env.reset() during state building (let game naturally progress)
- Save states RIGHT when fight starts (FIGHT! appears)
- Validate all states after creation

### 4. Fresh Training
The multi-character training scripts prevent loading existing models by default - better to start fresh than try to extend a single-character model.

### 5. Model Naming
Models saved with `multichar` prefix to avoid conflicts with any existing single-character models.

## Current State Requirements

Before training can begin, each game needs:

### Required States
- **Character starting states:** One for each playable character
  - MK2: 12 states (Fight_LiuKang, Fight_KungLao, etc.)
  - SF2: 12 states (Fight_Ryu, Fight_EHonda, etc.)
  - SSF2: 16 states (all SF2 + Fight_Cammy, Fight_FeiLong, etc.)

### Optional States (for Phase 2)
- **Opponent progression states:** Fight_vs_Opponent2 through Opponent7+
- These can be created later for Phase 2 training

## Next Steps

1. **Choose a game** to start with (MK2, SF2, or SSF2)
2. **Create character states** using `./create_character_states.sh`
3. **Validate states** with `./validate_states.sh`
4. **Start training** with `./train_multichar.sh`
5. **Monitor progress** via TensorBoard and `./watch.sh`

## File Structure

```
mortal_kombat_ii/
├── manual_state_creator.py          # Create states manually
├── train_multi_character.py         # Multi-char training script
├── validate_states.py               # Validate all states (symlink)
├── validate_single_state.py         # Validate one state (symlink)
├── watch.py                         # Watch agent (symlink)
├── train_multichar.sh               # Training launcher
├── validate_states.sh               # Validation wrapper
├── watch.sh                         # Watch wrapper
├── create_character_states.sh       # Character state creator
├── TRAINING_GUIDE.md                # Full training documentation
├── STATE_CREATION_GUIDE.md          # State creation documentation
└── custom_integrations/
    └── MortalKombatII-Snes/
        ├── Fight_MortalKombatII.state      # Base state
        ├── Fight_LiuKang.state             # Character states (to create)
        ├── Fight_KungLao.state
        └── ...
```

Same structure for `street_fighter_ii/` and `super_street_fighter_ii/`.

## Troubleshooting

### "No states found"
Run `./create_character_states.sh` first to create character starting states.

### "Can't play with controller"
The manual state creator supports both keyboard and gamepad. Check if your controller is detected with `ls /dev/input/js*`.

### "Training is slow"
- Check GPU usage: `nvidia-smi` (should be ~90%+)
- Consider fewer parallel envs in TrainConfig
- Reduce steps for initial testing

### "Out of memory"
- Reduce batch size in TrainConfig
- Reduce N_ENVS (fewer parallel environments)

## References

All three games are configured in `fighters_common/game_configs.py` with proper:
- Game IDs
- RAM addresses (including overrides for MK2)
- Action spaces (MK vs SF2-style)
- Menu navigation sequences

See the individual `TRAINING_GUIDE.md` and `STATE_CREATION_GUIDE.md` in each game directory for game-specific details.
