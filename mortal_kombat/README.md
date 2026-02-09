# Mortal Kombat - RL Training

Multi-character fighting game RL training for MK1.

## Quick Start

### Watch Trained Agent

```bash
# Watch against different opponents
./watch.sh --state Fight_vs_Opponent2  # 2nd opponent
./watch.sh --state Fight_vs_Opponent3  # 3rd opponent
./watch.sh --state Fight_vs_Opponent4  # 4th opponent
# ... up to Opponent7

# Watch as specific characters
./watch.sh --state Fight_JohnnyCage
./watch.sh --state Fight_Raiden
./watch.sh --state Fight_Scorpion

# Watch with combo tracking
./watch.sh --show-kombos

# Watch with specific model
./watch.sh --model models/mk1_ppo_1375000_steps.zip
```

### Train

```bash
# Start fresh multi-character training (2M steps)
./train_multichar.sh

# Resume from checkpoint
./train_multichar.sh --load models/mk1_ppo_1375000_steps.zip --steps 2000000

# Monitor training in another terminal
./monitor_training.sh
```

### Create Round 2 States (Optional but Recommended)

Round 2 states add difficulty and variety to training:

```bash
# Create Round2 states for all 7 characters
./create_round2_states.sh

# Or just create them
python create_round2_states.py

# Validate Round2 states
python validate_round2_states.py
```

See `QUICKSTART_ROUND2.md` for step-by-step guide.

### Validate States

```bash
# Validate all character states
./validate_states.sh

# Validate single state
python validate_single_state.py --state Fight_vs_Opponent2
```

## Training Phases

1. **Phase 1: Multi-Character** (2M steps) - Random character each episode
2. **Phase 2: Multi-Opponent** (1M steps) - Mix all opponents
3. **Phase 3: Fine-Tuning** (500k steps) - Focus on hard matchups

## File Structure

```
├── train_multi_character.py    # Main training script
├── watch.py                     # Watch agent play
├── validate_states.py           # Validate states
├── manual_state_creator.py     # Create states manually
├── *.sh                         # Shell wrappers
├── custom_integrations/         # Game ROMs and states
│   └── MortalKombat-Snes/
│       ├── Fight_*.state       # Character/opponent states
│       └── Practice_*.state    # Practice mode
├── models/                      # Trained models
└── logs/                        # Training logs
```

## Documentation

- `QUICKSTART_ROUND2.md` - **START HERE** to create Round2 states
- `ROUND2_STATES.md` - Complete Round2 state documentation
- `TRAINING_GUIDE.md` - Full training guide
- `STATE_CREATION_GUIDE.md` - How to create states
- `COMBO_GUIDE.md` - Combo system details
- `CLEANUP_PLAN.md` - File organization details

## Latest Model

Check latest checkpoint:
```bash
ls -lht models/mk1_ppo_*_steps.zip | head -1
```

Current: `mk1_ppo_1375000_steps.zip` (1.375M steps)
