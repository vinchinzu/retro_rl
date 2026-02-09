# SSF2 Multi-Character Training Guide

## Training from Scratch with All Characters

### Phase 1: Generic Fighter (START HERE)

Train a universal fighter using all 16 playable characters.

```bash
./train_multichar.sh
```

**What happens:**
- Trains for 2M steps (8-12 hours on GPU, 24-48 hours on CPU)
- Each episode randomly selects from 16 characters
- Learns fundamental fighting skills that work across all characters
- Saves as: `models/ssf2_multichar_ppo_final.zip`
- Checkpoints: `models/ssf2_multichar_ppo_*_steps.zip` every 25k steps

**What it learns:**
- Spacing and positioning
- Attack timing
- Defense (blocking, movement)
- Universal attack patterns
- Aggressive vs defensive balance

### Phase 2: Multi-Opponent Training (COMING NEXT)

After Phase 1 completes, add opponent variety:

```bash
./train_multi_opponent.sh --load models/ssf2_multichar_ppo_final.zip
```

**What happens:**
- Continues from Phase 1 model
- Trains for 1M more steps
- Mixes character states + opponent progression states
- Learns to handle different matchups

### Phase 3: Fine-Tuning (OPTIONAL)

If certain opponents are too difficult:

```bash
# Focus training on specific opponent
python ../fighters_common/train_ppo.py \
    --game ssf2 \
    --state Fight_vs_Opponent7 \
    --load models/ssf2_multichar_ppo_final.zip \
    --steps 500000
```

## Monitoring Training

### TensorBoard
```bash
cd models
tensorboard --logdir ppo_ssf2_tensorboard/
# Open http://localhost:6006
```

**Key metrics to watch:**
- `rollout/ep_rew_mean`: Should increase over time
- `train/entropy_loss`: Should decrease (exploration → exploitation)
- `custom/win_rate`: Percentage of matches won
- `custom/avg_damage_dealt`: Damage per episode

### Watch the Agent Train
```bash
# In another terminal
./watch.sh
```

This loads the latest checkpoint and shows the agent playing.

## Expected Performance

### Phase 1 (2M steps, ~12 hours)
- **100k steps:** Random flailing, occasional hits
- **500k steps:** Basic attacks, some blocking
- **1M steps:** Consistent pressure, good spacing
- **2M steps:** Wins ~60-80% against first opponent with all characters

### After Phase 2 (+1M steps)
- Handles multiple opponent types
- Adapts to different playstyles
- Wins ~50-70% across all 7 opponents

### After Phase 3 (optional)
- Masters specific difficult matchups
- Near-perfect play against early opponents
- Competitive against later opponents

## Troubleshooting

### Training is very slow
- **Check GPU usage:** `nvidia-smi` (should be ~90%+)
- **Reduce parallel envs:** Edit `TrainConfig.N_ENVS` in train_ppo.py
- **Use fewer steps:** `./train_multichar.sh --steps 1000000`

### Agent not learning
- **Check tensorboard:** Is reward increasing?
- **Validate states:** Do they all work? `./validate_states.sh`
- **Try single character first:** Easier to debug

### Out of memory
- **Reduce batch size:** Edit `TrainConfig.BATCH_SIZE` in train_ppo.py
- **Reduce N_ENVS:** Fewer parallel environments

### Want to resume training
```bash
# Resume from last checkpoint
python train_multi_character.py \
    --load models/ssf2_multichar_ppo_1000000_steps.zip \
    --steps 2000000  # Total steps (will train 1M more)
```

## Model Files

```
models/
├── ssf2_multichar_ppo_final.zip          # Final model after training
├── ssf2_multichar_ppo_25000_steps.zip    # Checkpoint at 25k
├── ssf2_multichar_ppo_50000_steps.zip    # Checkpoint at 50k
├── ...
└── ppo_ssf2_tensorboard/                 # TensorBoard logs
```

**Note:** The old single-character model is:
- `ssf2_ppo_final.zip` (1M steps, default character only)
- You can keep both! They won't conflict.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `./train_multichar.sh` | Start Phase 1 training (fresh) |
| `./watch.sh` | Watch latest model play |
| `./validate_states.sh` | Verify all states work |
| `nvidia-smi` | Check GPU usage |
| `htop` | Check CPU/RAM usage |

## What's Next?

After Phase 1 completes:
1. Watch the trained model: `./watch.sh`
2. Evaluate performance across characters
3. Decide: Continue to Phase 2, or fine-tune specific characters?
4. (Optional) Create more opponent states for harder challenges
