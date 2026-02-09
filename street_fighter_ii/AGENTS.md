# Agent Instructions — Street Fighter II Turbo

## Overview
SF2 Turbo - Hyper Fighting (SNES). PPO agent training to win fights against CPU opponents.

## Status
- Fight state: `Fight_StreetFighterIITurbo` (health 176/176, verified)
- Training: Active (500K steps, CUDA)
- Models checkpoint to `models/` every 25K steps
- TensorBoard logs in `logs/`

## Commands

```bash
# Interactive play
./run_bot.sh play --state Fight_StreetFighterIITurbo

# Train
python ../fighters_common/train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --steps 500000

# Evaluate
python ../fighters_common/train_ppo.py --game sf2 --state Fight_StreetFighterIITurbo --eval --load models/sf2_ppo_final.zip

# Monitor training
tail -f training.log
tensorboard --logdir logs/
```

## RAM Addresses (data.json)

| Variable | Address | Hex | Type |
|----------|---------|-----|------|
| health (P1) | 1590 | 0x0636 | \|u1 |
| enemy_health (P2) | 1840 | 0x0730 | \|u1 |
| timer | 6387 | 0x18F3 | \|u1 |
| rounds_won (P1) | 6300 | 0x189C | \|u1 |
| enemy_rounds_won (P2) | 6299 | 0x189B | \|u1 |
| matches_won | 3280 | 0x0CD0 | \|u1 |
| match_state | 6392 | 0x18F8 | \|u1 |

All addresses are < 0x2000, so they work directly in data.json.

## Next Steps
- [ ] Evaluate trained model win rate after 500K steps
- [ ] If win rate < 50%, increase to 1M steps or tune hyperparams
- [ ] Add character-specific states (Ken, Chun-Li, etc.)
- [ ] Add special move inputs to action space (QCF+P for hadouken, DP+P for shoryuken)
- [ ] Self-play training (2-player mode)
