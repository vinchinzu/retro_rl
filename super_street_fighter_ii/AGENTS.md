# Agent Instructions — Super Street Fighter II

## Overview
Super Street Fighter II - The New Challengers (SNES). Uses the same training pipeline as SF2 Turbo.

## Status
- Fight state: `Fight_SuperStreetFighterII` (health 176/176, verified)
- Training: Done (500K steps, avg_reward=-0.65, win tracking unavailable — data.json lacks rounds_won)

## Commands

```bash
./run_bot.sh play --state Fight_SuperStreetFighterII
python ../fighters_common/train_ppo.py --game ssf2 --state Fight_SuperStreetFighterII --steps 500000
python ../fighters_common/train_ppo.py --game ssf2 --state Fight_SuperStreetFighterII --eval --load models/ssf2_ppo_final.zip
```

## RAM Addresses (data.json)

| Variable | Address | Hex | Type | Notes |
|----------|---------|-----|------|-------|
| health (P1) | 1590 | 0x0636 | \|u1 | Same offset as SF2T |
| enemy_health (P2) | 2166 | 0x0876 | \|u1 | |
| timer | 6441 | 0x1929 | \|d1 | Signed/BCD |
| current_battle | 6332 | 0x18BC | \|u1 | |
| p1_x / p1_y | 1559/1562 | 0x0617/0x061A | <u2 | 16-bit LE positions |
| p2_x / p2_y | 2135/2138 | 0x0857/0x085A | <u2 | |
| p1_hit / p2_hit | 1428/2004 | 0x0594/0x07D4 | \|u1 | |

Has richer data than SF2T including positions, active states, hit detection.

## Next Steps
- [ ] Start training after SF2 shows good results
- [ ] Experiment with position-based rewards (approach enemy bonus)
- [ ] Test with the 4 new characters (Cammy, Fei Long, T. Hawk, Dee Jay)
