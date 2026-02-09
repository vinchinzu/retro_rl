# Agent Instructions — Mortal Kombat

## Overview
Mortal Kombat (SNES). First MK game, simpler mechanics than MK2/3.

## Status
- Fight state: `Fight_MortalKombat` (health 161/161, verified)
- Training: Active (500K steps, CUDA)

## Commands

```bash
./run_bot.sh play --state Fight_MortalKombat
python ../fighters_common/train_ppo.py --game mk1 --state Fight_MortalKombat --steps 500000
python ../fighters_common/train_ppo.py --game mk1 --state Fight_MortalKombat --eval --load models/mk1_ppo_final.zip
tail -f training.log
```

## RAM Addresses (data.json)

| Variable | Address | Hex | Type |
|----------|---------|-----|------|
| health (P1) | 1209 | 0x04B9 | \|u1 |
| enemy_health (P2) | 1211 | 0x04BB | \|u1 |
| timer | 290 | 0x0122 | \|u1 |
| continue_timer | 999 | 0x03E7 | \|u1 |
| p1_character | 6514 | 0x1972 | \|u1 |

Note: P1 and P2 health are only 2 bytes apart (0x04B9, 0x04BB). All addresses < 0x2000.

## MK-Specific Notes
- MK1 SNES is the censored version (no blood by default, sweat instead)
- Max health is 161 (0xA1), not 176 like SF2
- Blocking works differently from SF2 (hold back + block button)
- Fatalities require specific button sequences at "FINISH HIM" screen

## Next Steps
- [ ] Evaluate after training completes
- [ ] Add fatality detection (look for fatality_timer becoming nonzero)
- [ ] Create states for different characters
- [ ] Tune reward for MK's different combat pacing
