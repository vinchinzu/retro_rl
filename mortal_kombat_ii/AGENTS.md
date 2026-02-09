# Agent Instructions — Mortal Kombat II

## Overview
Mortal Kombat II (SNES). More complex than MK1 with more characters and mechanics.

## Status
- Fight state: `Fight_MortalKombatII` (health 161/161, verified)
- Training: Active (500K steps, CUDA)

## Commands

```bash
./run_bot.sh play --state Fight_MortalKombatII
python ../fighters_common/train_ppo.py --game mk2 --state Fight_MortalKombatII --steps 500000
python ../fighters_common/train_ppo.py --game mk2 --state Fight_MortalKombatII --eval --load models/mk2_ppo_final.zip
```

## RAM Addresses

### In data.json (< 0x2000, works with stable-retro)
| Variable | Address | Hex | Type |
|----------|---------|-----|------|
| fatality_timer | 562 | 0x0232 | \|u1 |

### In ram_overrides (>= 0x2000, read via DirectRAMReader)
| Variable | WRAM Addr | get_ram Index | Type | Notes |
|----------|-----------|---------------|------|-------|
| health (P1) | 0x2EFC | 0x4EFD | \|u1 | Verified: starts 161, decreases on hit |
| enemy_health (P2) | 0x30AA | 0x50AB | \|u1 | Verified: 161→135 when landing attacks |

Health addresses are in high WRAM (>= 0x2000). The get_ram() index = WRAM address + 0x2001 offset.
P1 and P2 player structs are 0x1AE (430 bytes) apart.

**Note**: Addresses 0x020A/0x020E are NOT health — they're transitional game state values that happen to show 161 at round boundaries but fluctuate wildly during gameplay.

## MK2-Specific Notes
- Boot sequence is very long (~15 seconds of logos: Sculptured Software, Acclaim, MK2 intro)
- Menu flow: Title → Character Select → Battle Plan → Story screens → Bio → VS → Fight
- Story/bio screens auto-advance but slowly; START mashing speeds through them
- Max health is 161 (0xA1), same as MK1
- Character Liu Kang used for default fight state

## Next Steps
- [ ] Evaluate after training completes
- [ ] Create states for different characters
- [ ] Find additional RAM addresses (rounds won, timer, character ID)
- [ ] Tune reward for MK2's different combat pacing
