# Agent Instructions — Mortal Kombat II

## Overview
Mortal Kombat II (SNES). More complex than MK1 with 12 characters and a longer tournament.

## Status
- Character select: `CharSelect_MortalKombatII.state` ✓
- Fight states: All 12 characters have Fight_{Char}.state ✓
- Tournament states: ALL EXTRACTED ✓ (134 total states)

## Quick Commands

```bash
# Play/watch
./run_bot.sh play --state Fight_LiuKang

# Train single character
python ../fighters_common/train_ppo.py --game mk2 --state Fight_LiuKang --steps 500000

# Train all characters
./train_multichar.sh

# Extract tournament states
python cheat_extractor.py --char LiuKang              # Single character
python cheat_extractor.py --all-chars                  # All 12 characters
./extract_all_states.sh                                  # Interactive script
```

## State Inventory

### Character Starting States (Created)
All 12 characters have Fight_{Char}.state at Match 1:
- Fight_LiuKang, Fight_KungLao, Fight_JohnnyCage, Fight_Reptile
- Fight_SubZero, Fight_ShangTsung, Fight_Kitana, Fight_Jax
- Fight_Mileena, Fight_Baraka, Fight_Scorpion, Fight_Raiden

### Tournament Progression States (Complete - 120 states)
All characters have states for each tournament stage:
- Match 2-8: Regular fights (8 matches × 12 chars = 96 states)
- ShangTsung: Sub-boss (12 states)
- Kintaro: Boss 1 - 4-armed (12 states) 
- ShaoKahn: Final boss (12 states)

**Total: 134 states** (1 CharSelect + 13 Fight + 120 Tournament)

### Character Select Waypoint
- `CharSelect_MortalKombatII.state` - Saved at character select screen
- Use to quickly create new character states without boot wait

## State Extraction

### Method 1: Automated Cheat Extractor (Fast - Recommended)
```bash
# Extract for all 12 characters (30-60 min)
./extract_all_states.sh

# Extract for single character
python cheat_extractor.py --char LiuKang

# Start from specific stage (e.g., Match 6)
python cheat_extractor.py --char LiuKang --start-from Match6
```

Uses RAM manipulation (`env.unwrapped.set_ram`) to instantly win matches and saves states at each stage.

### Method 2: Manual State Creator
```bash
./create_character_states.sh
```
Interactive tool: TAB for turbo, F1-F12 to save states for each character.

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

## MK2 Tournament Structure

MK2 has a longer tournament than MK1:
```
Match 1-8 → Shang Tsung (sub-boss) → Kintaro (boss 1) → Shao Kahn (final boss)
```

Opponent order varies by character. The cheat extractor saves states at each stage.

## MK2-Specific Notes
- Boot sequence is very long (~15 seconds of logos: Sculptured Software, Acclaim, MK2 intro)
- Menu flow: Title → Character Select → Battle Plan → Story screens → Bio → VS → Fight
- Story/bio screens auto-advance but slowly; START mashing speeds through them
- Max health is 161 (0xA1), same as MK1
- 12 playable characters (vs 7 in MK1)
- Kintaro replaces Goro as the 4-armed boss
- Shang Tsung is a sub-boss who morphs into other characters

## Key Scripts

| Script | Purpose |
|--------|---------|
| `cheat_extractor.py` | RAM-hack tournament state extraction |
| `extract_all_states.sh` | Interactive wrapper for all characters |
| `manual_state_creator.py` | Manual state creation with turbo |
| `train_multi_character.py` | Train on all 12 characters |
| `validate_states.sh` | Verify all states work |
| `watch.sh` | Watch agent play |

## Next Steps
- [x] Run cheat_extractor.py for all characters (134 states extracted)
- [x] Validate extracted tournament states (all verified)
- [ ] Train on multi-character tournament states
- [ ] Find additional RAM addresses (rounds won, timer, character ID)
