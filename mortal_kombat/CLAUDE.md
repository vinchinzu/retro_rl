# Mortal Kombat Training Directory

## Top-Level Goal
**Beat the entire MK1 SNES tournament with a LiuKang bot.**

Full plan in `SPEEDRUN_PLAN.md`. Tournament structure:
```
M1-M6 → M7 (mirror) → Endurance 1-3 (2 opponents each) → Goro → Shang Tsung
```

## Current Phase: Full Tournament Training
State extraction COMPLETE (105 validated states via `cheat_extractor.py`).
Now training LiuKang to beat the full tournament via `train_speedrun.py`.

## Best Models
- **General**: `models/mk1_multichar_ppo_2000000_steps.zip` (2M steps, all 7 chars, M1 57%)
- **Deep push**: `models/mk1_match7_ppo_final.zip` (16M steps, M1-M7 trained)
- **Fresh 14M**: `models/mk1_fresh_ppo_final.zip` (used for M6/M7 state extraction)

**CRITICAL:** Always start new training from the broad base models, NOT narrow fine-tunes.
Narrow fine-tuning causes catastrophic forgetting.

## Model Naming
- `mk1_multichar_ppo_*` - Phase 1: all 7 chars on Fight_* states
- `mk1_match{N}_ppo_*` - Phase N: multi-level training
- `mk1_fresh_ppo_*` - Fresh long runs (M1-M7 weighted)
- `mk1_liukang_ppo_*` - LiuKang speedrun: single-char push through tournament
- `mk1_speedrun_ppo_*` - Full tournament training (M1 through Shang Tsung)

**NEVER** use `mk1_ppo_*` naming - that's from old/broken runs.

## State Inventory (CORRECTED - visually validated)
| Level | States | Status |
|-------|--------|--------|
| Match 1 (Fight_*) | 7/7 | Complete |
| Match 2 (Match2_*) | 7/7 | Complete |
| Match 3 (Match3_*) | 7/7 | Complete |
| Match 4 (Match4_*) | 7/7 | Complete |
| Match 5 (Match5_*) | 7/7 | Complete |
| Match 6 (Match6_*) | 7/7 | Complete |
| Match 7 (Match7_*) | 7/7 | Complete - mirror matches confirmed |
| Endurance 1 (Endurance1_*) | 7/7 | Complete (+ 7 Endurance1B) |
| Endurance 2 (Endurance2_*) | 7/7 | Complete (+ 7 Endurance2B = Goro) |
| Goro (Goro_*) | 7/7 | Complete - alias of Endurance2B, 4-armed boss confirmed |
| Shang Tsung (ShangTsung_*) | 7/7 | Complete - shapeshifter boss confirmed |

**Total: 84 unique states** (12 per character × 7 characters) + 7 Goro aliases

## Benchmark History (stochastic, 10 matches each)
| Model | M1 Win% | M2 Win% | M3 Win% | M4 Win% |
|-------|---------|---------|---------|---------|
| Base 2M | 57% | 14% | - | - |
| Match2 V2 (2M) | 59% | 23% | 7% | - |
| Match4 (12M) | 40% | 9% | 20% | 21% |

M6 reality check: Only Scorpion/SubZero can win; other 5 chars go 0/10.

## State Naming Convention
```
Fight_{Char}              # Match 1
Match{N}_{Char}           # Match 2-7
Endurance{N}_{Char}       # Endurance round N (first opponent)
Endurance{N}B_{Char}      # Endurance round N (second opponent)
Goro_{Char}               # Goro fight
ShangTsung_{Char}         # Shang Tsung fight
```

## Known RAM Addresses
```
0x04B9 (1209): P1 health (max 161)
0x04BB (1211): P2/enemy health (max 161)
0x0122 (290):  Timer
0x03E7 (999):  Continue timer
0x1972 (6514): P1 character
???:           Difficulty setting
???:           Match counter / tower position
???:           Current opponent character
```

## Key Scripts

### State Extraction
- `cheat_extractor.py` - RAM-hack state extraction (2min for all 7 chars)
- `manual_state_creator.py` - Manual state creation with TAB turbo, F1-F7 save
- `match_manager.py` - Unified test/extract/validate tool

### Training
- `train_speedrun.py` - Full tournament training (the ONE training script)

### Testing & Utilities
- `speedrun_test.py` - Full tournament test (per-stage win rates, bottleneck analysis)
- `watch.py` - Visual watch agent play (--char, --game, R=reset, TAB=turbo)
- `benchmark_characters.py` - Benchmark all characters (--level N, --model)
- `model_registry.py` - Model tracking with benchmarks and lineage
- `validate_states.py` / `validate_single_state.py` - State validation
- `record_montage.py` / `record_zoom_montage.py` - Video recording

## Script Discipline
- NO one-off training scripts. Use `train_speedrun.py` with `--state` or adjust tier weights.
- NO one-off validation scripts. Use `match_manager.py validate` or `validate_states.py`.
- Debug scripts go in archive/ immediately after use.
- If a new capability is needed, add it as a flag/mode to an existing script.

## Model Registry
```bash
python model_registry.py list                          # Table of all models + scores
python model_registry.py show <model.zip>              # Full details
python model_registry.py benchmark <model.zip> --level N  # Run + record benchmark
python model_registry.py best                          # Best model per level
```

## Characters (7 playable)
LiuKang, Sonya, JohnnyCage, Kano, Raiden, SubZero, Scorpion

Non-playable bosses: Goro, Shang Tsung

## Win Detection
A match is WON only if: `rounds_won >= 2 AND rounds_won > rounds_lost`

### Timeout Round Detection
FightingEnv detects rounds ending via timeout (in-game timer expiry) in addition to KO.
When the game resets both health bars after timeout, the wrapper:
- Assigns round win/loss based on who had more health during combat
- Gives reduced win reward (50%) for timeout wins to prefer KOs
- Applies flat `REWARD_TIMEOUT_ROUND = -0.15` penalty on ALL timeout outcomes
- Tracks `timeout_rounds` in the info dict for logging

## State Validation Checklist
- Health = max (161) for both players
- Timer > 50 (153 = 99 display)
- Screenshot shows "ROUND 1" (not ROUND 2!)
- Use `match_manager.py validate --level N` (saves fresh screenshots)

## MK1 SNES Tournament Structure (CORRECTED)
```
Match 1-6:    Regular 1v1 fights (6 roster opponents)
Match 7:      Mirror match (fight yourself)
Endurance 1:  2 opponents back-to-back (damage carries over)
Endurance 2:  2 opponents back-to-back (2nd opponent = GORO)
Shang Tsung:  Final boss
```
NOTE: SNES has only 2 endurance rounds (not 3). Goro = Endurance2B (alias saved).
12 total fights across 10 stages. No separate Goro stage, no Endurance 3.

## Directory Structure
```
mortal_kombat/
├── SPEEDRUN_PLAN.md        # Full plan for beating the game
├── models/                 # Trained models + registry.json
├── custom_integrations/    # Save states
├── montages/               # Recorded montage videos
├── screenshots/            # Validated state screenshots
├── logs/                   # Training logs (tensorboard)
├── archive/
│   ├── old_scripts/       # Deprecated scripts
│   └── logs/              # Old benchmark/training logs
└── *.py                   # Active scripts
```
