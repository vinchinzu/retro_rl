# MK1 Speedrun Plan: Beat the Entire Game

## Goal
Train a LiuKang bot that can reliably beat the entire MK1 SNES tournament from start to finish.

## MK1 SNES Tournament Structure

```
Match 1-6:    Regular 1v1 fights (6 opponents from roster)
Match 7:      Mirror match (fight your own character)
Endurance 1:  2 opponents back-to-back (damage carries over)
Endurance 2:  2 opponents back-to-back (2nd opponent IS Goro!)
Shang Tsung:  Final boss
```

Total: 12 fights across 10 stages.
SNES MK1 has only 2 endurance rounds. Goro is the 2nd opponent of Endurance 2.

## Current Model Inventory (Feb 2026)

| Model | Trained On | Steps | ShangTsung Win% | Goro Win% | M1 Win% | Notes |
|-------|-----------|-------|-----------------|-----------|---------|-------|
| `mk1_fresh_ppo_final.zip` | M1-M7 all chars | 14M | 0% | 0% | ~40% | Broad base model |
| `mk1_shangtsung_ppo_final.zip` | ShangTsung only | 10M | **30%** | 0% | 10% | Boss specialist |
| `mk1_goro_ppo_*.zip` | Goro only | 8M (training) | ? | ? | ? | Boss specialist |

### Per-Stage Win Rates (mk1_shangtsung_ppo_final, 20 attempts each)

| Stage | Win% | Status |
|-------|------|--------|
| M1 | 10% | Weak transfer |
| M2-M5 | 0% | No transfer |
| M6 | 15% | Weak transfer |
| M7 (Mirror) | 0% | No transfer |
| Endurance 1 opp1 | 0% | Not trained |
| Endurance 1 opp2 | 0% | Not trained |
| Endurance 2 opp1 | 0% | Not trained |
| Goro (=E2 opp2) | 0% | Not trained |
| **Shang Tsung** | **30%** | **Specialist** |

### Videos Captured
- `montages/wins/ShangTsung_WIN.mp4` - LiuKang beats Shang Tsung (with credits)
- Post-win states saved: `PostWin_ShangTsung_300/600/1200.state`

## Full Tournament Strategy: Model Swapping

### The Approach
Train specialist models per stage (or stage group), then build a tournament runner
that loads the best model for each stage. This avoids catastrophic forgetting from
trying to make one model handle all 12 stages.

### Stage Groups & Models Needed

```
GROUP 1: Regular Matches (M1-M6)
  Model: mk1_fresh_ppo_final.zip (existing, ~40% on M1)
  Needs: More training focused on M1-M6 to get >60% each
  Training: train_ppo.py cycling through Fight/Match2-6 states
  Target: 8M steps, >50% per stage

GROUP 2: Mirror Match (M7)
  Model: Can share with Group 1 or separate specialist
  State: Match7_LiuKang (LiuKang vs LiuKang)
  Target: >50% win rate

GROUP 3: Endurance 1 (2 opponents)
  Model: New specialist or extension of Group 1
  States: Endurance1_LiuKang, Endurance1B_LiuKang
  Target: >50% per opponent

GROUP 4: Endurance 2 opp1
  Model: New specialist or extension of Group 1
  State: Endurance2_LiuKang
  Target: >50% win rate

GROUP 5: Goro (=Endurance 2 opp2)
  Model: mk1_goro_ppo_final.zip (training overnight, 8M steps)
  State: Goro_LiuKang
  Current: 6.7% at 3.2M steps, climbing
  Target: >30% win rate

GROUP 6: Shang Tsung
  Model: mk1_shangtsung_ppo_final.zip (DONE, 30% win rate)
  State: ShangTsung_LiuKang
  Could push higher with more training
```

### Tournament Runner Design

```python
# speedrun_multimodel.py - Chains the full tournament with per-stage models
STAGE_MODELS = {
    "Fight":       "mk1_regular_ppo_final.zip",     # M1
    "Match2":      "mk1_regular_ppo_final.zip",     # M2
    "Match3":      "mk1_regular_ppo_final.zip",     # M3
    "Match4":      "mk1_regular_ppo_final.zip",     # M4
    "Match5":      "mk1_regular_ppo_final.zip",     # M5
    "Match6":      "mk1_regular_ppo_final.zip",     # M6
    "Match7":      "mk1_regular_ppo_final.zip",     # Mirror
    "Endurance1":  "mk1_regular_ppo_final.zip",     # E1 opp1
    "Endurance1B": "mk1_regular_ppo_final.zip",     # E1 opp2
    "Endurance2":  "mk1_regular_ppo_final.zip",     # E2 opp1
    "Goro":        "mk1_goro_ppo_final.zip",        # Goro
    "ShangTsung":  "mk1_shangtsung_ppo_final.zip",  # Final boss
}
```

Each stage loads from its save state, plays with the designated model,
and reports win/loss. The "full clear probability" is the product of
all stage win rates.

### Probability Math

If we achieve these win rates per stage:
```
M1-M6:     60% each  →  0.6^6 = 4.7% chance of clearing all 6
M7:        60%
E1 opp1:   50%
E1 opp2:   50%
E2 opp1:   50%
Goro:      30%
ShangTsung: 30%

Full clear = 0.6^7 × 0.5^3 × 0.3^2 = 0.028 × 0.125 × 0.09 = 0.03%
```

That's ~1 in 3000. Not great. We need higher per-stage rates:
```
Target: 80% per regular stage, 50% bosses
0.8^7 × 0.7^3 × 0.5^2 = 0.21 × 0.34 × 0.25 = 1.8%
```

~1 in 55 attempts. Achievable with enough training + multiple tournament attempts.

## Execution Plan (Priority Order)

### Step 1: Finish Boss Specialists [IN PROGRESS]
- [x] Shang Tsung specialist: 30% (10M steps) ✓
- [ ] Goro specialist: training overnight (8M steps)
- [ ] Record Goro win video when training completes

### Step 2: Train Regular Match Model [NEXT]
Train a single model that handles M1-M7 + Endurance opponents well.
```bash
# Option A: Extend mk1_fresh_ppo_final.zip with all LiuKang states (M1-E2)
python train_speedrun.py --steps 12000000

# Option B: Train specialist per difficulty tier
python train_ppo.py --state Fight_LiuKang --steps 4000000      # Easy
python train_ppo.py --state Match4_LiuKang --steps 4000000     # Medium
```

### Step 3: Build Multi-Model Tournament Runner
- `speedrun_multimodel.py` that loads different models per stage
- Uses save states (not continuous play) - each stage independent
- Reports per-stage results and overall clear rate
- Run 100 tournament attempts to get statistics

### Step 4: Iterate on Weak Stages
- Run tournament, find bottleneck stages
- Train specialists for weak stages (same approach as Goro/ShangTsung)
- Repeat until full clear rate > 1%

### Step 5: Continuous Play (Stretch Goal)
- Build actual continuous tournament runner (no save state reloading)
- Handle VS screens, round transitions, endurance opponent swaps
- Record a full tournament clear video with credits

## Combo Actions (Experimental - Shelved)

Tested LiuKang's Fireball (F,F,HP) and Flying Kick (F,F,HK) as macro actions.
- ComboFrameSkip wrapper built and verified working (6 damage per fireball)
- Timing: F(4 frames) gap(2) F(4) gap(1) attack(4) = 15 raw frames
- Problem: PPO's random exploration never discovers the combo actions
- Needs: Curriculum learning, intrinsic reward, or imitation learning
- Code: `fighters_common/combo_wrapper.py`, test scripts in mortal_kombat/

## Technical Notes

### State Extraction
- `cheat_extractor.py` uses timer drain method (enemy_health=1, timer=1)
- Extracts all 7 chars through full tournament in 2.5 minutes
- NEVER use env.data.set_value("enemy_health", 0) - doesn't trigger KO logic

### FightingEnv Round Detection
- Round win: `prev_enemy_health > 0 and enemy_health <= 0`
- Round loss: `prev_health > 0 and health <= 0`
- Match win: `rounds_won >= 2` → terminated
- Works correctly for endurance (split-state approach, each half independent)
- Health refill between rounds doesn't cause false round counts

### Known RAM Addresses
```
0x04B9 (1209): P1 health (max 161)
0x04BB (1211): P2/enemy health (max 161)
0x0122 (290):  Timer
0x03E7 (999):  Continue timer
0x1972 (6514): P1 character
```
