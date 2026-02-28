# Super Metroid RL - Ceres Escape & Beyond

Train reinforcement learning agents to speedrun Super Metroid (SNES) using `stable-retro` and `stable-baselines3`.

## Project Goals

### Phase 1: Escape from Ceres Station
The opening sequence where Samus must escape the exploding Ceres Station after the Ridley fight. Key challenges:
- **Movement direction**: Goes DOWN and LEFT (opposite typical RL reward functions)
- **Time pressure**: Station explodes after countdown
- **Room transitions**: Multiple rooms with different layouts
- **Simple mechanics**: No items to collect, just movement and doors

### Phase 2: Arrival on Zebes
Landing on planet Zebes and navigating to acquire first items (Morphing Ball, Missiles, Bombs).

### Phase 3: Room-by-room Fine-tuning
Train specialized models per room/area, then combine for full runs.

---

## Approach

### Model Strategy (Layered)
1. **Naive baseline**: Hardcoded heuristics for initial testing
2. **Imitation learning**: Pre-train from recorded human demonstrations
3. **PPO training**: Full RL training with shaped rewards
4. **Room-specific models**: Fine-tune per room, use router to select

### Reward Function Design
Since Ceres goes DOWN-LEFT instead of RIGHT:
- **Progress reward**: Track room transitions and position within rooms
- **Room-specific rewards**: Different reward functions per room ID
- **Time bonus**: Reward faster completion
- **Stagnation penalty**: Penalize standing still

### Data Sources
- **ROM memory**: Position (X/Y), room ID, health, door transitions, game state
- **Visual input**: Screenshots/video frames (for later CNN-based approaches)
- **Human demonstrations**: Record manual runs with keypresses for imitation learning

### Training Modes
- **Headless training**: No visualization for speed
- **Visual debugging**: Optional pygame rendering
- **Recording (dataset)**: Save action indices + per-frame raw button arrays
- **Recording (movie)**: Optional `.bk2` replay capture for emulator playback/video
- **Playback videos**: Generate MP4 from `.bk2` recordings

---

## Directory Structure

```
super_metroid_rl/
├── README.md                     # This file
├── super_metroid_naive.py        # Hardcoded naive agent
├── play_human.py                 # Manual play + recording (in parent dir)
├── roms/
│   └── rom.sfc                   # Super Metroid ROM (ignored; copy/symlink into custom_integrations)
├── custom_integrations/
│   └── SuperMetroid-Snes/
│       ├── Start.state           # Save state at Ceres start
│       ├── data.json             # RAM address definitions
│       ├── metadata.json         # Default state config
│       └── scenario.json         # Reward/done conditions
├── run_bot.py                    # Navigation bot (Model + Heuristics)
├── metroid_rewards.py             # Custom reward wrapper
├── convert_bk2.py                 # BK2 to MP4 conversion script
├── train_bc_nav.py                # BC training script
├── models/                       # Trained policies (e.g., bc_nav_model.pth)
├── logs/                         # Training logs
└── recordings/                   # .bk2 replay files
```

---

## Key RAM Addresses (Super Metroid)

These need to be added to `data.json` for proper tracking:

| Variable | Address | Type | Description |
|----------|---------|------|-------------|
| room_id | $079B | u2 | Current room ID |
| samus_x | $0AF6 | u2 | Samus X position |
| samus_y | $0AFA | u2 | Samus Y position |
| health | $09C2 | u2 | Current energy |
| max_health | $09C4 | u2 | Max energy |
| game_state | $0998 | u1 | Game state machine |
| door_transition | $0797 | u1 | Door transition flag |
| timer_minutes | $0945 | u1 | Game timer minutes |
| timer_seconds | $0947 | u1 | Game timer seconds |
| timer_frames | $0949 | u1 | Game timer frames |

*Note: SNES uses LoROM mapping. Addresses are RAM offsets.*

---

## Ceres Station Room Layout

```
[Landing Site] ─> [Hallway 1] ─> [Ridley Room] ─> [Hallway 2]
                                      │
                                      v (after fight)
[Escape Start] <─ [Room 3] <─ [Room 2] <─ [Room 1]
      │
      v
[Ship Escape]
```

The escape goes: RIGHT at first, then DOWN, then LEFT to the ship.

---

## Scripts

### Record a Full Run (Recommended for Dataset Collection)
Run from repo root:

```bash
source .venv/bin/activate
uv run python -m super_metroid_rl play --state ZebesStart --scale 2
```

This is the best workflow for collecting training data from a 1-hour human run.

- Expected length for a 1-hour run: about `216000` frames at 60 FPS.
- Stop and save: `ESC`
- Restart and clear current recording: `R`
- Save/load in-memory checkpoints: `F1`-`F4` / `Shift+F1`-`Shift+F4`
- Export current emulator state to disk: `F5`

Keyboard controls:
- Arrow keys: D-Pad
- `Z`: B, `X`: A, `A`: Y, `S`: X
- `Q`: L, `W`: R
- `Enter`: Start, `Shift`: Select

Output files are written to:

`super_metroid_rl/optimizer/runs/sm_landing_site/`

- `recording_XXX.json`: action-space sequence (discrete action indices)
- `recording_XXX_raw.json`: per-frame button data + metadata

`recording_XXX_raw.json` now contains:
- `raw_buttons`: per-frame env-applied buttons (post-sanitize; replay-safe)
- `raw_buttons_pre_sanitize`: per-frame physical input snapshot before sanitize
- `actions`: per-frame discrete action indices

### Verify a Recording Quickly
```bash
uv run python -m super_metroid_rl verify \
  --actions super_metroid_rl/optimizer/runs/sm_landing_site/recording_XXX.json
```

Check frame alignment between raw and action outputs:

```bash
python - <<'PY'
from pathlib import Path
import json
runs = Path("super_metroid_rl/optimizer/runs/sm_landing_site")
raw_path = sorted(runs.glob("recording_*_raw.json"))[-1]
data = json.loads(raw_path.read_text())
print("file:", raw_path)
print("raw_buttons:", len(data.get("raw_buttons", [])))
print("raw_buttons_pre_sanitize:", len(data.get("raw_buttons_pre_sanitize", [])))
print("actions:", len(data.get("actions", [])))
PY
```

### Optional: Also Record `.bk2` Movies
If you want emulator movie files in addition to JSON datasets:

```bash
uv run python super_metroid_rl/record_tasker.py record --state ZebesStart
```

Convert `.bk2` to MP4:

```bash
uv run python super_metroid_rl/scripts/bk2_to_mp4.py <input.bk2> <output.mp4>
```

### PPO Training
```bash
uv run python super_metroid_rl/train_curriculum.py list-segments
uv run python super_metroid_rl/train_curriculum.py train --segment sm_landing_site --steps 50000 --device cuda
uv run python super_metroid_rl/train_curriculum.py run --render --start-state ZebesStart --device cuda
```

---

## Navigation Bot Features

The `run_bot.py` script combines a trained model with several heuristic boosters:

### 1. Unified Recovery Cycle
When Samus is stuck for > 1 second, the bot enters a 5-phase recovery loop:
- **Phase 0 (Pulse Shoot)**: Clears destructible blocks.
- **Phase 1 (Walk Left)**: Attempts horizontal movement.
- **Phase 2 (Walk Right)**: Attempts horizontal movement.
- **Phase 3 (Jump & Move)**: Attempts to jump over obstacles.
- **Phase 4 (Wiggle)**: Random movement to break out of complex stalls.

### 2. Elevator Assistance
Precise logic for handled complex transitions:
- **Crateria (0x94CC)**: Automatically centers Samus at `X=426` and holds `DOWN` to enter Brinstar.
- **Y-Guards**: Ensures elevator logic only triggers near actual platforms.

### 3. Focused Movement
- **Pulsed Firing**: Firing frequency is limited to preserve movement animations.
- **Clean Recovery**: Fire buttons are cleared during movement-heavy recovery phases.

### 4. Custom Rewards (`metroid_rewards.py`)
- **Depth Reward**: Incentivizes vertical progress ($+Y$).
- **Item Bonus**: Huge reward ($+1000$) for acquiring the **Morphing Ball**.
- **Stagnation Penalty**: Discourages standing still.

---

## Implementation Roadmap

### Stage 1: Foundation
- [ ] Fix data.json with proper RAM addresses
- [ ] Update scenario.json with position-based rewards
- [ ] Test RAM reading with naive agent
- [ ] Record 5-10 human demonstration runs

### Stage 2: Imitation Learning
- [ ] Create demonstration parser (bk2 → training data)
- [ ] Implement behavioral cloning baseline
- [ ] Pre-train model on human demonstrations

### Stage 3: PPO Training
- [ ] Create CeresReward wrapper with room-aware rewards
- [ ] Create Discretizer for useful action combinations
- [ ] Train PPO with shaped rewards
- [ ] Implement room transition detection

### Stage 4: Room-Specific Models
- [ ] Create save states for each room
- [ ] Train room-specific policies
- [ ] Implement room-based model router
- [ ] Combine for full escape run

### Stage 5: Zebes
- [ ] Create save states for Zebes sections
- [ ] Design exploration-based rewards
- [ ] Extend to item acquisition

---

## Technical Notes

### SNES Button Mapping
```
Index: [0:B, 1:Y, 2:Select, 3:Start, 4:Up, 5:Down, 6:Left, 7:Right, 8:A, 9:X, 10:L, 11:R]
```

### Useful Action Combinations for Super Metroid
- Run: Hold B + direction
- Jump: A
- Shoot: Y (or X for missile select)
- Aim up: L or R + direction
- Spin jump: A while moving
- Wall jump: A toward wall while spinning

### Recording Format
- `recording_XXX.json`: Discrete action indices (current action space)
- `recording_XXX_raw.json`:
  - `raw_buttons`: 12-button vectors per frame (post-sanitize, replay-safe)
  - `raw_buttons_pre_sanitize`: 12-button vectors per frame before sanitize
  - `actions`: action index per frame (same length as raw arrays)
- Optional `.bk2`: BizHawk movie format (initial state + per-frame buttons)

---

## References

- [Super Metroid RAM Map](https://wiki.supermetroid.run/RAM_Map)
- [stable-retro documentation](https://stable-retro.farama.org/)
- [stable-baselines3 PPO](https://stable-baselines3.readthedocs.io/)
- Mario implementation: `../beat_level_1_1.py`
