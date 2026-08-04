# SSF2 State Creation Guide

## Current States

### Opponent States (Tournament Progression)
- `Fight_SuperStreetFighterII.state` - vs Opponent 1
- `Fight_vs_Opponent2.state` - vs Opponent 2 ✓
- `Fight_vs_Opponent3.state` - vs Opponent 3 ✓
- `Fight_vs_Opponent4.state` - vs Opponent 4 ✓
- `Fight_vs_Opponent5.state` - vs Opponent 5 ✓
- `Fight_vs_Opponent6.state` - vs Opponent 6 ✓
- `Fight_vs_Opponent7.state` - vs Opponent 7 ✓

Note: Opponent order depends on which character you play as!

## All 16 Playable Characters

### Original World Warriors (12)
1. Ryu
2. E.Honda
3. Blanka
4. Guile
5. Ken
6. Chun-Li
7. Zangief
8. Dhalsim
9. Balrog
10. Vega
11. Sagat
12. M.Bison

### New Challengers (4)
13. Cammy
14. Fei Long
15. Dee Jay
16. T.Hawk

## Tools Available

### 1. Automated State Builder (build_opponent_states.py)
Beats opponents automatically and saves states at each level.

```bash
./build_states.sh
# Or manually:
python build_opponent_states.py --opponents 2 3 4 5 6 7
```

**Pros:**
- Fully automated
- Fast
- Consistent

**Cons:**
- Only works if AI can beat opponents
- Fixed character (default starting character)

### 2. AI-Assisted State Creator (ai_assisted_state_creator.py)
AI plays, you save at the right moments.

```bash
./create_states_ai_assisted.sh
```

**Controls:**
- SPACE: Toggle AI on/off
- TAB: Turbo mode (hold for 10x speed) - great for intro screens!
- F2-F7: Save opponent states
- ESC: Quit
- Keyboard/Controller: Manual control when AI is off

**Pros:**
- Easy to use
- AI does the hard work
- You control when to save

**Cons:**
- Requires watching and timing

### 3. Manual State Creator (manual_state_creator.py)
Full manual control.

```bash
# For opponent states (continue from current position):
python manual_state_creator.py

# For character states (start from ROM boot):
./create_character_states.sh
# or: python manual_state_creator.py --from-start

# TIP: Hold TAB for turbo (10x speed) during intro screens!
```

**Pros:**
- Complete control
- Can select any character
- Can save anytime

**Cons:**
- Need to play through manually (hard!)

## Creating Character Starting States

To train with different characters, you need starting states for each character.

### Batch System
Create character states in batches:
- **Batch 1:** Original 12 World Warriors
- **Batch 2:** New 4 Challengers

### Method 1: Manual
```bash
./create_character_states.sh
```

1. Navigate through boot/menus
2. Select the character you want
3. When first fight starts, press the corresponding F-key to save

**Original 12 Character Keys:**
- F1 = Ryu
- F2 = E.Honda
- F3 = Blanka
- F4 = Guile
- F5 = Ken
- F6 = Chun-Li
- F7 = Zangief
- F8 = Dhalsim
- F9 = Balrog
- F10 = Vega
- F11 = Sagat
- F12 = M.Bison

**New Challengers (run separately or use Shift+F-keys):**
- Shift+F1 = Cammy
- Shift+F2 = Fei Long
- Shift+F3 = Dee Jay
- Shift+F4 = T.Hawk

Creates: `Fight_CharacterName.state`

### Method 2: Manual Emulator
You can also create states using a regular SNES emulator:
1. Load SSF2 ROM in RetroArch/SNES9x/etc
2. Play to character select
3. Choose character
4. Save state when first fight starts
5. Copy .state file to `custom_integrations/SuperStreetFighterII-Snes/`
6. Name it `Fight_CharacterName.state`

## Validation

Always validate states after creation:

```bash
./validate_states.sh
```

This shows each state for 3 seconds. Verify:
- Correct opponent appears
- Fight is at the start (FIGHT! just appeared)
- Both fighters have full health
- Timer is full

## Training with Multiple Characters

Once you have character states:

```python
# In your training script:
character_states = [
    "Fight_Ryu",
    "Fight_EHonda",
    "Fight_Blanka",
    "Fight_Guile",
    "Fight_Ken",
    "Fight_ChunLi",
    "Fight_Zangief",
    "Fight_Dhalsim",
    "Fight_Balrog",
    "Fight_Vega",
    "Fight_Sagat",
    "Fight_MBison",
    "Fight_Cammy",
    "Fight_FeiLong",
    "Fight_DeeJay",
    "Fight_THawk",
]

# Randomly select on each reset
state = np.random.choice(character_states)
env = retro.make(game="SuperStreetFighterII-Snes", state=state)
```

This trains a more general fighter that works with multiple characters!

## Future Improvements

### Character Select Waypoint
**TODO for next time:** Create a `CharSelect_SuperStreetFighterII.state` saved right at the character select screen. This would make creating character states much faster:

1. Create waypoint once: Boot game → save at character select
2. Use waypoint: Load CharSelect state → pick character → save when fight starts
3. Benefit: Skip 30-second intro every time!

Implementation:
```bash
# Create the waypoint:
python manual_state_creator.py --from-start
# At character select screen, manually save as CharSelect_SuperStreetFighterII.state

# Then modify manual_state_creator to support:
python manual_state_creator.py --from-char-select
```

## Troubleshooting

### "AI never wins"
- Check if model exists: `ls -lh models/`
- Try using a later checkpoint: `ssf2_ppo_1000000_steps.zip`
- Use manual/AI-assisted creator instead

### "States are all the same"
- This was caused by `env.reset()` bug (now fixed)
- Rebuild states with updated script

### "Game runs too fast in manual mode"
- Fixed in latest version
- When AI is off, it steps the base env without frame skip

### "Controller doesn't work"
- Now supported in ai_assisted_state_creator.py
- Standard gamepad mapping (D-pad + face buttons)

### "Wrong opponent appears"
- Opponent order depends on your character
- States are numbered generically (Opponent2, Opponent3, etc.)
- Validate each state to know which opponent it is
