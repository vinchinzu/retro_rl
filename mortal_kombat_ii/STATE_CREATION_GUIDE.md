# MK2 State Creation Guide

## Current States

### Opponent States (Tournament Progression)
- `Fight_MortalKombatII.state` - vs Opponent 1
- `Fight_vs_Opponent2.state` - vs Opponent 2 ✓
- `Fight_vs_Opponent3.state` - vs Opponent 3 ✓
- `Fight_vs_Opponent4.state` - vs Opponent 4 ✓
- `Fight_vs_Opponent5.state` - vs Opponent 5 ✓
- `Fight_vs_Opponent6.state` - vs Opponent 6 ✓
- `Fight_vs_Opponent7.state` - vs Opponent 7 ✓

Note: Opponent order depends on which character you play as!

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

To train with different characters, you need starting states for each of the 12 characters.

### Method 1: Manual
```bash
./create_character_states.sh
```

1. Navigate through boot/menus
2. Select the character you want
3. When first fight starts, press F1-F12:
   - F1 = Liu Kang
   - F2 = Kung Lao
   - F3 = Johnny Cage
   - F4 = Reptile
   - F5 = Sub-Zero
   - F6 = Shang Tsung
   - F7 = Kitana
   - F8 = Jax
   - F9 = Mileena
   - F10 = Baraka
   - F11 = Scorpion
   - F12 = Raiden

Creates: `Fight_CharacterName.state`

### Method 2: Manual Emulator
You can also create states using a regular SNES emulator:
1. Load MK2 ROM in RetroArch/SNES9x/etc
2. Play to character select
3. Choose character
4. Save state when first fight starts
5. Copy .state file to `custom_integrations/MortalKombatII-Snes/`
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
    "Fight_LiuKang",
    "Fight_KungLao",
    "Fight_JohnnyCage",
    "Fight_Reptile",
    "Fight_SubZero",
    "Fight_ShangTsung",
    "Fight_Kitana",
    "Fight_Jax",
    "Fight_Mileena",
    "Fight_Baraka",
    "Fight_Scorpion",
    "Fight_Raiden",
]

# Randomly select on each reset
state = np.random.choice(character_states)
env = retro.make(game="MortalKombatII-Snes", state=state)
```

This trains a more general fighter that works with multiple characters!

## Future Improvements

### Character Select Waypoint
**TODO for next time:** Create a `CharSelect_MortalKombatII.state` saved right at the character select screen. This would make creating character states much faster:

1. Create waypoint once: Boot game → save at character select
2. Use waypoint: Load CharSelect state → pick character → save when fight starts
3. Benefit: Skip 30-second intro every time!

Implementation:
```bash
# Create the waypoint:
python manual_state_creator.py --from-start
# At character select screen, manually save as CharSelect_MortalKombatII.state

# Then modify manual_state_creator to support:
python manual_state_creator.py --from-char-select
```

## Troubleshooting

### "AI never wins"
- Check if model exists: `ls -lh models/`
- Try using a later checkpoint: `mk2_ppo_1000000_steps.zip`
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
