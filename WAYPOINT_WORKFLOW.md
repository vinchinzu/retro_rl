# Fast Character State Creation with Waypoints

The waypoint system makes creating character states **10x faster** by skipping intro screens!

## ⚡ Quick Start (Recommended)

### Step 1: Create Waypoint (One-Time, ~30 seconds)

Pick a game and create the waypoint:

```bash
cd mortal_kombat_ii        # or street_fighter_ii or super_street_fighter_ii
./create_waypoint.sh
```

**What to do:**
1. Hold TAB through intro screens (10x speed!)
2. Navigate to CHARACTER SELECT screen
3. Press F1 to save waypoint
4. ESC to quit

**You only do this once per game!**

### Step 2: Create Character States (Fast, ~5-10 min for 12 chars)

```bash
./create_character_states_fast.sh
```

**Workflow (FAST!):**
1. ✅ Starts at character select (waypoint loaded!)
2. Select character with arrows + button
3. Wait for "FIGHT!" to appear
4. Press F1-F12 to save character state
5. **Press R to RESET** back to character select
6. Repeat for next character!

**That's it!** With the R reset key, you can blast through all characters in minutes.

## 📋 Character Mappings

### Mortal Kombat II (12 characters)
```
F1  = Liu Kang       F7  = Kitana
F2  = Kung Lao       F8  = Jax
F3  = Johnny Cage    F9  = Mileena
F4  = Reptile        F10 = Baraka
F5  = Sub-Zero       F11 = Scorpion
F6  = Shang Tsung    F12 = Raiden
```

### Street Fighter II Turbo (12 characters)
```
F1  = Ryu          F7  = Zangief
F2  = E.Honda      F8  = Dhalsim
F3  = Blanka       F9  = Balrog
F4  = Guile        F10 = Vega
F5  = Ken          F11 = Sagat
F6  = Chun-Li      F12 = M.Bison
```

### Super Street Fighter II (16 characters, 2 batches)

**Batch 1 (Original 12):**
```bash
./create_character_states_fast.sh 1
```
```
F1  = Ryu          F7  = Zangief
F2  = E.Honda      F8  = Dhalsim
F3  = Blanka       F9  = Balrog
F4  = Guile        F10 = Vega
F5  = Ken          F11 = Sagat
F6  = Chun-Li      F12 = M.Bison
```

**Batch 2 (New Challengers):**
```bash
./create_character_states_fast.sh 2
```
```
F1 = Cammy
F2 = Fei Long
F3 = Dee Jay
F4 = T.Hawk
```

## 🎮 Controls Reference

**Movement:**
- Arrow keys: Move/Navigate

**MK2 Controls:**
- Z: High Punch, X: Low Punch
- A: High Kick, S: Low Kick
- Q: Block, W: Run
- ENTER: Start

**SF2/SSF2 Controls:**
- Z: Heavy Punch, X: Medium Punch
- Q: Heavy Kick, A: Medium Kick, S: Light Kick
- ENTER: Start

**Special Keys:**
- **TAB:** Hold for turbo mode (10x speed!)
- **F1-F12:** Save character state
- **R:** RESET to waypoint (super fast!)
- **ESC:** Quit

## ⏱️ Time Estimates

**Without waypoint (old way):**
- 12 characters × 30 sec intro each = **6 minutes of just waiting**
- Plus navigation time = **30-45 minutes total**

**With waypoint + R reset (new way):**
- Create waypoint once: 30 seconds
- Each character: ~20-30 seconds
- 12 characters: **~5-10 minutes total!**
- 16 characters (SSF2): **~8-12 minutes total!**

## 🔍 Verify Your States

After creating all character states:

```bash
./validate_states.sh
```

This shows each state for 3 seconds. Verify:
- ✅ Correct character appears
- ✅ "FIGHT!" appears or timer starts
- ✅ Both fighters have full health
- ✅ Timer is full (90 seconds)

## 🚀 Start Training

Once you have all character states validated:

```bash
./train_multichar.sh
```

This will:
- Train with ALL characters (random selection each episode)
- Learn generic fighting strategies
- Save models as `{game}_multichar_ppo_*.zip`
- Run for 2M steps (~8-12 hours on GPU)

## 📁 Files Created

### Waypoint (one per game)
```
custom_integrations/{GameName-Snes}/CharSelect_{GameName}.state
```

Examples:
- `CharSelect_MortalKombatII.state`
- `CharSelect_StreetFighterIITurbo.state`
- `CharSelect_SuperStreetFighterII.state`

### Character States
```
custom_integrations/{GameName-Snes}/Fight_{CharacterName}.state
```

Examples:
- `Fight_LiuKang.state`, `Fight_KungLao.state`, etc.
- `Fight_Ryu.state`, `Fight_Ken.state`, etc.
- `Fight_Cammy.state`, `Fight_FeiLong.state`, etc.

## 💡 Tips

1. **Use TAB liberally** - Hold it during any transition or menu
2. **R is your friend** - After saving, immediately press R to reset
3. **Don't rush** - Wait for "FIGHT!" before pressing F-key
4. **Save early in session** - Better to have states early than perfect
5. **Validate after** - Run `./validate_states.sh` when done

## 🆘 Troubleshooting

### "Waypoint not found"
Run `./create_waypoint.sh` first.

### "Wrong character saved"
Make sure you press the right F-key when "FIGHT!" appears, not during character select.

### "State looks wrong in validation"
Delete the bad state and recreate it:
```bash
rm custom_integrations/{GameName-Snes}/Fight_{CharName}.state
./create_character_states_fast.sh
```

### "R key not resetting"
Make sure you created the waypoint first, and you're using `--from-waypoint` mode.

## 🎯 Full Workflow Example (MK2)

```bash
cd mortal_kombat_ii

# One-time setup (30 seconds)
./create_waypoint.sh
# → Hold TAB through intro
# → Navigate to character select
# → Press F1
# → ESC

# Fast character creation (~5-10 minutes)
./create_character_states_fast.sh
# → Pick Liu Kang → wait for FIGHT → F1 → R
# → Pick Kung Lao → wait for FIGHT → F2 → R
# → Pick Johnny Cage → wait for FIGHT → F3 → R
# ... repeat for all 12 characters

# Validate (30 seconds)
./validate_states.sh

# Start training! (8-12 hours)
./train_multichar.sh
```

**Total time investment:** ~10-15 minutes to create all states!

Compare to 30-45 minutes the old way. The waypoint + R reset combo is a **massive** time saver!
