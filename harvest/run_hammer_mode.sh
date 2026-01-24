#!/bin/bash
# Run bot in HAMMER-ONLY mode (smash stones instead of lift+toss)
# This is faster than lift+carry+toss for small stones

cat << 'INFO'
========================================================
HAMMER-ONLY MODE
========================================================
Bot will use hammer to smash rocks and stones instead
of lifting and tossing them into ponds.

Clearing strategy:
  - Big rocks (0x04) → Hammer (can't be lifted)
  - Small stones (0x06, 0x09, 0x0A, 0x0B) → Hammer (faster than lift+toss)

Controls:
  - Bot starts in AUTO mode
  - Toggle: LB+RB+SELECT (controller) or A+S+TAB (keyboard)
  - ESC to quit
========================================================

INFO

python3 << 'PYTHON'
import sys
sys.path.insert(0, '.')
import harvest_bot as hb
import stable_retro as retro

# Create bot with hammer priority
bot = hb.AutoClearBot(priority=[hb.DebrisType.ROCK, hb.DebrisType.STONE])
bot.pond_dispose_enabled = True
bot.brain.prefer_tools_over_lift = True  # USE HAMMER INSTEAD OF LIFT!
bot.brain.only_liftable = False

print(f"Bot config:")
print(f"  Priority: {[d.name for d in bot.priority]}")
print(f"  Pond dispose: {bot.pond_dispose_enabled}")
print(f"  Prefer tools over lift: {bot.brain.prefer_tools_over_lift}")
print(f"  Only liftable: {bot.brain.only_liftable}")
print()

# Start session
session = hb.PlaySession(
    state="Y1_After_First_Rock_Smash",
    scale=4,
    bot=bot,
    autoplay=True
)
session.run()
PYTHON
