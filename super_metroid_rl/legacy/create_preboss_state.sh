#!/bin/bash
# Helper script to create the PreBoss state for curriculum learning
echo "==================================================="
echo "       SUPER METROID PRE-BOSS STATE CREATOR        "
echo "==================================================="
echo "1. The game will launch in 'ZebesStart' state."
echo "2. Use keyboard controls (Arrow Keys, Z=Dash, X=Jump/Shoot)."
echo "3. Navigate to the HALLWAY outside the Bomb Torizo room."
echo "4. Press 'M' to save the state."
echo "5. Close the window when done."
echo "==================================================="
read -p "Press Enter to start..."

./run_bot.sh play_human.py --game SuperMetroid-Snes --state ZebesStart

if [ -f "ManualSave.state" ]; then
    echo "State saved!"
    mkdir -p custom_integrations/SuperMetroid-Snes
    mv ManualSave.state custom_integrations/SuperMetroid-Snes/PreBoss.state
    echo "Moved to custom_integrations/SuperMetroid-Snes/PreBoss.state"
    echo "Ready for Phase 4!"
else
    echo "No state saved. (Did you press 'M'?)"
fi
