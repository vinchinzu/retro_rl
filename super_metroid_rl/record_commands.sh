#!/bin/bash
# Quick Recording Commands for Super Metroid RL
# Copy/paste these into your terminal with the .venv activated

# =============================================================================
# FULL ROUTE RECORDING (Parlor -> Morph Ball -> Torizo)
# =============================================================================

# Option 1: Start fresh from Parlor (after Landing Site)
# python record_tasker.py record --state "Parlor and Alcatraz [from Landing Site]"

# Option 2: Start from ZebesStart (full run)
# python record_tasker.py record --state ZebesStart

# =============================================================================
# DESCENT SAVES (for training going DOWN)
# =============================================================================

# Parlor going down to Climb
# python state_manager.py record --start "Parlor and Alcatraz [from Landing Site]" --name "Descent_Parlor_toClimb"

# Pit Room going down to Elevator
# python state_manager.py record --start "Pit Room [from Climb]" --name "Descent_PitRoom_toElevator"

# =============================================================================
# RETURN TRIP SAVES (for training going UP - these are the hard ones!)
# =============================================================================

# FROM PIT ROOM GOING UP (where aliens/enemies start)
# This is the key one you mentioned - Pit Room going UP to Climb
# python state_manager.py record --start "Pit Room [from Blue Brinstar Elevator Room]" --name "Return_PitRoom_toClimb_withMorph"

# Climb going UP to Parlor
# python state_manager.py record --start "Climb [from Pit Room]" --name "Return_Climb_toParlor_withMorph"

# Parlor going RIGHT to Flyway
# python state_manager.py record --start "Parlor and Alcatraz [from Climb]" --name "Return_Parlor_toFlyway_withMorph"

# Flyway going RIGHT to Torizo
# python state_manager.py record --start "Flyway [from Parlor and Alcatraz]" --name "Return_Flyway_toTorizo_withMorph"

# =============================================================================
# QUICK START COMMANDS (copy these directly)
# =============================================================================

echo "=== QUICK START COMMANDS ==="
echo ""
echo "# Full route from Parlor:"
echo "python record_tasker.py record --state 'Parlor and Alcatraz [from Landing Site]'"
echo ""
echo "# Return trip from Pit Room (with aliens):"
echo "python state_manager.py record --start 'Pit Room [from Blue Brinstar Elevator Room]' --name 'Return_PitRoom_toClimb_withMorph'"
echo ""
echo "# Return trip from Climb:"
echo "python state_manager.py record --start 'Climb [from Pit Room]' --name 'Return_Climb_toParlor_withMorph'"
echo ""
echo "# Return trip from Parlor to Flyway:"
echo "python state_manager.py record --start 'Parlor and Alcatraz [from Climb]' --name 'Return_Parlor_toFlyway_withMorph'"
