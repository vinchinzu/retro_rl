"""K5 reverse-tunnel geometry — room-prefixed constants only.

Warehouse elev exit (post Business→Warehouse pure) sits upper-left
~(20–60, 100–160). East Tunnel is the left blue door of Warehouse
(outbound ``play_east_to_warehouse`` runs RIGHT into Warehouse).
"""

from __future__ import annotations

# Warehouse 0xA6A1 — left blue door into East Tunnel 0xCF80.
WH_EAST_DOOR_X = 28
WH_ELEV_BAND_X_MAX = 60
WH_ELEV_BAND_Y_MAX = 160
WH_TO_EAST_FRAMES = 900
WH_TO_EAST_SETTLE = 360

# East Tunnel ordinary settle band after reverse entry (right side → y≈139
# shelf or mid-screen after multi-screen load). Room id gate is primary.
EAST_RETURN_X_MAX = 280
EAST_RETURN_Y_MIN = 100
EAST_RETURN_Y_MAX = 450

# East Tunnel 0xCF80 bottom-left blue door into Glass Tunnel 0xCEFB
# (outbound ``play_glass_to_east`` runs RIGHT into East).
# Tape: mid-floor x≈216 y≈395 LEFT-run to door x≤40.
EAST_GLASS_DOOR_X = 40
EAST_TO_GLASS_FRAMES = 600
EAST_TO_GLASS_SETTLE = 260

# Glass Tunnel 0xCEFB bottom-left blue door into West Tunnel 0xCF54
# (outbound ``play_west_to_glass`` runs RIGHT into Glass).
# Mirror of west_to_glass run/shoot/spin/hold/settle knobs; LEFT reverse.
# Pure pin mid-floor ~(216,395) after east_to_glass; tape left-side ~(17,395).
GLASS_WEST_DOOR_X = 40
GLASS_TO_WEST_RUN = 80
GLASS_TO_WEST_SHOOT = 5
GLASS_TO_WEST_SPIN = 50
GLASS_TO_WEST_HOLD = 300
GLASS_TO_WEST_SETTLE = 260

# West Tunnel 0xCF54 left blue door into Below Spazer 0xA408
# (outbound ``play_below_spazer_floor_to_west`` runs RIGHT into West).
# Mirror of west_to_glass / glass_to_west knobs; LEFT reverse from mid-right pin.
# Pure pin mid-floor ~(216,139) after glass_to_west; tape left-side ~(16,395).
WEST_BELOW_DOOR_X = 40
WEST_TO_BELOW_RUN = 80
WEST_TO_BELOW_SHOOT = 5
WEST_TO_BELOW_SPIN = 50
WEST_TO_BELOW_HOLD = 300
WEST_TO_BELOW_SETTLE = 260

__all__ = [
    "EAST_GLASS_DOOR_X",
    "EAST_RETURN_X_MAX",
    "EAST_RETURN_Y_MAX",
    "EAST_RETURN_Y_MIN",
    "EAST_TO_GLASS_FRAMES",
    "EAST_TO_GLASS_SETTLE",
    "GLASS_TO_WEST_HOLD",
    "GLASS_TO_WEST_RUN",
    "GLASS_TO_WEST_SETTLE",
    "GLASS_TO_WEST_SHOOT",
    "GLASS_TO_WEST_SPIN",
    "GLASS_WEST_DOOR_X",
    "WEST_BELOW_DOOR_X",
    "WEST_TO_BELOW_HOLD",
    "WEST_TO_BELOW_RUN",
    "WEST_TO_BELOW_SETTLE",
    "WEST_TO_BELOW_SHOOT",
    "WEST_TO_BELOW_SPIN",
    "WH_EAST_DOOR_X",
    "WH_ELEV_BAND_X_MAX",
    "WH_ELEV_BAND_Y_MAX",
    "WH_TO_EAST_FRAMES",
    "WH_TO_EAST_SETTLE",
]
