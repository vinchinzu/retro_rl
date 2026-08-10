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

__all__ = [
    "EAST_GLASS_DOOR_X",
    "EAST_RETURN_X_MAX",
    "EAST_RETURN_Y_MAX",
    "EAST_RETURN_Y_MIN",
    "EAST_TO_GLASS_FRAMES",
    "EAST_TO_GLASS_SETTLE",
    "WH_EAST_DOOR_X",
    "WH_ELEV_BAND_X_MAX",
    "WH_ELEV_BAND_Y_MAX",
    "WH_TO_EAST_FRAMES",
    "WH_TO_EAST_SETTLE",
]
