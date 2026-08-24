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

# Below Spazer 0xA408 left blue door into Bat Room 0xA3DD
# (outbound ``play_bat_to_below_spazer`` runs RIGHT into Below from Bat).
# Floor path reverse of below_spazer_floor_to_west (LEFT across water band).
# Pure pin right-floor ~(472,393) p82 after west_to_below.
BELOW_BAT_DOOR_X = 40
BELOW_TO_BAT_FRAMES = 2000
BELOW_TO_BAT_SETTLE = 260
# Dual-green leave pin post_ice_below_to_bat_pure ~(472, 139) p12.
# Compose from Ice can enter Bat morph/mid-platform; pin this sill before
# handing to bat_to_red. Stay left of the right door (~x=510) so we do not
# fall back into Below Spazer.
BAT_SILL_X_MIN = 400
BAT_SILL_X_MAX = 500
BAT_SILL_Y_MIN = 100
BAT_SILL_Y_MAX = 165
BAT_SILL_PIN_FRAMES = 240

# Bat Room 0xA3DD left blue door into Red Tower 0xA253 bottom
# (outbound ``play_red_tower_to_bat`` runs RIGHT from Red bottom into Bat left).
# High path is the three dry pipe platforms (reverse of bat→below). Water
# under the pipes is a different climb (HJ + no Gravity: crouch-jump +
# down-grab). Named pin right high sill ~(472,139) p12 (crouch).
BAT_TO_RED_DOOR_SEAT_X = 80
BAT_TO_RED_HIGH_Y = 165
BAT_TO_RED_TRAVERSE_BUDGET = 2200
BAT_TO_RED_PROGRESS_WINDOW = 42
BAT_TO_RED_JUMP_PERIOD = 36
BAT_TO_RED_JUMP_HOLD = 24
BAT_TO_RED_RUNUP = 20
BAT_TO_RED_WATER_CJ_CROUCH = 4
BAT_TO_RED_WATER_CJ_JUMP = 16
BAT_TO_RED_WATER_GRAB = 24
BAT_TO_RED_EXIT_RUN = 20
BAT_TO_RED_EXIT_SHOOT = 4
BAT_TO_RED_EXIT_SPIN = 30
BAT_TO_RED_EXIT_HOLD = 280
BAT_TO_RED_EXIT_SETTLE = 360

# Red Tower 0xA253 bottom → Hellway 0xA2F7 top-right door (K5 hop 12).
# Reverse of ``play_red_tower_to_bat`` descent bands (lower zigzag → tunnel →
# temporary floor → upper zigzag → Bat right), climbed with Hi-Jump spin hops.
# Pure pin bottom ~(216,2443) p10 after bat_to_red; Hellway door block [15,7]
# ≈ top-right sill y≈139 x≳220.
RED_BOTTOM_Y = 2440
RED_LOWER_LIP_Y = 2090
RED_TUNNEL_Y = 1880
RED_FLOOR_Y = 1600
RED_TOP_DOOR_Y = 180
RED_TOP_DOOR_X = 220
RED_ZIG_X_MIN = 45
RED_ZIG_X_MAX = 220
RED_CLIMB_FRAMES = 9000
RED_TO_HELLWAY_EXIT_RUN = 24
RED_TO_HELLWAY_EXIT_SHOOT = 4
RED_TO_HELLWAY_EXIT_SPIN = 36
RED_TO_HELLWAY_EXIT_HOLD = 320
RED_TO_HELLWAY_EXIT_SETTLE = 360

__all__ = [
    "BAT_TO_RED_DOOR_SEAT_X",
    "BAT_TO_RED_EXIT_HOLD",
    "BAT_TO_RED_EXIT_RUN",
    "BAT_TO_RED_EXIT_SETTLE",
    "BAT_TO_RED_EXIT_SHOOT",
    "BAT_TO_RED_EXIT_SPIN",
    "BAT_TO_RED_HIGH_Y",
    "BAT_TO_RED_JUMP_HOLD",
    "BAT_TO_RED_JUMP_PERIOD",
    "BAT_TO_RED_PROGRESS_WINDOW",
    "BAT_TO_RED_RUNUP",
    "BAT_TO_RED_TRAVERSE_BUDGET",
    "BAT_TO_RED_WATER_CJ_CROUCH",
    "BAT_TO_RED_WATER_CJ_JUMP",
    "BAT_TO_RED_WATER_GRAB",
    "BELOW_BAT_DOOR_X",
    "RED_BOTTOM_Y",
    "RED_CLIMB_FRAMES",
    "RED_FLOOR_Y",
    "RED_LOWER_LIP_Y",
    "RED_TO_HELLWAY_EXIT_HOLD",
    "RED_TO_HELLWAY_EXIT_RUN",
    "RED_TO_HELLWAY_EXIT_SETTLE",
    "RED_TO_HELLWAY_EXIT_SHOOT",
    "RED_TO_HELLWAY_EXIT_SPIN",
    "RED_TOP_DOOR_X",
    "RED_TOP_DOOR_Y",
    "RED_TUNNEL_Y",
    "RED_ZIG_X_MAX",
    "RED_ZIG_X_MIN",
    "BELOW_TO_BAT_FRAMES",
    "BELOW_TO_BAT_SETTLE",
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
