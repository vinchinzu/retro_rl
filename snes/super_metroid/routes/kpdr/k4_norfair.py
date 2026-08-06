"""Pure-first controllers for the K4 Business-to-Bubble Norfair path.

Business Center → Frog Save is the accepted K4.0 continuous extension (save
milestone). First Bubble visit is **Cathedral climb** (no Speed). Frog
Speedway is a post-Speed shortcut only (Boost Blocks need Speed Booster).

Public API is stable: registry, hops, tests, and probes import from this
module. Implementation lives in focused siblings:

- ``k4_business_frog`` — business↔frog, speedway, farm, farm→bubble scaffolds
- ``k4_cathedral`` — business→cathedral entrance→cathedral→rising tide
- ``k4_rising_tide`` — rising_tide→bubble
- ``k4_common`` — shared pose/elevator constants
- ``to_bat_cave`` — Bubble climb (re-exported for compat)
"""

from __future__ import annotations

# Bubble → Bat Cave product hop (re-exported for compat).
from super_metroid.routes.kpdr.to_bat_cave import (
    BUBBLE_PHASE_C_X_MIN,
    BUBBLE_PHASE_C_Y_MAX,
    BUBBLE_PHASE_C_Y_MIN,
    BUBBLE_PHASE_D_X,
    BUBBLE_PHASE_D_Y,
    BubblePhaseStop,
    bubble_phase_c_usable_right_contact,
    bubble_phase_d_top_band,
    play_bubble_to_bat_cave,
)
from super_metroid.routes.kpdr.k4_business_frog import (
    play_business_to_frog_save,
    play_farm_to_bubble,
    play_frog_save_to_business,
    play_frog_save_to_speedway,
    play_speedway_to_farm,
)
from super_metroid.routes.kpdr.k4_cathedral import (
    play_business_to_cathedral_entrance,
    play_cathedral_entrance_to_cathedral,
    play_cathedral_to_rising_tide,
)
from super_metroid.routes.kpdr.k4_rising_tide import (
    play_rising_tide_to_bubble,
)
from super_metroid.routes.kpdr.k4_wave import (
    play_bubble_to_single_chamber,
    play_single_to_double_chamber,
)
from super_metroid.routes.kpdr.speed_return import (
    play_speed_return_to_bubble,
)
from super_metroid.routes.kpdr.to_speed import (
    play_bat_cave_to_speed_hall,
    play_speed_hall_to_speed,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BAT_CAVE,
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_RISING_TIDE,
    ROOM_DOUBLE_CHAMBER,
    ROOM_SINGLE_CHAMBER,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
    ROOM_UPPER_NORFAIR_FARM,
)

__all__ = [
    "BUBBLE_PHASE_C_X_MIN",
    "BUBBLE_PHASE_C_Y_MAX",
    "BUBBLE_PHASE_C_Y_MIN",
    "BUBBLE_PHASE_D_X",
    "BUBBLE_PHASE_D_Y",
    "BubblePhaseStop",
    "ROOM_BAT_CAVE",
    "ROOM_BUBBLE",
    "ROOM_BUSINESS",
    "ROOM_CATHEDRAL",
    "ROOM_CATHEDRAL_ENTRANCE",
    "ROOM_FROG_SAVE",
    "ROOM_FROG_SPEEDWAY",
    "ROOM_RISING_TIDE",
    "ROOM_SPEED",
    "ROOM_SPEED_HALL",
    "ROOM_UPPER_NORFAIR_FARM",
    "bubble_phase_c_usable_right_contact",
    "bubble_phase_d_top_band",
    "play_bat_cave_to_speed_hall",
    "play_bubble_to_bat_cave",
    "play_business_to_cathedral_entrance",
    "play_business_to_frog_save",
    "play_cathedral_entrance_to_cathedral",
    "play_cathedral_to_rising_tide",
    "play_farm_to_bubble",
    "play_frog_save_to_business",
    "play_frog_save_to_speedway",
    "play_bubble_to_single_chamber",
    "play_single_to_double_chamber",
    "play_rising_tide_to_bubble",
    "play_speed_hall_to_speed",
    "play_speed_return_to_bubble",
    "play_speedway_to_farm",
    "ROOM_DOUBLE_CHAMBER",
    "ROOM_SINGLE_CHAMBER",
]
