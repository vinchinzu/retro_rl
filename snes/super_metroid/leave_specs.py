"""Named dest-glance LeaveSpecs for continuous SpineHops.

Compose, spine, and tests import these (do not copy numbers). Graded by
:mod:`super_metroid.hop_glance`. Bands are the hop leave still, not the
next-hop pin reload. In-room Main Shaft seats live in ``WS_MAIN_PHASE_SPECS``.
"""

from __future__ import annotations

from dataclasses import dataclass

from super_metroid.routes.kpdr.room_ids import (
    ROOM_ALPHA_PB,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BOWLING,
    ROOM_BUSINESS,
    ROOM_CATERPILLAR,
    ROOM_CRATERIA_KIHUNTER,
    ROOM_EAST_TUNNEL,
    ROOM_GLASS,
    ROOM_GRAVITY,
    ROOM_HELLWAY,
    ROOM_HOMING_GEEMER,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
    ROOM_MOAT,
    ROOM_PANCAKES,
    ROOM_PHANTOON,
    ROOM_RED_BRINSTAR_ELEVATOR,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WEST_OCEAN,
    ROOM_WEST_TUNNEL,
    ROOM_WS_ATTIC,
    ROOM_WS_BASEMENT,
    ROOM_WS_ENTRANCE,
    ROOM_WS_MAIN,
)

__all__ = [
    "LeaveSpec",
    "LEAVE_BY_HOP",
    "ICE_TO_SNAKE",
    "ICE_SNAKE_TO_TUTORIAL",
    "ICE_TUTORIAL_TO_GATE",
    "ICE_GATE_TO_BUSINESS",
    "ICE_BUSINESS_TO_WAREHOUSE",
    "WAREHOUSE_TO_EAST",
    "EAST_TO_GLASS",
    "GLASS_TO_WEST",
    "WEST_TO_BELOW",
    "BELOW_TO_BAT",
    "BAT_TO_RED",
    "RED_TO_HELLWAY",
    "HELLWAY_TO_CATERPILLAR",
    "CATERPILLAR_TO_ALPHA_PB",
    "ALPHA_PB_TO_CATERPILLAR",
    "CATERPILLAR_TO_ELEVATOR",
    "ELEVATOR_TO_KIHUNTER",
    "KIHUNTER_TO_MOAT",
    "MOAT_CROSS",
    "WEST_OCEAN_TO_WS",
    "WS_ENTRANCE_TO_MAIN",
    "WS_MAIN_TO_BASEMENT",
    "WS_BASEMENT_TO_PHANTOON",
    "PHANTOON_FIGHT",
    "PHANTOON_LEAVE",
    "WS_BASEMENT_TO_MAIN",
    "WS_MAIN_TO_ATTIC",
    "ATTIC_TO_WEST_OCEAN",
    "WEST_OCEAN_TO_PANCAKES",
    "PANCAKES_TO_HOMING_GEEMER",
    "HOMING_GEEMER_TO_BOWLING",
    "BOWLING_TO_GRAVITY",
    "GRAVITY_COLLECT",
    "WS_MAIN_PIT_SHOT",
    "WS_MAIN_GRATE_SEAT",
    "WS_MAIN_WEST_SUPER",
    "WS_MAIN_MID_CLIMB",
    "WS_MAIN_ATTIC_SEAT",
    "WS_MAIN_PHASE_SPECS",
]


@dataclass(frozen=True)
class LeaveSpec:
    """What a human would check in a couple of seconds on a still."""

    hop: str
    room: int
    x: tuple[int, int]
    y: tuple[int, int]
    pose_class: str = "any"
    gs: int = 8
    dt: int = 0
    boss_bit: int | None = None
    min_health: int = 1


def _spec(
    hop: str,
    room: int,
    x: tuple[int, int],
    y: tuple[int, int],
    pose_class: str = "stand",
    *,
    boss_bit: int | None = None,
) -> LeaveSpec:
    return LeaveSpec(
        hop=hop,
        room=room,
        x=x,
        y=y,
        pose_class=pose_class,
        boss_bit=boss_bit,
    )


# Ice return (K5 reverse prefix).
ICE_TO_SNAKE = _spec("ice_to_snake", ROOM_ICE_SNAKE, (430, 520), (360, 430))
ICE_SNAKE_TO_TUTORIAL = _spec(
    "ice_snake_to_tutorial", ROOM_ICE_TUTORIAL, (20, 80), (100, 160), "any"
)
ICE_TUTORIAL_TO_GATE = _spec(
    "ice_tutorial_to_gate", ROOM_ICE_GATE, (450, 900), (100, 200), "any"
)
ICE_GATE_TO_BUSINESS = _spec(
    "ice_gate_to_business", ROOM_BUSINESS, (20, 100), (880, 960), "any"
)
ICE_BUSINESS_TO_WAREHOUSE = _spec(
    "ice_business_to_warehouse", ROOM_WAREHOUSE, (20, 60), (100, 160)
)

# K5 reverse Brinstar.
WAREHOUSE_TO_EAST = _spec(
    "warehouse_to_east", ROOM_EAST_TUNNEL, (150, 280), (300, 420), "any"
)
EAST_TO_GLASS = _spec("east_to_glass", ROOM_GLASS, (150, 280), (350, 420), "any")
GLASS_TO_WEST = _spec("glass_to_west", ROOM_WEST_TUNNEL, (150, 280), (100, 180), "any")
WEST_TO_BELOW = _spec(
    "west_to_below", ROOM_BELOW_SPAZER, (400, 520), (350, 420), "any"
)
BELOW_TO_BAT = _spec("below_to_bat", ROOM_BAT, (400, 520), (100, 180), "any")
BAT_TO_RED = _spec("bat_to_red", ROOM_RED_TOWER, (150, 280), (2380, 2500), "any")
# Ice-pin ordinary left door ~(39,139) p11; tape morph p29 is not this still.
RED_TO_HELLWAY = _spec("red_to_hellway", ROOM_HELLWAY, (20, 80), (120, 180), "any")
HELLWAY_TO_CATERPILLAR = _spec(
    "hellway_to_caterpillar", ROOM_CATERPILLAR, (20, 80), (1380, 1450), "any"
)
CATERPILLAR_TO_ALPHA_PB = _spec(
    "caterpillar_to_alpha_pb", ROOM_ALPHA_PB, (320, 365), (150, 190)
)

# K6 Moat / over-ocean.
ALPHA_PB_TO_CATERPILLAR = _spec(
    "alpha_pb_to_caterpillar", ROOM_CATERPILLAR, (20, 80), (1910, 1940), "any"
)
CATERPILLAR_TO_ELEVATOR = _spec(
    "caterpillar_to_elevator",
    ROOM_RED_BRINSTAR_ELEVATOR,
    (110, 145),
    (270, 315),
    "any",
)
ELEVATOR_TO_KIHUNTER = _spec(
    "elevator_to_kihunter", ROOM_CRATERIA_KIHUNTER, (370, 415), (670, 725), "any"
)
KIHUNTER_TO_MOAT = _spec("kihunter_to_moat", ROOM_MOAT, (20, 80), (120, 170))
MOAT_CROSS = _spec("moat_cross", ROOM_WEST_OCEAN, (30, 80), (1140, 1185))
WEST_OCEAN_TO_WS = _spec(
    "west_ocean_to_ws", ROOM_WS_ENTRANCE, (40, 90), (120, 160)
)

# Wrecked Ship / Phantoon. Dual-green leaves (not dest-room only).
WS_ENTRANCE_TO_MAIN = _spec(
    "ws_entrance_to_main", ROOM_WS_MAIN, (1000, 1100), (880, 940)
)
# Hatch drop ~(657,92) p24; morph in the hatch is not a leave.
WS_MAIN_TO_BASEMENT = _spec(
    "ws_main_to_basement", ROOM_WS_BASEMENT, (600, 720), (60, 160), "any"
)
# Dual-green exit of basement: 0xCD13 ~(39,124) p81. Morph in the door is a miss.
WS_BASEMENT_TO_PHANTOON = _spec(
    "ws_basement_to_phantoon", ROOM_PHANTOON, (20, 80), (90, 160), "door"
)
PHANTOON_FIGHT = _spec(
    "phantoon_fight", ROOM_PHANTOON, (20, 80), (160, 210), boss_bit=1
)
PHANTOON_LEAVE = _spec(
    "phantoon_loot_exit",
    ROOM_WS_BASEMENT,
    (1200, 1280),
    (120, 160),
    boss_bit=1,
)
# Residual hop dest (not on POST_ICE_SPINE). Main Shaft floor hatch ~(1144,1900).
# RED leftover is the still (basement hatch ~(630-690, 160-190)), not this band.
WS_BASEMENT_TO_MAIN = _spec(
    "ws_basement_to_main", ROOM_WS_MAIN, (1100, 1200), (1800, 2000)
)
# Residual hop dest (not on POST_ICE_SPINE). Attic floor after ceiling-door
# settle — not the Main Shaft y=31 transition. Human dest ~(1135,184) p21.
WS_MAIN_TO_ATTIC = _spec(
    "ws_main_to_attic", ROOM_WS_ATTIC, (1050, 1220), (100, 220), "door"
)
# s23 tape leaves (wide bands; not Main Shaft GREEN). pose any: door/morph/idle.
ATTIC_TO_WEST_OCEAN = _spec(
    "attic_to_west_ocean", ROOM_WEST_OCEAN, (1, 80), (100, 180), "any"
)
WEST_OCEAN_TO_PANCAKES = _spec(
    "west_ocean_to_pancakes", ROOM_PANCAKES, (1, 80), (100, 180), "any"
)
PANCAKES_TO_HOMING_GEEMER = _spec(
    "pancakes_to_homing_geemer", ROOM_HOMING_GEEMER, (1, 80), (100, 180), "any"
)
HOMING_GEEMER_TO_BOWLING = _spec(
    "homing_geemer_to_bowling", ROOM_BOWLING, (1, 120), (100, 220), "any"
)
BOWLING_TO_GRAVITY = _spec(
    "bowling_to_gravity", ROOM_GRAVITY, (80, 320), (80, 220), "any"
)
# Dual from f022887: (127, 135) p46 gs=8 items 0x3125.
GRAVITY_COLLECT = _spec(
    "gravity_collect", ROOM_GRAVITY, (100, 160), (110, 160), "any"
)

# In-room Main Shaft seats (not dest hops). pose any: fire slope is p3.
# Approach still in Main, not Basement. Weak on purpose.
WS_MAIN_PIT_SHOT = _spec(
    "ws_main_pit_shot", ROOM_WS_MAIN, (1100, 1220), (1920, 2020), "any"
)
# Usable take02 outgoing pin: fire ~(1223,1860) p3 and LEFT+A takeoff
# (1227,1231)×(1852,1856). Not land (1189,1883) p2, not take04 (1195,1883).
WS_MAIN_GRATE_SEAT = _spec(
    "ws_main_grate_seat", ROOM_WS_MAIN, (1216, 1232), (1852, 1868), "any"
)
# y~1675 in shaft, not West Super room 0xCDA8.
WS_MAIN_WEST_SUPER = _spec(
    "ws_main_west_super", ROOM_WS_MAIN, (1080, 1220), (1650, 1700), "any"
)
WS_MAIN_MID_CLIMB = _spec(
    "ws_main_mid_climb", ROOM_WS_MAIN, (1080, 1220), (630, 710), "any"
)
WS_MAIN_ATTIC_SEAT = _spec(
    "ws_main_attic_seat", ROOM_WS_MAIN, (1111, 1159), (0, 160), "stand"
)

WS_MAIN_PHASE_SPECS: dict[str, LeaveSpec] = {
    "pit_shot": WS_MAIN_PIT_SHOT,
    "grate_seat": WS_MAIN_GRATE_SEAT,
    "west_super": WS_MAIN_WEST_SUPER,
    "mid_climb": WS_MAIN_MID_CLIMB,
    "attic_seat": WS_MAIN_ATTIC_SEAT,
    "attic_door": WS_MAIN_TO_ATTIC,
}

LEAVE_BY_HOP: dict[str, LeaveSpec] = {
    spec.hop: spec
    for spec in (
        ICE_TO_SNAKE,
        ICE_SNAKE_TO_TUTORIAL,
        ICE_TUTORIAL_TO_GATE,
        ICE_GATE_TO_BUSINESS,
        ICE_BUSINESS_TO_WAREHOUSE,
        WAREHOUSE_TO_EAST,
        EAST_TO_GLASS,
        GLASS_TO_WEST,
        WEST_TO_BELOW,
        BELOW_TO_BAT,
        BAT_TO_RED,
        RED_TO_HELLWAY,
        HELLWAY_TO_CATERPILLAR,
        CATERPILLAR_TO_ALPHA_PB,
        ALPHA_PB_TO_CATERPILLAR,
        CATERPILLAR_TO_ELEVATOR,
        ELEVATOR_TO_KIHUNTER,
        KIHUNTER_TO_MOAT,
        MOAT_CROSS,
        WEST_OCEAN_TO_WS,
        WS_ENTRANCE_TO_MAIN,
        WS_MAIN_TO_BASEMENT,
        WS_BASEMENT_TO_PHANTOON,
        PHANTOON_FIGHT,
        PHANTOON_LEAVE,
        WS_BASEMENT_TO_MAIN,
        WS_MAIN_TO_ATTIC,
        ATTIC_TO_WEST_OCEAN,
        WEST_OCEAN_TO_PANCAKES,
        PANCAKES_TO_HOMING_GEEMER,
        HOMING_GEEMER_TO_BOWLING,
        BOWLING_TO_GRAVITY,
        GRAVITY_COLLECT,
        WS_MAIN_PIT_SHOT,
        WS_MAIN_GRATE_SEAT,
        WS_MAIN_WEST_SUPER,
        WS_MAIN_MID_CLIMB,
        WS_MAIN_ATTIC_SEAT,
    )
}
