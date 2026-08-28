"""KPDR room ids, Super-collect evidence, and shared segment helpers.

Room int constants live in :mod:`super_metroid.routes.kpdr.room_ids` (cycle-free)
and are re-exported here for back-compat. Controllers keep importing from this
module; progression graph data imports ``room_ids`` directly.

Includes the early post-Spore rooms (Super / Farming / Pink PB) that are part
of the continuous KPDR spine (K0–K1), not a separate vanilla-first route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from super_metroid.routes.kpdr.room_ids import (  # noqa: F401
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BAT_CAVE,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_BLUE_BRINSTAR_ETANK,
    ROOM_BOMB_TORIZO,
    ROOM_BUBBLE,
    ROOM_BUSINESS,
    ROOM_CATHEDRAL,
    ROOM_CATHEDRAL_ENTRANCE,
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
    ROOM_CLIMB,
    ROOM_CONSTRUCTION,
    ROOM_DACHORA,
    ROOM_DOUBLE_CHAMBER,
    ROOM_EAST_TUNNEL,
    ROOM_FARMING,
    ROOM_FIRST_MISSILE,
    ROOM_FLYWAY,
    ROOM_FROG_SAVE,
    ROOM_FROG_SPEEDWAY,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_GREEN_ELEVATOR,
    ROOM_GREEN_MAIN_SHAFT,
    ROOM_GREEN_PIRATES,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_ICE,
    ROOM_ICE_ACID,
    ROOM_ICE_GATE,
    ROOM_ICE_SNAKE,
    ROOM_ICE_TUTORIAL,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_LANDING_SITE,
    ROOM_LOWER_MUSHROOMS,
    ROOM_MORPH,
    ROOM_NOOB,
    ROOM_PARLOR,
    ROOM_PINK_PB,
    ROOM_PIT,
    ROOM_RED_TOWER,
    ROOM_HELLWAY,
    ROOM_CATERPILLAR,
    ROOM_ALPHA_PB,
    ROOM_RED_BRINSTAR_ELEVATOR,
    ROOM_CRATERIA_KIHUNTER,
    ROOM_MOAT,
    ROOM_WEST_OCEAN,
    ROOM_WS_ENTRANCE,
    ROOM_WS_MAIN,
    ROOM_WS_ATTIC,
    ROOM_WS_BASEMENT,
    ROOM_PHANTOON,
    ROOM_RISING_TIDE,
    ROOM_SINGLE_CHAMBER,
    ROOM_SPAZER,
    ROOM_SPEED,
    ROOM_SPEED_HALL,
    ROOM_SPORE_KIHUNTER,
    ROOM_SPORE_SPAWN,
    ROOM_SUPER,
    ROOM_TERMINATOR,
    ROOM_UPPER_NORFAIR_FARM,
    ROOM_VARIA,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WAVE,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)

ITEM_HI_JUMP = 0x0100
ITEM_VARIA = 0x0001
ITEM_SPEED = 0x2000

@dataclass(frozen=True)
class SuperCollectEvidence:
    entry_frame: int
    collect_frame: int
    exit_frame: int | None
    max_super_missiles: int
    final_room_id: int
    samus_x: int
    samus_y: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class PowerBombEvidence:
    entry_frame: int
    collect_frame: int | None
    max_super_missiles: int
    max_power_bombs: int
    final_room_id: int
    samus_x: int
    samus_y: int
    reached_big_pink: bool
    reached_pb_room: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
