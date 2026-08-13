"""Human-recording guide polylines for KPDR practice segments.

Waypoints are approximate KPDR route pins (room world pixels). Guides are
**data** — split by arc so Ice/Moat stubs do not dump into Cathedral/Wave.

Modules
-------
* ``early`` — Parlor Alcatraz, Big Pink Charge, GHZ
* ``spazer`` — early Spazer climb/collect/top-drop
* ``norfair_k4`` — Cathedral → Bubble climb + post-Speed return
* ``wave`` — Bubble → Single → Double → Wave
* ``ice_moat`` — Ice branch, Moat, West Ocean stubs

Also: post-Torizo **Parlor Alcatraz** left wall-jump climb (Flyway door shaft),
sm-json-data “Alcatraz Escape” family — **not** the product Terminator platform
hop / bomb-tunnel path.

Post-supers **Big Pink Charge** (main shaft → Chozo collect → ordinary return →
optional GHZ green door) is the K1 detour inside ``play_big_pink_to_ghz``.

Early Spazer (K2.2 optional): continuous-like Below Spazer ``0xA408`` left
entry → left-shaft wall-jump → top green Super door → Spazer Room collect →
return. Overlay + one-pager: ``docs/tasks/EARLY_SPAZER_HUMAN.md``.

Post-Speed **Wave / Ice / Moat** re-record (``rr-dbu.12``): routes
``speed-to-wave`` / ``speed-to-ice-moat`` from ``scratch/post_speed_collected``.
Runbook: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md``. Partial legacy tape
``tasks/speed_to_ice_moat_human.json`` stops in Double Chamber — do not treat
as full-path evidence.

K6 **West Ocean → WS** product pure is dual GREEN
(``play_west_ocean_over_ocean_spark`` / ``pure-ws``). Human free-record:

* Optional WO practice: start ``scratch/post_moat_west_ocean_spark.state``
  (``0x93FE``), route ``west-ocean-to-ws``.
* **Ship work:** start ``scratch/post_west_ocean_ws_spark.state`` (``0xCA08``
  ~(57,139) gs=8), route ``ws-entrance``. Multi-take:
  ``practice_takes.py --segment ws-entrance``.
"""

from __future__ import annotations

from retro_harness.path_overlay import RoomGuide
from super_metroid.routes.kpdr.guides.early import (
    GUIDE_BIG_PINK_CHARGE,
    GUIDE_GHZ_ENTRY,
    GUIDE_PARLOR_ALCATRAZ,
)
from super_metroid.routes.kpdr.guides.ice_moat import (
    GUIDE_BUSINESS_TO_ICE,
    GUIDE_ICE_GATE,
    GUIDE_ICE_ROOM,
    GUIDE_ICE_SNAKE,
    GUIDE_ICE_TUTORIAL,
    GUIDE_KIHUNTER_MOAT,
    GUIDE_MOAT,
    GUIDE_WEST_OCEAN,
    GUIDE_WS_ENTRANCE,
)
from super_metroid.routes.kpdr.guides.norfair_k4 import (
    GUIDE_BAT_CAVE,
    GUIDE_BAT_CAVE_RETURN,
    GUIDE_BUBBLE,
    GUIDE_BUBBLE_SAVE,
    GUIDE_CATHEDRAL,
    GUIDE_CATHEDRAL_ENTRANCE,
    GUIDE_RISING_TIDE,
    GUIDE_SPEED_HALL_RETURN,
    GUIDE_SPEED_ROOM_EXIT,
)
from super_metroid.routes.kpdr.guides.spazer import (
    GUIDE_BELOW_SPAZER_EARLY,
    GUIDE_BELOW_SPAZER_TOP_DROP,
    GUIDE_SPAZER_ROOM,
)
from super_metroid.routes.kpdr.guides.wave import (
    GUIDE_BUBBLE_TO_SINGLE,
    GUIDE_DOUBLE_CHAMBER,
    GUIDE_DOUBLE_CHAMBER_RECOVER,
    GUIDE_SINGLE_CHAMBER,
    GUIDE_WAVE_ROOM,
)

GUIDE_BY_ROOM: dict[int, RoomGuide] = {
    g.room_id: g
    for g in (
        GUIDE_CATHEDRAL_ENTRANCE,
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
        GUIDE_PARLOR_ALCATRAZ,
        GUIDE_BIG_PINK_CHARGE,
        GUIDE_GHZ_ENTRY,
        GUIDE_BELOW_SPAZER_EARLY,
        GUIDE_SPAZER_ROOM,
        # Wave/Ice room defaults (Bat return / Bubble-to-Single override via route).
        GUIDE_SPEED_ROOM_EXIT,
        GUIDE_SPEED_HALL_RETURN,
        GUIDE_SINGLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER,
        GUIDE_WAVE_ROOM,
        GUIDE_BUSINESS_TO_ICE,
        GUIDE_ICE_GATE,
        GUIDE_ICE_TUTORIAL,
        GUIDE_ICE_SNAKE,
        GUIDE_ICE_ROOM,
        GUIDE_KIHUNTER_MOAT,
        GUIDE_MOAT,
        GUIDE_WEST_OCEAN,
        GUIDE_WS_ENTRANCE,
    )
}

ROUTE_PRESETS: dict[str, tuple[RoomGuide, ...]] = {
    "cathedral-to-bat": (
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
    ),
    "cathedral-to-bubble": (
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
    ),
    "entrance-to-bat": (
        GUIDE_CATHEDRAL_ENTRANCE,
        GUIDE_CATHEDRAL,
        GUIDE_RISING_TIDE,
        GUIDE_BUBBLE,
        GUIDE_BUBBLE_SAVE,
        GUIDE_BAT_CAVE,
    ),
    "bubble-to-bat": (GUIDE_BUBBLE, GUIDE_BUBBLE_SAVE, GUIDE_BAT_CAVE),
    "cathedral-only": (GUIDE_CATHEDRAL,),
    "rising-only": (GUIDE_RISING_TIDE,),
    "bubble-only": (GUIDE_BUBBLE, GUIDE_BUBBLE_SAVE),
    # Post-Torizo: Flyway door → Alcatraz left wall-jump shaft (human demo).
    "parlor-left": (GUIDE_PARLOR_ALCATRAZ,),
    "parlor-alcatraz": (GUIDE_PARLOR_ALCATRAZ,),
    # Post-supers K1: Charge detour (collect + ordinary return); optional GHZ.
    "charge-collect-return": (GUIDE_BIG_PINK_CHARGE,),
    "big-pink-to-ghz": (GUIDE_BIG_PINK_CHARGE, GUIDE_GHZ_ENTRY),
    # Early Spazer (100% / K2.2): wall-jump climb + collect + return.
    "early-spazer": (GUIDE_BELOW_SPAZER_EARLY, GUIDE_SPAZER_ROOM),
    "spazer-collect-return": (GUIDE_BELOW_SPAZER_EARLY, GUIDE_SPAZER_ROOM),
    "below-spazer-only": (GUIDE_BELOW_SPAZER_EARLY,),
    "spazer-only": (GUIDE_SPAZER_ROOM,),
    # Post-Spazer clean drop only (skip climb/bomb thrash).
    "spazer-top-drop": (GUIDE_BELOW_SPAZER_TOP_DROP,),
    "spazer-return-drop": (GUIDE_SPAZER_ROOM, GUIDE_BELOW_SPAZER_TOP_DROP),
    # Double Chamber only (Spazer missile ledge → Super → Wave). Practice from
    # post_single_to_double_chamber_* leave. Main (wave purple) + floor recover
    # fallback (trap red) from human take04.
    "double-chamber-to-wave": (
        GUIDE_DOUBLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER_RECOVER,
        GUIDE_WAVE_ROOM,
    ),
    "dc-missile-wave": (
        GUIDE_DOUBLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER_RECOVER,
        GUIDE_WAVE_ROOM,
    ),
    # Post-Speed Wave branch only (rr-dbu.12 phase A — min Wave collect).
    "speed-to-wave": (
        GUIDE_SPEED_ROOM_EXIT,
        GUIDE_SPEED_HALL_RETURN,
        GUIDE_BAT_CAVE_RETURN,
        GUIDE_BUBBLE_TO_SINGLE,
        GUIDE_SINGLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER_RECOVER,
        GUIDE_WAVE_ROOM,
    ),
    # Full aspirational path: Wave → Ice (via Business) → Moat stretch.
    # Free-play rooms between Wave return and Business have no polyline;
    # guide lights up when you re-enter a listed room.
    "speed-to-ice-moat": (
        GUIDE_SPEED_ROOM_EXIT,
        GUIDE_SPEED_HALL_RETURN,
        GUIDE_BAT_CAVE_RETURN,
        GUIDE_BUBBLE_TO_SINGLE,
        GUIDE_SINGLE_CHAMBER,
        GUIDE_DOUBLE_CHAMBER,
        GUIDE_WAVE_ROOM,
        GUIDE_BUSINESS_TO_ICE,
        GUIDE_ICE_GATE,
        GUIDE_ICE_TUTORIAL,
        GUIDE_ICE_SNAKE,
        GUIDE_ICE_ROOM,
        GUIDE_KIHUNTER_MOAT,
        GUIDE_MOAT,
    ),
    # Ice branch only (start from Business pin when available).
    "business-to-ice": (
        GUIDE_BUSINESS_TO_ICE,
        GUIDE_ICE_GATE,
        GUIDE_ICE_TUTORIAL,
        GUIDE_ICE_SNAKE,
        GUIDE_ICE_ROOM,
    ),
    # K6: post-Moat handoff West Ocean → Wrecked Ship (optional human; pure GREEN).
    "west-ocean-to-ws": (
        GUIDE_WEST_OCEAN,
        GUIDE_WS_ENTRANCE,
    ),
    "west-ocean": (
        GUIDE_WEST_OCEAN,
        GUIDE_WS_ENTRANCE,
    ),
    # K6: product WS pin after over-ocean spark — ship free-record start.
    "ws-entrance": (GUIDE_WS_ENTRANCE,),
    "wrecked-ship": (GUIDE_WS_ENTRANCE,),
}


def guide_for_room(room_id: int) -> RoomGuide | None:
    return GUIDE_BY_ROOM.get(int(room_id))


__all__ = [
    "GUIDE_BY_ROOM",
    "ROUTE_PRESETS",
    "guide_for_room",
    # early
    "GUIDE_PARLOR_ALCATRAZ",
    "GUIDE_BIG_PINK_CHARGE",
    "GUIDE_GHZ_ENTRY",
    # spazer
    "GUIDE_BELOW_SPAZER_EARLY",
    "GUIDE_SPAZER_ROOM",
    "GUIDE_BELOW_SPAZER_TOP_DROP",
    # norfair_k4
    "GUIDE_CATHEDRAL_ENTRANCE",
    "GUIDE_CATHEDRAL",
    "GUIDE_RISING_TIDE",
    "GUIDE_BUBBLE",
    "GUIDE_BUBBLE_SAVE",
    "GUIDE_BAT_CAVE",
    "GUIDE_SPEED_ROOM_EXIT",
    "GUIDE_SPEED_HALL_RETURN",
    "GUIDE_BAT_CAVE_RETURN",
    # wave
    "GUIDE_BUBBLE_TO_SINGLE",
    "GUIDE_SINGLE_CHAMBER",
    "GUIDE_DOUBLE_CHAMBER",
    "GUIDE_DOUBLE_CHAMBER_RECOVER",
    "GUIDE_WAVE_ROOM",
    # ice_moat
    "GUIDE_BUSINESS_TO_ICE",
    "GUIDE_ICE_GATE",
    "GUIDE_ICE_TUTORIAL",
    "GUIDE_ICE_SNAKE",
    "GUIDE_ICE_ROOM",
    "GUIDE_KIHUNTER_MOAT",
    "GUIDE_MOAT",
    "GUIDE_WEST_OCEAN",
    "GUIDE_WS_ENTRANCE",
]
