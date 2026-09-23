"""Level 7 dungeon stop contracts.

Entry ``0x79``, Red Candle cellar ``0x4A``, and the post-fanfare OW leave
``0x42`` are **spine-green** from power-on (Survival; Food is still the
disclosed poke, ``rr-8t4.4``). Not a Clean claim. Graph: ``level7.graph``.
Navigation belongs in ``path.py`` / purpose-named modules, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM, TF_BIT_L6, TF_BIT_L7
from zelda_i.dungeon.engine import DungeonRoomSpec
from zelda_i.overworld.stitch import OverworldHandoff
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

LEVEL7 = 7
TF_BEFORE_LEVEL7 = 0x3F
TF_AFTER_LEVEL7 = TF_BEFORE_LEVEL7 | TF_BIT_L7
RED_CANDLE = 2
_ROUTE_EVIDENCE = frozenset({"natural-segment", "spine-green"})


@dataclass(frozen=True)
class Level7StopSpec:
    """Exact RAM endpoint whose room/screen must be live-observed first."""

    stop_id: str
    level: int | None
    screen: int | None
    mode: int = PLAY_MODE
    evidence: str = "hypothesis"
    route_eligible: bool = False

    @property
    def observed(self) -> bool:
        return self.level is not None and self.screen is not None


# PROMOTED 2026-09-05 (rr-8t4.1). The condition this spec was waiting on --
# "a natural L6-leave drain" -- is met: from power-on the spine walks the
# post-L6 pocket to 0x24, warps out on the naturally-owned Recorder to the L4
# island door 0x45 (``level7/warp.py``), rejoins the green pond chain at 0x55,
# drains pond 0x42 and enters play 0x79 -- no Whistle poke anywhere. Live
# **2/2 byte-identical** from power-on (``recordings/l7entry_warp_v4_t0.json``
# / ``l7entry_warp_v5.json``, ``set_state=0``): warp 7 blows / 1982f, drain
# 466f on stair candidate 0, leftover L7 0x79 (120,205), ``writes=0``.
#
# SURVIVAL SCOPE, NOT A CLEAN CLAIM. ``food >= 1`` here is still satisfied by
# the disclosed ``ADDR_FOOD`` write in ``SurvivalBaitPurchaseController`` --
# the natural 60R bait shop is bead ``rr-8t4.4`` and the L6->shop overworld
# route is still unmapped. This flag makes the Survival spine green to 0x79;
# it does not make L7 entry Clean, and ``docs/STATUS.md`` is the planner's.
LEVEL7_ENTRY_STOP = Level7StopSpec(
    "level7_entry",
    LEVEL7,
    SCREEN_LEVEL7_ENTRY_ROOM,
    evidence="spine-green",
    route_eligible=True,
)
# PROMOTED 2026-09-05. Natural Red Candle cellar 0x4A on the power-on
# Survival tape (``--through level7``, 2/2).
LEVEL7_RED_CANDLE_STOP = Level7StopSpec(
    "level7_red_candle",
    LEVEL7,
    0x4A,
    mode=9,
    evidence="spine-green",
    route_eligible=True,
)
LEVEL7_COMPLETE_STOP = Level7StopSpec(
    "level7_complete",
    0,
    0x42,
    mode=PLAY_MODE,
    evidence="spine-green",
    route_eligible=True,
)
# Power-on ``--through level7`` 2/2, set_state=0, byte-identical
# (recordings/survival_spine.json + survival_spine_v2.json, 256779f).
# Do not copy the TF-0 fixture leftover (bombs 6 / 44R / 4 HC / B=recorder).
# Packet is route-eligible: L8 walked 0x42 -> 0x6D from this leftover
# (power-on ``--through level8-entry`` 2/2).
MEASURED_POST_L7_EXIT = OverworldHandoff(
    screen=0x42,
    link_x=96,
    link_y=93,
    mode=PLAY_MODE,
    triforce=TF_AFTER_LEVEL7,
    keys=1,
    bombs=1,
    rupees=66,
    # 12 = the gathered spine's continuous power-on arrival (2026-09-23,
    # full_poweron3, set_state=0, 132R); the fixture tape measured 9.
    heart_containers=12,
    selected_item=1,  # bombs leftover from the L7 bomb walls
    whistle=1,
    food=0,
    rod=1,
    bow=1,
    arrows=1,
    candle=RED_CANDLE,
    evidence="measured-level7-exit-2of2",
    verified=True,
    route_eligible=True,
)

# There are intentionally no executable DungeonRoomSpec rows yet.  Add one
# only after its room id, entry geometry, enemy census, and reward are live.
LEVEL7_ROOM_SPECS: tuple[DungeonRoomSpec, ...] = ()


def _at_exact_stop(snap: ZeldaSnapshot, spec: Level7StopSpec) -> bool:
    return bool(
        spec.observed
        and spec.route_eligible
        and spec.evidence in _ROUTE_EVIDENCE
        and snap.level == spec.level
        and snap.screen == spec.screen
        and snap.mode == spec.mode
        and not snap.transitioning
    )


def level7_entry_stop(
    snap: ZeldaSnapshot,
    *,
    whistle: int,
    food: int,
    spec: Level7StopSpec = LEVEL7_ENTRY_STOP,
) -> bool:
    """Live L7 entry with the exact L6 handoff and required natural items."""
    return bool(
        _at_exact_stop(snap, spec)
        and snap.triforce == TF_BEFORE_LEVEL7
        and (snap.triforce & TF_BIT_L6)
        and whistle >= 1
        and food >= 1
    )


def level7_red_candle_stop(
    snap: ZeldaSnapshot,
    *,
    candle: int,
    whistle: int,
    food: int,
    spec: Level7StopSpec = LEVEL7_RED_CANDLE_STOP,
) -> bool:
    """Natural Red Candle gate after the Hungry Goriya consumed Food."""
    return bool(
        _at_exact_stop(snap, spec)
        and snap.triforce == TF_BEFORE_LEVEL7
        and candle == RED_CANDLE
        and whistle >= 1
        and food == 0
    )


def level7_complete_stop(
    snap: ZeldaSnapshot,
    *,
    candle: int,
    whistle: int,
    incoming_heart_containers: int | None,
    spec: Level7StopSpec = LEVEL7_COMPLETE_STOP,
) -> bool:
    """Settled L7 leave with shard, one natural heart, and full health."""
    return bool(
        incoming_heart_containers is not None
        and _at_exact_stop(snap, spec)
        and snap.triforce == TF_AFTER_LEVEL7
        and candle == RED_CANDLE
        and whistle >= 1
        and snap.heart_containers == incoming_heart_containers + 1
        and snap.health_is_full
    )


__all__ = [
    "LEVEL7",
    "LEVEL7_COMPLETE_STOP",
    "LEVEL7_ENTRY_STOP",
    "LEVEL7_RED_CANDLE_STOP",
    "LEVEL7_ROOM_SPECS",
    "MEASURED_POST_L7_EXIT",
    "RED_CANDLE",
    "TF_AFTER_LEVEL7",
    "TF_BEFORE_LEVEL7",
    "Level7StopSpec",
    "level7_complete_stop",
    "level7_entry_stop",
    "level7_red_candle_stop",
]
