"""Level 7 dungeon stop contracts.

Entry room ``0x79`` is live (``Level7Entrance`` pin, recon Whistle poke).
Red Candle and leave rooms stay ``None``.  ``level7_entry_stop`` remains
fail-closed: evidence is ``fixture-live``, not spine-green.  The hypothesized
first-quest door/stair graph lives in ``level7.graph`` (source ids ``0x7xx``).
Navigation belongs in ``path.py`` / purpose-named modules, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import SCREEN_LEVEL7_ENTRY_ROOM, TF_BIT_L6, TF_BIT_L7
from zelda_i.dungeon.engine import DungeonRoomSpec
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


# Entry room 0x79 is live (drain_v2, Level7Entrance pin) but the pin is a
# recon Whistle poke on PostSwordStart — keep evidence off the spine set so
# ``level7_entry_stop`` stays fail-closed until a natural L6-leave drain.
LEVEL7_ENTRY_STOP = Level7StopSpec(
    "level7_entry", LEVEL7, SCREEN_LEVEL7_ENTRY_ROOM, evidence="fixture-live"
)
LEVEL7_RED_CANDLE_STOP = Level7StopSpec("level7_red_candle", LEVEL7, None)
# Settled post-fanfare OW leftover is UNMEASURED. Both level and screen stay
# None so ``level7_complete_stop`` fails closed. Do not invent an OW leave
# screen. A future MEASURED_POST_L7_EXIT copies MEASURED_POST_L6_EXIT's
# OverworldHandoff fields from a real fanfare leftover (handoff_from_ram):
# screen, link_x/y, mode, triforce (0x7F), candle (2), whistle, food (0),
# keys, bombs, rupees, selected_item, heart_containers (incoming+1),
# hearts lo==hi (ADDR_HEALTH) + ADDR_HEART_PARTIAL, rod, bow, arrows.
# verified stays False until that leftover is measured 2/2. L8 keeps
# PostLevel7Handoff.verified=False until then. See docs/tasks/l7c-prep-2026-09-03.md.
LEVEL7_COMPLETE_STOP = Level7StopSpec("level7_complete", None, None)

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
    """Settled L7 leave with shard, one natural heart, and full health.

    Fail-closed while ``LEVEL7_COMPLETE_STOP.screen`` is None (unmeasured OW
    leave). Filling TF ``0x7F``, Candle 2, Whistle, and HC+1 is not enough.
    """
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
    "RED_CANDLE",
    "TF_AFTER_LEVEL7",
    "TF_BEFORE_LEVEL7",
    "Level7StopSpec",
    "level7_complete_stop",
    "level7_entry_stop",
    "level7_red_candle_stop",
]
