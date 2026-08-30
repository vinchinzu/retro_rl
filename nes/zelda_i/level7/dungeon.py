"""Level 7 dungeon stop contracts.

No Level 7 dungeon room has been observed live yet.  Keep room ids out of this
module until RAM evidence exists; ``None`` therefore means "not verified" and
every predicate fails closed.  Navigation and controller loops belong in
``path.py`` / purpose-named modules, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from zelda_i.anchors import TF_BIT_L6, TF_BIT_L7
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


# Room ids deliberately remain absent.  Replace ``None`` only from live RAM
# evidence, never from a walkthrough grid or source topology decode.
LEVEL7_ENTRY_STOP = Level7StopSpec("level7_entry", LEVEL7, None)
LEVEL7_RED_CANDLE_STOP = Level7StopSpec("level7_red_candle", LEVEL7, None)
# The settled post-fanfare level/screen is part of the L7 -> L8 handoff and is
# not known yet, so both fields remain closed.
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
    "RED_CANDLE",
    "TF_AFTER_LEVEL7",
    "TF_BEFORE_LEVEL7",
    "Level7StopSpec",
    "level7_complete_stop",
    "level7_entry_stop",
    "level7_red_candle_stop",
]
