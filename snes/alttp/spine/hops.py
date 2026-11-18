"""One hop table for the ALTTP opening spine.

A row is a through-id, a stop name, the stage list, and the success node.
``--through`` plays rows until that id. Ids that are not rows fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from alttp.opening_route.escape_graph import (
    N_ROOM_01,
    N_ROOM_50,
    N_ROOM_72,
    N_ROOM_81,
    N_ROOM_82,
    N_ROOM_80,
    N_SANCTUARY,
    N_SANCTUARY_HEART,
    N_SANCTUARY_EXIT,
    N_KAKARIKO,
    N_ROOM_C9,
)
from alttp.opening_route.full_tip import VERIFIED_TIP_SEGMENT_IDS

# Kept so older imports still resolve. room_01 / room_72 are spine rows now.
ROOM_01_BLOCKER = "room_01 not on continuous tip"
ROOM_72_BLOCKER = "room_72 not on continuous tip; stairs not composed into full_tip"
ZELDA_BLOCKER = "zelda not on continuous tip; $F3CC==1 not measured"

FAIL_CLOSED_NOTES: dict[str, tuple[str, ...]] = {}
FAIL_CLOSED_BLOCKER: dict[str, str] = {}

@dataclass(frozen=True)
class SpineHop:
    """One spine stop. ``stages`` are continuous segment ids. ``success`` is the tip node."""

    through: str
    stop: str
    stages: tuple[str, ...]
    success: str


# Development power-on rows. Survival/search results require an audited
# forward replay and do not promote the Clean graph ladder.
SPINE_HOPS: tuple[SpineHop, ...] = (
    SpineHop(
        through=N_ROOM_50,
        stop=N_ROOM_50,
        stages=tuple(VERIFIED_TIP_SEGMENT_IDS),
        success=N_ROOM_50,
    ),
    SpineHop(
        through=N_ROOM_01,
        stop=N_ROOM_01,
        stages=("room_50_east_to_0x01",),
        success=N_ROOM_01,
    ),
    SpineHop(
        through=N_ROOM_72,
        stop=N_ROOM_72,
        stages=("room_01_down_to_0x72",),
        success=N_ROOM_72,
    ),
    SpineHop(
        through=N_ROOM_82,
        stop=N_ROOM_82,
        stages=("room_72_south_to_0x82",),
        success=N_ROOM_82,
    ),
    SpineHop(
        through=N_ROOM_81,
        stop=N_ROOM_81,
        stages=("room_82_west_to_0x81",),
        success=N_ROOM_81,
    ),
    SpineHop(
        through="zelda", stop="zelda", stages=("room_81_to_zelda",), success=N_ROOM_80,
    ),
    SpineHop(
        through="sanctuary", stop="sanctuary",
        stages=("escort_to_sanctuary",), success=N_SANCTUARY,
    ),
    SpineHop(
        through="sanctuary_heart", stop="sanctuary_heart",
        stages=("sanctuary_heart",), success=N_SANCTUARY_HEART,
    ),
    SpineHop(
        through="sanctuary_exit", stop="sanctuary_exit",
        stages=("sanctuary_exit",), success=N_SANCTUARY_EXIT,
    ),
    SpineHop(
        through="kakariko", stop="kakariko",
        stages=("sanctuary_to_kakariko",), success=N_KAKARIKO,
    ),
    # Branch from the Sanctuary exit. Not played by --through kakariko.
    SpineHop(
        through="eastern_entry", stop="eastern_entry",
        stages=("eastern_grounds", "sahasrahla_hut", "eastern_approach"),
        success=N_ROOM_C9,
    ),
)

SPINE_THROUGH: tuple[str, ...] = tuple(
    [hop.through for hop in SPINE_HOPS] + list(FAIL_CLOSED_BLOCKER)
)
DEFAULT_THROUGH = N_ROOM_50


def hop_for(through: str) -> SpineHop | None:
    """The row ``--through`` stops on, or None when that id is not continuous."""
    for hop in SPINE_HOPS:
        if hop.through == through:
            return hop
    return None


def hops_through(through: str) -> Sequence[SpineHop]:
    """Rows to play, inclusive of ``through``. Unknown and fail-closed ids yield nothing.

    ``eastern_entry`` continues the Sanctuary exit (heart, south door, proved
    screens, hut, hedge, lobby). It does not take the Kakariko branch.
    """
    if through == "eastern_entry":
        base = hops_through("sanctuary_exit")
        hop = hop_for("eastern_entry")
        if not base or hop is None:
            return ()
        return tuple(base) + (hop,)
    played: list[SpineHop] = []
    for hop in SPINE_HOPS:
        if hop.through == "eastern_entry":
            continue
        played.append(hop)
        if hop.through == through:
            return tuple(played)
    return ()


# Red-run pin stem. Not a stage in the hop table.
FAIL_STAGE = "fail"


def save_point_name(prefix: str, stage: str) -> str:
    """``<prefix>_<stage>`` pin stem. Both parts are single filenames."""
    for label, part in (("prefix", prefix), ("stage", stage)):
        if (
            not isinstance(part, str)
            or not part
            or part in {".", ".."}
            or "/" in part
            or "\\" in part
        ):
            raise ValueError(f"save-point {label} must be a filename, got {part!r}")
    return f"{prefix}_{stage}"


def last_stage(through: str) -> str | None:
    """Last stage played before ``through`` resolves. None when that id is not continuous."""
    hop = hop_for(through)
    if hop is None or not hop.stages:
        return None
    return hop.stages[-1]
