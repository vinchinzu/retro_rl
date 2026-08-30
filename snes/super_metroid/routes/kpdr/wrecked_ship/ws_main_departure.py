"""Tape-locked Main Shaft grate departure windows (rr-1xc2.8.2).

Living policy is take02 slope LEFT+A. Take04 walk-right to the save
alcove is a different policy. Data only — not ``climb_action``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from super_metroid.plm import SHOT_BLOCK_PLM_IDS
from super_metroid.takeoff import TakeoffWindow

# Grounded standing / aiming-up. Geometry GROUNDED_POSES + AIM_POSES; copied
# so this tape lock does not import the climb classifier.
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_AIM = frozenset({5, 6, 7, 8})
_FIRE_POSES = _GROUNDED | _AIM

SLOPE_LEFT_A_POLICY = "slope_left_a"
WALK_RIGHT_ALCOVE_POLICY = "walk_right_alcove"
LIVING_POLICY = SLOPE_LEFT_A_POLICY

# take02 last grounded takeoff (1231, 1852) p3; take03 in-place (1227, 1856) p1.
# Not airborne x≈1221–1227 on the take02 jump (y~1800).
SLOPE_LEFT_A = TakeoffWindow((1227, 1231), "LEFT", min_momentum=0)
SLOPE_LEFT_A_Y = (1852, 1856)

# take04/05 save-ledge jump. Not the living handoff.
ALCOVE_LEFT_A_X = (1242, 1243)
ALCOVE_LEFT_A_Y = (1851, 1851)

# take02 fire seat (usable grate_seat pin). Separate from the LEFT+A takeoff.
TAKE02_LIP_FIRE = (1223, 1860, 3)
TAKE02_LEFT_A_SEAT = (1231, 1852, 3)
TAKE02_PEAK_Y = 1763

TAKE03_LIP_FIRE = (1227, 1856, 3)
TAKE03_LEFT_A_SEAT = (1227, 1856, 1)
TAKE03_PEAK_Y = 1763

TAKE04_LIP_FIRE = (1195, 1883, 3)
TAKE04_LEFT_A_SEAT = (1242, 1851, 1)
TAKE04_PEAK_Y = 1795

TAKE05_LIP_FIRE = (1243, 1851, 6)
TAKE05_LEFT_A_SEAT = (1243, 1851, 2)
TAKE05_PEAK_Y = 1795


@dataclass(frozen=True)
class TapePose:
    frame: int
    x: int
    y: int
    pose: int
    buttons: tuple[str, ...]
    vy: int = 0


@dataclass(frozen=True)
class GrateDeparture:
    lip_fire: TapePose | None
    first_shot: TapePose | None
    grounded_takeoff: TapePose | None
    first_left_a: TapePose | None
    first_right_after_shot: TapePose | None
    peak_y: int | None
    policy: str


def at_slope_left_a(x: int, y: int) -> bool:
    """Living take02/03 grounded LEFT+A seat. Tight; not GRATE_LAND."""
    return (
        SLOPE_LEFT_A.x_range[0] <= int(x) <= SLOPE_LEFT_A.x_range[1]
        and SLOPE_LEFT_A_Y[0] <= int(y) <= SLOPE_LEFT_A_Y[1]
    )


def at_alcove_left_a(x: int, y: int) -> bool:
    """take04/05 save-ledge jump. Rejected as the living handoff."""
    return (
        ALCOVE_LEFT_A_X[0] <= int(x) <= ALCOVE_LEFT_A_X[1]
        and ALCOVE_LEFT_A_Y[0] <= int(y) <= ALCOVE_LEFT_A_Y[1]
    )


def _plm_ids(row: Mapping[str, object]) -> set[int]:
    raw_ids = row.get("plm_ids")
    if isinstance(raw_ids, (list, tuple)):
        return {int(i) for i in raw_ids}
    ids: set[int] = set()
    for item in row.get("plms") or ():
        if isinstance(item, (list, tuple)) and len(item) > 1:
            ids.add(int(item[1]))
        elif isinstance(item, Mapping) and item.get("id") is not None:
            ids.add(int(item["id"]))
    return ids


def _pose(row: Mapping[str, object]) -> TapePose:
    return TapePose(
        frame=int(row["frame"]),
        x=int(row["x"]),
        y=int(row["y"]),
        pose=int(row["pose"]),
        buttons=tuple(str(b) for b in (row.get("buttons") or ())),
        vy=int(row.get("vy") or 0),
    )


def _policy(takeoff: TapePose | None, first_left_a: TapePose | None) -> str:
    if takeoff is not None and at_slope_left_a(takeoff.x, takeoff.y):
        return SLOPE_LEFT_A_POLICY
    if takeoff is not None and at_alcove_left_a(takeoff.x, takeoff.y):
        return WALK_RIGHT_ALCOVE_POLICY
    if first_left_a is not None and first_left_a.x >= ALCOVE_LEFT_A_X[0]:
        return WALK_RIGHT_ALCOVE_POLICY
    return "unknown"


def scan_grate_departure(trace: Sequence[Mapping[str, object]]) -> GrateDeparture:
    """Lip fire, first 0xD080-family spawn, and the LEFT+A takeoff from a tape."""
    rows = list(trace)
    first_shot: TapePose | None = None
    for row in rows:
        if _plm_ids(row) & SHOT_BLOCK_PLM_IDS:
            first_shot = _pose(row)
            break
    lip_fire: TapePose | None = None
    if first_shot is not None:
        for row in rows:
            if int(row["frame"]) > first_shot.frame:
                break
            buttons = set(row.get("buttons") or ())
            if (
                "X" in buttons
                and int(row["y"]) >= 1850
                and int(row["pose"]) in _FIRE_POSES
            ):
                lip_fire = _pose(row)
    first_left_a: TapePose | None = None
    first_right: TapePose | None = None
    grounded: TapePose | None = None
    peak_y: int | None = None
    if first_shot is not None:
        for row in rows:
            frame = int(row["frame"])
            if frame < first_shot.frame:
                continue
            pose = _pose(row)
            if (
                pose.vy == 0
                and pose.pose in _GROUNDED
                and first_left_a is None
            ):
                grounded = pose
            buttons = set(pose.buttons)
            if first_right is None and "RIGHT" in buttons:
                first_right = pose
            if (
                first_left_a is None
                and "LEFT" in buttons
                and "A" in buttons
                and pose.y >= 1760
            ):
                first_left_a = pose
                peak_y = pose.y
            if first_left_a is not None and pose.y < peak_y:
                peak_y = pose.y
            if first_left_a is not None and frame > first_left_a.frame + 80:
                break
    return GrateDeparture(
        lip_fire=lip_fire,
        first_shot=first_shot,
        grounded_takeoff=grounded,
        first_left_a=first_left_a,
        first_right_after_shot=first_right,
        peak_y=peak_y,
        policy=_policy(grounded, first_left_a),
    )


__all__ = [
    "ALCOVE_LEFT_A_X",
    "ALCOVE_LEFT_A_Y",
    "GrateDeparture",
    "LIVING_POLICY",
    "SLOPE_LEFT_A",
    "SLOPE_LEFT_A_POLICY",
    "SLOPE_LEFT_A_Y",
    "TAKE02_LEFT_A_SEAT",
    "TAKE02_LIP_FIRE",
    "TAKE02_PEAK_Y",
    "TAKE03_LEFT_A_SEAT",
    "TAKE03_LIP_FIRE",
    "TAKE03_PEAK_Y",
    "TAKE04_LEFT_A_SEAT",
    "TAKE04_LIP_FIRE",
    "TAKE04_PEAK_Y",
    "TAKE05_LEFT_A_SEAT",
    "TAKE05_LIP_FIRE",
    "TAKE05_PEAK_Y",
    "TapePose",
    "WALK_RIGHT_ALCOVE_POLICY",
    "at_alcove_left_a",
    "at_slope_left_a",
    "scan_grate_departure",
]
