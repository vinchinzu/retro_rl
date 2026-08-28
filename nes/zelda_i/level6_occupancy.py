"""Shared L6 leftover dict, dest predicate, and occupancy-walk halt.

Hops own buttons. This grades 1px occupancy and leftover — not a phase machine.
"""

from __future__ import annotations

from typing import Any, Iterable

from zelda_i.level6_overworld import LEVEL6
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "l6_leftover",
    "l6_play_dest_success",
    "occupancy_new_miss",
    "record_l6_walk",
]


def l6_leftover(snap: ZeldaSnapshot) -> dict[str, int]:
    """x/y/mode/screen/tile/rod/bow/arrows/keys/bombs/triforce from a snap."""
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "rod": int(snap.rod),
        "bow": int(snap.bow),
        "arrows": int(snap.arrows),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
    }


def l6_play_dest_success(
    snap: ZeldaSnapshot,
    *,
    not_room: int,
    passage_ok: bool = True,
    forbid: Iterable[int] = (),
    dest_room: int | None = None,
) -> bool:
    """level==6, TF 0x1F, rod!=0, play != not_room (optional exact dest_room)."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F or snap.rod == 0:
        return False
    if snap.screen in forbid:
        return False
    if dest_room is not None:
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == dest_room
        )
    if passage_ok and snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != not_room
    )


def occupancy_new_miss(
    walker: OccupancyWalker,
    xy: tuple[int, int],
    *,
    allow_first: bool = False,
) -> str | None:
    """Observe ``xy``. Return last_dir when a new miss should halt.

    ``allow_first`` records the leftover's first miss but does not halt on it.
    """
    prev = walker.last_dir
    before = walker.misses
    walker.observe(xy)
    if walker.misses <= before:
        return None
    if allow_first and before == 0:
        return None
    return prev


def record_l6_walk(
    samples: list[dict[str, Any]],
    snap: ZeldaSnapshot,
    *,
    reason: str,
    frames: int,
    period: int,
    misses: int,
    force: bool = False,
) -> dict[str, int]:
    """Write leftover and, on the sample cadence, a walk sample."""
    if force or frames <= 2 or frames % period == 0:
        samples.append(
            {
                "frame": frames,
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "reason": reason,
                "tile": int(snap.colliding_tile),
                "misses": misses,
            }
        )
    return l6_leftover(snap)
