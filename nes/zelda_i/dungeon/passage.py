"""Side-view passage (mode 9) crossing: east column down, floor west, west ladder up.

Every underworld passage the route crosses east-to-west has the same room:
a ladder at the west wall (x=48) rising to the exit mouth (y=93), a floor at
y=189, and the arrival column in the east. What differs is only where the
east column is and how Link leaves the ledge onto it.
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import ZeldaSnapshot

PASSAGE_WEST_X = 48
PASSAGE_FLOOR_Y = 189
PASSAGE_MOUTH_Y = 93
PASSAGE_PIT_TILE = 250
PASSAGE_STAIRS_TILES = range(0x70, 0x74)


def passage_step(
    snap: ZeldaSnapshot,
    *,
    east_x: int,
    align: int = 4,
    drop: tuple[str, ...] = ("DOWN",),
    both_ways: bool = False,
    mouth_y: int = PASSAGE_MOUTH_Y,
) -> FrameAction:
    """One frame of the east-to-west crossing. Never UP in the east column.

    ``drop`` is the press that leaves the ledge onto the east column: a plain
    ``DOWN`` on a ladder column, ``LEFT+DOWN`` where the column is a gap in
    the ledge. That drop only works on the column itself, so a ``LEFT+DOWN``
    drop waits for ``x >= east_x`` (short of it the LEFT half walks Link back:
    L8 0x0F 172<->174 for 4000 frames). ``both_ways`` also steps LEFT onto a
    column Link has overshot.
    """
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    if y >= PASSAGE_FLOOR_Y - align:
        if x > PASSAGE_WEST_X + align:
            return FrameAction(nes_action("LEFT"), "cellar_floor_west")
        if x < PASSAGE_WEST_X - align:
            return FrameAction(nes_action("RIGHT"), "cellar_floor_east")
        return FrameAction(nes_action("UP"), "cellar_west_climb")
    if abs(x - PASSAGE_WEST_X) <= align:
        if y > mouth_y + align:
            return FrameAction(nes_action("UP"), "cellar_west_up")
        if tile in PASSAGE_STAIRS_TILES:
            return FrameAction(nes_idle_action(), "cellar_exit_warp")
        return FrameAction(nes_action("UP"), "cellar_west_lip")
    gap_drop = "LEFT" in drop
    short = x < east_x if gap_drop else x < east_x - align
    if tile == PASSAGE_PIT_TILE and short:
        return FrameAction(nes_action("RIGHT"), "cellar_pit_to_east")
    if short:
        return FrameAction(nes_action("RIGHT"), "cellar_to_east")
    if both_ways and x > east_x + align:
        return FrameAction(nes_action("LEFT"), "cellar_to_east")
    return FrameAction(nes_action(*drop), "cellar_east_drop")
