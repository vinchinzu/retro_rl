"""Level6Inland29Controller: south-mouth clip, reclear, door-band, and the
rr-mzxn grid-bounds regression (LEFT+UP clip must not strand the BFS walker
outside its own OccupancyGrid).
"""

from __future__ import annotations

from retro_harness.nes import nes_action
from zelda_i.level6.inland29 import (
    CLIP_Y,
    WEST_SPAWN_XMIN,
    level6_inland29_success,
    make_inland29_controller,
)
from zelda_i.level6.overworld import (
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    WIZZROBE_ORANGE_TYPE,
)
from zelda_i.ram import ADDR_OBJ_HP, ADDR_OBJ_TYPE, PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram
from zelda_i.walk.physics import OccupancyGrid

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 6,
    "x": 120,
    "y": 141,
    "triforce": 0x1F,
    "rod": 1,
}


def _snap(*, screen: int = LEVEL6_DARK_29_ROOM, wizzrobes: int = 0, hp: int = 64, **fields: int):
    ram = make_ram(_DEFAULTS, screen=screen, **fields)
    for slot in range(1, wizzrobes + 1):
        ram[ADDR_OBJ_TYPE + slot] = WIZZROBE_ORANGE_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    return read_snapshot(ram)


def test_south_mouth_leftover_clips_left_up() -> None:
    ctl = make_inland29_controller()
    act = ctl.step(_snap(x=120, y=205))
    assert act.reason == "inland_clip"
    assert list(act.action) == list(nes_action("LEFT", "UP"))
    assert list(act.action) != list(nes_action("UP"))


def test_clip_releases_at_threshold_and_reclears_live_wizzrobes() -> None:
    ctl = make_inland29_controller()
    # Just past the clip threshold, with a live wizzrobe still on the room:
    # must fight, not resume plain occupancy toward the door.
    act = ctl.step(_snap(x=96, y=CLIP_Y, wizzrobes=1))
    assert act.reason != "inland_clip"
    assert ctl.fighter is not None


def test_door_band_clip_near_north_door() -> None:
    ctl = make_inland29_controller()
    # North band, off-center: should diagonal toward x=120 rather than
    # push straight UP into the door column's known tile-244 face (v1
    # leftover (48,109) regression).
    act = ctl.step(_snap(x=48, y=100))
    assert act.reason == "door_clip"
    assert list(act.action) == list(nes_action("RIGHT", "UP"))


def test_success_and_backtrack_predicate() -> None:
    dest = _snap(screen=0x19, x=120, y=205)
    assert level6_inland29_success(dest)
    leftover = _snap(x=120, y=205)
    assert not level6_inland29_success(leftover)
    back = _snap(screen=LEVEL6_DARK_39_ROOM, x=120, y=93)
    assert not level6_inland29_success(back)


def test_grid_xmin_wide_enough_for_the_clip_drift() -> None:
    """Regression guard for rr-mzxn.

    The generic ``dungeon.door_hop.DoorHopController`` used the module
    default ``OccupancyGrid`` (xmin=40): from the real south-mouth entry
    (120,205), holding LEFT+UP until y clears the clip threshold drifts
    Link to the west wall (x=32) *before* y clears -- byte-identical
    power-on stall at (32,145), room 0x29, 2/2. x=32 is outside xmin=40,
    so the BFS's own start-cell exception let it sit there, but every
    neighbor step back toward the goal needs x>=40 too, so shortest_path
    returns None and the walker stands forever.

    The dedicated controller's grid uses xmin=16 (WEST_SPAWN_XMIN), which
    keeps x=32 comfortably inside bounds so the walker can still find a
    route back to the north door.
    """
    stranded = (32, 145)
    goal = (120, 93)

    narrow = OccupancyGrid(xmin=40)
    assert narrow.shortest_path(stranded, goal) is None

    ctl = make_inland29_controller()
    assert ctl.walker.grid.xmin == WEST_SPAWN_XMIN == 16
    assert ctl.walker.grid.shortest_path(stranded, goal) is not None
