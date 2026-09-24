"""Room 0x40 key: exact y-first ALIGN then fail-closed PATH. No emulator."""

from __future__ import annotations

import numpy as np

from zelda_i.level4.maze_path import (
    KEY_40_PATH_ANCHOR,
    Key40Phase,
    MAZE_40_TO_KEY,
    make_room_40_key_controller,
)
from zelda_i.level4.occupancy import ROOM_40_LEFTOVER_XY
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _pose(x: int, y: int, *, keys: int = 4) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x40
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    return ram


def test_the_open_loop_maze_starts_only_on_the_anchor() -> None:
    """The hold6 maze is open loop, so any start but the anchor walks it
    wrong; off the anchor ALIGN keeps walking (a lattice route: the old
    y-first press ran into the (160, 125) block from (158, 117))."""
    for xy in (ROOM_40_LEFTOVER_XY, (130, 159), (142, 159), (158, 117)):
        ctrl = make_room_40_key_controller()
        ctrl.phase = Key40Phase.ALIGN
        act = ctrl.step(read_snapshot(_pose(*xy)))
        assert ctrl.phase is Key40Phase.ALIGN, xy
        assert act.reason.startswith("align_"), xy


def test_exact_anchor_enters_path_next_frame() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.ALIGN
    act = ctrl.step(read_snapshot(_pose(*KEY_40_PATH_ANCHOR)))
    assert act.reason == "anchor_exact"
    assert ctrl.phase is Key40Phase.PATH
    assert ctrl.report()["path_start"] == [136, 165]


def test_path_exhaust_fails_closed_without_hunt() -> None:
    assert not hasattr(Key40Phase, "HUNT")
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.PATH
    ctrl.keys_before = 4
    act = None
    for _ in range(len(MAZE_40_TO_KEY) * 6 + 2):
        act = ctrl.step(read_snapshot(_pose(*ROOM_40_LEFTOVER_XY)))
        if ctrl.phase is Key40Phase.FAILED:
            break
    assert act is not None
    assert ctrl.phase is Key40Phase.FAILED
    assert act.reason == "path_done_no_key"
    assert ctrl.phase.name != "HUNT"


def test_keys_increment_in_play_arrives_done() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.PATH
    ctrl.keys_before = 4
    act = ctrl.step(read_snapshot(_pose(*KEY_40_PATH_ANCHOR, keys=5)))
    assert ctrl.phase is Key40Phase.DONE
    assert ctrl.success is True
    assert act.reason == "done"
