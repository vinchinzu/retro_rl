"""South-pocket 0x40 key boundary policies. No emulator."""

from __future__ import annotations

import numpy as np

from zelda_i.level4.dungeon import ROOM_L4_ZOLS_40
from zelda_i.level4.key40 import make_room_40_key_controller
from zelda_i.level4.maze_path import KEY_40_PATH_ANCHOR, Key40Phase
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


def _pose(x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = ROOM_L4_ZOLS_40
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = 4
    return ram


def test_key_path_requires_the_exact_anchor() -> None:
    ctrl = make_room_40_key_controller()
    ctrl.phase = Key40Phase.ALIGN
    ctrl.keys_before = 4
    for xy in ((130, 159), (142, 159)):
        ctrl.step(read_snapshot(_pose(*xy)))
        assert ctrl.phase is Key40Phase.ALIGN

    ctrl.step(read_snapshot(_pose(*KEY_40_PATH_ANCHOR)))

    assert ctrl.phase is Key40Phase.PATH
    assert ctrl.report()["path_start"] == list(KEY_40_PATH_ANCHOR)
