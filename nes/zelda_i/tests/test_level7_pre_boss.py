"""L7 0x29 east BOMB wall hop (`level7.pre_boss`).

Probe constants from ``scratch/probe_l7_7b_cellar_cross.py``: stand (208,141)
face RIGHT, approach (96,189)->(208,189)->(208,141), dest live 0x2A.
Factory configures ``BombWallController``; it does not write bombs.
"""

from __future__ import annotations

import ast

from retro_harness.controls import pressed_nes_buttons
from zelda_i.level7.pre_boss import (
    make_room29_east_bomb_controller,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.tests.ram_helpers import make_ram


def test_factory_walks_the_south_band_then_faces_the_east_wall() -> None:
    """The factory's controller walks (96,189) -> (208,189) -> (208,141)
    and faces RIGHT at the stand: the measured approach, not the centre."""
    ctl = make_room29_east_bomb_controller()
    ctl.wave_settle_frames = 0  # Test geometry after the live wave has settled.

    def press(x: int, y: int) -> list[str]:
        ram = make_ram(
            {"mode": PLAY_MODE, "level": 7, "screen": 0x29, "bombs": 4}, x=x, y=y
        )
        # A waypoint hand-off spends one idle frame; the second step walks.
        ctl.step(read_snapshot(ram))
        return pressed_nes_buttons(list(ctl.step(read_snapshot(ram)).action))

    assert press(96, 189) == ["RIGHT"]
    assert press(208, 189) == ["UP"]
    assert press(208, 141) == ["RIGHT"]


def test_factory_never_shares_instances() -> None:
    assert make_room29_east_bomb_controller() is not (
        make_room29_east_bomb_controller()
    )


def test_factory_does_not_write_bombs() -> None:
    import zelda_i.level7.pre_boss as pre_boss

    source_path = pre_boss.__file__
    assert source_path is not None
    tree = ast.parse(open(source_path, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            imported.add(node.id)
    assert "apply_owned_inventory" not in imported
    assert "ADDR_BOMBS" not in imported
    assert "BombWallPhase" not in imported
