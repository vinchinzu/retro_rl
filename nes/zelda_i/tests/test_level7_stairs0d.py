"""L7 play 0x0D walk-on of cellar 0x7B (`level7.stairs0d`, live 2/2).

Live evidence (rr-8t4.3, 2026-09-04): `20260904_S1` / `20260904_S2` from
`Level7Interior0DClearedReconFixture`, 458 controller frames each, dest RAM
mode 9 screen 0x7B, `position_writes=0`.
"""

from __future__ import annotations

import ast

import numpy as np

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level7.hops import make_tip_stairs_controller
from zelda_i.level7.stairs0d import (
    DOOR_ROW_Y,
    EAST_COLUMN_X,
    PHASE_MAX_FRAMES,
    RAM_CLAIM,
    ROOM,
    ROW_Y,
    STAIR_CELL,
    STALL_FRAMES,
    TIP_BLOCK_CELL_AFTER_RIGHT,
    TIP_BLOCK_XY,
    WEST_COLUMN_X,
    WEST_DOOR_GUARD_X,
    Level7Stairs0DController,
    Stairs0DPhase,
    level7_stairs0d_stages,
    level7_stairs0d_success,
    make_stairs0d_controller,
    tip_block,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PASSAGE_MODE,
    PLAY_MODE,
    read_snapshot,
)

_BLOCK_SLOT = 11


def _ram(
    *,
    x: int,
    y: int,
    screen: int = ROOM,
    mode: int = PLAY_MODE,
    block: tuple[int, int] | None = TIP_BLOCK_XY,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = 7
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    if block is not None:
        ram[ADDR_OBJ_TYPE + _BLOCK_SLOT] = 0x68
        ram[ADDR_LINK_X + _BLOCK_SLOT] = block[0]
        ram[ADDR_LINK_Y + _BLOCK_SLOT] = block[1]
    return ram


def _snap(**fields):
    return read_snapshot(_ram(**fields))


def _buttons(action) -> list[str]:
    from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX

    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def test_geometry_constants_match_the_measured_map() -> None:
    assert ROOM == 0x0D
    assert TIP_BLOCK_XY == (192, 144)
    assert TIP_BLOCK_CELL_AFTER_RIGHT == (208, 144)
    assert STAIR_CELL == (208, 96)
    # The two crossings of the y=112 / y=176 solid bands.
    assert (WEST_COLUMN_X, EAST_COLUMN_X) == (32, 208)
    assert DOOR_ROW_Y == ROW_Y[144] == 141
    assert ROW_Y[96] == 93 and ROW_Y[128] == 125


def test_ram_claim_names_the_miss_conditions() -> None:
    for phrase in ("Miss if", "0x7B", "(208,93)", "0x79"):
        assert phrase in RAM_CLAIM


def test_success_is_mode9_cellar_not_a_play_room() -> None:
    assert level7_stairs0d_success(
        _snap(x=208, y=93, screen=0x7B, mode=PASSAGE_MODE)
    )
    # Back in a play room is never success, including the source room.
    assert not level7_stairs0d_success(_snap(x=208, y=93))
    assert not level7_stairs0d_success(
        _snap(x=32, y=141, screen=0x79, mode=PLAY_MODE)
    )
    assert not level7_stairs0d_success(
        _snap(x=208, y=93, screen=0x4A, mode=PASSAGE_MODE)
    )


def test_tip_block_picks_the_0x68_nearest_the_known_cell() -> None:
    snap = _snap(x=63, y=149)
    block = tip_block(snap)
    assert block is not None
    assert (int(block.x), int(block.y)) == TIP_BLOCK_XY
    assert tip_block(_snap(x=63, y=149, block=None)) is None


def test_push_phases_are_single_axis_in_order() -> None:
    """Each leg presses one cardinal; mixing axes oscillates (20260904_S1)."""
    ctl = make_stairs0d_controller()
    assert ctl.phase is Stairs0DPhase.PUSH_UP
    # From the pin, rise to the block's row first.
    assert _buttons(ctl.step(_snap(x=63, y=149))) == ["UP"]
    assert ctl.phase is Stairs0DPhase.PUSH_UP
    # On the row, travel east with no y correction.
    act = ctl.step(_snap(x=63, y=141))
    assert _buttons(act) == ["RIGHT"]
    assert ctl.phase is Stairs0DPhase.PUSH_EAST
    # At the west face, align down onto the block's y, then hold RIGHT.
    act = ctl.step(_snap(x=176, y=141))
    assert _buttons(act) == ["DOWN"]
    assert ctl.phase is Stairs0DPhase.PUSH_ALIGN
    act = ctl.step(_snap(x=176, y=144))
    assert _buttons(act) == ["RIGHT"]
    assert ctl.phase is Stairs0DPhase.PUSH_HOLD
    assert act.reason == "push_block"


def test_ring_legs_follow_the_measured_corridors() -> None:
    ctl = make_stairs0d_controller()
    ctl.block_x0, ctl.block_y0 = TIP_BLOCK_XY
    ctl.phase = Stairs0DPhase.PUSH_HOLD
    # The pushed block's RAM x/y is repointed to the revealed stairs.
    moved = _snap(x=177, y=141, block=STAIR_CELL)
    assert _buttons(ctl.step(moved)) == ["LEFT"]
    assert ctl.phase is Stairs0DPhase.PEEL_WEST
    # Peel west, then climb to the y=128 row before crossing the room.
    act = ctl.step(_snap(x=160, y=141, block=STAIR_CELL))
    assert _buttons(act) == ["UP"]
    assert ctl.phase is Stairs0DPhase.NORTH_ROW
    act = ctl.step(_snap(x=160, y=125, block=STAIR_CELL))
    assert _buttons(act) == ["LEFT"]
    assert ctl.phase is Stairs0DPhase.WEST_COLUMN
    # West column reached (live legs settle at y=117, same y=128 cell row).
    act = ctl.step(_snap(x=32, y=117, block=STAIR_CELL))
    assert _buttons(act) == ["UP"]
    assert ctl.phase is Stairs0DPhase.NORTH_COLUMN
    act = ctl.step(_snap(x=32, y=93, block=STAIR_CELL))
    assert _buttons(act) == ["RIGHT"]
    assert ctl.phase is Stairs0DPhase.EAST_TOP
    assert act.reason == "east_top_row"


def test_arriving_in_cellar_0x7b_marks_done() -> None:
    ctl = make_stairs0d_controller()
    ctl.phase = Stairs0DPhase.EAST_TOP
    act = ctl.step(_snap(x=208, y=93, screen=0x7B, mode=PASSAGE_MODE))
    assert ctl.success and not ctl.failed
    assert act.reason == "cellar_0x7b"
    assert ctl.phase is Stairs0DPhase.DONE
    assert list(act.action) == list(nes_idle_action())


def test_left_on_the_west_door_row_fails_instead_of_exiting() -> None:
    """LEFT at x<=32 on y=141 walks into the (16,144) door and leaves for 0x79."""
    ctl = make_stairs0d_controller()
    ctl.block_x0, ctl.block_y0 = TIP_BLOCK_XY
    ctl.phase = Stairs0DPhase.WEST_COLUMN
    act = ctl.step(_snap(x=WEST_DOOR_GUARD_X, y=DOOR_ROW_Y, block=STAIR_CELL))
    assert ctl.failed
    assert any(n.startswith("west_door_row_left") for n in ctl.notes)
    assert list(act.action) != list(nes_action("LEFT"))
    # The live route crosses west one row higher, so the guard stays quiet.
    ok = make_stairs0d_controller()
    ok.block_x0, ok.block_y0 = TIP_BLOCK_XY
    ok.phase = Stairs0DPhase.WEST_COLUMN
    act = ok.step(_snap(x=WEST_DOOR_GUARD_X, y=ROW_Y[128], block=STAIR_CELL))
    assert not ok.failed
    assert _buttons(act) == ["LEFT"]


def test_leaving_0x0d_for_a_play_room_fails() -> None:
    ctl = make_stairs0d_controller()
    ctl.step(_snap(x=63, y=149, screen=0x79))
    assert ctl.failed
    assert any(note.startswith("left_0x0d_to_0x79") for note in ctl.notes)


def test_a_stalled_leg_fails_rather_than_burning_the_budget() -> None:
    ctl = make_stairs0d_controller()
    ctl.block_x0, ctl.block_y0 = TIP_BLOCK_XY
    ctl.phase = Stairs0DPhase.PUSH_EAST
    for _ in range(STALL_FRAMES + 2):
        if ctl.failed:
            break
        ctl.step(_snap(x=100, y=141))
    assert ctl.failed
    assert any("push_east_stall" in note for note in ctl.notes)
    assert ctl.frames <= PHASE_MAX_FRAMES


def test_report_is_fixture_live_and_write_free() -> None:
    report = make_stairs0d_controller().report()
    assert report["spec_id"] == "level7_tip_of_nose_stairs"
    assert report["dest_screen"] == 0x7B
    assert report["room"] == 0x0D
    assert report["door"] == "STAIRS"
    assert report["evidence"] == "fixture-live"
    assert report["route_eligible"] is False
    assert report["natural_entry"] is False
    assert report["writes"] == 0
    assert report["position_assist"] == {
        "position_writes": 0,
        "progression_writes": 0,
    }
    assert report["stair_cell"] == [208, 96]
    assert report["block_after_right"] == [208, 144]


def test_stage_and_spine_factory_use_the_live_controller() -> None:
    stages = level7_stairs0d_stages()
    assert [name for name, _c, _f in stages] == ["level7_tip_of_nose_stairs"]
    assert isinstance(stages[0][1], Level7Stairs0DController)
    assert stages[0][2] == stages[0][1].max_frames
    assert isinstance(make_tip_stairs_controller(), Level7Stairs0DController)


def test_no_occupancy_walker_in_room_0x0d() -> None:
    """OccupancyWalker is banned in 0x0D (standing order)."""
    import zelda_i.level7.stairs0d as mod

    assert not hasattr(mod, "OccupancyWalker")
    source = mod.__file__
    assert source is not None
    tree = ast.parse(open(source, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "OccupancyWalker" not in imported
    assert "zelda_i.walk.physics" not in imported
