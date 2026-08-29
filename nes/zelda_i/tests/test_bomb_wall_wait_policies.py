"""Behavioral unit tests for BombWallController wait policies (rr-knb1).

Synthetic snap sequences exercise FACE → PLACE → WAIT without the emulator.
Covers STEP_BACK_REQUIRE_CONSUME (6f/5f) vs HOLD_FACE (4f/1e) + south_band.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.bomb_wall import (
    BOMB_N_STEP_BACK,
    BombWallController,
    BombWallPhase,
)
from zelda_i.level2.bomb_path import (
    make_bomb_north_1e_controller,
    make_bomb_north_controller,
    make_post_boom_bomb_north_controller,
)
from zelda_i.level2.puzzles import BOMB_WALL_6F_NORTH
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
)


def _snap(
    *,
    room: int = 0x6F,
    x: int = 120,
    y: int = 101,
    bombs: int = 4,
    level: int = 2,
    mode: int = PLAY_MODE,
) -> ZeldaSnapshot:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_BOMBS] = bombs
    return read_snapshot(ram)


def test_6f_factory_is_step_back_require_consume() -> None:
    ctrl = make_bomb_north_controller()
    assert ctrl.wait_hold_face is False
    assert ctrl.require_bomb_consumed is True
    assert ctrl.step_back == BOMB_N_STEP_BACK
    assert ctrl.face_frames == 4
    assert ctrl.south_band_first is False


def test_post_boom_and_1e_are_hold_face() -> None:
    post = make_post_boom_bomb_north_controller()
    one_e = make_bomb_north_1e_controller()
    assert post.wait_hold_face is True
    assert post.require_bomb_consumed is False
    assert post.step_back == 0
    assert post.face_frames == 6
    assert one_e.wait_hold_face is True
    assert one_e.south_band_first is True
    assert one_e.step_back == 0


def _drive_to_wait(
    ctrl: BombWallController, *, bombs: int = 4
) -> list[str]:
    """From FACE at stand → through PLACE into WAIT. Returns action reasons."""
    sx, sy = ctrl.stand
    room = ctrl.from_room
    reasons: list[str] = []
    # FACE: first face_frames-1 steps return face_*; then same-frame fall-through place.
    for _ in range(ctrl.face_frames + 1):
        act = ctrl.step(_snap(room=room, x=sx, y=sy, bombs=bombs))
        reasons.append(act.reason)
        if act.reason == "place_bomb":
            break
    assert "place_bomb" in reasons
    assert ctrl.phase is BombWallPhase.WAIT
    assert ctrl.bombs_before_place == bombs
    return reasons


def test_step_back_wait_requires_bomb_consume() -> None:
    """6f-style: PLACE then step-back frames, idle wait; fail if bombs never drop."""
    wall = BOMB_WALL_6F_NORTH
    ctrl = BombWallController(
        wall=wall,
        level=2,
        face_frames=2,
        step_back=3,
        wait_blast=8,
        require_bomb_consumed=True,
        wait_hold_face=False,
    )
    ctrl.phase = BombWallPhase.FACE
    ctrl.phase_frames = 0
    sx, sy = wall.stand
    _drive_to_wait(ctrl, bombs=4)

    # step_back=N means phase_frames in 1..N-1 return step_back (strict < N).
    reasons: list[str] = []
    for _ in range(10):
        act = ctrl.step(_snap(room=wall.room, x=sx, y=sy + 4, bombs=4))
        reasons.append(act.reason)
        if ctrl.phase is BombWallPhase.FAILED:
            break
    assert reasons.count("step_back") == 2  # phase_frames 1,2 with step_back=3
    assert "wait_blast" in reasons
    assert ctrl.phase is BombWallPhase.FAILED
    assert reasons[-1] == "bomb_not_consumed"


def test_step_back_wait_advances_when_bomb_consumed() -> None:
    wall = BOMB_WALL_6F_NORTH
    ctrl = BombWallController(
        wall=wall,
        level=2,
        face_frames=1,
        step_back=2,
        wait_blast=5,
        require_bomb_consumed=True,
        wait_hold_face=False,
    )
    ctrl.phase = BombWallPhase.FACE
    ctrl.phase_frames = 0
    sx, sy = wall.stand
    _drive_to_wait(ctrl, bombs=3)

    # Step-back while bomb count drops (phase_frames resets on PLACE→WAIT)
    ctrl.step(_snap(room=wall.room, x=sx, y=sy + 2, bombs=2))
    ctrl.step(_snap(room=wall.room, x=sx, y=sy + 4, bombs=2))
    # Idle remaining wait_blast
    for _ in range(5):
        act = ctrl.step(_snap(room=wall.room, x=sx, y=sy + 4, bombs=2))
    assert ctrl.phase is BombWallPhase.PUSH
    assert ctrl.bombs_after_place == 2
    assert any("bomb_used_3->2" in n for n in ctrl.notes)


def test_hold_face_wait_never_steps_back() -> None:
    """4f/1e style: WAIT holds face; no step_back action reasons."""
    wall = BOMB_WALL_6F_NORTH  # reuse geometry; policy is what matters
    ctrl = BombWallController(
        wall=wall,
        level=2,
        face_frames=2,
        step_back=0,
        wait_blast=6,
        require_bomb_consumed=False,
        wait_hold_face=True,
    )
    ctrl.phase = BombWallPhase.FACE
    ctrl.phase_frames = 0
    sx, sy = wall.stand
    _drive_to_wait(ctrl, bombs=5)

    wait_reasons: list[str] = []
    for _ in range(6):
        act = ctrl.step(_snap(room=wall.room, x=sx, y=sy, bombs=5))
        wait_reasons.append(act.reason)
    assert all(r == "wait_blast" or r.startswith("push_") for r in wait_reasons)
    assert "step_back" not in wait_reasons
    assert ctrl.phase is BombWallPhase.PUSH


def test_1e_south_band_before_stand() -> None:
    ctrl = make_bomb_north_1e_controller()
    assert ctrl.south_band_first is True
    assert ctrl.approach_waypoints[0] == (96, 189)
    wall = ctrl.wall
    act = ctrl.step(
        _snap(room=wall.room, x=96, y=141, bombs=4, level=2)
    )
    assert ctrl.phase is BombWallPhase.SOUTH_BAND
    assert act.reason == "approach_y"


def test_1e_south_band_centers_x_before_stand() -> None:
    """0x1e live: south y=189 then east column 176, not mid-y laterals."""
    ctrl = make_bomb_north_1e_controller()
    wall = ctrl.wall
    act = ctrl.step(_snap(room=wall.room, x=96, y=141, bombs=4, level=2))
    assert act.reason == "approach_y"
    act = ctrl.step(_snap(room=wall.room, x=96, y=189, bombs=4, level=2))
    assert act.reason == "approach_next"
    act = ctrl.step(_snap(room=wall.room, x=96, y=189, bombs=4, level=2))
    assert act.reason == "approach_x"
