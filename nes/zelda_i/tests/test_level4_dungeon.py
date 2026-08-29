"""Unit tests for Level 4 leftover walks that would burn again."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i.dungeon_ids import VIRE_OBJECT_TYPE
from zelda_i.level4_dungeon import (
    GEL_OBJECT_TYPE,
    INVULN_MOVER_TYPE,
    LIKE_LIKE_OBJECT_TYPE,
    ROOM_21_SPEC,
    ROOM_30_SPEC,
    ROOM_32_SPEC,
    ROOM_L4_EAST_31,
    ROOM_L4_EAST_32,
    ROOM_L4_VIRES_50,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level4_maze_path import make_north_40_controller
from zelda_i.level4_stepladder import make_stepladder_controller


def test_stepladder_notch_stall_fails_without_bfs() -> None:
    """Dock-path stall fail-closes; do not fall through to BFS."""
    import numpy as np

    from zelda_i.level4_occupancy import ROOM_60_CLIP_BUDGET
    from zelda_i.level4_stepladder import StepladderPhase
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        read_snapshot,
    )

    ctl = make_stepladder_controller(clear_first=False)
    ctl.phase = StepladderPhase.PATH
    ctl._last_xy = (48, 189)
    ctl._stall = ROOM_60_CLIP_BUDGET
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = 9
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x60
    ram[ADDR_LINK_X] = 48
    ram[ADDR_LINK_Y] = 189
    act = ctl.step(read_snapshot(ram))
    assert ctl.phase is StepladderPhase.FAILED
    assert act.reason.startswith("dock_solid_48_189")


def test_maze_50_live_corner_turns_at_observed_coordinates() -> None:
    def snap(x: int, y: int) -> SimpleNamespace:
        return SimpleNamespace(
            level=4, screen=ROOM_L4_VIRES_50, mode=5,
            transitioning=False, link_x=x, link_y=y,
        )

    below_corner = make_north_40_controller()
    below_corner.path_index = 4
    assert below_corner.step(snap(128, 101)).reason == "maze50_seek_4_UP"

    at_corner = make_north_40_controller()
    at_corner.path_index = 4
    assert at_corner.step(snap(128, 93)).reason == "maze50_seek_4_LEFT"

    door_column = make_north_40_controller()
    door_column.path_index = 4
    assert door_column.step(snap(120, 93)).reason == "maze50_seek_4_UP"


def test_room_31_west_alcove_clip_is_right_up() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4_maze_path import make_maze_31_inland_controller

    def snap(x: int, y: int, *, screen: int = ROOM_L4_EAST_31):
        return SimpleNamespace(
            mode=5, level=4, screen=screen, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    alcove = make_maze_31_inland_controller()
    action = alcove.step(snap(32, 141))
    assert action.reason == "maze31_alcove_clip"
    assert list(action.action) == list(nes_action("RIGHT", "UP"))

    corridor = make_maze_31_inland_controller()
    first = corridor.step(snap(48, 133))
    assert first.reason == "maze31_thread_UP"
    assert not corridor.success
    north = make_maze_31_inland_controller()
    assert north.step(snap(48, 109)).reason == "maze31_thread_RIGHT"
    corner = make_maze_31_inland_controller()
    assert corner.step(snap(80, 109)).reason == "maze31_thread_DOWN"
    drop = make_maze_31_inland_controller()
    assert drop.step(snap(80, 141)).reason == "maze31_thread_DOWN"

    inland = make_maze_31_inland_controller()
    done = inland.step(snap(128, 133))
    assert done.reason == "done"
    assert inland.success
    assert list(done.action) == list(nes_idle_action())


def test_room_31_east_leftover_goes_up_not_through_water() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4_maze_path import make_maze_31_east_controller

    def snap(x: int, y: int, *, screen: int = ROOM_L4_EAST_31):
        return SimpleNamespace(
            mode=5, level=4, screen=screen, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    leftover = make_maze_31_east_controller()
    action = leftover.step(snap(112, 141))
    assert action.reason == "maze31_east_join_UP"
    assert list(action.action) == list(nes_action("UP"))

    clip = make_maze_31_east_controller()
    clipped = clip.step(snap(112, 113))
    assert clipped.reason == "maze31_east_se_clip"
    assert list(clipped.action) == list(nes_action("RIGHT", "DOWN"))

    band = make_maze_31_east_controller()
    assert band.step(snap(200, 136)).reason == "maze31_east_push"
    assert list(band.step(snap(200, 136)).action) == list(nes_action("RIGHT"))

    entered = make_maze_31_east_controller()
    done = entered.step(snap(16, 141, screen=ROOM_L4_EAST_32))
    assert done.reason == "done"
    assert entered.success
    assert list(done.action) == list(nes_idle_action())


def test_live_enemies_ignore_invuln_0x2b() -> None:
    invuln = SimpleNamespace(slot=1, type_id=INVULN_MOVER_TYPE, x=80, y=133, hp=64)
    block = SimpleNamespace(slot=2, type_id=0x68, x=80, y=144, hp=0)
    vire = SimpleNamespace(slot=3, type_id=VIRE_OBJECT_TYPE, x=160, y=100, hp=64)
    gel = SimpleNamespace(slot=3, type_id=GEL_OBJECT_TYPE, x=160, y=100, hp=0)
    zol = SimpleNamespace(slot=3, type_id=ZOL_OBJECT_TYPE, x=160, y=100, hp=64)
    like = SimpleNamespace(slot=4, type_id=LIKE_LIKE_OBJECT_TYPE, x=100, y=141, hp=64)
    empty = SimpleNamespace(objects=(invuln, block))
    assert ROOM_30_SPEC.live_enemies(SimpleNamespace(objects=(invuln,))) == ()
    assert [o.type_id for o in ROOM_30_SPEC.live_enemies(
        SimpleNamespace(objects=(invuln, vire))
    )] == [VIRE_OBJECT_TYPE]
    assert ROOM_21_SPEC.live_enemies(empty) == ()
    assert [o.type_id for o in ROOM_21_SPEC.live_enemies(
        SimpleNamespace(objects=(invuln, block, gel))
    )] == [GEL_OBJECT_TYPE]
    assert ROOM_32_SPEC.live_enemies(empty) == ()
    assert {o.type_id for o in ROOM_32_SPEC.live_enemies(
        SimpleNamespace(objects=(invuln, block, zol, like))
    )} == {ZOL_OBJECT_TYPE, LIKE_LIKE_OBJECT_TYPE}


def test_level4_clear12_attaches_after_key01() -> None:
    """key01 leftover (120,133) walks DOWN; bomb-east stand opens 0x12 not 0x11."""
    from retro_harness.nes import nes_action
    from zelda_i.level4_clear12 import (
        BOMB_11_EAST_STAND,
        BombWall11East,
        level4_clear12_success,
        make_south_11_controller,
    )
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        PLAY_MODE,
        read_snapshot,
    )
    import numpy as np

    wall = BombWall11East()
    assert wall.stand == BOMB_11_EAST_STAND == (192, 141)
    assert wall.opens_to == 0x12
    ctl = make_south_11_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x01
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 133
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("DOWN"))
    ram[ADDR_SCREEN] = 0x11
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert not level4_clear12_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x12
    assert level4_clear12_success(read_snapshot(ram))


def test_level4_gleeok13_attaches_after_clear12() -> None:
    """clear12 leftover (128,117) walks to PUSH_12_STAND; success is play 0x13."""
    from retro_harness.nes import nes_action
    from zelda_i.level4_dungeon import PUSH_12_STAND
    from zelda_i.level4_gleeok13 import (
        level4_gleeok13_success,
        make_gleeok13_controller,
    )
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        PLAY_MODE,
        read_snapshot,
    )
    import numpy as np

    ctl = make_gleeok13_controller()
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 4
    ram[ADDR_SCREEN] = 0x12
    ram[ADDR_LINK_X] = 128
    ram[ADDR_LINK_Y] = 117
    act = ctl.step(read_snapshot(ram))
    assert list(act.action) == list(nes_action("LEFT"))
    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = PUSH_12_STAND
    act = ctl.step(read_snapshot(ram))
    assert act.reason == "push_block"
    ram[ADDR_SCREEN] = 0x13
    ctl.step(read_snapshot(ram))
    assert ctl.success
    assert level4_gleeok13_success(read_snapshot(ram))
    ram[ADDR_SCREEN] = 0x12
    assert not level4_gleeok13_success(read_snapshot(ram))
