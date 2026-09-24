"""Unit tests for Level 4 leftover walks that would burn again."""

from __future__ import annotations

from types import SimpleNamespace

from retro_harness.controls import pressed_nes_buttons
from zelda_i.dungeon.ids import VIRE_OBJECT_TYPE
from zelda_i.level4.dungeon import (
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
from zelda_i.level4.maze_path import make_north_40_controller
from zelda_i.level4.stepladder import make_stepladder_controller


def test_stepladder_notch_stall_fails_without_bfs() -> None:
    """Dock-path stall fail-closes; do not fall through to BFS."""
    import numpy as np

    from zelda_i.level4.occupancy import ROOM_60_CLIP_BUDGET
    from zelda_i.level4.stepladder import StepladderPhase
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
    def press(x: int, y: int) -> list[str]:
        ctl = make_north_40_controller()
        ctl.path_index = 4
        snap = SimpleNamespace(
            level=4, screen=ROOM_L4_VIRES_50, mode=5,
            transitioning=False, link_x=x, link_y=y,
        )
        return pressed_nes_buttons(list(ctl.step(snap).action))

    assert press(128, 101) == ["UP"]  # below the corner
    assert press(128, 93) == ["LEFT"]  # at the corner
    assert press(120, 93) == ["UP"]  # on the door column


def test_room_31_west_alcove_clip_is_right_up() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4.maze_path import make_maze_31_inland_controller

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
    done = inland.step(snap(80, 173))
    assert done.reason == "done"
    assert inland.success
    assert list(done.action) == list(nes_idle_action())


def test_room_31_combat_does_not_chase_x120_water() -> None:
    from zelda_i.level4.dungeon import ROOM_31_SPEC
    from zelda_i.level4.occupancy import ROOM_31_CLEAR_XY, ROOM_31_SPAWN_XY

    blocked = set(ROOM_31_SPEC.combat.occupancy_blocked)
    assert ROOM_31_CLEAR_XY in blocked
    assert ROOM_31_SPAWN_XY not in blocked
    assert (200, 141) not in blocked
    assert ROOM_31_SPEC.combat.engage_distance <= 24
    assert (120, 141) not in ROOM_31_SPEC.combat.patrol


def test_room_31_leave_reshapes_to_floor() -> None:
    from retro_harness.nes import nes_idle_action
    from zelda_i.level4.maze_path import (
        Maze31LeavePhase,
        make_maze_31_leave_controller,
    )

    def snap(x: int, y: int):
        return SimpleNamespace(
            mode=5, level=4, screen=ROOM_L4_EAST_31, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    floor = make_maze_31_leave_controller()
    done = floor.step(snap(112, 141))
    assert floor.success
    assert done.reason == "done"
    assert list(done.action) == list(nes_idle_action())

    north_end = make_maze_31_leave_controller()
    act = north_end.step(snap(80, 109))
    assert not north_end.success
    assert north_end.phase is Maze31LeavePhase.PATH
    assert act.reason == "maze31_leave_DOWN"

    wall_pocket = make_maze_31_leave_controller()
    assert wall_pocket.step(snap(80, 101)).reason == "maze31_leave_DOWN"

    aisle = make_maze_31_leave_controller()
    assert aisle.step(snap(80, 141)).reason == "maze31_leave_DOWN"

    south_aisle = make_maze_31_leave_controller()
    assert south_aisle.step(snap(80, 173)).reason == "maze31_leave_RIGHT"

    lip = make_maze_31_leave_controller()
    assert lip.step(snap(103, 165)).reason == "maze31_leave_DOWN"

    corner = make_maze_31_leave_controller()
    assert corner.step(snap(128, 173)).reason == "maze31_leave_UP"

    island = make_maze_31_leave_controller()
    assert island.step(snap(128, 133)).reason == "maze31_leave_LEFT"

    join = make_maze_31_leave_controller()
    join.path_index = 4
    join._initialized = True
    assert join.step(snap(112, 133)).reason == "maze31_leave_DOWN"


def test_room_31_east_leftover_goes_up_not_through_water() -> None:
    from retro_harness.nes import nes_action, nes_idle_action
    from zelda_i.level4.maze_path import make_maze_31_east_controller

    def snap(x: int, y: int, *, screen: int = ROOM_L4_EAST_31):
        return SimpleNamespace(
            mode=5, level=4, screen=screen, transitioning=False,
            link_x=x, link_y=y, objects=(),
        )

    leftover = make_maze_31_east_controller()
    action = leftover.step(snap(112, 141))
    assert action.reason == "maze31_east_join_UP"
    assert list(action.action) == list(nes_action("UP"))

    from zelda_i.level4.maze_path import STALL_LIMIT, Maze31EastPhase

    water = make_maze_31_east_controller()
    for _ in range(STALL_LIMIT + 2):
        act = water.step(snap(120, 133))
        if act.reason != "maze31_east_join_UP":
            break
    assert water.phase is Maze31EastPhase.OCC
    assert not water.success
    assert water.phase is not Maze31EastPhase.FAILED
    assert act.reason.startswith("maze31_east_occ_") or act.reason == "maze31_east_stand"

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
    from zelda_i.level4.clear12 import (
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
    from zelda_i.level4.dungeon import PUSH_12_STAND
    from zelda_i.level4.gleeok13 import (
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


def test_level4_gleeok13_finishes_from_turn_node_without_replanning_flip() -> None:
    """The power-on 0x12 leave approaches the push stand at (112,141)."""
    from retro_harness.nes import nes_action
    from zelda_i.level4.gleeok13 import make_gleeok13_controller
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
    ram[ADDR_MODE], ram[ADDR_LEVEL], ram[ADDR_SCREEN] = PLAY_MODE, 4, 0x12
    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = 112, 141
    for y in (141, 142):
        ram[ADDR_LINK_Y] = y
        action = ctl.step(read_snapshot(ram))
        assert action.reason == "finish_push_stand"
        assert list(action.action) == list(nes_action("DOWN"))
    ram[ADDR_LINK_Y] = 143
    assert ctl.step(read_snapshot(ram)).reason == "push_block"


def test_level4_gleeok13_west_of_block_routes_around(monkeypatch) -> None:
    """West of the 0x68 block, the ROM lattice walks round it to the east push
    stand: never RIGHT into its west face (that pushes it the wrong way)."""
    from retro_harness.nes import nes_action
    from zelda_i.level4.dungeon import PUSH_12_STAND
    from zelda_i.level4.gleeok13 import make_gleeok13_controller
    from zelda_i.ram import (
        ADDR_LEVEL,
        ADDR_LINK_X,
        ADDR_LINK_Y,
        ADDR_MODE,
        ADDR_SCREEN,
        PLAY_MODE,
        read_snapshot,
    )
    from zelda_i.tests.ram_helpers import room_tile_env
    from zelda_i.walk import live_env

    env = room_tile_env("0x12", level=4)
    monkeypatch.setattr(live_env, "_ENV", env)
    ram = env.get_ram()
    ram[ADDR_MODE], ram[ADDR_LEVEL], ram[ADDR_SCREEN] = PLAY_MODE, 4, 0x12

    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = 80, 144  # against the block's west face
    act = make_gleeok13_controller().step(read_snapshot(ram))
    assert act.reason == "stand_lattice"
    assert list(act.action) != list(nes_action("RIGHT"))

    # At PUSH_12_STAND (112, 144): pushes LEFT
    ram[ADDR_LINK_X], ram[ADDR_LINK_Y] = PUSH_12_STAND
    act = make_gleeok13_controller().step(read_snapshot(ram))
    assert act.reason == "push_block"
    assert list(act.action) == list(nes_action("LEFT"))


def test_l4_bomb_walls_pause_select_leftover_slot() -> None:
    """L3 leftover is poke-assisted bombs; composition still pause-selects."""
    from types import SimpleNamespace

    import numpy as np

    from zelda_i.dungeon.pause_select import B_SLOT_BOMBS, B_SLOT_RECORDER
    from zelda_i.level4.bomb11 import make_bomb_21_north_controller
    from zelda_i.level4.clear12 import make_bomb_11_east_controller
    from zelda_i.level4.key01 import make_bomb_11_north_controller
    from zelda_i.level4.path import make_bomb_61_north_controller
    from zelda_i.ram import PLAY_MODE, read_snapshot
    from zelda_i.tests.ram_helpers import make_ram

    factories = (
        make_bomb_61_north_controller,
        make_bomb_21_north_controller,
        make_bomb_11_east_controller,
        make_bomb_11_north_controller,
    )
    for factory in factories:
        ctl = factory()
        assert ctl.select_item == B_SLOT_BOMBS, factory.__name__
        sx, sy = ctl.stand
        ram = make_ram(
            {},
            mode=PLAY_MODE,
            level=4,
            screen=ctl.from_room,
            x=sx,
            y=sy,
            bombs=8,
            selected=B_SLOT_RECORDER,
        )
        ctl.bind_env(SimpleNamespace(get_ram=lambda ram=ram: ram))
        reasons: list[str] = []
        for _ in range(80):
            before = ram.copy()
            act = ctl.step(read_snapshot(ram))
            assert np.array_equal(ram, before), "bomb wall must not write RAM"
            reasons.append(act.reason)
            if act.reason in {"place_bomb", "pause_open"}:
                break
        assert "pause_open" in reasons, (factory.__name__, reasons[-8:])
        assert "place_bomb" not in reasons, factory.__name__


def test_room_12_vires_combat_policy() -> None:
    """Room 0x12 Vires policy: avoid_walls, contact_backstep, west off-wall RIGHT."""
    from retro_harness.nes import nes_action
    from zelda_i.level4.dungeon import ROOM_12_SPEC
    from zelda_i.level4.path import (
        Room12ViresController,
        make_room_12_clear_controller,
    )

    ctl = make_room_12_clear_controller()
    assert isinstance(ctl, Room12ViresController)
    assert ROOM_12_SPEC.combat.avoid_walls is True
    assert ROOM_12_SPEC.combat.contact_backstep >= 16
    assert ROOM_12_SPEC.combat.engage_dominant_axis is True
    assert ROOM_12_SPEC.entry.direction == "RIGHT"

    # West door mouth (x < 56, y=141): Link steps RIGHT into open floor, not DOWN into wall.
    snap = SimpleNamespace(
        link_x=16,
        link_y=141,
        objects=(),
        mode=5,
        level=4,
        screen=0x12,
        transitioning=False,
    )
    off_wall = ctl._off_wall_step(snap)
    assert off_wall is not None
    assert off_wall.reason == "leave_wall_slash"
    assert list(off_wall.action) == list(nes_action("RIGHT", "A"))

    snap_32 = SimpleNamespace(
        link_x=32,
        link_y=141,
        objects=(),
        mode=5,
        level=4,
        screen=0x12,
        transitioning=False,
    )
    off_wall_32 = ctl._off_wall_step(snap_32)
    assert off_wall_32 is not None
    assert list(off_wall_32.action) == list(nes_action("RIGHT", "A"))
