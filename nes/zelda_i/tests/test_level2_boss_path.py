"""Pure unit tests for L2 Dodongo boss path library (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

from zelda_i import level2_boss_path as bp
from zelda_i import level2_puzzles as puz
from zelda_i.level2_boss_path import (
    BombNorth1EPhase,
    BossPathStart,
    Level2BombNorth1EController,
    Level2PostBossTfController,
    PostBossTfPhase,
)


def test_room_ids_and_path_constants() -> None:
    assert bp.ROOM_3F == 0x3F
    assert bp.ROOM_3E == 0x3E
    assert bp.ROOM_2E == 0x2E
    assert bp.ROOM_1E == 0x1E
    assert bp.ROOM_0E == puz.ROOM_L2_BOSS == 0x0E
    assert bp.ROOM_TF == puz.ROOM_L2_TF == 0x0D
    assert bp.LEVEL2_TF_BIT == puz.LEVEL2_TRIFORCE_BIT == 0x02
    assert bp.DODONGO_TYPE == 0x32
    assert bp.BOOM_TO_BOSS_ROOMS == (0x3F, 0x3E, 0x2E, 0x1E, 0x0E)
    assert bp.BOSS_PATH_ROOMS[-1] == 0x0D
    assert bp.is_boss_path_room(0x4F)
    assert bp.is_boss_path_room(0x0E)
    assert bp.is_boss_path_room(0x0D)
    assert not bp.is_boss_path_room(0x7D)


def test_bomb_wall_1e_reuses_catalog() -> None:
    assert bp.BOMB_WALL_1E is puz.BOMB_WALL_1E_NORTH
    assert bp.BOMB_STAND_1E == (120, 101)
    assert bp.BOMB_WALL_1E.face == "UP"
    assert bp.BOMB_WALL_1E.opens_to == 0x0E
    assert bp.bomb_1e_open_predicate(from_room=0x1E, to_room=0x0E)
    assert not bp.bomb_1e_open_predicate(from_room=0x1E, to_room=0x0D)
    assert puz.bomb_wall_for_room(0x1E, "UP") is puz.BOMB_WALL_1E_NORTH
    assert puz.is_at_bomb_stand(120, 101, bp.BOMB_WALL_1E)
    assert not puz.is_at_bomb_stand(120, 141, bp.BOMB_WALL_1E, tol=6)


def test_triforce_and_tf_policy_integration() -> None:
    assert bp.triforce_bit_02(0x02)
    assert bp.triforce_bit_02(0x03)
    assert not bp.triforce_bit_02(0x01)
    assert not bp.triforce_bit_02(0x00)
    wps = bp.default_tf_waypoints()
    assert wps == puz.L2_TF_COLLECT_WAYPOINTS
    assert wps == (
        (208, 141),
        (208, 189),
        (128, 189),
        (128, 149),
    )
    pol = bp.load_tf_policy()
    assert pol["live"] is True
    assert bp.policy_waypoints(pol) == list(wps)
    stand, face = bp.policy_push(pol)
    assert stand is None  # live path: push not required
    assert face is None or face is puz.L2_TF_PUSH_DIR


def test_bomb_north_1e_controller_default_phase() -> None:
    ctrl = Level2BombNorth1EController()
    assert ctrl.phase is BombNorth1EPhase.SETTLE
    assert ctrl.success is False
    assert ctrl.stand == (120, 101)
    assert ctrl.wall is puz.BOMB_WALL_1E_NORTH
    report = ctrl.report()
    assert report["phase"] == "SETTLE"
    assert report["success"] is False
    assert report["stand"] == [120, 101]
    assert report["opens_to"] == "0x0e"
    factory = bp.make_bomb_north_1e_controller()
    assert factory.phase is BombNorth1EPhase.SETTLE


def test_post_boss_tf_controller_default_phase() -> None:
    ctrl = Level2PostBossTfController()
    assert ctrl.phase is PostBossTfPhase.HEART
    assert ctrl.success is False
    assert ctrl.waypoints == list(puz.L2_TF_COLLECT_WAYPOINTS)
    assert ctrl.policy_live is True
    assert ctrl.push_done is True  # no push stand on live policy
    report = ctrl.report()
    assert report["phase"] == "HEART"
    assert report["waypoint_index"] == 0
    assert len(report["waypoints"]) == 4
    factory = bp.make_post_boss_tf_controller()
    assert factory.phase is PostBossTfPhase.HEART


def test_mouth_target_facing() -> None:
    d_e = SimpleNamespace(x=100, y=140, facing=bp.FACE_E)
    assert bp.mouth_target(d_e) == (112, 140, "LEFT")
    d_w = SimpleNamespace(x=100, y=140, facing=bp.FACE_W)
    assert bp.mouth_target(d_w) == (88, 140, "RIGHT")
    d_s = SimpleNamespace(x=100, y=140, facing=bp.FACE_S)
    assert bp.mouth_target(d_s) == (100, 152, "UP")
    d_n = SimpleNamespace(x=100, y=140, facing=bp.FACE_N)
    assert bp.mouth_target(d_n) == (100, 128, "DOWN")


def test_resolve_path_start() -> None:
    assert bp.resolve_path_start("Level2Boom", 0x4F) is BossPathStart.BOOM
    assert bp.resolve_path_start("Level2_0E", 0x0E) is BossPathStart.BOSS
    assert bp.resolve_path_start("Level2_0D_PostBoss", 0x0D) is BossPathStart.TF_ROOM
    assert bp.resolve_path_start("Level2Boom", 0x0E) is BossPathStart.BOSS
    assert bp.resolve_path_start("any", 0x0D) is BossPathStart.TF_ROOM


def test_goto_action_axes() -> None:
    snap = SimpleNamespace(link_x=100, link_y=140)
    act, at = bp.goto_action(snap, 120, 140, tol=6)  # type: ignore[arg-type]
    assert at is False
    snap2 = SimpleNamespace(link_x=120, link_y=140)
    act2, at2 = bp.goto_action(snap2, 120, 140, tol=6)  # type: ignore[arg-type]
    assert at2 is True
    del act, act2
