"""No-ROM gates for the 1-2 flag body and lift/pipe tail."""

from __future__ import annotations

from pathlib import Path

from smb.flag_12 import (
    Flag12Policy,
    FlagTailController,
    Phase,
    TailPhase,
    is_lift_pose,
    is_outdoor_flag_area,
    is_pipe_transition,
)
from smb.ram import PLAYER_STATE_AUTO_WALK, PLAYER_STATE_FLAGPOLE, SmbSnapshot
from smb.tas.stages import CONTROL_X_MAX, is_1_3_control

DEFAULT_MISSING = Path("/tmp/smb_1_2_flag_missing_for_tests.json")


def _snap(**kwargs: object) -> SmbSnapshot:
    player_x = int(kwargs.get("player_x", 40))
    timer = int(kwargs.get("timer", 400))
    return SmbSnapshot(
        frame=0,
        player_state=int(kwargs.get("player_state", 8)),
        player_x=player_x,
        player_y=int(kwargs.get("player_y", 176)),
        x_page=player_x // 256,
        x_offset=player_x % 256,
        lives=int(kwargs.get("lives", 2)),
        world=int(kwargs.get("world", 0)),
        level=int(kwargs.get("level", 2)),
        level_id=int(kwargs.get("world", 0)) * 4 + int(kwargs.get("level", 2)),
        oper_mode=int(kwargs.get("oper_mode", 1)),
        player_power=0,
        timer_hundreds=timer // 100,
        timer=timer,
        area_pointer=int(kwargs.get("area_pointer", 37)),
        x_speed=int(kwargs.get("x_speed", 0)),
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=bool(kwargs.get("in_air", False)),
        level_number=int(kwargs.get("level_number", kwargs.get("dash_level", 1))),
    )


def test_lift_pose_is_end_of_ug_platform() -> None:
    lift = _snap(player_x=2520, player_y=148, x_speed=40, dash_level=1, level_number=1)
    assert is_lift_pose(lift)
    assert not is_lift_pose(_snap(player_x=2520, player_y=148, x_speed=40, dash_level=2))
    assert not is_lift_pose(_snap(player_x=2856, player_y=176, x_speed=40, dash_level=1))


def test_outdoor_flag_area_is_world0_area_194() -> None:
    assert is_outdoor_flag_area(
        _snap(area_pointer=194, player_x=80, player_y=128, dash_level=1)
    )
    assert is_outdoor_flag_area(
        _snap(player_state=PLAYER_STATE_FLAGPOLE, dash_level=1, player_x=200)
    )
    assert not is_outdoor_flag_area(
        _snap(area_pointer=37, player_x=2646, player_y=128, dash_level=1)
    )
    assert not is_outdoor_flag_area(_snap(world=3, area_pointer=194, dash_level=0))


def test_pipe_walk_state_is_transition() -> None:
    assert is_pipe_transition(_snap(player_state=2, player_x=2646, player_y=128))
    assert is_pipe_transition(_snap(player_state=0, player_x=0, player_y=0))


def test_tail_waits_for_physics_ground_then_holds_a() -> None:
    tail = FlagTailController()
    lift = _snap(player_x=2520, player_y=148, x_speed=40, dash_level=1)
    idle = tail.step(lift, on_ground=False)
    assert tail.phase is TailPhase.JUMP
    assert int(idle.action.sum()) == 0
    a = tail.step(lift, on_ground=True)
    assert int(a.action[8]) == 1
    for _ in range(18):
        tail.step(lift, on_ground=False)
    assert tail.phase is TailPhase.COAST


def test_tail_walks_onto_lip_not_standing_down() -> None:
    tail = FlagTailController()
    tail.phase = TailPhase.WALK
    land = _snap(player_x=2620, player_y=128, x_speed=40, dash_level=1)
    walk = tail.step(land, on_ground=True)
    assert int(walk.action[7]) == 1  # RIGHT
    assert int(walk.action[5]) == 0  # not DOWN yet
    lip = _snap(player_x=2646, player_y=128, x_speed=8, dash_level=1)
    downish = tail.step(lip, on_ground=True)
    assert int(downish.action[5]) == 1  # DOWN on the lip
    pipe = _snap(player_state=2, player_x=2646, player_y=128, dash_level=1)
    entered = tail.step(pipe, on_ground=True)
    assert tail.phase is TailPhase.PIPE
    assert int(entered.action[5]) == 1


def test_tail_idles_on_flagpole_and_stops_at_1_3() -> None:
    tail = FlagTailController()
    tail.phase = TailPhase.OUTDOOR
    pole = _snap(
        player_state=PLAYER_STATE_AUTO_WALK,
        player_x=2000,
        dash_level=1,
        area_pointer=194,
    )
    idle = tail.step(pole, on_ground=True)
    assert int(idle.action.sum()) == 0
    done = tail.step(
        _snap(dash_level=2, level_number=2, player_x=40, timer=300, player_state=7),
        on_ground=True,
    )
    assert tail.phase is TailPhase.DONE
    assert int(done.action.sum()) == 0


def test_flag12_policy_waits_then_replays_then_1_3() -> None:
    frames = [[0, 0, 0, 0, 0, 0, 0, 1, 0]] * 4
    pol = Flag12Policy(frames=frames, seed_path=DEFAULT_MISSING)
    castle = _snap(level=0, level_number=0, player_x=3000, area_pointer=194, timer=200)
    a = pol.step(castle)
    assert pol.phase is Phase.WAIT_SURFACE
    assert int(a.action.sum()) == 0
    surf = _snap(
        level=1,
        level_number=1,
        player_x=40,
        player_y=176,
        area_pointer=41,
        timer=400,
    )
    a = pol.step(surf)
    assert pol.phase is Phase.BODY
    assert int(a.action[7]) == 1
    ctrl = _snap(dash_level=2, level_number=2, player_x=40, timer=300, player_state=8)
    assert is_1_3_control(ctrl)
    a = pol.step(ctrl)
    assert pol.phase is Phase.DONE
    assert pol.success


def test_1_3_control_rejects_ug_and_midstage() -> None:
    assert is_1_3_control(
        _snap(dash_level=2, level_number=2, player_x=40, timer=300)
    )
    assert not is_1_3_control(
        _snap(dash_level=1, level_number=1, player_x=40, timer=300)
    )
    assert not is_1_3_control(
        _snap(dash_level=2, level_number=2, player_x=CONTROL_X_MAX + 1, timer=300)
    )
