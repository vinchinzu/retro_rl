"""Unit tests for stroke-play mission recovery and sequencing."""

from __future__ import annotations

import numpy as np

from hals_golf.core.ram import (
    WRAM_AIM_OFFSET,
    WRAM_HOLE_INDEX,
    WRAM_LIE_TYPE,
    WRAM_OPPONENT_STROKE_COUNT,
    WRAM_REST_DISTANCE,
    WRAM_STROKE_COUNT,
    read_aim_offset,
)
from hals_golf.core.recovery import RecoveryController
from hals_golf.core.scene import is_command_screen
from hals_golf.tasks.menus import (
    ClubSet,
    Difficulty,
    MenuBootstrapTask,
    PlayMode,
    _metal_play_name_nav,
    title_to_stroke_play_amateur_frames,
    title_to_stroke_play_frames,
    title_to_vs_hal_amateur_frames,
    title_to_vs_hal_frames,
)
from hals_golf.tasks.scorecard import ScorecardBook
from hals_golf.tasks.mission import (
    COMMAND_STALL_FRAMES,
    FUTILE_SHOT_LIMIT,
    OPPONENT_WAIT_LIMIT,
    MissionPhase,
    StrokePlayMission,
)
from hals_golf.tasks.shot import PuttTask, ShotPhase, ShotTask
from retro_harness.protocol import TaskStatus, WorldState


def _command_obs() -> np.ndarray:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    panel = obs[160:205, 200:256]
    panel[:20, :] = (0, 60, 200)
    panel[20:30, :] = 255
    panel[30:, :] = 0
    return obs


def _world(
    *,
    hole: int = 1,
    strokes: int = 1,
    rest: int = 200,
    lie: int = 2,
    frame: int = 0,
    command: bool = False,
) -> WorldState:
    ram = np.zeros(0x2000, dtype=np.uint8)
    ram[WRAM_HOLE_INDEX] = hole - 1
    ram[WRAM_STROKE_COUNT] = strokes
    ram[WRAM_REST_DISTANCE] = rest & 0xFF
    ram[WRAM_REST_DISTANCE + 1] = rest >> 8
    ram[WRAM_LIE_TYPE] = lie
    return WorldState(
        frame=frame,
        ram=ram,
        info={
            "hole_index": hole - 1,
            "stroke_count": strokes,
            "rest_distance": rest,
            "lie_type": lie,
        },
        obs=_command_obs() if command else np.zeros((224, 256, 3), dtype=np.uint8),
    )


def test_scorecard_book_records_match_and_stroke_totals() -> None:
    book = ScorecardBook()
    book.record(4, 1)
    book.record(3, 2, opponent=4)
    book.record(5, 3, opponent=5)
    assert book.total == 12
    assert book.match_lead == 1
    card = book.as_dict([4, 4, 5])
    assert card["holes"] == [4, 3, 5]
    assert card["to_par"] == -1
    assert card["holes_won"] == 1
    assert card["holes_tied"] == 1


def test_shot_task_reaches_success() -> None:
    task = ShotTask(power_delay=1, impact_delay=1, flight_wait=1)
    world = _world()
    task.reset(world)
    status = TaskStatus.RUNNING
    for _ in range(500):
        result = task.step(world)
        status = result.status
        if status != TaskStatus.RUNNING:
            break
    assert status == TaskStatus.SUCCESS


def test_shot_task_accepts_signed_aim_steps() -> None:
    task = ShotTask(
        power_delay=1,
        impact_delay=1,
        flight_wait=1,
        aim_steps=-3,
    )
    world = _world()
    task.reset(world)
    for _ in range(200):
        result = task.step(world)
        if task._phase.name == "CONFIRM_AIM" and result.action is not None:
            assert int(result.action.action.sum()) == 1
            return
    raise AssertionError("aim phase did not emit an action")


def test_shot_task_accepts_club_downs() -> None:
    task = ShotTask(
        power_delay=1,
        impact_delay=1,
        flight_wait=1,
        club_downs=12,
    )
    world = _world()
    task.reset(world)
    for _ in range(500):
        result = task.step(world)
        if task._phase.name == "CONFIRM_CLUB" and result.action is not None:
            assert int(result.action.action.sum()) <= 1
            return
    raise AssertionError("club phase did not emit an action")


def test_mission_skip_bootstrap_starts_shot() -> None:
    mission = StrokePlayMission(skip_bootstrap=True, max_holes=18)
    world = _world(hole=1, strokes=1)
    mission.reset(world)
    assert mission._phase == MissionPhase.PLAY_HOLE
    assert mission._shot is not None


def test_autopilot_resume_starts_recovery() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world()
    mission.reset(world)
    mission.on_autopilot_resume()
    assert mission._recovery.active
    result = mission.step(world)
    assert result.status == TaskStatus.RUNNING
    assert result.action is not None
    assert result.action.reason == "recovery"


def test_recovery_controller_finishes() -> None:
    from hals_golf.core.scene import SceneDecision
    from hals_golf.core.ram import GameScene

    ctl = RecoveryController(warmup_frames=5)
    ctl.start()
    decision = SceneDecision(
        GameScene.COMMAND, needs_dismiss=False, wait_only=False, reason="ok"
    )
    saw_action = False
    for _ in range(200):
        action = ctl.step(decision)
        if action is None:
            break
        saw_action = True
    assert saw_action
    assert not ctl.active


def test_mission_status_snapshot() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=3, strokes=2))
    status = mission.mission_status()
    assert status.mission_id == "stroke_play"
    assert "hole=3" in status.objective


def test_mission_uses_putt_inside_15_yards() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(rest=12, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 42


def test_mission_uses_calibrated_tap_in_power() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world(rest=2, lie=6)
    mission.reset(world)
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 42


def test_mission_varies_repeated_putt_distance() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world(rest=10, lie=6)
    mission.reset(world)
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 42

    mission._start_shot(world)
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 41


def test_mission_uses_short_approach_inside_55_yards() -> None:
    mission = StrokePlayMission(skip_bootstrap=True, power_delay=42)
    mission.reset(_world(rest=37))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 32


def test_hole_six_splashes_short_bunker_lie_onto_green() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=6, strokes=3, rest=12, lie=3))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 28


def test_holes_four_through_six_use_par_routes() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=4, strokes=0, rest=178, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 39
    assert mission._shot.aim_steps == 1
    assert mission._shot.club_downs == 11

    mission.reset(_world(hole=5, strokes=2, rest=31, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 27
    assert mission._shot.aim_steps == 5

    mission.reset(_world(hole=6, strokes=0, rest=360, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 1

    mission.reset(_world(hole=6, strokes=2, rest=42, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 35


def test_hole_five_uses_calibrated_birdie_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=5, strokes=2, rest=8, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 38


def test_hole_three_lays_up_with_eight_iron() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=3, strokes=0, rest=505, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 32
    assert mission._shot.aim_steps == -20
    assert mission._shot.club_downs == 9


def test_distance_fallback_uses_iron_not_driver() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=5, strokes=8, rest=157, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40
    assert mission._shot.club_downs == 6


def test_bunker_does_not_use_tee_plan_slot() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    # strokes=0 would match plan slot 0 (driver) if we did not require tee lie.
    mission.reset(_world(hole=5, strokes=0, rest=193, lie=3, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs >= 8
    assert mission._shot.power_delay <= 38


def test_require_rest_change_ignores_stroke_only() -> None:
    """Phantom VS HAL stroke bumps must not complete a shot at the same REST."""
    task = ShotTask(
        power_delay=1,
        impact_delay=1,
        flight_wait=5,
        require_rest_change=True,
    )
    world = _world(strokes=3, rest=80, lie=2, command=True)
    task.reset(world)
    # Advance to WAIT_FLIGHT.
    for _ in range(400):
        result = task.step(world)
        if task._phase == ShotPhase.WAIT_FLIGHT:
            break
        assert result.status == TaskStatus.RUNNING
    assert task._phase == ShotPhase.WAIT_FLIGHT
    # Stroke byte advances, REST stays put — still running.
    world.info["stroke_count"] = 4
    world.ram[WRAM_STROKE_COUNT] = 4
    for _ in range(10):
        result = task.step(world)
        assert result.status == TaskStatus.RUNNING
    # Soft escape after grace returns FAILURE, not SUCCESS.
    task._flight_elapsed = task.flight_wait + 480
    result = task.step(world)
    assert result.status == TaskStatus.FAILURE


def test_require_rest_change_accepts_green_stroke_bump() -> None:
    """Hole-outs often leave REST stale; green stroke bumps may settle."""
    task = ShotTask(
        power_delay=1,
        impact_delay=1,
        flight_wait=5,
        require_rest_change=True,
    )
    world = _world(strokes=2, rest=19, lie=6, command=True)
    task.reset(world)
    for _ in range(400):
        task.step(world)
        if task._phase == ShotPhase.WAIT_FLIGHT:
            break
    world.info["stroke_count"] = 3
    world.ram[WRAM_STROKE_COUNT] = 3
    task._flight_elapsed = 120
    result = task.step(world)
    assert result.status == TaskStatus.RUNNING
    assert task._phase == ShotPhase.DONE


def test_full_shot_reports_hole_out_without_command_panel() -> None:
    task = ShotTask(
        power_delay=1,
        impact_delay=1,
        flight_wait=1400,
        require_rest_change=True,
        complete_on_rest_zero=True,
    )
    world = _world(strokes=2, rest=20, lie=2, command=False)
    task.reset(world)
    task._queue = []
    task._wait = 0
    task._phase = ShotPhase.WAIT_FLIGHT
    task._flight_elapsed = 120
    world.info["rest_distance"] = 0
    world.ram[WRAM_REST_DISTANCE] = 0

    result = task.step(world)

    assert result.status == TaskStatus.SUCCESS


def test_putt_reports_hole_out_without_command_panel() -> None:
    task = PuttTask(
        power_delay=24,
        flight_wait=1200,
        complete_on_rest_zero=True,
    )
    world = _world(strokes=4, rest=20, lie=6, command=False)
    task.reset(world)
    task._queue = []
    task._wait = 0
    task._phase = "flight"
    task._flight_elapsed = 120
    world.info["rest_distance"] = 0
    world.ram[WRAM_REST_DISTANCE] = 0

    result = task.step(world)

    assert result.status == TaskStatus.SUCCESS


def test_putt_supports_signed_aim_taps() -> None:
    world = _world(strokes=1, rest=13, lie=6, command=True)
    straight = PuttTask(power_delay=18)
    aimed = PuttTask(power_delay=18, aim_steps=-3)

    straight.reset(world)
    aimed.reset(world)

    # Each aim step adds a two-frame tap and three release frames.
    assert len(aimed._queue) == len(straight._queue) + 15


def test_bunker_overrides_wood_plan_with_loft() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=7, strokes=1, rest=332, lie=3, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == -12


def test_vs_hal_hole_three_rest_band_overrides_desynced_stroke() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    # Stroke index 2 would use 7I; the 282y band keeps the -6 metal 4W.
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=3, strokes=2, rest=282, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 2
    assert mission._shot.aim_steps == -6
    assert mission._shot.power_delay == 42


def test_vs_hal_futile_success_does_not_advance_plan_stroke() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    world = _world(hole=3, strokes=1, rest=472, lie=2, command=True)
    mission.reset(world)
    mission._phase = MissionPhase.PLAY_HOLE
    mission._strokes_this_hole = 1
    mission._shot_start_key = (3, 1, 472, 2)
    mission._shot = ShotTask(power_delay=1, impact_delay=1, flight_wait=1)
    mission._shot.reset(world)
    mission._shot._queue = []
    mission._shot._wait = 0
    mission._shot._phase = ShotPhase.DONE
    result = mission.step(world)
    assert result.status == TaskStatus.RUNNING
    assert mission._strokes_this_hole == 1


def test_vs_hal_hole_three_uses_metal_wood_route_and_seven_iron_finish() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=3, strokes=0, rest=519, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -6
    assert mission._shot.club_downs == 0
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=3, strokes=2, rest=149, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 8
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == -2


def test_vs_hal_hole_three_preserves_standard_club_route() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.STANDARD,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=3, strokes=0, rest=519, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 9
    assert mission._shot.power_delay == 32
    assert mission._shot.aim_steps == -20


def test_vs_hal_metal_hole_four_uses_nine_iron_green_route() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=4, strokes=0, rest=188, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 10
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 0


def test_vs_hal_metal_hole_two_uses_pw_from_shifted_fairway() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=2, strokes=1, rest=90, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 11
    assert mission._shot.power_delay == 32
    assert mission._shot.aim_steps == 0

    mission._shot = None
    mission._start_shot(_world(hole=2, strokes=2, rest=23, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 20


def test_vs_hal_metal_hole_ten_uses_four_wood_and_green_approach() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=10, strokes=1, rest=295, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 4
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == -6

    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=10, strokes=2, rest=138, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 10
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 0

    mission._shot = None
    mission._start_shot(_world(hole=10, strokes=3, rest=14, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 18


def test_vs_hal_metal_hole_six_uses_three_shot_chip_in_route() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=1, rest=183, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 5
    assert mission._shot.power_delay == 42

    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=2, rest=65, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 28
    assert mission._shot.aim_steps == 6

    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=2, rest=77, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 30
    assert mission._shot.aim_steps == 4


def test_vs_hal_metal_hole_seven_avoids_bunker_lock() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=7, strokes=0, rest=509, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 44
    assert mission._shot.aim_steps == -4

    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=7, strokes=2, rest=210, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 3
    assert mission._shot.power_delay == 44
    assert mission._shot.aim_steps == -4

    mission._shot = None
    mission._start_shot(_world(hole=7, strokes=2, rest=82, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 34
    assert mission._shot.aim_steps == -3


def test_vs_hal_standard_hole_seven_keeps_five_iron_approach() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.STANDARD,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=7, strokes=2, rest=210, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 6
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == -6


def test_vs_hal_metal_hole_eleven_uses_sand_wedge_finish() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 3
    mission._shot = None
    mission._start_shot(_world(hole=11, strokes=3, rest=91, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 32
    assert mission._shot.aim_steps == -4

    mission._shot = None
    mission._start_shot(_world(hole=11, strokes=3, rest=151, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 9
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 0


def test_vs_hal_metal_hole_twelve_uses_three_shot_finish() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=12, strokes=0, rest=402, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == 0

    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=12, strokes=2, rest=24, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 26
    assert mission._shot.aim_steps == -8


def test_vs_hal_metal_hole_five_uses_sand_wedge_approach() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=5, strokes=2, rest=47, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 26
    assert mission._shot.aim_steps == -8


def test_vs_hal_hole_eight_uses_sand_wedge_chip_in() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=8, strokes=2, rest=110, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == 2


def test_vs_hal_metal_hole_eight_uses_wedge_from_shifted_fairway() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=8, strokes=1, rest=106, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == 2


def test_vs_hal_hole_nine_aims_right_onto_green() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=9, strokes=0, rest=208, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 11
    assert mission._shot.aim_steps == 2
    assert mission._shot.power_delay == 39


def test_vs_hal_hole_seven_and_eight_hole_long_putts() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=7, strokes=5, rest=20, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 36
    mission.reset(_world(hole=8, strokes=3, rest=23, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 28


def test_vs_hal_hole_twelve_holes_twenty_three_yard_putt() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=12, strokes=2, rest=23, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 23


def test_vs_hal_hole_four_holes_seventeen_yard_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=4, strokes=1, rest=17, lie=6, command=True))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 18


def test_vs_hal_hole_five_uses_fairway_corridor() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=5, strokes=0, rest=416, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == -2
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=5, strokes=2, rest=100, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 10
    assert mission._shot.power_delay == 36
    assert mission._shot.aim_steps == 5


def test_short_rest_ignores_midhole_wood_plan() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    # Stroke-1 plan is driver; short fairway must demote to SW.
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=1, rest=44, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 12
    assert mission._shot.power_delay == 32


def test_vs_hal_hole_eleven_uses_seven_iron_to_green() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 3
    mission._shot = None
    mission._start_shot(_world(hole=11, strokes=3, rest=118, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 8
    assert mission._shot.power_delay == 38


def test_vs_hal_hole_ten_avoids_tree_corridor() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=10, strokes=0, rest=426, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 6
    assert mission._shot.aim_steps == -12
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=10, strokes=2, rest=169, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -8


def test_vs_hal_hole_thirteen_uses_eight_iron_to_green() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=13, strokes=0, rest=170, lie=1, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 9
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 0


def test_short_fairway_uses_soft_driver_not_sw() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 7
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=7, rest=17, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 32
    assert mission._shot.aim_steps == 0


def test_vs_hal_hole_six_uses_verified_driver_finish() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=2, rest=123, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == 0

    mission._strokes_this_hole = 3
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=3, rest=32, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 28
    assert mission._shot.aim_steps == 4


def test_vs_hal_hole_seven_escapes_full_course_tree_lie() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=7, strokes=2, rest=259, lie=0, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 36
    assert mission._shot.aim_steps == 6


def test_deep_bunker_uses_driver_left_escape() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 4
    mission._shot = None
    mission._start_shot(_world(hole=3, strokes=4, rest=299, lie=3, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == -12


def test_mid_bunker_uses_three_iron_escape() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=5, strokes=1, rest=193, lie=3, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 4
    assert mission._shot.power_delay == 42


def test_short_bunker_uses_soft_driver_splash() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 2
    mission._shot = None
    mission._start_shot(_world(hole=6, strokes=2, rest=44, lie=3, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 0
    assert mission._shot.power_delay == 36
    assert mission._shot.aim_steps == 4


def test_vs_hal_hole_two_keeps_three_wood_approach() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    mission._strokes_this_hole = 1
    mission._shot = None
    mission._start_shot(_world(hole=2, strokes=1, rest=124, lie=2, command=True))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 1
    assert mission._shot.power_delay == 42


def test_hole_two_reaches_birdie_putt_with_three_wood() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=2, strokes=0, rest=329, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42

    mission.reset(_world(hole=2, strokes=1, rest=111, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.club_downs == 1

    mission.reset(_world(hole=2, strokes=2, rest=11, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 40


def test_hole_three_uses_clubs_for_water_route_and_chip_in() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=3, strokes=1, rest=452, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -4
    assert mission._shot.club_downs == 1

    mission.reset(_world(hole=3, strokes=2, rest=357, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -6
    assert mission._shot.club_downs == 4

    mission.reset(_world(hole=3, strokes=3, rest=205, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == 1
    assert mission._shot.club_downs == 2

    mission.reset(_world(hole=3, strokes=4, rest=51, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 26
    assert mission._shot.aim_steps == -7
    assert mission._shot.club_downs == 7


def test_hole_three_uses_calibrated_par_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=3, strokes=4, rest=8, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 39


def test_hole_seven_avoids_tee_hazard() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=7, strokes=0, rest=516, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == 0

    mission.reset(_world(hole=7, strokes=3, rest=32, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 36
    assert mission._shot.aim_steps == -2

    mission.reset(_world(hole=7, strokes=4, rest=13, lie=6))
    assert mission._shot.power_delay == 20


def test_hole_eight_avoids_bunker_and_reaches_birdie_putt() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=8, strokes=0, rest=356, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -4

    mission.reset(_world(hole=8, strokes=1, rest=136))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == 2
    assert mission._shot.club_downs == 9

    mission.reset(_world(hole=8, strokes=2, rest=41))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 31
    assert mission._shot.aim_steps == -5


def test_hole_nine_uses_wedge_and_calibrated_birdie_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=9, strokes=0, rest=198, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40
    assert mission._shot.aim_steps == 0
    assert mission._shot.club_downs == 11

    mission.reset(_world(hole=9, strokes=1, rest=8, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 40

    mission.reset(_world(hole=9, strokes=1, rest=17, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 40

    mission.reset(_world(hole=9, strokes=1, rest=19, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 21

    mission.reset(_world(hole=9, strokes=1, rest=22, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 18


def test_hole_thirteen_uses_eight_iron_on_tee() -> None:
    """Live-clear wind: 5I stops ~85y; 8I reaches an 18y birdie putt."""
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=13, strokes=0, rest=176, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 0
    assert mission._shot.club_downs == 9

    mission.reset(_world(hole=13, strokes=1, rest=18, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 20


def test_hole_ten_uses_safe_par_route() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=10, strokes=0, rest=452, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -8
    assert mission._shot.club_downs == 6

    mission.reset(_world(hole=10, strokes=1, rest=275, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -8
    assert mission._shot.club_downs == 4

    mission.reset(_world(hole=10, strokes=2, rest=114, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 37
    assert mission._shot.aim_steps == -1
    assert mission._shot.club_downs == 8


def test_hole_eleven_uses_safe_tee_power() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=11, strokes=0, rest=523, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40

    mission.reset(_world(hole=11, strokes=2, rest=295, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -11
    assert mission._shot.club_downs == 1

    mission.reset(_world(hole=11, strokes=3, rest=162, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -4


def test_hole_eleven_uses_calibrated_par_putts() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=11, strokes=4, rest=11, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 37

    mission.reset(_world(hole=11, strokes=4, rest=16, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 18


def test_hole_twelve_avoids_out_of_bounds_tee_shot() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=12, strokes=0, rest=400, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40

    mission.reset(_world(hole=12, strokes=2, rest=53, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 37
    assert mission._shot.aim_steps == -1
    assert mission._shot.club_downs == 0


def test_safe_tee_plan_is_reused_after_penalty() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=12, strokes=2, rest=400, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40


def test_hole_fifteen_uses_safe_tee_power() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=15, strokes=0, rest=457))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42

    mission.reset(_world(hole=15, strokes=1, rest=236, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.club_downs == 2

    mission.reset(_world(hole=15, strokes=2, rest=84, lie=0))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 38
    assert mission._shot.aim_steps == 3
    assert mission._shot.club_downs == 0


def test_holes_fourteen_and_seventeen_use_par_approaches() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=14, strokes=1, rest=133, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 41
    assert mission._shot.aim_steps == 1
    assert mission._shot.club_downs == 10

    mission.reset(_world(hole=14, strokes=2, rest=42, lie=3))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 38
    assert mission._shot.club_downs == 4

    mission.reset(_world(hole=17, strokes=0, rest=173, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 36
    assert mission._shot.club_downs == 8


def test_hole_sixteen_uses_pitching_wedge_from_long_bunker() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=16, strokes=0, rest=522, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == 0

    mission.reset(_world(hole=16, strokes=1, rest=282, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -5
    assert mission._shot.club_downs == 2

    mission.reset(_world(hole=16, strokes=2, rest=280, lie=3))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -4
    assert mission._shot.club_downs == 11

    mission.reset(_world(hole=16, strokes=3, rest=224, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 40

    mission.reset(_world(hole=16, strokes=4, rest=46, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 25
    assert mission._shot.aim_steps == -8
    assert mission._shot.club_downs == 6


def test_hole_sixteen_uses_calibrated_birdie_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=16, strokes=3, rest=15, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 13


def test_hole_eighteen_uses_safe_three_shot_route() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)

    mission.reset(_world(hole=18, strokes=0, rest=416, lie=1))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.club_downs == 2

    mission.reset(_world(hole=18, strokes=1, rest=207, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 42
    assert mission._shot.aim_steps == -5
    assert mission._shot.club_downs == 1

    mission.reset(_world(hole=18, strokes=2, rest=50, lie=2))
    assert isinstance(mission._shot, ShotTask)
    assert mission._shot.power_delay == 26
    assert mission._shot.aim_steps == -6
    assert mission._shot.club_downs == 12


def test_hole_eighteen_uses_stable_eight_yard_putt() -> None:
    from hals_golf.tasks.shot import PuttTask

    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=18, strokes=3, rest=8, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 39

    mission.reset(_world(hole=18, strokes=2, rest=29, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 25

    mission.reset(_world(hole=18, strokes=3, rest=15, lie=6))
    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 16


def test_raw_hole_index_18_completes_course() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world(hole=18, strokes=8, rest=1)
    mission.reset(world)
    ram = world.ram.copy()
    ram[WRAM_HOLE_INDEX] = 18
    finished = WorldState(
        frame=1,
        ram=ram,
        info={
            "hole_index": 18,
            "stroke_count": 9,
            "rest_distance": 0,
        },
        obs=world.obs,
    )
    result = mission.step(finished)
    assert result.status == TaskStatus.SUCCESS
    assert "course_complete" in (result.reason or "")
    assert "total=9" in (result.reason or "")
    assert mission.scorecard()["total"] == 9


def test_read_aim_offset_uses_confirmed_address() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    ram[WRAM_AIM_OFFSET] = 74
    assert read_aim_offset(ram) == 74
    assert read_aim_offset(ram, {"aim_offset": 12}) == 12


def test_hole_advance_records_peak_stroke_count() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=1, strokes=0, rest=369, lie=1))
    mission.step(_world(hole=1, strokes=4, rest=20, lie=2))
    result = mission.step(_world(hole=2, strokes=0, rest=400, lie=1))
    assert result.status == TaskStatus.RUNNING
    assert mission._hole_scores == [4]
    assert mission.scorecard()["total"] == 4


def test_hole_advance_does_not_carry_stale_strokes_into_next_hole() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=1, strokes=0, rest=369, lie=1))
    mission.step(_world(hole=1, strokes=5, rest=4, lie=6))

    # The game can advance the hole byte while the old stroke byte is visible.
    mission.step(_world(hole=2, strokes=5, rest=4, lie=6))

    assert mission._hole_scores == [5]
    assert mission._peak_strokes_this_hole == 0

    # The stale byte can remain visible for several frames on the new hole.
    mission.step(_world(hole=2, strokes=5, rest=4, lie=6))
    assert mission._peak_strokes_this_hole == 0

    mission.step(_world(hole=2, strokes=0, rest=329, lie=1))
    mission.step(_world(hole=2, strokes=1, rest=329, lie=1))
    assert mission._peak_strokes_this_hole == 1


def test_hole_advance_falls_back_to_peak_strokes() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=1, strokes=0, rest=369, lie=1))
    mission.step(_world(hole=1, strokes=4, rest=10, lie=6))
    result = mission.step(_world(hole=2, strokes=0, rest=400, lie=1))
    assert result.status == TaskStatus.RUNNING
    assert mission._hole_scores == [4]
    assert mission.scorecard()["total"] == 4


def test_round_total_sums_recorded_hole_scores() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    mission.reset(_world(hole=1, strokes=0, rest=369, lie=1))
    mission.step(_world(hole=1, strokes=4, rest=10, lie=6))
    mission.step(_world(hole=2, strokes=0, rest=400, lie=1))
    mission.step(_world(hole=2, strokes=3, rest=20, lie=2))
    mission.step(_world(hole=3, strokes=0, rest=505, lie=1))

    assert mission._hole_scores == [4, 3]
    assert mission.scorecard()["total"] == 7
    assert mission.scorecard()["hole_numbers"] == [1, 2]
    assert mission.scorecard()["pars"] == [4, 4]
    assert mission.scorecard()["to_par"] == -1
    assert mission.scorecard()["over_par_holes"] == []


def test_command_stall_nudges_shot_plan() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world(hole=8, strokes=1, rest=147, lie=2, command=True)
    assert is_command_screen(world.obs)
    mission.reset(world)
    mission._stall_key = (8, 1, 147, 2)
    mission._stall_frames = COMMAND_STALL_FRAMES - 1
    mission._stall_nudges = 0
    action = mission._maybe_break_command_stall(world, 8, 1, 147)
    assert action is not None
    assert mission._stall_nudges == 1
    assert mission._recovery.active
    assert mission._shot is not None


def test_futile_shots_trigger_nudge() -> None:
    mission = StrokePlayMission(skip_bootstrap=True)
    world = _world(hole=8, strokes=1, rest=147, lie=2)
    mission.reset(world)
    mission._phase = MissionPhase.PLAY_HOLE
    mission._stall_key = (8, 1, 147, 2)
    mission._futile_shots = FUTILE_SHOT_LIMIT - 1
    mission._stall_nudges = 0
    mission._shot_start_key = (8, 1, 147, 2)
    mission._shot = ShotTask(power_delay=1, impact_delay=1, flight_wait=1)
    mission._shot.reset(world)
    mission._shot._queue = []
    mission._shot._wait = 0
    mission._shot._phase = ShotPhase.DONE
    result = mission.step(world)
    assert result.status == TaskStatus.RUNNING
    assert mission._stall_nudges == 1
    assert mission._recovery.active


def test_vs_hal_bootstrap_differs_from_stroke_play() -> None:
    stroke = title_to_stroke_play_amateur_frames()
    vs_hal = title_to_vs_hal_amateur_frames()
    assert len(vs_hal) > 0
    assert len(vs_hal) != len(stroke)
    mission = StrokePlayMission(play_mode=PlayMode.VS_HAL)
    world = _world(hole=1, strokes=0, rest=0)
    mission.reset(world)
    assert mission.name == "vs_hal"
    assert mission._bootstrap.play_mode is PlayMode.VS_HAL
    assert len(mission._bootstrap.frames) == len(vs_hal)

    metal_vs_hal = title_to_vs_hal_amateur_frames(club_set=ClubSet.METAL)
    metal_mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
    )
    metal_mission.reset(world)
    assert len(metal_vs_hal) > len(vs_hal)
    assert len(metal_mission._bootstrap.frames) == len(metal_vs_hal)

    metal_stroke = title_to_stroke_play_amateur_frames(club_set=ClubSet.METAL)
    metal_stroke_mission = StrokePlayMission(club_set=ClubSet.METAL)
    metal_stroke_mission.reset(world)
    assert len(metal_stroke) > len(stroke)
    assert len(metal_stroke_mission._bootstrap.frames) == len(metal_stroke)


def test_pro_stroke_play_frames_differ_from_amateur() -> None:
    amateur = title_to_stroke_play_frames(difficulty=Difficulty.AMATEUR)
    pro = title_to_stroke_play_frames(difficulty=Difficulty.PRO)
    assert len(amateur) == len(title_to_stroke_play_amateur_frames())
    # Pro adds a DOWN tap + settle before START on the difficulty screen.
    assert len(pro) > len(amateur)


def test_pro_vs_hal_frames_differ_from_amateur() -> None:
    amateur = title_to_vs_hal_frames(difficulty=Difficulty.AMATEUR)
    pro = title_to_vs_hal_frames(difficulty=Difficulty.PRO)
    assert len(amateur) == len(title_to_vs_hal_amateur_frames())
    assert len(pro) > len(amateur)


def test_pro_bootstrap_task_uses_longer_script() -> None:
    amateur = MenuBootstrapTask(difficulty=Difficulty.AMATEUR)
    pro = MenuBootstrapTask(difficulty=Difficulty.PRO)
    assert len(pro.frames) > len(amateur.frames)


def test_mission_pro_difficulty_flows_to_bootstrap_and_profile() -> None:
    mission = StrokePlayMission(difficulty=Difficulty.PRO)
    world = _world(hole=1, strokes=0, rest=0)
    mission.reset(world)
    assert mission._bootstrap.difficulty is Difficulty.PRO
    assert mission.profile.is_pro
    amateur_bootstrap = MenuBootstrapTask(difficulty=Difficulty.AMATEUR)
    assert len(mission._bootstrap.frames) > len(amateur_bootstrap.frames)


def test_vs_hal_name_script_enters_metal_play_and_confirms_ok() -> None:
    steps = _metal_play_name_nav()
    assert steps[-1] == ("B", 3)
    # Nine letter selects, one cursor-right select, and the final OK.
    assert sum(1 for button, _frames in steps if button == "B") == 11


def test_vs_hal_hole_one_uses_calibrated_birdie_putt() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        club_set=ClubSet.METAL,
        skip_bootstrap=True,
    )
    mission.reset(_world(hole=1, strokes=2, rest=8, lie=6, command=True))

    assert isinstance(mission._shot, PuttTask)
    assert mission._shot.power_delay == 42


def test_vs_hal_idles_without_command_panel() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    world = _world(hole=1, strokes=1, rest=196, command=False)
    mission.reset(world)
    mission._shot = None
    mission._phase = MissionPhase.PLAY_HOLE
    result = mission.step(world)
    assert result.status == TaskStatus.RUNNING
    assert mission._phase == MissionPhase.PLAY_HOLE
    assert result.action is not None
    assert int(result.action.action.sum()) == 0
    assert mission._shot is None


def test_vs_hal_opponent_wait_times_out_to_resume() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    world = _world(hole=5, strokes=2, rest=185, command=False)
    mission.reset(world)
    mission._shot = None
    mission._phase = MissionPhase.WAIT_OPPONENT
    mission._opponent_wait_frames = OPPONENT_WAIT_LIMIT
    result = mission.step(world)
    assert result.status == TaskStatus.RUNNING
    assert mission._phase == MissionPhase.PLAY_HOLE


def test_vs_hal_never_plays_opponent_command_after_hole_out() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    world = _world(hole=3, strokes=5, rest=13, lie=6, command=True)
    mission.reset(world)
    mission._shot = None
    mission._phase = MissionPhase.WAIT_OPPONENT
    mission._waiting_after_hole_out = True

    result = mission.step(world)

    assert result.status == TaskStatus.RUNNING
    assert mission._phase == MissionPhase.WAIT_OPPONENT
    assert mission._shot is None
    assert result.action is not None
    assert int(result.action.action.sum()) == 0


def test_vs_hal_match_won_at_twelve_hole_boundary() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
        max_holes=18,
    )
    world = _world(hole=13, strokes=0, rest=300, command=True)
    mission.reset(world)
    # The game ends VS HAL after Hole 12 even though stroke play has 18.
    mission._holes_completed = 12
    mission._card.holes_won = 3
    mission._card.holes_lost = 2
    mission._card.holes_tied = 7
    mission._card.holes = [4] * 12
    mission._card.hole_numbers = list(range(1, 13))
    mission._last_hole = 13
    mission._phase = MissionPhase.PLAY_HOLE
    mission._shot = None
    result = mission.step(world)
    assert result.status == TaskStatus.SUCCESS
    assert result.reason is not None
    assert result.reason.startswith("match_won")


def test_vs_hal_records_opponent_hole_score() -> None:
    mission = StrokePlayMission(
        play_mode=PlayMode.VS_HAL,
        skip_bootstrap=True,
    )
    world = _world(hole=2, strokes=0, rest=300, command=True)
    world.ram[WRAM_OPPONENT_STROKE_COUNT] = 5
    mission.reset(world)
    mission._last_hole = 1
    mission._peak_strokes_this_hole = 4
    mission._strokes_this_hole = 4
    mission._record_hole_score(world)
    assert mission._card.holes == [4]
    assert mission._card.opponent_holes == [5]
    assert mission._card.holes_won == 1
    assert mission.match_lead() == 1
