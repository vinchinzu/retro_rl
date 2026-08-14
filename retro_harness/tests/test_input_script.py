"""Tests for reusable title/menu and deterministic input scripts."""

from __future__ import annotations

from retro_harness.actions import idle_action, snes_action
from retro_harness.input_script import (
    InputStep,
    PeriodPulse,
    StartupPlan,
    idle_frames,
    iter_input_steps,
    mash_button,
    mash_start,
    parse_input_script,
    period_script,
    press_button_sequence,
    run_startup,
)


class _FakeEnv:
    def __init__(self, *, old_api: bool = False) -> None:
        self.frame = 0
        self.old_api = old_api
        self.actions: list[list[int]] = []

    def reset(self):
        self.frame = 0
        self.actions.clear()
        return "reset-observation", {"frame": 0}

    def step(self, action):
        self.frame += 1
        self.actions.append(list(action))
        result = (f"obs-{self.frame}", 0.0, False, {"frame": self.frame})
        if self.old_api:
            return result
        obs, reward, done, info = result
        return obs, reward, done, False, info


def test_parse_input_script_supports_chords_and_waits() -> None:
    assert parse_input_script("WAIT:0:20 RIGHT+Y:3:4") == [
        InputStep([], 0, 20),
        InputStep([7, 1], 3, 4),
    ]


def test_iter_input_steps_releases_between_inputs() -> None:
    frames = list(iter_input_steps([InputStep([8], 2, 1)]))

    assert [frame.action[8] for frame in frames] == [1, 1, 0]
    assert [frame.reason for frame in frames] == ["a", "a", "wait"]


def test_startup_plan_encodes_common_title_menu_flow() -> None:
    plan = StartupPlan.title_menu(
        "DOWN",
        initial_wait=1,
        start_hold=1,
        start_wait=1,
        move_hold=1,
        move_wait=1,
        confirm_hold=1,
        confirm_wait=1,
    )
    frames = list(iter_input_steps(plan.steps))

    assert len(frames) == 7
    assert [frame.action[3] for frame in frames].count(1) == 1
    assert [frame.action[5] for frame in frames].count(1) == 1
    assert [frame.action[8] for frame in frames].count(1) == 1


def test_run_startup_stops_at_first_playable_frame_with_old_gym_api() -> None:
    env = _FakeEnv(old_api=True)
    plan = StartupPlan.parse("START:1:2 A:1:4")

    result = run_startup(
        env,
        plan,
        is_ready=lambda active_env, _info: active_env.frame >= 4,
        max_cycles=2,
    )

    assert result.ready is True
    assert result.frames == 4
    assert env.actions[0][3] == 1
    assert env.actions[3][8] == 1


def test_run_startup_replays_plan_across_cycles_until_ready() -> None:
    env = _FakeEnv()
    plan = StartupPlan.parse("START:1:2")  # 3 frames per cycle

    result = run_startup(
        env,
        plan,
        is_ready=lambda active_env, _info: active_env.frame >= 5,
        max_cycles=3,
    )

    assert result.ready is True
    assert result.frames == 5
    assert result.completed is False
    assert env.frame == 5


def test_run_startup_exhausts_cycles_without_readiness() -> None:
    env = _FakeEnv()
    plan = StartupPlan.parse("START:1:2")

    result = run_startup(
        env,
        plan,
        is_ready=lambda _a, _i: False,
        max_cycles=3,
    )

    assert result.ready is False
    assert result.completed is True
    assert result.frames == 9
    assert env.frame == 9


def test_run_startup_returns_reset_ready_without_stepping() -> None:
    env = _FakeEnv()
    plan = StartupPlan.parse("START:1:2")

    result = run_startup(
        env,
        plan,
        is_ready=lambda _a, _i: True,
        max_cycles=3,
    )

    assert result.ready is True
    assert result.frames == 0
    assert env.actions == []


def test_period_script_pulses_slots_on_repeating_period() -> None:
    start = snes_action("START")
    idle = idle_action()
    frames = list(
        period_script(
            max_frames=10,
            period=5,
            pulses=(PeriodPulse(1, 3, start, "boot_start"),),
            idle=idle,
        )
    )

    assert len(frames) == 10
    for frame_no, frame in enumerate(frames, start=1):
        if 1 <= (frame_no % 5) < 3:
            assert frame.action == start
            assert frame.reason == "boot_start"
        else:
            assert frame.action == idle
            assert frame.reason == "boot_wait"


def test_period_script_overlapping_pulses_first_wins() -> None:
    start = snes_action("START")
    confirm = snes_action("A")
    frames = list(
        period_script(
            max_frames=5,
            period=5,
            pulses=(
                PeriodPulse(1, 4, start, "boot_start"),
                PeriodPulse(2, 5, confirm, "boot_confirm"),
            ),
            idle=idle_action(),
        )
    )

    assert [frame.reason for frame in frames] == [
        "boot_start",
        "boot_start",
        "boot_start",
        "boot_confirm",
        "boot_wait",
    ]
    assert frames[1].action == start
    assert frames[3].action == confirm


def test_period_script_default_range_matches_one_based_inclusive() -> None:
    start = snes_action("START")
    frames = list(
        period_script(
            max_frames=3,
            period=10,
            pulses=(PeriodPulse(1, 2, start, "boot_start"),),
            idle=idle_action(),
            start_frame=1,
        )
    )

    # range(1, max_frames + 1): first yielded frame is 1, last is max_frames.
    assert len(frames) == 3
    assert frames[0].action == start
    assert frames[0].reason == "boot_start"
    assert frames[1].reason == "boot_wait"
    assert frames[2].reason == "boot_wait"


def test_press_button_sequence_is_shared_numpy_neutral_primitive() -> None:
    actions = press_button_sequence(
        "A",
        face="UP",
        face_frames=1,
        pre_press_settle_frames=1,
        hold_frames=2,
        settle_frames=1,
        hold_face_with_button=True,
    )

    assert len(actions) == 5
    assert actions[0][4] == 1
    assert actions[2][4] == 1 and actions[2][8] == 1
    assert sum(actions[-1]) == 0


def test_mash_button_keeps_custom_reasons() -> None:
    frames = mash_button(
        "START",
        pulses=2,
        hold=2,
        gap=1,
        hold_reason="char_confirm",
        wait_reason="char_wait",
    )
    assert [f.reason for f in frames] == [
        "char_confirm",
        "char_confirm",
        "char_wait",
        "char_confirm",
        "char_confirm",
        "char_wait",
    ]
    start = mash_start(pulses=1, hold=1, gap=1)
    assert [f.reason for f in start] == ["start", "wait"]
    idle = idle_frames(2, "post_title_idle")
    assert [f.reason for f in idle] == ["post_title_idle", "post_title_idle"]
    assert all(sum(f.action) == 0 for f in idle)
