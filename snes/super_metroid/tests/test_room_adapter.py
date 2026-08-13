from __future__ import annotations

from super_metroid.room_adapter import (
    AdapterSearchConfig,
    TimedAction,
    beam_search_adapter,
)


IDLE = (0,) * 12


def test_beam_adapter_minimizes_frames_to_rejoin() -> None:
    actions = (
        TimedAction(IDLE, 1, "one"),
        TimedAction(IDLE, 2, "two"),
        TimedAction(IDLE, 4, "four"),
    )

    plan = beam_search_adapter(
        0,
        0,
        actions=actions,
        expand=lambda token, action: (
            token + action.frames,
            token + action.frames,
        ),
        score=lambda state: abs(5 - state),
        reached=lambda state, score: state == 5 and score == 0,
        state_key=lambda state: (state,),
        config=AdapterSearchConfig(beam_width=8, max_depth=3, frame_penalty=1.0),
    )

    assert plan.reached
    assert plan.frame_count == 5
    assert plan.score_before == 5
    assert plan.score_after == 0


def test_beam_adapter_returns_best_bounded_partial_plan() -> None:
    action = TimedAction(IDLE, 2, "two")

    plan = beam_search_adapter(
        0,
        0,
        actions=(action,),
        expand=lambda token, timed: (
            token + timed.frames,
            token + timed.frames,
        ),
        score=lambda state: abs(10 - state),
        reached=lambda _state, score: score == 0,
        state_key=lambda state: (state,),
        config=AdapterSearchConfig(beam_width=1, max_depth=2),
    )

    assert not plan.reached
    assert plan.frame_count == 4
    assert plan.score_after == 6
