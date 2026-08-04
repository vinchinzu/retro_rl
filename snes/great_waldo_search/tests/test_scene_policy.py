"""Pure-logic tests for Great Waldo Search helpers."""

from __future__ import annotations

import numpy as np
import pytest

from great_waldo_search.ram import (
    deltas_for_move,
    filter_byte_range,
    rank_axis_candidates,
    read_u8,
)
from great_waldo_search.scene_policy import (
    CursorPose,
    CursorTarget,
    at_target,
    plan_cursor_path,
    playing_state,
    step_toward_target,
)
from great_waldo_search.targets import (
    CONFIRMED_FIND_POINTS,
    SCENE1_TARGETS,
    TargetStatus,
    confirmed_targets,
    score_u16,
)
from retro_harness.ram_state import RamDelta


def test_read_u8() -> None:
    ram = np.arange(16, dtype=np.uint8)
    assert read_u8(ram, 5) == 5


def test_rank_axis_candidates_requires_hits() -> None:
    d1 = [RamDelta(10, 1, 2), RamDelta(11, 0, 1)]
    d2 = [RamDelta(10, 2, 3)]
    ranked = rank_axis_candidates([d1, d2], axis="x", min_hits=2)
    assert len(ranked) == 1
    assert ranked[0].address == 10
    assert ranked[0].hits == 2


def test_filter_and_delta_helpers() -> None:
    before = np.zeros(8, dtype=np.uint8)
    after = before.copy()
    after[3] = 5
    deltas = deltas_for_move(before, after)
    assert len(deltas) == 1
    assert filter_byte_range(deltas, lo=0, hi=4) == []
    assert filter_byte_range(deltas, lo=0, hi=5)[0].address == 3


def test_step_toward_prefers_larger_axis() -> None:
    pose = CursorPose(0, 0)
    target = CursorTarget(10, 3)
    action = step_toward_target(pose, target)
    assert action.reason == "cursor_right"


def test_step_toward_confirm_in_deadzone() -> None:
    pose = CursorPose(50, 50)
    target = CursorTarget(51, 49, deadzone=2)
    assert at_target(pose, target)
    action = step_toward_target(pose, target)
    assert action.reason == "confirm_at_target"
    assert action.action[8] == 1  # A


def test_plan_cursor_path_reaches_target() -> None:
    frames = plan_cursor_path(CursorPose(0, 0), CursorTarget(4, 2, deadzone=0))
    assert len(frames) == 6
    assert frames[0].reason == "cursor_right"


def test_playing_state_maps_cursor() -> None:
    state = playing_state(frame=3, cursor_x=12, cursor_y=34, scene_id=1)
    assert state.player_x == 12
    assert state.player_y == 34
    assert state.room == 1


def test_score_u16_and_confirmed_target() -> None:
    assert score_u16(232, 3) == CONFIRMED_FIND_POINTS
    confirmed = confirmed_targets()
    labels = {t.label for t in confirmed}
    assert "p2a_primary_1000" in labels
    assert "waldo_pan_right80" in labels
    assert "scene2_scroll_right" in labels
    assert "scene2_waldo_p2a500" in labels
    assert "scene3_scroll_p2a300" in labels
    assert "scene3_waldo_p2a200" in labels
    assert "scene4_scroll_p2a500" in labels
    assert "scene4_waldo_p2a500" in labels
    assert "scene5_scroll_p2a300" in labels
    assert "scene5_waldo_p2a500" in labels
    scroll = next(t for t in confirmed if t.label == "p2a_primary_1000")
    waldo = next(t for t in confirmed if t.label == "waldo_pan_right80")
    s2_scroll = next(t for t in confirmed if t.label == "scene2_scroll_right")
    s2_waldo = next(t for t in confirmed if t.label == "scene2_waldo_p2a500")
    s4_scroll = next(t for t in confirmed if t.label == "scene4_scroll_p2a500")
    s4_waldo = next(t for t in confirmed if t.label == "scene4_waldo_p2a500")
    s5_scroll = next(t for t in confirmed if t.label == "scene5_scroll_p2a300")
    s5_waldo = next(t for t in confirmed if t.label == "scene5_waldo_p2a500")
    assert (scroll.x, scroll.y) == (32, 100)
    assert (waldo.x, waldo.y) == (36, 28)
    assert (s2_scroll.x, s2_scroll.y) == (224, 100)
    assert (s2_waldo.x, s2_waldo.y) == (32, 120)
    assert (s4_scroll.x, s4_scroll.y) == (34, 100)
    assert (s4_waldo.x, s4_waldo.y) == (196, 140)
    assert (s5_scroll.x, s5_scroll.y) == (32, 100)
    assert (s5_waldo.x, s5_waldo.y) == (180, 60)
    assert len(SCENE1_TARGETS) >= 3
    assert TargetStatus.CONFIRMED is TargetStatus.CONFIRMED


@pytest.mark.skipif(
    not (
        __import__("pathlib").Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("custom_integrations/GreatWaldoSearch-Snes/rom.sfc")
        .exists()
    ),
    reason="Waldo ROM not linked",
)
def test_emulator_boot_smoke() -> None:
    """Optional emulator smoke; skipped when ROM/integration missing."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from great_waldo_search.paths import GAME, GAME_DIR
    from retro_harness.env import make_env

    env = make_env(game=GAME, state="NONE", game_dir=GAME_DIR, render_mode="rgb_array")
    try:
        obs, _info = env.reset()
        assert obs.ndim == 3
        obs2, *_rest = env.step([0] * 12)
        assert obs2.shape == obs.shape
    finally:
        env.close()
