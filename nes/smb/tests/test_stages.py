"""Pure unit tests for StageSpec table + control/goal predicates (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smb.ram import (
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_OPER_MODE,
    ADDR_WORLD,
    OPER_MODE_END,
    WORLD_INDEX_4,
    WORLD_INDEX_8,
)
from smb.tas.stages import (
    HL_1_2_FM2_START,
    HL_1_2_W4_FRAMES,
    HL_4_1_FM2_START,
    HL_8_1_FM2_START,
    STAGE_1_2,
    STAGES,
    GoalKind,
    StageSpec,
    get_stage,
    goal_hit,
    is_4_1_control,
    is_8_3_control,
)


def _snap(**kw: int | bool) -> SimpleNamespace:
    defaults: dict[str, int | bool] = {
        "world": 3,
        "level": 0,
        "oper_mode": 1,
        "player_state": 7,
        "dying": False,
        "timer": 401,
        "player_x": 40,
        "player_y": 176,
        "lives": 2,
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_stages_keys_complete() -> None:
    expected = {"1-2", "4-1", "4-2", "8-1", "8-2", "8-3", "8-4"}
    assert set(STAGES) == expected
    for sid, stage in STAGES.items():
        assert isinstance(stage, StageSpec)
        assert stage.id == sid
        assert callable(stage.control)
        assert isinstance(stage.goal, GoalKind)
        assert stage.seed_name.endswith(".json")
        assert stage.body_frames >= 0


def test_get_stage_lookup() -> None:
    s12 = get_stage("1-2")
    assert s12 is STAGE_1_2
    assert s12.fm2_start == HL_1_2_FM2_START == 2109
    assert s12.body_frames == HL_1_2_W4_FRAMES
    assert s12.goal is GoalKind.WORLD
    assert s12.goal_world == WORLD_INDEX_4

    s81 = get_stage("8-1")
    assert s81.id == "8-1"
    assert s81.fm2_start == HL_8_1_FM2_START
    assert s81.goal is GoalKind.LEVEL
    assert s81.goal_world == WORLD_INDEX_8
    assert s81.goal_level == 1

    # normalize underscore / case
    assert get_stage("4_1").id == "4-1"
    assert get_stage(" 8-2 ").id == "8-2"

    with pytest.raises(KeyError, match="unknown stage"):
        get_stage("9-9")


def test_stage_1_2_constants_match_table() -> None:
    """STAGE_1_2 fm2 starts match module-level constants used by exporters."""
    assert STAGE_1_2.fm2_start == HL_1_2_FM2_START
    assert STAGE_1_2.body_frames == HL_1_2_W4_FRAMES
    assert STAGES["4-1"].fm2_start == HL_4_1_FM2_START


def test_is_4_1_control() -> None:
    assert is_4_1_control(_snap())
    assert not is_4_1_control(_snap(timer=0))
    assert not is_4_1_control(_snap(level=1))
    assert not is_4_1_control(_snap(player_x=250))
    assert not is_4_1_control(_snap(dying=True))


def test_is_8_3_control() -> None:
    assert is_8_3_control(_snap(world=WORLD_INDEX_8, level=2, player_x=40))
    assert not is_8_3_control(_snap(world=WORLD_INDEX_8, level=3, player_x=40))
    assert not is_8_3_control(_snap(world=WORLD_INDEX_8, level=2, player_x=200))
    assert not is_8_3_control(_snap(world=3, level=2, player_x=40))


def test_goal_hit_world_level_ending() -> None:
    snap_w4 = _snap(world=WORLD_INDEX_4, level=0)
    assert goal_hit(
        GoalKind.WORLD,
        snap=snap_w4,
        ram=None,
        key=(WORLD_INDEX_4, 0),
        goal_world=WORLD_INDEX_4,
        goal_level=None,
        start_lives=2,
    )
    assert not goal_hit(
        GoalKind.WORLD,
        snap=_snap(world=0),
        ram=None,
        key=(0, 0),
        goal_world=WORLD_INDEX_4,
        goal_level=None,
        start_lives=2,
    )

    snap_82 = _snap(world=WORLD_INDEX_8, level=1)
    assert goal_hit(
        GoalKind.LEVEL,
        snap=snap_82,
        ram=None,
        key=(WORLD_INDEX_8, 1),
        goal_world=WORLD_INDEX_8,
        goal_level=1,
        start_lives=2,
    )
    assert not goal_hit(
        GoalKind.LEVEL,
        snap=snap_82,
        ram=None,
        key=(WORLD_INDEX_8, 0),
        goal_world=WORLD_INDEX_8,
        goal_level=1,
        start_lives=2,
    )

    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_WORLD] = WORLD_INDEX_8
    ram[ADDR_LEVEL] = 3
    ram[ADDR_OPER_MODE] = OPER_MODE_END
    ram[ADDR_LIVES] = 2
    assert goal_hit(
        GoalKind.ENDING,
        snap=_snap(world=WORLD_INDEX_8, level=3),
        ram=ram,
        key=(WORLD_INDEX_8, 3),
        goal_world=None,
        goal_level=None,
        start_lives=2,
    )
    # lives drop → not ending success
    ram[ADDR_LIVES] = 1
    assert not goal_hit(
        GoalKind.ENDING,
        snap=_snap(world=WORLD_INDEX_8, level=3),
        ram=ram,
        key=(WORLD_INDEX_8, 3),
        goal_world=None,
        goal_level=None,
        start_lives=2,
    )
