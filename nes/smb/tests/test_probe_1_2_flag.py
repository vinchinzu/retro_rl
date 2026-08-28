"""No-ROM gates for the 1-2 UG floor-pipe truth table."""

from __future__ import annotations

import numpy as np

from smb.ram import PLAYER_STATE_AUTO_WALK, PLAYER_STATE_FLAGPOLE
from smb.scripts.probe_1_2_flag import (
    CEILING,
    CONTROL_X_MAX,
    DEATH,
    ENEMY_TYPE_PIRANHA,
    OUTDOOR_FLAG,
    PIPE_B_X,
    PIPE_C_X,
    PLANT_HIDDEN_Y,
    STILL_UG,
    WARP,
    aligned_with_pipe,
    classify_destination,
    is_1_3_control,
    plant_hidden,
    sky_is_overworld,
)
from smb.ram import SmbSnapshot


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
        area_pointer=int(kwargs.get("area_pointer", 194)),
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
        level_number=int(kwargs.get("level_number", kwargs.get("dash_level", 1))),
    )


def test_plant_hidden_when_absent_or_ducked() -> None:
    assert plant_hidden([], PIPE_B_X) is True
    up = [{"type": ENEMY_TYPE_PIRANHA, "x": PIPE_B_X, "y": 120}]
    assert plant_hidden(up, PIPE_B_X) is False
    duck = [{"type": ENEMY_TYPE_PIRANHA, "x": PIPE_B_X, "y": PLANT_HIDDEN_Y}]
    assert plant_hidden(duck, PIPE_B_X) is True
    other = [{"type": ENEMY_TYPE_PIRANHA, "x": PIPE_C_X, "y": 120}]
    assert plant_hidden(other, PIPE_B_X) is True


def test_aligned_with_pipe_slop() -> None:
    assert aligned_with_pipe(PIPE_B_X, PIPE_B_X)
    assert aligned_with_pipe(PIPE_B_X + 12, PIPE_B_X)
    assert not aligned_with_pipe(PIPE_B_X + 40, PIPE_B_X)


def test_classify_halts_on_world0_outdoor_not_warp() -> None:
    assert (
        classify_destination(
            world=0, player_state=PLAYER_STATE_FLAGPOLE, outdoor_sky=False, dying=False
        )
        == OUTDOOR_FLAG
    )
    assert (
        classify_destination(
            world=0, player_state=PLAYER_STATE_AUTO_WALK, outdoor_sky=False, dying=False
        )
        == OUTDOOR_FLAG
    )
    assert (
        classify_destination(world=0, player_state=8, outdoor_sky=True, dying=False)
        == OUTDOOR_FLAG
    )
    assert classify_destination(world=3, player_state=8, outdoor_sky=False, dying=False) == WARP
    assert classify_destination(world=0, player_state=8, outdoor_sky=False, dying=True) == DEATH
    assert (
        classify_destination(
            world=0, player_state=8, outdoor_sky=False, dying=False, ceiling=True
        )
        == CEILING
    )
    assert classify_destination(world=0, player_state=8, outdoor_sky=False, dying=False) == STILL_UG


def test_sky_is_overworld_rejects_ug_black() -> None:
    black = np.zeros((224, 256, 3), dtype=np.uint8)
    assert sky_is_overworld(black) is False
    wipe = np.zeros((224, 256, 3), dtype=np.uint8)
    wipe[:, :, 2] = 180
    wipe[:, :, 1] = 140
    assert sky_is_overworld(wipe) is True  # blue pipe fade still left UG
    sky = np.zeros((224, 256, 3), dtype=np.uint8)
    sky[32:56, :, 2] = 180
    sky[32:56, :, 0] = 90
    sky[32:56, :, 1] = 140
    assert sky_is_overworld(sky) is True
    assert sky_is_overworld(None) is False


def test_ceiling_is_standing_on_y64_not_jump_apex() -> None:
    from smb.scripts.probe_1_2_flag import is_ceiling, is_pipe_transition

    assert is_ceiling(_snap(player_y=50, player_x=2550)) is True
    assert is_ceiling(_snap(player_y=70, player_x=2550)) is True  # grounded default
    assert is_ceiling(_snap(player_y=148, player_x=2520)) is False
    trans = _snap(player_y=0, player_x=0, player_state=0)
    assert is_pipe_transition(trans) is True
    assert is_ceiling(trans) is False
    assert is_pipe_transition(_snap(player_state=2, player_x=2646, player_y=128)) is True


def test_1_3_control_is_dash_2_low_x() -> None:
    good = _snap(world=0, level=2, level_number=2, player_x=40, timer=400, player_state=8)
    assert is_1_3_control(good)
    ug = _snap(world=0, level=2, level_number=1, player_x=40, timer=400)
    assert not is_1_3_control(ug)
    mid = _snap(world=0, level=2, level_number=2, player_x=CONTROL_X_MAX + 1, timer=400)
    assert not is_1_3_control(mid)
    warp = _snap(world=3, level=0, level_number=0, player_x=40, timer=400)
    assert not is_1_3_control(warp)
