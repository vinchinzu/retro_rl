"""Unit tests for reactive 1-2 policy (no emulator)."""

from __future__ import annotations

from smb.policy import expand_nes9_rle
from smb.ram import SmbSnapshot
from smb.reactive_12 import (
    DEFAULT_FRAGMENTS,
    Phase,
    Reactive12Policy,
    is_surface_control,
    is_underground_control,
    load_reactive_fragments,
    underground_frames,
)


def _snap(**kwargs) -> SmbSnapshot:
    base = dict(
        frame=0,
        player_state=8,
        player_x=40,
        player_y=176,
        x_page=0,
        x_offset=40,
        lives=2,
        world=0,
        level=1,
        level_id=1,
        oper_mode=1,
        player_power=0,
        timer_hundreds=0,
        timer=0,
        area_pointer=41,
        x_speed=0,
        y_speed=0,
        facing=1,
        screen_x=0,
        player_screen_x=40,
        in_air=False,
    )
    base.update(kwargs)
    return SmbSnapshot(**base)


def test_fragments_file_exists_and_loads() -> None:
    assert DEFAULT_FRAGMENTS.exists()
    data = load_reactive_fragments()
    ug = underground_frames()
    assert data["underground_from_control"]["num_frames"] == len(ug)
    assert len(ug) == 1545
    assert all(len(f) == 9 for f in ug[:5])


def test_surface_control_gate() -> None:
    assert is_surface_control(_snap())
    assert not is_surface_control(_snap(level=0))
    assert not is_surface_control(_snap(player_x=200))
    assert not is_surface_control(_snap(area_pointer=192))


def test_underground_control_gate() -> None:
    ug = _snap(level=2, level_id=2, timer=401, timer_hundreds=4, area_pointer=192)
    assert is_underground_control(ug)
    assert not is_underground_control(_snap(level=2, timer=0))


def test_policy_waits_then_moves_right() -> None:
    pol = Reactive12Policy()
    # Not yet on surface control (still 1-1 castle walk)
    mid = _snap(level=0, level_id=0, area_pointer=194, player_x=3000)
    a = pol.step(mid)
    assert pol.phase is Phase.WAIT_SURFACE
    assert int(a.action.sum()) == 0

    # Surface control → RIGHT
    a = pol.step(_snap())
    assert pol.phase is Phase.SURFACE
    assert int(a.action[7]) == 1  # RIGHT


def test_policy_down_on_pipe() -> None:
    pol = Reactive12Policy()
    pol.step(_snap())  # enter surface
    a = pol.step(_snap(player_x=160, x_offset=160))
    assert int(a.action[5]) == 1  # DOWN


def test_policy_replays_underground_after_gate() -> None:
    pol = Reactive12Policy()
    # Skip to underground wait
    pol.phase = Phase.WAIT_UNDERGROUND
    ug = _snap(level=2, level_id=2, timer=401, timer_hundreds=4, area_pointer=192, player_y=32)
    a = pol.step(ug)
    assert pol.phase is Phase.UNDERGROUND
    # First underground macro frame from fragments (often idle/right)
    assert a.action.shape[0] >= 9
    assert pol.ug_index == 1
