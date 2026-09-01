"""Unit tests for Dead Scientist Room outbound (no emulator)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import numpy as np

from super_metroid.ram import FACING_RIGHT, GS_ORDINARY, parse_state
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_SCI_DOOR_Y,
    _CERES_SCI_FLOOR_Y,
)
from super_metroid.routes.kpdr.ceres.scientist import (
    CeresScientistCross,
    play_ceres_scientist_to_flat,
    scientist_on_entry_ledge,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FLAT,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.takeoff import shoulder_pump_button


def _state(**overrides):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8), frame=0)
    values = {
        "room_id": ROOM_CERES_SCIENTIST,
        "game_state": GS_ORDINARY,
        "samus_x": 39,
        "samus_y": _CERES_SCI_DOOR_Y,
        "pose": 17,
        "facing": FACING_RIGHT,
        "momentum_x": 2,
        "speed_flag": 1,
        "samus_x_sub": 100,
    }
    values.update(overrides)
    return replace(base, **values)


def test_entry_ledge_runs_without_jump() -> None:
    st = _state(samus_x=39, samus_y=_CERES_SCI_DOOR_Y)
    assert scientist_on_entry_ledge(st)
    act = CeresScientistCross().action(st)
    assert act == ("RIGHT", "B", shoulder_pump_button(0))
    assert "A" not in act


def test_door_settle_does_not_jump() -> None:
    st = _state(game_state=11, samus_x=39, samus_y=_CERES_SCI_DOOR_Y)
    act = CeresScientistCross().action(st)
    assert act == ("RIGHT",)
    assert "A" not in act


def test_floor_never_jumps() -> None:
    ready = _state(
        samus_x=380,
        samus_y=_CERES_SCI_FLOOR_Y,
        facing=FACING_RIGHT,
        momentum_x=2,
        samus_x_sub=100,
    )
    act = CeresScientistCross().action(ready)
    assert "A" not in act
    assert "RIGHT" in act
    assert "B" in act


def test_arm_pump_alternates() -> None:
    cross = CeresScientistCross()
    st = _state(samus_x=200, samus_y=_CERES_SCI_FLOOR_Y)
    first = cross.action(st)
    second = cross.action(st)
    assert first == ("RIGHT", "B", "L")
    assert second == ("RIGHT", "B", "L")
    third = cross.action(st)
    fourth = cross.action(st)
    assert third == ("RIGHT", "B", "R")
    assert fourth == ("RIGHT", "B", "R")


def test_play_is_noop_when_already_in_flat() -> None:
    session = Mock()
    session.state = _state(room_id=ROOM_CERES_FLAT, samus_x=39, game_state=GS_ORDINARY)
    play_ceres_scientist_to_flat(session)
    session.step.assert_not_called()


def test_door_transition_into_flat_is_not_done() -> None:
    st = _state(room_id=ROOM_CERES_FLAT, game_state=11, samus_x=493)
    from super_metroid.routes.kpdr.ceres.scientist import _scientist_past

    assert not _scientist_past(st)


def test_play_is_noop_when_already_in_ridley() -> None:
    session = Mock()
    session.state = _state(room_id=ROOM_CERES_RIDLEY, samus_x=39)
    play_ceres_scientist_to_flat(session)
    session.step.assert_not_called()


def test_reverse_never_jumps() -> None:
    st = _state(samus_x=472, samus_y=_CERES_SCI_DOOR_Y)
    act = CeresScientistCross("LEFT").action(st)
    assert act == ("LEFT", "B", shoulder_pump_button(0))
    assert "A" not in act


def test_reverse_floor_never_jumps() -> None:
    st = _state(samus_x=250, samus_y=_CERES_SCI_FLOOR_Y)
    act = CeresScientistCross("LEFT").action(st)
    assert "A" not in act
    assert "LEFT" in act
    assert "B" in act


def test_reverse_rejects_bad_direction() -> None:
    import pytest

    with pytest.raises(ValueError):
        CeresScientistCross("UP")


def test_reverse_is_noop_when_already_in_magnet() -> None:
    from super_metroid.routes.kpdr.ceres.scientist import play_ceres_scientist_to_magnet
    from super_metroid.routes.kpdr.room_ids import ROOM_CERES_MAGNET

    session = Mock()
    session.state = _state(room_id=ROOM_CERES_MAGNET, samus_x=216, game_state=GS_ORDINARY)
    play_ceres_scientist_to_magnet(session)
    session.step.assert_not_called()
