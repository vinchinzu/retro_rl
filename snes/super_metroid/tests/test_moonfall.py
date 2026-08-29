"""ROM-free tests for moonwalk / moonfall builders and Climb/Parlor descent."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from super_metroid.ram import (
    ADDR_MOONWALK,
    FACING_LEFT,
    FACING_RIGHT,
    GameplayPhase,
    parse_state,
)
from super_metroid.routes.kpdr.climb_descent import (
    CLIMB_MOONFALL_ON_CLEAN,
    ClimbMoonfallTrack,
    LIP_X,
    climb_moonfall_action,
    climb_moonfall_enabled,
)
from super_metroid.routes.kpdr.parlor_descent import (
    LEDGE_X,
    LIP_X as PARLOR_LIP_X,
    SHAFT_LIP_X,
    PARLOR_MOONFALL_ON_CLEAN,
    ParlorMoonfallTrack,
    parlor_moonfall_action,
    parlor_moonfall_enabled,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PARLOR, ROOM_PIT
from super_metroid.routes.skills.moonfall import (
    MOVEMENT_FALLING,
    MOVEMENT_MOONWALKING,
    initiate_moonfall,
    is_moonfalling,
    is_moonwalking,
    moonwalk_buttons,
    moonwalk_direction,
    require_moonwalk_on,
    uncapped_fall,
)


def _state(**kwargs: Any):
    base = parse_state(np.zeros(0x10000, dtype=np.uint8), frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "game_state": 8,
        "room_id": ROOM_CLIMB,
        "samus_x": 400,
        "samus_y": 80,
        "pose": 1,
        "facing": FACING_LEFT,
        "movement_type": 0,
        "vertical_direction": 0,
        "velocity_y": 0,
        "moonwalk": 1,
    }
    values.update(kwargs)
    return replace(base, **values)


class _FakeSession:
    def __init__(self, state: Any) -> None:
        self.state = state
        self.frame = int(state.frame)
        self.reasons: list[str] = []
        self.button_names: list[tuple[str, ...]] = []

    def step(self, action, reason: str = "") -> Any:
        del action
        self.frame += 1
        self.reasons.append(reason)
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_moonwalk_flag_parses_from_wram() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    ram[ADDR_MOONWALK] = 1
    state = parse_state(ram)
    assert state.moonwalk == 1
    assert state.moonwalk_enabled
    assert not parse_state(np.zeros(0x10000, dtype=np.uint8)).moonwalk_enabled


def test_moonwalk_buttons_are_shot_plus_opposite_facing() -> None:
    assert moonwalk_direction(FACING_LEFT) == "RIGHT"
    assert moonwalk_direction(FACING_RIGHT) == "LEFT"
    assert moonwalk_buttons(FACING_LEFT) == ("RIGHT", "X", "L")
    assert moonwalk_buttons(FACING_RIGHT, aim="UP") == ("LEFT", "X", "R")
    assert moonwalk_buttons(FACING_LEFT, extra=("A",)) == ("RIGHT", "X", "L", "A")


def test_moonfall_detects_airborne_zero_vertical_dir() -> None:
    grounded = _state(movement_type=0, vertical_direction=0)
    assert not is_moonfalling(grounded)
    falling = _state(
        movement_type=MOVEMENT_FALLING,
        vertical_direction=0,
        velocity_y=3,
        samus_y=400,
    )
    assert is_moonfalling(falling)
    assert not uncapped_fall(falling)
    fast = replace(falling, velocity_y=12)
    assert uncapped_fall(fast)
    ordinary_fall = replace(falling, vertical_direction=2, velocity_y=5)
    assert not is_moonfalling(ordinary_fall)


def test_is_moonwalking_uses_movement_type() -> None:
    assert is_moonwalking(_state(movement_type=MOVEMENT_MOONWALKING))
    assert not is_moonwalking(_state(movement_type=1))


def test_require_moonwalk_on_raises_when_flag_off() -> None:
    try:
        require_moonwalk_on(_state(moonwalk=0), label="test")
    except RuntimeError as exc:
        assert "$09E4" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    require_moonwalk_on(_state(moonwalk=1))


def test_initiate_moonfall_emits_wiki_button_order() -> None:
    session = _FakeSession(_state())
    initiate_moonfall(session, walk_frames=3, jump_frames=1, release_frames=1, timeout=0)
    assert any(r.endswith("_moonwalk") for r in session.reasons)
    assert any(r.endswith("_jump") for r in session.reasons)
    assert any(r.endswith("_spin") for r in session.reasons)


def test_climb_policy_buffers_moonfall_while_dropping_in() -> None:
    dropping = _state(
        facing=FACING_LEFT,
        samus_y=50,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=2,
    )
    names, track = climb_moonfall_action(dropping, ClimbMoonfallTrack("plant"))
    assert names == ("X", "L")
    assert "RIGHT" not in names
    assert track.phase == "plant"
    landed = _state(facing=FACING_LEFT, samus_y=80, movement_type=0)
    names, track = climb_moonfall_action(landed, ClimbMoonfallTrack("plant"))
    assert names == ("RIGHT",)
    assert track.phase == "face"


def test_climb_policy_moonwalks_left_to_lip() -> None:
    faced = _state(facing=FACING_RIGHT, samus_x=357, samus_y=91, movement_type=0)
    names, track = climb_moonfall_action(faced, ClimbMoonfallTrack("face", held=2))
    assert names == ("LEFT", "X", "L")
    assert track.phase == "moonwalk"
    walking = _state(
        facing=FACING_LEFT,
        samus_x=355,
        samus_y=91,
        movement_type=MOVEMENT_MOONWALKING,
    )
    names, track = climb_moonfall_action(walking, ClimbMoonfallTrack("moonwalk"))
    assert names == ("LEFT", "X", "L")
    at_lip = _state(
        facing=FACING_LEFT,
        samus_x=LIP_X,
        samus_y=91,
        movement_type=MOVEMENT_MOONWALKING,
    )
    names, track = climb_moonfall_action(at_lip, ClimbMoonfallTrack("moonwalk"))
    assert "A" in names
    assert track.phase == "jump"


def test_climb_policy_falls_left_then_exits_right_to_pit() -> None:
    falling = _state(
        samus_x=300,
        samus_y=900,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=0,
        velocity_y=8,
    )
    names, track = climb_moonfall_action(falling, ClimbMoonfallTrack("fall"))
    assert track.phase == "fall"
    assert names == ("LEFT",)
    bottom = _state(samus_x=300, samus_y=2187, movement_type=0)
    names, track = climb_moonfall_action(bottom, ClimbMoonfallTrack("fall"))
    assert track.phase == "bottom"
    assert "RIGHT" in names
    pit = _state(room_id=ROOM_PIT, samus_x=40, samus_y=139)
    names, track = climb_moonfall_action(pit, ClimbMoonfallTrack("exit"))
    assert track.phase == "done"
    assert names == ()


def test_clean_moonfall_flag_off_until_probe_green() -> None:
    assert CLIMB_MOONFALL_ON_CLEAN is False

    class _S:
        climb_moonfall = True

    assert climb_moonfall_enabled(_S()) is True  # type: ignore[arg-type]

    class _Off:
        pass

    assert climb_moonfall_enabled(_Off()) is False  # type: ignore[arg-type]


def test_parlor_policy_runs_left_then_moonwalks_to_lip() -> None:
    dropping = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_LEFT,
        samus_x=1270,
        samus_y=80,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=2,
    )
    names, track = parlor_moonfall_action(dropping, ParlorMoonfallTrack("plant"))
    assert "LEFT" in names
    assert "RIGHT" not in names
    assert track.phase == "plant"
    door = _state(
        room_id=ROOM_PARLOR,
        game_state=11,
        facing=FACING_LEFT,
        samus_x=19,
        samus_y=1163,
        movement_type=1,
    )
    names, track = parlor_moonfall_action(door, ParlorMoonfallTrack("plant"))
    assert "LEFT" in names
    assert track.phase == "plant"
    landed = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_LEFT,
        samus_x=1200,
        samus_y=139,
        movement_type=0,
    )
    names, track = parlor_moonfall_action(landed, ParlorMoonfallTrack("plant"))
    assert "LEFT" in names
    assert track.phase == "run"
    at_ledge = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_LEFT,
        samus_x=LEDGE_X,
        samus_y=171,
        movement_type=0,
    )
    names, track = parlor_moonfall_action(at_ledge, ParlorMoonfallTrack("run"))
    assert names == ("RIGHT",)
    assert track.phase == "face"
    faced = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_RIGHT,
        samus_x=LEDGE_X,
        samus_y=171,
        movement_type=0,
    )
    names, track = parlor_moonfall_action(faced, ParlorMoonfallTrack("face", held=2))
    assert names == ("LEFT", "X", "L")
    assert track.phase == "moonwalk"
    at_lip = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_LEFT,
        samus_x=PARLOR_LIP_X,
        samus_y=139,
        movement_type=MOVEMENT_MOONWALKING,
    )
    names, track = parlor_moonfall_action(at_lip, ParlorMoonfallTrack("moonwalk"))
    assert "A" in names
    assert track.phase == "jump"
    shaft_lip = _state(
        room_id=ROOM_PARLOR,
        facing=FACING_LEFT,
        samus_x=SHAFT_LIP_X,
        samus_y=171,
        movement_type=MOVEMENT_MOONWALKING,
    )
    names, track = parlor_moonfall_action(shaft_lip, ParlorMoonfallTrack("moonwalk"))
    assert "A" in names
    assert track.phase == "jump"


def test_parlor_policy_falls_then_exits_to_climb() -> None:
    falling = _state(
        room_id=ROOM_PARLOR,
        samus_x=400,
        samus_y=600,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=0,
        velocity_y=8,
    )
    names, track = parlor_moonfall_action(falling, ParlorMoonfallTrack("fall"))
    assert track.phase == "fall"
    bottom = _state(
        room_id=ROOM_PARLOR,
        samus_x=400,
        samus_y=1200,
        movement_type=0,
    )
    names, track = parlor_moonfall_action(bottom, ParlorMoonfallTrack("fall"))
    assert track.phase == "downback"
    assert "L" in names
    climb = _state(room_id=ROOM_CLIMB, samus_x=357, samus_y=49)
    names, track = parlor_moonfall_action(climb, ParlorMoonfallTrack("exit"))
    assert track.phase == "done"
    assert names == ()


def test_parlor_clean_moonfall_flag_off_until_probe_green() -> None:
    assert PARLOR_MOONFALL_ON_CLEAN is False

    class _S:
        parlor_moonfall = True

    assert parlor_moonfall_enabled(_S()) is True  # type: ignore[arg-type]

    class _Off:
        pass

    assert parlor_moonfall_enabled(_Off()) is False  # type: ignore[arg-type]
