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
from super_metroid.routes.kpdr.ceres.outbound import (
    CERES_FIRST_TAS_PAD,
    CeresFallingTrack,
    CeresFirstMoonfallTrack,
    CeresMagnetTrack,
    ceres_falling_magnet_feet_action,
    ceres_first_moonfall_action,
    ceres_magnet_to_scientist_action,
)
from super_metroid.routes.kpdr.crateria.climb_descent import (
    CLIMB_MOONFALL_ON_CLEAN,
    ClimbMoonfallTrack,
    LIP_X,
    climb_moonfall_action,
    climb_moonfall_enabled,
)
from super_metroid.routes.kpdr.crateria.parlor_descent import (
    LEDGE_X,
    LIP_X as PARLOR_LIP_X,
    SHAFT_LIP_X,
    PARLOR_MOONFALL_ON_CLEAN,
    ParlorMoonfallTrack,
    parlor_moonfall_action,
    parlor_moonfall_enabled,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
    ROOM_CLIMB,
    ROOM_PARLOR,
    ROOM_PIT,
)
from super_metroid.routes.skills.moonfall import (
    MOVEMENT_FALLING,
    MOVEMENT_JUMPING,
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


def test_ceres_first_waits_for_pad_then_hops() -> None:
    elev = _state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=128,
        samus_y=20,
        pose=0,
        movement_type=0,
    )
    names, track = ceres_first_moonfall_action(elev, CeresFirstMoonfallTrack("ride"))
    assert names == ()
    assert track.phase == "ride"
    pad = replace(elev, samus_y=72)
    names, track = ceres_first_moonfall_action(pad, CeresFirstMoonfallTrack("ride"))
    assert names == ("B", "RIGHT")
    assert track.phase == "hop"
    assert track.held == 1
    names, track = ceres_first_moonfall_action(pad, CeresFirstMoonfallTrack("hop", held=1))
    assert names == ("B", "Y", "RIGHT", "A")
    assert track.phase == "hop"


def test_ceres_first_air_turns_at_tas_seat() -> None:
    spinning = _state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=142,
        samus_y=79,
        pose=25,
        movement_type=MOVEMENT_JUMPING,
        vertical_direction=1,
    )
    assert len(CERES_FIRST_TAS_PAD) == 150
    assert CERES_FIRST_TAS_PAD[12] == ("B", "LEFT")
    names, track = ceres_first_moonfall_action(
        spinning, CeresFirstMoonfallTrack("hop", held=12)
    )
    assert names == ("B", "LEFT")
    assert track.phase == "air_turn"
    names, track = ceres_first_moonfall_action(
        spinning, CeresFirstMoonfallTrack("air_turn", held=13)
    )
    assert "L" in names and "B" in names
    assert track.phase == "moon_arm"
    names, track = ceres_first_moonfall_action(
        spinning, CeresFirstMoonfallTrack("moon_arm", held=15)
    )
    assert names[0] == "B" and "RIGHT" in names and "X" in names
    names, track = ceres_first_moonfall_action(
        spinning, CeresFirstMoonfallTrack("moon_arm", held=16)
    )
    assert names[0] == "B" and "RIGHT" in names and "A" in names


def test_ceres_first_weaves_then_exits_falling() -> None:
    falling = _state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=160,
        samus_y=200,
        pose=25,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=0,
        velocity_y=8,
    )
    names, track = ceres_first_moonfall_action(
        falling, CeresFirstMoonfallTrack("fall", held=40)
    )
    assert track.phase == "fall"
    assert names == ()
    near_first_platform = _state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=150,
        samus_y=80,
        pose=25,
        movement_type=MOVEMENT_FALLING,
        vertical_direction=0,
        velocity_y=8,
    )
    names, track = ceres_first_moonfall_action(
        near_first_platform, CeresFirstMoonfallTrack("fall", held=25)
    )
    assert "RIGHT" in names
    floor = _state(
        room_id=ROOM_CERES_ELEVATOR,
        samus_x=187,
        samus_y=651,
        pose=9,
        movement_type=0,
    )
    names, track = ceres_first_moonfall_action(
        floor, CeresFirstMoonfallTrack("fall", held=len(CERES_FIRST_TAS_PAD))
    )
    assert track.phase == "land"
    assert "RIGHT" in names
    dest = _state(
        room_id=ROOM_CERES_FALLING,
        game_state=8,
        samus_x=40,
        samus_y=139,
        pose=17,
    )
    names, track = ceres_first_moonfall_action(dest, CeresFirstMoonfallTrack("exit"))
    assert track.phase == "done"
    assert names == ()


def test_ceres_first_idles_door_fade() -> None:
    door = _state(
        room_id=ROOM_CERES_ELEVATOR,
        game_state=9,
        samus_x=237,
        samus_y=651,
        pose=17,
        movement_type=1,
    )
    names, track = ceres_first_moonfall_action(door, CeresFirstMoonfallTrack("land", held=4))
    assert names == ()
    assert track.phase == "exit"
    fade = replace(door, game_state=11, room_id=ROOM_CERES_FALLING, samus_x=39, samus_y=139)
    names, track = ceres_first_moonfall_action(fade, CeresFirstMoonfallTrack("exit"))
    assert names == ()
    assert track.phase == "exit"


def test_ceres_falling_runs_off_entry_then_hops_floor() -> None:
    ledge = _state(
        room_id=ROOM_CERES_FALLING,
        facing=FACING_RIGHT,
        samus_x=39,
        samus_y=139,
        pose=17,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_falling_magnet_feet_action(ledge, CeresFallingTrack())
    assert names == ("RIGHT", "B", "L")
    assert "A" not in names
    names, track = ceres_falling_magnet_feet_action(ledge, track)
    assert names == ("RIGHT", "B", "L")
    floor = replace(ledge, samus_x=155, samus_y=187, pose=9, momentum_x=2)
    names, track = ceres_falling_magnet_feet_action(floor, CeresFallingTrack())
    assert names == ("RIGHT", "B", "A")
    assert "LEFT" not in names
    assert track.phase == "floor_hop"
    air = replace(
        floor,
        samus_x=166,
        samus_y=183,
        pose=25,
        movement_type=MOVEMENT_JUMPING,
        vertical_direction=1,
    )
    names, track = ceres_falling_magnet_feet_action(
        air, CeresFallingTrack(phase="floor_hop", floor_hopped=True, hop_held=1)
    )
    assert names == ("B", "A")
    names, track = ceres_falling_magnet_feet_action(
        air, CeresFallingTrack(phase="floor_hop", floor_hopped=True, hop_held=2)
    )
    assert names == ("LEFT", "RIGHT", "B", "A")
    assert track.phase == "floor_hop"
    names, track = ceres_falling_magnet_feet_action(
        replace(air, samus_y=178),
        CeresFallingTrack(phase="floor_hop", floor_hopped=True, hop_held=3),
    )
    assert names[0] == "RIGHT"
    assert "B" in names
    assert "X" not in names
    assert track.phase == "magnet_feet"


def test_ceres_falling_exit_hop_then_magnet_done() -> None:
    plat = _state(
        room_id=ROOM_CERES_FALLING,
        facing=FACING_RIGHT,
        samus_x=330,
        samus_y=171,
        pose=9,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_falling_magnet_feet_action(
        plat, CeresFallingTrack(floor_hopped=True)
    )
    assert names == ("RIGHT", "B", "A")
    assert track.phase == "exit_hop"
    dest = _state(
        room_id=ROOM_CERES_MAGNET,
        game_state=8,
        samus_x=39,
        samus_y=139,
        pose=9,
    )
    names, track = ceres_falling_magnet_feet_action(dest, CeresFallingTrack(phase="exit"))
    assert names == ()
    assert track.phase == "done"


def test_ceres_magnet_jumps_top_ledge_no_shoulder() -> None:
    seat = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_RIGHT,
        samus_x=39,
        samus_y=139,
        pose=9,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_magnet_to_scientist_action(seat, CeresMagnetTrack())
    assert names[0] == "RIGHT"
    assert "B" in names
    assert "A" not in names
    assert "L" in names or "R" in names
    lip = replace(seat, samus_x=133, pose=9)
    names, track = ceres_magnet_to_scientist_action(lip, CeresMagnetTrack())
    assert names == ("RIGHT", "B", "A")
    assert track.phase == "jump1"


def test_ceres_magnet_jumps_mid_slope_then_scientist_done() -> None:
    slope = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_LEFT,
        samus_x=131,
        samus_y=255,
        pose=10,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_magnet_to_scientist_action(
        slope, CeresMagnetTrack(phase="mid")
    )
    assert names == ("LEFT", "B", "A")
    assert track.phase == "jump2"
    dest = _state(
        room_id=ROOM_CERES_SCIENTIST,
        game_state=8,
        samus_x=39,
        samus_y=139,
        pose=9,
    )
    names, track = ceres_magnet_to_scientist_action(
        dest, CeresMagnetTrack(phase="exit")
    )
    assert names == ()
    assert track.phase == "done"


def test_ceres_magnet_jump1_releases_after_short_spin() -> None:
    air = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_RIGHT,
        samus_x=150,
        samus_y=139,
        pose=25,
        movement_type=MOVEMENT_JUMPING,
        vertical_direction=1,
        momentum_x=2,
    )
    names, track = ceres_magnet_to_scientist_action(
        air, CeresMagnetTrack(phase="jump1", hop_held=3)
    )
    assert names == ("DOWN",)
    assert "A" not in names
    assert track.phase == "drop1"


def test_ceres_magnet_early_slope_does_not_jump2() -> None:
    early = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_LEFT,
        samus_x=151,
        samus_y=235,
        pose=10,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
    )
    names, track = ceres_magnet_to_scientist_action(
        early, CeresMagnetTrack(phase="mid")
    )
    assert "A" not in names
    assert names == ("LEFT", "B")
    assert track.phase == "mid"


def test_ceres_magnet_jump2_releases_to_a_only() -> None:
    falling = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_LEFT,
        samus_x=118,
        samus_y=268,
        pose=26,
        movement_type=MOVEMENT_JUMPING,
        vertical_direction=2,
        momentum_x=2,
    )
    names, track = ceres_magnet_to_scientist_action(
        falling, CeresMagnetTrack(phase="jump2", hop_held=3)
    )
    assert names == ("LEFT",)
    assert "A" not in names
    assert track.phase == "drop2"
    held = replace(falling, samus_y=245, vertical_direction=1)
    names, track = ceres_magnet_to_scientist_action(
        held, CeresMagnetTrack(phase="drop2", hop_held=4)
    )
    assert names == ("A",)
    names, track = ceres_magnet_to_scientist_action(
        held, CeresMagnetTrack(phase="drop2", hop_held=19)
    )
    assert names == ("DOWN", "A")


def test_ceres_magnet_shoots_door_steam() -> None:
    steam = _state(
        room_id=ROOM_CERES_MAGNET,
        facing=FACING_RIGHT,
        samus_x=177,
        samus_y=395,
        pose=9,
        movement_type=1,
        momentum_x=2,
        speed_flag=1,
        enemy0_hp=20,
        enemy0_x=211,
        enemy0_y=395,
    )
    names, track = ceres_magnet_to_scientist_action(steam, CeresMagnetTrack(phase="bot"))
    assert names == ("LEFT", "B", "X")
    assert track.steam_shot is True
    resume, track = ceres_magnet_to_scientist_action(
        steam, CeresMagnetTrack(phase="exit", steam_shot=True)
    )
    assert resume == ("RIGHT", "B")
    assert "X" not in resume


def test_ceres_magnet_idles_fade() -> None:
    fade = _state(
        room_id=ROOM_CERES_MAGNET,
        game_state=11,
        facing=FACING_RIGHT,
        samus_x=236,
        samus_y=395,
        pose=9,
        movement_type=1,
    )
    names, track = ceres_magnet_to_scientist_action(
        fade, CeresMagnetTrack(phase="bot")
    )
    assert names == ()
    assert track.phase == "exit"
