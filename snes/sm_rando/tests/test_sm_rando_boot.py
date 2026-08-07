"""Boot helper unit tests (no emulator required)."""

from __future__ import annotations

from sm_rando.boot import (
    CERES_ELEVATOR_ROOM_ID,
    ORDINARY_GAME_STATE,
    BootResult,
    BootSnapshot,
)
from sm_rando.paths import FIRST_PLAY_STATE, INTEGRATION, SM_SHA1


def test_first_play_constant() -> None:
    assert FIRST_PLAY_STATE == "FirstPlay"
    assert INTEGRATION == "SMRando-Snes"
    assert SM_SHA1 == "da957f0d63d14cb441d215462904c4fa8519c613"


def test_ceres_elevator_room_id() -> None:
    assert CERES_ELEVATOR_ROOM_ID == 0xDF45
    assert ORDINARY_GAME_STATE == 8


def test_boot_snapshot_controllable() -> None:
    snap = BootSnapshot(
        frame=900,
        game_state=8,
        room_id=CERES_ELEVATOR_ROOM_ID,
        door_transition=0,
        health=99,
        samus_x=100,
        samus_y=200,
    )
    assert snap.controllable is True
    d = snap.to_dict()
    assert d["ceres_elevator"] is True
    assert d["room_id_hex"] == "0xDF45"


def test_boot_snapshot_not_controllable_menu() -> None:
    snap = BootSnapshot(
        frame=10,
        game_state=1,
        room_id=0,
        door_transition=0,
        health=0,
        samus_x=0,
        samus_y=0,
    )
    assert snap.controllable is False


def test_boot_result_dict() -> None:
    snap = BootSnapshot(
        frame=1200,
        game_state=8,
        room_id=CERES_ELEVATOR_ROOM_ID,
        door_transition=0,
        health=99,
        samus_x=80,
        samus_y=160,
    )
    result = BootResult(
        ok=True,
        frames=1200,
        room_id=CERES_ELEVATOR_ROOM_ID,
        game_state=8,
        detail="test",
        snapshot=snap,
        state_path="/tmp/FirstPlay.state",
    )
    d = result.to_dict()
    assert d["ok"] is True
    assert d["frames"] == 1200
    assert d["ceres_elevator"] is True
    assert d["game_state"] == 8
    assert d["snapshot"]["controllable"] is True
