"""Unit tests for Morph+Bombs Gauntlet side-quest (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from retro_harness.controls import pressed_snes_buttons

from super_metroid.hop_glance import grade_final
from super_metroid.ram import FACING_LEFT, GameplayPhase, parse_state
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.gauntlet import play_parlor_to_gauntlet
from super_metroid.routes.kpdr.gauntlet.geometry import (
    IBJ_CENTER_X,
    IBJ_FIRST_WAIT,
    IBJ_WAIT2,
    LANDING_TO_GAUNTLET,
    PARLOR_TO_GAUNTLET,
    SHIP_FLOOR_MIN_X,
    at_cliff_lip,
    at_flyway_door,
    at_gauntlet_entry,
    at_gauntlet_ledge,
    at_landing_floor,
    at_parlor_top,
    at_ship_floor,
    in_parlor_shaft,
    is_wall_latch,
)
from super_metroid.routes.kpdr.gauntlet.landing_to_gauntlet import (
    drift_to_bomb_wall,
    ibj_cycle,
    ibj_first_bomb,
)
from super_metroid.routes.kpdr.gauntlet.parlor_to_landing import play_parlor_to_landing
from super_metroid.routes.kpdr.room_ids import (
    ROOM_GAUNTLET_ENTRANCE,
    ROOM_GAUNTLET_ETANK,
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
)


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_PARLOR,
        "samus_x": 968,
        "samus_y": 651,
        "pose": 2,
        "facing": FACING_LEFT,
        "health": 99,
        "max_health": 99,
        "game_state": 8,
        "door_transition": 0,
        "selected_item": 0,
        "collected_items": 0x1004,
        "velocity_y": 0,
        "velocity_x": 0,
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    def __init__(self, state):
        self.state = state
        self.frame = int(state.frame)
        self.actions: list[tuple[object, str]] = []

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_room_ids() -> None:
    assert ROOM_GAUNTLET_ENTRANCE == 0x92B3
    assert ROOM_GAUNTLET_ETANK == 0x965B
    assert ROOM_LANDING_SITE == 0x91F8
    assert ROOM_PARLOR == 0x92FD


def test_flyway_and_shaft_bands() -> None:
    pin = _state()
    assert at_flyway_door(pin)
    assert in_parlor_shaft(pin)
    assert not at_parlor_top(pin)
    top = _state(samus_x=1000, samus_y=120)
    assert at_parlor_top(top)
    assert not at_flyway_door(_state(samus_x=200, samus_y=651))


def test_landing_ledge_and_entry_bands() -> None:
    floor = _state(room_id=ROOM_LANDING_SITE, samus_x=80, samus_y=1163, pose=1)
    assert at_landing_floor(floor)
    assert not at_ship_floor(floor)
    ship = _state(room_id=ROOM_LANDING_SITE, samus_x=708, samus_y=1173, pose=1)
    assert at_ship_floor(ship)
    assert ship.samus_x >= SHIP_FLOOR_MIN_X
    ledge = _state(room_id=ROOM_LANDING_SITE, samus_x=520, samus_y=720, pose=1)
    assert at_gauntlet_ledge(ledge)
    assert not at_gauntlet_ledge(floor)
    lip = _state(room_id=ROOM_LANDING_SITE, samus_x=613, samus_y=801, pose=2)
    assert at_cliff_lip(lip)
    assert at_gauntlet_ledge(lip)
    entry = _state(
        room_id=ROOM_GAUNTLET_ENTRANCE,
        samus_x=1200,
        samus_y=140,
        pose=1,
        game_state=8,
        door_transition=0,
    )
    assert at_gauntlet_entry(entry)
    assert not at_gauntlet_entry(_state(room_id=ROOM_GAUNTLET_ENTRANCE, game_state=11))


def test_leave_spec_grades_gauntlet_entry() -> None:
    good = {
        "room": "0x92B3",
        "xy": [1200, 140],
        "pose": 1,
        "gs": 8,
        "dt": 0,
        "health": 99,
    }
    assert grade_final(good, PARLOR_TO_GAUNTLET) == []
    assert grade_final(good, LANDING_TO_GAUNTLET) == []
    miss = dict(good, room="0x91F8")
    assert grade_final(miss, PARLOR_TO_GAUNTLET)


def test_ibj_cycle_lays_bombs_and_recenters() -> None:
    # Morph pose 0x1D = 29. Start right of center so cycle taps LEFT then X.
    session = _Session(
        _state(
            room_id=ROOM_LANDING_SITE,
            samus_x=IBJ_CENTER_X + 100,
            samus_y=1163,
            pose=29,
        )
    )
    done = ibj_cycle(session, stop_y=0)
    assert done is False
    names = [pressed_snes_buttons(act) for act, _ in session.actions]
    reasons = [reason for _, reason in session.actions]
    assert any("X" in n for n in names)
    assert any(r.endswith("_b1") for r in reasons)
    assert any(r.endswith("_b2") for r in reasons)
    assert any(r.endswith("_cL") for r in reasons)
    assert IBJ_WAIT2 == 30


def test_ibj_first_bomb_waits_from_rest() -> None:
    session = _Session(
        _state(room_id=ROOM_LANDING_SITE, samus_x=800, samus_y=1171, pose=29)
    )
    done = ibj_first_bomb(session, stop_y=0)
    assert done is False
    reasons = [reason for _, reason in session.actions]
    assert any(r.endswith("_b0") for r in reasons)
    assert sum(1 for r in reasons if r.endswith("_w0")) == IBJ_FIRST_WAIT


def test_peak_to_obstacle_a_preserves_pin_chain_seats() -> None:
    session = _Session(
        _state(
            room_id=ROOM_LANDING_SITE,
            samus_x=867,
            samus_y=519,
            pose=49,
        )
    )
    drift_to_bomb_wall(session)
    reasons = [reason for _, reason in session.actions]
    assert sum(r.endswith("_b1") for r in reasons) == 74
    assert sum(r.endswith("_b2") for r in reasons) == 74
    assert reasons.count("landing_ibj_cliff_seat") == 50
    assert reasons.count("landing_ibj_face_seat") == 30
    assert reasons.count("landing_ibj_a_seat") == 40
    assert reasons.count("landing_ibj_dL4_L") == 40
    assert reasons.count("landing_ibj_dL12_L") == 72


def test_wall_latch_pose() -> None:
    assert is_wall_latch(POSE_WALL_LATCH)
    assert not is_wall_latch(1)


def test_wrong_room_raises() -> None:
    session = _Session(_state(room_id=ROOM_LANDING_SITE, samus_x=80, samus_y=1163))
    with pytest.raises(RuntimeError, match="parlor_to_landing"):
        play_parlor_to_landing(session)


def test_play_parlor_to_gauntlet_exported() -> None:
    assert callable(play_parlor_to_gauntlet)
