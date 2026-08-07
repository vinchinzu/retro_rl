"""Unit tests for door leave/entry kinematics (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.door_kinematics import (
    DoorKinematics,
    DoorKinematicsRequirement,
    SpeedBand,
    classify_speed_band,
    require_door_kinematics,
)
from super_metroid.policy import StateRequirement
from super_metroid.progression import ObservedTransition
from super_metroid.ram import (
    ADDR_MOMENTUM_X,
    ADDR_SAMUS_FACING,
    ADDR_SAMUS_X,
    ADDR_SAMUS_X_SUB,
    ADDR_SAMUS_Y,
    ADDR_SPEED_COUNTER,
    ADDR_VELOCITY_X,
    ADDR_VELOCITY_Y,
    FACING_LEFT,
    FACING_RIGHT,
    parse_state,
)
from super_metroid.rooms.segment_contract import EntryContract


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_parse_state_reads_door_kinematics_fields() -> None:
    ram = np.zeros(0x10000, dtype=np.uint8)
    _put_u16(ram, ADDR_SAMUS_X, 400)
    _put_u16(ram, ADDR_SAMUS_X_SUB, 0x8000)
    _put_u16(ram, ADDR_SAMUS_Y, 180)
    _put_u16(ram, ADDR_VELOCITY_X, 5)
    _put_u16(ram, ADDR_VELOCITY_Y, 2)
    _put_u16(ram, ADDR_MOMENTUM_X, 3)
    # Speed counter word: hi=4 (boost), lo=0x12 anim
    _put_u16(ram, ADDR_SPEED_COUNTER, 0x0412)
    ram[ADDR_SAMUS_FACING] = FACING_RIGHT

    state = parse_state(ram, frame=10)

    assert state.samus_x == 400
    assert state.samus_x_sub == 0x8000
    assert state.velocity_x == 5
    assert state.momentum_x == 3
    assert state.speed_counter == 4
    assert state.speed_boosting
    assert state.facing_right
    assert not state.facing_left
    kin = DoorKinematics.from_state(state)
    assert kin.speed_band is SpeedBand.SPEED_BOOST
    assert kin.to_dict()["speed_boosting"] is True


def test_classify_speed_bands() -> None:
    assert classify_speed_band(velocity_x=0, speed_counter=0) is SpeedBand.STATIONARY
    assert classify_speed_band(velocity_x=2, speed_counter=0) is SpeedBand.WALK
    assert classify_speed_band(velocity_x=4, speed_counter=0) is SpeedBand.RUN
    assert classify_speed_band(velocity_x=1, speed_counter=4) is SpeedBand.SPEED_BOOST
    assert (
        classify_speed_band(velocity_x=0, speed_counter=0, shinespark_timer=30)
        is SpeedBand.SHINESPARK
    )


def test_door_kinematics_requirement_speed_and_position() -> None:
    state = replace(
        parse_state(np.zeros(0x10000, dtype=np.uint8)),
        samus_x=120,
        samus_y=200,
        velocity_x=5,
        speed_counter=4,
        facing=FACING_RIGHT,
        pose=9,
    )
    req = DoorKinematicsRequirement(
        x_range=(100, 140),
        y_range=(190, 210),
        velocity_x_range=(3, 8),
        speed_counter_min=4,
        require_speed_boost=True,
        facings=frozenset({FACING_RIGHT}),
        speed_bands=frozenset({SpeedBand.SPEED_BOOST}),
    )
    assert req.matches(state)
    assert not req.failures(state)

    bad = replace(state, speed_counter=1, velocity_x=0)
    fails = req.failures(bad)
    assert any("speed_counter" in f for f in fails)
    assert any("speed_boost" in f or "speed_band" in f for f in fails)

    with pytest.raises(RuntimeError, match="door kinematics mismatch"):
        require_door_kinematics(bad, req, label="unit")


def test_state_requirement_kinematics_fields() -> None:
    state = replace(
        parse_state(np.zeros(0x10000, dtype=np.uint8)),
        room_id=0xA7DE,
        velocity_x=6,
        speed_counter=4,
        facing=FACING_LEFT,
    )
    req = StateRequirement(
        room_id=0xA7DE,
        velocity_x_range=(5, 8),
        speed_counter_min=4,
        require_speed_boost=True,
        facings=frozenset({FACING_LEFT}),
    )
    assert req.matches(state)
    assert not StateRequirement(
        require_speed_boost=False
    ).matches(state)


def test_observed_transition_carries_leave_entry_kinematics() -> None:
    leave = {
        "frame": 10,
        "room_id": 0xA6A1,
        "samus_x": 500,
        "velocity_x": 5,
        "speed_counter": 4,
        "speed_band": "speed_boost",
    }
    entry = {
        "frame": 40,
        "room_id": 0xA7DE,
        "samus_x": 32,
        "velocity_x": 5,
        "speed_counter": 4,
    }
    hop = ObservedTransition(
        frame=40,
        source_room_id=0xA6A1,
        target_room_id=0xA7DE,
        edge_id="warehouse_to_business",
        leave_kinematics=leave,
        entry_kinematics=entry,
    )
    payload = hop.to_dict()
    assert payload["leave_kinematics"]["speed_counter"] == 4
    assert payload["entry_kinematics"]["samus_x"] == 32
    # Positional construction still works without kinematics.
    bare = ObservedTransition(100, 0xA6A1, 0xA7DE, "warehouse_to_business")
    assert bare.leave_kinematics is None
    assert "leave_kinematics" not in bare.to_dict()


def test_entry_contract_roundtrip_kinematics() -> None:
    kin = DoorKinematicsRequirement(
        x_range=(20, 40),
        speed_counter_min=0,
        speed_counter_max=1,
        require_speed_boost=False,
    )
    contract = EntryContract(
        door_ptr=0x1234,
        entry_source_room_id=0xA6A1,
        door_orientation="left",
        spawn_x=32,
        spawn_y=180,
        spawn_pose=1,
        entry_kinematics=kin,
        leave_kinematics=DoorKinematicsRequirement(
            velocity_x_range=(3, 8),
            speed_bands=frozenset({SpeedBand.RUN, SpeedBand.SPEED_BOOST}),
        ),
    )
    raw = contract.to_dict()
    assert "entryKinematics" in raw
    assert "leaveKinematics" in raw
    restored = EntryContract.from_dict(raw)
    assert restored is not None
    assert restored.entry_kinematics is not None
    assert restored.entry_kinematics.x_range == (20, 40)
    assert restored.leave_kinematics is not None
    assert SpeedBand.RUN in restored.leave_kinematics.speed_bands


def test_door_kinematics_from_mapping_roundtrip() -> None:
    state = replace(
        parse_state(np.zeros(0x10000, dtype=np.uint8)),
        frame=7,
        room_id=0xB07A,
        samus_x=90,
        samus_y=400,
        velocity_x=-4,
        speed_counter=2,
        facing=FACING_LEFT,
    )
    snap = DoorKinematics.from_state(state)
    again = DoorKinematics.from_mapping(snap.to_dict())
    assert again.samus_x == 90
    assert again.velocity_x == -4
    assert again.speed_band is SpeedBand.RUN
