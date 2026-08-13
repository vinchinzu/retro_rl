"""Tests for physics_sim trajectory predictor protocol.

Pure offline tests with StubPredictor — no ROM required.
Validates protocol contract and data structure serialization.
"""

from __future__ import annotations

import json

import pytest

from super_metroid.physics_sim import (
    FrameInput,
    PhysicsPredictor,
    SimState,
    SmRevClient,
    StubPredictor,
    Trajectory,
    TrajectoryFrame,
    decode_frame_mnemonic,
    encode_frame_mnemonic,
    load_predictor,
)


class TestSimState:
    """Test SimState dataclass and conversions."""

    def test_create_minimal(self) -> None:
        state = SimState(
            frame=100,
            room_id=0x91F8,
            samus_x=400,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        assert state.frame == 100
        assert state.room_id == 0x91F8
        assert state.samus_x == 400
        assert state.samus_y == 200

    def test_to_dict_roundtrip(self) -> None:
        state = SimState(
            frame=100,
            room_id=0x91F8,
            samus_x=400,
            samus_y=200,
            samus_x_sub=32768,
            samus_y_sub=16384,
            velocity_x=2,
            velocity_y=-3,
            velocity_x_sub=1000,
            velocity_y_sub=2000,
            momentum_x=5,
            momentum_x_sub=8192,
            pose=1,
            facing=8,
            movement_type=1,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        data = state.to_dict()
        restored = SimState.from_dict(data)
        assert restored == state

    def test_json_serializable(self) -> None:
        state = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=100,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        # Should not raise
        json_str = json.dumps(state.to_dict())
        data = json.loads(json_str)
        restored = SimState.from_dict(data)
        assert restored == state


class TestFrameInput:
    """Test FrameInput dataclass."""

    def test_create(self) -> None:
        inp = FrameInput(buttons=0x40)
        assert inp.buttons == 0x40

    def test_to_dict_roundtrip(self) -> None:
        inp = FrameInput(buttons=0x41)
        data = inp.to_dict()
        restored = FrameInput.from_dict(data)
        assert restored == inp


class TestTrajectoryFrame:
    """Test TrajectoryFrame dataclass."""

    def test_create(self) -> None:
        frame = TrajectoryFrame(
            frame=10,
            room_id=0x91F8,
            samus_x=450,
            samus_y=180,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=2,
            velocity_y=-1,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=2,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=1,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        assert frame.frame == 10
        assert frame.room_id == 0x91F8
        assert frame.samus_x == 450
        assert frame.velocity_x == 2

    def test_to_dict_roundtrip(self) -> None:
        frame = TrajectoryFrame(
            frame=10,
            room_id=0x91F8,
            samus_x=450,
            samus_y=180,
            samus_x_sub=1000,
            samus_y_sub=2000,
            velocity_x=2,
            velocity_y=-1,
            velocity_x_sub=500,
            velocity_y_sub=1500,
            momentum_x=2,
            momentum_x_sub=100,
            pose=0,
            facing=8,
            movement_type=1,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        data = frame.to_dict()
        restored = TrajectoryFrame.from_dict(data)
        assert restored == frame


class TestTrajectory:
    """Test Trajectory dataclass."""

    def test_create_empty(self) -> None:
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=100,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        traj = Trajectory(start=start, frames=(), predictor="test")
        assert traj.start == start
        assert len(traj.frames) == 0
        assert traj.predictor == "test"

    def test_to_dict_roundtrip(self) -> None:
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=100,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        frames = (
            TrajectoryFrame(
                frame=1,
                room_id=0x91F8,
                samus_x=102,
                samus_y=100,
                samus_x_sub=0,
                samus_y_sub=0,
                velocity_x=2,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=2,
                momentum_x_sub=0,
                pose=0,
                facing=8,
                movement_type=0,
                speed_counter=0,
                speed_flag=0,
                shinespark_timer=0,
            ),
        )
        traj = Trajectory(start=start, frames=frames, predictor="test")
        data = traj.to_dict()
        restored = Trajectory.from_dict(data)
        assert restored.start == traj.start
        assert restored.frames == traj.frames
        assert restored.predictor == traj.predictor


class TestStubPredictor:
    """Test StubPredictor implementation."""

    def test_protocol_compliance(self) -> None:
        pred = StubPredictor()
        assert isinstance(pred, PhysicsPredictor)

    def test_name(self) -> None:
        pred = StubPredictor(name="test-stub")
        assert pred.name() == "test-stub"

    def test_predict_empty_inputs(self) -> None:
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=100,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        traj = pred.predict(start, [])
        assert traj.start == start
        assert len(traj.frames) == 0
        assert traj.predictor == "stub"

    def test_predict_right_movement(self) -> None:
        """RIGHT (0x80) should move +x."""
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]  # Hold RIGHT
        traj = pred.predict(start, inputs)

        assert len(traj.frames) == 10
        # Should move right (fake physics)
        assert traj.frames[-1].samus_x > start.samus_x
        # Frame numbers should increment
        assert traj.frames[0].frame == 1
        assert traj.frames[-1].frame == 10
        # Room ID preserved
        assert all(f.room_id == 0x91F8 for f in traj.frames)

    def test_predict_left_movement(self) -> None:
        """LEFT (0x40) should move -x."""
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x40) for _ in range(10)]  # Hold LEFT
        traj = pred.predict(start, inputs)

        assert len(traj.frames) == 10
        # Should move left (fake physics)
        assert traj.frames[-1].samus_x < start.samus_x
        # Room ID preserved
        assert all(f.room_id == 0x91F8 for f in traj.frames)

    def test_predict_deterministic(self) -> None:
        """Same inputs should produce same trajectory."""
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x40) for _ in range(5)]

        traj1 = pred.predict(start, inputs)
        traj2 = pred.predict(start, inputs)

        assert len(traj1.frames) == len(traj2.frames)
        for f1, f2 in zip(traj1.frames, traj2.frames):
            assert f1 == f2

    def test_predict_preserves_start(self) -> None:
        pred = StubPredictor()
        start = SimState(
            frame=42,
            room_id=0xA000,
            samus_x=123,
            samus_y=456,
            samus_x_sub=789,
            samus_y_sub=1011,
            velocity_x=5,
            velocity_y=-3,
            velocity_x_sub=100,
            velocity_y_sub=200,
            momentum_x=5,
            momentum_x_sub=100,
            pose=1,
            facing=4,
            movement_type=1,
            speed_counter=2,
            speed_flag=1,
            shinespark_timer=30,
        )
        inputs = [FrameInput(buttons=0) for _ in range(3)]
        traj = pred.predict(start, inputs)

        assert traj.start == start


class TestSmRevClient:
    """Test SmRevClient stub (graceful unavailable handling)."""

    def test_protocol_compliance(self) -> None:
        client = SmRevClient()
        assert isinstance(client, PhysicsPredictor)

    def test_name(self) -> None:
        client = SmRevClient(binary_path="/path/to/sm_rev")
        assert "sm_rev" in client.name()

    def test_predict_unavailable(self) -> None:
        """Should raise when sm_rev binary is not available."""
        client = SmRevClient(binary_path="/nonexistent/sm_rev")
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=100,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0)]

        with pytest.raises(RuntimeError, match="not available"):
            client.predict(start, inputs)


class TestLoadPredictor:
    """Test load_predictor factory."""

    def test_load_stub(self) -> None:
        pred = load_predictor("stub")
        assert isinstance(pred, StubPredictor)
        assert pred.name() == "stub"

    def test_load_stub_with_name(self) -> None:
        pred = load_predictor("stub", name="custom")
        assert isinstance(pred, StubPredictor)
        assert pred.name() == "custom"

    def test_load_sm_rev(self) -> None:
        pred = load_predictor("sm_rev")
        assert isinstance(pred, SmRevClient)

    def test_load_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown predictor backend"):
            load_predictor("nonexistent")


class TestFrameMnemonic:
    """Test frame mnemonic encoding/decoding."""

    def test_encode_all_released(self) -> None:
        assert encode_frame_mnemonic(0) == "............"

    def test_encode_b_only(self) -> None:
        assert encode_frame_mnemonic(0x01) == "b..........."

    def test_encode_left(self) -> None:
        # LEFT = bit 6 = 0x40
        assert encode_frame_mnemonic(0x40) == "......l....."

    def test_encode_right(self) -> None:
        # RIGHT = bit 7 = 0x80
        assert encode_frame_mnemonic(0x80) == ".......r...."

    def test_encode_b_and_right(self) -> None:
        # B (0x01) + RIGHT (0x80) = 0x81
        assert encode_frame_mnemonic(0x81) == "b......r...."

    def test_encode_all_buttons(self) -> None:
        # All 12 buttons pressed
        assert encode_frame_mnemonic(0xFFF) == "bysSSudlraxLR"

    def test_decode_all_released(self) -> None:
        assert decode_frame_mnemonic("............") == 0

    def test_decode_b_only(self) -> None:
        assert decode_frame_mnemonic("b...........") == 0x01

    def test_decode_left(self) -> None:
        assert decode_frame_mnemonic("......l.....") == 0x40

    def test_decode_right(self) -> None:
        assert decode_frame_mnemonic(".......r....") == 0x80

    def test_decode_b_and_right(self) -> None:
        assert decode_frame_mnemonic("b......r....") == 0x81

    def test_roundtrip(self) -> None:
        for buttons in [0, 0x01, 0x40, 0x80, 0x81, 0xFFF]:
            mnemonic = encode_frame_mnemonic(buttons)
            decoded = decode_frame_mnemonic(mnemonic)
            assert decoded == buttons

    def test_decode_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="must be 12 chars"):
            decode_frame_mnemonic("short")


class TestSmeditTasExport:
    """Test smedit-tas-1 export format."""

    def test_export_basic_structure(self) -> None:
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80), FrameInput(buttons=0x81)]
        traj = pred.predict(start, inputs)

        result = traj.to_smedit_tas(start_state_name="TestStart")

        assert result["format"] == "smedit-tas-1"
        assert result["meta"]["gameName"] == "SuperMetroid-Snes"
        assert result["meta"]["startState"] == "TestStart"
        assert result["meta"]["romSha1"] is None
        assert result["buttonOrder"] == [
            "B",
            "Y",
            "Select",
            "Start",
            "Up",
            "Down",
            "Left",
            "Right",
            "A",
            "X",
            "L",
            "R",
        ]
        assert len(result["frames"]) == 2
        assert result["frames"][0] == ".......r...."  # RIGHT only
        assert result["frames"][1] == "b......r...."  # B + RIGHT
        assert len(result["trace"]) == 2

    def test_export_trace_fields(self) -> None:
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80)]
        traj = pred.predict(start, inputs)

        result = traj.to_smedit_tas()

        trace = result["trace"]
        assert len(trace) == 1
        assert trace[0]["frame"] == 1
        assert trace[0]["x"] == 102  # Moved right by 2
        assert trace[0]["y"] == 200
        assert trace[0]["roomId"] == 0x91F8

    def test_export_trace_stride(self) -> None:
        pred = StubPredictor()
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]
        traj = pred.predict(start, inputs)

        result = traj.to_smedit_tas(trace_stride=5)

        # Should have frames 0 and 5 (indices 0, 5 with stride 5)
        assert len(result["trace"]) == 2
        assert result["trace"][0]["frame"] == 1
        assert result["trace"][1]["frame"] == 6


class TestGoldenFixtures:
    """Test against golden request/response fixtures."""

    def test_golden_internal_format(self) -> None:
        """Golden fixture for internal Trajectory.to_dict() format."""
        pred = StubPredictor(name="test-stub")
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80), FrameInput(buttons=0x81)]
        traj = pred.predict(start, inputs)

        result = traj.to_dict()

        # Verify structure
        assert result["predictor"] == "test-stub"
        assert "start" in result
        assert "frames" in result
        assert "inputs" in result

        # Verify start state
        assert result["start"]["room_id"] == 0x91F8
        assert result["start"]["samus_x"] == 100
        assert result["start"]["samus_y"] == 200

        # Verify frames
        assert len(result["frames"]) == 2
        assert result["frames"][0]["frame"] == 1
        assert result["frames"][0]["room_id"] == 0x91F8
        assert result["frames"][0]["samus_x"] == 102  # Moved right
        assert result["frames"][1]["frame"] == 2
        assert result["frames"][1]["samus_x"] == 104

        # Verify inputs
        assert len(result["inputs"]) == 2
        assert result["inputs"][0]["buttons"] == 0x80
        assert result["inputs"][1]["buttons"] == 0x81

    def test_golden_smedit_tas_format(self) -> None:
        """Golden fixture for smedit-tas-1 format."""
        pred = StubPredictor(name="test-stub")
        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        inputs = [FrameInput(buttons=0x80), FrameInput(buttons=0x81)]
        traj = pred.predict(start, inputs)

        result = traj.to_smedit_tas(start_state_name="LandingSite", rom_sha1=None)

        # Verify exact golden structure
        expected = {
            "format": "smedit-tas-1",
            "meta": {
                "gameName": "SuperMetroid-Snes",
                "startState": "LandingSite",
                "romSha1": None,
            },
            "buttonOrder": [
                "B",
                "Y",
                "Select",
                "Start",
                "Up",
                "Down",
                "Left",
                "Right",
                "A",
                "X",
                "L",
                "R",
            ],
            "frames": [".......r....", "b......r...."],
            "trace": [
                {"frame": 1, "x": 102, "y": 200, "roomId": 37368},
                {"frame": 2, "x": 104, "y": 200, "roomId": 37368},
            ],
        }

        assert result == expected
