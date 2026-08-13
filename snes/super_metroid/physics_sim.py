"""Physics simulation client for Super Metroid trajectory prediction.

Provides a protocol for predicting Samus trajectories given a start state
and input sequence. Designed to integrate with external physics engines
(e.g., sm_rev MiniStep predictor) or stub implementations for testing.

This module defines:
- Data structures for simulation state, inputs, and trajectory frames
- Protocol/ABC for physics predictors
- StubPredictor for offline testing
- Optional client stub for external sm_rev predict binary

Integration points:
- Route planning: query candidate trajectories without full emulation
- SMEDIT route panel: live trajectory preview
- Hop optimization: evaluate input-tape variants offline
"""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from super_metroid.ram import SuperMetroidState

__all__ = [
    "SimState",
    "FrameInput",
    "TrajectoryFrame",
    "Trajectory",
    "PhysicsPredictor",
    "StubPredictor",
    "SmRevClient",
    "load_predictor",
    "encode_frame_mnemonic",
    "decode_frame_mnemonic",
]

# SNES button order for smedit-tas-1 format (matches retro_harness.controls)
# [B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]
_BUTTON_ORDER = ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"]
_BUTTON_MNEMONICS = ["b", "y", "s", "S", "u", "d", "l", "r", "a", "x", "L", "R"]


def encode_frame_mnemonic(buttons: int) -> str:
    """Encode button mask to 12-char mnemonic for smedit-tas-1 format.

    Button order: B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R
    Mnemonic: lowercase/char for pressed, '.' for released

    Args:
        buttons: Button mask (bit i = button i pressed)

    Returns:
        12-character mnemonic string (e.g., "b......r...." for B+RIGHT)

    Examples:
        >>> encode_frame_mnemonic(0)  # All released
        '............'
        >>> encode_frame_mnemonic(1)  # B pressed
        'b...........'
        >>> encode_frame_mnemonic(0x81)  # B + RIGHT (bits 0 and 7)
        'b......r....'
    """
    chars = []
    for i in range(12):
        if buttons & (1 << i):
            chars.append(_BUTTON_MNEMONICS[i])
        else:
            chars.append(".")
    return "".join(chars)


def decode_frame_mnemonic(mnemonic: str) -> int:
    """Decode 12-char mnemonic to button mask.

    Args:
        mnemonic: 12-character mnemonic string

    Returns:
        Button mask (bit i = button i pressed)

    Raises:
        ValueError: If mnemonic length is not 12
    """
    if len(mnemonic) != 12:
        raise ValueError(f"Mnemonic must be 12 chars, got {len(mnemonic)}")

    buttons = 0
    for i, char in enumerate(mnemonic):
        if char != ".":
            buttons |= 1 << i
    return buttons


@dataclass(frozen=True)
class SimState:
    """Minimal Super Metroid state for physics prediction.

    Represents Samus kinematics and relevant RAM state at a point in time.
    Subset of SuperMetroidState focused on physics-relevant fields.
    """

    frame: int
    room_id: int
    samus_x: int
    samus_y: int
    samus_x_sub: int
    samus_y_sub: int
    velocity_x: int
    velocity_y: int
    velocity_x_sub: int
    velocity_y_sub: int
    momentum_x: int
    momentum_x_sub: int
    pose: int
    facing: int
    movement_type: int
    speed_counter: int
    speed_flag: int
    shinespark_timer: int

    @classmethod
    def from_sm_state(cls, state: SuperMetroidState) -> SimState:
        """Extract sim state from full SuperMetroidState."""
        return cls(
            frame=state.frame,
            room_id=state.room_id,
            samus_x=state.samus_x,
            samus_y=state.samus_y,
            samus_x_sub=state.samus_x_sub,
            samus_y_sub=state.samus_y_sub,
            velocity_x=state.velocity_x,
            velocity_y=state.velocity_y,
            velocity_x_sub=state.velocity_x_sub,
            velocity_y_sub=state.velocity_y_sub,
            momentum_x=state.momentum_x,
            momentum_x_sub=state.momentum_x_sub,
            pose=state.pose,
            facing=state.facing,
            movement_type=state.movement_type,
            speed_counter=state.speed_counter,
            speed_flag=state.speed_flag,
            shinespark_timer=state.shinespark_timer,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimState:
        return cls(**data)


@dataclass(frozen=True)
class FrameInput:
    """Controller input for a single frame.

    Buttons are SNES button masks matching retro_harness.controls conventions.
    """

    buttons: int

    def to_dict(self) -> dict[str, Any]:
        return {"buttons": self.buttons}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrameInput:
        return cls(buttons=data["buttons"])


@dataclass(frozen=True)
class TrajectoryFrame:
    """Samus state at one frame of a predicted trajectory.

    Includes full kinematics plus relevant RAM state for route analysis.
    Optional enemies list for damage-boost routing and SMEDIT overlay.
    """

    frame: int
    room_id: int
    samus_x: int
    samus_y: int
    samus_x_sub: int
    samus_y_sub: int
    velocity_x: int
    velocity_y: int
    velocity_x_sub: int
    velocity_y_sub: int
    momentum_x: int
    momentum_x_sub: int
    pose: int
    facing: int
    movement_type: int
    speed_counter: int
    speed_flag: int
    shinespark_timer: int
    enemies: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Omit enemies when empty (not required)
        if not result["enemies"]:
            del result["enemies"]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryFrame:
        # enemies is optional, defaults to empty tuple
        enemies_data = data.get("enemies", [])
        data_copy = dict(data)
        data_copy["enemies"] = tuple(enemies_data)
        return cls(**data_copy)


@dataclass(frozen=True)
class Trajectory:
    """Complete predicted trajectory from start state + input sequence.

    Contains per-frame state predictions for route analysis and validation.
    """

    start: SimState
    frames: tuple[TrajectoryFrame, ...]
    predictor: str
    inputs: tuple[FrameInput, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.to_dict(),
            "frames": [f.to_dict() for f in self.frames],
            "predictor": self.predictor,
            "inputs": [inp.to_dict() for inp in self.inputs] if self.inputs else [],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            start=SimState.from_dict(data["start"]),
            frames=tuple(TrajectoryFrame.from_dict(f) for f in data["frames"]),
            predictor=data["predictor"],
            inputs=tuple(FrameInput.from_dict(inp) for inp in data.get("inputs", [])),
        )

    def to_smedit_tas(
        self,
        *,
        start_state_name: str = "ZebesStart",
        rom_sha1: str | None = None,
        trace_stride: int = 1,
    ) -> dict[str, Any]:
        """Export trajectory as SMEDIT TasMovie format (smedit-tas-1).

        Compatible with SMEDIT route panel for trajectory preview and
        route planning workflows.

        Args:
            start_state_name: Human-readable start state name
            rom_sha1: ROM SHA1 hash (None for stub/CI without ROM)
            trace_stride: Include trace entry every N frames (1 = every frame)

        Returns:
            smedit-tas-1 format dict with format, meta, buttonOrder, frames, trace

        Notes:
            - frames[]: 12-char mnemonics encoded from button masks (native format)
            - trace[].roomId: Room header pointer (WRAM $079B), NOT door ID.
              Door transitions show as roomId changing between trace points.
            - trace[].enemies: Optional list for SMEDIT overlay and damage-boost
              routing. Omitted when empty. Point structure: {id/slot, x, y, hp?}.
              Used for Ceres→Morph→Bomb damage-boost routing once sm_rev fills it.
            - Extra kinematics fields (subX, subY, pose, etc.) are optional;
              route planner ignores unknown keys.

        Example:
            >>> trajectory.to_smedit_tas(start_state_name="LandingSite", trace_stride=10)
            {
              "format": "smedit-tas-1",
              "meta": {
                "gameName": "SuperMetroid-Snes",
                "startState": "LandingSite",
                "romSha1": null
              },
              "buttonOrder": ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
              "frames": ["............", "b......r...."],
              "trace": [{"frame": 0, "x": 184, "y": 312, "roomId": 37368}]
            }
        """
        # Encode inputs to mnemonic frames
        frame_mnemonics = [encode_frame_mnemonic(inp.buttons) for inp in self.inputs]

        # Build sparse trace with required x/y and optional fields
        trace = []
        for i, traj_frame in enumerate(self.frames):
            if i % trace_stride == 0:
                entry: dict[str, Any] = {
                    "frame": traj_frame.frame,
                    "x": traj_frame.samus_x,
                    "y": traj_frame.samus_y,
                }
                # Optional fields when available
                if traj_frame.samus_x_sub != 0:
                    entry["subX"] = traj_frame.samus_x_sub
                if traj_frame.samus_y_sub != 0:
                    entry["subY"] = traj_frame.samus_y_sub
                if traj_frame.pose != 0:
                    entry["pose"] = traj_frame.pose
                entry["roomId"] = traj_frame.room_id
                # Optional enemies list for damage-boost routing / SMEDIT overlay
                if traj_frame.enemies:
                    entry["enemies"] = [dict(e) for e in traj_frame.enemies]
                trace.append(entry)

        return {
            "format": "smedit-tas-1",
            "meta": {
                "gameName": "SuperMetroid-Snes",
                "startState": start_state_name,
                "romSha1": rom_sha1,
            },
            "buttonOrder": _BUTTON_ORDER,
            "frames": frame_mnemonics,
            "trace": trace,
        }


class PhysicsPredictor(ABC):
    """Abstract predictor protocol for Super Metroid physics simulation.

    Implementations predict Samus trajectories given start state + inputs
    without requiring full emulator for every candidate sequence.
    """

    @abstractmethod
    def predict(
        self, start: SimState, inputs: Sequence[FrameInput]
    ) -> Trajectory:
        """Predict trajectory from start state with input sequence.

        Args:
            start: Initial Samus kinematics and RAM state
            inputs: Button sequence, one per frame

        Returns:
            Trajectory with per-frame predictions

        Raises:
            RuntimeError: If predictor backend fails or is unavailable
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable predictor identity for trajectory provenance."""
        ...


class StubPredictor(PhysicsPredictor):
    """Deterministic stub predictor for offline testing.

    Returns fake but consistent motion for tests that don't need
    real Super Metroid physics. Useful for:
    - Protocol contract tests
    - CLI smoke tests
    - Route planning unit tests without ROM

    Physics model: simple linear motion with gravity and input-based
    velocity changes (not accurate Super Metroid physics).
    """

    def __init__(self, name: str = "stub") -> None:
        self._name = name

    def predict(
        self, start: SimState, inputs: Sequence[FrameInput]
    ) -> Trajectory:
        """Predict with simple deterministic fake physics.
        
        Button masks (bit positions):
        - B=0 (0x01), Y=1 (0x02), Select=2 (0x04), Start=3 (0x08)
        - Up=4 (0x10), Down=5 (0x20), Left=6 (0x40), Right=7 (0x80)
        - A=8 (0x100), X=9 (0x200), L=10 (0x400), R=11 (0x800)
        """
        frames: list[TrajectoryFrame] = []
        x, y = start.samus_x, start.samus_y
        x_sub, y_sub = start.samus_x_sub, start.samus_y_sub
        vx, vy = start.velocity_x, start.velocity_y
        vx_sub, vy_sub = start.velocity_x_sub, start.velocity_y_sub
        mx, mx_sub = start.momentum_x, start.momentum_x_sub
        pose = start.pose
        facing = start.facing
        movement = start.movement_type
        speed = start.speed_counter
        speed_flag = start.speed_flag
        spark = start.shinespark_timer

        for i, inp in enumerate(inputs):
            # Fake physics: simple linear motion with gravity
            # Not accurate SM physics, just deterministic test motion

            # Apply fake gravity if airborne (not grounded)
            if movement != 0:
                vy_sub += 1000
                if vy_sub >= 65536:
                    vy += vy_sub // 65536
                    vy_sub = vy_sub % 65536

            # Apply fake horizontal input response
            # LEFT = bit 6 (0x40), RIGHT = bit 7 (0x80)
            if inp.buttons & 0x40:  # LEFT
                vx = -2
                facing = 0x04
            elif inp.buttons & 0x80:  # RIGHT
                vx = 2
                facing = 0x08
            else:
                vx = 0

            # Apply fake jump (B = bit 0, 0x01)
            if inp.buttons & 0x01:  # B (jump)
                if movement == 0:  # grounded
                    vy = -5
                    movement = 1

            # Update position (simple pixel-per-frame for stub)
            x += vx
            y += vy

            # Fake ground collision at y=200
            if y >= 200 and vy > 0:
                y = 200
                y_sub = 0
                vy = 0
                vy_sub = 0
                movement = 0

            frames.append(
                TrajectoryFrame(
                    frame=start.frame + i + 1,
                    room_id=start.room_id,
                    samus_x=x,
                    samus_y=y,
                    samus_x_sub=x_sub,
                    samus_y_sub=y_sub,
                    velocity_x=vx,
                    velocity_y=vy,
                    velocity_x_sub=vx_sub,
                    velocity_y_sub=vy_sub,
                    momentum_x=mx,
                    momentum_x_sub=mx_sub,
                    pose=pose,
                    facing=facing,
                    movement_type=movement,
                    speed_counter=speed,
                    speed_flag=speed_flag,
                    shinespark_timer=spark,
                )
            )

        return Trajectory(
            start=start,
            frames=tuple(frames),
            predictor=self._name,
            inputs=tuple(inputs),
        )

    def name(self) -> str:
        return self._name


class SmRevClient(PhysicsPredictor):
    """Client for sm_rev_predict external physics predictor.

    Calls sm_rev MiniStep-based physics predictor for better search heuristics.
    Gracefully raises RuntimeError if binary not available (CI stays green).

    IMPORTANT: sm_rev/MiniStep is for search speed only, NOT ground truth.
    Final validation requires emulator (stable-retro / SMEDIT snes9x).

    Environment:
        SM_REV_PATH: Path to sm_rev_predict binary (default: sm_rev_predict in PATH)
    """

    def __init__(self, binary_path: str | Path | None = None) -> None:
        """Initialize sm_rev client.

        Args:
            binary_path: Path to sm_rev_predict binary, or None to use
                        SM_REV_PATH env var or 'sm_rev_predict' in PATH
        """
        if binary_path is None:
            binary_path = os.environ.get("SM_REV_PATH", "sm_rev_predict")
        self._binary = Path(binary_path) if isinstance(binary_path, str) else binary_path

    def _is_available(self) -> bool:
        """Check if sm_rev_predict binary is available."""
        try:
            result = subprocess.run(
                [str(self._binary), "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def predict(
        self, start: SimState, inputs: Sequence[FrameInput]
    ) -> Trajectory:
        """Predict trajectory using sm_rev_predict external binary.

        Note: sm_rev/MiniStep is for search speed only, NOT ground truth.
        Winners must be validated on emulator (stable-retro / SMEDIT snes9x).

        Raises:
            RuntimeError: If sm_rev_predict binary is not available or fails
        """
        if not self._is_available():
            raise RuntimeError(
                f"sm_rev_predict binary not available at {self._binary}. "
                "Set SM_REV_PATH env var or ensure sm_rev_predict is in PATH."
            )

        # Prepare input JSON for sm_rev_predict
        request = {
            "start": start.to_dict(),
            "inputs": [inp.to_dict() for inp in inputs],
        }

        # Invoke binary (with or without 'predict' subcommand)
        # sm_rev_predict accepts both `sm_rev_predict` and `sm_rev_predict predict`
        binary_name = self._binary.name if hasattr(self._binary, 'name') else str(self._binary)
        
        # If binary is named sm_rev_predict, invoke without subcommand
        # Otherwise include 'predict' subcommand for compatibility
        if 'sm_rev_predict' in binary_name:
            cmd = [str(self._binary)]
        else:
            cmd = [str(self._binary), "predict"]

        try:
            result = subprocess.run(
                cmd,
                input=json.dumps(request).encode(),
                capture_output=True,
                timeout=30,
                check=True,
            )
            response = json.loads(result.stdout)
            return Trajectory.from_dict(response)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"sm_rev_predict failed: {e.stderr.decode()}"
            ) from e
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            raise RuntimeError(f"sm_rev_predict communication failed: {e}") from e

    def name(self) -> str:
        return f"sm_rev_predict@{self._binary}"


def load_predictor(
    backend: str = "stub", **kwargs: Any
) -> PhysicsPredictor:
    """Load a physics predictor by backend name.

    Args:
        backend: Predictor backend ("stub" or "sm_rev")
        **kwargs: Backend-specific configuration

    Returns:
        Configured predictor instance

    Raises:
        ValueError: If backend name is unknown
    """
    if backend == "stub":
        return StubPredictor(**kwargs)
    if backend == "sm_rev":
        return SmRevClient(**kwargs)
    raise ValueError(
        f"Unknown predictor backend: {backend!r}. "
        "Supported: 'stub', 'sm_rev'"
    )
