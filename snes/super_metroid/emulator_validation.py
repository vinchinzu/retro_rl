"""Emulator validation for trajectory predictions.

Ground truth validation: run winning trajectories on real emulator
(stable-retro / SMEDIT snes9x) to verify predictor results.

This module provides the validation path required for room-clear claims.
Predictor is for search speed only; emulator is authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from retro_harness.actions import indexed_action
from retro_harness.env import read_state_bytes

from super_metroid.physics_sim import FrameInput, position_out_of_range
from super_metroid.ram import parse_env_state

__all__ = [
    "EmulatorValidationResult",
    "validate_trajectory_on_emulator",
    "ROM_AVAILABLE",
]


def _check_rom_available() -> bool:
    """True when a ROM file exists and stable-retro exposes Integrations.CUSTOM."""
    rom_paths = [
        Path("roms/SuperMetroid.sfc"),
        Path("roms/Super Metroid.sfc"),
        Path("../roms/SuperMetroid.sfc"),
    ]
    if not any(p.exists() for p in rom_paths):
        return False
    try:
        import stable_retro as retro

        return hasattr(getattr(retro.data, "Integrations", None), "CUSTOM")
    except Exception:
        return False


ROM_AVAILABLE = _check_rom_available()


@dataclass(frozen=True)
class EmulatorValidationResult:
    """Result of emulator validation for a trajectory.

    This is ground truth — if validation fails, predictor was wrong.
    """

    success: bool
    """Whether the trajectory succeeded on emulator (ground truth)."""

    final_room_id: int | None
    """Room ID after trajectory execution (None if crashed/failed)."""

    final_x: int | None
    """Final X position (None if crashed/failed)."""

    final_y: int | None
    """Final Y position (None if crashed/failed)."""

    frames_executed: int
    """Number of frames successfully executed."""

    reason: str = ""
    """Failure reason when success=False."""

    checked_room_id: int | None = None
    """Room id that was explicitly checked and matched. None = no room-clear claim."""

    @property
    def room_clear(self) -> bool:
        """True only when an explicit room target was checked and passed."""
        return self.success and self.checked_room_id is not None


def validate_trajectory_on_emulator(
    start_state_path: Path | str,
    inputs: Sequence[FrameInput],
    *,
    target_room_id: int | None = None,
    target_x_range: tuple[int, int] | None = None,
    target_y_range: tuple[int, int] | None = None,
) -> EmulatorValidationResult:
    """Validate trajectory on real emulator (ground truth).

    This is the authoritative validation path for room-clear claims.
    Predictor results are heuristics; emulator result is ground truth.

    Args:
        start_state_path: Path to emulator save state (.state file)
        inputs: Button input sequence to execute
        target_room_id: Expected final room (None = don't check)
        target_x_range: Expected final X range (None = don't check)
        target_y_range: Expected final Y range (None = don't check)

    Returns:
        EmulatorValidationResult with ground truth outcome

    Raises:
        FileNotFoundError: If ROM is not available
        RuntimeError: If emulator fails to load

    Note:
        This function requires ROM and stable-retro. Tests can use
        @pytest.mark.skipif(not ROM_AVAILABLE) to skip without ROM.
    """
    if not ROM_AVAILABLE:
        raise FileNotFoundError(
            "ROM not available. Cannot validate on emulator. "
            "Place Super Metroid ROM in roms/ directory."
        )

    start_path = Path(start_state_path)
    if not start_path.exists():
        raise FileNotFoundError(f"Start state not found: {start_path}")

    from super_metroid.dev.common import make_dev_env

    env = make_dev_env()
    frames_executed = 0
    try:
        env.reset()
        env.em.set_state(read_state_bytes(start_path))

        for i, frame_input in enumerate(inputs):
            indices = [b for b in range(12) if frame_input.buttons & (1 << b)]
            env.step(indexed_action(indices, action_size=12))
            frames_executed = i + 1

        final_state = parse_env_state(env, mode="full")
        final_room = final_state.room_id
        final_x = final_state.samus_x
        final_y = final_state.samus_y
    except Exception as e:
        return EmulatorValidationResult(
            success=False,
            final_room_id=None,
            final_x=None,
            final_y=None,
            frames_executed=frames_executed,
            reason=f"Emulator failed: {e}",
        )
    finally:
        env.close()

    reason = ""
    if target_room_id is not None and final_room != target_room_id:
        reason = f"Room mismatch: expected {target_room_id:04X}, got {final_room:04X}"
    if not reason:
        reason = position_out_of_range(
            final_x, final_y, x_range=target_x_range, y_range=target_y_range
        )
    success = not reason
    checked_room_id = (
        final_room if target_room_id is not None and success else None
    )

    return EmulatorValidationResult(
        success=success,
        final_room_id=final_room,
        final_x=final_x,
        final_y=final_y,
        frames_executed=frames_executed,
        reason=reason,
        checked_room_id=checked_room_id,
    )
