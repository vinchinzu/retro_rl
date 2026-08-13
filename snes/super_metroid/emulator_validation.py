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

from super_metroid.physics_sim import FrameInput

__all__ = [
    "EmulatorValidationResult",
    "validate_trajectory_on_emulator",
    "ROM_AVAILABLE",
]


# Check if ROM is available for validation tests
def _check_rom_available() -> bool:
    """Check if Super Metroid ROM is available."""
    # Check common ROM locations
    rom_paths = [
        Path("roms/SuperMetroid.sfc"),
        Path("roms/Super Metroid.sfc"),
        Path("../roms/SuperMetroid.sfc"),
    ]
    return any(p.exists() for p in rom_paths)


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

    @property
    def room_clear(self) -> bool:
        """True if trajectory cleared target room (ground truth claim)."""
        return self.success


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

    # Import here to avoid requiring stable-retro for planning-only code
    try:
        from retro_harness.snes import SuperMetroidEnv
    except ImportError as e:
        raise RuntimeError(
            "stable-retro not available. Install retro_harness with SNES support."
        ) from e

    start_path = Path(start_state_path)
    if not start_path.exists():
        raise FileNotFoundError(f"Start state not found: {start_path}")

    # Load emulator from state
    env = SuperMetroidEnv()
    try:
        env.load_state(str(start_path))
    except Exception as e:
        return EmulatorValidationResult(
            success=False,
            final_room_id=None,
            final_x=None,
            final_y=None,
            frames_executed=0,
            reason=f"Failed to load state: {e}",
        )

    # Execute input sequence on emulator
    frames_executed = 0
    for i, frame_input in enumerate(inputs):
        try:
            # Step emulator with button mask
            env.step(frame_input.buttons)
            frames_executed = i + 1
        except Exception as e:
            return EmulatorValidationResult(
                success=False,
                final_room_id=None,
                final_x=None,
                final_y=None,
                frames_executed=frames_executed,
                reason=f"Emulator crash at frame {i}: {e}",
            )

    # Read final state from emulator
    try:
        final_state = env.get_state()
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
            reason=f"Failed to read final state: {e}",
        )

    # Check success criteria
    success = True
    reason = ""

    if target_room_id is not None and final_room != target_room_id:
        success = False
        reason = f"Room mismatch: expected {target_room_id:04X}, got {final_room:04X}"

    if success and target_x_range is not None:
        x_lo, x_hi = target_x_range
        if not (x_lo <= final_x <= x_hi):
            success = False
            reason = f"X out of range: {final_x} not in [{x_lo}, {x_hi}]"

    if success and target_y_range is not None:
        y_lo, y_hi = target_y_range
        if not (y_lo <= final_y <= y_hi):
            success = False
            reason = f"Y out of range: {final_y} not in [{y_lo}, {y_hi}]"

    return EmulatorValidationResult(
        success=success,
        final_room_id=final_room,
        final_x=final_x,
        final_y=final_y,
        frames_executed=frames_executed,
        reason=reason,
    )
