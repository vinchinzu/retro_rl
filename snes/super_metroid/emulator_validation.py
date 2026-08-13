"""Emulator validation for trajectory predictions.

Ground truth validation: run winning trajectories on real emulator
(stable-retro / SMEDIT snes9x) to verify predictor results.

This module provides the validation path required for room-clear claims.
Predictor is for search speed only; emulator is authoritative.

Tag E = SuperMetroidEnv (product name for RetroEnv-based validation, not a class).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from retro_harness.actions import indexed_action
from retro_harness.env import GameSpec, resync_custom_state

from super_metroid.observation import Observation
from super_metroid.paths import GAME_DIR
from super_metroid.physics_sim import FrameInput
from super_metroid.ram import parse_env_state, read_wram_u16, read_wram_u8
from super_metroid.residual import ResidualProfile, compute_residual_profile

__all__ = [
    "EmulatorValidationResult",
    "observation_from_env",
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
    
    residual: ResidualProfile | None = None
    """Residual profile R(τ) when Mini observations provided."""

    @property
    def room_clear(self) -> bool:
        """True if trajectory cleared target room (ground truth claim)."""
        return self.success


def observation_from_env(env: Any) -> Observation:
    """Extract Observation from emulator (RetroEnv from GameSpec.make_env()).

    Reads Oπ/Oσ/Oσ+/O† fields from emulator RAM:
    - Oπ: pixels x/y ($0AF6, $0AFA), pose ($0A1C), room ($079B)
    - Oσ: Oπ plus subpixels ($0AF8, $0AFC)
    - Oσ+: Oσ plus enemy energy ($0F8C) / i-frames ($18A8)
    - O†: energy ($09C2)
    - Lag: frame counters ($1842, $09DA)
    - Speeds: velocity/momentum for first-differing-field

    Args:
        env: RetroEnv from GameSpec.make_env()

    Returns:
        Observation with all fields populated from emulator RAM
    """
    state = parse_env_state(env, mode="full")
    
    # Peek lag counters and enemy/invuln fields for Oσ+
    enemy_energy = read_wram_u16(env, 0x0F8C)
    invuln_timer = read_wram_u16(env, 0x18A8)
    frame_counter_1 = read_wram_u8(env, 0x1842)
    frame_counter_2 = read_wram_u16(env, 0x09DA)

    return Observation(
        frame=state.frame,
        x=state.samus_x,
        y=state.samus_y,
        pose=state.pose,
        room=state.room_id,
        sub_x=state.samus_x_sub,
        sub_y=state.samus_y_sub,
        velocity_x=state.velocity_x,
        velocity_y=state.velocity_y,
        velocity_x_sub=state.velocity_x_sub,
        velocity_y_sub=state.velocity_y_sub,
        momentum_x=state.momentum_x,
        momentum_x_sub=state.momentum_x_sub,
        speed_counter=state.speed_counter,
        speed_flag=state.speed_flag,
        energy=state.health,
        frame_counter_1=frame_counter_1,
        frame_counter_2=frame_counter_2,
        enemy_energy=enemy_energy,
        invulnerability_timer=invuln_timer,
    )


def _buttons_mask_to_action(buttons: int) -> list[int]:
    """Convert FrameInput packed button mask to 12-length action vector.
    
    Args:
        buttons: Packed mask where bit i = button i pressed
        
    Returns:
        12-length list with 1 at pressed button indices, 0 elsewhere
        
    Example:
        0x80 (bit 7) → [0,0,0,0,0,0,0,1,0,0,0,0] (RIGHT pressed)
    """
    indices = [i for i in range(12) if buttons & (1 << i)]
    return indexed_action(indices, action_size=12)


def validate_trajectory_on_emulator(
    start_state_path: Path | str,
    inputs: Sequence[FrameInput],
    *,
    target_room_id: int | None = None,
    target_x_range: tuple[int, int] | None = None,
    target_y_range: tuple[int, int] | None = None,
    mini_observations: list[Observation] | None = None,
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
        mini_observations: Optional Mini/Stub predictor observations for residual

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

    # Create emulator using GameSpec
    game_spec = GameSpec(game="SuperMetroid-Snes", game_dir=GAME_DIR)
    
    try:
        env = game_spec.make_env(state=None)
        env.reset()
        
        # Load the custom state
        from retro_harness.env import read_state_bytes
        state_data = read_state_bytes(start_path)
        env.em.set_state(state_data)
        
        # Resync to drop free frame (if this is a custom state)
        state_name = start_path.stem
        resync_custom_state(env, GAME_DIR, "SuperMetroid-Snes", state_name)
        
    except Exception as e:
        return EmulatorValidationResult(
            success=False,
            final_room_id=None,
            final_x=None,
            final_y=None,
            frames_executed=0,
            reason=f"Failed to load state: {e}",
        )

    # Collect emulator observations for residual profiling
    emu_observations: list[Observation] = []
    
    # Execute input sequence on emulator
    frames_executed = 0
    for i, frame_input in enumerate(inputs):
        try:
            # Convert packed button mask to 12-length action vector
            action = _buttons_mask_to_action(frame_input.buttons)
            env.step(action)
            frames_executed = i + 1
            
            # Collect observation for residual profiling
            if mini_observations is not None:
                emu_observations.append(observation_from_env(env))
                
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

    # Compute residual profile when Mini observations provided
    residual = None
    if mini_observations is not None:
        emu_obs_for_residual = emu_observations if emu_observations else None
        residual = compute_residual_profile(mini_observations, emu_obs_for_residual)

    return EmulatorValidationResult(
        success=success,
        final_room_id=final_room,
        final_x=final_x,
        final_y=final_y,
        frames_executed=frames_executed,
        reason=reason,
        residual=residual,
    )
