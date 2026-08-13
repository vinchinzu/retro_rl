"""Third Ceres room (Magnet Stairs → Scientist) hop/tape fixture.

Real tape extracted from existing Ceres outbound route
(`routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`) for the third
room of Ceres → Morph → Bomb sequence.

Tape Source:
- NOT greedy search or invented physics
- From `snes/super_metroid/routes/kpdr/ceres/outbound.py`
- ActionSpan list: LEFT, RIGHT+B with arm-pump, idle
- Covers Elevator → Falling Tile → Magnet → Scientist
- This fixture takes the documented prefix for Magnet → Scientist only

Policy:
- Predictor (StubPredictor / sm_rev_predict) = search speed only
- Emulator (stable-retro / SMEDIT snes9x) = ground truth
- Mini/stub results are heuristics; emulator wins
- Room-clear claims require emulator validation

Validation:
- Start state: env var `SM_CERES_MAGNET_STATE` (path to .state file on disk)
- Tests skip without ROM_AVAILABLE or missing start state
- Never commit .state or ROM blobs to repo
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX
from super_metroid.emulator_validation import (
    EmulatorValidationResult,
    ROM_AVAILABLE,
    validate_trajectory_on_emulator,
)
from super_metroid.physics_sim import FrameInput
from super_metroid.routes.kpdr.ceres.arm_pump import _arm_pump_dash_spans
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
)

__all__ = [
    "CeresThirdRoomFixture",
    "get_ceres_third_room_tape",
    "validate_ceres_third_room",
    "button_names_to_mask",
]


def button_names_to_mask(names: tuple[str, ...]) -> int:
    """Convert SNES button names to button mask for FrameInput.

    Args:
        names: Button names (e.g., ("RIGHT", "A", "B"))

    Returns:
        Button mask where bit i = button i pressed

    Examples:
        >>> button_names_to_mask(("LEFT",))
        64  # 0x40, bit 6
        >>> button_names_to_mask(("RIGHT", "B"))
        129  # 0x81, bits 7+0
    """
    mask = 0
    for name in names:
        idx = SNES_BUTTON_NAME_TO_INDEX.get(name.strip().upper())
        if idx is not None:
            mask |= 1 << idx
    return mask


@dataclass(frozen=True)
class CeresThirdRoomFixture:
    """Third Ceres room hop/tape fixture (real tape, not search).

    Tape extracted from `routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`.
    Prefix covers Magnet Stairs 0xDFD7 → Scientist 0xE021 only.

    Emulator validation uses env var SM_CERES_MAGNET_STATE for start state path.
    """

    from_room_id: int
    to_room_id: int
    inputs: tuple[FrameInput, ...]
    tape_source: str
    emulator_validated: bool = False
    emulator_success: bool = False
    emulator_final_room: int | None = None
    emulator_final_x: int | None = None
    emulator_final_y: int | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (smedit-tas-1 compatible)."""
        return {
            **asdict(self),
            "inputs": [
                {"buttons": inp.buttons, "frame": i}
                for i, inp in enumerate(self.inputs)
            ],
        }

    @property
    def frames(self) -> int:
        """Total frames in tape."""
        return len(self.inputs)

    @property
    def room_clear(self) -> bool:
        """True if emulator validation confirmed room clear (ground truth).

        Never claim room-clear without emulator validation.
        """
        return self.emulator_validated and self.emulator_success


def get_ceres_third_room_tape() -> CeresThirdRoomFixture:
    """Get real tape for Ceres Magnet Stairs → Scientist (EXACT prefix).

    Tape source: `routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`
    - EXACT prefix through room 3 (lines 54-56 in outbound.py)
    - Continues after room 1's 564 frames + room 2's 552 frames
    - Final LEFT-ward segment (stops BEFORE scientist→Ridley arm-pump)
    - RIGHT+B span expanded via `_arm_pump_dash_spans` (period=2, L↔R)
    - One RIGHT+B pump span at 96 frames

    Returns:
        CeresThirdRoomFixture with real tape (emulator_validated=False)

    Note:
        This is the EXACT product tape with full arm-pump expansion, not
        a shortened sketch. Frame count must match expanding the raw spans.
    """
    # EXACT raw prefix from _ceres_outbound_to_scientist_spans (lines 54-56):
    # (("LEFT",), 120, False),
    # (("RIGHT", "B"), 96, True),
    # ((), 24, False),

    inputs: list[FrameInput] = []

    # Span 1: LEFT, 120 frames
    mask = button_names_to_mask(("LEFT",))
    for _ in range(120):
        inputs.append(FrameInput(buttons=mask))

    # Span 2: RIGHT+B, 96 frames with arm-pump expansion
    arm_pump_spans = _arm_pump_dash_spans("RIGHT", 96, "ceres_third_room")
    for span in arm_pump_spans:
        mask = button_names_to_mask(span.names)
        for _ in range(span.frames):
            inputs.append(FrameInput(buttons=mask))

    # Span 3: idle, 24 frames
    mask = button_names_to_mask(())
    for _ in range(24):
        inputs.append(FrameInput(buttons=mask))

    # Total: 120 + 96 + 24 = 240 frames
    return CeresThirdRoomFixture(
        from_room_id=ROOM_CERES_MAGNET,
        to_room_id=ROOM_CERES_SCIENTIST,
        inputs=tuple(inputs),
        tape_source="routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans (EXACT prefix, 240 frames)",
        emulator_validated=False,
    )


def validate_ceres_third_room(
    fixture: CeresThirdRoomFixture,
    start_state_path: Path | str | None = None,
) -> CeresThirdRoomFixture:
    """Validate fixture on real emulator (ground truth).

    Runs tape on stable-retro / SMEDIT snes9x. This is the authoritative
    validation path for room-clear claims.

    Args:
        fixture: Tape fixture to validate
        start_state_path: Path to Ceres Magnet Stairs start state (optional)
            If None, uses env var SM_CERES_MAGNET_STATE

    Returns:
        Updated fixture with emulator validation results

    Raises:
        FileNotFoundError: If ROM or start state not available
        RuntimeError: If emulator fails to load
        ValueError: If start state path not provided and env var not set

    Note:
        Tests skip validation if:
        - ROM_AVAILABLE is False
        - SM_CERES_MAGNET_STATE env var not set
        - Start state file does not exist
    """
    if start_state_path is None:
        start_state_path = os.environ.get("SM_CERES_MAGNET_STATE")
        if not start_state_path:
            raise ValueError(
                "start_state_path not provided and SM_CERES_MAGNET_STATE not set"
            )

    if not Path(start_state_path).exists():
        raise FileNotFoundError(f"Start state not found: {start_state_path}")

    result = validate_trajectory_on_emulator(
        start_state_path,
        fixture.inputs,
        target_room_id=fixture.to_room_id,
    )

    return CeresThirdRoomFixture(
        from_room_id=fixture.from_room_id,
        to_room_id=fixture.to_room_id,
        inputs=fixture.inputs,
        tape_source=fixture.tape_source,
        emulator_validated=True,
        emulator_success=result.success,
        emulator_final_room=result.final_room_id,
        emulator_final_x=result.final_x,
        emulator_final_y=result.final_y,
    )
