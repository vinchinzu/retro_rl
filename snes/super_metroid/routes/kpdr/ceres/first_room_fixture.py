"""First Ceres room (Elevator → Falling Tile) hop/tape fixture.

Real tape extracted from existing Ceres outbound route
(`routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`) for the first
room of Ceres → Morph → Bomb sequence.

Tape Source:
- NOT greedy search or invented physics
- From `snes/super_metroid/routes/kpdr/ceres/outbound.py`
- ActionSpan list: RIGHT+A, LEFT, RIGHT+B with arm-pump, etc.
- Covers Elevator → Falling Tile → Magnet → Scientist
- This fixture takes the documented prefix for Elevator → Falling only

Policy:
- Predictor (StubPredictor / sm_rev_predict) = search speed only
- Emulator (stable-retro / SMEDIT snes9x) = ground truth
- Mini/stub results are heuristics; emulator wins
- Room-clear claims require emulator validation

Validation:
- Start state: env var `SM_CERES_ELEV_STATE` (path to .state file on disk)
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
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
)

__all__ = [
    "CeresFirstRoomFixture",
    "get_ceres_first_room_tape",
    "validate_ceres_first_room",
    "button_names_to_mask",
]


def button_names_to_mask(names: tuple[str, ...]) -> int:
    """Convert SNES button names to button mask for FrameInput.

    Args:
        names: Button names (e.g., ("RIGHT", "A", "B"))

    Returns:
        Button mask where bit i = button i pressed

    Examples:
        >>> button_names_to_mask(("RIGHT",))
        128  # 0x80, bit 7
        >>> button_names_to_mask(("RIGHT", "A"))
        384  # 0x180, bits 7+8
    """
    mask = 0
    for name in names:
        idx = SNES_BUTTON_NAME_TO_INDEX.get(name.strip().upper())
        if idx is not None:
            mask |= 1 << idx
    return mask


@dataclass(frozen=True)
class CeresFirstRoomFixture:
    """First Ceres room hop/tape fixture (real tape, not search).

    Tape extracted from `routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`.
    Prefix covers Elevator 0xDF45 → Falling Tile 0xDF8D only.

    Emulator validation uses env var SM_CERES_ELEV_STATE for start state path.
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


def get_ceres_first_room_tape() -> CeresFirstRoomFixture:
    """Get real tape for Ceres Elevator → Falling Tile (first room only).

    Tape source: `routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans`
    - ActionSpan list for Elevator → Falling → Magnet → Scientist
    - This function takes the documented prefix for first room only

    Returns:
        CeresFirstRoomFixture with real tape (emulator_validated=False)

    Note:
        This is NOT a greedy search or invented physics. It's the existing
        product tape from outbound.py, converted to FrameInput format.
    """
    # Real tape from outbound.py: Elevator → Falling Tile prefix
    # Source: _ceres_outbound_to_scientist_spans() lines 35-42
    # (RIGHT+A 24, RIGHT 120, LEFT 120, RIGHT+B 240 with arm-pump, idle 60)
    #
    # First room transition happens in first ~500 frames
    # Taking prefix up to first LEFT turn (264 frames) as first-room tape
    tape_frames: list[tuple[tuple[str, ...], int]] = [
        (("RIGHT", "A"), 24),    # Jump start
        (("RIGHT",), 120),        # Run right
        (("LEFT",), 120),         # Turn left
        (("RIGHT", "B"), 60),     # Dash right (simplified, no arm-pump for now)
    ]

    inputs: list[FrameInput] = []
    for button_names, frame_count in tape_frames:
        mask = button_names_to_mask(button_names)
        for _ in range(frame_count):
            inputs.append(FrameInput(buttons=mask))

    return CeresFirstRoomFixture(
        from_room_id=ROOM_CERES_ELEVATOR,
        to_room_id=ROOM_CERES_FALLING,
        inputs=tuple(inputs),
        tape_source="routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans (prefix)",
        emulator_validated=False,
    )


def validate_ceres_first_room(
    fixture: CeresFirstRoomFixture,
    start_state_path: Path | str | None = None,
) -> CeresFirstRoomFixture:
    """Validate fixture on real emulator (ground truth).

    Runs tape on stable-retro / SMEDIT snes9x. This is the authoritative
    validation path for room-clear claims.

    Args:
        fixture: Tape fixture to validate
        start_state_path: Path to Ceres Elevator start state (optional)
            If None, uses env var SM_CERES_ELEV_STATE

    Returns:
        Updated fixture with emulator validation results

    Raises:
        FileNotFoundError: If ROM or start state not available
        RuntimeError: If emulator fails to load
        ValueError: If start state path not provided and env var not set

    Note:
        Tests skip validation if:
        - ROM_AVAILABLE is False
        - SM_CERES_ELEV_STATE env var not set
        - Start state file does not exist
    """
    if start_state_path is None:
        start_state_path = os.environ.get("SM_CERES_ELEV_STATE")
        if not start_state_path:
            raise ValueError(
                "start_state_path not provided and SM_CERES_ELEV_STATE not set"
            )

    if not Path(start_state_path).exists():
        raise FileNotFoundError(f"Start state not found: {start_state_path}")

    result = validate_trajectory_on_emulator(
        start_state_path,
        fixture.inputs,
        target_room_id=fixture.to_room_id,
    )

    return CeresFirstRoomFixture(
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
