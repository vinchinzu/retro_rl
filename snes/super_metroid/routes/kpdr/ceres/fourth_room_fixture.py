"""Fourth Ceres room (Scientist → Flat) room-gated fixture.

This fixture represents the room-gated arm-pump from Scientist 0xE021 → Flat
0xE06B. Unlike rooms 1-3, there is no fixed tape in the source code for this
segment. The product code (`routes.kpdr.ceres.outbound.play_ceres_to_ridley_door`)
treats Scientist→Flat→Ridley as one continuous room-gated arm-pump (line 90-99).

Tape Source:
- Room-gated behavior from `_ceres_arm_pump_until` in arm_pump.py
- Product code stops at Ridley (0xE0B5), not Flat
- Reason "ceres_out_flat_band" suggests Flat is part of a continuous band
- This fixture estimates Scientist→Flat segment using same arm-pump pattern

Implementation:
- RIGHT direction with classic L↔R arm-pump (period=2)
- Estimated ~300 frames based on typical room-gated behavior
- Uses `_arm_pump_dash_spans` helper (same as product code)
- Not extracted from fixed tape; represents adaptive room-gated behavior

Policy:
- Predictor (StubPredictor / sm_rev_predict) = search speed only
- Emulator (stable-retro / SMEDIT snes9x) = ground truth
- Room-clear claims require emulator validation

Validation:
- Start state: env var `SM_CERES_SCIENTIST_STATE` (path to .state file on disk)
- Tests skip without ROM_AVAILABLE or missing start state
- Never commit .state or ROM blobs to repo

Note:
    Product tape treats Scientist→Flat→Ridley as one continuous segment.
    This fixture creates an artificial stop at Flat for modular testing.
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
    ROOM_CERES_FLAT,
    ROOM_CERES_SCIENTIST,
)

__all__ = [
    "CeresFourthRoomFixture",
    "get_ceres_fourth_room_tape",
    "validate_ceres_fourth_room",
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
class CeresFourthRoomFixture:
    """Fourth Ceres room fixture (room-gated, not fixed tape).

    Represents Scientist 0xE021 → Flat 0xE06B using classic arm-pump pattern.
    Product code treats this as part of continuous Scientist→Flat→Ridley band.

    Emulator validation uses env var SM_CERES_SCIENTIST_STATE for start state.
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


def get_ceres_fourth_room_tape() -> CeresFourthRoomFixture:
    """Get tape for Ceres Scientist → Flat (room-gated arm-pump pattern).

    This tape represents the room-gated behavior from product code's
    `_ceres_arm_pump_until` call (outbound.py lines 92-99). Product stops at
    Ridley; this fixture stops at Flat for modular room testing.

    Estimated frame count (~300) based on:
    - Product comment: Scientist→Flat→Ridley drops ~600f of old tape
    - Max frames allowed: 900
    - Typical single-room arm-pump: 100-400 frames

    Returns:
        CeresFourthRoomFixture with estimated tape (emulator_validated=False)

    Note:
        This is NOT extracted from fixed product tape. Product uses room-gated
        `_ceres_arm_pump_until(done=lambda s: s.room_id == ROOM_CERES_RIDLEY)`.
        Frame count is estimated; emulator validation determines actual success.
    """
    # Estimated frame count for Scientist → Flat segment
    # Product code: "drops ~600f" for full Scientist→Ridley
    # This fixture: ~300 frames for Scientist→Flat (roughly half)
    estimated_frames = 300

    inputs: list[FrameInput] = []

    # Classic RIGHT+B arm-pump with L↔R pattern (same as product)
    arm_pump_spans = _arm_pump_dash_spans(
        "RIGHT", estimated_frames, "ceres_fourth_room"
    )
    for span in arm_pump_spans:
        mask = button_names_to_mask(span.names)
        for _ in range(span.frames):
            inputs.append(FrameInput(buttons=mask))

    return CeresFourthRoomFixture(
        from_room_id=ROOM_CERES_SCIENTIST,
        to_room_id=ROOM_CERES_FLAT,
        inputs=tuple(inputs),
        tape_source=(
            "room-gated arm-pump pattern (estimated 300f) — product code "
            "treats Scientist→Flat→Ridley as continuous segment in "
            "routes.kpdr.ceres.outbound.play_ceres_to_ridley_door"
        ),
        emulator_validated=False,
    )


def validate_ceres_fourth_room(
    fixture: CeresFourthRoomFixture,
    start_state_path: Path | str | None = None,
) -> CeresFourthRoomFixture:
    """Validate fixture on real emulator (ground truth).

    Runs tape on stable-retro / SMEDIT snes9x. This is the authoritative
    validation path for room-clear claims.

    Args:
        fixture: Tape fixture to validate
        start_state_path: Path to Ceres Scientist start state (optional)
            If None, uses env var SM_CERES_SCIENTIST_STATE

    Returns:
        Updated fixture with emulator validation results

    Raises:
        FileNotFoundError: If ROM or start state not available
        RuntimeError: If emulator fails to load
        ValueError: If start state path not provided and env var not set

    Note:
        Tests skip validation if:
        - ROM_AVAILABLE is False
        - SM_CERES_SCIENTIST_STATE env var not set
        - Start state file does not exist
    """
    if start_state_path is None:
        start_state_path = os.environ.get("SM_CERES_SCIENTIST_STATE")
        if not start_state_path:
            raise ValueError(
                "start_state_path not provided and "
                "SM_CERES_SCIENTIST_STATE not set"
            )

    if not Path(start_state_path).exists():
        raise FileNotFoundError(f"Start state not found: {start_state_path}")

    result = validate_trajectory_on_emulator(
        start_state_path,
        fixture.inputs,
        target_room_id=fixture.to_room_id,
    )

    return CeresFourthRoomFixture(
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
