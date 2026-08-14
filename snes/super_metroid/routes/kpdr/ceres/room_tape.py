"""Shared Ceres room-1..3 tape: slices of the product outbound prefix.

Rooms 1–3 are fixed prefixes of ``CERES_OUTBOUND_TO_SCIENTIST_RAW``.
Room 4 (Scientist → Ridley) is room-gated and lives in fourth_room_fixture.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX
from super_metroid.emulator_validation import validate_trajectory_on_emulator
from super_metroid.physics_sim import FrameInput
from super_metroid.routes.kpdr.ceres.outbound import (
    CERES_OUTBOUND_TO_SCIENTIST_RAW,
    expand_ceres_outbound_spans,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.runtime import ActionSpan

__all__ = [
    "CeresRoomTape",
    "button_names_to_mask",
    "get_ceres_room_tape",
    "validate_ceres_room_tape",
    "CERES_ROOM_SLICES",
]


def button_names_to_mask(names: tuple[str, ...]) -> int:
    """Convert SNES button names to a FrameInput packed mask."""
    mask = 0
    for name in names:
        idx = SNES_BUTTON_NAME_TO_INDEX.get(name.strip().upper())
        if idx is not None:
            mask |= 1 << idx
    return mask


def spans_to_inputs(spans: list[ActionSpan]) -> tuple[FrameInput, ...]:
    return tuple(
        FrameInput(buttons=button_names_to_mask(span.names))
        for span in spans
        for _ in range(span.frames)
    )


@dataclass(frozen=True)
class CeresRoomTape:
    """Fixed-tape Ceres room hop (product outbound prefix, not search)."""

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
        return {
            **asdict(self),
            "inputs": [
                {"buttons": inp.buttons, "frame": i}
                for i, inp in enumerate(self.inputs)
            ],
        }

    @property
    def frames(self) -> int:
        return len(self.inputs)

    @property
    def room_clear(self) -> bool:
        return self.emulator_validated and self.emulator_success


# raw-table slices + env var for the enter pin
CERES_ROOM_SLICES: dict[str, tuple[int, int, int, int, str]] = {
    "first": (0, 5, ROOM_CERES_ELEVATOR, ROOM_CERES_FALLING, "SM_CERES_ELEV_STATE"),
    "second": (5, 18, ROOM_CERES_FALLING, ROOM_CERES_MAGNET, "SM_CERES_FALLING_STATE"),
    "third": (18, 21, ROOM_CERES_MAGNET, ROOM_CERES_SCIENTIST, "SM_CERES_MAGNET_STATE"),
}


def get_ceres_room_tape(which: str) -> CeresRoomTape:
    start, end, from_id, to_id, _env = CERES_ROOM_SLICES[which]
    raw = CERES_OUTBOUND_TO_SCIENTIST_RAW[start:end]
    spans = expand_ceres_outbound_spans(raw, reason=f"ceres_{which}_room")
    inputs = spans_to_inputs(spans)
    return CeresRoomTape(
        from_room_id=from_id,
        to_room_id=to_id,
        inputs=inputs,
        tape_source=(
            "routes.kpdr.ceres.outbound.CERES_OUTBOUND_TO_SCIENTIST_RAW "
            f"(slice {start}:{end}, {len(inputs)} frames)"
        ),
    )


def validate_ceres_room_tape(
    fixture: CeresRoomTape,
    which: str,
    start_state_path: Path | str | None = None,
) -> CeresRoomTape:
    _start, _end, _from_id, _to_id, env_var = CERES_ROOM_SLICES[which]
    if start_state_path is None:
        start_state_path = os.environ.get(env_var)
        if not start_state_path:
            raise ValueError(
                f"start_state_path not provided and {env_var} not set"
            )
    if not Path(start_state_path).exists():
        raise FileNotFoundError(f"Start state not found: {start_state_path}")

    result = validate_trajectory_on_emulator(
        start_state_path,
        fixture.inputs,
        target_room_id=fixture.to_room_id,
    )
    return CeresRoomTape(
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
