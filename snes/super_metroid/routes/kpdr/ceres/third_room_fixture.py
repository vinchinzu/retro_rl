"""Ceres magnet → scientist tape (slice of product outbound prefix)."""

from __future__ import annotations

from pathlib import Path

from super_metroid.routes.kpdr.ceres.room_tape import (
    CeresRoomTape,
    button_names_to_mask,
    get_ceres_room_tape,
    validate_ceres_room_tape,
)

__all__ = [
    "CeresThirdRoomFixture",
    "get_ceres_third_room_tape",
    "validate_ceres_third_room",
    "button_names_to_mask",
]

CeresThirdRoomFixture = CeresRoomTape


def get_ceres_third_room_tape() -> CeresRoomTape:
    return get_ceres_room_tape("third")


def validate_ceres_third_room(
    fixture: CeresRoomTape,
    start_state_path: Path | str | None = None,
) -> CeresRoomTape:
    return validate_ceres_room_tape(fixture, "third", start_state_path)
