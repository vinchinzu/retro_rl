"""Ceres elevator → falling-tile tape (slice of product outbound prefix)."""

from __future__ import annotations

from pathlib import Path

from super_metroid.routes.kpdr.ceres.room_tape import (
    CeresRoomTape,
    button_names_to_mask,
    get_ceres_room_tape,
    validate_ceres_room_tape,
)

__all__ = [
    "CeresFirstRoomFixture",
    "get_ceres_first_room_tape",
    "validate_ceres_first_room",
    "button_names_to_mask",
]

CeresFirstRoomFixture = CeresRoomTape


def get_ceres_first_room_tape() -> CeresRoomTape:
    return get_ceres_room_tape("first")


def validate_ceres_first_room(
    fixture: CeresRoomTape,
    start_state_path: Path | str | None = None,
) -> CeresRoomTape:
    return validate_ceres_room_tape(fixture, "first", start_state_path)
