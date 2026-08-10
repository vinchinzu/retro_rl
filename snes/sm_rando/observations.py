"""Ordered observation builders shared by SM-rando corpus and trainers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

LANDING_ENTRY_METADATA_VERSION = 2
LANDING_ENTRY_FEATURE_KEYS = (
    "room_id",
    "game_state",
    "door_transition",
    "samus_x",
    "samus_x_sub",
    "samus_y",
    "samus_y_sub",
    "velocity_x",
    "velocity_y",
    "health",
    "missiles",
    "pose",
)


def _u16(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def landing_entry_features(
    ram: np.ndarray,
    *,
    prev_action: list[int] | None = None,
) -> np.ndarray:
    """Small ordered SM boundary vector used by the corpus/BC contract."""
    del prev_action
    return np.asarray(
        (
            _u16(ram, 0x079B),  # room
            _u16(ram, 0x0998),  # game state
            _u16(ram, 0x0797),  # door transition
            _u16(ram, 0x0AF6),  # x
            _u16(ram, 0x0AF8),  # x subpixel
            _u16(ram, 0x0AFA),  # y
            _u16(ram, 0x0AFC),  # y subpixel
            _u16(ram, 0x0B42),  # x velocity
            _u16(ram, 0x0B2E),  # y velocity
            _u16(ram, 0x09C2),  # energy
            _u16(ram, 0x09C6),  # missiles
            _u16(ram, 0x0A1C),  # pose
        ),
        dtype=np.float32,
    )


def landing_entry_features_from_metadata(
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Apply the explicit v2 retained-metadata → observation migration."""
    version = metadata.get("observation_metadata_version")
    if version != LANDING_ENTRY_METADATA_VERSION:
        raise ValueError(
            "Landing entry metadata is not v2; re-harvest or run the explicit "
            "state-backed metadata migration"
        )
    missing = [key for key in LANDING_ENTRY_FEATURE_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"Landing entry metadata v2 lacks fields: {missing}")
    return np.asarray(
        tuple(metadata[key] for key in LANDING_ENTRY_FEATURE_KEYS),
        dtype=np.float64,
    )


__all__ = [
    "LANDING_ENTRY_FEATURE_KEYS",
    "LANDING_ENTRY_METADATA_VERSION",
    "landing_entry_features",
    "landing_entry_features_from_metadata",
]
