"""Ordered observation builders shared by SM-rando corpus and trainers."""

from __future__ import annotations

import numpy as np


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


__all__ = ["landing_entry_features"]
