"""MK2 SNES health layout and RAM-gated match result.

``get_ram()`` index = WRAM + ``GETRAM_OFFSET`` (0x2001). Health lives in high
WRAM, so it cannot go in ``data.json`` (stable-retro maps only the first 8KB).

Confirmed on ``Fight_LiuKang``: P1/P2 start at 161. 0x020A/0x020E are
transitional bytes — **not** health.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_HEALTH = 161
GETRAM_OFFSET = 0x2001
WRAM_P1_HEALTH = 0x2EFC
WRAM_P2_HEALTH = 0x30AA
ADDR_P1_HEALTH = WRAM_P1_HEALTH + GETRAM_OFFSET  # 0x4EFD
ADDR_P2_HEALTH = WRAM_P2_HEALTH + GETRAM_OFFSET  # 0x50AB
PAR_P1_HEALTH = 0x7E0000 + WRAM_P1_HEALTH
PAR_P2_HEALTH = 0x7E0000 + WRAM_P2_HEALTH
# Low WRAM decoys previously claimed as health in fighters/AGENTS.md.
DECOY_NOT_HEALTH = (0x020A, 0x020E)


@dataclass(frozen=True)
class FightSnapshot:
    """Health-only snapshot used to gate match-win on KO zero-crossings."""

    p1_health: int
    p2_health: int
    ram_len: int


def _u8(ram: np.ndarray, addr: int) -> int:
    if addr < 0 or addr >= len(ram):
        return 0
    return int(ram[addr]) & 0xFF


def parse_ram(ram: np.ndarray) -> FightSnapshot:
    """Parse a ``get_ram()`` buffer. Ignores 0x020A/0x020E decoys."""
    return FightSnapshot(
        p1_health=_u8(ram, ADDR_P1_HEALTH),
        p2_health=_u8(ram, ADDR_P2_HEALTH),
        ram_len=int(len(ram)),
    )


def is_match_won(p1_kos: int, p2_kos: int) -> bool:
    """Best-of-three, strict majority (2-2 is not a win)."""
    return p1_kos >= 2 and p1_kos > p2_kos


def is_match_lost(p1_kos: int, p2_kos: int) -> bool:
    """True when P2 has taken the match (strict majority)."""
    return p2_kos >= 2 and p2_kos > p1_kos


def make_test_ram(**fields: int) -> np.ndarray:
    """Synthetic WRAM buffer large enough for high-WRAM health."""
    ram = np.zeros(ADDR_P2_HEALTH + 1, dtype=np.uint8)
    ram[ADDR_P1_HEALTH] = int(fields.get("p1_health", MAX_HEALTH))
    ram[ADDR_P2_HEALTH] = int(fields.get("p2_health", MAX_HEALTH))
    ram[DECOY_NOT_HEALTH[0]] = int(fields.get("decoy_020a", MAX_HEALTH))
    ram[DECOY_NOT_HEALTH[1]] = int(fields.get("decoy_020e", MAX_HEALTH))
    return ram
