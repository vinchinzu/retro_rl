"""Enemy slot scan (``$0F78 + i*$40``).

One scanner for every room overlay. Hops do not re-encode the layout.
Unknown species are still listed — Stance decides what to do with them.
"""

from __future__ import annotations

from typing import Any, NamedTuple

ENEMY_BASE = 0x0F78
ENEMY_STRIDE = 0x40
# Super Metroid reserves 32 entries in the room-enemy table. Scan the whole
# table even though the first Species catalog contains only three rows: slot
# coverage and Species coverage are separate concerns.
MAX_ENEMY_SLOTS = 32
OFF_MAP = 0xFE00


class Enemy(NamedTuple):
    """One live room slot. Not a Boss. Frozen is not dead."""

    slot: int
    enemy_id: int
    x: int
    y: int
    hp: int
    freeze_timer: int


def _u16(ram: Any, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def ram_from(source: object) -> Any | None:
    """Session ``env.get_ram()``, a RAM buffer, or None."""
    env = getattr(source, "env", None)
    if env is not None:
        get_ram = getattr(env, "get_ram", None)
        if get_ram is None:
            return None
        return get_ram()
    if source is None:
        return None
    try:
        _ = source[0]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None
    return source


def enemies_from_ram(ram: Any) -> tuple[Enemy, ...]:
    """Scan slots. Drops empty, dead, and off-map."""
    out: list[Enemy] = []
    for slot in range(MAX_ENEMY_SLOTS):
        base = ENEMY_BASE + slot * ENEMY_STRIDE
        try:
            enemy_id = _u16(ram, base)
        except (IndexError, TypeError):
            break
        if enemy_id == 0:
            continue
        hp = _u16(ram, base + 0x14)
        if hp <= 0:
            continue
        x = _u16(ram, base + 0x02)
        y = _u16(ram, base + 0x06)
        if x >= OFF_MAP or y >= OFF_MAP:
            continue
        out.append(
            Enemy(
                slot=slot,
                enemy_id=enemy_id,
                x=x,
                y=y,
                hp=hp,
                freeze_timer=_u16(ram, base + 0x26),
            )
        )
    return tuple(out)


def list_enemies(source: object) -> tuple[Enemy, ...]:
    """Scan a session or a RAM buffer. Empty when RAM is missing."""
    ram = ram_from(source)
    if ram is None:
        return ()
    return enemies_from_ram(ram)


__all__ = [
    "ENEMY_BASE",
    "ENEMY_STRIDE",
    "MAX_ENEMY_SLOTS",
    "Enemy",
    "enemies_from_ram",
    "list_enemies",
    "ram_from",
]
