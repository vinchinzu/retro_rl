"""Declarative RAM reading for SNES games via stable-retro.

Provides typed value readers for numpy RAM arrays, a declarative RAMSchema
for mapping field names to addresses, and a RAMWatcher for detecting
value changes between frames.
"""
from __future__ import annotations

import numpy as np

# -- Typed value readers -----------------------------------------------------

def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])

def read_u16(ram: np.ndarray, addr: int) -> int:
    """Little-endian unsigned 16-bit read."""
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)

def read_u16_be(ram: np.ndarray, addr: int) -> int:
    """Big-endian unsigned 16-bit read."""
    return (int(ram[addr]) << 8) | int(ram[addr + 1])

def read_s8(ram: np.ndarray, addr: int) -> int:
    v = int(ram[addr])
    return v - 256 if v > 127 else v

def read_s16(ram: np.ndarray, addr: int) -> int:
    """Little-endian signed 16-bit read."""
    v = read_u16(ram, addr)
    return v - 65536 if v > 32767 else v

_READERS = {
    "u8": read_u8,
    "u16": read_u16,
    "u16_be": read_u16_be,
    "s8": read_s8,
    "s16": read_s16,
}

# -- RAMSchema ---------------------------------------------------------------

class RAMSchema:
    """Declarative RAM address map.

    Usage::

        schema = RAMSchema({
            "player_x": (0x00D6, "u8"),
            "player_y": (0x00D8, "u8"),
            "level_id": (0x0076, "u16"),
            "health":   (0x04B9, "u16_be"),
        })
        values = schema.read(ram_array)
        # {"player_x": 42, "player_y": 100, "level_id": 233, "health": 161}

    Supported types: ``"u8"``, ``"u16"``, ``"u16_be"``, ``"s8"``, ``"s16"``
    """

    def __init__(self, addresses: dict[str, tuple[int, str]]) -> None:
        for name, (addr, type_str) in addresses.items():
            if type_str not in _READERS:
                raise ValueError(f"Unknown type {type_str!r} for field {name!r}")
        self._addresses = dict(addresses)

    @classmethod
    def from_dict(cls, d: dict[str, tuple[int, str]]) -> RAMSchema:
        """Create a RAMSchema from a plain dict."""
        return cls(d)

    @property
    def fields(self) -> list[str]:
        """Return the list of field names."""
        return list(self._addresses)

    def read(self, ram: np.ndarray) -> dict[str, int]:
        """Read all fields from *ram* and return a name -> value dict."""
        return {
            name: _READERS[type_str](ram, addr)
            for name, (addr, type_str) in self._addresses.items()
        }

    def read_one(self, ram: np.ndarray, field_name: str) -> int:
        """Read a single field by name."""
        addr, type_str = self._addresses[field_name]
        return _READERS[type_str](ram, addr)

# -- RAMWatcher --------------------------------------------------------------

class RAMWatcher:
    """Track RAM value changes between frames.

    Usage::

        watcher = RAMWatcher(schema)
        changes = watcher.update(ram)
        # {"health": (161, 150), "level_id": (1, 2)}  -- (old, new)
    """

    def __init__(self, schema: RAMSchema) -> None:
        self._schema = schema
        self._prev: dict[str, int] | None = None

    def update(self, ram: np.ndarray) -> dict[str, tuple[int, int]]:
        """Return fields that changed since the last call.

        Returns ``{field_name: (old_value, new_value)}`` for every field whose
        value differs.  First call returns an empty dict (no previous state).
        """
        current = self._schema.read(ram)
        if self._prev is None:
            self._prev = current
            return {}
        changes = {
            name: (self._prev[name], current[name])
            for name in current
            if current[name] != self._prev[name]
        }
        self._prev = current
        return changes
