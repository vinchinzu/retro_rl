"""Hop identity: stable ``hop_key`` and inventory/room parsers.

``hop_key`` is direction-aware::

    {room_hex}:{from_room_hex|start}->{to_room_hex|goal}:{items_hex}

Parsers live in ``human_tape.anchors``; this module is the skill-bank /
PB-board entry point so callers need not import the full tape package.

``make_hop_key`` is defined before the anchors import so
``human_tape`` → ``pb_board`` → ``hop_id`` can resolve mid-cycle.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "make_hop_key",
    "parse_items",
    "parse_items_value",
    "parse_room_id",
]


def make_hop_key(
    room_id: int,
    *,
    from_room_id: int | None = None,
    to_room_id: int | None = None,
    items: int | None = None,
    goal: str | None = None,
) -> str:
    """Stable skill identity for PB / bank lookup."""
    room = f"0x{int(room_id):04X}"
    src = f"0x{int(from_room_id):04X}" if from_room_id else "start"
    if goal:
        dest = goal
    elif to_room_id is not None:
        dest = f"0x{int(to_room_id):04X}"
    else:
        dest = "leave"
    inv = f"0x{int(items):04X}" if items is not None else "any"
    return f"{room}:{src}->{dest}:{inv}"


# anchors must not import hop_id / skill_bank (verified).
from super_metroid.human_tape.anchors import parse_items_value, parse_room_id  # noqa: E402

parse_items = parse_items_value
