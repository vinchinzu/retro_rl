"""Explicit assist contracts for SMZ3 Bronze/dev routes.

Assists help *reach* a verified natural transition; they are not part of the
combo teleport itself. Publish results must declare any assist used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AssistContract:
    """One named, auditable assist."""

    assist_id: str
    description: str
    class_: str = "dev"  # dev | bronze | clean
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assist_id": self.assist_id,
            "description": self.description,
            "class": self.class_,
            "notes": self.notes,
        }


# Red door $8976 needs missiles; natural morph → first-missile not yet on combo.
MISSILE_RED_DOOR_ASSIST = AssistContract(
    assist_id="missile_red_door",
    description=(
        "Grant missile capacity/ammo and select missiles so Parlor red door "
        "$8976 can open before natural morph → missiles is wired."
    ),
    class_="dev",
    notes="Does not poke Z3 RAM after portal. Drop when natural missile path lands.",
)


def grant_missiles(env: Any, *, count: int = 20) -> None:
    """Apply :data:`MISSILE_RED_DOOR_ASSIST` (SM WRAM only)."""
    from super_metroid.ram import write_wram_u16

    write_wram_u16(env, 0x09C8, count)  # max
    write_wram_u16(env, 0x09C6, count)  # current
    write_wram_u16(env, 0x09D2, 1)  # select missiles
