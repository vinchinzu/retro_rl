"""SMZ3 cross-world portals (fixed; not randomized).

Source: tewtal ``alttp_sm_combo_randomizer_rom`` ``src/sm/teleport.asm`` /
``src/z3/teleport.asm`` and https://samus.link/information.

Closest early portal from Zebes start:

  Landing Site → Parlor → (red door, missiles) Pre-Map Flyway → Crateria Map Room
  → portal door ``$8976`` → ALttP Lake Hylia Fortune Teller (cave ``$0122``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Portal:
    """One fixed SM ↔ Z3 link."""

    portal_id: str
    sm_name: str
    sm_room_id: int | None
    sm_door_ptr: int
    z3_name: str
    z3_cave_id: int
    dark_world: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "portal_id": self.portal_id,
            "sm_name": self.sm_name,
            "sm_room_id": self.sm_room_id,
            "sm_room_id_hex": (
                f"0x{self.sm_room_id:04X}" if self.sm_room_id is not None else None
            ),
            "sm_door_ptr": f"0x{self.sm_door_ptr:04X}",
            "z3_name": self.z3_name,
            "z3_cave_id": f"0x{self.z3_cave_id:04X}",
            "dark_world": self.dark_world,
            "notes": self.notes,
        }


# Door pointers are SM ``$078D`` values from the combo teleport tables.
PORTALS: tuple[Portal, ...] = (
    Portal(
        portal_id="crateria_map_fortune_teller",
        sm_name="Crateria Map Room",
        sm_room_id=0x9994,
        sm_door_ptr=0x8976,
        z3_name="Lake Hylia Fortune Teller",
        z3_cave_id=0x0122,
        dark_world=False,
        notes=(
            "Earliest SM-side portal. Path: Landing Site (0x91F8) → Parlor "
            "(0x92FD) → red door (missiles) → Pre-Map Flyway (0x98E2) → Map "
            "(0x9994). Z3 return door to Parlor uses SM door $8BCE."
        ),
    ),
    Portal(
        portal_id="norfair_map_old_man",
        sm_name="Norfair Map Room",
        sm_room_id=None,
        sm_door_ptr=0x9306,
        z3_name="Death Mountain Old Man Cave (back)",
        z3_cave_id=0x00E5,
        dark_world=False,
        notes="Mid-game portal; needs Norfair access.",
    ),
    Portal(
        portal_id="maridia_missile_ice_rod",
        sm_name="Maridia Missile Refill",
        sm_room_id=None,
        sm_door_ptr=0xA8F4,
        z3_name="Dark World Ice Rod Cave (right)",
        z3_cave_id=0x010E,
        dark_world=True,
        notes="Dark World portal.",
    ),
    Portal(
        portal_id="ln_refill_mire_fairy",
        sm_name="Lower Norfair Energy Refill (GT)",
        sm_room_id=None,
        sm_door_ptr=0x0000,  # NewLNRefillDoorData_exit — ROM-local label
        z3_name="Misery Mire right fairy",
        z3_cave_id=0x0115,
        dark_world=True,
        notes="Door ptr is a combo-asm label; resolve from built IPS if needed.",
    ),
)

PORTALS_BY_ID: dict[str, Portal] = {p.portal_id: p for p in PORTALS}

# Early SM rooms on the map-portal approach (power-on natural order).
EARLY_SM_ROOMS: dict[int, str] = {
    0x91F8: "Landing Site",
    0x92FD: "Parlor and Alcatraz",
    0x98E2: "Pre-Map Flyway",
    0x9994: "Crateria Map Room",
    0x96BA: "Climb",  # morph route branch, not portal path
    0x95FF: "Terminator Room",  # near parlor morph path
}

# Z3 cave id written when Crateria map portal fires (verified mid-transition).
FORTUNE_TELLER_CAVE_ID = 0x0122


def room_name(room_id: int) -> str:
    return EARLY_SM_ROOMS.get(room_id, f"room_0x{room_id:04X}")


def early_portal() -> Portal:
    return PORTALS_BY_ID["crateria_map_fortune_teller"]


def portals_to_dict() -> list[dict[str, Any]]:
    return [p.to_dict() for p in PORTALS]
