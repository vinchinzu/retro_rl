"""Combo-ROM WRAM reads for SMZ3 (shared $7E layout, dual interpretation).

While Super Metroid is active, low WRAM matches vanilla SM addresses used by
``super_metroid.ram``. While ALttP is active, the same bytes match vanilla Z3
addresses used by ``alttp.ram``. World detection chooses which parser to trust.

SRAM_CURRENT_GAME (combo flag at bus ``$A1:73FE``) is documented for a future
direct read once stable-retro maps cart SRAM; until then we classify from WRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# --- Super Metroid (vanilla offsets; verified on combo at Landing Site) ---
SM_ADDR_ROOM_ID = 0x079B
SM_ADDR_AREA_INDEX = 0x079F
SM_ADDR_DOOR_TRANSITION = 0x0797
SM_ADDR_GAME_STATE = 0x0998
SM_ADDR_HEALTH = 0x09C2
SM_ADDR_MAX_HEALTH = 0x09C4
SM_ADDR_SAMUS_X = 0x0AF6
SM_ADDR_SAMUS_Y = 0x0AFA
SM_ADDR_POSE = 0x0A1C

# Ordinary controllable gameplay (same as super_metroid.room_timer).
SM_ORDINARY_GAME_STATE = 8

# --- ALttP (vanilla offsets) ---
Z3_ADDR_MODULE = 0x10
Z3_ADDR_SUBMODULE = 0x11
Z3_ADDR_INDOORS = 0x1B
Z3_ADDR_LINK_Y = 0x20
Z3_ADDR_LINK_X = 0x22
Z3_ADDR_ROOM_ID = 0xA0
Z3_ADDR_SCREEN_ID = 0x8A

# Combo randomizer (tewtal alttp_sm_combo_randomizer_rom src/sram.asm):
#   !SRAM_CURRENT_GAME = $a173fe
# NMI (8-bit A): 0 = ALTTP, negative ($80-$FF) = SM, positive nonzero = credits.
# Reset stores #$00FF (SM). Not yet exposed via stable-retro get_ram() / blocks.
SRAM_CURRENT_GAME_BUS = 0xA173FE

# SM game states treated as title / file select / boot (see super_metroid.ram).
SM_MENU_GAME_STATES = frozenset({0, 1, 2, 3, 4, 5, 6, 30, 31, 40, 41, 42, 43, 44})
# States that mean SM engine is driving the machine (not Z3).
SM_ENGINE_GAME_STATES = frozenset(range(0, 45))

# Z3 modules that indicate the ALTTP engine owns the frame.
# 0x07 dungeon, 0x09 overworld, 0x0E text, plus common transitions.
Z3_ACTIVE_MODULES = frozenset(
    {
        0x06,  # underworld load-ish
        0x07,  # dungeon
        0x08,  # underworld special
        0x09,  # overworld
        0x0A,  # overworld special
        0x0B,  # overworld special / mirror
        0x0C,
        0x0E,  # text / messaging
        0x0F,  # SM→Z3 portal mid-transition (observed)
        0x10,
        0x11,
        0x12,
    }
)
Z3_MENU_MODULES = frozenset({0x00, 0x01, 0x02, 0x03, 0x04, 0x05})


def _u8(ram: np.ndarray | bytes | bytearray, address: int) -> int:
    return int(ram[address]) & 0xFF


def _u16(ram: np.ndarray | bytes | bytearray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def read_env_ram(env: Any) -> np.ndarray:
    """Return a uint8 view of ``env.get_ram()`` (stable-retro WRAM slice)."""
    return np.asarray(env.get_ram(), dtype=np.uint8)


@dataclass(frozen=True)
class ComboSnapshot:
    """Minimal dual-world fields from one frame of combo WRAM."""

    frame: int
    sm_game_state: int
    sm_room_id: int
    sm_area_index: int
    sm_door_transition: int
    sm_health: int
    sm_max_health: int
    sm_samus_x: int
    sm_samus_y: int
    sm_pose: int
    z3_module: int
    z3_submodule: int
    z3_indoors: int
    z3_room_id: int
    z3_screen_id: int
    z3_link_x: int
    z3_link_y: int

    @property
    def sm_controllable(self) -> bool:
        """True when SM ordinary gameplay is settled (game_state 8, no door)."""
        return (
            self.sm_game_state == SM_ORDINARY_GAME_STATE
            and self.sm_door_transition == 0
            and self.sm_room_id != 0
            and self.sm_health > 0
        )

    @property
    def z3_controllable(self) -> bool:
        """True when Link has overworld/dungeon control (module 7/9, sub 0)."""
        return self.z3_module in (0x07, 0x09) and self.z3_submodule == 0x00

    @property
    def sm_room_plausible(self) -> bool:
        """SM room IDs are ROM pointers; headers sit in high banks (~$79xx+)."""
        return self.sm_room_id >= 0x7900

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "sm_game_state": self.sm_game_state,
            "sm_room_id": self.sm_room_id,
            "sm_room_id_hex": f"0x{self.sm_room_id:04X}",
            "sm_area_index": self.sm_area_index,
            "sm_door_transition": self.sm_door_transition,
            "sm_health": self.sm_health,
            "sm_max_health": self.sm_max_health,
            "sm_samus_x": self.sm_samus_x,
            "sm_samus_y": self.sm_samus_y,
            "sm_pose": self.sm_pose,
            "sm_controllable": self.sm_controllable,
            "z3_module": self.z3_module,
            "z3_submodule": self.z3_submodule,
            "z3_indoors": self.z3_indoors,
            "z3_room_id": self.z3_room_id,
            "z3_screen_id": self.z3_screen_id,
            "z3_link_x": self.z3_link_x,
            "z3_link_y": self.z3_link_y,
            "z3_controllable": self.z3_controllable,
        }


def read_snapshot(ram: np.ndarray | bytes | bytearray, *, frame: int = 0) -> ComboSnapshot:
    """Parse dual SM/Z3 fields from a WRAM-like buffer."""
    return ComboSnapshot(
        frame=frame,
        sm_game_state=_u16(ram, SM_ADDR_GAME_STATE),
        sm_room_id=_u16(ram, SM_ADDR_ROOM_ID),
        sm_area_index=_u16(ram, SM_ADDR_AREA_INDEX),
        sm_door_transition=_u16(ram, SM_ADDR_DOOR_TRANSITION),
        sm_health=_u16(ram, SM_ADDR_HEALTH),
        sm_max_health=_u16(ram, SM_ADDR_MAX_HEALTH),
        sm_samus_x=_u16(ram, SM_ADDR_SAMUS_X),
        sm_samus_y=_u16(ram, SM_ADDR_SAMUS_Y),
        sm_pose=_u16(ram, SM_ADDR_POSE),
        z3_module=_u8(ram, Z3_ADDR_MODULE),
        z3_submodule=_u8(ram, Z3_ADDR_SUBMODULE),
        z3_indoors=_u8(ram, Z3_ADDR_INDOORS),
        z3_room_id=_u16(ram, Z3_ADDR_ROOM_ID),
        z3_screen_id=_u8(ram, Z3_ADDR_SCREEN_ID),
        z3_link_x=_u16(ram, Z3_ADDR_LINK_X),
        z3_link_y=_u16(ram, Z3_ADDR_LINK_Y),
    )


def snapshot_env(env: Any, *, frame: int = 0) -> ComboSnapshot:
    return read_snapshot(read_env_ram(env), frame=frame)
