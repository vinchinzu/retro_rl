"""ALTTP WRAM readers and readiness helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# stable-retro get_ram() index mapping:
#   WRAM offset < 0x2000 → index = offset (block 0 mirror)
#   WRAM offset >= 0x2000 → index = 16384 + offset
WRAM_IDX = 16384

MODULE = 0x10
SUBMODULE = 0x11
INDOORS = 0x1B
ROOM_ID = 0xA0
LINK_Y = 0x20
LINK_X = 0x22
LINK_DIRECTION = 0x2F
LINK_ACTION = 0x5D
SCREEN_ID = 0x008A
DARK_WORLD_FLAG = 0x0FFF
BG2_VOFS = 0x00E8
CAMERA_X = 0xE2
LINK_HP = 0xF36D
LINK_MAX_HP = 0xF36C
EQUIP_SWORD = 0xF359

HYRULE_CASTLE_SCREEN = 0x1B
LINKS_HOUSE_SCREEN = 0x2C
LINKS_HOUSE_ROOM = 0x0004


@dataclass(frozen=True)
class AlttpSnapshot:
    """Frame snapshot of the fields needed for title→castle routing."""

    game_mode: int
    submodule: int
    room_id: int
    indoors: bool
    screen_id: int
    link_x: int
    link_y: int
    link_direction: int
    link_action: int
    camera_x: int
    camera_y: int
    dark_world: bool

    @property
    def room_base_id(self) -> int:
        return int(self.room_id) & 0x00FF

    @property
    def has_control(self) -> bool:
        return self.game_mode in (0x07, 0x09) and self.submodule == 0x00

    @property
    def is_text_mode(self) -> bool:
        return self.game_mode == 0x0E

    @property
    def is_file_select(self) -> bool:
        return self.game_mode == 0x02

    @property
    def is_title_screen(self) -> bool:
        return self.game_mode == 0x01

    @property
    def on_castle_grounds(self) -> bool:
        return (
            (not self.indoors)
            and (not self.dark_world)
            and self.screen_id == HYRULE_CASTLE_SCREEN
            and self.has_control
        )


def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])


def read_u16(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def read_snapshot(ram: np.ndarray) -> AlttpSnapshot:
    """Read a routing snapshot from a stable-retro RAM buffer."""
    return AlttpSnapshot(
        game_mode=read_u8(ram, MODULE),
        submodule=read_u8(ram, SUBMODULE),
        room_id=read_u16(ram, ROOM_ID),
        indoors=bool(read_u8(ram, INDOORS)),
        screen_id=read_u8(ram, SCREEN_ID),
        link_x=read_u16(ram, LINK_X),
        link_y=read_u16(ram, LINK_Y),
        link_direction=read_u8(ram, LINK_DIRECTION),
        link_action=read_u8(ram, LINK_ACTION),
        camera_x=read_u16(ram, CAMERA_X),
        camera_y=read_u16(ram, BG2_VOFS),
        dark_world=bool(read_u8(ram, DARK_WORLD_FLAG)),
    )


def player_has_control(env: object, _info: dict | None = None) -> bool:
    """Readiness predicate for shared StartupPlan runners."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).has_control


def on_hyrule_castle_grounds(env: object, _info: dict | None = None) -> bool:
    """True when Link is controllable on light-world castle screen 0x1B."""
    ram = np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]
    return read_snapshot(ram).on_castle_grounds
