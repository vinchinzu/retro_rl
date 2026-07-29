"""Unit tests for shared development helpers (no emulator required)."""

from __future__ import annotations

from super_metroid.dev.common import enemy_hps
from super_metroid.dev.phantoon_dev import SHIP_ROUTE, ROOM_PHANTOON, ROOM_PINK_PB
from super_metroid.dev.kraid_dev import DOOR_EYE_TO_KRAID, ROOM_KRAID


def test_ship_route_ends_at_phantoon() -> None:
    assert SHIP_ROUTE[-1][0] == "phantoon"
    assert SHIP_ROUTE[-1][2] == ROOM_PHANTOON
    assert len(SHIP_ROUTE) == 10


def test_room_constants_match_known_ids() -> None:
    assert ROOM_PINK_PB == 0x9E11
    assert ROOM_PHANTOON == 0xCD13
    assert ROOM_KRAID == 0xA59F
    assert DOOR_EYE_TO_KRAID == 0x91B6


def test_enemy_hps_reads_slots() -> None:
    class _Env:
        def get_ram(self):
            ram = bytearray(0x2000)
            # enemy 0 HP = 1000
            ram[0x0F8C] = 0xE8
            ram[0x0F8D] = 0x03
            # enemy 1 HP = 42
            ram[0x0F8C + 0x40] = 42
            ram[0x0F8D + 0x40] = 0
            return ram

    assert enemy_hps(_Env(), 2) == [1000, 42]
