"""Unit locks for the development-only Wrecked Ship scaffold."""

from __future__ import annotations

from collections.abc import Callable

from super_metroid.routes.kpdr import wrecked_ship


def test_wrecked_ship_room_constants_match_k6_route() -> None:
    assert wrecked_ship.ROOM_MOAT == 0x95FF
    assert wrecked_ship.ROOM_WEST_OCEAN == 0x93FE
    assert wrecked_ship.ROOM_WS_ENTRANCE == 0xCA08
    assert wrecked_ship.ROOM_WS_MAIN == 0xCAF6
    assert wrecked_ship.ROOM_WS_BASEMENT == 0xCC6F
    assert wrecked_ship.ROOM_PHANTOON == 0xCD13


def test_wrecked_ship_scaffold_exports_callables() -> None:
    segments: tuple[Callable[..., object], ...] = (
        wrecked_ship.play_moat_to_west_ocean,
        wrecked_ship.play_west_ocean_to_ws,
        wrecked_ship.play_ws_entrance_to_main,
        wrecked_ship.play_ws_main_to_basement,
        wrecked_ship.play_ws_basement_to_phantoon,
    )

    assert all(callable(segment) for segment in segments)
