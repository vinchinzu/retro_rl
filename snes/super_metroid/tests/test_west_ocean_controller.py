"""Unit locks for West Ocean shinespark controllers (no emulator)."""

from __future__ import annotations

from super_metroid.routes.kpdr import west_ocean as wo
from super_metroid.routes.kpdr import wrecked_ship as ws


def test_west_ocean_rooms_and_exports() -> None:
    assert wo.ROOM_WEST_OCEAN == 0x93FE
    assert wo.ROOM_BOWLING == 0xC98E
    assert wo.ROOM_WS_ENTRANCE == 0xCA08
    assert wo.SPIT_PLACE_XY == (350, 550)
    assert wo.OCEAN_FLOOR_PLACE_XY == (48, 1140)
    assert wo.DEFAULT_OCEAN_HOP_FRAMES == 4
    assert wo.DEFAULT_OCEAN_PRE_STAND == 4
    for name in (
        "play_west_ocean_edge_spark",
        "play_west_ocean_over_ocean_spark",
        "play_west_ocean_to_ws",
        "open_green_super_ws",
        "run_to_water_edge",
    ):
        assert callable(getattr(wo, name))


def test_wrecked_ship_reexports_product_ws() -> None:
    assert ws.play_west_ocean_to_ws is wo.play_west_ocean_to_ws
    assert ws.play_west_ocean_over_ocean_spark is wo.play_west_ocean_over_ocean_spark
    assert callable(ws.play_west_ocean_to_ws)
