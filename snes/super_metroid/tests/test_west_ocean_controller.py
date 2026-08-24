"""Unit locks for West Ocean shinespark controllers (no emulator)."""

from __future__ import annotations

from super_metroid.routes.kpdr import west_ocean as wo
from super_metroid.routes.kpdr import wrecked_ship as ws
from super_metroid.routes.kpdr.guides import ROUTE_PRESETS
from super_metroid.source_states import get_source


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


def test_ws_recording_routes_and_source_catalog() -> None:
    """Product WS pin + human record routes are wired after pure-ws / chain-ws."""
    assert "west-ocean-to-ws" in ROUTE_PRESETS
    assert "ws-entrance" in ROUTE_PRESETS
    ws_guides = ROUTE_PRESETS["ws-entrance"]
    assert len(ws_guides) == 1
    assert ws_guides[0].room_id == 0xCA08

    pre = get_source("post_kihunter_pre_moat_spark")
    assert pre.room_id == 0x948C

    moat_pin = get_source("post_moat_west_ocean_spark")
    assert moat_pin.room_id == 0x93FE
    assert moat_pin.relative_path.endswith("post_moat_west_ocean_spark.state")

    ws_pin = get_source("post_west_ocean_ws_spark")
    assert ws_pin.room_id == 0xCA08
    assert ws_pin.relative_path.endswith("post_west_ocean_ws_spark.state")

    poweron = get_source("post_moat_poweron")
    assert poweron.room_id == 0x93FE
    assert poweron.relative_path.endswith("post_moat_poweron.state")
    ws_cont = get_source("post_moat_poweron_wo_to_ws")
    assert ws_cont.room_id == 0xCA08
    ws_poweron = get_source("post_ws_poweron")
    assert ws_poweron.room_id == 0xCA08
    assert ws_poweron.relative_path.endswith("post_ws_poweron.state")

    assert callable(ws.play_moat_to_ws)
    assert callable(ws.play_moat_to_west_ocean)

    grav = get_source("post_gravity_caterpillar")
    assert grav.room_id == 0xA322
    assert grav.relative_path.endswith("post_gravity_caterpillar.state")
