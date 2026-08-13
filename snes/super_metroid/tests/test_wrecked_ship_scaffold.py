"""Unit locks for the development-only Wrecked Ship scaffold."""

from __future__ import annotations

from super_metroid.routes.kpdr import wrecked_ship
from super_metroid.routes.kpdr.guides import ROUTE_PRESETS
from super_metroid.source_states import get_source


def test_wrecked_ship_rooms_and_controllers() -> None:
    assert wrecked_ship.ROOM_KIHUNTER == 0x948C
    assert wrecked_ship.ROOM_MOAT == 0x95FF
    assert wrecked_ship.ROOM_WEST_OCEAN == 0x93FE
    assert wrecked_ship.ROOM_WS_ENTRANCE == 0xCA08
    assert wrecked_ship.ROOM_WS_MAIN == 0xCAF6
    assert wrecked_ship.ROOM_WS_BASEMENT == 0xCC6F
    assert wrecked_ship.ROOM_PHANTOON == 0xCD13
    for name in (
        "play_moat_to_west_ocean",
        "play_moat_to_ws",
        "play_west_ocean_to_ws",
        "play_ws_entrance_to_main",
        "play_ws_main_to_basement",
        "play_ws_basement_to_phantoon",
    ):
        assert callable(getattr(wrecked_ship, name))


def test_moat_to_ws_compose_and_phantoon_recording_wire() -> None:
    """Compose surface + WS pin + recording routes for Phantoon ship tape."""
    assert "west-ocean-to-ws" in ROUTE_PRESETS
    assert "ws-entrance" in ROUTE_PRESETS

    pre = get_source("post_kihunter_pre_moat_spark")
    assert pre.room_id == 0x948C
    assert pre.relative_path.endswith("post_kihunter_pre_moat_spark.state")

    moat_end = get_source("alpha_pb_to_moat_human_end")
    assert moat_end.room_id == 0x95FF

    wo = get_source("post_moat_west_ocean_spark")
    assert wo.room_id == 0x93FE

    ws_pin = get_source("post_west_ocean_ws_spark")
    assert ws_pin.room_id == 0xCA08
    assert "Phantoon" in ws_pin.use_for or "ship free-record" in ws_pin.use_for

    phant_entry = get_source("ws_ship_human_end")
    assert phant_entry.room_id == 0xCD13
