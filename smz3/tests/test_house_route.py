"""Unit tests for Link's House map-driven route (no emulator)."""

from __future__ import annotations

from smz3.house_route import (
    APPROACH_Y,
    CHEST_OPEN_X,
    CHEST_OPEN_Y,
    ENTRANCE_X,
    ENTRANCE_Y,
    HOUSE_APPROACH_WAYPOINTS,
    HouseSegmentResult,
    LINKS_HOUSE_INTERIOR_ROOM,
    indoors_links_house,
)
from smz3.ram import ComboSnapshot


def _snap(**overrides: object) -> ComboSnapshot:
    base = dict(
        frame=0,
        sm_game_state=0,
        sm_room_id=0,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=0,
        sm_max_health=0,
        sm_samus_x=0,
        sm_samus_y=0,
        sm_pose=0,
        z3_module=0x07,
        z3_submodule=0,
        z3_indoors=1,
        z3_room_id=LINKS_HOUSE_INTERIOR_ROOM,
        z3_screen_id=0x2C,
        z3_link_x=2424,
        z3_link_y=8664,
    )
    base.update(overrides)
    return ComboSnapshot(**base)  # type: ignore[arg-type]


def test_map_constants_match_yaze_and_vanilla() -> None:
    # Yaze Link's House Post-intro entrance_id 0x01.
    assert ENTRANCE_X == 2224
    assert ENTRANCE_Y == 2800
    # Under-house approach band (west ramp path).
    assert APPROACH_Y == 2846
    # Vanilla lamp-script end standing XY.
    assert CHEST_OPEN_X == 2491
    assert CHEST_OPEN_Y == 8632
    assert len(HOUSE_APPROACH_WAYPOINTS) == 3
    assert HOUSE_APPROACH_WAYPOINTS[0].label == "south_clear"



def test_indoors_links_house() -> None:
    assert indoors_links_house(_snap())
    assert not indoors_links_house(_snap(z3_indoors=0))
    assert not indoors_links_house(_snap(z3_room_id=0x55))


def test_result_dict_includes_map_sources() -> None:
    result = HouseSegmentResult(
        ok=True,
        frames=900,
        detail="ok",
        entered=True,
        chest_opened=True,
        max_hp_before=24,
        max_hp_after=32,
    )
    d = result.to_dict()
    assert d["ok"] is True
    assert d["chest_opened"] is True
    assert d["map"]["entrance_xy"] == [2224, 2800]
    assert d["map"]["chest_open_xy"] == [2491, 8632]
    assert any("yaze" in s.lower() for s in d["map"]["sources"])
