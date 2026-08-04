"""Early route unit tests (no emulator)."""

from __future__ import annotations

from smz3.early_route import (
    EARLY_ROOM_BASELINES,
    PARLOR_ROOM_ID,
    EarlySegmentResult,
    RoomVisit,
)
from smz3.portals import EARLY_SM_ROOMS, early_portal, portals_to_dict, room_name
from smz3.ram import ComboSnapshot
from smz3.world import ActiveWorld


def test_room_names() -> None:
    assert room_name(0x91F8) == "Landing Site"
    assert room_name(PARLOR_ROOM_ID) == "Parlor and Alcatraz"
    assert "0x" in room_name(0xABCD)


def test_early_portal_is_parlor_red_door() -> None:
    p = early_portal()
    assert p.sm_door_ptr == 0x8976
    assert p.z3_cave_id == 0x0122
    # Combo table uses parlor red door $8976 (not map room 0x9994).
    assert p.sm_room_id == 0x92FD
    assert not p.dark_world


def test_portals_catalog_size() -> None:
    cats = portals_to_dict()
    assert len(cats) >= 3
    ids = {c["portal_id"] for c in cats}
    assert "crateria_map_fortune_teller" in ids


def test_baselines_cover_early_rooms() -> None:
    for rid in (0x91F8, 0x92FD, 0x98E2, 0x9994):
        assert f"0x{rid:04X}" in EARLY_ROOM_BASELINES
    assert set(EARLY_SM_ROOMS) >= {0x91F8, 0x92FD, 0x98E2, 0x9994}


def test_segment_result_dict() -> None:
    snap = ComboSnapshot(
        frame=2000,
        sm_game_state=8,
        sm_room_id=PARLOR_ROOM_ID,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=99,
        sm_max_health=99,
        sm_samus_x=1240,
        sm_samus_y=139,
        sm_pose=1,
        z3_module=151,
        z3_submodule=0,
        z3_indoors=0,
        z3_room_id=0,
        z3_screen_id=0,
        z3_link_x=0,
        z3_link_y=0,
    )
    result = EarlySegmentResult(
        ok=True,
        goal="landing_to_parlor",
        frames=2000,
        boot_frames=900,
        visits=[
            RoomVisit(0x91F8, 900, 1800, "super_metroid"),
            RoomVisit(PARLOR_ROOM_ID, 1800, 2000, "super_metroid"),
        ],
        final_snapshot=snap,
        world=ActiveWorld.SUPER_METROID,
        detail="ok",
    )
    d = result.to_dict()
    assert d["reached_parlor"] is True
    assert d["room_names"] == ["Landing Site", "Parlor and Alcatraz"]
    assert d["visits"][0]["dwell_frames"] == 900
