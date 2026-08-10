"""Unit tests for KPDR topology and controller exports (no emulator)."""

from __future__ import annotations

from super_metroid.dev.kpdr_dev import (
    DOOR_BUSINESS_TO_HJ_SHAFT,
    DOOR_HJ_SHAFT_TO_HJ,
    HOP_BY_ID,
    ITEM_HI_JUMP,
    KPDR_TO_HIJUMP,
    ROOM_BUSINESS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_VARIA,
)


def test_hijump_doors() -> None:
    assert DOOR_BUSINESS_TO_HJ_SHAFT == 0x92D6
    assert DOOR_HJ_SHAFT_TO_HJ == 0x9426
    assert ROOM_HJ == 0xA9E5
    assert ROOM_HJ_SHAFT == 0xAA41
    assert ITEM_HI_JUMP == 0x0100


def test_hop_table_reaches_hijump() -> None:
    names = [h[0] for h in KPDR_TO_HIJUMP]
    assert names[0] == "ghz"
    assert names[-1] == "hj_room"
    assert "kraid" in names
    assert "varia" in names
    assert "business" in names
    assert HOP_BY_ID["hj_room"][2] == ROOM_HJ
    assert HOP_BY_ID["business"][2] == ROOM_BUSINESS
    assert HOP_BY_ID["varia"][2] == ROOM_VARIA
    assert HOP_BY_ID["kraid"][2] == ROOM_KRAID


def test_hop_ids_unique() -> None:
    names = [h[0] for h in KPDR_TO_HIJUMP]
    assert len(names) == len(set(names))


def test_kpdr_controller_exports() -> None:
    from super_metroid.routes.kpdr import (
        play_bat_to_below_spazer,
        play_below_spazer_to_west,
        play_big_pink_to_ghz,
        play_east_to_glass,
        play_east_to_warehouse,
        play_glass_to_east,
        play_glass_to_west,
        play_warehouse_to_east,
        play_west_to_below,
        play_ghz_to_noob,
        play_hijump_to_warehouse,
        play_noob_to_red_tower,
        play_red_tower_to_bat,
        play_red_tower_to_warehouse,
        play_run_shoot_exit,
        play_super_room_collect,
        play_warehouse_hijump_kraid,
        play_warehouse_to_hijump,
        play_warehouse_to_kraid_with_hijump,
        play_warehouse_wall_to_lower_lip,
        play_west_to_glass,
        ROOM_SUPER,
    )

    # K0 Super collect (formerly post_spore/) is part of the KPDR package.
    assert ROOM_SUPER == 0x9B5B
    assert callable(play_super_room_collect)
    assert callable(play_big_pink_to_ghz)
    assert callable(play_ghz_to_noob)
    assert callable(play_noob_to_red_tower)
    assert callable(play_red_tower_to_bat)
    assert callable(play_bat_to_below_spazer)
    assert callable(play_below_spazer_to_west)
    assert callable(play_west_to_glass)
    assert callable(play_glass_to_east)
    assert callable(play_east_to_warehouse)
    assert callable(play_east_to_glass)
    assert callable(play_glass_to_west)
    assert callable(play_warehouse_to_east)
    assert callable(play_west_to_below)
    assert callable(play_red_tower_to_warehouse)
    assert callable(play_warehouse_wall_to_lower_lip)
    assert callable(play_warehouse_to_hijump)
    assert callable(play_hijump_to_warehouse)
    assert callable(play_warehouse_to_kraid_with_hijump)
    assert callable(play_warehouse_hijump_kraid)
    assert callable(play_run_shoot_exit)

    from super_metroid.routes.kpdr import play_kraid_entry_to_varia
    from super_metroid.routes.kpdr.warehouse_stack import resolve_warehouse_entry_mode
    from super_metroid.ram import parse_state
    from dataclasses import replace
    import numpy as np

    assert callable(play_kraid_entry_to_varia)
    base = parse_state(np.zeros(0x2000, dtype=np.uint8))
    assert resolve_warehouse_entry_mode(replace(base, samus_x=50)) == "left_elevator"
    assert (
        resolve_warehouse_entry_mode(replace(base, samus_x=500))
        == "right_reverse_stack"
    )
    assert (
        resolve_warehouse_entry_mode(
            replace(base, samus_x=500), entry_mode="left_elevator"
        )
        == "left_elevator"
    )


def test_kpdr_segment_registry_includes_super_collect() -> None:
    from super_metroid.routes.kpdr import KPDR_SEGMENTS, get_segment

    assert "super_room_collect" in KPDR_SEGMENTS
    assert "big_pink_into_main_shaft" in KPDR_SEGMENTS
    assert "big_pink_to_ghz" in KPDR_SEGMENTS
    assert get_segment("super_room_collect") is KPDR_SEGMENTS["super_room_collect"]


def test_post_hijump_climb_segments_resolve_from_registry() -> None:
    from super_metroid.routes.kpdr import get_segment
    from super_metroid.routes.kpdr.business_climb import play_business_to_warehouse
    from super_metroid.routes.kpdr.return_hijump import play_hj_shaft_to_business

    assert get_segment("hj_shaft_to_business") is play_hj_shaft_to_business
    assert get_segment("business_to_warehouse") is play_business_to_warehouse


def test_kpdr_controller_has_no_progression_writes_or_state_loads() -> None:
    """Package surface must stay controller-only (no progression writes)."""
    import importlib
    import inspect
    from pathlib import Path

    # Spot-check segment modules that used to live behind the deleted shim.
    modules = (
        "super_metroid.routes.kpdr.super_collect",
        "super_metroid.routes.kpdr.red_stack",
        "super_metroid.routes.kpdr.warehouse_stack",
    )
    forbidden = (
        "write_wram",
        "write_ram",
        "set_collected_items",
        "load_state(",
        "place_samus(",
    )
    for name in modules:
        mod = importlib.import_module(name)
        source = inspect.getsource(mod)
        for token in forbidden:
            assert token not in source, f"{name} contains {token!r}"
        # Ensure the module file still exists under routes/kpdr/
        assert Path(mod.__file__).is_file()


def test_tracker_csv_exists_and_parses() -> None:
    from pathlib import Path
    import csv

    path = Path(__file__).resolve().parents[1] / "docs" / "routes" / "KPDR_TRACKER.csv"
    assert path.is_file()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) >= 30
    ids = {r["seg_id"] for r in rows}
    assert "K0.6" in ids
    assert "K2.10" in ids
    assert "K2.18" in ids
    assert next(r for r in rows if r["seg_id"] == "K1.5")["status"] == "continuous"
    assert next(r for r in rows if r["seg_id"] == "K1.6")["status"] == "continuous"
    assert next(r for r in rows if r["seg_id"] == "K2.0")["status"] == "continuous"
    assert next(r for r in rows if r["seg_id"] == "K2.1")["status"] == "continuous"
    hj = next(r for r in rows if r["seg_id"] == "K2.10")
    assert hj["item_or_boss"] == "hi_jump"
    assert hj["status"] == "continuous"
    kraid_entry = next(r for r in rows if r["seg_id"] == "K2.18")
    assert kraid_entry["room_id_hex"] == "0xA59F"
    assert kraid_entry["status"] == "continuous"
    varia = next(r for r in rows if r["seg_id"] == "K3.1")
    assert varia["status"] == "continuous"
    post = next(r for r in rows if r["seg_id"] == "K3.2")
    assert post["status"] == "continuous"
    assert post["room_id_hex"] == "0xA59F"
