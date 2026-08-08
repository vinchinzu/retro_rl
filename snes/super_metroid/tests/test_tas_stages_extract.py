"""Unit tests for TAS room stages + hop extraction (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_ICE,
    ROOM_ICE_ACID,
    ROOM_ICE_SNAKE,
    ROOM_LANDING_SITE,
    ROOM_PARLOR,
)
from super_metroid.tas.extract_hops import (
    build_extraction_board,
    build_hops,
    extract_run,
    room_hex,
    write_board,
)
from super_metroid.tas.stages import (
    STAGE_CATALOG,
    GoalKind,
    control_in,
    export_room_body_spec,
    get_stage,
    is_room_settled,
    movie_window_from_pins,
)


def test_stage_catalog_has_ceres_and_ice_p0() -> None:
    assert "ceres_first_control" in STAGE_CATALOG
    assert "ice_acid_to_snake" in STAGE_CATALOG
    ice = get_stage("ice_acid_to_snake")
    assert ice.room_id == ROOM_ICE_ACID
    assert ice.goal_room_id == ROOM_ICE_SNAKE
    assert ice.track == "product"
    # Acid→Snake dual GREEN (rr-5cf); product_p0 tag retained as Ice stack track.


def test_control_and_goal_on_pin_dict() -> None:
    pin = {
        "room_id": ROOM_LANDING_SITE,
        "game_state": 8,
        "door_transition": 0,
        "phase": "ORDINARY_GAMEPLAY",
    }
    assert is_room_settled(pin, ROOM_LANDING_SITE)
    assert control_in(ROOM_LANDING_SITE)(pin)
    assert not control_in(ROOM_PARLOR)(pin)

    stage = get_stage("landing_to_parlor")
    assert stage.control(pin)
    assert not stage.goal(pin)
    goal_pin = {**pin, "room_id": ROOM_PARLOR}
    assert stage.goal(goal_pin)


def test_movie_window_from_pins() -> None:
    pins = [
        {"kind": "room_enter", "frame": 100, "room_id": ROOM_CERES_ELEVATOR},
        {"kind": "room_enter", "frame": 500, "room_id": ROOM_CERES_FALLING},
        {"kind": "room_enter", "frame": 900, "room_id": ROOM_CERES_ELEVATOR},
    ]
    win = movie_window_from_pins(
        pins, from_room=ROOM_CERES_ELEVATOR, to_room=ROOM_CERES_FALLING
    )
    assert win == (100, 500)
    assert (
        movie_window_from_pins(
            pins, from_room=ROOM_LANDING_SITE, to_room=ROOM_PARLOR
        )
        is None
    )


def test_export_room_body_spec() -> None:
    stage = get_stage("ceres_elev_to_falling")
    pins = [
        {
            "kind": "room_enter",
            "frame": 11182,
            "room_id": ROOM_CERES_ELEVATOR,
        },
        {
            "kind": "room_enter",
            "frame": 17821,
            "room_id": ROOM_CERES_FALLING,
        },
    ]
    spec = export_room_body_spec(stage, pins)
    assert spec["schema"] == "sm_tas_room_body_v1"
    assert spec["movie_start"] == 11182
    assert spec["body_frames"] == 17821 - 11182
    assert spec["status"] == "plan_only"
    assert "never_sanitize_L+R" in spec["hard_rules"]


def _sample_events() -> list[dict]:
    return [
        {
            "kind": "control",
            "frame": 100,
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 128,
            "y": 0,
            "detail": "first_control",
        },
        {
            "kind": "room_enter",
            "frame": 100,
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 128,
            "y": 0,
            "detail": "enter",
        },
        {
            "kind": "pose_cluster",
            "frame": 150,
            "room_id": ROOM_CERES_ELEVATOR,
            "detail": "walljump",
        },
        {
            "kind": "room_enter",
            "frame": 400,
            "room_id": ROOM_CERES_FALLING,
            "pose": 9,
            "x": 39,
            "y": 139,
            "detail": "hop",
        },
        {
            "kind": "desync_suspect",
            "frame": 450,
            "room_id": ROOM_CERES_FALLING,
            "detail": "stall",
        },
        {
            "kind": "room_enter",
            "frame": 2000,
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 0,
            "y": 0,
            "detail": "back",
        },
    ]


def test_build_hops_and_board() -> None:
    hops = build_hops(_sample_events(), run_id="test")
    assert len(hops) == 3
    assert hops[0].from_room == ROOM_CERES_ELEVATOR
    assert hops[0].to_room == ROOM_CERES_FALLING
    assert hops[0].frames == 300
    assert "walljump" in hops[0].tech_tags
    assert hops[1].desync_in_hop is True

    board = build_extraction_board(hops, pins=_sample_events(), run_id="test")
    assert board["schema"] == "sm_tas_extraction_board_v1"
    assert board["summary"]["hop_count"] == 3
    # Product P0 always listed: Snake→Ice PLM (rr-5if); Acid→Snake dual GREEN.
    tops = board["top_skill_room_candidates"]
    assert any(
        c["from_room"] == ROOM_ICE_SNAKE and c["to_room"] == ROOM_ICE for c in tops
    )
    assert any(c.get("bead_hint") == "rr-5if" for c in tops)


def test_extract_run_any_full_if_present() -> None:
    run = (
        Path(__file__).resolve().parents[1]
        / "recordings"
        / "tas_import"
        / "sniq_any_full"
    )
    if not (run / "trace.json").is_file() and not (run / "summary.json").is_file():
        pytest.skip("sniq_any_full annotate artifacts not present")
    board = extract_run(run)
    assert board["summary"]["hop_count"] >= 1
    assert board["annotate_summary"].get("first_control_frame") in (11182, None) or (
        board["annotate_summary"].get("first_control_frame") == 11182
    )


def test_write_board_roundtrip(tmp_path: Path) -> None:
    hops = build_hops(_sample_events(), run_id="t")
    board = build_extraction_board(hops, run_id="t")
    path = write_board(board, tmp_path / "board.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["hop_count"] == 3
    assert room_hex(ROOM_CERES_ELEVATOR) == "0xDF45"


def test_goal_item_bit() -> None:
    from super_metroid.tas.stages import RoomStageSpec

    st = RoomStageSpec(
        id="morph_item",
        room_id=0x9E9F,
        goal_kind=GoalKind.ITEM_BIT,
        goal_mask=0x0004,
    )
    assert not st.goal({"collected_items": 0})
    assert st.goal({"collected_items": 0x0004})
    assert st.goal({"items": "0x0004"})
