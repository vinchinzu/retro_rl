"""Unit tests for TAS room stages + hop extraction (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
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
    assert hops[0].usable is True
    assert hops[1].desync_in_hop is True
    assert hops[1].usable is False  # desync_in_hop
    # After first desync frame (450), later hop is thrash — not research.
    assert hops[2].enter_frame > 450
    assert hops[2].usable is False
    assert hops[2].notes == "post_desync_thrash"

    board = build_extraction_board(hops, pins=_sample_events(), run_id="test")
    assert board["schema"] == "sm_tas_extraction_board_v1"
    assert board["summary"]["hop_count"] == 3
    assert board["summary"]["usable_hops"] == 1
    assert "Zebes-first" in " ".join(board["rules"])
    # Product P0 always listed: Snake→Ice PLM (rr-5if); Acid→Snake dual GREEN.
    tops = board["top_skill_room_candidates"]
    assert any(
        c["from_room"] == ROOM_ICE_SNAKE and c["to_room"] == ROOM_ICE for c in tops
    )
    assert any(c.get("bead_hint") == "rr-5if" for c in tops)
    # No Acid→Snake pure_open injection (rr-5cf dual GREEN).
    assert not any(
        c["from_room"] == ROOM_ICE_ACID
        and c["to_room"] == ROOM_ICE_SNAKE
        and c["pure_status"] == "pure_open"
        for c in tops
    )
    # Post-desync thrash never appears as a skill candidate.
    thrash_ids = {h.hop_id for h in hops if not h.usable}
    assert not any(c["hop_id"] in thrash_ids for c in tops)
    thrash_cand = next(c for c in board["candidates"] if c["hop_id"] == hops[2].hop_id)
    assert thrash_cand["pure_status"] == "post_desync_thrash"


def test_post_desync_thrash_not_top_skill() -> None:
    """Ceres bounce after desync must not become continuous_green skill port."""
    events = [
        {
            "kind": "room_enter",
            "frame": 100,
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 0,
            "y": 0,
        },
        {
            "kind": "desync_suspect",
            "frame": 200,
            "room_id": ROOM_CERES_ELEVATOR,
            "detail": "stall",
        },
        {
            "kind": "room_enter",
            "frame": 500,
            "room_id": ROOM_CERES_FALLING,
            "pose": 0,
            "x": 0,
            "y": 0,
        },
        {
            "kind": "room_enter",
            "frame": 800,
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 0,
            "y": 0,
        },
        {
            "kind": "room_enter",
            "frame": 1200,
            "room_id": ROOM_CERES_FALLING,
            "pose": 0,
            "x": 0,
            "y": 0,
        },
    ]
    hops = build_hops(events, run_id="thrash")
    assert hops[0].usable is False  # contains desync
    assert all(h.usable is False for h in hops[1:])
    assert all(h.notes == "post_desync_thrash" for h in hops[1:])
    board = build_extraction_board(hops, pins=events, run_id="thrash")
    assert board["summary"]["usable_hops"] == 0
    tops = board["top_skill_room_candidates"]
    # Only synthetic Ice product row (or empty of Ceres thrash).
    assert not any(
        c["from_room"] in (ROOM_CERES_ELEVATOR, ROOM_CERES_FALLING)
        for c in tops
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


def test_extract_run_lsnes_oracle_without_trace(tmp_path: Path) -> None:
    """events.jsonl + proof.json (no trace/summary) stays source=lsnes_oracle."""
    oracle = tmp_path / "sniq_100_lsnes"
    oracle.mkdir()
    # Ceres elev → falling → … → Ridley → reverse escape → landing.
    path = [
        (8319, ROOM_CERES_ELEVATOR),
        (8832, ROOM_CERES_FALLING),
        (9114, ROOM_CERES_MAGNET),
        (9458, ROOM_CERES_SCIENTIST),
        (9723, ROOM_CERES_FLAT),
        (9979, ROOM_CERES_RIDLEY),
        (11821, ROOM_CERES_FLAT),
        (12079, ROOM_CERES_SCIENTIST),
        (12342, ROOM_CERES_MAGNET),
        (12671, ROOM_CERES_FALLING),
        (12952, ROOM_CERES_ELEVATOR),
        (15198, ROOM_LANDING_SITE),
    ]
    events: list[dict] = [
        {"frame": 0, "kind": "start"},
        {
            "frame": 8538,
            "kind": "control",
            "room_id": ROOM_CERES_ELEVATOR,
            "pose": 0,
            "x": 128,
            "y": 0,
        },
    ]
    for frame, room_id in path:
        events.append(
            {
                "frame": frame,
                "kind": "room_enter",
                "room_id": room_id,
                "pose": 0,
                "x": 0,
                "y": 0,
                "energy": 99,
            }
        )
    events.append({"frame": 15198, "kind": "green", "landing_frame": 15198})
    (oracle / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    (oracle / "proof.json").write_text(
        json.dumps(
            {
                "status": "GREEN",
                "source": "lsnes_oracle",
                "landing_frame": 15198,
                "first_control_frame": 8538,
                "unique_rooms": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    board = extract_run(oracle)
    assert board["source"] == "lsnes_oracle"
    hops = board["hops"]
    assert len(hops) == 12
    assert hops[0]["from_room"] == ROOM_CERES_ELEVATOR
    assert hops[0]["to_room"] == ROOM_CERES_FALLING
    assert hops[5]["from_room"] == ROOM_CERES_RIDLEY
    assert hops[5]["to_room"] == ROOM_CERES_FLAT
    assert hops[-2]["from_room"] == ROOM_CERES_ELEVATOR
    assert hops[-2]["to_room"] == ROOM_LANDING_SITE
    assert hops[-1]["from_room"] == ROOM_LANDING_SITE
    assert hops[-1]["to_room"] is None
    assert all(h["usable"] is True for h in hops)
    assert board["summary"]["usable_hops"] == 12
    assert board["summary"]["desync_hops"] == 0
    assert board["summary"]["post_desync_thrash_hops"] == 0
    assert not any(h["desync_in_hop"] or h["death_in_hop"] for h in hops)
    assert not any(
        h["notes"] in ("desync_in_hop", "post_desync_thrash", "death") for h in hops
    )
    ann = board["annotate_summary"]
    assert ann["first_control_frame"] == 8538
    assert ann["landing_frame"] == 15198
    assert ann["unique_rooms"] == 7
    assert ann["status"] == "GREEN"


def test_committed_ceres_lsnes_hops_table() -> None:
    """Oracle hop table is plan_only reference, not a product tape."""
    path = (
        Path(__file__).resolve().parents[1]
        / "tas"
        / "bodies"
        / "sniq_100_ceres_lsnes_hops.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "sm_tas_ceres_hops_v2"
    assert payload["source"] == "lsnes_oracle"
    assert payload["status"] == "plan_only"
    assert payload["first_control_frame"] == 8538
    assert payload["first_pad_frame"] == 8639
    assert payload["landing_frame"] == 15198
    assert payload["station_frames"] == 15198 - 8538
    hops = payload["hops"]
    assert hops[0]["from"] == "0xDF45"
    assert hops[-1]["to"] == "0x91F8"
    assert hops[-1]["frames"] == 2246
    assert payload["elev_wj"]["policy"].startswith("PreciseWallJumpTiming")


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
