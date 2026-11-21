"""Unit tests for KPDR segment benchmarking and swappable route policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.splice.benchmark import (
    KPDR_REFERENCE_SPLITS,
    SUB55_BUDGET_FRAMES,
    build_benchmark_report,
    format_benchmark_report,
    format_rta,
    parse_rta_seconds,
    record_benchmark_history,
)
from super_metroid.splice.route_policy import (
    SKILL_REGISTRY,
    RoutePolicy,
    SegmentRule,
    list_policy_ids,
    load_policy,
    resolve_skill,
)


def test_reference_splits_match_target_image() -> None:
    assert len(KPDR_REFERENCE_SPLITS) == 23
    assert KPDR_REFERENCE_SPLITS[0].id == "ceres_station"
    assert KPDR_REFERENCE_SPLITS[-1].id == "ship"
    assert format_rta(KPDR_REFERENCE_SPLITS[-1].ref_run_seconds) == "56:28.88"
    assert format_rta(KPDR_REFERENCE_SPLITS[-1].ref_pb_seconds) == "57:11.75"
    assert SUB55_BUDGET_FRAMES == 198_000
    # Reference times are strictly increasing.
    run_times = [s.ref_run_seconds for s in KPDR_REFERENCE_SPLITS]
    assert run_times == sorted(run_times)


def test_rta_parse_and_format_roundtrip() -> None:
    assert parse_rta_seconds("01:31.31") == pytest.approx(91.31)
    assert parse_rta_seconds("1:58:13.95") == pytest.approx(7093.95)
    assert format_rta(91.31) == "01:31.31"
    assert format_rta(7093.95) == "1:58:13.95"
    assert format_rta(-42.87, signed=True) == "-00:42.87"
    assert format_rta(42.87, signed=True) == "+00:42.87"


def test_benchmark_report_detects_route_order_shifts_and_history(tmp_path: Path) -> None:
    stem = tmp_path / "dummy_clean"
    master = {
        "tape_sha256": "abc123",
        "total_frames": 220_000,
        "credits_complete": True,
        "rows": [
            {"row": 1, "room": "0x0000", "to_room": "0xDF45", "name": "?", "frames": 10000, "method": "tape+idle:0", "energy": 99},
            {"row": 2, "room": "0xDF45", "to_room": "0x91F8", "name": "Ceres", "frames": 6000, "method": "tape+idle:0", "energy": 99},
            {"row": 3, "room": "0xA890", "to_room": None, "name": "Ice Beam Room", "frames": 100000, "method": "room-controller", "energy": 200},
            {"row": 4, "room": "0xD2AA", "to_room": None, "name": "Plasma Room", "frames": 90000, "method": "override:tape+idle:0", "energy": 200},
            {"row": 5, "room": "0x91F8", "to_room": None, "name": "Landing Site", "frames": 14000, "method": "room-controller", "energy": 450, "goal_phase": "ending_or_credits"},
        ],
    }
    stem.with_suffix(".json").write_text(json.dumps(master))
    splits = {
        "kind": "kpdr_splits_scan",
        "tape_sha256": "abc123",
        "ceres_control_frame": 10000,
        "rta_ship_frames": 210_000,
        "credits_complete": True,
        "splits": {
            "ceres_station": {"rta_frames": 6000, "rta_seconds": 100.0, "row": 2},
            # Early Ice before Plasma should be flagged as route_order_shifts!
            "ice_beam": {"rta_frames": 106000, "rta_seconds": 1766.67, "row": 3},
            "plasma_beam": {"rta_frames": 196000, "rta_seconds": 3266.67, "row": 4},
            "mother_brain_1": {"rta_frames": 200000, "rta_seconds": 3333.33, "row": 5},
            "mother_brain_2": {"rta_frames": 204000, "rta_seconds": 3400.0, "row": 5},
            "ship": {"rta_frames": 210000, "rta_seconds": 3500.0, "row": 5},
        },
    }
    bench = build_benchmark_report(stem, splits_data=splits)
    assert "ice_beam" in bench["route_order_shifts"]
    assert bench["rta_ship"] == "58:20.00"
    assert bench["overall_policy_coverage"]["skill_pct"] > 50.0
    segments = {seg["id"]: seg for seg in bench["actual_segments"]}
    assert not segments["ice_beam"]["same_start_split"]
    assert segments["ice_beam"]["seg_frames"] == 100_000
    assert segments["mother_brain_2"]["total_frames"] == 4_000
    assert segments["mother_brain_2"]["start_row"] == segments["mother_brain_2"]["end_row"] == 5
    assert sum(seg["total_frames"] for seg in segments.values()) == 210_000

    hist_path = tmp_path / "history.json"
    hist = record_benchmark_history(bench, label="v0", notes="init", history_path=hist_path)
    assert len(hist) == 1
    assert hist[0]["split_frames"]["ship"] == 210_000
    # Re-recording same label + hash is idempotent.
    hist2 = record_benchmark_history(bench, label="v0", notes="init", history_path=hist_path)
    assert len(hist2) == 1
    md = format_benchmark_report(bench, history=hist2)
    assert "Standard KPDR Splits vs. Reference Image" in md
    assert "Longitudinal Progress History" in md
    assert "Ice Beam †" in md


def test_route_policy_inheritance_and_rollback_toggles() -> None:
    ids = list_policy_ids()
    assert "baseline_tape" in ids
    assert "speed_detours_v1" in ids
    assert "speed_v1_verified" in ids

    base = load_policy("baseline_tape")
    assert base.active_rules() == ()

    v1 = load_policy("speed_detours_v1")
    active_ids = {r.rule_id for r in v1.active_rules()}
    assert "business_to_frog_r237" in active_ids

    # Instant rollback to old tape code via with_toggles(use_tape=...).
    rolled_back = v1.with_toggles(use_tape=["business_to_frog_r237"])
    assert rolled_back.rule_by_id("business_to_frog_r237").mode == "tape"
    assert not rolled_back.rule_by_id("business_to_frog_r237").is_active()

    # Enable / disable toggles preserve rule definitions.
    enabled = v1.with_toggles(enable=["bubble_direct_r88"])
    assert enabled.rule_by_id("bubble_direct_r88").is_active()

    verified = load_policy("speed_v1_verified")
    assert [r.rule_id for r in verified.active_rules()] == ["climb_fast_r337"]
    assert verified.with_toggles(use_tape=["climb_fast_r337"]).active_rules() == ()

    v2 = load_policy("speed_v2_tourian")
    v2_ids = [r.rule_id for r in v2.active_rules()]
    assert v2_ids == [
        "tourian_direct_r313",
        "metroid_rooms_r316",
        "big_boy_room_r322",
        "rinka_shaft_direct_r328",
        "mb_tourian_v2_r331",
        "climb_fast_r337",
    ]
    assert "metroid_rooms_r316" not in [
        r.rule_id for r in v2.with_toggles(use_tape=["metroid_rooms_r316"]).active_rules()
    ]
    assert "maridia.halfie_climb_wj" in SKILL_REGISTRY


def test_overlapping_active_rules_fail_closed() -> None:
    pol = RoutePolicy(
        policy_id="bad",
        description="overlap",
        rules=(
            SegmentRule("a", 10, 15, "0xA7DE", "0xB167", mode="skill", target="norfair.business_to_frog", enabled=True),
            SegmentRule("b", 15, 20, "0xB167", "0xB106", mode="skill", target="norfair.business_to_frog", enabled=True),
        ),
    )
    with pytest.raises(ValueError, match="overlapping active rules"):
        pol.active_rules()


def test_policy_starts_from_actual_predecessor_after_collapsed_rows() -> None:
    from super_metroid.splice.route_policy import _policy_predecessor

    slices = {
        299: (0, 2900, {"replaced_rows": [299, 301]}),
        302: (2900, 472, {}),
    }
    assert _policy_predecessor(SegmentRule("hellway", 302, 302, "0xA2F7", "0xA322"), slices) == 299
    with pytest.raises(ValueError, match="row 300 is absent"):
        _policy_predecessor(SegmentRule("cut", 300, 301, "0x9FBA", "0xA2F7"), slices)
    with pytest.raises(ValueError, match="row 300 is not a source tape boundary"):
        _policy_predecessor(SegmentRule("partial", 299, 300, "0xA253", "0xA253"), slices)


def test_registered_skills_resolve() -> None:
    for skill_id in SKILL_REGISTRY:
        fn, resolved = resolve_skill(skill_id)
        assert callable(fn)
        assert resolved == skill_id


def test_policy_list_and_ab_format() -> None:
    from super_metroid.splice.route_policy import format_ab_report, format_policy_list

    text = format_policy_list()
    assert "baseline_tape" in text
    assert "speed_detours_v1" in text
    assert "norfair.business_to_frog" in text
    assert "tourian.first_room_direct" in text

    dummy_ab = {
        "rule": {
            "rule_id": "business_to_frog_r237",
            "start_row": 237,
            "end_row": 239,
            "from_room": "0xA7DE",
            "to_room": "0xB167",
            "mode": "skill",
            "target": "norfair.business_to_frog",
        },
        "verdict": "PASS",
        "error": None,
        "checkpoint": "row236_leave.state",
        "entry": {"energy": 168, "max_energy": 399, "missiles": 35, "supers": 2, "power_bombs": 5},
        "baseline": {"segment_frames": 6650, "total_frames": 7938},
        "candidate": {"segment_frames": 1172, "total_frames": 2461},
        "local_frames_saved": 5478,
        "local_seconds_saved": 91.3,
        "net_frames_saved": 5477,
        "net_seconds_saved": 91.28,
        "energy_delta_at_end": -20,
        "downstream_results": [
            {"row": 240, "name": "Frog Savestation", "ok": True, "method": "saved+idle:0", "baseline_frames": 300, "candidate_frames": 301},
        ],
    }
    ab_text = format_ab_report(dummy_ab)
    assert "PASS" in ab_text
    assert "+91.28s" in ab_text


def test_row_crossing_matches_single_and_multi_row_policy() -> None:
    from super_metroid.splice.assemble import _row_crossing_matches

    # Single-row rule still requires a direct source -> target transition.
    ok, detour = _row_crossing_matches(
        {"row": 100},
        [{"source_room_id": 0xDAE1, "target_room_id": 0xDB31}],
        0xDAE1,
        0xDB31,
    )
    assert (ok, detour) == (True, False)

    # Single-row rule rejects chained multi-hop transition without replaced_rows span.
    chained_obs = [
        {"source_room_id": 0xDAE1, "target_room_id": 0xDB31},
        {"source_room_id": 0xDB31, "target_room_id": 0xDB7D},
        {"source_room_id": 0xDB7D, "target_room_id": 0xDBCD},
        {"source_room_id": 0xDBCD, "target_room_id": 0xDC19},
    ]
    ok_single, _ = _row_crossing_matches({"row": 316}, chained_obs, 0xDAE1, 0xDC19)
    assert ok_single is False

    # Multi-row policy rule (replaced_rows [316, 319]) accepts a connected 4-hop chain with no detour.
    ok_multi, detour_multi = _row_crossing_matches(
        {"row": 316, "replaced_rows": [316, 319]},
        chained_obs,
        0xDAE1,
        0xDC19,
    )
    assert (ok_multi, detour_multi) == (True, False)

    # Multi-row policy rule still rejects broken chains.
    broken_obs = [
        {"source_room_id": 0xDAE1, "target_room_id": 0xDB31},
        {"source_room_id": 0xFFFF, "target_room_id": 0xDC19},
    ]
    ok_broken, _ = _row_crossing_matches(
        {"row": 316, "replaced_rows": [316, 319]},
        broken_obs,
        0xDAE1,
        0xDC19,
    )
    assert ok_broken is False
