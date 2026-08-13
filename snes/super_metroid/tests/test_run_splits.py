"""Hierarchical room → item / boss fold-up (no ROM)."""

from __future__ import annotations

from super_metroid.run_splits import (
    TimingEvent,
    best_room_deltas,
    build_run_timing,
    compare_room_pbs,
    events_from_anchors,
    events_from_boss_room_splits,
    events_from_task_payload,
    events_from_trace_item_deltas,
    fold_boss_fights,
    fold_item_to_item,
    frankenstein_pb,
    room_splits_from_hops,
    room_splits_from_timer_visits,
)
from super_metroid.skill_bank import (
    SkillBank,
    compose_plan,
    make_hop_key,
    merge_runs_into_bank,
    stitch_route_plan,
)


def _hop(index: int, room: int, start: int, end: int, name: str = "") -> dict:
    return {
        "index": index,
        "room_id": room,
        "room": f"0x{room:04X}",
        "name": name or f"0x{room:04X}",
        "frame": start,
        "end_frame": end,
        "start_index": start,
        "end_index": end,
        "dwell": end - start + 1,
    }


def test_room_splits_from_hops_and_item_fold() -> None:
    hops = [
        _hop(0, 0x91F8, 0, 100, "Landing Site"),
        _hop(1, 0x92FD, 101, 200, "Parlor and Alcatraz"),
        _hop(2, 0x9E9F, 201, 350, "Morph Ball Room"),
    ]
    rooms = room_splits_from_hops(hops)
    assert len(rooms) == 3
    assert rooms[0].name == "Landing Site"
    assert rooms[2].dwell_frames == 150

    events = [
        TimingEvent(frame=0, kind="item_delta", label="start"),
        TimingEvent(frame=350, kind="item_delta", room_id=0x9E9F, label="Morph"),
        TimingEvent(frame=500, kind="item_delta", room_id=0x9804, label="Bombs"),
    ]
    items = fold_item_to_item(events, rooms)
    assert len(items) == 2
    assert items[0].id == "start_to_Morph"
    assert items[1].label == "Morph → Bombs"
    assert items[1].frames == 150


def test_boss_fold_pairs_start_finish() -> None:
    rooms = room_splits_from_hops(
        [
            _hop(0, 0xA56B, 0, 50, "Kraid Eye Door Room"),
            _hop(1, 0xA59F, 51, 400, "Kraid Room"),
            _hop(2, 0xA6E2, 401, 450, "Varia Suit Room"),
        ]
    )
    events = [
        TimingEvent(frame=51, kind="boss_start", room_id=0xA59F, label="Kraid"),
        TimingEvent(frame=400, kind="boss_finish", room_id=0xA59F, label="Kraid"),
    ]
    bosses = fold_boss_fights(events, rooms)
    assert len(bosses) == 1
    assert bosses[0].label == "Kraid"
    assert bosses[0].frames == 349


def test_build_run_timing_report() -> None:
    visits = [
        {
            "room_id": 0x91F8,
            "entry_frame": 0,
            "leave_frame": 80,
            "exit_frame": 100,
            "dest_room_id": 0x92FD,
            "dwell_frames": 80,
            "room_frames": 100,
            "transition_frames": 20,
        }
    ]
    rooms = room_splits_from_timer_visits(visits)
    report = build_run_timing(
        rooms,
        [TimingEvent(frame=0, kind="segment", label="boot")],
        source="unit",
        total_frames=100,
    )
    d = report.to_dict()
    assert d["summary"]["room_visits"] == 1
    assert d["rooms"][0]["name"] == "Landing Site"


def test_trace_item_deltas() -> None:
    trace = [
        {"frame": 0, "room": 0x9E9F, "items": 0},
        {"frame": 10, "room": 0x9E9F, "items": 0},
        {"frame": 20, "room": 0x9E9F, "items": 0x4},
    ]
    ev = events_from_trace_item_deltas(trace)
    assert len(ev) == 1
    assert ev[0].frame == 20
    assert ev[0].detail["to"] == 4


def test_skill_bank_merge_and_pb() -> None:
    run_a = room_splits_from_hops(
        [_hop(0, 0x9E9F, 0, 100, "Morph Ball Room")]
    )
    run_b = room_splits_from_hops(
        [_hop(0, 0x9E9F, 0, 80, "Morph Ball Room")]
    )
    bank = SkillBank()
    merge_runs_into_bank(bank, [("a", run_a), ("b", run_b)])
    key = make_hop_key(0x9E9F, to_room_id=None)
    # keys include dest leave
    keys = list(bank.records)
    assert keys
    best = bank.best(keys[0])
    assert best is not None
    assert best.frames == 81  # dwell 80..100 inclusive style: end-start+1 = 81 for 0..80


def test_frankenstein_and_stitch() -> None:
    theory = frankenstein_pb({"h1": 100, "h2": 50}, ["h1", "h2", "h3"])
    assert theory["frames"] == 150
    assert theory["missing"] == ["h3"]
    assert not theory["complete"]

    bank = SkillBank()
    from super_metroid.skill_bank import HopSkillRecord

    bank.add(
        HopSkillRecord(
            hop_key="h1",
            room_id=1,
            name="A",
            frames=100,
            source="x",
            dual_green=True,
            entry_anchor="a.state",
            body_path="a.json",
        )
    )
    plan = compose_plan(bank, ["h1", "h2"], require_dual_green=True)
    assert plan["missing"] == ["h2"]
    assert plan["steps"][0]["status"] == "ready"
    assert stitch_route_plan is compose_plan  # back-compat alias


def test_compare_room_pbs() -> None:
    a = room_splits_from_hops([_hop(0, 0x91F8, 0, 100, "Landing Site")])
    b = room_splits_from_hops([_hop(0, 0x91F8, 0, 90, "Landing Site")])
    deltas = compare_room_pbs(a, b)
    assert deltas[0]["delta"] < 0
    slow = best_room_deltas(compare_room_pbs(b, a))
    assert slow[0]["delta"] > 0


def test_events_from_anchors_room_enter_and_item_delta() -> None:
    anchors = {
        "anchors": [
            {
                "kind": "boot",
                "frame": 0,
                "room": "0x91F8",
                "room_id": 0x91F8,
                "items": "0x0000",
            },
            {
                "kind": "room_enter",
                "frame": 100,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "items": "0x0000",
            },
            {
                "kind": "item_delta",
                "frame": 150,
                "room": "0x9E9F",
                "room_id": 0x9E9F,
                "items": "0x3105",
            },
            {
                "kind": "manual",
                "frame": 200,
                "room_id": 0x9E9F,
                "label": "checkpoint",
            },
            {
                "kind": "end",
                "frame": 300,
                "room": "0x9804",
                "room_id": 0x9804,
            },
        ]
    }
    ev = events_from_anchors(anchors)
    kinds = [e.kind for e in ev]
    assert kinds == ["room_enter", "room_enter", "item_delta", "segment", "segment"]
    assert ev[0].label == "boot"
    assert ev[1].kind == "room_enter" and ev[1].room_id == 0x9E9F
    assert ev[2].kind == "item_delta" and ev[2].label == "0x3105"
    assert ev[3].label == "checkpoint"
    assert ev[4].label == "end"
    # Bare list of rows also accepted
    bare = events_from_anchors(anchors["anchors"])
    assert len(bare) == len(ev)


def test_events_from_boss_room_splits_pairs() -> None:
    rooms = room_splits_from_hops(
        [
            _hop(0, 0xA56B, 0, 50, "Kraid Eye Door Room"),
            _hop(1, 0xA59F, 51, 400, "Kraid Room"),
            _hop(2, 0xA6E2, 401, 450, "Varia Suit Room"),
        ]
    )
    ev = events_from_boss_room_splits(rooms)
    assert len(ev) == 2
    assert ev[0].kind == "boss_start" and ev[0].frame == 51
    assert ev[0].label == "Kraid"
    assert ev[1].kind == "boss_finish" and ev[1].frame == 400
    bosses = fold_boss_fights(ev, rooms)
    assert len(bosses) == 1
    assert bosses[0].frames == 349


def test_events_from_task_payload_merges() -> None:
    trace = [
        {"frame": 0, "room": 0x9E9F, "items": 0},
        {"frame": 20, "room": 0x9E9F, "items": 0x4},
    ]
    anchors = [
        {
            "kind": "room_enter",
            "frame": 0,
            "room_id": 0x9E9F,
            "room": "0x9E9F",
        },
        # Same frame as trace item_delta — prefer trace label
        {
            "kind": "item_delta",
            "frame": 20,
            "room_id": 0x9E9F,
            "items": "0x0004",
        },
        {
            "kind": "item_delta",
            "frame": 40,
            "room_id": 0x9804,
            "items": "0x1005",
        },
    ]
    rooms = room_splits_from_hops([_hop(0, 0xA59F, 50, 100, "Kraid Room")])
    ev = events_from_task_payload(trace=trace, anchors=anchors, rooms=rooms)
    item_events = [e for e in ev if e.kind == "item_delta"]
    assert len(item_events) == 2
    # Trace wins at frame 20
    at_20 = next(e for e in item_events if e.frame == 20)
    assert at_20.label == "items_0004"
    assert at_20.detail.get("to") == 4
    # Anchor-only item kept
    at_40 = next(e for e in item_events if e.frame == 40)
    assert at_40.label == "0x1005"
    assert any(e.kind == "room_enter" for e in ev)
    assert any(e.kind == "boss_start" for e in ev)
    assert any(e.kind == "boss_finish" for e in ev)
    # Dedup: no duplicate (kind, frame, room_id)
    keys = [(e.kind, e.frame, e.room_id) for e in ev]
    assert len(keys) == len(set(keys))


def test_room_splits_from_hops_timeline_index() -> None:
    """Index timeline uses start_index/end_index even when frame fields differ."""
    hop = {
        "index": 0,
        "room_id": 0x91F8,
        "room": "0x91F8",
        "name": "Landing Site",
        "frame": 9000,  # renumbered / absolute — ignore for index mode
        "end_frame": 9500,
        "start_index": 10,
        "end_index": 110,
        "dwell": 101,
        "transition_frames": 5,
    }
    rooms = room_splits_from_hops([hop], timeline="index")
    assert rooms[0].entry_frame == 10
    assert rooms[0].leave_frame == 110
    assert rooms[0].dwell_frames == 101
    assert rooms[0].transition_frames == 5

    rooms_frame = room_splits_from_hops([hop], timeline="frame")
    assert rooms_frame[0].entry_frame == 9000
    assert rooms_frame[0].leave_frame == 9500
    assert rooms_frame[0].dwell_frames == 101
