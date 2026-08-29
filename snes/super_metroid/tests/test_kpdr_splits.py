"""KPDR Any% tracker-format split table (no ROM)."""

from __future__ import annotations

from super_metroid.human_tape.kpdr_splits import (
    build_kpdr_split_rows,
    format_kpdr_split_table,
    fmt_tracker,
    fmt_tracker_delta,
)


def test_fmt_tracker_matches_auto_tracker_clock() -> None:
    assert fmt_tracker(0) == "00:00.00"
    assert fmt_tracker(60) == "00:01.00"
    assert fmt_tracker(90) == "00:01.50"
    assert fmt_tracker(392208) == "108:56.80"
    assert fmt_tracker_delta(-209) == "-03.48"
    assert fmt_tracker_delta(0) == "+00.00"


def test_ceres_uses_elevator_leave_not_landing_site() -> None:
    """ASL ceresEscape = leave ordinary in Ceres Elevator (gs 8→32)."""
    timeline = [
        {
            "room_id": 0xDF45,
            "abs_entry": 0,
            "dwell": 480,
            "dest_room_id": 0xDF8D,
            "hop_key": "elev_out",
            "source": "s1",
        },
        {
            "room_id": 0xDF45,
            "abs_entry": 6416,
            "dwell": 2640,
            "dest_room_id": 0x91F8,
            "hop_key": "elev_esc",
            "source": "s1",
        },
        {
            "room_id": 0x91F8,
            "abs_entry": 9286,
            "dwell": 961,
            "hop_key": "ls",
            "source": "s1",
        },
    ]
    rows = build_kpdr_split_rows(timeline, hop_pb={"elev_esc": 300})
    ceres = {r.split_id: r for r in rows}["ceres_station"]
    assert ceres.hit
    assert ceres.best_frames == 6416 + 2640  # leave, not LS 9286
    # gold = leave - (dwell - pb) = 9056 - (2640 - 300) = 6716
    assert ceres.gold_frames == 6716


def test_mb_escape_tail_from_mother_brain_entry() -> None:
    from super_metroid.human_tape.kpdr_splits import MB_ESCAPE_TAIL

    timeline = [
        {"room_id": 0xA66A, "abs_entry": 351678, "dwell": 3160, "hop_key": "g4", "source": "live"},
        {"room_id": 0xDD58, "abs_entry": 371330, "dwell": 20878, "hop_key": "mb", "source": "live"},
    ]
    rows = build_kpdr_split_rows(timeline)
    by_id = {r.split_id: r for r in rows}
    assert by_id["golden_four"].best_frames == 351678
    assert by_id["mother_brain_1"].hit
    assert by_id["mother_brain_1"].best_frames == 371330 + MB_ESCAPE_TAIL["mother_brain_1"][0]
    assert by_id["mother_brain_2"].best_frames == 371330 + MB_ESCAPE_TAIL["mother_brain_2"][0]
    assert by_id["ship"].best_frames == 371330 + MB_ESCAPE_TAIL["ship"][0]
    assert by_id["ship"].source == "g4_tourian_human_mb"
    table = format_kpdr_split_table(rows, product_frames=392208)
    assert "spliced MB finish" in table
    assert fmt_tracker(by_id["ship"].best_frames) in table
