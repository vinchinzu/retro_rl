"""Tests for retro_harness.splits module."""
import json
import tempfile
from pathlib import Path

import pytest

from retro_harness.splits import SplitResult, SplitTracker


class TestSplitResult:
    def test_fields(self):
        assert SplitResult._fields == (
            "segment_id", "elapsed_seconds", "is_pb", "pb_seconds", "diff_seconds"
        )

    def test_construction(self):
        r = SplitResult("s1", 10.0, True, 10.0, None)
        assert r.segment_id == "s1"
        assert r.is_pb is True


class TestSplitTrackerBasic:
    def test_first_segment_returns_none(self):
        t = SplitTracker(session_id="test")
        result = t.on_segment_change("s1", elapsed_seconds=5.0)
        assert result is None

    def test_current_segment(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("s1")
        assert t.current_segment == "s1"

    def test_second_segment_returns_result(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("s1")
        r = t.on_segment_change("s2", elapsed_seconds=10.0)
        assert r is not None
        assert r.segment_id == "s1"
        assert r.elapsed_seconds == 10.0

    def test_splits_list(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("a")
        t.on_segment_change("b", elapsed_seconds=5.0)
        t.on_segment_change("c", elapsed_seconds=3.0)
        assert len(t.splits) == 2

    def test_none_elapsed(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("a")
        r = t.on_segment_change("b", elapsed_seconds=None)
        assert r.elapsed_seconds is None
        assert r.is_pb is False


class TestSplitTrackerPB:
    def test_first_time_is_pb(self):
        with tempfile.TemporaryDirectory() as td:
            pb = Path(td) / "pb.json"
            t = SplitTracker(best_times_path=pb, session_id="t1")
            t.on_segment_change("s1")
            r = t.on_segment_change("s2", elapsed_seconds=20.0)
            assert r.is_pb is True
            assert r.pb_seconds == 20.0

    def test_slower_is_not_pb(self):
        with tempfile.TemporaryDirectory() as td:
            pb = Path(td) / "pb.json"
            t1 = SplitTracker(best_times_path=pb, session_id="t1")
            t1.on_segment_change("s1")
            t1.on_segment_change("s2", elapsed_seconds=20.0)

            t2 = SplitTracker(best_times_path=pb, session_id="t2")
            t2.on_segment_change("s1")
            r = t2.on_segment_change("s2", elapsed_seconds=25.0)
            assert r.is_pb is False
            assert r.pb_seconds == 20.0
            assert r.diff_seconds == 5.0

    def test_faster_is_new_pb(self):
        with tempfile.TemporaryDirectory() as td:
            pb = Path(td) / "pb.json"
            t1 = SplitTracker(best_times_path=pb, session_id="t1")
            t1.on_segment_change("s1")
            t1.on_segment_change("s2", elapsed_seconds=20.0)

            t2 = SplitTracker(best_times_path=pb, session_id="t2")
            t2.on_segment_change("s1")
            r = t2.on_segment_change("s2", elapsed_seconds=15.0)
            assert r.is_pb is True
            assert r.pb_seconds == 15.0
            assert r.diff_seconds == -5.0

    def test_pb_persists_to_file(self):
        with tempfile.TemporaryDirectory() as td:
            pb = Path(td) / "pb.json"
            t = SplitTracker(best_times_path=pb, session_id="t1")
            t.on_segment_change("s1")
            t.on_segment_change("s2", elapsed_seconds=10.0)

            data = json.loads(pb.read_text())
            assert data["s1"] == 10.0


class TestSplitTrackerLog:
    def test_split_logged(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.jsonl"
            t = SplitTracker(log_path=log, session_id="t1")
            t.on_segment_change("s1", segment_name="Stage 1")
            t.on_segment_change("s2", elapsed_seconds=10.0)

            lines = log.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["event"] == "split"
            assert entry["segment_id"] == "s1"

    def test_finish_run_logged(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.jsonl"
            t = SplitTracker(log_path=log, session_id="t1")
            t.on_segment_change("s1")
            t.on_segment_change("s2", elapsed_seconds=10.0)
            t.finish_run(info={"game": "test"})

            lines = log.read_text().strip().split("\n")
            assert len(lines) == 2
            run = json.loads(lines[1])
            assert run["event"] == "run_complete"
            assert run["game"] == "test"

    def test_info_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.jsonl"
            t = SplitTracker(log_path=log, session_id="t1")
            t.on_segment_change("s1")
            t.on_segment_change("s2", elapsed_seconds=5.0, info={"custom": 42})

            entry = json.loads(log.read_text().strip())
            assert entry["custom"] == 42


class TestSplitTrackerHUD:
    def test_hud_basic(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("a", segment_name="Alpha")
        t.on_segment_change("b", elapsed_seconds=10.0, segment_name="Beta")
        lines = t.hud_lines()
        assert len(lines) == 2
        assert "Alpha" in lines[0]
        assert "Current: Beta" in lines[1]

    def test_hud_max_lines(self):
        t = SplitTracker(session_id="test")
        for i in range(10):
            t.on_segment_change(f"s{i}", elapsed_seconds=float(i + 1))
        lines = t.hud_lines(max_lines=3)
        assert len(lines) <= 3

    def test_hud_none_elapsed_shows_dash(self):
        t = SplitTracker(session_id="test")
        t.on_segment_change("a")
        t.on_segment_change("b", elapsed_seconds=None)
        lines = t.hud_lines()
        assert "--" in lines[0]
