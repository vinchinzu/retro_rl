"""Shared leftover glance grammar (no emulator)."""

from __future__ import annotations

from retro_harness.glance import (
    GlanceLeftover,
    LeaveMiss,
    band_miss,
    grade_report,
    in_band,
    parse_int,
    pick,
    xy_of,
)


def test_parse_int_hex_and_decimal() -> None:
    assert parse_int("0x3A") == 0x3A
    assert parse_int("0xcd13") == 0xCD13
    assert parse_int(58) == 58
    assert parse_int(None) == 0
    assert parse_int(None, default=-1) == -1


def test_xy_of_pair_and_aliases() -> None:
    assert xy_of({"xy": [120, 141]}) == (120, 141)
    assert xy_of({"x": 8, "y": 9}) == (8, 9)
    assert xy_of(
        {"link_x": 176, "link_y": 77},
        x_keys=("x", "link_x"),
        y_keys=("y", "link_y"),
    ) == (176, 77)


def test_in_band_and_band_miss() -> None:
    assert in_band(120, (112, 128))
    assert not in_band(200, (112, 128))
    assert in_band(8, 8)
    assert in_band(1, None)
    assert band_miss("x", 200, (112, 128)) == "x=200 not in [112, 128]"
    assert band_miss("x", 120, (112, 128)) is None


def test_pick_skips_none() -> None:
    assert pick({"room": None, "screen": 0x3A}, "room", "screen") == 0x3A
    assert pick({}, "x") is None


def test_grade_report_dual_run_and_top_level_final() -> None:
    def grade(final):
        x, _y = xy_of(final)
        miss = band_miss("x", x, (100, 140))
        return [] if miss is None else [miss]

    assert grade_report({"final": {"xy": [120, 141]}}, grade) == []
    assert grade_report({}, grade) == ["missing final"]
    dual = {
        "success": True,
        "runs": [
            {"success": True, "final": {"x": 120, "y": 141}},
            {"success": True, "final": {"x": 200, "y": 141}},
        ],
    }
    misses = grade_report(dual, grade)
    assert misses == ["run 2: x=200 not in [100, 140]"]


def test_grade_report_prepare_final_and_failed_run() -> None:
    def grade(final):
        return [] if final.get("boss") == 1 else ["boss miss"]

    def prepare(final, run):
        if "boss" not in final and "boss" in run:
            out = dict(final)
            out["boss"] = run["boss"]
            return out
        return final

    report = {
        "success": False,
        "runs": [{"success": False, "final": {"x": 1}, "boss": 1}],
    }
    misses = grade_report(report, grade, prepare_final=prepare)
    assert "success is false" in misses
    assert "run 1 success is false" in misses
    assert all("boss miss" not in m for m in misses)


def test_leave_miss_and_glance_leftover() -> None:
    leftover = {"room": 0x3A, "xy": [144, 141]}
    exc = LeaveMiss("clear3a", leftover, ["room 0x22 != 0x3A"])
    assert exc.hop_id == "clear3a"
    assert exc.leftover == leftover
    assert "clear3a" in str(exc)
    glance = GlanceLeftover(ok=False, leftover=leftover, misses=exc.misses)
    assert not glance.ok
    assert glance.leftover["room"] == 0x3A
