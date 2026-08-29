"""Unit tests for shinespark.diagnose_trace (no emulator)."""

from __future__ import annotations

from super_metroid.routes.skills import shinespark as spark


def test_empty() -> None:
    d = spark.diagnose_trace([])
    assert d["ok"] is False
    assert d["grade"] == "EMPTY"


def test_red_no_charge() -> None:
    rows = [
        {
            "frame": i,
            "x": 900 + i,
            "y": 1163,
            "pose": 9,
            "buttons": ["RIGHT", "B"],
            "speed_echoes": 0,
            "spark_timer": 0,
        }
        for i in range(40)
    ]
    d = spark.diagnose_trace(rows)
    assert d["grade"] == "RED"
    assert any("charge" in f for f in d["failures"])


def test_green_spark() -> None:
    rows: list[dict] = []
    for i in range(90):
        rows.append(
            {
                "frame": i,
                "x": 900 + i,
                "y": 1163,
                "pose": 9,
                "buttons": ["RIGHT", "B"],
                "speed_echoes": 4 if i > 85 else 2,
                "spark_timer": 0,
            }
        )
    for i in range(90, 95):
        rows.append(
            {
                "frame": i,
                "x": 1100,
                "y": 1163,
                "pose": 53,
                "buttons": ["DOWN"],
                "speed_echoes": 0,
                "spark_timer": 179,
            }
        )
    for i in range(95, 99):
        rows.append(
            {
                "frame": i,
                "x": 1100,
                "y": 1163,
                "pose": 39,
                "buttons": ["UP"],
                "speed_echoes": 0,
                "spark_timer": 170,
            }
        )
    for i in range(99, 130):
        rows.append(
            {
                "frame": i,
                "x": 1100 + i * 3,
                "y": 1163,
                "pose": 201,
                "buttons": ["RIGHT", "A"],
                "speed_echoes": 0,
                "spark_timer": 100,
            }
        )
    d = spark.diagnose_trace(rows)
    assert d["ok"] is True
    assert d["grade"] == "GREEN"
    assert d["peaks"]["spark_travel_frames"] >= 3


def test_orange_late_store_after_charge_died() -> None:
    """Charge full while holding B, never DOWN in window, DOWN a few frames later."""
    rows: list[dict] = []
    # build to echoes 4
    for i in range(90):
        rows.append(
            {
                "frame": i,
                "x": 900 + i,
                "y": 1163,
                "pose": 9,
                "buttons": ["RIGHT", "B"],
                "speed_echoes": min(4, i // 22),
                "spark_timer": 0,
            }
        )
    # hold B through full charge ~20f (no DOWN)
    for i in range(90, 110):
        rows.append(
            {
                "frame": i,
                "x": 990 + (i - 90),
                "y": 1163,
                "pose": 9,
                "buttons": ["RIGHT", "B"],
                "speed_echoes": 4,
                "spark_timer": 0,
            }
        )
    # charge dies the way humans do: release RIGHT, keep B → e dumps
    for i in range(110, 114):
        rows.append(
            {
                "frame": i,
                "x": 1010,
                "y": 1163,
                "pose": 9,
                "buttons": ["B"],  # direction released
                "speed_echoes": 0,
                "spark_timer": 0,
            }
        )
    # late crouch
    for i in range(114, 130):
        rows.append(
            {
                "frame": i,
                "x": 1010,
                "y": 1163,
                "pose": 53 if i < 116 else 39,
                "buttons": ["DOWN"],
                "speed_echoes": 0,
                "spark_timer": 0,
            }
        )
    d = spark.diagnose_trace(rows)
    assert d["ok"] is False
    assert d["grade"] == "ORANGE"
    assert d["milestones"]["late_store_after_charge_died"] is True
    assert d["peaks"]["missed_store_windows"] >= 1
    assert any("late crouch" in f for f in d["failures"])
    assert any(
        "releasing LEFT/RIGHT" in f or "B alone" in f or "late crouch" in f
        for f in d["failures"]
    )
    assert any("PRESS DOWN" in c or "ALSO press DOWN" in c or "CRITICAL" in c for c in d["cues"])


def test_yellow_crouch_walk() -> None:
    rows: list[dict] = []
    for i in range(90):
        rows.append(
            {
                "frame": i,
                "x": 900 + i,
                "y": 1163,
                "pose": 9,
                "buttons": ["RIGHT", "B"],
                "speed_echoes": 4 if i > 80 else min(4, i // 20),
                "spark_timer": 0,
            }
        )
    for i in range(90, 110):
        rows.append(
            {
                "frame": i,
                "x": 1000,
                "y": 1163,
                "pose": 53,
                "buttons": ["DOWN"],
                "speed_echoes": 0,
                "spark_timer": max(0, 179 - (i - 90)),
            }
        )
    for i in range(110, 140):
        rows.append(
            {
                "frame": i,
                "x": 1000 + (i - 110),
                "y": 1163,
                "pose": 39,
                "buttons": ["RIGHT", "A"],
                "speed_echoes": 0,
                "spark_timer": max(0, 179 - (i - 90)),
            }
        )
    d = spark.diagnose_trace(rows)
    assert d["ok"] is False
    assert d["grade"] == "YELLOW"
    assert d["milestones"]["activate_from_crouch_walk"] is True
