"""Sweep verdicts: CLEAN needs the ending, zero rescues and a Clean intervention class."""
from __future__ import annotations

import json
from pathlib import Path

from harvest.tools.sweep_report import render, summarize


def _report(**over) -> dict:
    base = {
        "success": True,
        "terminal": True,
        "until_ending": True,
        "ending": {"finished": True},
        "end_date": "Y3S1D30",
        "days_completed": 210,
        "frames": 7_000_000,
        "end": {"money": 90_000, "num_cows": 4, "num_chickens": 2},
        "day_journal": [{"plan_season": 0, "plan_day": 3, "bad_day": True, "rescues": 0}],
        "day_failures": [],
        "clean_run": {"intervention_class": "Clean", "mid_run_state_loads": 0, "ram_writes": 0},
    }
    base.update(over)
    return base


def _verdict(tmp_path: Path, **over) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_report(**over)))
    return summarize(path)["verdict"]


def test_a_bad_day_without_rescue_is_still_clean(tmp_path):
    assert _verdict(tmp_path) == "CLEAN"


def test_rescues_ending_and_intervention_decide_the_verdict(tmp_path):
    assert _verdict(tmp_path, day_failures=[{"status": "rescue"}]) == "FINISHED+RESCUES"
    assert _verdict(tmp_path, day_journal=[{"plan_season": 1, "plan_day": 4, "rescues": 2}]) == "FINISHED+RESCUES"
    assert _verdict(tmp_path, ending={"finished": False}) == "UNFINISHED"
    assert _verdict(tmp_path, success=False) == "UNFINISHED"
    assert _verdict(tmp_path, clean_run={"intervention_class": "Clean", "ram_writes": 3}) == "FINISHED+INTERVENED"


def test_render_counts_clean_runs(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_report()))
    text = render([summarize(path)])
    assert "clean 1/1" in text and "S0D3" in text
