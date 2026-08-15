"""Unit tests for Survival damage heatmap ranking (fixture JSON only)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from zelda_i.damage_heatmap import (
    L5_SUFFIX_HEAT,
    RoomHeat,
    extract_assist_block,
    format_heatmap_table,
    l5_suffix_fixture_report,
    rank_damage,
    rank_report_paths,
)


def _cli_main():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rank_damage_heatmap.py"
    spec = importlib.util.spec_from_file_location("rank_damage_heatmap_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_l5_suffix_fixture_ranks_hot_rooms() -> None:
    ranked = rank_damage(l5_suffix_fixture_report(), source="l5_suffix")
    assert [room.location for room in ranked] == [
        "L5:0x24",
        "L5:0x66",
        "L5:0x26",
        "L5:0x56",
        "L5:0x57",
    ]
    assert ranked[0] == RoomHeat("L5:0x24", 27, 0, ("l5_suffix",))
    assert ranked[1].damage == 10
    assert ranked[2].damage == 4
    assert ranked[3].damage == ranked[4].damage == 1
    assert sum(room.damage for room in ranked) == 43
    assert L5_SUFFIX_HEAT["L5:0x24"] == 27


def test_nested_assist_block_and_samples_count_writes() -> None:
    report = {
        "segment": "l5_whistle_04_to_tf",
        "assist": {
            "total_damage": 43,
            "damage_by_location": dict(L5_SUFFIX_HEAT),
            "health": {"writes": 20, "restored": 44},
            "damage_samples": (
                [{"location": "L5:0x24", "amount": 3}] * 10
                + [{"location": "L5:0x66", "amount": 2}] * 5
                + [{"level": 5, "screen": 0x26, "amount": 2}] * 2
                + [{"location": "L5:0x56", "amount": 1}]
                + [{"location": "L5:0x57", "amount": 1}]
            ),
        },
    }
    block = extract_assist_block(report)
    assert block["total_damage"] == 43
    ranked = rank_damage(report)
    by_loc = {room.location: room for room in ranked}
    assert by_loc["L5:0x24"].writes == 10
    assert by_loc["L5:0x66"].writes == 5
    assert by_loc["L5:0x26"].writes == 2
    assert ranked[0].sources == ("l5_whistle_04_to_tf",)


def test_merges_multiple_reports() -> None:
    a = {"_source": "a", "damage_by_location": {"L5:0x24": 10, "L2:0x5c": 2}}
    b = {"_source": "b", "damage_by_location": {"L5:0x24": 17, "L5:0x66": 4}}
    ranked = rank_damage([a, b])
    assert ranked[0].location == "L5:0x24"
    assert ranked[0].damage == 27
    assert set(ranked[0].sources) == {"a", "b"}
    assert {room.location: room.damage for room in ranked} == {
        "L5:0x24": 27,
        "L5:0x66": 4,
        "L2:0x5c": 2,
    }


def test_cli_prints_table_from_fixture_json(tmp_path: Path, capsys) -> None:
    path = tmp_path / "l5_suffix_fixture.json"
    path.write_text(json.dumps(l5_suffix_fixture_report()), encoding="utf-8")
    assert _cli_main()([str(path)]) == 0
    out = capsys.readouterr().out
    assert "L5:0x24" in out
    assert "27" in out
    assert "L5:0x66" in out
    # Hottest row first.
    assert out.index("L5:0x24") < out.index("L5:0x66")
    assert rank_report_paths([path])[0].location == "L5:0x24"
    table = format_heatmap_table([])
    assert "no damage_by_location" in table


def test_cli_missing_file(tmp_path: Path) -> None:
    assert _cli_main()([str(tmp_path / "missing.json")]) == 2
