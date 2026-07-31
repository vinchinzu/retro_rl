"""Offline tests for ALTTP Sanctuary-path work queue."""

from __future__ import annotations

from pathlib import Path

from alttp.paths import INTEGRATION_DIR
from alttp.work_queue import (
    build_catalog,
    build_work_queue,
    classify_group,
    export_work_queue,
    list_state_names,
    rank_score,
    work_queue_to_markdown,
)


def test_list_state_names_nonempty() -> None:
    names = list_state_names()
    assert len(names) > 0
    assert all(not n.endswith(".state") for n in names)
    # Integration dir should match on-disk count when present.
    if INTEGRATION_DIR.is_dir():
        on_disk = sorted(p.stem for p in INTEGRATION_DIR.glob("*.state"))
        assert names == on_disk


def test_catalog_size_and_unique_names() -> None:
    items = build_catalog()
    assert len(items) > 0
    names = [i.state_name for i in items]
    assert len(names) == len(set(names))
    assert len(items) == len(list_state_names())


def test_known_states_present() -> None:
    names = {i.state_name for i in build_catalog()}
    assert "FighterSword" in names
    assert "HyruleCastleGrounds" in names
    # Mantle state is in the integration set used for escort.
    if (INTEGRATION_DIR / "CastleMantleZelda.state").is_file():
        assert "CastleMantleZelda" in names


def test_ranking_stable_and_ordered() -> None:
    a = build_catalog()
    b = build_catalog()
    assert [i.state_name for i in a] == [i.state_name for i in b]
    scores = [i.rank_score for i in a]
    assert scores == sorted(scores)
    ranks = [i.rank for i in a]
    assert ranks == list(range(1, len(a) + 1))


def test_sanctuary_priority_policy() -> None:
    """After sword: 0x55 / key / shutter before random B1; escort later."""
    items = {i.state_name: i for i in build_catalog()}
    assert "FighterSword" in items
    assert "Castle_55" in items

    fs = items["FighterSword"]
    c55 = items["Castle_55"]
    # Critical path ranks ahead of a random B1 island clear when present.
    if "CastleB1IslandCleared" in items:
        island = items["CastleB1IslandCleared"]
        assert rank_score(c55) < rank_score(island)
        assert rank_score(items.get("CastleB1Key", c55)) <= rank_score(island) + 0

    if "CastleB1Key" in items:
        assert items["CastleB1Key"].rank < items.get(
            "CastleB1IslandCleared", items["CastleB1Key"]
        ).rank or "CastleB1IslandCleared" not in items

    if "CastleMantleZelda" in items:
        mantle = items["CastleMantleZelda"]
        assert c55.rank < mantle.rank
        assert mantle.goal == "sanctuary"
        assert mantle.tier == "later"

    # Opening natural_chain is not the top "next work" vs 0x55 blocker.
    grounds = items["HyruleCastleGrounds"]
    assert grounds.status == "natural_chain"
    assert c55.status == "blocker"
    assert c55.rank < grounds.rank


def test_classify_group_heuristics() -> None:
    assert classify_group("CastleB1Key") == "key_shutter"
    assert classify_group("CastleB1ShutterRoom") == "key_shutter"
    assert classify_group("CastleB1Pit") == "b1"
    assert classify_group("CastleB2Landing") == "b2"
    assert classify_group("CastleB3BallApproach") == "b3"
    assert classify_group("FighterSword") == "post_sword"
    assert classify_group("HyruleCastleGrounds") == "opening"
    assert classify_group("CastleMantleZelda") == "escort"
    assert classify_group("CastleZeldaFollower") == "zelda"
    assert classify_group("Castle_55") == "room_55"


def test_curated_statuses() -> None:
    items = {i.state_name: i for i in build_catalog()}
    assert items["FighterSword"].status == "segment_scripted"
    assert items["HyruleCastleGrounds"].status == "natural_chain"
    assert items["Castle_55"].status == "blocker"
    assert items["FighterSword"].goal == "exit_0x55"


def test_build_work_queue_payload() -> None:
    payload = build_work_queue()
    assert payload["catalogId"] == "alttp_sanctuary_work_queue"
    assert payload["schemaVersion"] == 1
    assert payload["summary"]["stateCount"] == len(payload["items"])
    assert payload["summary"]["sanctuaryClaimed"] is False
    assert "workFocus" in payload
    md = work_queue_to_markdown(payload)
    assert "Sanctuary" in md
    assert "FighterSword" in md


def test_export_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "room_work_queue.json"
    md_out = tmp_path / "ROOM_WORK_QUEUE.md"
    payload = export_work_queue(json_output=json_out, md_output=md_out)
    assert json_out.is_file()
    assert md_out.is_file()
    assert payload["summary"]["stateCount"] > 0
    text = md_out.read_text(encoding="utf-8")
    assert "work queue" in text.lower() or "Work Queue" in text
    assert "exit_0x55" in text or "0x55" in text


def test_synthetic_names_unique_rank() -> None:
    """Catalog works without integration dir via injected names."""
    names = [
        "YazeSlot000",
        "HyruleCastleGrounds",
        "FighterSword",
        "Castle_55",
        "CastleB1Key",
        "CastleB1Pit",
        "CastleMantleZelda",
    ]
    items = build_catalog(state_names=names)
    assert len(items) == len(names)
    assert [i.rank for i in items] == list(range(1, len(names) + 1))
    by_name = {i.state_name: i for i in items}
    assert by_name["Castle_55"].rank < by_name["CastleB1Pit"].rank
    assert by_name["CastleB1Key"].rank < by_name["CastleMantleZelda"].rank
