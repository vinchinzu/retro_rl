"""Offline tests for ALTTP Sanctuary-path work queue."""

from __future__ import annotations

from pathlib import Path

from alttp.paths import INTEGRATION_DIR
from alttp.opening_route.work_queue import (
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
    """Room-0x50 frontier ranks ahead of alternate / historical practice."""
    items = {i.state_name: i for i in build_catalog()}
    assert "FighterSword" in items
    assert "Castle_55" in items
    assert "CastleMain" in items
    assert "CastleRoom50" in items

    main = items["CastleMain"]
    frontier = items["CastleRoom50"]
    c55 = items["Castle_55"]
    fs = items["FighterSword"]

    # The room-0x50 frontier ranks ahead of alternate 0x55 / key path.
    assert frontier.rank < c55.rank
    assert frontier.tier == "standard"
    assert frontier.goal == "discover_b1_stairs"
    assert frontier.status == "probe_state"
    assert main.goal == "castle_dungeon_prefix"
    assert main.status == "segment_scripted"

    if "CastleMainZeldaReady" in items:
        ready = items["CastleMainZeldaReady"]
        assert ready.rank < c55.rank
        assert ready.goal == "reach_zelda_cell"
        if "CastleB1Key" in items:
            assert ready.rank < items["CastleB1Key"].rank

    if "CastleB1Key" in items:
        key = items["CastleB1Key"]
        assert frontier.rank < key.rank
        assert key.tier in {"standard", "later"}
        assert key.tier != "blocker"
        # key_shutter not ranked as primary blockers above frontier/Zelda.
        assert rank_score(frontier) < rank_score(key)

    if "CastleB1IslandCleared" in items and "CastleB1Key" in items:
        # Alternate key path still ranks near other B1 practice; not above tip.
        island = items["CastleB1IslandCleared"]
        assert rank_score(frontier) < rank_score(island)

    if "CastleMantleZelda" in items:
        mantle = items["CastleMantleZelda"]
        assert frontier.rank < mantle.rank
        if "CastleMainZeldaReady" in items:
            assert items["CastleMainZeldaReady"].rank < mantle.rank
        assert mantle.goal == "sanctuary"
        assert mantle.tier == "later"

    # Opening natural_chain is not the top "next work" vs continuous tip.
    grounds = items["HyruleCastleGrounds"]
    assert grounds.status == "natural_chain"
    assert frontier.rank < grounds.rank

    # Castle_55 is not a primary blocker; secret entrance clear is continuous.
    assert c55.status in {"segment_scripted", "probe_state"}
    assert c55.status != "blocker"
    assert c55.tier != "blocker"

    # FighterSword is done secret-entrance checkpoint, not tip.
    assert fs.goal == "secret_entrance_clear"
    assert fs.status == "segment_scripted"
    assert frontier.rank < fs.rank


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
    assert classify_group("CastleMain") == "main"
    assert classify_group("CastleMainZeldaReady") == "zelda"


def test_curated_statuses() -> None:
    items = {i.state_name: i for i in build_catalog()}
    assert items["FighterSword"].status == "segment_scripted"
    assert items["HyruleCastleGrounds"].status == "natural_chain"
    assert items["Castle_55"].status in {"segment_scripted", "probe_state"}
    assert items["Castle_55"].status != "blocker"
    assert items["FighterSword"].goal == "secret_entrance_clear"
    assert items["CastleMain"].goal == "castle_dungeon_prefix"
    if "CastleB1Key" in items:
        assert items["CastleB1Key"].tier in {"standard", "later"}


def test_build_work_queue_payload() -> None:
    payload = build_work_queue()
    assert payload["catalogId"] == "alttp_sanctuary_work_queue"
    assert payload["schemaVersion"] == 1
    assert payload["summary"]["stateCount"] == len(payload["items"])
    assert payload["summary"]["sanctuaryClaimed"] is False
    assert "workFocus" in payload
    # Focus groups are continuous-tip primary work, not post_sword/key_shutter.
    for row in payload["workFocus"]:
        assert row["group"] in {"frontier", "zelda", "b1"}
    milestones = payload["summary"]["verifiedMilestones"]
    assert "secret_entrance_clear" in milestones
    assert any("0x50" in m or "dungeon_prefix" in m for m in milestones)
    md = work_queue_to_markdown(payload)
    assert "Sanctuary" in md
    assert "0x50" in md
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
    assert "0x50" in text


def test_synthetic_names_unique_rank() -> None:
    """Catalog works without integration dir via injected names."""
    names = [
        "YazeSlot000",
        "HyruleCastleGrounds",
        "FighterSword",
        "Castle_55",
        "CastleMain",
        "CastleMainZeldaReady",
        "CastleB1Key",
        "CastleB1Pit",
        "CastleMantleZelda",
    ]
    items = build_catalog(state_names=names)
    assert len(items) == len(names)
    assert [i.rank for i in items] == list(range(1, len(names) + 1))
    by_name = {i.state_name: i for i in items}
    # Main / Zelda tip ahead of alternate key path and Castle_55.
    assert by_name["CastleMain"].rank < by_name["CastleB1Key"].rank
    assert by_name["CastleMain"].rank < by_name["Castle_55"].rank
    assert by_name["CastleMainZeldaReady"].rank < by_name["CastleB1Key"].rank
    assert by_name["CastleMain"].rank < by_name["CastleMantleZelda"].rank
    assert by_name["CastleMainZeldaReady"].rank < by_name["CastleMantleZelda"].rank
