"""Offline tests for multi-truth anchors and Segment registry."""

from __future__ import annotations

from alttp.opening_route.anchors import (
    STATE_SEMANTICS,
    anchor_by_id,
    match_anchors,
    opening_anchors,
)
from alttp.opening_route.segment import get_segment, list_segments, segment_registry
from alttp.paths import FIGHTER_SWORD_STATE, HYRULE_CASTLE_GROUNDS_STATE, STATE_SEMANTIC_NAMES
from alttp.ram import SECRET_PASSAGE_ROOM, AlttpSnapshot


def _snap(**kwargs: object) -> AlttpSnapshot:
    base: dict[str, object] = dict(
        game_mode=0x07,
        submodule=0x00,
        room_id=0,
        indoors=False,
        screen_id=0x1B,
        link_x=2000,
        link_y=1600,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=0,
        lamp_level=0,
        num_keys=0xFF,
        follower=0,
    )
    base.update(kwargs)
    return AlttpSnapshot(**base)  # type: ignore[arg-type]


def test_opening_anchors_nonempty_and_unique_ids() -> None:
    anchors = opening_anchors()
    assert len(anchors) >= 6
    ids = [a.anchor_id for a in anchors]
    assert len(ids) == len(set(ids))
    tiers = {a.tier for a in anchors}
    assert "route" in tiers
    assert "approach" in tiers
    assert "trigger" in tiers


def test_grounds_spawn_matches_controllable_screen() -> None:
    a = anchor_by_id("HyruleCastle_GroundsSpawn_Controllable")
    assert a is not None
    assert a.matches(_snap(screen_id=0x1B, indoors=False, game_mode=0x09))
    assert not a.matches(_snap(screen_id=0x2C, indoors=False))


def test_secret_hole_approach_position_window() -> None:
    a = anchor_by_id("HyruleCastle_SecretPassageApproach")
    assert a is not None
    near = _snap(screen_id=0x1B, indoors=False, link_x=2430, link_y=1704, game_mode=0x09)
    far = _snap(screen_id=0x1B, indoors=False, link_x=1000, link_y=1000, game_mode=0x09)
    assert a.matches(near)
    assert not a.matches(far)


def test_fighter_sword_anchor_requires_inventory() -> None:
    a = anchor_by_id("HyruleCastle_SecretEntrance_FighterSword")
    assert a is not None
    bare = _snap(indoors=True, room_id=SECRET_PASSAGE_ROOM, sword_level=0)
    armed = _snap(indoors=True, room_id=SECRET_PASSAGE_ROOM, sword_level=1)
    assert not a.matches(bare)
    assert a.matches(armed)


def test_courtyard_pocket_anchor() -> None:
    a = anchor_by_id("HyruleCastle_Courtyard_SecretStairsPocket")
    assert a is not None
    pocket = _snap(
        indoors=False,
        screen_id=0x1B,
        link_x=2248,
        link_y=1755,
        sword_level=1,
        game_mode=0x09,
    )
    assert a.matches(pocket)
    assert any(
        m.anchor_id == "HyruleCastle_Courtyard_SecretStairsPocket"
        for m in match_anchors(pocket)
    )


def test_main_door_approach_and_hall_anchors() -> None:
    approach = anchor_by_id("HyruleCastle_MainDoorApproach")
    assert approach is not None
    at_door = _snap(
        indoors=False,
        screen_id=0x1B,
        link_x=2040,
        link_y=1790,
        sword_level=1,
        game_mode=0x09,
    )
    assert approach.matches(at_door)
    hall = anchor_by_id("HyruleCastle_MainHall")
    assert hall is not None
    from alttp.ram import HYRULE_CASTLE_MAIN_HALL_ROOM

    indoor = _snap(
        indoors=True,
        room_id=HYRULE_CASTLE_MAIN_HALL_ROOM,
        sword_level=1,
        game_mode=0x07,
    )
    assert hall.matches(indoor)
    assert "pocket_to_main_hall" in list_segments()


def test_state_semantics_cover_key_dev_states() -> None:
    assert HYRULE_CASTLE_GROUNDS_STATE in STATE_SEMANTICS
    assert FIGHTER_SWORD_STATE in STATE_SEMANTICS
    assert "bridge" not in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE].lower() or (
        "NOT" in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE]
        or "not" in STATE_SEMANTICS[HYRULE_CASTLE_GROUNDS_STATE]
    )
    assert STATE_SEMANTIC_NAMES[HYRULE_CASTLE_GROUNDS_STATE].startswith("HyruleCastle_")


def test_segment_registry_has_continuous_segments() -> None:
    reg = segment_registry()
    assert "castle_to_sword" in reg
    assert "sword_to_secret_entrance_clear" in reg
    assert set(list_segments()) == set(reg)
    seg = get_segment("castle_to_sword")
    assert seg.exit.verification == "continuous"
    assert seg.entry.graph_node_id == "castle_grounds"
