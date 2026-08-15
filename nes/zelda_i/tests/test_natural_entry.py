"""Catalog tests for Zelda I natural-entry vs isolated/assisted greens."""

from __future__ import annotations

from zelda_i.natural_entry import (
    SEGMENTS,
    get_segment,
    missing_natural_entry,
    status_claim_allowed,
)

_REQUIRED_IDS = {
    "l1_complete",
    "l2_path_prefix_0x4a",
    "l2_door_path_0x3c",
    "l2_room_0x6d",
    "l2_room_0x6c",
    "l2_room_0x7e",
    "l2_room_0x6f",
    "l2_complete_tf",
    "l3_entry_from_l2",
    "l4_entry_from_l3",
    "l5_entry_from_l4",
    "l5_room_0x66",
    "l5_east_key_0x77",
    "l5_east_to_whistle_0x04",
    "l5_whistle_0x04_to_tf",
    "l9_room_0x51_to_0x41",
    "l9_room_0x41_suffix",
}


def test_required_segments_are_seeded() -> None:
    ids = {entry.segment_id for entry in SEGMENTS}
    assert _REQUIRED_IDS <= ids
    assert len(ids) == len(SEGMENTS)


def test_l1_complete_is_only_status_eligible() -> None:
    l1 = get_segment("l1_complete")
    assert l1 is not None
    assert l1.natural_entry is True
    assert l1.isolated_clean is True
    assert l1.status_eligible is True
    assert l1.predecessor == "power_on"
    assert l1.blocker == ""
    assert status_claim_allowed("l1_complete") is True
    for entry in SEGMENTS:
        allowed = status_claim_allowed(entry.segment_id)
        if entry.segment_id == "l1_complete":
            assert allowed is True
        else:
            assert allowed is False
            assert entry.status_eligible is False
    assert status_claim_allowed("not_a_segment") is False


def test_l2_prefix_natural_from_l1_exit() -> None:
    prefix = get_segment("l2_path_prefix_0x4a")
    assert prefix is not None
    assert prefix.isolated_clean is True
    assert prefix.natural_entry is True
    assert prefix.predecessor == "l1_exit_overworld"
    assert prefix.status_eligible is False


def test_l2_door_path_blocked_by_0x5c_hearts() -> None:
    door = get_segment("l2_door_path_0x3c")
    assert door is not None
    assert door.isolated_clean is True
    assert door.assisted_green is True
    assert door.natural_entry is False
    assert door.blocker == "heart_starvation_0x5c"
    assert status_claim_allowed("l2_door_path_0x3c") is False


def test_l2_interior_rooms_isolated_not_natural() -> None:
    for segment_id in ("l2_room_0x6d", "l2_room_0x6c", "l2_room_0x7e", "l2_room_0x6f"):
        room = get_segment(segment_id)
        assert room is not None
        assert room.isolated_clean is True
        assert room.natural_entry is False
        assert room.blocker == "mid_run_state_load"
        assert room.status_eligible is False


def test_assisted_entries_are_not_natural_status() -> None:
    l2tf = get_segment("l2_complete_tf")
    l3 = get_segment("l3_entry_from_l2")
    l4 = get_segment("l4_entry_from_l3")
    l5 = get_segment("l5_entry_from_l4")
    assert l2tf is not None and l2tf.assisted_green and not l2tf.natural_entry
    assert l3 is not None and l3.assisted_green and l3.predecessor == "Level2ExitOverworld"
    assert l4 is not None and l4.assisted_green and l4.predecessor == "Level3ExitOverworld"
    assert l5 is not None
    assert l5.assisted_green is True
    assert l5.natural_entry is False
    assert l5.predecessor == "Level4Complete"


def test_l5_and_l9_seams() -> None:
    gibdo = get_segment("l5_room_0x66")
    east = get_segment("l5_east_key_0x77")
    e2w = get_segment("l5_east_to_whistle_0x04")
    whistle = get_segment("l5_whistle_0x04_to_tf")
    l9 = get_segment("l9_room_0x41_suffix")
    assert gibdo is not None
    assert gibdo.isolated_clean is True
    assert gibdo.assisted_green is True
    assert gibdo.predecessor == "Level5EntranceFromL4"
    assert east is not None and east.predecessor == "Level5Cleared66"
    assert e2w is not None
    assert e2w.assisted_green is True
    assert e2w.natural_entry is False
    assert e2w.predecessor == "Level5EastKey"
    assert whistle is not None
    assert whistle.assisted_green is True
    assert whistle.natural_entry is False
    assert whistle.blocker == "not_composed_onto_east_key"
    assert whistle.predecessor == "Level5WhistleFrom77"
    pred = get_segment("l9_room_0x51_to_0x41")
    assert pred is not None
    assert pred.blocker == "dest_no_statue_diamond"
    assert l9 is not None
    assert l9.blocker == "fixture_only"
    assert l9.predecessor == "l9_room_0x51_to_0x41"
    assert status_claim_allowed("l9_room_0x41_suffix") is False


def test_missing_natural_entry_lists_assisted_and_isolated_gaps() -> None:
    missing = {entry.segment_id for entry in missing_natural_entry()}
    assert "l1_complete" not in missing
    assert "l2_path_prefix_0x4a" not in missing
    assert "l2_door_path_0x3c" in missing
    assert "l5_whistle_0x04_to_tf" in missing
    assert "l9_room_0x41_suffix" in missing
