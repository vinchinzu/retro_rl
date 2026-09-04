"""L8 interior fixture-recon rows stay development-only (no emulator).

rr-6o7.2.  Every ``Level8InteriorRoomRecon`` row is a live-RAM record from a
disclosed fixture replay: never route eligible, never a ``DungeonRoomSpec``,
never attached to ``L8_THROUGH``, and never a room_id promotion in the
walkthrough hypothesis graph.
"""

from __future__ import annotations

from zelda_i.level8.dungeon import (
    LEVEL8_INTERIOR_0X2E_RECON,
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_ROOM_RECON,
    LEVEL8_ROOM_SPECS,
    Level8InteriorRoomRecon,
    hypothesis_room_ids_unobserved,
)


def test_interior_recon_chain_is_ordered_south_to_north() -> None:
    assert LEVEL8_INTERIOR_ROOM_RECON == (
        LEVEL8_INTERIOR_0X3E_RECON,
        LEVEL8_INTERIOR_0X2E_RECON,
    )
    rooms = [row.room_id for row in LEVEL8_INTERIOR_ROOM_RECON]
    assert rooms == [0x3E, 0x2E]
    # Each row is entered from the row before it (0x4E is the 0x3E predecessor
    # and predates this table).
    assert LEVEL8_INTERIOR_0X3E_RECON.entered_from == 0x4E
    assert LEVEL8_INTERIOR_0X2E_RECON.entered_from == 0x3E
    # Screen ids walk north one row per boundary: high nibble -1, low nibble E.
    for row in LEVEL8_INTERIOR_ROOM_RECON:
        assert row.room_id & 0x0F == 0x0E
        assert row.entered_from - row.room_id == 0x10
        assert row.entry_direction == "UP"


def test_interior_recon_rows_are_never_route_eligible() -> None:
    from zelda_i.level8.spine import L8_THROUGH

    for row in LEVEL8_INTERIOR_ROOM_RECON:
        assert isinstance(row, Level8InteriorRoomRecon)
        assert row.route_eligible is False
        assert row.evidence == "live_recon_fixture"
        assert row.fixture  # a disclosed state + provenance sidecar exists
        assert row.recording_tag
        assert f"level8-interior-0x{row.room_id:02x}" not in L8_THROUGH
    assert LEVEL8_ROOM_SPECS == ()
    assert hypothesis_room_ids_unobserved()


def test_0x2e_recon_matches_the_live_bomb_north_sitting() -> None:
    # probe_l8_3e_north.py A2/A3, 2/2 byte-identical, 4788 frames each.
    r = LEVEL8_INTERIOR_0X2E_RECON
    assert r.room_id == 0x2E
    assert r.entered_from == 0x3E
    assert r.entry_gate == "north_bomb_wall"
    assert r.entry_pose == (120, 189)  # blasted wall lands lower than a door
    assert (r.keys_in, r.keys_out) == (9, 9)  # no key spent at a bomb wall
    assert (r.bombs_in, r.bombs_out) == (7, 6)  # exactly one natural bomb
    assert r.census == ((0x3C, 64, 5),)  # Manhandla body + 4 heads
    assert r.room_item_id == 0x17
    assert r.fixture == "Level8Interior2EReconFixture"


def test_recon_key_and_bomb_ledger_is_continuous() -> None:
    # Counts must chain: each row's outgoing budget is the next row's incoming.
    for before, after in zip(
        LEVEL8_INTERIOR_ROOM_RECON, LEVEL8_INTERIOR_ROOM_RECON[1:]
    ):
        assert before.keys_out == after.keys_in
        assert before.bombs_out == after.bombs_in
    # Nothing was gained: keys and bombs only ever go down across the chain.
    for row in LEVEL8_INTERIOR_ROOM_RECON:
        assert row.keys_out <= row.keys_in
        assert row.bombs_out <= row.bombs_in
