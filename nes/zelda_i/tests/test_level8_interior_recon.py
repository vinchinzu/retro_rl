"""L8 interior fixture-recon rows stay development-only (no emulator).

rr-6o7.2.  Every ``Level8InteriorRoomRecon`` row is a live-RAM record from a
disclosed fixture replay: never route eligible, never a ``DungeonRoomSpec``,
never attached to ``L8_THROUGH``, and never a room_id promotion in the
walkthrough hypothesis graph.
"""

from __future__ import annotations

from zelda_i.level8.dungeon import (
    BLUE_GOHMA_ARROWS_REQUIRED,
    LEVEL8_INTERIOR_0X1E_RECON,
    LEVEL8_INTERIOR_0X1F_RECON,
    LEVEL8_INTERIOR_0X2E_RECON,
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_ROOM_RECON,
    LEVEL8_ROOM_SPECS,
    Level8InteriorRoomRecon,
    hypothesis_room_ids_unobserved,
)


def test_interior_recon_chain_is_ordered_south_to_north_then_east() -> None:
    assert LEVEL8_INTERIOR_ROOM_RECON == (
        LEVEL8_INTERIOR_0X3E_RECON,
        LEVEL8_INTERIOR_0X2E_RECON,
        LEVEL8_INTERIOR_0X1E_RECON,
        LEVEL8_INTERIOR_0X1F_RECON,
    )
    rooms = [row.room_id for row in LEVEL8_INTERIOR_ROOM_RECON]
    assert rooms == [0x3E, 0x2E, 0x1E, 0x1F]
    # Each row is entered from the row before it (0x4E is the 0x3E predecessor
    # and predates this table).
    assert LEVEL8_INTERIOR_0X3E_RECON.entered_from == 0x4E
    assert LEVEL8_INTERIOR_0X2E_RECON.entered_from == 0x3E
    assert LEVEL8_INTERIOR_0X1E_RECON.entered_from == 0x2E
    assert LEVEL8_INTERIOR_0X1F_RECON.entered_from == 0x1E
    # North-column prefix: high nibble -1, low nibble E, entered UP.
    for row in LEVEL8_INTERIOR_ROOM_RECON[:-1]:
        assert row.room_id & 0x0F == 0x0E
        assert row.entered_from - row.room_id == 0x10
        assert row.entry_direction == "UP"
    east = LEVEL8_INTERIOR_0X1F_RECON
    assert east.room_id == 0x1E + 1
    assert east.entry_direction == "RIGHT"


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


def test_0x1e_recon_matches_the_live_north_key_door_sitting() -> None:
    # probe_l8_2e_north.py C1/C2, 2/2 byte-identical, 2348 frames each.
    r = LEVEL8_INTERIOR_0X1E_RECON
    assert r.room_id == 0x1E
    assert r.entered_from == 0x2E
    assert r.entry_gate == "north_key_door"
    assert r.entry_pose == (120, 205)  # a door lands on the south door tile
    assert (r.keys_in, r.keys_out) == (9, 8)  # exactly one natural key
    assert (r.bombs_in, r.bombs_out) == (6, 6)  # no bomb at a key door
    # ONE body, observed type + HP only. dungeon/ids.py registers 0x33 as the
    # L6 "gohma_red" body; the walkthrough calls this room a *blue* Gohma. The
    # row records what RAM showed and asserts no colour.
    assert r.census == ((0x33, 96, 1),)
    assert r.room_item_id == 0x03
    assert r.fixture == "Level8Interior1EReconFixture"


def test_0x1f_recon_matches_the_live_east_kill_clear_sitting() -> None:
    # probe_l8_1e_gohma.py D1b/D2, 2/2 byte-identical, 1731 frames each.
    r = LEVEL8_INTERIOR_0X1F_RECON
    assert r.room_id == 0x1F
    assert r.entered_from == 0x1E
    assert r.entry_gate == "east_kill_clear_shutter"
    assert r.entry_pose == (16, 141)  # west mouth of a RIGHT door
    assert (r.keys_in, r.keys_out) == (8, 8)  # no key at a kill-clear shutter
    assert (r.bombs_in, r.bombs_out) == (6, 6)  # no bomb
    # Mixed 0x1F population; 0x68 stairs sprite is not a census row.
    assert r.census == ((0x16, 160, 2), (0x0C, 128, 2), (0x0B, 64, 2))
    assert r.room_item_id == 0x03
    assert r.fixture == "Level8Interior1FReconFixture"
    # rr-gw0x: three connecting wooden arrows (HP 96→64→32→0). Colour is not
    # asserted — RAM type was 0x33, never 0x34. 9 shots loosed, 6 misses.
    assert BLUE_GOHMA_ARROWS_REQUIRED == 3
