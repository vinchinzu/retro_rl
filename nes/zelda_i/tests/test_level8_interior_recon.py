"""L8 interior fixture-recon rows stay development-only (no emulator).

rr-6o7.2.  Every ``Level8InteriorRoomRecon`` row is a live-RAM record from a
disclosed fixture replay: never route eligible, never a ``DungeonRoomSpec``,
never attached to ``L8_THROUGH``, and never a room_id promotion in the
walkthrough hypothesis graph.
"""

from __future__ import annotations

import pytest

from zelda_i.level8 import gleeok_entry, passage, path, stairs, triforce
from zelda_i.level8.dungeon import (
    LEVEL8_INTERIOR_0X0F_RECON,
    LEVEL8_INTERIOR_0X1E_RECON,
    LEVEL8_INTERIOR_0X1E_WEST_RECON,
    LEVEL8_INTERIOR_0X1F_RECON,
    LEVEL8_INTERIOR_0X2C_TF_RECON,
    LEVEL8_INTERIOR_0X2E_RECON,
    LEVEL8_INTERIOR_0X2E_SOUTH_RECON,
    LEVEL8_INTERIOR_0X2F_STAIRS_RECON,
    LEVEL8_INTERIOR_0X3C_NORTH_RECON,
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_0X3E_SOUTH_RECON,
    LEVEL8_INTERIOR_0X3F_EAST_RECON,
    LEVEL8_INTERIOR_0X4C_WEST_RECON,
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
        LEVEL8_INTERIOR_0X0F_RECON,
    )
    rooms = [row.room_id for row in LEVEL8_INTERIOR_ROOM_RECON]
    assert rooms == [0x3E, 0x2E, 0x1E, 0x1F, 0x0F]
    # Each row is entered from the row before it (0x4E is the 0x3E predecessor
    # and predates this table).
    assert LEVEL8_INTERIOR_0X3E_RECON.entered_from == 0x4E
    assert LEVEL8_INTERIOR_0X2E_RECON.entered_from == 0x3E
    assert LEVEL8_INTERIOR_0X1E_RECON.entered_from == 0x2E
    assert LEVEL8_INTERIOR_0X1F_RECON.entered_from == 0x1E
    assert LEVEL8_INTERIOR_0X0F_RECON.entered_from == 0x1F
    # North-column prefix: high nibble -1, low nibble E, entered UP.
    for row in LEVEL8_INTERIOR_ROOM_RECON[:3]:
        assert row.room_id & 0x0F == 0x0E
        assert row.entered_from - row.room_id == 0x10
        assert row.entry_direction == "UP"
    east = LEVEL8_INTERIOR_0X1F_RECON
    assert east.room_id == 0x1E + 1
    assert east.entry_direction == "RIGHT"
    cellar = LEVEL8_INTERIOR_0X0F_RECON
    assert cellar.entry_direction == "STAIRS"


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


@pytest.mark.parametrize(
    "row, dest, pose",
    [
        (LEVEL8_INTERIOR_0X1E_WEST_RECON, path.WEST_DEST, path.WEST_DEST_POSE),
        (LEVEL8_INTERIOR_0X2E_SOUTH_RECON, path.SOUTH_DEST, path.SOUTH_DEST_POSE),
        (LEVEL8_INTERIOR_0X3E_SOUTH_RECON, path.SOUTH_2E_DEST, path.SOUTH_2E_DEST_POSE),
        (LEVEL8_INTERIOR_0X3F_EAST_RECON, path.EAST_3E_DEST, path.EAST_3E_DEST_POSE),
        (LEVEL8_INTERIOR_0X2F_STAIRS_RECON, stairs.STAIRS_3F_DEST, passage.SPAWN_XY),
        (LEVEL8_INTERIOR_0X4C_WEST_RECON, passage.DEST, passage.DEST_POSE),
        (LEVEL8_INTERIOR_0X3C_NORTH_RECON, gleeok_entry.DEST, gleeok_entry.DEST_POSE),
        (LEVEL8_INTERIOR_0X2C_TF_RECON, triforce.NORTH_3C_DEST, triforce.NORTH_3C_DEST_POSE),
    ],
    ids=lambda v: f"0x{v.room_id:02x}" if isinstance(v, Level8InteriorRoomRecon) else "",
)
def test_off_chain_recon_row_is_the_pose_its_controller_arrives_at(row, dest, pose) -> None:
    """Each off-chain recon row records the room and settled pose that the
    controller claiming it treats as success; the two must not drift."""
    assert row.room_id == dest
    assert row.entry_pose == pose
    assert row.route_eligible is False
    assert row not in LEVEL8_INTERIOR_ROOM_RECON
