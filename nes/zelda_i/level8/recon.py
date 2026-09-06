"""Level 8 interior room recon records from fixture replays.

These records capture development-recon evidence only: route_eligible is
always False and these rows are never attached to L8_THROUGH or promoted
to DungeonRoomSpec. They record live RAM observations from 2/2 byte-identical
fixture-recon probes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level8InteriorRoomRecon:
    """A single interior room observed live from a disclosed fixture replay.

    Development recon only: route_eligible is always False and these rows
    are never attached to L8_THROUGH or promoted to DungeonRoomSpec.
    They only record what RAM showed, with a 2/2 byte-identical recording.
    """

    room_id: int
    entered_from: int
    entry_direction: str
    entry_gate: str
    entry_pose: tuple[int, int]
    keys_in: int
    keys_out: int
    bombs_in: int
    bombs_out: int
    room_item_id: int
    census: tuple[tuple[int, int, int], ...]  # (type_id, hp, count)
    evidence: str = "live_recon_fixture"
    route_eligible: bool = False
    fixture: str = ""
    recording_tag: str = ""


LEVEL8_INTERIOR_0X3E_RECON = Level8InteriorRoomRecon(
    room_id=0x3E,
    entered_from=0x4E,
    entry_direction="UP",
    entry_gate="north_key_door",
    entry_pose=(120, 205),
    keys_in=10,
    keys_out=9,
    bombs_in=7,
    bombs_out=7,
    room_item_id=0x03,
    census=((0x0C, 128, 6),),
    fixture="Level8Interior3EReconFixture",
    recording_tag="l8_4e_north_fixture_B1",
)

LEVEL8_INTERIOR_0X2E_RECON = Level8InteriorRoomRecon(
    room_id=0x2E,
    entered_from=0x3E,
    entry_direction="UP",
    entry_gate="north_bomb_wall",
    entry_pose=(120, 189),
    keys_in=9,
    keys_out=9,
    bombs_in=7,
    bombs_out=6,
    room_item_id=0x17,
    census=((0x3C, 64, 5),),
    fixture="Level8Interior2EReconFixture",
    recording_tag="l8_3e_north_fixture_20260904_A2",
)

LEVEL8_INTERIOR_0X1E_RECON = Level8InteriorRoomRecon(
    room_id=0x1E,
    entered_from=0x2E,
    entry_direction="UP",
    entry_gate="north_key_door",
    entry_pose=(120, 205),
    keys_in=9,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x03,
    census=((0x33, 96, 1),),
    fixture="Level8Interior1EReconFixture",
    recording_tag="l8_2e_north_fixture_20260904_C1",
)

LEVEL8_INTERIOR_0X1F_RECON = Level8InteriorRoomRecon(
    room_id=0x1F,
    entered_from=0x1E,
    entry_direction="RIGHT",
    entry_gate="east_kill_clear_shutter",
    entry_pose=(16, 141),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x03,
    census=(
        (0x16, 160, 2),
        (0x0C, 128, 2),
        (0x0B, 64, 2),
    ),
    fixture="Level8Interior1FReconFixture",
    recording_tag="l8_1e_gohma_fixture_20260904_D1b",
)

LEVEL8_INTERIOR_0X0F_RECON = Level8InteriorRoomRecon(
    room_id=0x0F,
    entered_from=0x1F,
    entry_direction="STAIRS",
    entry_gate="center_0x68_west_block_slide",
    entry_pose=(128, 141),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x0B,
    census=(),
    fixture="Level8InteriorMKReconFixture",
    recording_tag="l8_1f_magic_key_fixture_20260904_E2",
)

LEVEL8_INTERIOR_ROOM_RECON: tuple[Level8InteriorRoomRecon, ...] = (
    LEVEL8_INTERIOR_0X3E_RECON,
    LEVEL8_INTERIOR_0X2E_RECON,
    LEVEL8_INTERIOR_0X1E_RECON,
    LEVEL8_INTERIOR_0X1F_RECON,
    LEVEL8_INTERIOR_0X0F_RECON,
)

LEVEL8_INTERIOR_0X1E_WEST_RECON = Level8InteriorRoomRecon(
    room_id=0x1E,
    entered_from=0x1F,
    entry_direction="LEFT",
    entry_gate="west_open_door",
    entry_pose=(208, 141),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x03,
    census=(),
    fixture="Level8Interior1EWestReconFixture",
    recording_tag="l8_1f_west_fixture_20260904_G3",
)

LEVEL8_INTERIOR_0X2E_SOUTH_RECON = Level8InteriorRoomRecon(
    room_id=0x2E,
    entered_from=0x1E,
    entry_direction="DOWN",
    entry_gate="south_open_door",
    entry_pose=(120, 77),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x17,
    census=(),
    fixture="Level8Interior2ESouthReconFixture",
    recording_tag="l8_1e_south_fixture_20260904_H3",
)

LEVEL8_INTERIOR_0X3E_SOUTH_RECON = Level8InteriorRoomRecon(
    room_id=0x3E,
    entered_from=0x2E,
    entry_direction="DOWN",
    entry_gate="south_open_door",
    entry_pose=(120, 93),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x03,
    census=(),
    fixture="Level8Interior3ESouthReconFixture",
    recording_tag="l8_2e_south_fixture_20260904_I3",
)

LEVEL8_INTERIOR_0X3F_EAST_RECON = Level8InteriorRoomRecon(
    room_id=0x3F,
    entered_from=0x3E,
    entry_direction="RIGHT",
    entry_gate="east_open_shutter",
    entry_pose=(32, 141),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x00,
    census=(),
    fixture="Level8Interior3FEastReconFixture",
    recording_tag="l8_3e_east_fixture_20260904_J2",
)

LEVEL8_INTERIOR_0X2F_STAIRS_RECON = Level8InteriorRoomRecon(
    room_id=0x2F,
    entered_from=0x3F,
    entry_direction="STAIRS",
    entry_gate="tile_0x71_y141",
    entry_pose=(192, 93),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x00,
    census=(),
    fixture="Level8Interior2FCellarReconFixture",
    recording_tag="l8_2f_settle_20260904_S2",
)

LEVEL8_INTERIOR_0X4C_WEST_RECON = Level8InteriorRoomRecon(
    room_id=0x4C,
    entered_from=0x2F,
    entry_direction="STAIRS",
    entry_gate="west_ladder",
    entry_pose=(112, 125),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=6,
    room_item_id=0x19,
    census=(),
    fixture="Level8Interior4CWestReconFixture",
    recording_tag="l8_2f_cross_fixture_20260904_P2",
)

LEVEL8_INTERIOR_0X3C_NORTH_RECON = Level8InteriorRoomRecon(
    room_id=0x3C,
    entered_from=0x4C,
    entry_direction="UP",
    entry_gate="north_bomb_wall",
    entry_pose=(120, 189),
    keys_in=8,
    keys_out=8,
    bombs_in=6,
    bombs_out=5,
    room_item_id=0x1A,
    census=((0x45, 160, 1),),
    fixture="Level8Interior3CNorthReconFixture",
    recording_tag="l8_4c_north_fixture_20260904_N8",
)

LEVEL8_INTERIOR_0X3C_KILL_RECON = Level8InteriorRoomRecon(
    room_id=0x3C,
    entered_from=0x4C,
    entry_direction="UP",
    entry_gate="south_stand_0x45_heart",
    entry_pose=(32, 181),
    keys_in=8,
    keys_out=8,
    bombs_in=5,
    bombs_out=5,
    room_item_id=0x1A,
    census=(),
    fixture="Level8Interior3CKillReconFixture",
    recording_tag="l8_3c_gleeok_fixture_20260904_F7",
)

LEVEL8_INTERIOR_0X2C_TF_RECON = Level8InteriorRoomRecon(
    room_id=0x2C,
    entered_from=0x3C,
    entry_direction="UP",
    entry_gate="north_shutter",
    entry_pose=(120, 205),
    keys_in=8,
    keys_out=8,
    bombs_in=5,
    bombs_out=5,
    room_item_id=0x1B,
    census=(),
    fixture="Level8Interior2CTriforceReconFixture",
    recording_tag="l8_3c_north_fixture_20260904_T3",
)

__all__ = [
    "LEVEL8_INTERIOR_0X0F_RECON",
    "LEVEL8_INTERIOR_0X1E_RECON",
    "LEVEL8_INTERIOR_0X1E_WEST_RECON",
    "LEVEL8_INTERIOR_0X1F_RECON",
    "LEVEL8_INTERIOR_0X2C_TF_RECON",
    "LEVEL8_INTERIOR_0X2E_RECON",
    "LEVEL8_INTERIOR_0X2E_SOUTH_RECON",
    "LEVEL8_INTERIOR_0X2F_STAIRS_RECON",
    "LEVEL8_INTERIOR_0X3C_KILL_RECON",
    "LEVEL8_INTERIOR_0X3C_NORTH_RECON",
    "LEVEL8_INTERIOR_0X3E_RECON",
    "LEVEL8_INTERIOR_0X3E_SOUTH_RECON",
    "LEVEL8_INTERIOR_0X3F_EAST_RECON",
    "LEVEL8_INTERIOR_0X4C_WEST_RECON",
    "LEVEL8_INTERIOR_ROOM_RECON",
    "Level8InteriorRoomRecon",
]
