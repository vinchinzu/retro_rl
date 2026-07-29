"""Unit tests for post-Spore Super controller helpers (no emulator)."""

from __future__ import annotations

from super_metroid.routes.post_spore_controller import (
    MORPH_POSES,
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_PINK_PB,
    ROOM_SUPER,
    PowerBombEvidence,
    SuperCollectEvidence,
    bomb_roll_left_safe,
    ensure_morph,
    is_morph,
    play_big_pink_bomb_to_walkway_edge,
    play_big_pink_clear_super_block,
    play_big_pink_crest_pocket,
    play_big_pink_drop_to_pocket,
    play_big_pink_enter_pb_door_from_sill,
    play_big_pink_enter_pb_door_from_top_ledge,
    play_big_pink_into_main_shaft,
    play_big_pink_morph_to_tunnel,
    play_big_pink_tunnel_west,
    play_farming_to_big_pink,
    play_pink_pb_break_maze_wall,
    play_pink_pb_from_left_zone,
    play_pink_pb_mid_maze_to_collect,
    play_pink_pb_morph_bomb_collect,
    play_super_room_collect,
    play_super_room_to_farming,
    wait_until,
)


def test_room_constants() -> None:
    assert ROOM_SUPER == 0x9B5B
    assert ROOM_FARMING == 0xA0A4
    assert ROOM_BIG_PINK == 0x9D19
    assert ROOM_PINK_PB == 0x9E11


def test_controller_exports() -> None:
    assert callable(play_super_room_collect)
    assert callable(play_super_room_to_farming)
    assert callable(play_farming_to_big_pink)
    assert callable(play_big_pink_crest_pocket)
    assert callable(play_big_pink_clear_super_block)
    assert callable(play_big_pink_morph_to_tunnel)
    assert callable(play_big_pink_tunnel_west)
    assert callable(play_big_pink_drop_to_pocket)
    assert callable(play_big_pink_bomb_to_walkway_edge)
    assert callable(play_big_pink_into_main_shaft)
    assert callable(play_big_pink_enter_pb_door_from_sill)
    assert callable(play_big_pink_enter_pb_door_from_top_ledge)
    assert callable(play_pink_pb_break_maze_wall)
    assert callable(play_pink_pb_morph_bomb_collect)
    assert callable(play_pink_pb_mid_maze_to_collect)
    assert callable(play_pink_pb_from_left_zone)
    assert callable(ensure_morph)
    assert callable(bomb_roll_left_safe)
    assert callable(wait_until)
    assert is_morph(65)
    assert is_morph(31)
    assert not is_morph(40)  # crouch, not morph
    assert 65 in MORPH_POSES


def test_super_collect_evidence_dict() -> None:
    evidence = SuperCollectEvidence(
        entry_frame=10,
        collect_frame=100,
        exit_frame=200,
        max_super_missiles=5,
        final_room_id=ROOM_BIG_PINK,
        samus_x=100,
        samus_y=200,
    )
    payload = evidence.to_dict()
    assert payload["max_super_missiles"] == 5
    assert payload["final_room_id"] == ROOM_BIG_PINK
    assert payload["exit_frame"] == 200


def test_power_bomb_evidence_dict() -> None:
    evidence = PowerBombEvidence(
        entry_frame=0,
        collect_frame=None,
        max_super_missiles=5,
        max_power_bombs=0,
        final_room_id=ROOM_BIG_PINK,
        samus_x=1,
        samus_y=2,
        reached_big_pink=True,
        reached_pb_room=False,
    )
    payload = evidence.to_dict()
    assert payload["reached_big_pink"] is True
    assert payload["reached_pb_room"] is False
    assert payload["collect_frame"] is None
