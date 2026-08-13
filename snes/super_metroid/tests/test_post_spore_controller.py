"""Unit tests for Super-collect / Pink PB helpers under routes.kpdr.

Historical ``post_spore_controller`` shim is deleted; import from
``super_metroid.routes.kpdr`` (or submodules) directly.
"""

from __future__ import annotations

from super_metroid.routes.kpdr import (
    MORPH_POSES,
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_PINK_PB,
    ROOM_SUPER,
    PowerBombEvidence,
    SuperCollectEvidence,
    ensure_morph,
    is_morph,
    play_big_pink_into_main_shaft,
    play_super_room_collect,
    wait_until,
)


def test_post_spore_rooms_and_morph_helpers() -> None:
    assert ROOM_SUPER == 0x9B5B
    assert ROOM_FARMING == 0xA0A4
    assert ROOM_BIG_PINK == 0x9D19
    assert ROOM_PINK_PB == 0x9E11
    assert is_morph(65)
    assert is_morph(31)
    assert is_morph(29)  # falling morph
    assert is_morph(30)
    assert not is_morph(40)  # crouch, not morph
    assert 65 in MORPH_POSES
    assert 29 in MORPH_POSES
    # Spot-check package still re-exports product entry points.
    assert callable(play_super_room_collect)
    assert callable(play_big_pink_into_main_shaft)
    assert callable(ensure_morph)
    assert callable(wait_until)


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
