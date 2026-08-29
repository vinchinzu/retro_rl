"""Unit tests for L2 puzzle catalog constants (no emulator)."""

from __future__ import annotations

from zelda_i.level2 import puzzles as puz


def test_bomb_wall_6f_north_stand() -> None:
    """Documented live stand is (120, 101) facing UP → 0x5f."""
    bw = puz.BOMB_WALL_6F_NORTH
    assert bw.room == 0x6F
    assert bw.stand == (120, 101)
    assert bw.face == "UP"
    assert bw.opens_to == 0x5F
    assert bw.live is True
    assert puz.bomb_wall_for_room(0x6F, "UP") is bw
    assert puz.is_at_bomb_stand(120, 101, bw)
    assert puz.is_at_bomb_stand(118, 103, bw, tol=6)
    assert not puz.is_at_bomb_stand(120, 141, bw, tol=6)


def test_bomb_wall_open_predicate() -> None:
    assert puz.bomb_wall_open_predicate(from_room=0x6F, to_room=0x5F)
    assert puz.bomb_wall_open_predicate(from_room=0x5F, to_room=0x4F)
    assert not puz.bomb_wall_open_predicate(from_room=0x6F, to_room=0x6E)
    assert not puz.bomb_wall_open_predicate(from_room=0x5F, to_room=0x5E)


def test_key_doors_live() -> None:
    k6e = puz.KEY_DOOR_6E_RIGHT
    assert k6e.room == 0x6E
    assert k6e.direction == "RIGHT"
    assert k6e.destination == 0x6F
    assert k6e.key_cost == 1
    assert k6e.approach_band_y == puz.DIAMOND_BAND_6E
    assert k6e.live is True

    k5f = puz.KEY_DOOR_5F_LEFT
    assert k5f.room == 0x5F
    assert k5f.direction == "LEFT"
    assert k5f.destination == 0x5E
    assert k5f.key_cost == 1
    assert k5f.live is True

    assert puz.key_door_for(0x6E, "RIGHT") is k6e
    assert puz.key_door_for(0x5F, "LEFT") is k5f
    assert set(puz.KEY_DOORS_LIVE) == {k6e, k5f}


def test_key_door_open_predicate() -> None:
    assert puz.key_door_open_predicate(
        from_room=0x6E,
        to_room=0x6F,
        keys_before=2,
        keys_after=1,
        door=puz.KEY_DOOR_6E_RIGHT,
    )
    assert puz.key_door_open_predicate(
        from_room=0x5F,
        to_room=0x5E,
        keys_before=4,
        keys_after=3,
        door=puz.KEY_DOOR_5F_LEFT,
    )
    # Wrong key delta
    assert not puz.key_door_open_predicate(
        from_room=0x6E,
        to_room=0x6F,
        keys_before=2,
        keys_after=2,
        door=puz.KEY_DOOR_6E_RIGHT,
    )
    # Residual RIGHT has destination 0 → never open
    assert not puz.key_door_open_predicate(
        from_room=0x5F,
        to_room=0x5E,
        keys_before=1,
        keys_after=0,
        door=puz.KEY_DOOR_5F_RIGHT_RESIDUAL,
    )
