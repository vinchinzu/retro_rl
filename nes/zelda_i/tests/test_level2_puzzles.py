"""Unit tests for L2 puzzle catalog constants (no emulator)."""

from __future__ import annotations

from zelda_i import level2_puzzles as puz
from zelda_i.nav_common import DIAMOND_BAND_6E, DIAMOND_BAND_7D


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


def test_bomb_wall_5f_north_boom() -> None:
    """0x5f bomb-N @ (120,101) → Magical Boomerang room 0x4f."""
    bw = puz.BOMB_WALL_5F_NORTH
    assert bw.room == 0x5F
    assert bw.stand == (120, 101)
    assert bw.face == "UP"
    assert bw.opens_to == 0x4F
    assert bw.live is True
    assert puz.bomb_wall_for_room(0x5F, "UP") is bw
    assert puz.ROOM_L2_BOOM == 0x4F


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


def test_diamond_bands_match_nav_common() -> None:
    assert puz.DIAMOND_BAND_7D == DIAMOND_BAND_7D == 157
    assert puz.DIAMOND_BAND_6E == DIAMOND_BAND_6E == 113
    assert puz.diamond_band_for_room(0x7D) == 157
    assert puz.diamond_band_for_room(0x6E) == 113
    assert puz.DOOR_Y_MIN_OPEN == 137
    assert puz.DIAMOND_EAST_SEQUENCE == ("free", "band", "wall", "door_y", "push")


def test_negatives_and_sealed() -> None:
    faces = {f for f, _ in puz.BOMB_WALL_NEGATIVES_6F}
    assert faces >= {"UP", "RIGHT", "DOWN", "LEFT"}
    # Successful stand must not be listed as a negative miss.
    assert ( "UP", (120, 101) ) not in puz.BOMB_WALL_NEGATIVES_6F
    sealed = {(s.room, s.direction) for s in puz.SEALED_EXITS}
    assert (0x7D, "LEFT") in sealed
    assert (0x6C, "LEFT") in sealed
    assert (0x5F, "RIGHT") in sealed
    # 0x5f UP is bombable (BOMB_WALL_5F_NORTH) — not sealed-exit catalog.


def test_room_ids_align_level2_dungeon() -> None:
    """Catalog room IDs match level2_dungeon without importing heavy specs logic."""
    from zelda_i import level2_dungeon as d

    assert puz.ROOM_L2_COMPASS == d.ROOM_L2_COMPASS == 0x6F
    assert puz.ROOM_L2_BOMB_N == d.ROOM_L2_BOMB_N == 0x5F
    assert puz.ROOM_L2_GORIYA_WEST == d.ROOM_L2_GORIYA_WEST == 0x5E
    assert puz.ROOM_L2_EAST_OF_ROPES == d.ROOM_L2_EAST_OF_ROPES == 0x6E


def test_post_boss_tf_policy_live() -> None:
    """0x0d south-band TF maze is LIVE assisted (west of boss, not east)."""
    assert puz.ROOM_L2_BOSS == 0x0E
    assert puz.ROOM_L2_TF == 0x0D
    assert puz.LEVEL2_TRIFORCE_BIT == 0x02
    assert puz.L2_TF_COLLECT_WAYPOINTS == (
        (208, 141),
        (208, 189),
        (128, 189),
        (128, 149),
    )
    pol = puz.POST_BOSS_TF_POLICY
    assert pol.live is True
    assert pol.collect_xy == (128, 149)
    assert pol.waypoints == puz.L2_TF_COLLECT_WAYPOINTS
    assert pol.push_stand is None  # push not required for live collect
