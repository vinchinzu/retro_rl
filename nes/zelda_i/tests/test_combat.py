"""Unit tests for Zelda I sword hitbox / threat helpers (no emulator)."""

from __future__ import annotations

from zelda_i.combat import (
    CONTACT_CHEBYSHEV,
    FACING_EAST,
    FACING_NORTH,
    FACING_SOUTH,
    FACING_WEST,
    SWORD_HALF_WIDTH,
    SWORD_REACH,
    THREAT_RADIUS,
    direction_to_facing,
    facing_to_direction,
    in_sword_hitbox,
    manhattan,
    nearest_enemy,
    overworld_threat_objects,
    should_swing_at,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot


def _obj(
    slot: int = 1,
    *,
    type_id: int = 0x2A,
    x: int = 100,
    y: int = 100,
    hp: int = 0x20,
) -> ZeldaObject:
    return ZeldaObject(
        slot=slot,
        type_id=type_id,
        x=x,
        y=y,
        facing=FACING_SOUTH,
        hp=hp,
        state=0,
    )


def _snap(
    *,
    link_x: int = 120,
    link_y: int = 141,
    objects: tuple[ZeldaObject, ...] = (),
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=5,
        level=0,
        screen=0x77,
        next_screen=0x77,
        link_x=link_x,
        link_y=link_y,
        facing=FACING_EAST,
        sword=1,
        bombs=0,
        rupees=0,
        keys=0,
        health=0x2F,
        triforce=0,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=0,
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=objects,
    )


def test_direction_facing_roundtrip() -> None:
    assert direction_to_facing("UP") == FACING_NORTH
    assert direction_to_facing("down") == FACING_SOUTH
    assert direction_to_facing("RIGHT") == FACING_EAST
    assert direction_to_facing("LEFT") == FACING_WEST
    assert facing_to_direction(FACING_NORTH) == "UP"
    assert facing_to_direction(FACING_SOUTH) == "DOWN"
    assert facing_to_direction(FACING_EAST) == "RIGHT"
    assert facing_to_direction(FACING_WEST) == "LEFT"


def test_manhattan() -> None:
    assert manhattan(10, 20, 13, 17) == 6


def test_in_sword_hitbox_front_true_behind_false() -> None:
    lx, ly = 120, 141
    # North: enemy above Link within reach and width.
    assert in_sword_hitbox(lx, ly, "UP", lx, ly - 12)
    assert in_sword_hitbox(lx, ly, FACING_NORTH, lx + SWORD_HALF_WIDTH, ly - 8)
    assert not in_sword_hitbox(lx, ly, "UP", lx, ly + 12)  # behind
    assert not in_sword_hitbox(lx, ly, "UP", lx, ly - (SWORD_REACH + 5))  # far
    assert not in_sword_hitbox(
        lx, ly, "UP", lx + SWORD_HALF_WIDTH + 5, ly - 8
    )  # side

    # South / East / West
    assert in_sword_hitbox(lx, ly, "DOWN", lx, ly + 10)
    assert not in_sword_hitbox(lx, ly, "DOWN", lx, ly - 10)
    assert in_sword_hitbox(lx, ly, "RIGHT", lx + 15, ly)
    assert not in_sword_hitbox(lx, ly, "RIGHT", lx - 15, ly)
    assert in_sword_hitbox(lx, ly, "LEFT", lx - 15, ly)
    assert not in_sword_hitbox(lx, ly, "LEFT", lx + 15, ly)


def test_nearest_enemy() -> None:
    far = _obj(1, x=200, y=200)
    near = _obj(2, x=125, y=145)
    assert nearest_enemy(120, 141, (far, near)) is near
    assert nearest_enemy(120, 141, ()) is None


def test_should_swing_in_hitbox_or_contact_only() -> None:
    lx, ly = 120, 141
    # In front, in reach → swing
    front = _obj(1, x=lx + 12, y=ly)
    assert should_swing_at(lx, ly, "RIGHT", (front,))

    # Far in front of engage range but outside sword → no swing
    far = _obj(1, x=lx + 40, y=ly)
    assert not should_swing_at(lx, ly, "RIGHT", (far,))

    # Behind while facing right, outside contact band → no swing
    behind = _obj(1, x=lx - 30, y=ly)
    assert not should_swing_at(lx, ly, "RIGHT", (behind,))

    # Contact-close off-axis still swings (softlock guard)
    contact = _obj(1, x=lx + CONTACT_CHEBYSHEV, y=ly + CONTACT_CHEBYSHEV)
    assert should_swing_at(lx, ly, "UP", (contact,))

    # Empty list
    assert not should_swing_at(lx, ly, "UP", ())


def test_threat_radius_does_not_authorize_swing() -> None:
    """THREAT_RADIUS is for approach; should_swing stays hitbox/contact only."""
    lx, ly = 120, 141
    # Inside threat radius but outside sword + contact → no swing
    mid = _obj(1, x=lx + (THREAT_RADIUS - 5), y=ly)
    assert THREAT_RADIUS - 5 > SWORD_REACH
    assert not should_swing_at(lx, ly, "RIGHT", (mid,))


def test_overworld_threat_objects_filters_slots_and_bounds() -> None:
    good = _obj(1, type_id=0x07, x=100, y=100)
    slot0 = _obj(0, type_id=0x07, x=100, y=100)
    empty = _obj(2, type_id=0, x=100, y=100)
    oob_y = _obj(3, type_id=0x07, x=100, y=30)
    oob_x = _obj(4, type_id=0x07, x=4, y=100)
    snap = _snap(objects=(good, slot0, empty, oob_y, oob_x))
    threats = overworld_threat_objects(snap)
    assert threats == (good,)
