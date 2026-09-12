"""Unit tests for Zelda I sword hitbox / threat helpers (no emulator)."""

from __future__ import annotations

from zelda_i.combat import (
    CONTACT_CHEBYSHEV,
    FACING_EAST,
    FACING_NORTH,
    FACING_SOUTH,
    FLOOR_DROP_TYPES,
    HEART_OR_FAIRY_TYPES,
    SWORD_HALF_WIDTH,
    SWORD_REACH,
    THREAT_RADIUS,
    floor_drops,
    in_sword_hitbox,
    is_floor_drop,
    is_heart_or_fairy_drop,
    nearest_enemy,
    nearest_floor_drop,
    overworld_threat_objects,
    should_swing_at,
    wants_heart_pickup,
)
from zelda_i.dungeon.ids import (
    GHINI_FLYING_OBJECT_TYPE,
    HEART_DROP_OBJECT_TYPE,
    HEART_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot


def _obj(
    slot: int = 1,
    *,
    type_id: int = 0x2A,
    x: int = 100,
    y: int = 100,
    hp: int = 0x20,
    state: int = 0,
) -> ZeldaObject:
    return ZeldaObject(
        slot=slot,
        type_id=type_id,
        x=x,
        y=y,
        facing=FACING_SOUTH,
        hp=hp,
        state=state,
    )


def _snap(
    *,
    link_x: int = 120,
    link_y: int = 141,
    health: int = 0x2F,
    objects: tuple[ZeldaObject, ...] = (),
    world_kill_count: int = 0,
    help_drop_count: int = 0,
    help_drop_value: int = 0,
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
        health=health,
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
        world_kill_count=world_kill_count,
        help_drop_count=help_drop_count,
        help_drop_value=help_drop_value,
    )


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


def test_should_swing_consumes_engagement_hint_veto() -> None:
    """Hint can veto; it cannot authorize a swing outside the hitbox."""
    from zelda_i.dungeon.behaviors import EngagementHint

    lx, ly = 120, 141
    front = _obj(1, x=lx + 12, y=ly)
    allow = EngagementHint(
        preferred_distance=48, face="RIGHT", swing=True, retreat=False
    )
    assert should_swing_at(lx, ly, "RIGHT", (front,), hint=allow)

    no_sword = EngagementHint(
        preferred_distance=48, face="RIGHT", swing=False, retreat=False
    )
    assert not should_swing_at(lx, ly, "RIGHT", (front,), hint=no_sword)

    retreat = EngagementHint(
        preferred_distance=48, face="RIGHT", swing=True, retreat=True
    )
    assert not should_swing_at(lx, ly, "RIGHT", (front,), hint=retreat)

    far = _obj(1, x=lx + 40, y=ly)
    want = EngagementHint(
        preferred_distance=48, face="RIGHT", swing=True, retreat=False
    )
    assert not should_swing_at(lx, ly, "RIGHT", (far,), hint=want)


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


def test_overworld_threat_objects_drops_rupee_and_hp_zero() -> None:
    live = _obj(1, type_id=0x07, x=100, y=100, hp=0x20)
    dead = _obj(2, type_id=0x07, x=110, y=100, hp=0)
    drop = _obj(3, type_id=0x60, x=120, y=100, hp=1)
    snap = _snap(objects=(live, dead, drop))
    assert overworld_threat_objects(snap) == (live,)


def test_overworld_threat_objects_skips_heart_drop_even_with_hp() -> None:
    live = _obj(1, type_id=0x07, x=100, y=100, hp=0x20)
    heart = _obj(
        2,
        type_id=HEART_DROP_OBJECT_TYPE,
        x=120,
        y=100,
        hp=1,
        state=HEART_DROP_STATE,
    )
    snap = _snap(objects=(live, heart))
    assert overworld_threat_objects(snap) == (live,)
    assert heart.type_id in FLOOR_DROP_TYPES


def test_floor_drops_finds_heart_ignores_living_ghini() -> None:
    """Heart is ObjType 0x60 + state 0x22; 0x22 as type is ghini_flying."""
    heart = _obj(
        1,
        type_id=HEART_DROP_OBJECT_TYPE,
        x=140,
        y=141,
        hp=0,
        state=HEART_DROP_STATE,
    )
    ghini = _obj(2, type_id=GHINI_FLYING_OBJECT_TYPE, x=160, y=141, hp=0x20)
    octorok = _obj(3, type_id=0x07, x=80, y=141, hp=0x10)
    snap = _snap(objects=(heart, ghini, octorok))
    assert is_floor_drop(heart)
    assert not is_floor_drop(ghini)
    assert not is_floor_drop(octorok)
    assert floor_drops(snap) == (heart,)
    assert ghini not in floor_drops(snap)
    assert is_heart_or_fairy_drop(heart)
    assert heart.type_id in HEART_OR_FAIRY_TYPES


def test_floor_drops_rupee_is_drop_not_heart() -> None:
    rupee = _obj(
        1,
        type_id=RUPEE_DROP_OBJECT_TYPE,
        x=140,
        y=141,
        hp=0,
        state=RUPEE_DROP_STATE,
    )
    snap = _snap(objects=(rupee,))
    assert is_floor_drop(rupee)
    assert not is_heart_or_fairy_drop(rupee)
    assert floor_drops(snap) == (rupee,)


def test_nearest_floor_drop() -> None:
    far = _obj(
        1,
        type_id=HEART_DROP_OBJECT_TYPE,
        x=200,
        y=141,
        hp=0,
        state=HEART_DROP_STATE,
    )
    near = _obj(
        2,
        type_id=RUPEE_DROP_OBJECT_TYPE,
        x=130,
        y=141,
        hp=0,
        state=RUPEE_DROP_STATE,
    )
    ghini = _obj(3, type_id=GHINI_FLYING_OBJECT_TYPE, x=125, y=141, hp=0x20)
    snap = _snap(link_x=120, link_y=141, objects=(far, near, ghini))
    assert nearest_floor_drop(snap) is near
    assert nearest_floor_drop(snap, types=FLOOR_DROP_TYPES) is near
    assert nearest_floor_drop(_snap(objects=())) is None


def test_wants_heart_pickup_two_of_three_vs_full() -> None:
    """2/3 (0x21) wants a heart; 3/3 (0x22, lo==hi) does not."""
    assert wants_heart_pickup(_snap(health=0x21)) is True
    assert wants_heart_pickup(_snap(health=0x22)) is False
    two_of_four = _snap(health=0x31)
    assert two_of_four.filled_hearts == 1
    assert two_of_four.heart_containers == 4
    assert wants_heart_pickup(two_of_four) is True
