"""Edge-case review of ``dungeon.tracking`` / ``dungeon.threat`` /
``dungeon.postmortem``: identity across room transitions, the projectile
tie-break leaking into the reported distance, and death with no captured
hit. Companion to ``test_reactive_combat.py``.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.postmortem import DamageLog
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_LEVEL,
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)

WIZZROBE_TYPE = 0x24
FACE_SOUTH = 0x04
FACE_WEST = 0x02


def _snap(
    link: tuple[int, int],
    objects: tuple = (),
    *,
    mode: int = PLAY_MODE,
    health: int = 0x22,
    partial: int = 0xFF,
    facing: int = FACE_SOUTH,
    screen: int = 0x1E,
    level: int = 8,
):
    ram = np.zeros(0x1000, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = link[0]
    ram[ADDR_LINK_Y] = link[1]
    ram[ADDR_LINK_FACING] = facing
    ram[ADDR_HEALTH] = health
    ram[ADDR_HEART_PARTIAL] = partial
    for slot, type_id, x, y, hp, state, obj_facing in objects:
        ram[ADDR_OBJ_TYPE + slot] = type_id
        ram[ADDR_LINK_X + slot] = x
        ram[ADDR_LINK_Y + slot] = y
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_OBJ_STATE + slot] = state
        ram[ADDR_LINK_FACING + slot] = obj_facing
    return read_snapshot(ram)


# --- tracking: room transitions ----------------------------------------


def test_room_transition_does_not_inherit_velocity() -> None:
    """A same-type object landing near the old room's last slot position
    must not read as a continuation of that room's motion.

    Both rooms place a 0x24 wizzrobe near the same waist row (a real
    pattern in this codebase: L6 0x78/0x7a both use y=141), so the
    (slot, type_id) + 32px-teleport identity alone is not enough — a
    room change has to hard-reset every track.
    """
    tracker = ObjectTracker()
    room_a = [
        _snap(
            (120, 141),
            ((1, WIZZROBE_TYPE, 150 - 2 * i, 141, 64, 0, FACE_WEST),),
            screen=0x78,
            level=6,
        )
        for i in range(4)
    ]
    for f in room_a:
        tracker.observe(f)

    # New room, same slot, same type, landing within RESPAWN_JUMP (32px) of
    # where the old room's object last sat — but it is a different object.
    room_b = _snap(
        (60, 90),
        ((1, WIZZROBE_TYPE, 148, 141, 64, 0, FACE_WEST),),
        screen=0x79,
        level=6,
    )
    tracked = tracker.observe(room_b)
    assert tracked[0].vx == 0.0
    assert tracked[0].vy == 0.0
    assert tracked[0].age == 1


def test_room_transition_resets_even_when_level_is_unchanged() -> None:
    """Same level, different screen: still a different room's object pool."""
    tracker = ObjectTracker()
    frames = [
        _snap(
            (120, 141),
            ((2, WIZZROBE_TYPE, 100 + 3 * i, 141, 64, 0, 0),),
            screen=0x1E,
            level=8,
        )
        for i in range(4)
    ]
    for f in frames:
        tracker.observe(f)
    next_room = _snap(
        (120, 141),
        ((2, WIZZROBE_TYPE, 110, 141, 64, 0, 0),),
        screen=0x1F,
        level=8,
    )
    tracked = tracker.observe(next_room)
    assert tracked[0].vx == 0.0
    assert tracked[0].age == 1


def test_same_room_identity_is_unaffected_by_the_room_guard() -> None:
    """Sanity: the room guard must not disturb ordinary same-room tracking."""
    tracker = ObjectTracker()
    frames = [
        _snap(
            (144, 141),
            ((1, WIZZROBE_TYPE, 200 - 4 * i, 141, 0, 0, FACE_WEST),),
            screen=0x78,
            level=6,
        )
        for i in range(4)
    ]
    tracked = ()
    for f in frames:
        tracked = tracker.observe(f)
    assert tracked[0].vx == -4.0
    assert tracked[0].age == 4


# --- postmortem: reported distance must be the true gap -----------------


def test_attributed_distance_is_not_the_projectile_tiebreak_fudge() -> None:
    """``_attribute`` nudges a tied projectile ahead of an idling body by 4
    px of *ranking* only. The reported ``distance`` must stay the real gap,
    not the post-nudge value used to pick the winner.
    """
    tracker = ObjectTracker()
    log = DamageLog()
    # Body at true (projected) distance 6, projectile at true distance 8 —
    # farther, but its rank of 8-4=4 beats the body's rank of 6, so the
    # projectile wins the tie-break. The reported ``distance`` must still
    # read 8, the real gap, not the 4 used only to pick the winner. The
    # projectile needs an established velocity (>= PROJECTILE_SPEED) before
    # the frame DamageLog uses as ``_prev``, so warm it up first.
    warmup = _snap(
        (120, 141),
        (
            (1, 0x23, 126, 141, 64, 0, 0),  # body, |dx|=6, stationary
            (2, 0x59, 136, 141, 0, 0, FACE_WEST),  # projectile, closing
        ),
        screen=0x78,
        level=6,
    )
    alive = _snap(
        (120, 141),
        (
            (1, 0x23, 126, 141, 64, 0, 0),
            (2, 0x59, 132, 141, 0, 0, FACE_WEST),  # vx=-4.0; +1f -> |dx|=8
        ),
        screen=0x78,
        level=6,
    )
    hurt = _snap(
        (120, 141),
        (
            (1, 0x23, 126, 141, 64, 0, 0),
            (2, 0x59, 128, 141, 0, 0, FACE_WEST),
        ),
        screen=0x78,
        level=6,
        health=0x21,
    )
    log.observe(warmup, tracker.observe(warmup))
    log.observe(alive, tracker.observe(alive))
    event = log.observe(hurt, tracker.observe(hurt))
    assert event is not None
    assert event.type_id == 0x59
    assert event.hazard == HazardClass.PROJECTILE.value
    # True (projected) gap from the previous frame, not (true - 4).
    assert event.distance == 8


# --- postmortem: death with no captured hit ------------------------------


def test_death_with_no_prior_hit_is_still_reported() -> None:
    """Mode 17 with an empty hit history must not silently drop the death.

    Health can already be pinned at the floor when a fresh ``DamageLog``
    starts watching (damage carried over from a previous room), so the
    killing blow never registers as a decrement. ``death`` should still
    exist so ``report()['death_cause']`` is not a bare ``None`` next to a
    real death.
    """
    log = DamageLog()
    tracker = ObjectTracker()
    dead = _snap((128, 181), (), health=0x20, partial=0x00, mode=17)
    event = log.observe(dead, tracker.observe(dead), action="idle", phase="FIGHT")
    assert event is None
    assert log.death is not None
    assert log.death.fatal
    report = log.report()
    assert report["death_cause"] is not None
    assert "unattributed" in report["death_cause"]
