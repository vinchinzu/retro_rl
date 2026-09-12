from __future__ import annotations

import numpy as np

from zelda_i.combat import in_sword_hitbox, overworld_threat_objects, should_swing_at
from zelda_i.overworld.common import (
    EDGE_WEST_X,
    KNOCKBACK_STUCK_PENALTY,
    answer_projectile,
    overworld_projectiles,
    scoop_floor_drop,
    scoop_toward_drop,
    swing_action,
    track_knockback,
    walk_or_swing,
)
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops
from retro_harness.nes import nes_action
from zelda_i.dungeon.behaviors import FIREBALL_TYPE
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_MAGIC_SHIELD,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    ZeldaObject,
    read_snapshot,
)


def _snap(
    *,
    x: int = 120,
    y: int = 140,
    screen: int = 0x37,
    mode: int = PLAY_MODE,
    obj: tuple[int, int, int, int] | None = None,
):
    """Optional ``obj`` is (slot, type_id, ox, oy)."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    if obj is not None:
        slot, type_id, ox, oy = obj
        ram[ADDR_OBJ_TYPE + slot] = type_id
        ram[ADDR_LINK_X + slot] = ox
        ram[ADDR_LINK_Y + slot] = oy
        ram[ADDR_OBJ_HP + slot] = 0x20
    return read_snapshot(ram)


def test_path_screens_from_hops() -> None:
    hops = (
        ScreenHop(0x38, "RIGHT", align_y=140),
        ScreenHop(0x48, "DOWN", align_x=120),
    )
    assert path_screens_from_hops(0x37, hops) == (0x37, 0x38, 0x48)


def test_walk_or_swing_no_enemies_no_slash() -> None:
    snap = _snap(x=120, y=140)
    # Pulse frame would slash under bare swing_action; empty screen stays walk.
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action != swing_action(0, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_enemy_in_front_slashes() -> None:
    # Enemy just to the right of Link → in sword hitbox for RIGHT.
    snap = _snap(x=120, y=140, obj=(1, 0x06, 135, 140))
    threats = overworld_threat_objects(snap)
    assert in_sword_hitbox(120, 140, "RIGHT", 135, 140)
    assert should_swing_at(120, 140, "RIGHT", threats)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    walk = walk_or_swing(3, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert walk.reason == "walk"


def test_walk_or_swing_faces_contact_side_enemy() -> None:
    """Contact-range north of Link: face UP this frame. Far side hitbox must not."""
    snap = _snap(x=120, y=140, obj=(1, 0x07, 120, 132))
    assert in_sword_hitbox(120, 140, "UP", 120, 132)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert slash.action == swing_action(0, "UP", "walk", period=10, hold=3).action


def test_walk_or_swing_far_side_hitbox_keeps_travel() -> None:
    """16px north is in the UP blade but not contact — keep the hop RIGHT."""
    snap = _snap(x=120, y=140, obj=(1, 0x07, 120, 124))
    assert in_sword_hitbox(120, 140, "UP", 120, 124)
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action == swing_action(3, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_front_slash_not_vetoed_by_nearer_side_enemy() -> None:
    """Nearest off-axis enemy must not cancel a travel-direction hitbox slash."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = 0x37
    ram[ADDR_LINK_X] = 120
    ram[ADDR_LINK_Y] = 140
    ram[ADDR_OBJ_TYPE + 1] = 0x07
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 100
    ram[ADDR_OBJ_HP + 1] = 0x20
    ram[ADDR_OBJ_TYPE + 2] = 0x07
    ram[ADDR_LINK_X + 2] = 135
    ram[ADDR_LINK_Y + 2] = 140
    ram[ADDR_OBJ_HP + 2] = 0x20
    snap = read_snapshot(ram)
    slash = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert slash.reason == "walk_slash"
    assert slash.action == swing_action(0, "RIGHT", "walk", period=10, hold=3).action


def test_walk_or_swing_does_not_slash_far_behind_enemy() -> None:
    snap = _snap(x=120, y=140, obj=(1, 0x07, 80, 140))
    assert not in_sword_hitbox(120, 140, "RIGHT", 80, 140)
    assert not in_sword_hitbox(120, 140, "LEFT", 80, 140)
    act = walk_or_swing(0, "RIGHT", "walk", snap, period=10, hold=3)
    assert act.reason == "walk"
    assert act.action == swing_action(3, "RIGHT", "walk", period=10, hold=3).action


def _shot_snap(
    *,
    x: int,
    y: int,
    ox: int,
    oy: int,
    type_id: int = 0x53,
    magic_shield: int = 0,
):
    """Link plus one projectile slot (Octorok rock by default, hp stays 0)."""
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MAGIC_SHIELD] = magic_shield
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = 0x37
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_OBJ_TYPE + 1] = type_id
    ram[ADDR_LINK_X + 1] = ox
    ram[ADDR_LINK_Y + 1] = oy
    return read_snapshot(ram)


def test_overworld_projectiles_survive_the_hp_filter() -> None:
    """Shots carry hp=0, so ``overworld_threat_objects`` cannot see them."""
    snap = _shot_snap(x=120, y=140, ox=150, oy=140)
    assert overworld_threat_objects(snap) == ()
    assert len(overworld_projectiles(snap)) == 1


def test_walk_or_swing_shields_a_blockable_shot_without_pressing_a() -> None:
    """Octorok rock on Link's row: keep walking into it, but drop the A pulse."""
    snap = _shot_snap(x=120, y=140, ox=144, oy=140)
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert act.reason == "hop0_shield"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_walk_or_swing_dodges_a_fireball_without_the_magic_shield() -> None:
    """Fireball 0x1e is not blockable bare: leave the lane."""
    snap = _shot_snap(x=120, y=140, ox=144, oy=148, type_id=FIREBALL_TYPE)
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert act.reason == "hop0_dodge"
    assert list(act.action) == list(nes_action("UP"))


def test_magic_shield_blocks_the_fireball_instead() -> None:
    snap = _shot_snap(
        x=120, y=140, ox=144, oy=148, type_id=FIREBALL_TYPE, magic_shield=1
    )
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert act.reason == "hop0_shield"


def test_dodge_flips_side_at_the_screen_edge() -> None:
    """Shot below would send Link north, but y=66 is the north wall."""
    snap = _shot_snap(x=120, y=66, ox=144, oy=74, type_id=FIREBALL_TYPE)
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert list(act.action) == list(nes_action("DOWN"))


def test_answer_projectile_is_none_for_an_empty_lane() -> None:
    assert answer_projectile(120, 140, "RIGHT", (), "hop0") is None


def test_shot_outside_the_travel_lane_does_not_dodge() -> None:
    snap = _shot_snap(x=120, y=140, ox=144, oy=190)
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert act.reason == "hop0"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_shot_behind_link_does_not_dodge() -> None:
    """Only the approach band matters; a passed shot is not a threat."""
    snap = _shot_snap(x=120, y=140, ox=80, oy=140)
    act = walk_or_swing(0, "RIGHT", "hop0", snap, period=10, hold=3)
    assert act.reason == "hop0"


def test_track_knockback_charges_stuck_on_damage() -> None:
    hits, health, stuck = track_knockback(
        _snap(), last_health=-1, hits=0, stuck=0
    )
    assert (hits, stuck) == (0, 0)  # first read only seeds the baseline
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_HEALTH] = 0x25
    hurt = read_snapshot(ram)
    hits, health, stuck = track_knockback(
        hurt, last_health=0x35, hits=0, stuck=0
    )
    assert hits == 1
    assert health == 0x25
    assert stuck == KNOCKBACK_STUCK_PENALTY


def test_track_knockback_ignores_healing() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_HEALTH] = 0x45
    healed = read_snapshot(ram)
    hits, health, stuck = track_knockback(
        healed, last_health=0x25, hits=2, stuck=7
    )
    assert (hits, health, stuck) == (2, 0x45, 7)


def test_knockback_loop_reaches_the_unstick_ladder() -> None:
    """Three hits inside one hop trip the 50-frame stuck bar (rr Phase 4.5)."""
    stuck = 0
    for _ in range(3):
        ram = np.zeros(0x800, dtype=np.uint8)
        ram[ADDR_MODE] = 8
        ram[ADDR_HEALTH] = 0x25
        _, _, stuck = track_knockback(
            read_snapshot(ram), last_health=0x35, hits=0, stuck=stuck
        )
    assert stuck > 50


_HEART_TYPE = 0xFE


def _heart(*, x: int, y: int, slot: int = 1) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=_HEART_TYPE, x=x, y=y, facing=0, hp=0, state=0
    )


def test_scoop_toward_drop_walks_cardinal_to_a_nearby_heart() -> None:
    snap = _snap(x=100, y=140)
    act = scoop_toward_drop(
        snap,
        _heart(x=120, y=140),
        reason="scoop_heart",
        travel_dir="RIGHT",
        radius=48,
    )
    assert act is not None
    assert act.reason == "scoop_heart"
    assert list(act.action) == list(nes_action("RIGHT"))


def test_scoop_toward_drop_refuses_opposite_edge() -> None:
    snap = _snap(x=40, y=140)
    act = scoop_toward_drop(
        snap,
        _heart(x=EDGE_WEST_X, y=140),
        reason="scoop_heart",
        travel_dir="RIGHT",
        radius=48,
    )
    assert act is None


def test_scoop_toward_drop_noops_when_obj_is_none_or_far() -> None:
    snap = _snap(x=100, y=140)
    assert (
        scoop_toward_drop(
            snap, None, reason="scoop_heart", travel_dir="RIGHT", radius=48
        )
        is None
    )
    assert (
        scoop_toward_drop(
            snap,
            _heart(x=220, y=200),
            reason="scoop_heart",
            travel_dir="RIGHT",
            radius=48,
        )
        is None
    )


def test_scoop_floor_drop_noops_when_want_is_false() -> None:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_SCREEN] = 0x37
    ram[ADDR_LINK_X] = 100
    ram[ADDR_LINK_Y] = 140
    ram[ADDR_OBJ_TYPE + 1] = _HEART_TYPE
    ram[ADDR_LINK_X + 1] = 120
    ram[ADDR_LINK_Y + 1] = 140
    snap = read_snapshot(ram)
    nearby = scoop_floor_drop(
        snap,
        types={_HEART_TYPE},
        travel_dir="RIGHT",
        radius=48,
        reason="scoop_heart",
        want=True,
    )
    assert nearby is not None
    assert nearby.reason == "scoop_heart"
    assert (
        scoop_floor_drop(
            snap,
            types={_HEART_TYPE},
            travel_dir="RIGHT",
            radius=48,
            reason="scoop_heart",
            want=False,
        )
        is None
    )
