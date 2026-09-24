"""Unit tests for L9 Patra/Ganon dodge + leftover-relative door-band. No emulator."""

from __future__ import annotations

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import (
    CONTACT_MANHATTAN,
    FACING_EAST,
    FACING_NORTH,
    FACING_WEST,
)
from zelda_i.dungeon.door_hop import door_band_goal
from zelda_i.dungeon.ids import FIREBALL_OBJECT_TYPE
from zelda_i.level9.dungeon import FULL_TRIFORCE
from zelda_i.level9.ganon import (
    DODGE_DIST,
    DODGE_X_HI,
    DODGE_X_LO,
    OBJ_GANON,
    ROOM_BEFORE_GANON,
    ROOM_GANON,
    ganon_action,
    ganon_contact,
    hazard_dodge_dir,
)
from zelda_i.level9.natural_path import NaturalPatraJoinController, PatraJoinPhase
from zelda_i.level9.path import NORTH_DOOR_GOAL, leftover_door_step
from zelda_i.level9.patra import OBJ_PATRA, OBJ_PATRA_EYE, patra_action
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot


def _obj(
    type_id: int,
    x: int,
    y: int,
    *,
    slot: int = 1,
    hp: int = 0xF0,
    state: int = 0,
) -> ZeldaObject:
    return ZeldaObject(
        slot=slot, type_id=type_id, x=x, y=y, facing=0, hp=hp, state=state
    )


def _snap(
    *,
    screen: int = ROOM_GANON,
    link_x: int = 120,
    link_y: int = 150,
    facing: int = FACING_NORTH,
    objects: tuple[ZeldaObject, ...] = (),
) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=PLAY_MODE,
        level=9,
        screen=screen,
        next_screen=screen,
        link_x=link_x,
        link_y=link_y,
        facing=facing,
        sword=2,
        bombs=8,
        rupees=0,
        keys=0,
        health=0xFF,
        triforce=FULL_TRIFORCE,
        compass=0,
        dialog_timer=0,
        colliding_tile=0,
        room_item_id=0,
        room_all_dead=0,
        room_obj_count=len(objects),
        cur_opened_doors=0,
        open_doorway_mask=0,
        objects=objects,
        bow=1,
        arrows=2,
    )


def test_hazard_dodge_fires_at_manhattan_thr_not_beyond() -> None:
    assert DODGE_DIST == CONTACT_MANHATTAN
    link = _snap(link_x=120, link_y=150)
    on_thr = (_obj(FIREBALL_OBJECT_TYPE, 127, 157, slot=2),)  # 7+7=14
    over = (_obj(FIREBALL_OBJECT_TYPE, 128, 157, slot=2),)  # 8+7=15
    assert hazard_dodge_dir(link, on_thr, thr=DODGE_DIST) == "LEFT"
    assert hazard_dodge_dir(link, over, thr=DODGE_DIST) is None


def test_hazard_dodge_flips_at_left_and_right_edges() -> None:
    left = _snap(link_x=DODGE_X_LO, link_y=150)
    right = _snap(link_x=DODGE_X_HI, link_y=150)
    ball_right = (_obj(FIREBALL_OBJECT_TYPE, DODGE_X_LO + 4, 150, slot=2),)
    ball_left = (_obj(FIREBALL_OBJECT_TYPE, DODGE_X_HI - 4, 150, slot=2),)
    assert hazard_dodge_dir(left, ball_right) == "RIGHT"
    assert hazard_dodge_dir(right, ball_left) == "LEFT"


def test_patra_cooldown_dodges_eye_at_boundary_not_idle() -> None:
    body = _obj(OBJ_PATRA, 120, 120, slot=1, hp=0xB0)
    eye = _obj(OBJ_PATRA_EYE, 127, 157, slot=2, hp=0x60)  # manhattan 14 from (120,150)
    snap = _snap(
        screen=ROOM_BEFORE_GANON,
        link_x=120,
        link_y=150,  # body.y+30
        facing=FACING_NORTH,
        objects=(body, eye),
    )
    action, reason, cooldown = patra_action(snap, cooldown=6, stand_dy=30)
    assert reason == "attack_dodge"
    assert list(action) == list(nes_action("LEFT"))
    assert cooldown == 5
    assert list(action) != list(nes_idle_action())


def test_patra_cooldown_without_hazard_stands_idle() -> None:
    body = _obj(OBJ_PATRA, 120, 120, slot=1, hp=0xB0)
    snap = _snap(
        screen=ROOM_BEFORE_GANON,
        link_x=120,
        link_y=150,
        facing=FACING_EAST,
        objects=(body,),
    )
    action, reason, cooldown = patra_action(snap, cooldown=3, stand_dy=30)
    assert reason == "cooldown_stand"
    assert list(action) == list(nes_idle_action())
    assert cooldown == 2


def test_patra_default_stand_is_the_bottom_row_outside_the_orbit() -> None:
    """30 px south stood inside the eyes' orbit (0x52 24h, 0x61 23h per
    fight); the default stand clamps to y=173 under the body."""
    body = _obj(OBJ_PATRA, 120, 120, slot=1, hp=0xB0)
    snap = _snap(
        screen=ROOM_BEFORE_GANON, link_x=120, link_y=150, facing=FACING_NORTH,
        objects=(body,),
    )
    action, reason, _ = patra_action(snap, cooldown=0)
    assert reason == "align_south"
    assert list(action) == list(nes_action("DOWN"))


def test_patra_faces_north_then_fires() -> None:
    body = _obj(OBJ_PATRA, 120, 120, slot=1, hp=0xB0)
    sideways = _snap(
        screen=ROOM_BEFORE_GANON,
        link_x=120,
        link_y=150,
        facing=FACING_EAST,
        objects=(body,),
    )
    face, face_reason, face_cd = patra_action(sideways, cooldown=0, stand_dy=30)
    assert face_reason == "face_up"
    assert list(face) == list(nes_action("UP"))
    assert face_cd == 0

    north = _snap(
        screen=ROOM_BEFORE_GANON,
        link_x=120,
        link_y=150,
        facing=FACING_NORTH,
        objects=(body,),
    )
    fire, fire_reason, fire_cd = patra_action(north, cooldown=0, stand_dy=30)
    assert fire_reason == "sword_pulse_up"
    assert list(fire) == list(nes_action("UP", "A"))
    assert fire_cd > 0


def test_ganon_blade_from_below_window_not_inside_sprite() -> None:
    # Link 26 px under Ganon's corner, on his column: the measured UP window.
    boss = _obj(OBJ_GANON, 112, 125, slot=1, hp=0xF0, state=0)
    north = _snap(link_x=120, link_y=151, facing=FACING_NORTH, objects=(boss,))
    fire, reason, cooldown = ganon_action(north, cooldown=0)
    assert reason == "sword_pulse"
    assert list(fire) == list(nes_action("A"))
    assert cooldown == 12
    hold, reason, cooldown = ganon_action(north, cooldown=8)
    assert reason == "cooldown_stand"
    assert list(hold) == list(nes_idle_action())
    assert cooldown == 7


def test_ganon_turns_only_from_inside_the_stand() -> None:
    boss = _obj(OBJ_GANON, 112, 125, slot=1, hp=0xF0, state=0)
    sideways = _snap(link_x=120, link_y=151, facing=FACING_EAST, objects=(boss,))
    face, reason, _ = ganon_action(sideways, cooldown=0)
    assert reason == "face_sword"
    assert list(face) == list(nes_action("UP"))
    # dy 21 is inside the window but not its stand: the turn would walk out.
    edge = _snap(link_x=120, link_y=146, facing=FACING_EAST, objects=(boss,))
    _, reason, _ = ganon_action(edge, cooldown=0)
    assert reason != "face_sword"


def test_ganon_steps_out_of_the_sprite() -> None:
    # The old chase stood here: 8 px into the 32 px sprite, hit every iframe.
    boss = _obj(OBJ_GANON, 112, 125, slot=1, hp=0xF0, state=0)
    inside = _snap(link_x=120, link_y=133, facing=FACING_NORTH, objects=(boss,))
    assert ganon_contact(inside, boss)
    _, reason, _ = ganon_action(inside, cooldown=0)
    assert reason == "leave_ganon_body"


def test_ganon_silver_arrow_from_a_lane_outside_the_sprite() -> None:
    boss = _obj(OBJ_GANON, 64, 133, slot=1, hp=0xF0, state=0xF0)
    lane = _snap(link_x=120, link_y=141, facing=FACING_WEST, objects=(boss,))
    fire, reason, cooldown = ganon_action(lane, cooldown=0)
    assert reason == "silver_arrow"
    assert list(fire) == list(nes_action("B"))
    assert cooldown == 16
    # dy -3: the arrow flies over him (Blue Ring power-on 10).
    over = _snap(link_x=120, link_y=130, facing=FACING_WEST, objects=(boss,))
    _, reason, _ = ganon_action(over, cooldown=0)
    assert reason != "silver_arrow"


def test_dying_ganon_is_walked_onto_not_shot() -> None:
    # The killing arrow drops a brown Ganon's HP below 240; the defeat flag
    # waits for Link on the remains.
    boss = _obj(OBJ_GANON, 64, 141, slot=1, hp=176, state=0xF4)
    lane = _snap(link_x=120, link_y=141, facing=FACING_WEST, objects=(boss,))
    _, reason, _ = ganon_action(lane, cooldown=0)
    assert reason == "to_ganon_remains"


def test_leftover_door_band_off_column_uses_door_x() -> None:
    assert door_band_goal("UP", (208, 157), NORTH_DOOR_GOAL)[0] == NORTH_DOOR_GOAL[0]
    assert door_band_goal("UP", (118, 157), NORTH_DOOR_GOAL)[0] == 118
    off = _snap(screen=ROOM_BEFORE_GANON, link_x=208, link_y=157)
    step = leftover_door_step(off, (208, 157), "UP", NORTH_DOOR_GOAL, reason="ganon")
    assert list(step.action) == list(nes_action("LEFT"))
    in_band = _snap(screen=ROOM_BEFORE_GANON, link_x=118, link_y=157)
    hold = leftover_door_step(
        in_band, (118, 157), "UP", NORTH_DOOR_GOAL, reason="ganon"
    )
    assert list(hold.action) == list(nes_action("UP"))


def test_north_41_uses_leftover_column_not_frozen_spawn() -> None:
    ctl = NaturalPatraJoinController()
    ctl.start_checked = True
    ctl.phase = PatraJoinPhase.NORTH_41
    in_band = _snap(screen=0x41, link_x=118, link_y=189)
    first = ctl.step(in_band)
    assert ctl.phase_leftover == (118, 189)
    assert list(first.action) == list(nes_action("UP"))

    ctl2 = NaturalPatraJoinController()
    ctl2.start_checked = True
    ctl2.phase = PatraJoinPhase.NORTH_41
    off = _snap(screen=0x41, link_x=208, link_y=141)
    align = ctl2.step(off)
    assert ctl2.phase_leftover == (208, 141)
    assert list(align.action) == list(nes_action("LEFT"))
