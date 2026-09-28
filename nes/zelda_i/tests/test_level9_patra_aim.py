"""Patra eye aim (rr-e59v): fire only on a predicted hit, stand on the orbit's lane.

Run 23's last eye orbited a point 50-130 px below the body and was inside
the room 6% of its 5674 frames; the body-lane stand pulsed A ~800 times for
24 eye hits, and every miss held the one sword shot for ~76 frames.
"""

from __future__ import annotations

import math

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.combat import FACING_EAST, FACING_NORTH, FACING_WEST
from zelda_i.level9.ganon import ROOM_BEFORE_GANON
from zelda_i.level9.patra import (
    AIM_EYE_CLEAR,
    BEAM_LEAD,
    EYE_FIT_FRAMES,
    OBJ_PATRA,
    OBJ_PATRA_EYE,
    PatraAim,
    PatraEyeModel,
    _lane_windows,
    beam_hit_frame,
)
from zelda_i.ram import PLAY_MODE, SWORD_SHOT_SLOT, ZeldaObject, ZeldaSnapshot

PERIOD = 73


def _obj(type_id: int, x: int, y: int, *, slot: int, hp: int) -> ZeldaObject:
    return ZeldaObject(slot=slot, type_id=type_id, x=x & 0xFF, y=y & 0xFF, facing=0, hp=hp, state=0)


def _snap(link: tuple[int, int], facing: int, *objects: ZeldaObject) -> ZeldaSnapshot:
    return ZeldaSnapshot(
        mode=PLAY_MODE, level=9, screen=ROOM_BEFORE_GANON, next_screen=ROOM_BEFORE_GANON,
        link_x=link[0], link_y=link[1], facing=facing, sword=2, bombs=8, rupees=0, keys=0,
        health=0xFF, triforce=0xFF, compass=0, dialog_timer=0, colliding_tile=0,
        room_item_id=0, room_all_dead=0, room_obj_count=len(objects), cur_opened_doors=0,
        open_doorway_mask=0, objects=objects, bow=1, arrows=2,
        sword_shot=ZeldaObject(slot=SWORD_SHOT_SLOT, type_id=0, x=0, y=0, facing=0, hp=0, state=0),
    )


def _orbit(frame: int, body: tuple[int, int], center: tuple[int, int], r: int) -> tuple[int, int]:
    """Eye on a circle about ``body + center`` (the drifted orbit run 23 measured)."""
    a = 2 * math.pi * frame / PERIOD
    return (
        round(body[0] + center[0] + r * math.cos(a)),
        round(body[1] + center[1] + r * math.sin(a)),
    )


def _feed(model: PatraEyeModel, frames: range, body, center, r, link=(40, 173)):
    snap = None
    for f in frames:
        ex, ey = _orbit(f, body, center, r)
        snap = _snap(
            link, FACING_EAST,
            _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
            _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x20),
        )
        model.observe(snap)
    return snap


def test_eye_model_unwraps_an_orbit_across_the_screen_bottom() -> None:
    # Center 110 px below a body at y=150: the eye's y runs 214..255, 0..50.
    model = PatraEyeModel()
    _feed(model, range(EYE_FIT_FRAMES + 40), (120, 150), (0, 110), 46)
    k = 20
    fx, fy = model.predict(k)[2]
    ex, ey = _orbit(model.frame - 1 + k, (120, 150), (0, 110), 46)  # frame 1 = f0
    assert abs(fx - ex) <= 2
    assert abs((fy - ey + 128) % 256 - 128) <= 2


def test_beam_hit_frame_is_the_first_frame_the_shot_box_meets_the_eye() -> None:
    # A still eye 43 px right of Link on his row: the shot spawns at +19 on
    # frame 4 (dx 24) and closes 3 px/f; the box's near edge (dx <= 18) is frame 6.
    eye = {2: (lambda k: (83, 173))}
    paths = {slot: [fn(k) for k in range(41)] for slot, fn in eye.items()}
    assert beam_hit_frame((40, 173), "RIGHT", paths) == BEAM_LEAD + 2
    assert beam_hit_frame((40, 173), "LEFT", paths) is None
    assert beam_hit_frame((40, 120), "RIGHT", paths) is None


def test_aim_holds_fire_while_the_orbit_is_below_the_room() -> None:
    # The old body-lane stand pulsed A here: the shot flew the room and the
    # eye was never in it.
    aim = PatraAim()
    body, center = (120, 150), (0, 130)
    reasons = set()
    for f in range(EYE_FIT_FRAMES + PERIOD):
        ex, ey = _orbit(f, body, center, 46)
        snap = _snap(
            (40, 173), FACING_EAST,
            _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
            _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x20),
        )
        _, reason = aim.step(snap)
        reasons.add(reason)
    assert not any(r.startswith("sword_pulse") for r in reasons), reasons


def test_aim_fires_when_the_orbit_top_crosses_the_bottom_row() -> None:
    # Orbit top at y=164: the eye runs along the bottom rows for ~20 frames.
    aim = PatraAim()
    body, center, r = (104, 165), (0, 45), 46
    fired = []
    for f in range(EYE_FIT_FRAMES + 2 * PERIOD):
        ex, ey = _orbit(f, body, center, r)
        snap = _snap(
            (40, 173), FACING_EAST,
            _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
            _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x20),
        )
        action, reason = aim.step(snap)
        if reason.startswith("sword_pulse"):
            fired.append(f)
            assert list(action) == list(nes_action("A"))
    assert fired, "never fired at an eye crossing Link's row"
    # Every shot is a predicted hit: the eye is on the row when it lands.
    for f in fired:
        k = aim.last_fire_hit_frame
        assert k is not None and BEAM_LEAD <= k <= 40


def test_stand_search_picks_a_safe_lane_the_lap_crosses() -> None:
    # Lap top at y~164: only the bottom rows see the eye.
    model = PatraEyeModel()
    body = (104, 150)
    snap = _feed(model, range(EYE_FIT_FRAMES + 8), body, (0, 60), 46, link=(120, 101))
    aim = PatraAim(model=model)
    stand = aim.choose_stand(snap)
    assert stand is not None
    (sx, sy), facing = stand
    assert sx % 8 == 0 and sy % 8 == 5  # a turn-lattice node
    lap = np.concatenate(list(model.paths(np.arange(PERIOD), body_motion=False).values()))
    count, _ = _lane_windows(np.array([(sx, sy)], dtype=float), facing, lap)
    assert count[0] > 0
    assert np.abs(lap - (sx, sy)).max(axis=1).min() >= AIM_EYE_CLEAR


def test_stand_is_kept_while_it_still_sees_the_lap() -> None:
    # Re-planning every 8 frames must not flip between two near-equal lanes.
    model = PatraEyeModel()
    body = (104, 150)
    aim = PatraAim(model=model)
    _feed(model, range(EYE_FIT_FRAMES), body, (0, 60), 46, link=(40, 173))
    stands = set()
    for f in range(EYE_FIT_FRAMES, EYE_FIT_FRAMES + 3 * PERIOD, 8):
        snap = _feed(model, range(f, f + 8), body, (0, 60), 46, link=(40, 173))
        aim.stand = aim.choose_stand(snap)
        stands.add(aim.stand)
    assert len(stands) == 1, stands


def test_aim_falls_back_to_the_body_lane_once_the_eyes_are_gone() -> None:
    aim = PatraAim()
    body = _obj(OBJ_PATRA, 120, 93, slot=1, hp=0x70)
    _, reason = aim.step(_snap((120, 157), FACING_NORTH, body))
    assert reason == "sword_pulse_up"


def test_turning_on_the_stand_does_not_restart_the_walk() -> None:
    # Run 23: stand x=64, arrive tolerance 4; Link stopped at 68, the RIGHT
    # turn put him at 69, and the walk pressed LEFT again -- 270 frames.

    aim = PatraAim()
    body, center = (140, 120), (0, 130)
    for f in range(EYE_FIT_FRAMES):
        ex, ey = _orbit(f, body, center, 46)
        aim.model.observe(_snap((64, 173), FACING_WEST, _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
                                _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x20)))
    aim.stand, aim.replan_in, aim.arrived = ((64, 173), "RIGHT"), 99, True
    ex, ey = _orbit(EYE_FIT_FRAMES, body, center, 46)
    snap = _snap((69, 173), FACING_EAST, _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
                 _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x20))
    _, reason = aim.step(snap)
    assert not reason.startswith("aim_align"), reason


def test_a_lap_centered_on_the_body_keeps_the_lane_stand() -> None:
    # Healthy pins: the eyes ring the body and the blind lane pulse crosses
    # several of them; the aim cost ~430 frames there.
    aim = PatraAim()
    body = (120, 110)
    reasons = set()
    for f in range(EYE_FIT_FRAMES + PERIOD):
        ex, ey = _orbit(f, body, (0, 0), 46)
        snap = _snap((120, 173), FACING_NORTH, _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
                     _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x60))
        reasons.add(aim.step(snap)[1])
    assert not aim.drifted
    assert not any(r.startswith("aim_") for r in reasons), reasons


def test_an_eye_parked_at_its_spawn_point_is_not_a_drift() -> None:
    # Pin 17 frame 24: slots 6-9 still sat at (48,173)/(160,125), 79 px off
    # the body, and a still point fits a lap exactly.
    model = PatraEyeModel()
    body = (128, 111)
    for f in range(EYE_FIT_FRAMES + 4):
        ex, ey = _orbit(f, body, (0, 0), 46)
        model.observe(_snap((120, 173), FACING_NORTH, _obj(OBJ_PATRA, *body, slot=1, hp=0xB0),
                            _obj(OBJ_PATRA_EYE, ex, ey, slot=2, hp=0x60),
                            _obj(OBJ_PATRA_EYE, 48, 173, slot=6, hp=0x60)))
    assert model.drift() <= 8


def test_lane_side_sticks_while_the_walk_faces_link_the_other_way() -> None:
    # Run 23 body phase: body (127,127), Link west at (103,133). Walking to
    # the LEFT-facing stand (east of the body) faces Link RIGHT, and the
    # held-facing bonus then picked the west stand: x 103<->104, 425 frames.
    from zelda_i.level9.patra import _patra_stands

    body = _obj(OBJ_PATRA, 127, 127, slot=1, hp=0x50)
    walking_east = _snap((104, 133), FACING_EAST, body)
    sides = [s[2] for s in _patra_stands(walking_east, body, 84, (32, 192, 93, 173), prefer="LEFT")]
    assert sides[0] == "LEFT"
    walking_west = _snap((103, 133), FACING_WEST, body)
    sides = [s[2] for s in _patra_stands(walking_west, body, 84, (32, 192, 93, 173), prefer="RIGHT")]
    assert sides[0] == "RIGHT"


def test_lane_stand_snaps_to_a_walkable_lane_node() -> None:
    # 0x52's east rows are blocks except y=133/141: the clamped (192,107)
    # stand pressed UP into a wall for 200 frames.
    from zelda_i.level9.patra import _patra_stands

    body = _obj(OBJ_PATRA, 108, 107, slot=1, hp=0x50)
    nodes = {(x, y) for x in range(32, 145, 8) for y in range(93, 174, 8)}
    nodes |= {(x, y) for x in range(152, 193, 8) for y in (133, 141)}
    stands = _patra_stands(_snap((192, 133), FACING_WEST, body), body, 84, (32, 192, 93, 173),
                           prefer="LEFT", nodes=nodes)
    for _, goal, facing in stands:
        assert goal in nodes
    assert all(goal != (192, 107) for _, goal, _ in stands)
