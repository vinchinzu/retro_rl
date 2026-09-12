"""Tracking / threat / damage-attribution units (no emulator).

The three blocked-pose cases are the ones ``docs/tasks/rr-npv.*`` burned a
sitting each on: L8 ``0x1E`` ``(128,181)``, L6 ``0x78`` ``(144,141)``,
L5 ``0x77`` ``(120,173)``.
"""

from __future__ import annotations

import numpy as np

from zelda_i.dungeon.behaviors import FIREBALL_TYPE, KEESE_TYPE
from zelda_i.dungeon.postmortem import DamageLog, heart_value
from zelda_i.dungeon.threat import (
    MIN_DODGE_SHOT,
    ReactiveEvader,
    assess,
    contact_frames,
    dodgeable,
    firing_axis,
    in_firing_line,
    off_line_step,
)
from zelda_i.dungeon.tracking import (
    HazardClass,
    ObjectTracker,
    RESPAWN_JUMP,
)
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

# Live IDs from the residual object censuses.
WIZZROBE_TYPE = 0x24
WIZZ_BEAM_TYPE = 0x59  # rr-d6v census; in no dungeon.ids projectile table
POLS_VOICE_TYPE = 0x27
BLUE_GOHMA_TYPE = 0x34
FLOOR_DROP_TYPE = 0x60
FACE_WEST = 0x02
FACE_SOUTH = 0x04

Obj = tuple  # (slot, type, x, y, hp, state, facing)


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


def _drive(tracker, frames):
    """Feed a list of snapshots; return the last tracked tuple."""
    tracked = ()
    for snap in frames:
        tracked = tracker.observe(snap)
    return tracked


# --- tracking ---------------------------------------------------------


def test_velocity_is_measured_across_frames() -> None:
    tracker = ObjectTracker()
    frames = [
        _snap((120, 141), ((1, WIZZ_BEAM_TYPE, 200 - 3 * i, 141, 0, 0, FACE_WEST),))
        for i in range(6)
    ]
    beam = _drive(tracker, frames)[0]
    assert beam.vx == -3.0
    assert beam.vy == 0.0
    assert beam.at(4.0) == (185 - 12.0, 141.0)


def test_respawn_jump_is_not_velocity() -> None:
    tracker = ObjectTracker()
    tracker.observe(_snap((120, 141), ((1, WIZZROBE_TYPE, 80, 141, 64, 0, 0),)))
    far = 80 + RESPAWN_JUMP + 8
    tracked = tracker.observe(
        _snap((120, 141), ((1, WIZZROBE_TYPE, far, 141, 64, 0, 0),))
    )
    assert tracked[0].vx == 0.0
    assert tracked[0].age == 1


def test_same_snapshot_is_observed_once() -> None:
    """``super().step`` re-entry must not sample velocity at double rate."""
    tracker = ObjectTracker()
    first = _snap((120, 141), ((1, WIZZ_BEAM_TYPE, 200, 141, 0, 0, FACE_WEST),))
    second = _snap((120, 141), ((1, WIZZ_BEAM_TYPE, 196, 141, 0, 0, FACE_WEST),))
    tracker.observe(first)
    tracker.observe(first)
    tracked = tracker.observe(second)
    assert tracker.frames == 2
    assert tracked[0].vx == -4.0


def test_untyped_fast_slot_is_a_projectile() -> None:
    """L6 0x59 is in no ids table; 3 px/frame with hp=128 is still a shot."""
    tracker = ObjectTracker()
    frames = [
        _snap((144, 141), ((1, WIZZ_BEAM_TYPE, 200 - 3 * i, 141, 128, 0, FACE_WEST),))
        for i in range(4)
    ]
    beam = _drive(tracker, frames)[0]
    assert beam.hp == 128
    assert beam.hazard is HazardClass.PROJECTILE
    assert beam.is_hazard


def test_age1_unknown_hp128_is_not_a_body() -> None:
    """Point-blank 0x59 spawn: age-1, hp=128, vx=0 must not classify BODY.

    DamageLog ranks the previous frame, so an age-1 body is how a spawn
    on top of Link still reports ``body 0x59`` after the hp=128 fix.
    Motion is not in yet; the unknown slot is a shot until the next frame
    confirms speed.
    """
    tracker = ObjectTracker()
    snap = _snap(
        (144, 141),
        ((1, WIZZ_BEAM_TYPE, 144, 141, 128, 0, FACE_WEST),),
        screen=0x78,
        level=6,
    )
    beam = tracker.observe(snap)[0]
    assert beam.age == 1
    assert beam.vx == 0.0
    assert beam.hp == 128
    assert beam.hazard is HazardClass.PROJECTILE
    assert beam.hazard is not HazardClass.BODY


def test_floor_drop_and_keese_classes() -> None:
    tracker = ObjectTracker()
    tracked = tracker.observe(
        _snap(
            (120, 141),
            (
                (1, FLOOR_DROP_TYPE, 100, 141, 0, 0x22, 0),
                (2, KEESE_TYPE, 140, 141, 0, 0, 0),
            ),
        )
    )
    drop, keese = tracked[0], tracked[1]
    assert drop.hazard is HazardClass.DROP and not drop.is_hazard
    # Keese live at HP 0 (type-only liveness) — a body, never a shot.
    assert keese.hazard is HazardClass.BODY and keese.is_hazard


# --- threat arithmetic ------------------------------------------------


def test_contact_frames_counts_down_a_travelling_shot() -> None:
    tracker = ObjectTracker()
    frames = [
        _snap((144, 141), ((1, WIZZ_BEAM_TYPE, 200 - 4 * i, 141, 0, 0, FACE_WEST),))
        for i in range(4)
    ]
    beam = _drive(tracker, frames)[0]
    stand = contact_frames((144, 141), beam)
    # 44 px of gap, 12 px of pad (Link 8 + shot 4), 4 px/frame.
    assert stand == 9
    # 9 frames is under MIN_DODGE_SHOT: at 1 px/frame Link cannot clear the
    # 12 px pad in time, so leaving the row gains nothing from here.
    assert stand < MIN_DODGE_SHOT
    assert contact_frames((144, 141), beam, step=(0, -1)) == stand


def test_assess_names_the_earliest_source() -> None:
    tracker = ObjectTracker()
    frames = [
        _snap(
            (144, 141),
            (
                (1, WIZZROBE_TYPE, 192, 141, 64, 0, FACE_WEST),
                (2, WIZZ_BEAM_TYPE, 200 - 4 * i, 141, 0, 0, FACE_WEST),
            ),
        )
        for i in range(4)
    ]
    tracked = _drive(tracker, frames)
    impact = assess((144, 141), tracked)
    assert impact.imminent
    assert impact.source is not None and impact.source.slot == 2


# --- blocked pose: L6 0x78 (144,141) ----------------------------------


def test_l6_0x78_beam_is_left_by_leaving_the_row() -> None:
    """rr-d6v: the sidestep walked back along y=141 and died 3/3.

    ``_off_band_dir`` discards the beams entirely; a time-to-contact policy
    must break the shared row, not slide along it.
    """
    tracker = ObjectTracker()
    frames = [
        _snap(
            (144, 141),
            (
                (1, WIZZROBE_TYPE, 192, 141, 64, 0, FACE_WEST),
                (2, WIZZ_BEAM_TYPE, 224 - 4 * i, 141, 0, 0, FACE_WEST),
            ),
            screen=0x78,
            level=6,
        )
        for i in range(4)
    ]
    tracked = _drive(tracker, frames)
    evader = ReactiveEvader(bounds=(56, 200, 109, 173))
    decision = evader.decide(frames[-1], tracked)
    assert decision is not None
    assert decision.direction in {"UP", "DOWN"}
    assert decision.ttc > decision.stand_ttc


def test_l6_0x78_firing_line_is_known_before_the_beam_exists() -> None:
    """The only window a 1 px/frame walker can use is before the shot."""
    tracker = ObjectTracker()
    snap = _snap(
        (144, 141),
        ((1, WIZZROBE_TYPE, 192, 141, 64, 0, FACE_WEST),),
        screen=0x78,
        level=6,
    )
    tracked = tracker.observe(snap)
    wizz = tracked[0]
    assert firing_axis(wizz) == "row"
    assert in_firing_line((144, 141), wizz)
    assert not in_firing_line((144, 117), wizz)
    step = off_line_step((144, 141), tracked, bounds=(56, 200, 109, 173))
    assert step in {"UP", "DOWN"}


# --- blocked pose: L8 0x1E (128,181) ----------------------------------


def _gohma_frames(link, gx=128, gy=141, vy=3, n=4, fireball=True):
    out = []
    for i in range(n):
        objs = [(1, BLUE_GOHMA_TYPE, gx, gy, 100, 0, FACE_SOUTH)]
        if fireball:
            objs.append((2, FIREBALL_TYPE, gx, gy + 8 + vy * i, 0, 0, FACE_SOUTH))
        out.append(_snap(link, tuple(objs)))
    return out


def test_l8_0x1e_stand_line_does_not_oscillate() -> None:
    """rr-npv.4: 3/3 deaths peeling 128<->112 on STAND_Y=181.

    A committed escape is held; the reverse is only taken when it clearly
    beats every other option. Re-deciding each frame must not flip.
    """
    tracker = ObjectTracker()
    evader = ReactiveEvader(bounds=(112, 200, 109, 189))
    frames = _gohma_frames((128, 181), gx=140, vy=3, n=8)
    chosen = []
    for snap in frames:
        tracked = tracker.observe(snap)
        decision = evader.decide(snap, tracked)
        if decision is not None and decision.direction is not None:
            chosen.append(decision.direction)
    flips = sum(
        1
        for a, b in zip(chosen, chosen[1:])
        if {a, b} in ({"LEFT", "RIGHT"}, {"UP", "DOWN"})
    )
    assert flips == 0, chosen


def test_l8_0x1e_reports_no_gain_instead_of_peeling() -> None:
    """The honest answer on the stand line: a 1 px peel buys nothing.

    Three sittings tuned the peel. The model says the peel cannot work, so
    the fix has to be positional (``off_firing_line``), not a wider band.
    """
    tracker = ObjectTracker()
    frames = _gohma_frames((128, 181), gx=128, vy=3, n=6)
    tracked = _drive(tracker, frames)
    impact = assess((128, 181), tracked)
    assert impact.imminent
    assert not dodgeable(impact)
    evader = ReactiveEvader(bounds=(112, 144, 173, 189))
    decision = evader.decide(frames[-1], tracked)
    assert decision is not None
    assert decision.stands
    assert decision.reason in {"evade_no_gain", "evade_boxed_in"}


def test_l8_0x1e_steps_off_gohma_column_when_nothing_is_in_flight() -> None:
    tracker = ObjectTracker()
    frames = _gohma_frames((128, 181), gx=128, n=3, fireball=False)
    tracked = _drive(tracker, frames)
    evader = ReactiveEvader(
        bounds=(112, 200, 109, 189), avoid_firing_lines=True
    )
    decision = evader.decide(frames[-1], tracked)
    assert decision is not None
    assert decision.reason == "off_firing_line"
    assert decision.direction in {"LEFT", "RIGHT"}


# --- blocked pose: L5 0x77 (120,173) ----------------------------------


def test_l5_0x77_peels_off_the_landing_column() -> None:
    """rr-npv.2: the hold line re-centred under a hopping Pols Voice."""
    tracker = ObjectTracker()
    frames = [
        _snap(
            (120, 173),
            ((1, POLS_VOICE_TYPE, 120, 117 + 2 * i, 32, 1, FACE_SOUTH),),
            screen=0x77,
            level=5,
        )
        for i in range(4)
    ]
    tracked = _drive(tracker, frames)
    evader = ReactiveEvader(bounds=(88, 152, 141, 181))
    decision = evader.decide(frames[-1], tracked)
    assert decision is not None
    assert decision.direction in {"LEFT", "RIGHT"}
    assert decision.ttc > decision.stand_ttc


# --- damage attribution -----------------------------------------------


def test_hit_is_blamed_on_the_shot_not_the_idle_body() -> None:
    tracker = ObjectTracker()
    log = DamageLog()
    frames = []
    for i in range(5):
        frames.append(
            _snap(
                (144, 141),
                (
                    (1, WIZZROBE_TYPE, 80, 141, 64, 0, 0),
                    (2, WIZZ_BEAM_TYPE, 176 - 8 * i, 141, 0, 0, FACE_WEST),
                ),
                screen=0x78,
                level=6,
            )
        )
    frames.append(
        _snap(
            (144, 141),
            ((1, WIZZROBE_TYPE, 80, 141, 64, 0, 0),),
            screen=0x78,
            level=6,
            health=0x21,
        )
    )
    hit = None
    for snap in frames:
        tracked = tracker.observe(snap)
        event = log.observe(snap, tracked, action="waist_hold", phase="FIGHT")
        hit = event or hit
    assert hit is not None
    assert hit.type_id == WIZZ_BEAM_TYPE
    assert hit.bearing == "E"  # travelling west, so fired from the east
    assert hit.action == "waist_hold"
    assert "0x59" in hit.label


def test_half_heart_loss_is_a_hit() -> None:
    """Only the partial byte moves on a half heart; whole hearts miss it."""
    log = DamageLog()
    tracker = ObjectTracker()
    full = _snap((120, 141), ((1, KEESE_TYPE, 130, 141, 0, 0, 0),))
    half = _snap(
        (120, 141), ((1, KEESE_TYPE, 130, 141, 0, 0, 0),), partial=0x7F
    )
    assert heart_value(full) > heart_value(half)
    log.observe(full, tracker.observe(full))
    event = log.observe(half, tracker.observe(half), action="combat_wait")
    assert event is not None
    assert event.type_id == KEESE_TYPE


def test_death_carries_the_cause_label() -> None:
    log = DamageLog()
    tracker = ObjectTracker()
    alive = _snap((128, 181), ((1, FIREBALL_TYPE, 128, 168, 0, 0, FACE_SOUTH),))
    dead = _snap(
        (128, 181),
        ((1, FIREBALL_TYPE, 128, 176, 0, 0, FACE_SOUTH),),
        health=0x20,
        partial=0x00,
        mode=17,
    )
    log.observe(alive, tracker.observe(alive))
    log.observe(dead, tracker.observe(dead), action="eye_wait", phase="FIGHT")
    report = log.report()
    assert report["hits"] == 1
    assert report["death_cause"] is not None
    assert "eye_wait" in report["death_cause"]
    assert log.death is not None and log.death.fatal


# --- controller regression: L6 0x78 at the blocked pose ----------------


def test_l6_0x78_controller_breaks_the_row_not_the_column() -> None:
    """rr-d6v v4–v9 died at (144,141) sliding along the beam row.

    Driven with real motion, the controller must answer with a vertical
    break; ``wizzrobe_sidestep`` (LEFT/RIGHT along y=141) is the blocked
    class.
    """
    from zelda_i.level6.wizzrobe import make_west_wizzrobe_controller
    from zelda_i.ram import ADDR_OBJ_TYPE as _T

    ctl = make_west_wizzrobe_controller()
    reasons = []
    for i in range(5):
        ram = np.zeros(0x1000, dtype=np.uint8)
        ram[ADDR_MODE] = PLAY_MODE
        ram[ADDR_LEVEL] = 6
        ram[ADDR_SCREEN] = 0x78
        ram[ADDR_LINK_X] = 144
        ram[ADDR_LINK_Y] = 141
        ram[ADDR_HEALTH] = 0x22
        ram[ADDR_HEART_PARTIAL] = 0xFF
        # Wizzrobe facing west down the waist, beam closing along the row.
        ram[_T + 1] = WIZZROBE_TYPE
        ram[ADDR_LINK_X + 1] = 208
        ram[ADDR_LINK_Y + 1] = 141
        ram[ADDR_OBJ_HP + 1] = 64
        ram[ADDR_LINK_FACING + 1] = FACE_WEST
        ram[_T + 2] = WIZZ_BEAM_TYPE
        ram[ADDR_LINK_X + 2] = 224 - 4 * i
        ram[ADDR_LINK_Y + 2] = 141
        ram[ADDR_LINK_FACING + 2] = FACE_WEST
        reasons.append(ctl.step(read_snapshot(ram)).reason)
    assert "wizzrobe_beam_peel" in reasons
    assert "wizzrobe_sidestep" not in reasons


def test_off_line_step_prefers_open_floor_over_a_corner() -> None:
    """L6 0x78 ROM trial: stepping east off one row died at (189,149).

    Clearing the most firing lines is not enough when the step ends in a
    corner; the tie has to break toward the room's interior.
    """
    tracker = ObjectTracker()
    snap = _snap(
        (176, 141),
        ((1, WIZZROBE_TYPE, 176, 109, 64, 0, FACE_SOUTH),),
        screen=0x78,
        level=6,
    )
    tracked = tracker.observe(snap)
    step = off_line_step((176, 141), tracked, bounds=(56, 200, 109, 173))
    # Both LEFT and RIGHT clear the column; RIGHT ends 24 px from the wall.
    assert step == "LEFT"


def test_live_beam_hp_is_not_the_projectile_test() -> None:
    """L6 ROM census: 0x59 beams carry hp=128, yet they are shots.

    Classifying them as bodies gives them a 16 px pad and a body's dodge
    threshold, which is how the death report read ``body 0x59``.
    """
    tracker = ObjectTracker()
    frames = [
        _snap(
            (144, 141),
            ((1, WIZZ_BEAM_TYPE, 200 - 3 * i, 141, 128, 0, FACE_WEST),),
            screen=0x78,
            level=6,
        )
        for i in range(4)
    ]
    beam = _drive(tracker, frames)[0]
    assert beam.hp == 128
    assert beam.hazard is HazardClass.PROJECTILE


def test_a_known_enemy_stays_a_body_however_fast() -> None:
    tracker = ObjectTracker()
    frames = [
        _snap((120, 141), ((1, KEESE_TYPE, 200 - 4 * i, 141, 0, 0, 0),))
        for i in range(4)
    ]
    keese = _drive(tracker, frames)[0]
    assert keese.hazard is HazardClass.BODY


# --- controller regression: L1 0x33 Stalfos body at d=8 ----------------

STALFOS_TYPE = 0x2A


def test_l1_0x33_does_not_walk_into_a_closing_stalfos_body() -> None:
    """Natural-entry leftover: 0x2a_N then 0x2a_E at d=8, evades=0, lo=1.

    Hits landed during combat_backstep / combat_engage. Tracker saw the
    body; _combat never asked threat.decide, so Link walked into pad=16.
    """
    from retro_harness.nes import nes_action
    from zelda_i.dungeon.engine import GenericDungeonRoomController
    from zelda_i.level1.dungeon import ROOM_33_SPEC

    ctl = GenericDungeonRoomController(ROOM_33_SPEC)
    last = None
    toward = (nes_action("UP"), nes_action("UP", "A"))
    walked_in = 0
    for i in range(20):
        snap = _snap(
            (120, 172),
            ((1, STALFOS_TYPE, 128, 148 + i, 32, 0, FACE_SOUTH),),
            health=0x22,
            screen=0x33,
            level=1,
        )
        last = ctl.step(snap)
        # Velocity needs TRACK_HISTORY samples before decide can peel.
        if i >= 8 and any(np.array_equal(last.action, a) for a in toward):
            walked_in += 1
    assert last is not None
    assert last.reason.startswith("combat_evade")
    assert "engage" not in last.reason
    assert walked_in == 0


def test_l1_0x33_still_chases_a_stalfos_that_is_not_inbound() -> None:
    """Evade is silent when standing is already safe; chase still kills."""
    from zelda_i.dungeon.engine import GenericDungeonRoomController
    from zelda_i.level1.dungeon import ROOM_33_SPEC

    ctl = GenericDungeonRoomController(ROOM_33_SPEC)
    snap = _snap(
        (120, 172),
        ((1, STALFOS_TYPE, 48, 93, 32, 0, FACE_SOUTH),),
        health=0x22,
        screen=0x33,
        level=1,
    )
    action = ctl.step(snap)
    assert action.reason.startswith("combat_")
    assert ctl.report()["tuning"]["evades"] == 0
