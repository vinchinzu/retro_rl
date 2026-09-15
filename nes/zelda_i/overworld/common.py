"""Shared overworld movement helpers for Zelda I controllers.

Both the Level 1 phase controller and the Level 2 hop controller use the same
stuck tracking, periodic sword swing, edge recovery, and align-and-push
primitives. Keep route-specific geometry in the owning module.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.combat import (
    CONTACT_CHEBYSHEV,
    CONTACT_MANHATTAN,
    HEART_OR_FAIRY_STATES,
    RUPEE_DROP_STATES,
    chebyshev,
    in_sword_hitbox,
    manhattan,
    nearest_enemy,
    nearest_heart_or_fairy,
    overworld_threat_objects,
    should_swing_at,
)
from zelda_i.dungeon import ids as _dungeon_ids
from zelda_i.dungeon.behaviors import (
    engagement_hint,
    face_toward,
    is_projectile,
    kind_for_type,
    projectile_threats,
    shield_blocks,
)
from zelda_i.dungeon.hop_controller import dungeon_align_then_push as dungeon_align_then_push
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

DEFAULT_SWING_PERIOD = 12
DEFAULT_SWING_FRAMES = 3
DEFAULT_STUCK_THRESHOLD = 50
# Mode 8 is the hurt-freeze. A knockback loop moves Link, so ``track_stuck``
# reads it as progress; charge the stuck counter per hit instead.
HURT_MODE = 8
KNOCKBACK_STUCK_PENALTY = 20

# Screen-edge thresholds (overworld playfield)
EDGE_SOUTH_Y = 212
EDGE_NORTH_Y = 62
EDGE_EAST_X = 232
EDGE_WEST_X = 14
ARRIVAL_EAST_X = 220
ARRIVAL_WEST_X = 30
ARRIVAL_NORTH_Y = 70
ARRIVAL_SOUTH_Y = 200

# Floor drops share ObjType 0x60; item identity is ObjState (live At4A).
HEART_FAIRY_DROP_TYPES: frozenset[int] = frozenset(
    int(v)
    for name in ("HEART_DROP_OBJECT_TYPE", "FAIRY_DROP_OBJECT_TYPE")
    if (v := getattr(_dungeon_ids, name, None)) is not None
)
HEART_FAIRY_DROP_STATES: frozenset[int] = frozenset(HEART_OR_FAIRY_STATES)


def swing_action(
    phase_frames: int,
    direction: str,
    reason: str,
    *,
    period: int = DEFAULT_SWING_PERIOD,
    hold: int = DEFAULT_SWING_FRAMES,
) -> FrameAction:
    """Walk in ``direction``, pulsing A for a few frames each period."""
    if period > 0 and phase_frames % period < hold:
        return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
    return FrameAction(nes_action(direction), reason)


def _in_contact(link_x: int, link_y: int, obj: ZeldaObject) -> bool:
    return (
        chebyshev(link_x, link_y, obj.x, obj.y) <= CONTACT_CHEBYSHEV
        or manhattan(link_x, link_y, obj.x, obj.y) <= CONTACT_MANHATTAN
    )


def _any_in_hitbox(
    link_x: int, link_y: int, direction: str, threats: tuple[ZeldaObject, ...]
) -> bool:
    return any(
        in_sword_hitbox(link_x, link_y, direction, obj.x, obj.y) for obj in threats
    )


def _off_axis_face(
    link_x: int, link_y: int, threats: tuple[ZeldaObject, ...]
) -> str | None:
    """Face a contact-range threat. Do not abandon a hop for a far side hitbox."""
    best: ZeldaObject | None = None
    best_d = 10**9
    for obj in threats:
        if not _in_contact(link_x, link_y, obj):
            continue
        d = manhattan(link_x, link_y, obj.x, obj.y)
        if d < best_d:
            best_d = d
            best = obj
    if best is None:
        return None
    return face_toward(link_x, link_y, best.x, best.y)


def overworld_projectiles(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Live shots on screen. ``overworld_threat_objects`` drops these (hp=0)."""
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1 and obj.type_id not in (0, 0xFF) and is_projectile(obj)
    )


def answer_projectile(
    link_x: int,
    link_y: int,
    direction: str,
    projectiles: tuple[ZeldaObject, ...],
    reason: str,
    *,
    magic_shield: bool = False,
) -> FrameAction | None:
    """Shield or sidestep an inbound shot; None when the lane is clear.

    Walking ``direction`` means Link faces the band, and the shield blocks
    while facing and not attacking — so a blockable shot costs nothing but
    the A press (``swing_action`` pulses it on a fixed period and would
    cancel the block). Anything the shield cannot eat has to leave the lane;
    perpendicular keeps hop progress on the other axis.
    """
    hits = projectile_threats(link_x, link_y, projectiles, direction=direction)
    if not hits:
        return None
    nearest = min(hits, key=lambda o: manhattan(link_x, link_y, o.x, o.y))
    if all(shield_blocks(obj, magic_shield=magic_shield) for obj in hits):
        return FrameAction(nes_action(direction), f"{reason}_shield")
    if direction in ("LEFT", "RIGHT"):
        step = "UP" if int(nearest.y) - int(link_y) > 0 else "DOWN"
        if step == "UP" and link_y <= EDGE_NORTH_Y + 8:
            step = "DOWN"
        elif step == "DOWN" and link_y >= EDGE_SOUTH_Y - 8:
            step = "UP"
    else:
        step = "LEFT" if int(nearest.x) - int(link_x) > 0 else "RIGHT"
        if step == "LEFT" and link_x <= EDGE_WEST_X + 8:
            step = "RIGHT"
        elif step == "RIGHT" and link_x >= EDGE_EAST_X - 8:
            step = "LEFT"
    return FrameAction(nes_action(step), f"{reason}_dodge")


def track_knockback(
    snap: ZeldaSnapshot,
    *,
    last_health: int,
    hits: int,
    stuck: int,
    penalty: int = KNOCKBACK_STUCK_PENALTY,
) -> tuple[int, int, int]:
    """Return updated (hits, last_health, stuck) after charging a fresh hit.

    Health is the raw byte (hearts high nibble, partial low), so any decrease
    is damage. Each hit charges ``stuck`` so a hit/shove/walk-back loop
    reaches the unstick ladder instead of reading as progress forever.
    """
    health = int(snap.health)
    if last_health >= 0 and health < last_health:
        return hits + 1, health, stuck + penalty
    return hits, health, stuck


def walk_or_swing(
    phase_frames: int,
    direction: str,
    reason: str,
    snap: ZeldaSnapshot | None = None,
    *,
    period: int = DEFAULT_SWING_PERIOD,
    hold: int = DEFAULT_SWING_FRAMES,
    always_swing: bool = False,
) -> FrameAction:
    """Walk in ``direction``; pulse A only if always_swing or a nearby threat is swingable.

    When ``snap is None`` or ``always_swing``, keep the old periodic swing
    (tests / stuck recovery). Otherwise slash only for hitbox or contact-range
    threats — not merely because any enemy exists on screen.

    Contact-range off-axis threats turn Link toward them this frame so a hop
    does not walk through a body. Far side hitboxes do not steal the travel
    direction (that stalled sword-cave on the first ROM eval).
    """
    if snap is None or always_swing:
        return swing_action(
            phase_frames, direction, reason, period=period, hold=hold
        )
    threats = overworld_threat_objects(snap)
    lx, ly = snap.link_x, snap.link_y
    nearest = nearest_enemy(lx, ly, threats)
    hint = (
        engagement_hint(kind_for_type(nearest.type_id), snap, nearest)
        if nearest is not None
        else None
    )

    # Travel-direction hitbox is never vetoed by the nearest-enemy hint
    # (that hint's swing flag is for its own face).
    if _any_in_hitbox(lx, ly, direction, threats) and should_swing_at(
        lx, ly, direction, threats
    ):
        return swing_action(
            phase_frames, direction, reason, period=period, hold=hold
        )

    face = None
    if hint is not None and nearest is not None and _in_contact(lx, ly, nearest):
        face = hint.face
    if face is None:
        face = _off_axis_face(lx, ly, threats)
    if face is not None:
        if should_swing_at(lx, ly, face, threats):
            return swing_action(
                phase_frames, face, reason, period=period, hold=hold
            )
        return FrameAction(nes_action(face), reason)
    # Nothing to hit: block what the shield eats, sidestep the rest. Never
    # walk the lane pressing A — that cancels the block for free damage.
    answer = answer_projectile(
        lx,
        ly,
        direction,
        overworld_projectiles(snap),
        reason,
        magic_shield=bool(snap.magic_shield),
    )
    if answer is not None:
        return answer
    return FrameAction(nes_action(direction), reason)


def track_stuck(
    snap: ZeldaSnapshot,
    *,
    last_x: int,
    last_y: int,
    last_screen: int,
    stuck: int,
) -> tuple[int, int, int, int]:
    """Return updated (stuck, last_x, last_y, last_screen)."""
    if (
        snap.link_x == last_x
        and snap.link_y == last_y
        and snap.screen == last_screen
        and not snap.transitioning
    ):
        stuck += 1
    else:
        stuck = 0
    return stuck, snap.link_x, snap.link_y, snap.screen


def on_arrival_edge(direction: str, snap: ZeldaSnapshot) -> bool:
    """True while Link is still on the edge that produced this hop's arrival."""
    if direction == "RIGHT":
        return snap.link_x > ARRIVAL_EAST_X
    if direction == "LEFT":
        return snap.link_x < ARRIVAL_WEST_X
    if direction == "UP":
        return snap.link_y < ARRIVAL_NORTH_Y
    if direction == "DOWN":
        return snap.link_y > ARRIVAL_SOUTH_Y
    return False


def recover_off_edge(
    snap: ZeldaSnapshot,
    travel_direction: str,
    *,
    swing: Callable[[str, str], FrameAction],
) -> FrameAction | None:
    """Nudge inward if Link is scraping the wrong screen edge."""
    if snap.link_y >= EDGE_SOUTH_Y and travel_direction != "DOWN":
        return swing("UP", "off_south")
    if snap.link_y <= EDGE_NORTH_Y and travel_direction != "UP":
        return swing("DOWN", "off_north")
    if snap.link_x >= EDGE_EAST_X and travel_direction != "RIGHT":
        return swing("LEFT", "off_east")
    if snap.link_x <= EDGE_WEST_X and travel_direction != "LEFT":
        return swing("RIGHT", "off_west")
    return None


def _nearest_typed_drop(
    snap: ZeldaSnapshot,
    types: Iterable[int],
    states: Iterable[int] | None = None,
) -> ZeldaObject | None:
    type_set = frozenset(int(t) for t in types)
    if not type_set:
        return None
    state_set = None if states is None else frozenset(int(s) for s in states)
    if state_set == HEART_OR_FAIRY_STATES:
        hit = nearest_heart_or_fairy(snap)
        if hit is not None:
            return hit
    candidates = [
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and int(obj.type_id) in type_set
        and (state_set is None or int(obj.state) in state_set)
        and 40 < obj.y < 220
        and 8 < obj.x < 248
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda obj: manhattan(snap.link_x, snap.link_y, obj.x, obj.y),
    )


def scoop_toward_drop(
    snap: ZeldaSnapshot,
    obj: ZeldaObject | None,
    *,
    reason: str,
    travel_dir: str | None,
    radius: int,
) -> FrameAction | None:
    """Walk onto a nearby floor drop. Contact pickup; no A.

    None if ``obj`` is missing, farther than ``radius``, or sitting on the
    opposite scroll edge from ``travel_dir`` (RIGHT refuses a west-edge
    drop, and so on).
    """
    if obj is None:
        return None
    dist = manhattan(snap.link_x, snap.link_y, obj.x, obj.y)
    if dist > radius:
        return None
    if travel_dir == "RIGHT" and obj.x < EDGE_WEST_X + 16:
        return None
    if travel_dir == "LEFT" and obj.x > EDGE_EAST_X - 16:
        return None
    if travel_dir == "DOWN" and obj.y < EDGE_NORTH_Y + 16:
        return None
    if travel_dir == "UP" and obj.y > EDGE_SOUTH_Y - 16:
        return None
    if dist <= 4:
        return FrameAction(nes_idle_action(), reason)
    dx = obj.x - snap.link_x
    dy = obj.y - snap.link_y
    if abs(dx) >= abs(dy) and abs(dx) > 2:
        direction = "RIGHT" if dx > 0 else "LEFT"
    elif abs(dy) > 2:
        direction = "DOWN" if dy > 0 else "UP"
    else:
        return FrameAction(nes_idle_action(), reason)
    return FrameAction(nes_action(direction), reason)


def scoop_floor_drop(
    snap: ZeldaSnapshot,
    *,
    types: Iterable[int],
    travel_dir: str | None,
    radius: int,
    reason: str,
    want: bool,
    states: Iterable[int] | None = None,
) -> FrameAction | None:
    """Nearest in-bounds drop of ``types``/``states``, else None.

    Floor drops share ObjType 0x60; pass ``states`` for heart vs rupee.
    No-op when ``want`` is False.
    """
    if not want:
        return None
    return scoop_toward_drop(
        snap,
        _nearest_typed_drop(snap, types, states=states),
        reason=reason,
        travel_dir=travel_dir,
        radius=radius,
    )


# One 4-direction cycle, then stand. Never reset stuck (that restarts the spam).
IN_PLACE_WIGGLE_FRAMES = 16


def unstick_wiggle(
    stuck: int,
    *,
    reason: str = "unstick",
    reset_after: int = 140,
    wiggle_frames: int = IN_PLACE_WIGGLE_FRAMES,
) -> tuple[FrameAction, int]:
    """Brief cardinal nudge, then stand still. Returns (action, new_stuck).

    ``reset_after`` is accepted for call-site compat and ignored: resetting
    stuck caused thousands of LEFT/RIGHT/DOWN frames in place. If the walk
    is blocked, wait; do not loop a wiggle.
    """
    del reset_after
    if stuck > wiggle_frames:
        return FrameAction(nes_idle_action(), f"{reason}_wait"), stuck
    wiggle = ("UP", "DOWN", "LEFT", "RIGHT")[stuck % 4]
    return FrameAction(nes_action(wiggle, "A"), reason), stuck


def align_and_push(
    snap: ZeldaSnapshot,
    *,
    direction: str,
    reason: str,
    align_x: int | None = None,
    align_y: int | None = None,
    y_band: tuple[int, int] | None = None,
    stuck: int = 0,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
    x_tol: int = 5,
    y_tol: int = 5,
    swing: Callable[[str, str], FrameAction] | None = None,
    swing_period: int = DEFAULT_SWING_PERIOD,
    swing_hold: int = DEFAULT_SWING_FRAMES,
    phase_frames: int = 0,
    align_x_at_wall: bool = False,
) -> FrameAction:
    """Align to optional x/y or y-band, then push in ``direction``.

    Default movement uses :func:`walk_or_swing` (threat-gated). Callers that
    pass ``swing=`` own the slash policy (controllers usually close over snap).
    Stuck recovery tries one short :func:`unstick_wiggle` cycle, then idles.

    ``align_x_at_wall`` is opt-in per hop: it keeps ``align_x`` live past the
    ``80 < y < 205`` interior band for the hop's own direction. Leave it off
    unless that hop's wall stall was measured -- a blanket wall strafe walked
    the post-L6 0x22 DOWN leftover (120,221) left onto ``L6_CAVE_MOUTH_X``.
    """

    def _swing(dir_: str, why: str) -> FrameAction:
        if swing is not None:
            return swing(dir_, why)
        return walk_or_swing(
            phase_frames,
            dir_,
            why,
            snap,
            period=swing_period,
            hold=swing_hold,
        )

    if stuck > stuck_threshold:
        action, _ = unstick_wiggle(stuck, reason=f"{reason}_unstick")
        return action

    if y_band is not None:
        lo, hi = y_band
        if snap.link_y < lo:
            return _swing("DOWN", "band_down")
        if snap.link_y > hi:
            return _swing("UP", "band_up")
        return _swing(direction, reason)

    # Interior: skip x-align at the north/south walls (a strafe there walks
    # past the mouth). An opted-in vertical hop keeps the gap column at its
    # own far wall — y=205 is rock, not EDGE_SOUTH_Y=212.
    y = snap.link_y
    can_align_x = 80 < y < 205
    if align_x_at_wall:
        if direction == "DOWN" and y >= 205:
            can_align_x = True
        elif direction == "UP" and y <= 80:
            can_align_x = True
    if (
        align_x is not None
        and abs(snap.link_x - align_x) > x_tol
        and can_align_x
    ):
        btn = "LEFT" if snap.link_x > align_x else "RIGHT"
        return _swing(btn, f"{reason}_ax")

    if (
        align_y is not None
        and abs(snap.link_y - align_y) > y_tol
        and 25 < snap.link_x < 230
    ):
        # Force the travel direction near the entry edge so corridor alignment
        # does not scrape rocks after a screen scroll.
        if direction == "RIGHT" and snap.link_x <= 18:
            return _swing("RIGHT", "enter_corridor")
        if direction == "RIGHT" and snap.link_y > 200:
            return _swing("UP", "climb_entry")
        if direction == "RIGHT" and snap.link_y < 70:
            return _swing("DOWN", "drop_entry")
        btn = "UP" if snap.link_y > align_y else "DOWN"
        return _swing(btn, f"{reason}_ay")

    return _swing(direction, reason)


def wake_or_wait_mode(phase_frames: int, mode: int) -> FrameAction:
    """Brief A pulse, then idle, while waiting out non-play modes."""
    if phase_frames % 30 < 3:
        return FrameAction(nes_action("A"), f"wake_mode_{mode}")
    return FrameAction(nes_idle_action(), f"wait_mode_{mode}")


# --- Dungeon diamond-block door approach (L2 0x7d east, 0x6e east, …) ---
# Mid-room diamond solids block a straight y≈141 corridor near x≈128–176.
# Correct policy (verified 2026-08-06): reach east wall on an open y-band, then
# cycle LEFT+vertical to free y≈141 *at the wall*, then RIGHT through the door.
# Do NOT fully retreat west of ~x=180 on y=141 (re-enters the solid).
# Do NOT y-align only with micro-LEFT at x≥200 without longer LEFT pulses.

DOOR_Y_DEFAULT = 141
DIAMOND_WALL_X = 200
DIAMOND_BAND_7D = 157  # open band for entry-east 0x7d → 0x7e
DIAMOND_BAND_6E = 113  # open band for 0x6e RIGHT key door → 0x6f


def diamond_east_phase(
    snap: ZeldaSnapshot,
    *,
    phase: str,
    band_y: int = DIAMOND_BAND_7D,
    door_y: int = DOOR_Y_DEFAULT,
    wall_x: int = DIAMOND_WALL_X,
    cycle: int = 0,
) -> tuple[FrameAction, str]:
    """One-frame policy for diamond-blocked east doors.

    Phases (caller advances when geometry matches):
      free     — leave west/east/south/north alcoves toward mid-room
      band     — align ``band_y`` at mid-x
      wall     — RIGHT on band until ``link_x >= wall_x``
      door_y   — at wall: LEFT×6 → vertical to door_y → RIGHT×10 cycles
      push     — hold RIGHT on door_y (re-nudge if y drifts)

    Returns (action, next_phase_hint). Caller may keep phase until transition.
    """
    x, y = snap.link_x, snap.link_y

    if phase == "free":
        if 70 <= x <= 180 and 110 <= y <= 175:
            return FrameAction(nes_action("UP" if y > band_y else "DOWN"), "band"), "band"
        if x >= 200:
            if not (120 <= y <= 170):
                return FrameAction(nes_action("UP" if y > 170 else "DOWN"), "free_ey"), "free"
            return FrameAction(nes_action("LEFT"), "free_ex"), "free"
        if x <= 48:
            if not (120 <= y <= 170):
                return FrameAction(nes_action("DOWN" if y < 120 else "UP"), "free_wy"), "free"
            return FrameAction(nes_action("RIGHT"), "free_wx"), "free"
        if y >= 195:
            if abs(x - 120) > 10:
                return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_sx"), "free"
            return FrameAction(nes_action("UP"), "free_sy"), "free"
        if y <= 95:
            if abs(x - 120) > 10:
                return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_nx"), "free"
            return FrameAction(nes_action("DOWN"), "free_ny"), "free"
        if abs(x - 120) >= abs(y - 141):
            return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_cx"), "free"
        return FrameAction(nes_action("DOWN" if y < 141 else "UP"), "free_cy"), "free"

    if phase == "band":
        if abs(y - band_y) <= 4 and 90 <= x <= 160:
            return FrameAction(nes_action("RIGHT"), "to_wall"), "wall"
        if abs(y - band_y) > 4:
            return FrameAction(nes_action("DOWN" if y < band_y else "UP"), "band_y"), "band"
        if x < 90:
            return FrameAction(nes_action("RIGHT"), "band_x"), "band"
        if x > 160:
            return FrameAction(nes_action("LEFT"), "band_x"), "band"
        return FrameAction(nes_action("RIGHT"), "band"), "band"

    if phase == "wall":
        if x >= wall_x:
            return FrameAction(nes_action("LEFT"), "at_wall"), "door_y"
        if abs(y - band_y) > 8:
            return FrameAction(nes_action("DOWN" if y < band_y else "UP"), "wall_y"), "wall"
        return FrameAction(nes_action("RIGHT"), "wall_r"), "wall"

    if phase == "door_y":
        # S2 cycle: LEFT block → vertical to door_y → RIGHT block.
        # Longer LEFT when still off door_y (vertical is solid at x≈200).
        step_in_cycle = cycle % 28
        if abs(y - door_y) <= 2 and x >= wall_x - 6:
            return FrameAction(nes_action("RIGHT"), "door_ready"), "push"
        left_hold = 10 if abs(y - door_y) > 2 else 6
        if step_in_cycle < left_hold:
            return FrameAction(nes_action("LEFT"), "door_left"), "door_y"
        if step_in_cycle < left_hold + 12:
            if abs(y - door_y) <= 2:
                return FrameAction(nes_action("RIGHT"), "door_r_early"), "door_y"
            return FrameAction(
                nes_action("UP" if y > door_y else "DOWN"), "door_vert"
            ), "door_y"
        return FrameAction(nes_action("RIGHT"), "door_right"), "door_y"

    # push — pure y-align + RIGHT. Do NOT LEFT-nudge here: that re-enters the
    # mid-room diamond solid on door_y (observed fail (208,149)→(176,141)).
    if abs(y - door_y) > 4:
        return FrameAction(nes_action("UP" if y > door_y else "DOWN"), "push_y"), "push"
    return FrameAction(nes_action("RIGHT"), "push_r"), "push"
