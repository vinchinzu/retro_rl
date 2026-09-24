"""Shared overworld movement helpers for Zelda I controllers.

Both the Level 1 phase controller and the Level 2 hop controller use the same
stuck tracking, periodic sword swing, edge recovery, and align-and-push
primitives. Keep route-specific geometry in the owning module.
"""

from __future__ import annotations

from zelda_i.walk.physics import WALK_DELTA, lattice_step

from collections.abc import Iterable
from typing import Callable

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
from zelda_i.combat import (
    CONTACT_CHEBYSHEV,
    CONTACT_MANHATTAN,
    HEART_OR_FAIRY_STATES,
    chebyshev,
    facing_to_direction,
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
from zelda_i.dungeon.threat import MIN_DODGE_BODY
from zelda_i.overworld.prey import SKIP_TYPES as NO_ENGAGE_TYPES
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
# One pixel inside each scroll line, so an escape step never hands the screen
# back. Same shape as ``hunt.HUNT_BOX``: ``(xlo, xhi, ylo, yhi)``.
DODGE_BOX = (EDGE_WEST_X + 1, EDGE_EAST_X - 1, EDGE_NORTH_Y + 1, EDGE_SOUTH_Y - 1)

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


def facing_direction(snap: ZeldaSnapshot | None) -> str | None:
    """Link's own facing as a direction name, or ``None`` if it reads odd.

    A Zora reads ``0x03`` (Right|Left) and Link's own byte can be read
    mid-write, so an unmapped value is not an error here.
    """
    if snap is None:
        return None
    try:
        return facing_to_direction(int(snap.facing))
    except ValueError:
        return None


def swing_or_turn(
    phase_frames: int,
    direction: str,
    reason: str,
    snap: ZeldaSnapshot | None,
    *,
    period: int = DEFAULT_SWING_PERIOD,
    hold: int = DEFAULT_SWING_FRAMES,
) -> FrameAction:
    """:func:`swing_action`, but never press A across a turn.

    A turn and a swing cannot share a frame: measured
    (``scratch/probe_turn_swing.py``, ``turn4``) a walking Link keeps his old
    facing through 22 of 64 ``dir+A`` presses, and the blade then goes out
    along an axis the body is not on — a miss that pins him for 13 frames.
    Holding the direction alone turns him in 1-4 frames and he is walking,
    not pinned, while it happens. The frame this returns instead is the one
    the non-swing phase of the period returns anyway, so nothing that could
    stall on a wall is new.
    """
    held = facing_direction(snap)
    if held is not None and held != direction:
        return FrameAction(nes_action(direction), f"{reason}_turn")
    return swing_action(phase_frames, direction, reason, period=period, hold=hold)


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


def box_step(
    x: int, y: int, direction: str, box: tuple[int, int, int, int]
) -> str | None:
    """``direction`` if one step of it stays inside ``box``, else ``None``."""
    nx, ny = x, y
    if direction == "LEFT":
        nx = x - 2
    elif direction == "RIGHT":
        nx = x + 2
    elif direction == "UP":
        ny = y - 2
    elif direction == "DOWN":
        ny = y + 2
    else:
        return None
    xlo, xhi, ylo, yhi = box
    if xlo <= nx <= xhi and ylo <= ny <= yhi:
        return direction
    return None


def perpendicular(
    lx: int, ly: int, mx: int, my: int, box: tuple[int, int, int, int],
    bodies: tuple[ZeldaObject, ...] = (),
    avoid: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """Step that opens the angle to a muzzle at ``(mx, my)``, or ``None``.

    Zora aim is quantized at launch; never walk into the muzzle. ``bodies``
    is the same rule :meth:`ShotPolicy.face` keeps: a shot is not a reason to
    walk into an octorok. ``avoid`` is what the *caller* has measured to be
    unwalkable — ``box`` is a rectangle and the overworld is not, so a step
    that Link has already failed to take belongs here and nowhere else.
    """
    dx, dy = int(mx) - int(lx), int(my) - int(ly)
    # Cross the *major* axis of the bearing. Shared row (dy == 0) -> UP/DOWN.
    if abs(dx) >= abs(dy):
        options = ("DOWN", "UP") if dy <= 0 else ("UP", "DOWN")
    else:
        options = ("RIGHT", "LEFT") if dx <= 0 else ("LEFT", "RIGHT")
    for direction in options:
        if direction in avoid:
            continue
        # Room, not one step: a sidestep only clears a hitbox once Link has
        # walked the whole pad, so a side with 3 px of wall left is not an
        # escape — it is the wall he is about to be pinned against. This is
        # the old edge flip stated as what it was for, and it still answers
        # when Link is *already* inside the margin (both box tests pass there
        # and only the room test can tell the two sides apart).
        if _room(lx, ly, direction, box) < MIN_DODGE_BODY:
            continue
        nx = lx + WALK_DELTA[direction][0] * MIN_DODGE_BODY
        ny = ly + WALK_DELTA[direction][1] * MIN_DODGE_BODY
        if any(
            chebyshev(nx, ny, int(b.x), int(b.y)) < MIN_DODGE_BODY for b in bodies
        ):
            continue
        return direction
    return None


# ------------------------------------------------ the shot escape ---
# ``Z_01.asm`` ``CheckLinkCollision``: a shot hits when its middle is inside
# 9 px of Link's middle on both axes. Link's middle is ``(x+8, y+8)``; a
# half-width shot's is ``(x+4, y+8)``, a full one's ``(x+8, y+8)``.
SHOT_HIT_PX = 9
# Link's walk, measured: 1 and 2 px frames alternating.
LINK_WALK_PX = 1.5
SHOT_ESCAPE_HORIZON = 48
# Contact with a body inside this many frames costs the candidate its tie.
SHOT_ESCAPE_BODY_FRAMES = 16
_LATTICE = 8


def _sim_walk(
    x: int, y: int, direction: str | None, n: int,
    nodes: frozenset[tuple[int, int]] | None, box: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    """Link's position for ``n`` frames holding ``direction``.

    Turn rule (measured on ``OW_39``): off the grid on the other axis, he
    first slides at walk speed to the nearest grid line, then turns. With
    ``nodes`` he stops on the last walkable lattice node before rock.
    """
    fx, fy = float(x), float(y)
    out: list[tuple[float, float]] = []
    xlo, xhi, ylo, yhi = box
    for _ in range(n):
        if direction is not None:
            step = LINK_WALK_PX
            vertical = direction in ("UP", "DOWN")
            off = (fx % _LATTICE) if vertical else ((fy - 5) % _LATTICE)
            if off:
                # Slide to the nearest line on the other axis first.
                back = off
                fwd = _LATTICE - off
                delta = -min(step, back) if back < fwd else min(step, fwd)
                if vertical:
                    fx += delta
                else:
                    fy += delta
            else:
                sign = -1 if direction in ("UP", "LEFT") else 1
                cur = fy if vertical else fx
                base = (cur - 5) if vertical else cur
                # The ROM tests the next node only from a node; between two
                # he is already committed to the far one.
                if nodes is not None and base % _LATTICE == 0:
                    line = base + sign * _LATTICE
                    node = (int(fx), int(line + 5)) if vertical else (int(line), int(fy))
                    if node not in nodes:
                        step = 0.0
                nxt = cur + sign * step
                # Land on the next grid line rather than step over it, as the
                # ROM's 1/2 px frames do, so the node test above sees it.
                ahead = (base // _LATTICE + 1) * _LATTICE if sign > 0 else (
                    (base // _LATTICE - (0 if base % _LATTICE else 1)) * _LATTICE
                )
                if abs(nxt - cur) > abs(ahead - base):
                    nxt = ahead + (5 if vertical else 0)
                if vertical:
                    fy = min(max(nxt, ylo), yhi)
                else:
                    fx = min(max(nxt, xlo), xhi)
        out.append((fx, fy))
    return out


def shot_escape(
    lx: int,
    ly: int,
    shots: Iterable[tuple[float, float, float, float, int]],
    box: tuple[int, int, int, int],
    *,
    nodes: frozenset[tuple[int, int]] | None = None,
    bodies: tuple[ZeldaObject, ...] = (),
    prefer: tuple[str, ...] = (),
    horizon: int = SHOT_ESCAPE_HORIZON,
) -> tuple[str | None, bool]:
    """Best held input against shots flying straight, and whether it is needed.

    ``shots`` is ``(x, y, vx, vy, x_off)``: position, measured velocity and
    the middle's x offset (4 half-width, 8 full). Candidates are standing and
    the four walks, each simulated with the turn rule and the lattice. Score
    is: never hit, then latest first hit, then no body contact early, then
    distance off each shot's line at the horizon, then the widest miss. Returns ``(direction, needed)``: ``needed`` is False when
    every candidate is safe, so the caller's own ladder can keep the frame.
    """
    shots = tuple(shots)
    candidates: tuple[str | None, ...] = (None, "UP", "DOWN", "LEFT", "RIGHT")
    scored: list[tuple[tuple, str | None]] = []
    all_safe = True
    for cand in candidates:
        path = _sim_walk(lx, ly, cand, horizon, nodes, box)
        first_hit = horizon + 1
        margin = 10**6
        for sx, sy, vx, vy, ox in shots:
            for k, (px, py) in enumerate(path, start=1):
                mx = sx + vx * k + ox - (px + 8)
                my = sy + vy * k - py
                m = max(abs(mx), abs(my)) - SHOT_HIT_PX
                if m < 0:
                    first_hit = min(first_hit, k)
                    break
                margin = min(margin, m)
        # Off the line, not ahead of it: fleeing down a shot's own line is
        # "not hit yet" for a long horizon and still ends on it (the 0x7B
        # hits that ran with the spit). Score the end pose's distance from
        # each shot's line through Link's middle.
        ex, ey = path[-1]
        off_line = min(
            (
                abs((ex + 8 - sx - ox) * vy - (ey - sy) * vx)
                / max(1e-6, (vx * vx + vy * vy) ** 0.5)
                for sx, sy, vx, vy, ox in shots
            ),
            default=0.0,
        )
        touch = any(
            max(abs(px - int(b.x)), abs(py - int(b.y))) < MIN_DODGE_BODY
            for px, py in path[:SHOT_ESCAPE_BODY_FRAMES]
            for b in bodies
        )
        safe = first_hit > horizon
        all_safe = all_safe and safe
        rank = prefer.index(cand) if cand in prefer else len(prefer)
        scored.append(((safe, first_hit, not touch, min(off_line, 24.0), min(margin, 24), -rank), cand))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1], not all_safe


def keep_y_band(
    step: str | None,
    lx: int,
    ly: int,
    sx: int,
    sy: int,
    box: tuple[int, int, int, int],
    bodies: tuple[ZeldaObject, ...] = (),
    band: tuple[int, int] | None = None,
    avoid: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """Stay on a hop's y-band when the shot is not on it.

    ``perpendicular`` crosses the bearing's major axis. A spit in the water
    south of the band is still mostly east of Link, so that rule walks UP.
    Live ``pre_l1_shortfall1`` on 0x7D: five of those UP steps left
    ``SCREEN_7E_EAST_BAND`` (137–145) and the last one died on the octorok
    rock at y=109. A shot already inside the band still leaves the row.
    Two pixels is the step ``box_step`` already uses; one pixel still reads
    as inside on the frame that walks out.
    """
    if step not in ("UP", "DOWN") or band is None:
        return step
    lo, hi = int(band[0]), int(band[1])
    ly, sy = int(ly), int(sy)
    if not (lo <= ly <= hi):
        return step
    ny = ly - 2 if step == "UP" else ly + 2
    if lo <= ny <= hi or lo <= sy <= hi:
        return step
    options = ("LEFT", "RIGHT") if int(sx) >= int(lx) else ("RIGHT", "LEFT")
    for direction in options:
        if direction in avoid:
            continue
        if _room(int(lx), ly, direction, box) < MIN_DODGE_BODY:
            continue
        nx = int(lx) + WALK_DELTA[direction][0] * MIN_DODGE_BODY
        if any(
            chebyshev(nx, ly, int(b.x), int(b.y)) < MIN_DODGE_BODY for b in bodies
        ):
            continue
        return direction
    return None


def _room(lx: int, ly: int, direction: str, box: tuple[int, int, int, int]) -> int:
    """Pixels of ``direction`` left inside ``box``. Negative outside it."""
    xlo, xhi, ylo, yhi = box
    if direction == "LEFT":
        return int(lx) - int(xlo)
    if direction == "RIGHT":
        return int(xhi) - int(lx)
    if direction == "UP":
        return int(ly) - int(ylo)
    if direction == "DOWN":
        return int(yhi) - int(ly)
    return -1


def answer_projectile(
    link_x: int,
    link_y: int,
    direction: str,
    projectiles: tuple[ZeldaObject, ...],
    reason: str,
    *,
    magic_shield: bool = False,
    bodies: tuple[ZeldaObject, ...] = (),
) -> FrameAction | None:
    """Shield or sidestep an inbound shot; None when the lane is clear.

    Walking ``direction`` means Link faces the band, and the shield blocks
    while facing and not attacking — so a blockable shot costs nothing but
    the A press (``swing_action`` pulses it on a fixed period and would
    cancel the block). Anything the shield cannot eat has to leave the lane.

    The escape crosses the bearing to the *shot*, not the travel axis. The
    old rule sidestepped perpendicular to ``direction`` and then flipped the
    step at the screen edge, which on a wall reverses it **into** the shot:
    live 0x7C (``scratch/zhit1.json`` f=5806), Link pinned at x=16 walking
    DOWN with the spit 19 px east, stepped RIGHT three frames running and
    took it. :func:`perpendicular` answers with the bearing's minor axis and
    only returns a step the box has room for, so there is no flip to make.
    None now means *no escape step is better than the push* — the caller
    keeps walking rather than being handed a direction that closes the gap.
    """
    hits = projectile_threats(link_x, link_y, projectiles, direction=direction)
    if not hits:
        return None
    nearest = min(hits, key=lambda o: manhattan(link_x, link_y, o.x, o.y))
    if all(shield_blocks(obj, magic_shield=magic_shield) for obj in hits):
        return FrameAction(nes_action(direction), f"{reason}_shield")
    step = perpendicular(
        link_x, link_y, int(nearest.x), int(nearest.y), DODGE_BOX, bodies
    )
    if step is None:
        return None
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
    # Turning off the travel axis *is* engaging, and a Zora is never a fight
    # this walk takes: ``prey.SKIP_TYPES`` is the same list the hunt refuses
    # to chase. Leave it in ``threats`` — a swing the travel direction was
    # already pulsing costs nothing — but never let it own the face.
    engageable = tuple(
        obj for obj in threats if (int(obj.type_id) & 0xFF) not in NO_ENGAGE_TYPES
    )
    lx, ly = snap.link_x, snap.link_y
    nearest = nearest_enemy(lx, ly, engageable)
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
        return swing_or_turn(
            phase_frames, direction, reason, snap, period=period, hold=hold
        )

    face = None
    if hint is not None and nearest is not None and _in_contact(lx, ly, nearest):
        face = hint.face
    if face is None:
        face = _off_axis_face(lx, ly, engageable)
    if face is not None:
        if should_swing_at(lx, ly, face, threats):
            return swing_or_turn(
                phase_frames, face, reason, snap, period=period, hold=hold
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
        bodies=threats,
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
    # A drop in a live body's pad is not a pickup, it is a contact: 10 of 33
    # 0x7B hits over twelve RNG offsets (``e3``) were ``scoop_rupee`` walking
    # onto a rupee with a red leever beside it or beside Link. The drop
    # outlasts the wave; leave it until the body has moved.
    bodies = overworld_threat_objects(snap)
    if any(
        chebyshev(int(obj.x), int(obj.y), int(b.x), int(b.y)) <= MIN_DODGE_BODY
        or chebyshev(int(snap.link_x), int(snap.link_y), int(b.x), int(b.y)) <= MIN_DODGE_BODY
        for b in bodies
    ):
        return None
    if dist <= 4:
        return _stand_on_drop(snap, reason)
    if max(abs(obj.x - snap.link_x), abs(obj.y - snap.link_y)) <= 2:
        return _stand_on_drop(snap, reason)
    # A press Link can take from here: UP off a column slides him sideways,
    # which fought the hunt's scoop frame by frame (live 0x79 x 80<->82).
    direction = lattice_step(snap.link_x, snap.link_y, (obj.x, obj.y))
    if direction is None:
        return _stand_on_drop(snap, reason)
    return FrameAction(nes_action(direction), reason)


def _stand_on_drop(snap: ZeldaSnapshot, reason: str) -> FrameAction | None:
    """Idle on the drop — unless something is close enough to walk into Link.

    Standing on a drop is not instant: live 0x7B (``zhit6`` f=4837-4856) Link
    sat 3 px from a 1-rupee for **twenty** frames before the ROM handed it
    over, and a leever closed 10 px → 8 px and took the heart in frame 4860.
    The drop keeps for hundreds of frames and the wave does not, so inside
    the dodge pad this rung hands the frame back to the layers whose job the
    body is.
    """
    if any(
        chebyshev(int(snap.link_x), int(snap.link_y), int(b.x), int(b.y))
        <= MIN_DODGE_BODY
        for b in overworld_threat_objects(snap)
    ):
        return None
    return FrameAction(nes_idle_action(), reason)


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
    if direction == "UP" and y == 205:
        # 205 is the arrival side of an UP hop and a lattice row Link can
        # strafe on. Excluding it swapped the UP push (205 -> 203) with the
        # align's turn-grid slide (203 -> 205) for 4000f on 0x49 (gathered
        # power-on, walk_pond_l1).
        can_align_x = True
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
        and _y_unaligned(int(snap.link_y), int(align_y), direction, y_tol)
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


def lattice_row(y: int) -> int:
    """The turn-lattice row (y % 8 == 5) nearest ``y``; ties go down-screen."""
    return (int(y) - 1) // 8 * 8 + 5


def _y_unaligned(y: int, align_y: int, direction: str, y_tol: int) -> bool:
    """Whether a push still needs its row fixed first.

    A LEFT/RIGHT press off the lattice slides Link onto the NEAREST row, so
    "within 5 px" of an off-row target is not aligned: 0x37's align_y=140
    stopped the DOWN at 135, RIGHT slid back to 133, and the two swapped
    for 12500 frames (last-heart run 29). Horizontal pushes align until
    Link's own nearest row is the target's.
    """
    if direction in ("LEFT", "RIGHT"):
        return lattice_row(y) != lattice_row(align_y)
    return abs(y - align_y) > y_tol


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
