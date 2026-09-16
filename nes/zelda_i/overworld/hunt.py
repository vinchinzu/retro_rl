"""On-route screen clearing: kill the wave, then bank what it drops.

Closes to wooden-sword reach of every live body in :data:`HUNT_BOX`, and at
full health answers a lined-up body with the sword shot instead
(``zelda_i.beam``) — same damage, same drop, no contact. For
:data:`BEAM_STAND_KINDS` the approach goal becomes the *beam* stand, which is
outside the body's reach rather than inside Link's.
:class:`ShotPolicy` is a *modifier*, never a mode that stands Link still
while a body closes; it also steps out from under unblockable shots.
``prey`` picks the richest body, not the nearest; ``transit_screens`` skip
a chase. Budgets: ``screen_max_frames``, ``target_max_frames``, :data:`HUNT_BOX`.
``kills`` is a slot census; ``kills_counter`` is the ROM forced-drop
counters (``Link_BeHarmed`` zeros them). ``min_hearts`` is
``ram.whole_hearts`` (the ``$066F`` nibble reads one low); below it the
hunt *guards* — blade in the pad, bank a heart — and stops chasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.beam import BeamPolicy, beam_ready
from zelda_i.combat import (
    SWORD_REACH,
    CombatLedger,
    bodies_in_box,
    chebyshev,
    closest_body,
    direction_to_facing,
    facing_to_direction,
    floor_pickups,
    in_sword_hitbox,
    live_enemies,
    manhattan,
    heal_wanted,
    nearest_to,
)
from zelda_i.dungeon.behaviors import EnemyKind, face_toward, kind_for_type
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.dungeon.postmortem import DamageLog, HitEvent
from zelda_i.dungeon.threat import (
    MIN_DODGE_BODY,
    MIN_DODGE_SHOT,
    assess,
    off_line_step,
)
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker, TrackedObject
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.heart_farm import LEAVE_GOALS, FarmOccupancy
from zelda_i.overworld.prey import PreyPolicy, prey_name
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

__all__ = [
    "BEAM_STAND_KINDS",
    "HUNT_BOX",
    "HUNT_LANE_MAX_FRAMES",
    "HUNT_LANE_TOL",
    "HUNT_MIN_HEARTS",
    "HUNT_MUZZLE_ALARM",
    "HUNT_OFF_LINE_MAX_FRAMES",
    "HUNT_SCREEN_MAX_FRAMES",
    "HUNT_STRIKE_SLOT_MAX_FRAMES",
    "HUNT_SETTLE_FRAMES",
    "HUNT_SHIELD_WINDOW",
    "HUNT_SPAWN_WAIT_FRAMES",
    "HUNT_TARGET_MAX_FRAMES",
    "HuntCensus",
    "MUZZLE_PENALTY",
    "STAND_HOLD_PX",
    "PEAHAT_LANDED_SPEED",
    "SHIELD_CLOSING_PAD",
    "SHOT_DWELL_SPEED",
    "ScreenHunter",
    "ShieldPolicy",
    "ShotCensus",
    "ShotPolicy",
    "TargetBook",
    "attackable",
    "hit_cause",
    "hop_exit_goal",
    "hop_lane",
    "perpendicular",
    "sword_stand",
]

HUNT_SCREEN_MAX_FRAMES = 600  # one screen; caps the detour rather than max_frames
HUNT_TARGET_MAX_FRAMES = 180  # one body; closing the box is ~120f at ~1 px/frame
HUNT_SETTLE_FRAMES = 24  # drop lands a frame or two after the body
HUNT_SPAWN_WAIT_FRAMES = 110  # wave is not live the moment the scroll ends
# Whole hearts (``ram.whole_hearts``), not the raw ``$066F`` nibble. Guard, do not chase.
HUNT_MIN_HEARTS = 1
HUNT_LANE_TOL = 6  # walk back to the hop lane; align_and_push has no answer to a bush
HUNT_LANE_MAX_FRAMES = 240
# How far a drop is worth walking to *while the screen still has a wave*.
# Manhattan, from Link. The whole box was the first shape and it priced
# itself: 0x7E is four octorok_fast plus a Zora, and one pass spent 240
# frames on ``hunt_heal`` and 179 on ``hunt_scoop`` crossing it for one heart
# and 2R, taking five of the walk's twelve hits doing it (``pre_l1_anyrow1``).
# Link walks 1 px/frame, so this is ~72 frames of exposure each way. Once the
# screen is empty the ``_hunt`` pickup branch still takes the whole box: with
# nothing alive, distance costs frames and nothing else.
HUNT_PICKUP_RADIUS = 72
# How long one slot may hold the contact-strike rung. A wooden sword takes
# two hits to kill anything on this corridor and ``_a_edge`` spends ~7 frames
# a press, so ~20 presses is already far past "this is not working".
HUNT_STRIKE_SLOT_MAX_FRAMES = 300
# The heal is the shorter of the two budgets on purpose: it runs above the
# beam and the chase, so an unbounded one starves the screen it is on.
HUNT_HEAL_MAX_FRAMES = 120
# Interior box. Scroll lines are x=14/232, y=62/212 (``overworld.common.EDGE_*``).
HUNT_BOX = (32, 214, 76, 198)
# ``$00AC`` slot 0 non-zero for the whole swing; idle is the ROM's A-release edge.
LINK_SLOT = 0
HUNT_SHIELD_WINDOW = 16  # swing pins Link with the shield down; do not A a shot in it
SHIELD_CLOSING_PAD = 48  # only a *closing* body silences the shield
HUNT_OFF_LINE_MAX_FRAMES = 40  # leave a shooter's axis only while there is room to close
HUNT_DESTINATION_FRAMES = 2400  # last hop-table screen has no route left to protect
HUNT_DESTINATION_TARGET_FRAMES = 420
PEAHAT_LANDED_SPEED = 0.35  # flying Peahat: CheckMonsterCollisions only at $0444 == 5
MUZZLE_PENALTY = 32  # extra walk (px) to attack a shooter from a side it is not facing
STAND_HOLD_PX = 12  # hold a sword-stand cell while the body hops inside this pad
# Zora spit dwells ZORA_MUZZLE_DWELL (17) f at ~0 vel; launched is ZORA_SHOT_SPEED (1.75).
SHOT_DWELL_SPEED = 0.5
HUNT_MUZZLE_ALARM = 176  # dwelling muzzle still worth stepping away from. Measured.
# Bodies the full-health sword shot is worth *waiting* for. Their whole attack
# is walking into Link, so his standing 56 px off in their own row costs
# nothing; a shooter's axis is the one place the beam's reach does not pay,
# and keeps ``_off_line``.
BEAM_STAND_KINDS = frozenset({EnemyKind.TEKTITE, EnemyKind.LEEVER})
_SIDE_FACE = {"N": "UP", "S": "DOWN", "E": "RIGHT", "W": "LEFT"}
_FACING_SIDE = {0x08: "N", 0x04: "S", 0x01: "E", 0x02: "W"}  # $0098 facing -> muzzle side


def _box_step(x: int, y: int, direction: str, box: tuple[int, int, int, int]) -> str | None:
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
) -> str | None:
    """Step that opens the angle to a muzzle at ``(mx, my)``, or ``None``.

    Zora aim is quantized at launch; never walk into the muzzle. ``bodies``
    is the same rule :meth:`ShotPolicy.face` keeps: a shot is not a reason to
    walk into an octorok.
    """
    dx, dy = int(mx) - int(lx), int(my) - int(ly)
    # Cross the *major* axis of the bearing. Shared row (dy == 0) -> UP/DOWN.
    if abs(dx) >= abs(dy):
        options = ("DOWN", "UP") if dy <= 0 else ("UP", "DOWN")
    else:
        options = ("RIGHT", "LEFT") if dx <= 0 else ("LEFT", "RIGHT")
    for direction in options:
        if _box_step(lx, ly, direction, box) is None:
            continue
        nx, ny = lx + _STEP[direction][0] * MIN_DODGE_BODY, ly + _STEP[direction][1] * MIN_DODGE_BODY
        if any(
            chebyshev(nx, ny, int(b.x), int(b.y)) < MIN_DODGE_BODY for b in bodies
        ):
            continue
        return direction
    return None


_STEP = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

def _side_cells(obj: ZeldaObject) -> dict[str, tuple[int, int]]:
    ox, oy = int(obj.x), int(obj.y)
    return {
        "E": (ox + SWORD_REACH, oy),
        "W": (ox - SWORD_REACH, oy),
        "S": (ox, oy + SWORD_REACH),
        "N": (ox, oy - SWORD_REACH),
    }


def sword_stand(
    link_x: int, link_y: int, obj: ZeldaObject,
    box: tuple[int, int, int, int] = HUNT_BOX, *,
    avoid_muzzle: bool = False, prefer: str | None = None,
) -> tuple[int, int]:
    """Cell beside ``obj`` from which the wooden sword reaches.

    ``SWORD_REACH`` (20) off the body, outside ``MIN_DODGE_BODY`` (16):
    occupancy-walking onto the sprite is ``Link_BeHarmed``. ``avoid_muzzle``
    adds :data:`MUZZLE_PENALTY` on the side the body faces.
    """
    if not avoid_muzzle:
        face = face_toward(int(link_x), int(link_y), int(obj.x), int(obj.y))
        side = {"RIGHT": "W", "LEFT": "E", "DOWN": "N", "UP": "S"}[face]
        return _clamp(_side_cells(obj)[side], box)
    return _clamp(_side_cells(obj)[muzzle_free_side(link_x, link_y, obj, prefer)], box)


def muzzle_free_side(link_x: int, link_y: int, obj: ZeldaObject, prefer: str | None = None) -> str:
    """Compass side of ``obj`` to attack from: nearest that is not its muzzle."""
    muzzle = _FACING_SIDE.get(int(obj.facing) & 0x0F)
    best, best_cost = "E", 10**9
    for side, (gx, gy) in _side_cells(obj).items():
        cost = abs(gx - int(link_x)) + abs(gy - int(link_y))
        if side == muzzle:
            cost += MUZZLE_PENALTY
        if side == prefer:
            cost -= MUZZLE_PENALTY
        if cost < best_cost:
            best, best_cost = side, cost
    return best


def _clamp(cell: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[int, int]:
    xlo, xhi, ylo, yhi = box
    return (max(xlo, min(xhi, cell[0])), max(ylo, min(yhi, cell[1])))


def link_busy(snap: ZeldaSnapshot) -> bool:
    """True while Link's own slot is mid-animation (sword out, knockback)."""
    first = snap.objects[0] if snap.objects else None
    if first is None or int(first.slot) != LINK_SLOT:
        return False
    return int(first.state) != 0


def hit_cause(event: HitEvent) -> str:
    """Name the thing that took the health: ``octorok_fast_E``, ``rock_E``."""
    if event.type_id is None:
        return "unattributed"
    name = OBJECT_NAMES.get(int(event.type_id), f"unk_{int(event.type_id):#04x}")
    return f"{name}_{event.bearing}"


def _held_face(snap: ZeldaSnapshot) -> str | None:
    try:
        return facing_to_direction(int(snap.facing))
    except ValueError:
        return None


def hop_lane(hop: ScreenHop) -> tuple[str, int] | None:
    """The row or column this hop travels on, for the hunt's walk back."""
    if hop.align_y is not None:
        return ("y", int(hop.align_y))
    if hop.align_x is not None:
        return ("x", int(hop.align_x))
    if hop.y_band is not None:
        lo, hi = hop.y_band
        return ("y", (int(lo) + int(hi)) // 2)
    return None


def hop_exit_goal(hop: ScreenHop) -> tuple[int, int] | None:
    """The scroll-line cell this hop is trying to reach, on its own lane."""
    goal = LEAVE_GOALS.get(hop.direction)
    if goal is None:
        return None
    gx, gy = goal
    lane = hop_lane(hop)
    if lane is None:
        return (gx, gy)
    axis, value = lane
    if axis == "y" and hop.direction in ("LEFT", "RIGHT"):
        return (gx, int(value))
    if axis == "x" and hop.direction in ("UP", "DOWN"):
        return (int(value), gy)
    return (gx, gy)


def attackable(obj: ZeldaObject, track: TrackedObject | None) -> bool:
    """False while the wooden sword cannot hurt this body.

    Peahat: ``CheckMonsterCollisions`` only while ``$0444 == 5`` (landed).
    """
    if kind_for_type(int(obj.type_id)) is not EnemyKind.PEAHAT:
        return True
    return track is None or track.speed < PEAHAT_LANDED_SPEED


@dataclass
class ShotCensus:
    """Frames the shield/duck modifier took, split by what it did with them.

    Accounting, not tuning: the reason ``ShotPolicy``'s own fields can all be
    read as knobs. Reaches the outside through :meth:`ShotPolicy.report`.
    """

    turns: int = 0
    holds: int = 0
    swings_held: int = 0
    ducks: int = 0

    @property
    def frames(self) -> int:
        return self.turns + self.holds + self.swings_held + self.ducks

    def reset(self) -> None:
        self.turns = self.holds = self.swings_held = self.ducks = 0

    def report(self) -> dict[str, Any]:
        return {
            "shield_frames": self.frames,
            "shield_turns": self.turns,
            "shield_holds": self.holds,
            "shield_swings_held": self.swings_held,
            "duck_frames": self.ducks,
        }


@dataclass
class ShotPolicy:
    """What to do about a shot. A modifier, never a mode.

    Speaks only when not closing on a body: :meth:`hold_swing` (already
    faces the shot — A is the only thing dropping the shield) and
    :meth:`face`. :meth:`duck` walks away from a Zora ``0x55`` spit during
    ``ZORA_MUZZLE_DWELL`` (the shield cannot eat it; a velocity tracker
    reports the motionless muzzle as safe).

    Every field here is a knob; what it spent is on :class:`ShotCensus`.
    """

    enabled: bool = True
    window: int = HUNT_SHIELD_WINDOW
    closing_pad: int = SHIELD_CLOSING_PAD
    duck_enabled: bool = True  # own switch so ablating the shield keeps fireball dodge
    muzzle_alarm: int = HUNT_MUZZLE_ALARM
    census: ShotCensus = field(default_factory=ShotCensus)

    @property
    def frames(self) -> int:
        return self.census.frames

    def _shot(self, link: tuple[int, int], tracked: tuple[TrackedObject, ...]) -> TrackedObject | None:
        if not self.enabled:
            return None
        impact = assess(link, tracked, horizon=self.window)
        shot = impact.source
        if shot is None or not shot.blockable:
            return None
        return shot

    def hold_swing(self, snap: ZeldaSnapshot, tracked: tuple[TrackedObject, ...], closing: bool) -> bool:
        """True when pressing A would drop the shield onto an arriving shot."""
        if closing:
            return False
        link = (int(snap.link_x), int(snap.link_y))
        shot = self._shot(link, tracked)
        if shot is None:
            return False
        face = _SIDE_FACE.get(shot.approach_side(*link))
        if face is None or int(snap.facing) != direction_to_facing(face):
            return False  # turning costs the swing and does not block until next frame
        self.census.swings_held += 1
        return True

    def face(
        self, snap: ZeldaSnapshot, tracked: tuple[TrackedObject, ...],
        body: TrackedObject | None, body_pad: int,
    ) -> tuple[str | None, str] | None:
        """``(direction, reason)`` to keep the shield between Link and a shot.

        ``direction`` None means already facing it. Whole return None when a
        body is close enough that standing to block would eat a contact.
        """
        link = (int(snap.link_x), int(snap.link_y))
        if (
            body is not None
            and body_pad <= self.closing_pad
            and body.closing_on(*link)
        ):
            return None
        shot = self._shot(link, tracked)
        if shot is None:
            return None
        face = _SIDE_FACE.get(shot.approach_side(*link))
        if face is None:
            return None
        if int(snap.facing) != direction_to_facing(face):
            self.census.turns += 1
            return (face, "hunt_shield_turn")
        self.census.holds += 1
        return (None, "hunt_shield")

    def duck(
        self, snap: ZeldaSnapshot, tracked: tuple[TrackedObject, ...],
        box: tuple[int, int, int, int], bodies: tuple[ZeldaObject, ...] = (),
    ) -> tuple[str, str] | None:
        """Leave the line of a shot the shield cannot eat.

        Dwelling (speed below :data:`SHOT_DWELL_SPEED`): motionless, so
        ``assess`` scores it safe. Launched inside ``MIN_DODGE_SHOT`` is the
        late case. A surfaced Zora with no shot yet is *not* a reason to
        move — the shot is aimed when it leaves, not when the mouth opens.
        """
        if not self.enabled or not self.duck_enabled:
            return None
        link = (int(snap.link_x), int(snap.link_y))
        shot = self._unblockable(link, tracked)
        if shot is None:
            return None
        step = perpendicular(
            link[0], link[1], int(shot.x), int(shot.y), box, bodies
        )
        if step is None:
            return None
        self.census.ducks += 1
        return (step, "hunt_duck")

    def _unblockable(self, link: tuple[int, int], tracked: tuple[TrackedObject, ...]) -> TrackedObject | None:
        best: TrackedObject | None = None
        best_range = 10**9
        for track in tracked:
            if track.hazard is not HazardClass.PROJECTILE or track.blockable:
                continue
            gap = chebyshev(link[0], link[1], int(track.x), int(track.y))
            if track.speed < SHOT_DWELL_SPEED:
                if gap > self.muzzle_alarm:
                    continue
            elif not assess(link, (track,), horizon=MIN_DODGE_SHOT).imminent:
                continue
            if gap < best_range:
                best, best_range = track, gap
        return best

    def reset(self) -> None:
        self.census.reset()

    def report(self) -> dict[str, Any]:
        return self.census.report()


ShieldPolicy = ShotPolicy  # ablation probes that poke ShieldPolicy still name this class


@dataclass
class TargetBook:
    """Which body the hunt holds, writes off, or declines.

    Re-picking nearest every frame oscillates. A currently unhittable body
    (Peahat in flight) spends no budget. ``prey.worth_chasing`` / ``score``
    pick value, not nearest. Neither stops a swing already in the blade box.
    """

    max_frames: int = HUNT_TARGET_MAX_FRAMES
    prey: PreyPolicy = field(default_factory=PreyPolicy)
    slot: int | None = None
    frames: int = 0
    skipped: set[int] = field(default_factory=set)
    skips: int = 0
    passed: dict[str, int] = field(default_factory=dict)  # declined on value, by type

    def pick(
        self, snap: ZeldaSnapshot, prey: tuple[ZeldaObject, ...],
        tracks: dict[int, TrackedObject], budget_left: int = 10**6,
    ) -> tuple[ZeldaObject | None, str | None]:
        """``(target, note)``. ``target`` None means nothing is worth holding."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        live = tuple(o for o in prey if int(o.slot) not in self.skipped)
        hittable = tuple(o for o in live if attackable(o, tracks.get(int(o.slot))))
        note: str | None = None
        held = next((o for o in hittable if int(o.slot) == self.slot), None)
        if held is not None:
            self.frames += 1
            if self.frames <= self.max_frames:
                return (held, None)
            self.skipped.add(int(held.slot))
            self.skips += 1
            note = f"slot{int(held.slot)}"
            hittable = tuple(o for o in hittable if int(o.slot) != int(held.slot))
        chosen = self._best(
            lx, ly, hittable, int(snap.whole_hearts), int(budget_left)
        )
        if chosen is None:
            self.slot = None
            self.frames = 0
            return (None, note)
        self.slot = int(chosen.slot)
        self.frames = 1
        return (chosen, note)

    def _best(
        self, lx: int, ly: int, hittable: tuple[ZeldaObject, ...],
        hearts: int, budget_left: int,
    ) -> ZeldaObject | None:
        worth = []
        for obj in hittable:
            # Chebyshev gates (contact is a square pad); manhattan orders walks.
            reach = chebyshev(lx, ly, int(obj.x), int(obj.y))
            if self.prey.worth_chasing(
                obj, reach, hearts=hearts, budget_left=budget_left
            ):
                worth.append((obj, manhattan(lx, ly, int(obj.x), int(obj.y))))
            else:
                name = prey_name(int(obj.type_id))
                self.passed[name] = self.passed.get(name, 0) + 1
        if not worth:
            return None
        return max(
            worth, key=lambda pair: (self.prey.score(*pair), -int(pair[0].slot))
        )[0]

    def clear(self) -> None:
        self.slot = None
        self.frames = 0
        self.skipped.clear()

    def release(self) -> None:
        self.slot = None
        self.frames = 0


@dataclass
class HuntCensus:
    """What the hunt spent, and what came of it, screen by screen.

    These counters used to sit on :class:`ScreenHunter` beside its knobs, so
    a tuning field and an accounting field looked the same from outside.
    ``overworld.prey.PreyPolicy`` is frozen with no counters on it and is the
    module in this cluster that reads cleanly; this is that rule applied to
    the hunt. Everything here reaches the outside through
    :meth:`ScreenHunter.report` and :meth:`ScreenHunter.screen_table`.

    ``collect_frames`` and ``off_line_frames`` are *also* per-screen budgets —
    :meth:`ScreenHunter._enter` zeroes them on a scroll, not just on a reset.
    """

    hunt_frames: int = 0
    guard_frames: int = 0
    peel_frames: int = 0
    transit_frames: int = 0
    collect_frames: int = 0
    heal_frames: int = 0
    # Per-screen budgets reset on entry; this one is the run total, so a
    # census can say whether the heal rung ever claimed a frame at all
    # (``ScreenTally.heal_hearts`` says whether it reached the drop).
    heal_frames_total: int = 0
    off_line_frames: int = 0
    release_frames: int = 0  # A-edge idle frames; a swing that never starts is invisible
    screens_cleared: int = 0
    screens_retired: int = 0
    frames_by_screen: dict[int, int] = field(default_factory=dict)
    prey_by_screen: dict[int, int] = field(default_factory=dict)
    hits_by_screen: dict[int, dict[str, int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def spend(self, screen: int) -> None:
        """One frame of chase budget, on the whole hunt and on this screen."""
        self.hunt_frames += 1
        self.frames_by_screen[screen] = self.frames_by_screen.get(screen, 0) + 1

    def saw_prey(self, screen: int, live: int) -> None:
        """Peak, not total: the wave is what the screen cost, not each body."""
        self.prey_by_screen[screen] = max(self.prey_by_screen.get(screen, 0), live)

    def hurt(self, screen: int, cause: str) -> None:
        causes = self.hits_by_screen.setdefault(screen, {})
        causes[cause] = causes.get(cause, 0) + 1

    def hits_by_cause(self) -> dict[str, int]:
        causes: dict[str, int] = {}
        for per_screen in self.hits_by_screen.values():
            for key, n in per_screen.items():
                causes[key] = causes.get(key, 0) + n
        return causes

    def note(self, note: str) -> None:
        self.notes.append(note)

    def note_once(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def reset(self) -> None:
        self.hunt_frames = 0
        self.guard_frames = 0
        self.peel_frames = 0
        self.transit_frames = 0
        self.collect_frames = 0
        self.heal_frames = 0
        self.heal_frames_total = 0
        self.off_line_frames = 0
        self.release_frames = 0
        self.screens_cleared = 0
        self.screens_retired = 0
        self.frames_by_screen.clear()
        self.prey_by_screen.clear()
        self.hits_by_screen.clear()
        self.notes.clear()


@dataclass
class ScreenHunter:
    """Clear the current overworld screen, then hand the frame back.

    ``step`` returns ``None`` when the hunt does not want the frame. Ladder:
    strike a body in the pad or blade box (even retired — ``Link_BeHarmed``
    does not care), else shield, else guard/collect, else hunt.

    The fields here are knobs, budgets and live screen state. What the hunt
    *spent* is on :class:`HuntCensus` (``census``), read through
    :meth:`report` and :meth:`screen_table`.
    """

    screen_max_frames: int = HUNT_SCREEN_MAX_FRAMES
    target_max_frames: int = HUNT_TARGET_MAX_FRAMES
    settle_frames: int = HUNT_SETTLE_FRAMES
    spawn_wait_frames: int = HUNT_SPAWN_WAIT_FRAMES
    min_hearts: int = HUNT_MIN_HEARTS  # whole hearts; the nibble reads one low
    lane_tol: int = HUNT_LANE_TOL
    lane_max_frames: int = HUNT_LANE_MAX_FRAMES
    strike_slot_max_frames: int = HUNT_STRIKE_SLOT_MAX_FRAMES
    collect_max_frames: int = HUNT_LANE_MAX_FRAMES
    # The heal has its own budget, not a share of ``collect_max_frames``: a
    # screen that spent its collect budget on rupees would otherwise walk
    # past the fairy that hands the beam back.
    heal_max_frames: int = HUNT_HEAL_MAX_FRAMES
    pickup_radius: int = HUNT_PICKUP_RADIUS
    shield: bool = True
    shield_window: int = HUNT_SHIELD_WINDOW
    # Sword shot (``zelda_i.beam``). Live at full health only: one wooden chip
    # takes ``$0670`` from ``$FF`` to ``$7F`` and the weapon is gone.
    beam: BeamPolicy = field(default_factory=BeamPolicy)
    beam_stand_kinds: frozenset[EnemyKind] = BEAM_STAND_KINDS
    duck: bool = True  # Zora 0x55; separate from shield so an ablation can split them
    transit_screens: frozenset[int] = frozenset()  # cross, do not chase; still strike/duck/scoop
    reopen_on_enter: bool = False  # lap needs this; one-pass must not reopen. See respawn.
    avoid_firing_lines: bool = True  # sword_stand puts Link on the muzzle axis by construction
    off_line_max_frames: int = HUNT_OFF_LINE_MAX_FRAMES
    destination_frames: int = HUNT_DESTINATION_FRAMES
    destination_target_frames: int = HUNT_DESTINATION_TARGET_FRAMES
    _destination: bool = field(default=False, repr=False)
    _stand_side: str | None = field(default=None, repr=False)
    _stand_goal: tuple[int, int] | None = field(default=None, repr=False)
    _stand_slot: int | None = field(default=None, repr=False)
    _strike_key: tuple[int, int] | None = field(default=None, repr=False)
    _strike_frames: int = field(default=0, repr=False)
    _unkillable: set[tuple[int, int]] = field(default_factory=set, repr=False)
    box: tuple[int, int, int, int] = HUNT_BOX
    prey: PreyPolicy = field(default_factory=PreyPolicy)
    ledger: CombatLedger = field(default_factory=CombatLedger)
    damage: DamageLog = field(default_factory=DamageLog, repr=False)
    census: HuntCensus = field(default_factory=HuntCensus)
    shield_policy: ShotPolicy = field(default_factory=ShotPolicy)
    targets: TargetBook = field(default_factory=TargetBook)
    _pressed: bool = field(default=False, repr=False)
    screen: int = -1
    screen_frames: int = 0
    since_enter: int = 0
    settle: int = 0
    lane_frames: int = 0
    done: set[int] = field(default_factory=set)
    # ``done`` is "stop chasing here"; ``cleared`` is the stronger claim that
    # the wave is actually gone. They are not the same screen set — a budget
    # retire lands in ``done`` with bodies still walking around — and the
    # difference is what a floor drop is worth: on a cleared screen the only
    # thing left to walk into is money, on a retired one it is the wave that
    # just outlasted the budget.
    cleared: set[int] = field(default_factory=set)
    _occ: FarmOccupancy = field(default_factory=FarmOccupancy, repr=False)
    _tracker: ObjectTracker = field(default_factory=ObjectTracker, repr=False)
    _tracked: tuple[TrackedObject, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        self.shield_policy.enabled = bool(self.shield)
        self.shield_policy.window = int(self.shield_window)
        self.shield_policy.duck_enabled = bool(self.duck)
        self.targets.max_frames = int(self.target_max_frames)
        self.targets.prey = self.prey

    @property
    def kills(self) -> int:
        return self.ledger.kills

    @property
    def streak(self) -> int:
        return self.ledger.streak

    def observe(self, snap: ZeldaSnapshot) -> None:
        self.beam.observe(snap, self.box)
        self.ledger.observe(snap)
        self._tracked = self._tracker.observe(snap)
        event = self.damage.observe(
            snap, self._tracked, phase=f"{int(snap.screen):#04x}"
        )
        if event is not None:
            self.census.hurt(int(snap.screen), hit_cause(event))

    def _track(self, obj: ZeldaObject | None) -> TrackedObject | None:
        if obj is None:
            return None
        slot = int(obj.slot)
        return next((t for t in self._tracked if t.slot == slot), None)

    def _tracks_by_slot(self) -> dict[int, TrackedObject]:
        return {t.slot: t for t in self._tracked}

    def step(self, snap: ZeldaSnapshot, frames: int, lane: tuple[str, int] | None = None) -> FrameAction | None:
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE or snap.transitioning:
            return None
        screen = int(snap.screen)
        if screen != self.screen:
            self._enter(screen)
        self.since_enter += 1
        in_box = len(bodies_in_box(snap, self.box))
        if in_box:
            self.census.saw_prey(screen, in_box)
        lx, ly = int(snap.link_x), int(snap.link_y)
        close = closest_body(snap, lx, ly)
        pad = 10**6 if close is None else chebyshev(lx, ly, int(close.x), int(close.y))
        if close is not None and self._at_contact(lx, ly, close, pad):
            if attackable(close, self._track(close)) and self._strike_budget(
                screen, close
            ):
                return self._strike(snap, frames, close, f"hunt_{screen:02x}")
            peel = self._peel(snap, close, f"hunt_{screen:02x}")
            if peel is not None:
                return peel
        block = self._shield_action(snap, close, pad)
        if block is not None:
            return block
        duck = self._duck_action(snap)
        if duck is not None:
            return duck
        heal = self._take_heal(snap, frames)
        if heal is not None:
            return heal
        shot = self._beam_action(snap, screen)
        if shot is not None:
            return shot

        if screen in self.transit_screens:
            self.census.transit_frames += 1
            self._note_once(f"hunt_transit_{screen:02x}")
            return self._collect(snap, frames, lane, heal_only=False)

        if int(snap.whole_hearts) <= self.min_hearts:
            self.census.guard_frames += 1
            self._note_once(f"hunt_guard_{screen:02x}")
            if screen not in self.done:
                self._spend(screen)
                if self.screen_frames > self.screen_max_frames:
                    self._retire(screen, "guard_budget")
            return self._collect(snap, frames, lane, heal_only=True)

        if screen in self.done:
            # Money only once the wave is gone. ``heal_only`` here was the
            # 5R-on-the-floor gap: the kill that empties a screen drops on the
            # frame the screen goes ``done``, and the path's own
            # ``_rupee_scoop`` only reaches 48 px.
            return self._collect(
                snap, frames, lane, heal_only=screen not in self.cleared
            )

        return self._hunt(snap, frames, screen, lane)

    def striking(self, snap: ZeldaSnapshot) -> bool:
        """True when the blade already reaches the nearest body.

        The caller's reactive evader runs first; stepping away from a body
        already in the blade box trades a kill for a frame it gives back.
        """
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE or snap.transitioning:
            return False
        lx, ly = int(snap.link_x), int(snap.link_y)
        body = closest_body(snap, lx, ly)
        if body is None:
            return False
        pad = chebyshev(lx, ly, int(body.x), int(body.y))
        return self._at_contact(lx, ly, body, pad) and attackable(
            body, self._track(body)
        )

    def take_beam(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """The shot, for a caller that owns the frame (the hop, not the chase)."""
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE or snap.transitioning:
            return None
        screen = int(snap.screen)
        if screen != self.screen:
            self._enter(screen)
        return self._beam_action(snap, screen)

    def take_destination(self, snap: ZeldaSnapshot, frames: int) -> FrameAction | None:
        if not self._destination:
            self._destination = True
            self.screen_max_frames = self.destination_frames
            self.targets.max_frames = self.destination_target_frames
            self.targets.skipped.clear()
            self.screen_frames = 0
        return self.step(snap, frames, lane=None)

    def _at_contact(self, lx: int, ly: int, body: ZeldaObject, pad: int) -> bool:
        face = face_toward(lx, ly, int(body.x), int(body.y))
        return pad <= MIN_DODGE_BODY or in_sword_hitbox(
            lx, ly, face, int(body.x), int(body.y)
        )

    def _a_edge(
        self, snap: ZeldaSnapshot, face: str, reason: str, *, closing: bool,
    ) -> FrameAction:
        """One A press then idle release. ButtonsPressed is an edge."""
        self._freeze_occ()
        if link_busy(snap):
            self._pressed = False
            return FrameAction(nes_idle_action(), f"{reason}_recover")
        if self._pressed:
            self._pressed = False
            self.census.release_frames += 1
            return FrameAction(nes_idle_action(), f"{reason}_release")
        if self.shield_policy.hold_swing(snap, self._tracked, closing):
            self._pressed = False
            return FrameAction(nes_idle_action(), f"{reason}_block")
        self._pressed = True
        return FrameAction(nes_action(face, "A"), reason)

    def _strike_budget(self, screen: int, body: ZeldaObject) -> bool:
        """False once one slot has held the contact rung too long.

        The contact strike is the *top* of the hunt ladder and it carried no
        budget at all: not the per-target one (``TargetBook`` is consulted in
        ``_hunt``, three rungs below), not the per-screen one (``_hunt``
        again). A body that will not die at contact therefore owns every
        frame for as long as it stands there, and ``pre_l1_bound1`` is what
        that looks like — 24877 of a 30000 frame timeout on 0x7C, 3072 A
        presses and 19965 ``link_busy`` recovery frames, two hits taken, the
        walk never moving again. Identity is ``(slot, type)`` because a slot
        is reused the moment its occupant dies.

        The frames are also spent against the screen budget, so a screen that
        burns its whole chase on one wedged body still retires and hands the
        hop back.
        """
        key = (int(body.slot), int(body.type_id) & 0xFF)
        if key in self._unkillable:
            return False
        if key != self._strike_key:
            self._strike_key = key
            self._strike_frames = 0
        self._strike_frames += 1
        self._spend(screen)
        if self._strike_frames > self.strike_slot_max_frames:
            self._unkillable.add(key)
            self.census.note(f"hunt_wedged_{screen:02x}_slot{key[0]}")
            return False
        return True

    def _strike(self, snap: ZeldaSnapshot, frames: int, body: ZeldaObject, reason: str) -> FrameAction:
        """Swing at the body, one press per edge."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        face = face_toward(lx, ly, int(body.x), int(body.y))
        held = _held_face(snap)
        if held is not None and in_sword_hitbox(
            lx, ly, held, int(body.x), int(body.y)
        ):
            face = held  # hop that flips dx/dy is not a reason to turn
        track = self._track(body)
        closing = track.closing_on(lx, ly) if track is not None else True
        return self._a_edge(snap, face, f"{reason}_slash", closing=closing)

    def _beam_action(self, snap: ZeldaSnapshot, screen: int) -> FrameAction | None:
        """Fire the full-health shot at a body already standing in a lane.

        Transit screens and any range, including inside sword reach: the
        same press swings. ``prey.skipped`` too: a Zora is never a target.
        """
        if not self.beam.ready(snap):
            return None
        tracks = self._tracks_by_slot()
        bodies = tuple(
            obj
            for obj in bodies_in_box(snap, self.box)
            if attackable(obj, tracks.get(int(obj.slot)))
            and not self.prey.skipped(obj)
        )
        aim = self.beam.aim(snap, bodies, prefer=_held_face(snap))
        if aim is None:
            return None
        face, _body = aim
        reason = f"beam_{screen:02x}"
        act = self._a_edge(snap, face, reason, closing=True)
        if act.reason == reason:
            self.beam.press()
        return act

    def _beam_stand_goal(
        self, snap: ZeldaSnapshot, target: ZeldaObject
    ) -> tuple[int, int] | None:
        """Where to wait for a body the shot can kill before it arrives.

        Only for :data:`BEAM_STAND_KINDS` — a tektite's whole attack is the
        hop that lands on Link, and standing 56 px off in its own row takes
        it for nothing. Shooters keep ``_off_line``: their axis is the one
        place the beam's reach does not pay for. Capped, because a lane can
        be walled and a shot that never lands must not hold the chase.
        """
        if not self.beam.enabled or not beam_ready(snap):
            self.beam.enter()
            return None
        if kind_for_type(int(target.type_id)) not in self.beam_stand_kinds:
            return None
        return self.beam.stand(snap, target, self.box)

    def _peel(self, snap: ZeldaSnapshot, body: ZeldaObject, reason: str) -> FrameAction | None:
        lx, ly = int(snap.link_x), int(snap.link_y)
        dx, dy = int(body.x) - lx, int(body.y) - ly
        away = ("LEFT" if dx > 0 else "RIGHT", "UP" if dy > 0 else "DOWN")
        if abs(dy) > abs(dx):
            away = away[::-1]
        for direction in away:
            if _box_step(lx, ly, direction, self.box) is not None:
                self.census.peel_frames += 1
                self._freeze_occ()
                return FrameAction(nes_action(direction), f"{reason}_peel")
        return None

    def _shield_action(self, snap: ZeldaSnapshot, body: ZeldaObject | None, pad: int) -> FrameAction | None:
        verdict = self.shield_policy.face(
            snap, self._tracked, self._track(body), pad
        )
        if verdict is None:
            return None
        face, reason = verdict
        self._freeze_occ()
        if face is None:
            return FrameAction(nes_idle_action(), reason)
        return FrameAction(nes_action(face), reason)

    def _duck_action(self, snap: ZeldaSnapshot) -> FrameAction | None:
        verdict = self.shield_policy.duck(
            snap, self._tracked, self.box, live_enemies(snap)
        )
        if verdict is None:
            return None
        direction, reason = verdict
        self._freeze_occ()
        return FrameAction(nes_action(direction), reason)

    def _approach(
        self, snap: ZeldaSnapshot, frames: int, target: ZeldaObject,
        close: ZeldaObject, pad: int, reason: str,
    ) -> FrameAction:
        lx, ly = int(snap.link_x), int(snap.link_y)
        if pad <= SWORD_REACH:
            self._freeze_occ()
            cx, cy = int(close.x), int(close.y)
            held = _held_face(snap)
            if held is not None and in_sword_hitbox(lx, ly, held, cx, cy):
                return self._strike(snap, frames, close, reason)
            dx, dy = cx - lx, cy - ly
            if abs(dx) >= abs(dy) and dy != 0:
                align = "DOWN" if dy > 0 else "UP"
            elif abs(dy) > abs(dx) and dx != 0:
                align = "RIGHT" if dx > 0 else "LEFT"
            else:
                align = face_toward(lx, ly, cx, cy)
            if _box_step(lx, ly, align, self.box) is not None:
                self._pressed = False
                return FrameAction(nes_action(align), f"{reason}_align")
            return self._strike(snap, frames, close, reason)  # same A-edge as _strike
        stand = self._beam_stand_goal(snap, target)
        if stand is not None:
            goal = self._hold_stand(int(target.slot), stand)
            return self._occ.walk(snap, frames, goal, f"{reason}_beam", slash=False)
        off = self._off_line(lx, ly, pad)
        if off is not None:
            return FrameAction(nes_action(off), f"{reason}_offline")
        if self.avoid_firing_lines:
            self._stand_side = muzzle_free_side(lx, ly, target, self._stand_side)
            goal = sword_stand(
                lx, ly, target, self.box, avoid_muzzle=True, prefer=self._stand_side
            )
        else:
            goal = sword_stand(lx, ly, target, self.box)
        goal = self._hold_stand(int(target.slot), goal)
        return self._occ.walk(snap, frames, goal, reason, slash=False)

    def _hold_stand(self, slot: int, goal: tuple[int, int]) -> tuple[int, int]:
        if (
            self._stand_goal is not None
            and self._stand_slot == slot
            and manhattan(goal[0], goal[1], *self._stand_goal) <= STAND_HOLD_PX
        ):
            return self._stand_goal
        self._stand_goal = goal
        self._stand_slot = slot
        return goal

    def _off_line(self, lx: int, ly: int, pad: int) -> str | None:
        if not self.avoid_firing_lines:
            return None
        if pad <= SWORD_REACH:
            return None
        if self.census.off_line_frames >= self.off_line_max_frames:
            return None
        bodies = tuple(t for t in self._tracked if t.hazard is HazardClass.BODY)
        step = off_line_step((lx, ly), bodies, bounds=self.box)
        if step is None or _box_step(lx, ly, step, self.box) is None:
            return None
        self.census.off_line_frames += 1
        self._freeze_occ()
        return step

    def _hunt(
        self, snap: ZeldaSnapshot, frames: int, screen: int,
        lane: tuple[str, int] | None,
    ) -> FrameAction | None:
        prey = bodies_in_box(snap, self.box)
        if prey:
            self.settle = 0
            self._spend(screen)
            if self.screen_frames > self.screen_max_frames:
                self._retire(screen, "budget")
                return self._collect(snap, frames, lane, heal_only=True)
            held = self.targets.slot
            target, note = self.targets.pick(
                snap,
                prey,
                self._tracks_by_slot(),
                budget_left=self.screen_max_frames - self.screen_frames,
            )
            if self.targets.slot != held:
                self._stand_side = None
                self._stand_goal = None
                self._stand_slot = None
            if note is not None:
                self.census.note(f"hunt_skip_{screen:02x}_{note}")
            if target is None:
                return self._collect(snap, frames, lane, heal_only=False)
            lx, ly = int(snap.link_x), int(snap.link_y)
            close = closest_body(snap, lx, ly) or target
            pad = chebyshev(lx, ly, int(close.x), int(close.y))
            return self._approach(
                snap, frames, target, close, pad, f"hunt_{screen:02x}"
            )

        pickup = nearest_to(int(snap.link_x), int(snap.link_y), floor_pickups(snap, self.box))
        if pickup is not None:
            self.settle = 0
            self.targets.release()
            self._spend(screen)
            if self.screen_frames > self.screen_max_frames:
                self._retire(screen, "budget")
                return None
            return self._occ.walk(
                snap, frames, (int(pickup.x), int(pickup.y)), "hunt_drop", slash=False
            )

        self.settle += 1
        if (
            self.settle >= self.settle_frames
            and self.since_enter >= self.spawn_wait_frames
        ):
            self._clear(screen)
        return self._lane_return(snap, frames, lane)

    def _near_drop(
        self, snap: ZeldaSnapshot, lx: int, ly: int, *, heal_only: bool
    ) -> ZeldaObject | None:
        """Nearest drop worth walking to with a wave still on the screen.

        ``pickup_radius`` is the whole difference from ``floor_pickups`` +
        ``nearest_to``: those answer "is it on the screen", which on a five
        body screen is a licence to walk the long way across it.
        """
        drop = nearest_to(lx, ly, floor_pickups(snap, self.box, heal_only=heal_only))
        if drop is None:
            return None
        if manhattan(lx, ly, int(drop.x), int(drop.y)) > self.pickup_radius:
            return None
        return drop

    def _take_heal(
        self, snap: ZeldaSnapshot, frames: int
    ) -> FrameAction | None:
        """Walk onto a heart or fairy on the floor, above the shot.

        Sits above ``_beam_action`` on purpose. The beam is a full-health
        weapon — one wooden chip takes ``$0670`` from ``$FF`` to ``$7F`` and
        ``MakeSwordShot`` stops firing — so on every frame this branch can
        claim, the beam below it is already dead. Below the strike and the
        shield, because a heart is not worth walking through a body for.

        The live case is 0x7B: a heart *and* a fairy on the floor, ``$0670``
        already chipped, ``heal_hearts`` 0.0 for the screen, and the walk
        entered 0x7C in guard at one heart and died there (``pre_l1_beam4``,
        reproduced in ``pre_l1_shot2``). ``_collect`` could have taken them
        and never got the frame: it is the *last* rung, under the chase.
        """
        if not heal_wanted(snap):
            return None
        if self.census.heal_frames >= self.heal_max_frames:
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        drop = self._near_drop(snap, lx, ly, heal_only=True)
        if drop is None:
            return None
        self.census.heal_frames += 1
        self.census.heal_frames_total += 1
        return self._occ.walk(
            snap, frames, (int(drop.x), int(drop.y)), "hunt_heal", slash=False
        )

    def _collect(
        self, snap: ZeldaSnapshot, frames: int, lane: tuple[str, int] | None, *,
        heal_only: bool,
    ) -> FrameAction | None:
        if self.census.collect_frames < self.collect_max_frames:
            lx, ly = int(snap.link_x), int(snap.link_y)
            drop = self._near_drop(snap, lx, ly, heal_only=heal_only)
            if drop is not None:
                self.census.collect_frames += 1
                return self._occ.walk(
                    snap, frames, (int(drop.x), int(drop.y)), "hunt_scoop", slash=False
                )
        return self._lane_return(snap, frames, lane)

    def _lane_return(
        self, snap: ZeldaSnapshot, frames: int, lane: tuple[str, int] | None,
    ) -> FrameAction | None:
        if lane is None or self.screen_frames <= 0:
            return None
        axis, value = lane
        lx, ly = int(snap.link_x), int(snap.link_y)
        off = abs(ly - int(value)) if axis == "y" else abs(lx - int(value))
        if off <= self.lane_tol:
            self.lane_frames = 0
            return None
        self.lane_frames += 1
        if self.lane_frames > self.lane_max_frames:
            return None
        goal = (lx, int(value)) if axis == "y" else (int(value), ly)
        act = self._occ.walk(snap, frames, goal, "hunt_lane", slash=True)
        return None if act.reason == "occupancy_stand" else act

    def escape_for(self, snap: ZeldaSnapshot, frames: int, hop: ScreenHop, reason: str) -> FrameAction | None:
        goal = hop_exit_goal(hop)
        return None if goal is None else self.escape(snap, frames, goal, reason)

    def escape(self, snap: ZeldaSnapshot, frames: int, goal: tuple[int, int], reason: str) -> FrameAction | None:
        """Walk toward ``goal`` on the grid this screen's chase learned.

        Stall branch only: ``align_and_push`` holds one direction and
        ``unstick_wiggle`` waits forever once spent.
        """
        act = self._occ.walk(snap, frames, goal, reason, slash=True)
        return None if act.reason == "occupancy_stand" else act

    def _spend(self, screen: int) -> None:
        self.screen_frames += 1
        self.census.spend(screen)

    def _freeze_occ(self) -> None:
        walker = self._occ.walker
        if walker is not None:
            walker.last_dir = None
            walker.last_xy = None

    def _note_once(self, note: str) -> None:
        self.census.note_once(note)

    def _enter(self, screen: int) -> None:
        if self.reopen_on_enter:
            self.done.discard(screen)
            self.cleared.discard(screen)
        self.screen = screen
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self._strike_key = None
        self._strike_frames = 0
        self._unkillable.clear()
        self.census.collect_frames = 0
        self.census.heal_frames = 0
        self.census.off_line_frames = 0
        self._pressed = False
        self.beam.enter()
        self.targets.clear()
        self._stand_side = None
        self._stand_goal = None
        self._stand_slot = None
        self._occ.reset()

    def _clear(self, screen: int) -> None:
        self.done.add(screen)
        self.cleared.add(screen)
        self.census.screens_cleared += 1
        self.census.note(f"hunt_clear_{screen:02x}")

    def _retire(self, screen: int, why: str) -> None:
        self.done.add(screen)
        self.census.screens_retired += 1
        self.census.note(f"hunt_{why}_{screen:02x}")

    def reset(self) -> None:
        self.ledger.reset()
        self.damage = DamageLog()
        self.census.reset()
        self.shield_policy.reset()
        self.targets.clear()
        self.targets.skips = 0
        self.targets.passed.clear()
        self.beam.reset()
        self._pressed = False
        self.screen = -1
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self.done.clear()
        self.cleared.clear()
        self._strike_key = None
        self._strike_frames = 0
        self._unkillable.clear()
        self._destination = False
        self._stand_side = None
        self._stand_goal = None
        self._stand_slot = None
        self._tracked = ()
        self._tracker = ObjectTracker()
        self._occ.reset()

    def screen_table(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.ledger.screen_rows():
            screen = int(row["screen"], 16)
            rows.append(
                {
                    **row,
                    "hunt_frames": self.census.frames_by_screen.get(screen, 0),
                    "peak_prey": self.census.prey_by_screen.get(screen, 0),
                    "beam_fired": self.beam.by_screen.get(screen, 0),
                    "cleared": screen in self.cleared,
                    "retired": screen in self.done and screen not in self.cleared,
                    "transit": screen in self.transit_screens,
                    "hits_by_cause": dict(self.census.hits_by_screen.get(screen, {})),
                }
            )
        return rows

    def report(self) -> dict[str, Any]:
        census = self.census
        return {
            **self.ledger.report(),
            **self.shield_policy.report(),
            **self.beam.report(),
            "hits_by_cause": census.hits_by_cause(),
            "screens": self.screen_table(),
            "hunt_frames": census.hunt_frames,
            "guard_frames": census.guard_frames,
            "peel_frames": census.peel_frames,
            "transit_frames": census.transit_frames,
            "off_line_frames": census.off_line_frames,
            "heal_frames": census.heal_frames_total,
            "release_frames": census.release_frames,
            "target_skips": self.targets.skips,
            "prey_passed": dict(self.targets.passed),
            "transit_screens": sorted(self.transit_screens),
            "screens_cleared": census.screens_cleared,
            "screens_retired": census.screens_retired,
            "screens_done": sorted(self.done),
            "hunt_frames_by_screen": {
                f"{k:#04x}": v for k, v in sorted(census.frames_by_screen.items())
            },
            "peak_prey_by_screen": {
                f"{k:#04x}": v for k, v in sorted(census.prey_by_screen.items())
            },
            "occupancy_misses": self._occ.misses,
            "notes": list(census.notes),
        }
