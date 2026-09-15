"""On-route screen clearing: kill the wave, then bank what it drops.

The pre-L1 errand is money. ``OverworldPathController`` crosses a screen on
one lane and only scoops a drop that lands beside it, so the wave two lanes
north is never fought and never pays. This module is the opposite policy:
close to wooden-sword reach of every live body in the interior box, kill it,
and walk onto what it leaves behind. Four collaborators, one rule each:
:class:`CombatLedger` (the census), :class:`ShieldPolicy` (blockable shots, as
a *modifier* — it never stands Link still while a body closes on him),
:class:`TargetBook` (which body is held, written off, or unhittable now), and
:class:`ScreenHunter` (screen lifecycle, strike ladder, lane return).

Three budgets keep it off the route: ``screen_max_frames``, ``target_max_frames``
and the interior box :data:`HUNT_BOX` (a chase that walks a scroll line changes
screen under the hop table). Kills are counted two ways because neither is free
of doubt: ``kills`` is a slot census, ``kills_counter`` the ROM's own
forced-drop counters, which ``Link_BeHarmed`` zeros on collision.

**Low health is not a reason to stop fighting.** ``$066F``'s low nibble is
whole hearts *minus one* (``ram.whole_hearts``), so the first gate compared
``filled_hearts <= 1`` and retired the screen at two of three hearts: the
2026-09-15 baseline gave up on ``0x58``, ``0x59`` and ``0x49`` after one chip
hit and the hop — which has no combat at all — then walked Link into two
octoroks at 9 px. Below ``min_hearts`` the hunt *guards*: it answers a body in
the pad with the blade and banks a heart off the floor, it just stops chasing.
Numbers and the per-run ledger live in ``docs/PRE_L1.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import (
    SWORD_REACH,
    CombatLedger,
    bodies_in_box,
    chebyshev,
    closest_body,
    direction_to_facing,
    floor_pickups,
    in_sword_hitbox,
    nearest_to,
)
from zelda_i.dungeon.behaviors import EnemyKind, face_toward, kind_for_type
from zelda_i.dungeon.ids import OBJECT_NAMES
from zelda_i.dungeon.postmortem import DamageLog, HitEvent
from zelda_i.dungeon.threat import MIN_DODGE_BODY, assess, off_line_step
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker, TrackedObject
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.heart_farm import LEAVE_GOALS, FarmOccupancy
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

__all__ = [
    "HUNT_BOX",
    "HUNT_LANE_MAX_FRAMES",
    "HUNT_LANE_TOL",
    "HUNT_MIN_HEARTS",
    "HUNT_OFF_LINE_MAX_FRAMES",
    "HUNT_SCREEN_MAX_FRAMES",
    "HUNT_SETTLE_FRAMES",
    "HUNT_SHIELD_WINDOW",
    "HUNT_SPAWN_WAIT_FRAMES",
    "HUNT_TARGET_MAX_FRAMES",
    "MUZZLE_PENALTY",
    "PEAHAT_LANDED_SPEED",
    "SHIELD_CLOSING_PAD",
    "ScreenHunter",
    "ShieldPolicy",
    "TargetBook",
    "attackable",
    "hit_cause",
    "hop_exit_goal",
    "hop_lane",
    "sword_stand",
]

# One screen's worth of fighting. The unhunted 0x77 -> 0x4A walk is 1924
# frames across seven screens; this caps the detour rather than ``max_frames``.
HUNT_SCREEN_MAX_FRAMES = 600
# One body. An overworld octorok dies in 1-2 wooden swings once Link closes,
# and closing across the box is ~120 frames at ~1 px/frame.
HUNT_TARGET_MAX_FRAMES = 180
# The drop lands a frame or two after the body, so do not declare the screen
# clear on the first empty frame.
HUNT_SETTLE_FRAMES = 24
# A wave does not exist the moment the scroll ends (a dungeon room's settle
# spawn is 80-100f). These frames go to the hop, so the wait is free unless
# Link leaves first — and a screen he leaves stays un-retired, which is why.
HUNT_SPAWN_WAIT_FRAMES = 110
# Whole hearts (``ram.whole_hearts``), not the raw ``$066F`` nibble. At or
# below this the hunt guards instead of chasing: the route still has to reach
# L1, but a screen Link walks across un-fought hits him just as hard.
HUNT_MIN_HEARTS = 1
# Walking back to the hop's lane after a fight. ``align_and_push`` holds one
# direction and has no answer to a bush: the first live hunt left Link at
# 0x49 (56,125) against one and burned 27,000 frames on ``unstick_wait``. The
# hunt took him off the lane, so the hunt walks him back.
HUNT_LANE_TOL = 6
HUNT_LANE_MAX_FRAMES = 240
# Interior box (xlo, xhi, ylo, yhi). The scroll lines are x=14/232, y=62/212
# (``overworld.common.EDGE_*``); this keeps a whole chase off them.
HUNT_BOX = (32, 214, 76, 198)
# Link's own object state (``$00AC`` slot 0) is non-zero for the whole sword
# animation, so the hunt presses A the frame the blade box fills and idles
# only while the swing runs. The blind ``frames % 8 < 3`` cadence left up to
# five idle frames per swing with a body closing ~1 px a frame (0x49 f=3653:
# ten frames standing while slot 4 walked 16 -> 9). The idle is also the
# release edge the ROM needs before the next swing can start.
LINK_SLOT = 0
# A wooden swing pins Link for about a dozen frames with the shield down, so
# a shot that lands inside that window must not be answered with A.
HUNT_SHIELD_WINDOW = 16
# Only a *closing* body silences the shield. A stationary shooter 30 px away
# is not a reason to eat its rock, and the flat 36 px pad this replaced
# silenced the shield on every frame of 0x58 f=2058-2068 (nearest body 26-35
# px, never moving toward Link) while the rock crossed his row.
SHIELD_CLOSING_PAD = 48
# Stepping off a shooter's axis is only worth it while there is room to close
# again afterwards; past this many frames on one screen, close and swing.
HUNT_OFF_LINE_MAX_FRAMES = 40
# The screen a walk ends on has no route left to protect, so it gets its own
# budget. On the 600/180 defaults the pre-L1 walk killed one of 0x4A's six
# blue tektites and skipped two (2026-09-15): those six bodies are the richest
# on the corridor, 0.891 R/kill against row 0's 0.156. Tektites hop, so they
# also need longer than a walking octorok.
HUNT_DESTINATION_FRAMES = 2400
HUNT_DESTINATION_TARGET_FRAMES = 420
# A Peahat is invulnerable while it flies (``KIND_POLICY``: the ROM runs
# CheckMonsterCollisions only at ``Flyer_ObjFlyingState $0444 == 5``), and the
# flying state is not in the snapshot. A landed one does not move, so the
# tracker answers what the type byte cannot.
PEAHAT_LANDED_SPEED = 0.35
# Extra walk (px) the hunt will pay to attack a shooter from a side it is not
# facing. Roughly one body length: worth a rock, not worth a lap of the screen.
MUZZLE_PENALTY = 32
# ``TrackedObject.approach_side`` names the side a shot was fired from; Link
# blocks by facing it.
_SIDE_FACE = {"N": "UP", "S": "DOWN", "E": "RIGHT", "W": "LEFT"}
# ``$0098``-style facing byte -> the compass side that body's muzzle points at.
_FACING_SIDE = {0x08: "N", 0x04: "S", 0x01: "E", 0x02: "W"}



# ---------------------------------------------------------------------- #
# Pure geometry
# ---------------------------------------------------------------------- #


def _box_step(
    x: int, y: int, direction: str, box: tuple[int, int, int, int]
) -> str | None:
    """``direction`` if one step stays in the hunt box, else None.

    A peel that walks a scroll line changes screen under the hop table.
    """
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


def link_busy(snap: ZeldaSnapshot) -> bool:
    """True while Link's own slot is mid-animation (sword out, knockback).

    ``$00AC`` slot 0 is non-zero for the whole swing, which is what the hunt
    needs instead of a blind ``frames % 8`` cadence: the cadence idled up to
    five frames per swing while a body closed ~1 px a frame.
    """
    first = snap.objects[0] if snap.objects else None
    if first is None or int(first.slot) != LINK_SLOT:
        return False
    return int(first.state) != 0


def hit_cause(event: HitEvent) -> str:
    """Name the thing that took the health: ``octorok_fast_E``, ``rock_E``.

    ``postmortem`` keys causes by raw ObjType, which reads the same for a
    body and the rock it spat. On this corridor half the streak resets are
    shots (2026-09-15 ``contact1``/``contact7``), so the census that decides
    whether to fix the chase or the dodge has to spell the difference out.
    """
    if event.type_id is None:
        return "unattributed"
    name = OBJECT_NAMES.get(int(event.type_id), f"unk_{int(event.type_id):#04x}")
    return f"{name}_{event.bearing}"


def _side_cells(obj: ZeldaObject) -> dict[str, tuple[int, int]]:
    ox, oy = int(obj.x), int(obj.y)
    return {
        "E": (ox + SWORD_REACH, oy),
        "W": (ox - SWORD_REACH, oy),
        "S": (ox, oy + SWORD_REACH),
        "N": (ox, oy - SWORD_REACH),
    }


def sword_stand(
    link_x: int,
    link_y: int,
    obj: ZeldaObject,
    box: tuple[int, int, int, int] = HUNT_BOX,
    *,
    avoid_muzzle: bool = False,
    prefer: str | None = None,
) -> tuple[int, int]:
    """Cell beside ``obj`` from which the wooden sword reaches.

    ``SWORD_REACH`` (20) off the body, outside ``MIN_DODGE_BODY`` (16):
    occupancy-walking onto the sprite is ``Link_BeHarmed``. With
    ``avoid_muzzle`` the side the body *faces* costs an extra
    :data:`MUZZLE_PENALTY` px — an octorok fires down the axis it faces, so
    the naive "Link's own side" cell is the muzzle on every approach it
    notices. ``prefer`` discounts the caller's held side so the choice does
    not flip each time the body turns.
    """
    if not avoid_muzzle:
        face = face_toward(int(link_x), int(link_y), int(obj.x), int(obj.y))
        side = {"RIGHT": "W", "LEFT": "E", "DOWN": "N", "UP": "S"}[face]
        return _clamp(_side_cells(obj)[side], box)
    return _clamp(_side_cells(obj)[muzzle_free_side(link_x, link_y, obj, prefer)], box)


def muzzle_free_side(
    link_x: int, link_y: int, obj: ZeldaObject, prefer: str | None = None
) -> str:
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


def _clamp(
    cell: tuple[int, int], box: tuple[int, int, int, int]
) -> tuple[int, int]:
    xlo, xhi, ylo, yhi = box
    return (max(xlo, min(xhi, cell[0])), max(ylo, min(yhi, cell[1])))


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
    """False while the wooden sword cannot hurt this body at all.

    Only the Peahat needs it today: the ROM runs ``CheckMonsterCollisions`` on
    one only while ``$0444 == 5`` (landed) and ``ZeldaObject.state`` is
    ``$00AC``, not that flying state. Motion is the fallback, as in
    ``tracking._hazard_class``: a landed Peahat is still.
    """
    if kind_for_type(int(obj.type_id)) is not EnemyKind.PEAHAT:
        return True
    return track is None or track.speed < PEAHAT_LANDED_SPEED


# ---------------------------------------------------------------------- #
# Shield
# ---------------------------------------------------------------------- #


@dataclass
class ShieldPolicy:
    """What to do about a blockable shot. A modifier, never a mode.

    The small shield eats an octorok rock for free while Link faces it and is
    not attacking, and half the contacts on this corridor are rocks. The first
    cut answered every rock inside the window by standing still and killed
    Link three times on ``0x49`` (mode 17) because octoroks walked the last
    9 px while he stood. So it speaks in two places only, neither of them a
    frame the hunt would have spent closing on a body: :meth:`hold_swing`
    (Link already faces the shot and the nearest body is not closing, so the A
    press is the only thing dropping the shield — ``0x58`` ``f=2056``) and
    :meth:`face` (no body closing at all — ``0x58`` ``f=2275``).
    """

    enabled: bool = True
    window: int = HUNT_SHIELD_WINDOW
    closing_pad: int = SHIELD_CLOSING_PAD
    turns: int = 0
    holds: int = 0
    swings_held: int = 0

    @property
    def frames(self) -> int:
        return self.turns + self.holds + self.swings_held

    def _shot(
        self, link: tuple[int, int], tracked: tuple[TrackedObject, ...]
    ) -> TrackedObject | None:
        if not self.enabled:
            return None
        impact = assess(link, tracked, horizon=self.window)
        shot = impact.source
        if shot is None or not shot.blockable:
            return None
        return shot

    def hold_swing(
        self,
        snap: ZeldaSnapshot,
        tracked: tuple[TrackedObject, ...],
        closing: bool,
    ) -> bool:
        """True when pressing A would drop the shield onto an arriving shot.

        ``closing`` is the caller's read of whether the body it is about to
        swing at is walking into Link; when it is, the sword wins.
        """
        if closing:
            return False
        link = (int(snap.link_x), int(snap.link_y))
        shot = self._shot(link, tracked)
        if shot is None:
            return False
        face = _SIDE_FACE.get(shot.approach_side(*link))
        if face is None or int(snap.facing) != direction_to_facing(face):
            # Turning costs the swing *and* does not block until next frame.
            return False
        self.swings_held += 1
        return True

    def face(
        self,
        snap: ZeldaSnapshot,
        tracked: tuple[TrackedObject, ...],
        body: TrackedObject | None,
        body_pad: int,
    ) -> tuple[str | None, str] | None:
        """``(direction, reason)`` to keep the shield between Link and a shot.

        ``direction`` None means Link already faces it and should stand;
        ``None`` means nothing to block, or a body close enough that standing
        to block is how the first three shield walks died.
        """
        link = (int(snap.link_x), int(snap.link_y))
        if (
            body is not None
            and body_pad <= self.closing_pad
            and body.closing_on(*link)
        ):
            # Standing to block while a body walks the last pixels is how
            # ``contact4`` died. A body that is not closing can wait.
            return None
        shot = self._shot(link, tracked)
        if shot is None:
            return None
        face = _SIDE_FACE.get(shot.approach_side(*link))
        if face is None:
            return None
        if int(snap.facing) != direction_to_facing(face):
            # One frame to bring the shield round. It also walks Link 1 px
            # into the shot, which is the cheap half of the trade.
            self.turns += 1
            return (face, "hunt_shield_turn")
        self.holds += 1
        return (None, "hunt_shield")

    def reset(self) -> None:
        self.turns = self.holds = self.swings_held = 0

    def report(self) -> dict[str, Any]:
        return {
            "shield_frames": self.frames,
            "shield_turns": self.turns,
            "shield_holds": self.holds,
            "shield_swings_held": self.swings_held,
        }


# ---------------------------------------------------------------------- #
# Targeting
# ---------------------------------------------------------------------- #


@dataclass
class TargetBook:
    """Which body the hunt holds, and which it has written off.

    Re-picking the nearest body every frame makes Link oscillate between two
    equidistant octoroks; the budget is what stops a body the wooden sword
    cannot reach from owning the screen. A body that is *currently* unhittable
    (a Peahat in flight) spends no budget — it is passed over and picked up
    again when it lands.
    """

    max_frames: int = HUNT_TARGET_MAX_FRAMES
    slot: int | None = None
    frames: int = 0
    skipped: set[int] = field(default_factory=set)
    skips: int = 0

    def pick(
        self,
        snap: ZeldaSnapshot,
        prey: tuple[ZeldaObject, ...],
        tracks: dict[int, TrackedObject],
    ) -> tuple[ZeldaObject | None, str | None]:
        """``(target, note)``. ``target`` None means nothing is worth holding."""
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
        chosen = nearest_to(int(snap.link_x), int(snap.link_y), hittable)
        if chosen is None:
            self.slot = None
            self.frames = 0
            return (None, note)
        self.slot = int(chosen.slot)
        self.frames = 1
        return (chosen, note)

    def clear(self) -> None:
        self.slot = None
        self.frames = 0
        self.skipped.clear()

    def release(self) -> None:
        self.slot = None
        self.frames = 0


# ---------------------------------------------------------------------- #
# The hunt
# ---------------------------------------------------------------------- #


@dataclass
class ScreenHunter:
    """Clear the current overworld screen, then hand the frame back.

    ``step`` returns ``None`` on every frame the hunt does not want, so the
    caller's hop rules keep driving. It never presses a direction that leaves
    the screen and never writes RAM. The ladder, every frame: strike a body in
    the pad or blade box (even on a retired screen and at one heart —
    ``Link_BeHarmed`` does not care that the hunt gave up), else shield, else
    guard/collect on a done screen, else hunt.
    """

    screen_max_frames: int = HUNT_SCREEN_MAX_FRAMES
    target_max_frames: int = HUNT_TARGET_MAX_FRAMES
    settle_frames: int = HUNT_SETTLE_FRAMES
    spawn_wait_frames: int = HUNT_SPAWN_WAIT_FRAMES
    # Whole hearts. See the module docstring: the nibble reads one low.
    min_hearts: int = HUNT_MIN_HEARTS
    lane_tol: int = HUNT_LANE_TOL
    lane_max_frames: int = HUNT_LANE_MAX_FRAMES
    collect_max_frames: int = HUNT_LANE_MAX_FRAMES
    shield: bool = True
    shield_window: int = HUNT_SHIELD_WINDOW
    # Pre-emptive: leave a shooter's axis on the way in, and attack from a
    # side it is not facing. ``sword_stand`` puts Link on the muzzle axis by
    # construction, which is what 0x58 f=2056 and f=2069 both measured.
    avoid_firing_lines: bool = True
    off_line_max_frames: int = HUNT_OFF_LINE_MAX_FRAMES
    destination_frames: int = HUNT_DESTINATION_FRAMES
    destination_target_frames: int = HUNT_DESTINATION_TARGET_FRAMES
    _destination: bool = field(default=False, repr=False)
    _stand_side: str | None = field(default=None, repr=False)
    box: tuple[int, int, int, int] = HUNT_BOX

    ledger: CombatLedger = field(default_factory=CombatLedger)
    # Magnitude and *cause* per screen. The ledger watches the health
    # bytes; only the tracker knows which body or shot was there a frame
    # before the knockback moved Link away from it.
    damage: DamageLog = field(default_factory=DamageLog, repr=False)
    hits_by_screen: dict[int, dict[str, int]] = field(default_factory=dict)
    shield_policy: ShieldPolicy = field(default_factory=ShieldPolicy)
    targets: TargetBook = field(default_factory=TargetBook)

    hunt_frames: int = 0
    guard_frames: int = 0
    peel_frames: int = 0
    collect_frames: int = 0
    off_line_frames: int = 0
    screens_cleared: int = 0
    screens_retired: int = 0
    frames_by_screen: dict[int, int] = field(default_factory=dict)
    # Peak prey *inside the box*, against the ledger's peak live *anywhere*.
    # A screen with live bodies and no prey was never the hunt's to fight.
    prey_by_screen: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    screen: int = -1
    screen_frames: int = 0
    since_enter: int = 0
    settle: int = 0
    lane_frames: int = 0
    done: set[int] = field(default_factory=set)
    _occ: FarmOccupancy = field(default_factory=FarmOccupancy, repr=False)
    _tracker: ObjectTracker = field(default_factory=ObjectTracker, repr=False)
    _tracked: tuple[TrackedObject, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        self.shield_policy.enabled = bool(self.shield)
        self.shield_policy.window = int(self.shield_window)
        self.targets.max_frames = int(self.target_max_frames)

    # -- census ---------------------------------------------------------

    @property
    def kills(self) -> int:
        return self.ledger.kills

    @property
    def streak(self) -> int:
        return self.ledger.streak

    def observe(self, snap: ZeldaSnapshot) -> None:
        """Bank kills and velocity. Call once per frame, before ``step``."""
        self.ledger.observe(snap)
        # ``ObjectTracker.observe`` is idempotent per snapshot, so a nested
        # controller re-observing the frame is free.
        self._tracked = self._tracker.observe(snap)
        event = self.damage.observe(
            snap, self._tracked, phase=f"{int(snap.screen):#04x}"
        )
        if event is not None:
            causes = self.hits_by_screen.setdefault(int(snap.screen), {})
            key = hit_cause(event)
            causes[key] = causes.get(key, 0) + 1

    def _track(self, obj: ZeldaObject | None) -> TrackedObject | None:
        if obj is None:
            return None
        slot = int(obj.slot)
        return next((t for t in self._tracked if t.slot == slot), None)

    def _tracks_by_slot(self) -> dict[int, TrackedObject]:
        return {t.slot: t for t in self._tracked}

    # -- main -----------------------------------------------------------

    def step(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        lane: tuple[str, int] | None = None,
    ) -> FrameAction | None:
        if int(snap.level) != 0 or int(snap.mode) != PLAY_MODE or snap.transitioning:
            return None
        screen = int(snap.screen)
        if screen != self.screen:
            self._enter(screen)
        self.since_enter += 1
        in_box = len(bodies_in_box(snap, self.box))
        if in_box:
            self.prey_by_screen[screen] = max(
                self.prey_by_screen.get(screen, 0), in_box
            )
        lx, ly = int(snap.link_x), int(snap.link_y)

        close = closest_body(snap, lx, ly)
        pad = 10**6 if close is None else chebyshev(lx, ly, int(close.x), int(close.y))
        if close is not None and self._at_contact(lx, ly, close, pad):
            if attackable(close, self._track(close)):
                return self._strike(snap, frames, close, f"hunt_{screen:02x}")
            peel = self._peel(snap, close, f"hunt_{screen:02x}")
            if peel is not None:
                return peel

        block = self._shield_action(snap, close, pad)
        if block is not None:
            return block

        if int(snap.whole_hearts) <= self.min_hearts:
            # Guard, do not chase. The baseline handed three live waves to a
            # hop with no combat at all at exactly this point. The budget
            # still runs: a caller that waits on ``done`` (the destination
            # hunt) would otherwise idle here for the whole stage.
            self.guard_frames += 1
            self._note_once(f"hunt_guard_{screen:02x}")
            if screen not in self.done:
                self._spend(screen)
                if self.screen_frames > self.screen_max_frames:
                    self._retire(screen, "guard_budget")
            return self._collect(snap, frames, lane, heal_only=True)

        if screen in self.done:
            return self._collect(snap, frames, lane, heal_only=True)

        return self._hunt(snap, frames, screen, lane)

    def striking(self, snap: ZeldaSnapshot) -> bool:
        """True when the blade already reaches the nearest body.

        The caller's reactive evader runs ahead of every hop rule, the hunt
        included, and that silences the sword: a red octorok is one wooden
        hit, so stepping away from a body already in the blade box trades a
        kill for a frame of separation it gives straight back.
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

    def take_destination(
        self, snap: ZeldaSnapshot, frames: int
    ) -> FrameAction | None:
        """:meth:`step` on the screen the caller's hop table ended on.

        No lane to walk back to, and the bigger budget: the route is over, so
        the only thing left to spend frames on is the wave.
        """
        if not self._destination:
            self._destination = True
            self.screen_max_frames = self.destination_frames
            self.targets.max_frames = self.destination_target_frames
            self.targets.skipped.clear()
            self.screen_frames = 0
        return self.step(snap, frames, lane=None)

    def _at_contact(self, lx: int, ly: int, body: ZeldaObject, pad: int) -> bool:
        """True when this body is close enough that the answer is the blade.

        ``in_sword_hitbox`` is true out to 20 px; the old ladder only turned
        to face at ``pad > MIN_DODGE_BODY + 2``, leaving a 17-18 px dead band
        where Link was in range facing the wrong way (0x68 f=1763). The swing
        carries its own direction now, so there is no band.
        """
        face = face_toward(lx, ly, int(body.x), int(body.y))
        return pad <= MIN_DODGE_BODY or in_sword_hitbox(
            lx, ly, face, int(body.x), int(body.y)
        )

    # -- tactics --------------------------------------------------------

    def _strike(
        self, snap: ZeldaSnapshot, frames: int, body: ZeldaObject, reason: str
    ) -> FrameAction:
        """Swing, in the direction of the body, on the cadence.

        No peel inside the pad: a sidestep must walk the whole pad before it
        clears the hitbox while the body closes ~1 px a frame, so one started
        inside cannot finish (0x49 f=4500 pressed LEFT into a bush for eight
        frames while slot 4 closed 16 -> 8). A red octorok is one wooden hit.
        Direction + A on one frame: the facing byte is read before the sword,
        so the blade lands in ``face`` and the attack state pins Link.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        face = face_toward(lx, ly, int(body.x), int(body.y))
        self._freeze_occ()
        if link_busy(snap):
            # The animation owns the frame, and the idle is also the release
            # edge the ROM needs before the next swing can start.
            return FrameAction(nes_idle_action(), f"{reason}_recover")
        track = self._track(body)
        closing = track.closing_on(lx, ly) if track is not None else True
        if self.shield_policy.hold_swing(snap, self._tracked, closing):
            # Facing the shot already: the A press is the only thing that
            # would drop the shield (0x58 f=2056).
            return FrameAction(nes_idle_action(), f"{reason}_block")
        return FrameAction(nes_action(face, "A"), f"{reason}_slash")

    def _peel(
        self, snap: ZeldaSnapshot, body: ZeldaObject, reason: str
    ) -> FrameAction | None:
        """Back off a body the sword cannot answer. The complement to
        :meth:`_strike`'s "no peel inside the pad".

        That rule holds because the body dies first. A Peahat in flight never
        does, so walking away is the only answer there is — 0x59 f=3165 walked
        Link to 9 px of one and stood, strike vetoed and nothing replacing it.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        dx, dy = int(body.x) - lx, int(body.y) - ly
        away = ("LEFT" if dx > 0 else "RIGHT", "UP" if dy > 0 else "DOWN")
        if abs(dy) > abs(dx):
            away = away[::-1]
        for direction in away:
            if _box_step(lx, ly, direction, self.box) is not None:
                self.peel_frames += 1
                self._freeze_occ()
                return FrameAction(nes_action(direction), f"{reason}_peel")
        return None

    def _shield_action(
        self, snap: ZeldaSnapshot, body: ZeldaObject | None, pad: int
    ) -> FrameAction | None:
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

    def _approach(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        target: ZeldaObject,
        close: ZeldaObject,
        pad: int,
        reason: str,
    ) -> FrameAction:
        """Close on ``target``, reacting to the body that is actually nearest.

        The body about to touch Link is not always the one the hunt picked:
        on ``0x49`` a 7-kill streak died to slot 4 closing 9 px on Link's lane
        while the walk aimed at a stand 30 px north of slot 1.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        if pad <= SWORD_REACH:
            # In reach but off-axis: strafe to the blade box. Walking the
            # toward-axis is how a 20 px stand-off became an 8 px collision.
            self._freeze_occ()
            cx, cy = int(close.x), int(close.y)
            dx, dy = cx - lx, cy - ly
            if abs(dx) >= abs(dy) and dy != 0:
                align = "DOWN" if dy > 0 else "UP"
            elif abs(dy) > abs(dx) and dx != 0:
                align = "RIGHT" if dx > 0 else "LEFT"
            else:
                align = face_toward(lx, ly, cx, cy)
            if _box_step(lx, ly, align, self.box) is not None:
                return FrameAction(nes_action(align), f"{reason}_align")
            return FrameAction(nes_action(face_toward(lx, ly, cx, cy), "A"), f"{reason}_slash")
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
        return self._occ.walk(snap, frames, goal, reason, slash=False)

    def _off_line(self, lx: int, ly: int, pad: int) -> str | None:
        """Leave a shooter's axis while there is still room to close again.

        ``threat.off_line_step`` is the pre-emptive half of the dungeon policy
        and this corridor is what it was written for. Capped per screen — a
        shooter turns to face Link after every sidestep, so it would spiral.
        """
        if not self.avoid_firing_lines:
            return None
        # Only while the sword has nothing to do. Inside the blade box the
        # answer is the swing, and a sidestep started there cannot finish.
        if pad <= SWORD_REACH:
            return None
        if self.off_line_frames >= self.off_line_max_frames:
            return None
        bodies = tuple(t for t in self._tracked if t.hazard is HazardClass.BODY)
        step = off_line_step((lx, ly), bodies, bounds=self.box)
        if step is None or _box_step(lx, ly, step, self.box) is None:
            return None
        self.off_line_frames += 1
        self._freeze_occ()
        return step

    # -- screen ---------------------------------------------------------

    def _hunt(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        screen: int,
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
            target, note = self.targets.pick(snap, prey, self._tracks_by_slot())
            if self.targets.slot != held:
                self._stand_side = None
            if note is not None:
                self.notes.append(f"hunt_skip_{screen:02x}_{note}")
            if target is None:
                # Written off, or unhittable this frame (a Peahat in flight).
                # Hand the frame back; the screen stays open so a landing
                # Peahat is still worth a kill.
                return self._collect(snap, frames, lane, heal_only=False)
            lx, ly = int(snap.link_x), int(snap.link_y)
            close = closest_body(snap, lx, ly) or target
            pad = chebyshev(lx, ly, int(close.x), int(close.y))
            return self._approach(
                snap, frames, target, close, pad, f"hunt_{screen:02x}"
            )

        pickup = nearest_to(int(snap.link_x), int(snap.link_y), floor_pickups(snap, self.box))
        if pickup is not None:
            # After the wave: scooping mid-fight walked Link onto a drop
            # sitting in an octorok's pad (0x68 (179,125), body at 187).
            self.settle = 0
            self.targets.release()
            self._spend(screen)
            if self.screen_frames > self.screen_max_frames:
                self._retire(screen, "budget")
                return None
            return self._occ.walk(
                snap, frames, (int(pickup.x), int(pickup.y)), "hunt_drop", slash=False
            )

        # The drop lands a frame or two after the body. Hand these frames to
        # the hop rather than idling; the pickup branch reclaims a late drop.
        self.settle += 1
        if (
            self.settle >= self.settle_frames
            and self.since_enter >= self.spawn_wait_frames
        ):
            self._clear(screen)
        return self._lane_return(snap, frames, lane)

    def _collect(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        lane: tuple[str, int] | None,
        *,
        heal_only: bool,
    ) -> FrameAction | None:
        """Bank a drop this screen still owes, then walk back to the lane.

        A guarding or retired screen used to go straight to the lane, leaving
        the heart that would have ended the guard on the floor. Budgeted, and
        hearts only.
        """
        if self.collect_frames < self.collect_max_frames:
            lx, ly = int(snap.link_x), int(snap.link_y)
            drop = nearest_to(lx, ly, floor_pickups(snap, self.box, heal_only=heal_only))
            if drop is not None:
                self.collect_frames += 1
                return self._occ.walk(
                    snap, frames, (int(drop.x), int(drop.y)), "hunt_scoop", slash=False
                )
        return self._lane_return(snap, frames, lane)

    def _lane_return(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        lane: tuple[str, int] | None,
    ) -> FrameAction | None:
        """Walk back onto the hop's lane on the grid the chase just learned.

        Only after a fight on this screen (``screen_frames``): a screen the
        hunt never touched is the hop's business and Link is on its lane
        already. Yields on the budget and on ``occupancy_stand``.
        """
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

    def escape_for(
        self, snap: ZeldaSnapshot, frames: int, hop: ScreenHop, reason: str
    ) -> FrameAction | None:
        """:meth:`escape` toward the scroll line ``hop`` is trying to reach."""
        goal = hop_exit_goal(hop)
        return None if goal is None else self.escape(snap, frames, goal, reason)

    def escape(
        self, snap: ZeldaSnapshot, frames: int, goal: tuple[int, int], reason: str
    ) -> FrameAction | None:
        """Walk toward ``goal`` on the grid this screen's chase learned.

        For the caller's stall branch only: ``align_and_push`` holds one
        direction and ``unstick_wiggle`` waits forever once spent, so a hop
        that walks Link into a pocket has no way out (two live runs burned
        27,000+ frames at (56,125)). ``None`` when this grid has no route.
        """
        act = self._occ.walk(snap, frames, goal, reason, slash=True)
        return None if act.reason == "occupancy_stand" else act

    # -- bookkeeping ----------------------------------------------------

    def _spend(self, screen: int) -> None:
        self.screen_frames += 1
        self.hunt_frames += 1
        self.frames_by_screen[screen] = self.frames_by_screen.get(screen, 0) + 1

    def _freeze_occ(self) -> None:
        """Stand/strafe/slash do not grade as a missed occupancy step."""
        walker = self._occ.walker
        if walker is not None:
            walker.last_dir = None
            walker.last_xy = None

    def _note_once(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _enter(self, screen: int) -> None:
        self.screen = screen
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self.collect_frames = 0
        self.off_line_frames = 0
        self.targets.clear()
        self._stand_side = None
        self._occ.reset()

    def _clear(self, screen: int) -> None:
        self.done.add(screen)
        self.screens_cleared += 1
        self.notes.append(f"hunt_clear_{screen:02x}")

    def _retire(self, screen: int, why: str) -> None:
        self.done.add(screen)
        self.screens_retired += 1
        self.notes.append(f"hunt_{why}_{screen:02x}")

    def reset(self) -> None:
        self.ledger.reset()
        self.damage = DamageLog()
        self.hits_by_screen.clear()
        self.shield_policy.reset()
        self.targets.clear()
        self.targets.skips = 0
        self.hunt_frames = 0
        self.guard_frames = 0
        self.peel_frames = 0
        self.collect_frames = 0
        self.off_line_frames = 0
        self.screens_cleared = 0
        self.screens_retired = 0
        self.frames_by_screen.clear()
        self.prey_by_screen.clear()
        self.notes.clear()
        self.screen = -1
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self.done.clear()
        self._destination = False
        self._tracked = ()
        self._tracker = ObjectTracker()
        self._occ.reset()

    def screen_table(self) -> list[dict[str, Any]]:
        """The per-screen bill: one row per screen, in walk order.

        The ledger owns what the health and rupee bytes did; the hunt owns
        what it spent getting there and who landed the hit. A flat total
        cannot say which screen the damage or the money came from, and on
        this corridor those are not the same screens.
        """
        rows = []
        for row in self.ledger.screen_rows():
            screen = int(row["screen"], 16)
            rows.append(
                {
                    **row,
                    "hunt_frames": self.frames_by_screen.get(screen, 0),
                    "peak_prey": self.prey_by_screen.get(screen, 0),
                    "cleared": screen in self.done,
                    "hits_by_cause": dict(self.hits_by_screen.get(screen, {})),
                }
            )
        return rows

    def report(self) -> dict[str, Any]:
        causes: dict[str, int] = {}
        for per_screen in self.hits_by_screen.values():
            for key, n in per_screen.items():
                causes[key] = causes.get(key, 0) + n
        return {
            **self.ledger.report(),
            **self.shield_policy.report(),
            "hits_by_cause": causes,
            "screens": self.screen_table(),
            "hunt_frames": self.hunt_frames,
            "guard_frames": self.guard_frames,
            "peel_frames": self.peel_frames,
            "off_line_frames": self.off_line_frames,
            "target_skips": self.targets.skips,
            "screens_cleared": self.screens_cleared,
            "screens_retired": self.screens_retired,
            "screens_done": sorted(self.done),
            "hunt_frames_by_screen": {
                f"{k:#04x}": v for k, v in sorted(self.frames_by_screen.items())
            },
            "peak_prey_by_screen": {
                f"{k:#04x}": v for k, v in sorted(self.prey_by_screen.items())
            },
            "occupancy_misses": self._occ.misses,
            "notes": list(self.notes),
        }
