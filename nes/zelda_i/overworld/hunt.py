"""On-route screen clearing: kill the wave, then bank what it drops.

The pre-L1 errand is money. ``OverworldPathController`` crosses a screen on
one lane and only scoops a drop that happens to land beside it, so the wave
standing two lanes north is never fought and never pays. This module is the
opposite policy: while a hop crosses a screen, walk to every live body in the
interior box, kill it, and walk onto whatever it leaves behind.

Three guards keep that from eating the route:

* a per-screen frame budget (``screen_max_frames``) — past it the screen is
  retired and the hop drives again. A retired screen is never re-hunted, so a
  hop that scrolls back and forth cannot restart the budget;
* a per-target budget (``target_max_frames``) — one body that will not die (a
  Zola that submerges, a Peahat in flight) is skipped, not chased to the
  screen budget;
* an interior box (:data:`HUNT_BOX`). A chase that walks Link into a scroll
  line changes screen under the hop table, and the hop then advances against
  the wrong arrival edge. Bodies outside the box are left where they are.

Kills are counted two ways, because neither is free of doubt.

``kills`` is a slot census: a live enemy slot that stops being that same live
enemy, on the same screen, in play mode, died. It over-counts anything that
despawns on its own (a Zola submerging) and under-counts nothing.

``kills_counter`` reads the ROM's own forced-drop counters — ``$0627`` (16
kills force a fairy) and ``$0050`` (10 force a rupee). Both increment once per
kill. ``Link_BeHarmed`` (aldonunez) zeros them on Link-enemy *collision*,
including 0-damage bubbles — not on a ``$066F`` change. A wooden octorok
chip is ``$0670`` only; Survival assist writes that byte back to ``$FF``
before the next ``observe``, so ``damage_taken`` can stay 0 while the
streak resets. ``hurt_events`` watches ``$04F0`` (Link iframes) instead:
that timer survives the refill. Live 2026-09-15: 6 resets, each with
iframes 24 / knockback 32 / hp still ``0x22``/``$FF``.

Both kill censuses are reported. Agreement is the evidence; a gap is a
measurement to chase, not a number to quote. The 2026-09-14 walk reads 14
and 13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.combat import FLOOR_DROP_TYPES
from zelda_i.dungeon.behaviors import is_projectile
from zelda_i.dungeon.ids import (
    FAIRY_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    HEART_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.heart_farm import LEAVE_GOALS, FarmOccupancy
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot

__all__ = [
    "CAVE_TRIGGER_TYPE",
    "HUNT_BOX",
    "HUNT_LANE_MAX_FRAMES",
    "HUNT_LANE_TOL",
    "HUNT_MIN_HEARTS",
    "HUNT_SCREEN_MAX_FRAMES",
    "HUNT_SETTLE_FRAMES",
    "HUNT_SPAWN_WAIT_FRAMES",
    "HUNT_TARGET_MAX_FRAMES",
    "MAX_PREY_HP",
    "ScreenHunter",
    "hop_exit_goal",
    "hop_lane",
    "hunt_prey",
]

# One screen's worth of fighting. The 0x77 -> 0x4A walk crosses seven screens
# and took 1924 frames without fighting anything; this caps the whole detour
# at a few thousand rather than ``max_frames``.
HUNT_SCREEN_MAX_FRAMES = 600
# One body. An overworld octorok dies in 1-2 wooden swings once Link closes,
# and closing across the box is ~120 frames at ~1 px/frame.
HUNT_TARGET_MAX_FRAMES = 180
# The drop appears a frame or two after the body does, so do not declare the
# screen clear on the first empty frame. These frames are handed back to the
# hop, not idled away.
HUNT_SETTLE_FRAMES = 24
# A wave does not exist the moment the scroll ends. Do not let an empty first
# look retire a screen for the rest of the run: a dungeon room's settle spawn
# is 80-100f (``door_graph.level3_exits``) and the overworld is no faster.
# These frames are handed to the hop, so the wait is free unless Link leaves
# first — and a screen he leaves before the wait is up stays un-retired, which
# is the point.
HUNT_SPAWN_WAIT_FRAMES = 110
# Never trade the last heart for a rupee: the route still has to reach L1.
HUNT_MIN_HEARTS = 1
# Walking back to the hop's lane after a fight. ``align_and_push`` holds one
# direction and has no answer to a bush: the first live hunt left Link at
# 0x49 (56,125) holding DOWN against one, and the stage burned 27,000 frames
# on ``unstick_wait``. The hunt is what took him off the lane, so the hunt
# walks him back — on the grid it just learned chasing bodies across it.
HUNT_LANE_TOL = 6
HUNT_LANE_MAX_FRAMES = 240
# Interior box (xlo, xhi, ylo, yhi). The scroll lines are x=14/232, y=62/212
# (``overworld.common.EDGE_*``); this keeps a whole chase off them.
HUNT_BOX = (32, 214, 76, 198)
# Slot 11 on a shop screen: type 0x64, hp 240, no sprite. It is the cave /
# NPC trigger, not prey — chasing it is how the rupee farm used to stall.
CAVE_TRIGGER_TYPE = 0x64
# Live overworld foes are single-digit HP. A 200+ reading is a trigger or a
# boss-shaped slot, neither of which the wooden sword resolves.
MAX_PREY_HP = 200

_PICKUP_STATES = frozenset(
    {RUPEE_DROP_STATE, FIVE_RUPEE_DROP_STATE, HEART_DROP_STATE, FAIRY_DROP_STATE}
)
_HEAL_STATES = frozenset({HEART_DROP_STATE, FAIRY_DROP_STATE})


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


def _live_enemies(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    """Typed, killable, non-drop slots. No position filter.

    Deliberately looser than :func:`combat.overworld_threat_objects`, whose
    ``40 < y < 220`` bound would read a body that wandered low as a corpse and
    bank a kill that never happened.

    Projectiles are out, and that is a census correction, not a targeting one:
    an octorok rock occupies a slot with hp and then vanishes against a wall,
    which the slot census reads as a death. Measured on the 0x77 -> 0x4A walk
    (2026-09-14): 10 census kills against 6 on the ROM counters, and the four
    extra were rocks.
    """
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and int(obj.type_id) not in (0, 0xFF, CAVE_TRIGGER_TYPE)
        and int(obj.type_id) not in FLOOR_DROP_TYPES
        and 0 < int(obj.hp) < MAX_PREY_HP
        and not is_projectile(obj)
    )


def hunt_prey(
    snap: ZeldaSnapshot, box: tuple[int, int, int, int] = HUNT_BOX
) -> tuple[ZeldaObject, ...]:
    """Live bodies worth walking to: in the box, killable, not a projectile."""
    xlo, xhi, ylo, yhi = box
    return tuple(
        obj
        for obj in _live_enemies(snap)
        if xlo <= int(obj.x) <= xhi
        and ylo <= int(obj.y) <= yhi
    )


def _pickups(
    snap: ZeldaSnapshot, box: tuple[int, int, int, int]
) -> tuple[ZeldaObject, ...]:
    """Floor drops worth walking to. Bombs and clocks are not in this set.

    Bomb is ROM item code ``0x00`` — the same ObjState a cleared slot reads as
    — so a bomb chase here would outrank every real rupee on the one leg where
    Link owns no bombs. Hearts and fairies only count while Link is damaged.
    """
    xlo, xhi, ylo, yhi = box
    # Not ``filled_hearts < heart_containers``: ``filled_hearts`` is *whole*
    # hearts and the container's own partial lives in ``$0670``, so that
    # comparison is true at full health on every container count.
    heal_wanted = not snap.health_is_full or int(snap.heart_partial) != 0xFF
    return tuple(
        obj
        for obj in snap.objects
        if obj.slot >= 1
        and int(obj.type_id) == RUPEE_DROP_OBJECT_TYPE
        and int(obj.state) in _PICKUP_STATES
        and (heal_wanted or int(obj.state) not in _HEAL_STATES)
        and xlo <= int(obj.x) <= xhi
        and ylo <= int(obj.y) <= yhi
    )


def _nearest(snap: ZeldaSnapshot, objs: tuple[ZeldaObject, ...]) -> ZeldaObject | None:
    if not objs:
        return None
    return min(
        objs,
        key=lambda o: abs(int(o.x) - int(snap.link_x)) + abs(int(o.y) - int(snap.link_y)),
    )


@dataclass
class ScreenHunter:
    """Clear the current overworld screen, then hand the frame back.

    ``step`` returns ``None`` on every frame the hunt does not want — nothing
    left to fight, screen retired, Link too hurt, wrong mode — so the caller's
    own hop rules keep driving. It never presses a direction that leaves the
    screen and it never writes RAM.
    """

    screen_max_frames: int = HUNT_SCREEN_MAX_FRAMES
    target_max_frames: int = HUNT_TARGET_MAX_FRAMES
    settle_frames: int = HUNT_SETTLE_FRAMES
    spawn_wait_frames: int = HUNT_SPAWN_WAIT_FRAMES
    min_hearts: int = HUNT_MIN_HEARTS
    lane_tol: int = HUNT_LANE_TOL
    lane_max_frames: int = HUNT_LANE_MAX_FRAMES
    box: tuple[int, int, int, int] = HUNT_BOX

    kills: int = 0
    kills_counter: int = 0
    rupees_banked: int = 0
    damage_taken: int = 0
    hurt_events: int = 0
    streak_best: int = 0
    streak_resets: int = 0
    hunt_frames: int = 0
    screens_cleared: int = 0
    screens_retired: int = 0
    by_screen: dict[int, int] = field(default_factory=dict)
    seen_by_screen: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    screen: int = -1
    screen_frames: int = 0
    since_enter: int = 0
    settle: int = 0
    lane_frames: int = 0
    target_slot: int | None = None
    target_frames: int = 0
    done: set[int] = field(default_factory=set)
    skipped: set[int] = field(default_factory=set)
    _census: dict[int, int] = field(default_factory=dict, repr=False)
    _census_screen: int = field(default=-1, repr=False)
    _world: int = field(default=-1, repr=False)
    _hp: int = field(default=-1, repr=False)
    _iframes: int = field(default=-1, repr=False)
    _rupees: int = field(default=-1, repr=False)
    _help: int = field(default=-1, repr=False)
    _occ: FarmOccupancy = field(default_factory=FarmOccupancy, repr=False)

    # ------------------------------------------------------------------ #
    # Kill census (runs every frame, hunting or not)
    # ------------------------------------------------------------------ #

    def observe(self, snap: ZeldaSnapshot) -> None:
        """Bank kills. Call once per frame, before ``step``."""
        self._observe_counters(snap)
        census = {int(o.slot): int(o.type_id) for o in _live_enemies(snap)}
        screen = int(snap.screen)
        same_screen = (
            screen == self._census_screen
            and int(snap.level) == 0
            and int(snap.mode) == PLAY_MODE
            and not snap.transitioning
        )
        if same_screen:
            for slot, type_id in self._census.items():
                if census.get(slot) != type_id:
                    self.kills += 1
                    self.by_screen[screen] = self.by_screen.get(screen, 0) + 1
            self.seen_by_screen[screen] = max(
                self.seen_by_screen.get(screen, 0), len(census)
            )
        self._census = census
        self._census_screen = screen if int(snap.level) == 0 else -1

    def _observe_counters(self, snap: ZeldaSnapshot) -> None:
        rupees = int(snap.rupees)
        if self._rupees >= 0:
            # Positive deltas only: a purchase is not a negative kill.
            self.rupees_banked += max(rupees - self._rupees, 0)
        self._rupees = rupees
        # Chip damage, which ``hits_taken`` cannot see: a half-heart lands in
        # ``$0670`` and never touches the whole-hearts byte the hop controller
        # watches. Survival assist refills ``$0670`` before the next observe,
        # so this census is empty on an assisted walk — ``hurt_events`` is
        # the one that still fires.
        hp = int(snap.filled_hearts) * 256 + int(snap.heart_partial)
        if self._hp >= 0 and hp < self._hp:
            self.damage_taken += 1
        self._hp = hp
        iframes = int(getattr(snap, "link_iframes", 0))
        if self._iframes >= 0 and iframes > 0 and self._iframes == 0:
            self.hurt_events += 1
        self._iframes = iframes
        world, help_ = int(snap.world_kill_count), int(snap.help_drop_count)
        if self._world >= 0:
            # A negative delta is a reset, never a kill. The two counters
            # were measured moving in lockstep on the live walk, so the max is
            # belt-and-braces rather than independent coverage of a wrap.
            gained = max(world - self._world, help_ - self._help, 0)
            self.kills_counter += gained
            # The streak is the only forced rupee on this corridor: ten kills
            # force a 5-rupee ($0050), sixteen force a fairy ($0627). Every
            # reset below those thresholds is money not paid. Link_BeHarmed
            # (collision, $04F0 0→24) is what zeros them — not a $066F
            # change, and not a screen scroll. Live 2026-09-15: 6 resets,
            # 6 hurt_events, damage_taken 0 on an assisted walk.
            if world == 0 and self._world > 0:
                self.streak_resets += 1
        self.streak_best = max(self.streak_best, world)
        self._world, self._help = world, help_

    # ------------------------------------------------------------------ #
    # Hunt
    # ------------------------------------------------------------------ #

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
        if screen in self.done:
            return self._lane_return(snap, frames, lane)
        if int(snap.filled_hearts) <= self.min_hearts:
            self._retire(screen, "hurt")
            return self._lane_return(snap, frames, lane)

        pickup = _nearest(snap, _pickups(snap, self.box))
        if pickup is not None:
            self.screen_frames += 1
            self.settle = 0
            self.target_slot = None
            self.target_frames = 0
            self.hunt_frames += 1
            if self.screen_frames > self.screen_max_frames:
                self._retire(screen, "budget")
                return None
            return self._occ.walk(
                snap, frames, (int(pickup.x), int(pickup.y)), "hunt_drop", slash=False
            )

        prey = tuple(
            o for o in hunt_prey(snap, self.box) if int(o.slot) not in self.skipped
        )
        if not prey:
            # The drop lands a frame or two after the body. Hand these frames
            # to the hop rather than idling: the pickup branch above reclaims
            # a late drop, and the hop has only moved a pixel or two.
            self.settle += 1
            if (
                self.settle >= self.settle_frames
                and self.since_enter >= self.spawn_wait_frames
            ):
                self._clear(screen)
            return self._lane_return(snap, frames, lane)

        self.settle = 0
        self.screen_frames += 1
        self.hunt_frames += 1
        if self.screen_frames > self.screen_max_frames:
            self._retire(screen, "budget")
            return self._lane_return(snap, frames, lane)

        target = self._pick(snap, prey)
        if target is None:
            self._retire(screen, "stubborn")
            return self._lane_return(snap, frames, lane)
        return self._occ.walk(
            snap,
            frames,
            (int(target.x), int(target.y)),
            f"hunt_{screen:02x}",
            slash=True,
        )

    def _pick(
        self, snap: ZeldaSnapshot, prey: tuple[ZeldaObject, ...]
    ) -> ZeldaObject | None:
        """Hold one target until it dies or outlives its budget, then skip it.

        Re-picking the nearest body every frame makes Link oscillate between
        two octoroks that are equidistant; the budget is what stops a body the
        wooden sword cannot reach from owning the screen.
        """
        held = next((o for o in prey if int(o.slot) == self.target_slot), None)
        if held is not None:
            self.target_frames += 1
            if self.target_frames <= self.target_max_frames:
                return held
            self.skipped.add(int(held.slot))
            self.notes.append(f"hunt_skip_{self.screen:02x}_slot{int(held.slot)}")
            prey = tuple(o for o in prey if int(o.slot) != int(held.slot))
        chosen = _nearest(snap, prey)
        if chosen is None:
            return None
        self.target_slot = int(chosen.slot)
        self.target_frames = 1
        return chosen

    def _lane_return(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        lane: tuple[str, int] | None,
    ) -> FrameAction | None:
        """Walk back onto the hop's lane on the grid the chase just learned.

        Only after a fight on this screen (``screen_frames``): a screen the
        hunt never touched is the hop's own business, and Link is on its lane
        already. Yields on the budget, and on ``occupancy_stand`` — a hunt that
        cannot find the lane hands the frame back rather than idling on it.
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

        For the caller's stall branch only. ``align_and_push`` holds one
        direction and ``unstick_wiggle`` waits forever once the wiggle is
        spent, so a hop that walks Link into a pocket has no way out: two live
        runs ended at (56,125) — one on 0x49, one on 0x58 — with 27,000 and
        28,000 frames of ``unstick_wait``. This grid learns the bush by
        bumping it and routes around. ``None`` when it has no route either, so
        the caller still falls through to its own recovery.
        """
        act = self._occ.walk(snap, frames, goal, reason, slash=True)
        return None if act.reason == "occupancy_stand" else act

    def _enter(self, screen: int) -> None:
        self.screen = screen
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self.target_slot = None
        self.target_frames = 0
        self.skipped.clear()
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
        self.kills = 0
        self.kills_counter = 0
        self.rupees_banked = 0
        self.damage_taken = 0
        self.hurt_events = 0
        self.streak_best = 0
        self.streak_resets = 0
        self.hunt_frames = 0
        self.screens_cleared = 0
        self.screens_retired = 0
        self.by_screen.clear()
        self.seen_by_screen.clear()
        self.notes.clear()
        self.screen = -1
        self.screen_frames = 0
        self.since_enter = 0
        self.settle = 0
        self.lane_frames = 0
        self.target_slot = None
        self.target_frames = 0
        self.done.clear()
        self.skipped.clear()
        self._census.clear()
        self._census_screen = -1
        self._world = -1
        self._help = -1
        self._rupees = -1
        self._hp = -1
        self._iframes = -1
        self._occ.reset()

    def report(self) -> dict[str, Any]:
        return {
            "kills": self.kills,
            "kills_counter": self.kills_counter,
            "rupees_banked": self.rupees_banked,
            "damage_taken": self.damage_taken,
            "hurt_events": self.hurt_events,
            "streak_best": self.streak_best,
            "streak_resets": self.streak_resets,
            "hunt_frames": self.hunt_frames,
            "screens_cleared": self.screens_cleared,
            "screens_retired": self.screens_retired,
            "screens_done": sorted(self.done),
            "kills_by_screen": {f"{k:#04x}": v for k, v in sorted(self.by_screen.items())},
            "peak_live_by_screen": {
                f"{k:#04x}": v for k, v in sorted(self.seen_by_screen.items()) if v
            },
            "occupancy_misses": self._occ.misses,
            "notes": list(self.notes),
        }
