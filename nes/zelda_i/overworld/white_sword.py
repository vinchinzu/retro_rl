"""White Sword detour: Level 9 approach screen 0x05 -> cave 0x0A -> back.

Route (every leg live-verified 2026-09-07 from the real power-on pin
``OW_07_Row0Real``; ``scratch/probe_white_sword_detour.py`` runs it end to
end in 5,554 frames, ``ADDR_SWORD`` 1 -> 2, zero memory writes)::

    0x05 -RIGHT y=141-> 0x06 -RIGHT y=141-> 0x07
    0x07 -DOWN  x=32 -> 0x17 -RIGHT y=141-> 0x18 -RIGHT y=141-> 0x19
    0x19 -RIGHT y=141-> 0x1A -UP from (208,157)-> 0x0A
    0x0A: climb the x=208 sand corridor to y<=87, west along the top band to
          x~34, UP into the cave mouth
    cave: walk to x=120, UP -> White Sword
    then the same legs reversed back to 0x05.

`route/item_gate_hops.py` had the cave screen right (0x0A) and the approach
wrong: it planned "west off the Level 5 door 0x0B", which is sealed at every
band, and its fallback through Lost Hills 0x1B wraps to itself in all four
directions. The live way in is 0x1A's single north opening at x=208 -- not a
maze count, despite 0x1A sitting next to the Lost Hills.

The Old Man gates on **heart containers**, not filled hearts, so
``UnlimitedHealthAssist`` cannot unlock him; the controller refuses up front
if Link is short. It writes nothing: controller inputs only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

CAVE_MODE = 11
SCREEN_L9_APPROACH = 0x05
SCREEN_ROW0_GATE = 0x07
SCREEN_WHITE_SWORD_CAVE = 0x0A
SCREEN_MAZE_GATE = 0x1A
WHITE_SWORD = 2
MIN_HEART_CONTAINERS = 5

BAND_Y = 141
# 0x07 <-> 0x17 connect through a single lane at x=64. Measured live
# (scratch/probe_ow_07_descent.py): from 0x07's y=141 arrival band, x=64 is
# the *only* column that reaches the bottom band -- every other one walls out
# or slides Link to a screen edge. The original 0x07 DOWN hop read x=32
# because it was measured from a pin already standing at (64,221), where the
# in-screen descent never had to happen.
GATE_07_X = 64          # 0x07 -> 0x17 drops on this column
GATE_17_X = 64          # 0x17 -> 0x07 climbs on this column
MAZE_NORTH_X = 208      # 0x1A's single north opening
MAZE_STAND_Y = 157
CORRIDOR_X = 208        # 0x0A's south sand corridor
TOP_BAND_Y = 85
TOP_BAND_REACHED_Y = 87
CAVE_MOUTH_X = 34
CORRIDOR_BOTTOM_Y = 213
CAVE_ITEM_X = 120
# The Old Man's text freezes input, so pressing early is a no-op, not a
# risk. ``overworld/sword_cave.py`` measured 35 f as enough to let the
# cave settle; 300 was a blind hold that cost ~265 f of dead air.
DIALOG_FRAMES = 35
TOL = 2
# 0x0A's blue Lynel (96 HP, two hearts a touch without the ring) spawns on
# the top band at (160,93), on the route, then roams the east lane and the
# bottom band. Climb from the corridor foot only while it is on the bottom
# band west of LYNEL_SAFE_X; the lake is then between it and the route to
# the cave. Nearer than LYNEL_BACKOFF, step back to 0x1A and re-enter (a
# re-entry respawns it). Wait at most LYNEL_WAIT_MAX frames per entry.
LYNEL_TYPE = 0x01
LYNEL_SAFE_Y = 165
LYNEL_SAFE_X = 176
LYNEL_BACKOFF = 40
LYNEL_WAIT_MAX = 900
CORRIDOR_FOOT_Y = 190
# Its sword shot crosses the lake: from (176,189) a beam flew up x=176 and
# met Link walking the top band at x=172, two hearts (back_1a, 6/6 offsets).
# Hold while a shot's column (walking LEFT/RIGHT) or row (UP/DOWN) is within
# BEAM_AHEAD px ahead of Link.
LYNEL_BEAM_TYPE = 0x57
BEAM_AHEAD = 28

# (direction, cross-axis target, screen expected on arrival)
OUT_LEGS: tuple[tuple[str, int, int], ...] = (
    ("RIGHT", BAND_Y, 0x06), ("RIGHT", BAND_Y, 0x07),
    ("DOWN", GATE_07_X, 0x17), ("RIGHT", BAND_Y, 0x18),
    ("RIGHT", BAND_Y, 0x19), ("RIGHT", BAND_Y, 0x1A),
)
BACK_LEGS: tuple[tuple[str, int, int], ...] = (
    ("LEFT", BAND_Y, 0x19), ("LEFT", BAND_Y, 0x18), ("LEFT", BAND_Y, 0x17),
    ("UP", GATE_17_X, 0x07), ("LEFT", BAND_Y, 0x06), ("LEFT", BAND_Y, 0x05),
)

TRIFORCE_UNUSED = None


class WhiteSwordPhase(Enum):
    OUT_LEGS = auto()
    MAZE_NORTH = auto()
    CLIMB_0A = auto()
    TO_MOUTH = auto()
    ENTER_CAVE = auto()
    TAKE_SWORD = auto()
    EXIT_CAVE = auto()
    RETURN_TOP = auto()
    RETURN_CORRIDOR = auto()
    MAZE_SOUTH = auto()
    BACK_LEGS = auto()
    DONE = auto()
    FAILED = auto()


def _axis_step(cur: int, target: int, lo: str, hi: str) -> str | None:
    if abs(cur - target) <= TOL:
        return None
    return hi if cur < target else lo


@dataclass
class WhiteSwordDetourController:
    """0x05 -> White Sword cave 0x0A -> 0x05. Controller inputs only."""

    max_frames: int = 20_000
    phase: WhiteSwordPhase = WhiteSwordPhase.OUT_LEGS
    leg_i: int = 0
    frames: int = 0
    phase_frames: int = 0
    dialog_waited: int = 0
    maze_lane_done: bool = False
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)
    start_checked: bool = False
    lynel_wait: int = 0
    # Per-entry cap on the hold for the Lynel's bottom-band window. The
    # gathering lowers it when Link can take the two-heart hit.
    lynel_wait_max: int = LYNEL_WAIT_MAX

    def _set_phase(self, phase: WhiteSwordPhase) -> None:
        self.phase = phase
        self.phase_frames = 0
        self.leg_i = 0

    def _action(self, action: list[int], reason: str) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return FrameAction(action, reason)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self.phase = WhiteSwordPhase.FAILED
        self.notes.append(reason)
        return self._action(nes_idle_action(), reason)

    # -- legs --------------------------------------------------------------
    def _walk_leg(
        self, snap: ZeldaSnapshot, legs: tuple[tuple[str, int, int], ...], nxt: WhiteSwordPhase
    ) -> FrameAction:
        if self.leg_i >= len(legs):
            self._set_phase(nxt)
            return self._action(nes_idle_action(), f"{nxt.name.lower()}_begin")
        direction, band, want = legs[self.leg_i]
        if snap.screen == want and snap.mode == PLAY_MODE and not snap.transitioning:
            self.leg_i += 1
            if self.leg_i >= len(legs):
                self._set_phase(nxt)
            return self._action(nes_idle_action(), f"leg_{want:02x}_arrived")
        if snap.mode != PLAY_MODE or snap.transitioning:
            return self._action(nes_action(direction), "leg_scroll")
        # Align on the cross axis, then hold the leg direction.
        if direction in ("LEFT", "RIGHT"):
            d = _axis_step(int(snap.link_y), band, "UP", "DOWN")
        else:
            d = _axis_step(int(snap.link_x), band, "LEFT", "RIGHT")
        if d is not None:
            return self._action(nes_action(d), f"leg_{want:02x}_align")
        return self._action(nes_action(direction), f"leg_{want:02x}_walk")

    def _lynel_gate(
        self,
        snap: ZeldaSnapshot,
        *,
        back: str = "DOWN",
        foot=lambda y: y >= CORRIDOR_FOOT_Y,
    ) -> FrameAction | None:
        """Hold until the Lynel is on the bottom band; too near, step ``back``.

        Outbound the hold is the corridor foot and ``back`` leaves for 0x1A;
        on the way home it is the cave mouth and ``back`` re-enters the cave.
        """
        if snap.mode != PLAY_MODE or snap.transitioning or not foot(int(snap.link_y)):
            return None
        lynels = [o for o in snap.objects if 1 <= o.slot <= 10 and int(o.type_id) == LYNEL_TYPE and o.hp > 0]
        if not lynels or self.lynel_wait >= self.lynel_wait_max:
            return None
        lx, ly = int(snap.link_x), int(snap.link_y)
        if all(int(o.y) >= LYNEL_SAFE_Y and int(o.x) <= LYNEL_SAFE_X for o in lynels):
            return None
        self.lynel_wait += 1
        if any(max(abs(int(o.x) - lx), abs(int(o.y) - ly)) < LYNEL_BACKOFF for o in lynels):
            return self._action(nes_action(back), "lynel_backoff")
        return self._action(nes_idle_action(), "lynel_wait")

    def _beam_hold(self, snap: ZeldaSnapshot, direction: str) -> FrameAction | None:
        """Idle while a Lynel shot is about to cross the lane ahead."""
        lx, ly = int(snap.link_x), int(snap.link_y)
        sign = 1 if direction in ("RIGHT", "DOWN") else -1
        for o in snap.objects:
            if int(o.type_id) != LYNEL_BEAM_TYPE:
                continue
            if direction in ("LEFT", "RIGHT"):
                ahead = (int(o.x) - lx) * sign
            else:
                ahead = (int(o.y) - ly) * sign
            if -8 <= ahead <= BEAM_AHEAD:
                return self._action(nes_idle_action(), "lynel_beam_hold")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("white_sword_detour_timeout")

        if not self.start_checked:
            self.start_checked = True
            if snap.level != 0 or snap.screen != SCREEN_L9_APPROACH:
                return self._fail("white_sword_predecessor_contract_miss")
            # Filled hearts are not containers: the assist refills the low
            # nibble only, so it can never open this gate.
            if snap.heart_containers < MIN_HEART_CONTAINERS:
                return self._fail("white_sword_heart_container_gate")
            if snap.sword >= WHITE_SWORD:
                self.success = True
                self._set_phase(WhiteSwordPhase.DONE)
                return self._action(nes_idle_action(), "white_sword_already_held")

        if self.phase is WhiteSwordPhase.OUT_LEGS:
            return self._walk_leg(snap, OUT_LEGS, WhiteSwordPhase.MAZE_NORTH)

        # 0x1A -> 0x0A through the single north opening at x=208. 0x1A abuts
        # the Lost Hills, but this is plain geometry, not a maze count: every
        # other column is walled and x=208 always crosses.
        if self.phase is WhiteSwordPhase.MAZE_NORTH:
            if snap.screen == SCREEN_WHITE_SWORD_CAVE and snap.mode == PLAY_MODE:
                self._set_phase(WhiteSwordPhase.CLIMB_0A)
                return self._action(nes_idle_action(), "maze_north_arrived")
            if snap.mode != PLAY_MODE or snap.transitioning:
                return self._action(nes_action("UP"), "maze_north_scroll")
            # y=157 is the lane that carries Link east to the opening; it is
            # not where the climb starts. Re-checking it every frame makes the
            # align and the climb fight each other -- Link oscillated at
            # (208,150..155) for 20,000 frames. Latch once the column is
            # reached and then only ever hold UP.
            if not self.maze_lane_done:
                if abs(int(snap.link_x) - MAZE_NORTH_X) <= TOL:
                    self.maze_lane_done = True
                else:
                    d = _axis_step(int(snap.link_y), MAZE_STAND_Y, "UP", "DOWN")
                    if d is not None:
                        return self._action(nes_action(d), "maze_north_align_y")
                    d = _axis_step(int(snap.link_x), MAZE_NORTH_X, "LEFT", "RIGHT")
                    if d is not None:
                        return self._action(nes_action(d), "maze_north_align_x")
            d = _axis_step(int(snap.link_x), MAZE_NORTH_X, "LEFT", "RIGHT")
            if d is not None:
                return self._action(nes_action(d), "maze_north_hold_column")
            return self._action(nes_action("UP"), "maze_north_climb")

        # A lake fills the middle of 0x0A; the only north-south lane is the
        # x=208 sand corridor, and the cave mouth is on the top band far west.
        if self.phase is WhiteSwordPhase.CLIMB_0A:
            if snap.screen == SCREEN_MAZE_GATE and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(WhiteSwordPhase.MAZE_NORTH)
                self.lynel_wait = 0
                return self._action(nes_idle_action(), "lynel_reroll")
            gate = self._lynel_gate(snap)
            if gate is not None:
                return gate
            if snap.link_y <= TOP_BAND_REACHED_Y:
                self._set_phase(WhiteSwordPhase.TO_MOUTH)
                return self._action(nes_idle_action(), "climb_0a_done")
            d = _axis_step(int(snap.link_x), CORRIDOR_X, "LEFT", "RIGHT")
            if d is not None:
                return self._action(nes_action(d), "climb_0a_align_x")
            return self._beam_hold(snap, "UP") or self._action(nes_action("UP"), "climb_0a_up")

        if self.phase is WhiteSwordPhase.TO_MOUTH:
            if snap.in_cave or snap.mode == CAVE_MODE:
                self._set_phase(WhiteSwordPhase.TAKE_SWORD)
                return self._action(nes_idle_action(), "mouth_entered")
            d = _axis_step(int(snap.link_x), CAVE_MOUTH_X, "LEFT", "RIGHT")
            if d is not None:
                return self._beam_hold(snap, d) or self._action(nes_action(d), "to_mouth_west")
            self._set_phase(WhiteSwordPhase.ENTER_CAVE)
            return self._action(nes_action("UP"), "to_mouth_enter")

        if self.phase is WhiteSwordPhase.ENTER_CAVE:
            if snap.in_cave or snap.mode == CAVE_MODE:
                self._set_phase(WhiteSwordPhase.TAKE_SWORD)
                return self._action(nes_idle_action(), "cave_entered")
            if self.phase_frames > 240:
                self._set_phase(WhiteSwordPhase.TO_MOUTH)
                return self._action(nes_idle_action(), "cave_reapproach")
            return self._action(nes_action("UP"), "enter_cave_up")

        if self.phase is WhiteSwordPhase.TAKE_SWORD:
            if snap.sword >= WHITE_SWORD:
                self._set_phase(WhiteSwordPhase.EXIT_CAVE)
                return self._action(nes_action("DOWN"), "sword_taken")
            # The Old Man's text freezes input; walking early is harmless but
            # wasted, so wait it out once before crossing to the pedestal.
            self.dialog_waited += 1
            if self.dialog_waited < DIALOG_FRAMES:
                return self._action(nes_idle_action(), "cave_dialog_wait")
            d = _axis_step(int(snap.link_x), CAVE_ITEM_X, "LEFT", "RIGHT")
            if d is not None:
                return self._action(nes_action(d), "cave_align_item")
            return self._action(nes_action("UP"), "cave_take_item")

        if self.phase is WhiteSwordPhase.EXIT_CAVE:
            if snap.screen == SCREEN_WHITE_SWORD_CAVE and snap.mode == PLAY_MODE \
                    and not snap.transitioning and not snap.in_cave:
                self._set_phase(WhiteSwordPhase.RETURN_TOP)
                return self._action(nes_idle_action(), "cave_exited")
            return self._action(nes_action("DOWN"), "exit_cave_down")

        # The cave spits Link out at the top-LEFT of 0x0A. Heading straight
        # DOWN walks him into the lake's west shore and stalls at (32,189),
        # so re-cross the top band east before descending the corridor.
        if self.phase is WhiteSwordPhase.RETURN_TOP:
            if snap.in_cave or snap.mode == CAVE_MODE:
                self._set_phase(WhiteSwordPhase.EXIT_CAVE)
                self.lynel_wait = 0
                return self._action(nes_action("DOWN"), "lynel_reroll_cave")
            if int(snap.link_x) < CORRIDOR_X - 32:
                gate = self._lynel_gate(snap, back="UP", foot=lambda y: True)
                if gate is not None:
                    return gate
            d = _axis_step(int(snap.link_y), TOP_BAND_Y, "UP", "DOWN")
            if d is not None:
                return self._action(nes_action(d), "return_top_align_y")
            d = _axis_step(int(snap.link_x), CORRIDOR_X, "LEFT", "RIGHT")
            if d is not None:
                return self._beam_hold(snap, d) or self._action(nes_action(d), "return_top_east")
            self._set_phase(WhiteSwordPhase.RETURN_CORRIDOR)
            return self._action(nes_action("DOWN"), "return_top_done")

        if self.phase is WhiteSwordPhase.RETURN_CORRIDOR:
            if snap.link_y >= CORRIDOR_BOTTOM_Y:
                self._set_phase(WhiteSwordPhase.MAZE_SOUTH)
                return self._action(nes_action("DOWN"), "return_corridor_done")
            d = _axis_step(int(snap.link_x), CORRIDOR_X, "LEFT", "RIGHT")
            if d is not None:
                return self._action(nes_action(d), "return_corridor_align_x")
            return self._action(nes_action("DOWN"), "return_corridor_down")

        if self.phase is WhiteSwordPhase.MAZE_SOUTH:
            if snap.screen == SCREEN_MAZE_GATE and snap.mode == PLAY_MODE \
                    and not snap.transitioning:
                self._set_phase(WhiteSwordPhase.BACK_LEGS)
                return self._action(nes_idle_action(), "maze_south_arrived")
            return self._action(nes_action("DOWN"), "maze_south_down")

        if self.phase is WhiteSwordPhase.BACK_LEGS:
            act = self._walk_leg(snap, BACK_LEGS, WhiteSwordPhase.DONE)
            if self.phase is WhiteSwordPhase.DONE:
                self.success = snap.sword >= WHITE_SWORD
                if not self.success:
                    return self._fail("white_sword_returned_without_sword")
            return act

        if self.phase is WhiteSwordPhase.DONE:
            self.success = True
            return self._action(nes_idle_action(), "done")

        return self._fail(f"unknown_phase_{self.phase}")

    def report(self) -> dict[str, Any]:
        return {
            "chapter": "white_sword_detour",
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase.name,
            "frames": self.frames,
            "evidence": "live",
            "notes": list(self.notes),
            "reasons": dict(self.reasons),
            "controller_memory_writes": 0,
            "progression_writes": 0,
            "capacity_writes": 0,
            "inventory_writes": 0,
        }


def make_white_sword_detour_controller() -> WhiteSwordDetourController:
    return WhiteSwordDetourController()


__all__ = [
    "MIN_HEART_CONTAINERS", "SCREEN_WHITE_SWORD_CAVE", "WHITE_SWORD",
    "WhiteSwordDetourController", "WhiteSwordPhase",
    "make_white_sword_detour_controller",
]
