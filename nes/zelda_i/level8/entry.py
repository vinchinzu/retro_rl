"""Fail-closed natural Red-Candle entry from the measured post-L7 leave.

The start-based Blue Candle path remains recon-only; defaults cannot execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL8_BUSH
from zelda_i.dungeon.pause_select import PauseSelectController
from zelda_i.level7.dungeon import MEASURED_POST_L7_EXIT
from zelda_i.level8.overworld import (
    L7_POND_TO_LEVEL8_BUSH_HOPS,
    LEVEL8_5C_MAZE_WAYPOINTS,
    pond_42_north_strip_action,
    pond_reverse_to_l8_extra_hop_action,
)
from zelda_i.dungeon.hop_controller import ow_edge_band_step
from zelda_i.overworld.graph import ScreenHop, hop_exit_band, is_5c_maze_hop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.stitch import OverworldHandoff
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_ROD,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

LEVEL8, POST_L7_TRIFORCE = 8, 0x7F
CANDLE_RED, B_ITEM_CANDLE = 2, 4
ADDR_CANDLE_USED = 0x0513
APPROACH_MAX_FRAMES, SELECT_MAX_FRAMES, BURN_MAX_FRAMES = 40_000, 240, 1_200


@dataclass(frozen=True)
class PostLevel7Handoff:
    """Measured L7 leave and inventory required before L8 may move.

    Every nullable value is part of the eventual handoff packet.  ``verified``
    must stay false until the cumulative L7 owner reports the actual value.
    """

    screen: int | None = None
    link_x: int | None = None
    link_y: int | None = None
    keys: int | None = None
    bombs: int | None = None
    rupees: int | None = None
    heart_containers: int | None = None
    selected_item: int | None = None
    whistle: int | None = None
    food: int | None = None
    rod: int | None = None
    bow: int | None = None
    arrows: int | None = None
    candle: int = CANDLE_RED
    xy_tolerance: int = 4
    evidence: str = "hypothesis"
    verified: bool = False
    route_eligible: bool = False

    def complete(self) -> bool:
        measured = (
            self.screen,
            self.link_x,
            self.link_y,
            self.keys,
            self.bombs,
            self.rupees,
            self.heart_containers,
            self.selected_item,
            self.whistle,
            self.food,
            self.rod,
            self.bow,
            self.arrows,
        )
        return self.verified and all(value is not None for value in measured)

    def mismatch(self, snap: ZeldaSnapshot, ram: Any) -> str | None:
        if not self.complete():
            return "post_l7_handoff_unmeasured"
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return "post_l7_not_settled_overworld"
        if snap.screen != self.screen:
            return "post_l7_screen_mismatch"
        if abs(snap.link_x - int(self.link_x)) > self.xy_tolerance:
            return "post_l7_x_mismatch"
        if abs(snap.link_y - int(self.link_y)) > self.xy_tolerance:
            return "post_l7_y_mismatch"
        if snap.triforce != POST_L7_TRIFORCE:
            return "post_l7_triforce_mismatch"
        if not snap.health_is_full or snap.heart_containers != self.heart_containers:
            return "post_l7_health_mismatch"
        # Consumables are lower bounds, as on the post-L6 handoff: the
        # gathered spine arrives richer than the fixture tape measured.
        for label, actual, floor in (
            ("keys", snap.keys, self.keys),
            ("bombs", snap.bombs, self.bombs),
            ("rupees", snap.rupees, self.rupees),
        ):
            if int(actual) < int(floor):
                return f"post_l7_{label}_mismatch"
        for label, actual, expected in (
            ("selected_item", read_u8(ram, ADDR_SELECTED_ITEM), self.selected_item),
            ("whistle", read_u8(ram, ADDR_WHISTLE), self.whistle),
            ("food", read_u8(ram, ADDR_FOOD), self.food),
            ("rod", read_u8(ram, ADDR_ROD), self.rod),
            ("bow", read_u8(ram, ADDR_BOW), self.bow),
            ("arrows", read_u8(ram, ADDR_ARROWS), self.arrows),
            ("candle", read_u8(ram, ADDR_CANDLE), self.candle),
        ):
            if int(actual) != int(expected):
                return f"post_l7_{label}_mismatch"
        return None


UNMEASURED_POST_L7_HANDOFF = PostLevel7Handoff()


def handoff_from_overworld(packet: OverworldHandoff) -> PostLevel7Handoff:
    """Copy a shared leftover into the L8-shaped packet.

    Incomplete packets stay unmeasured. Isolated factories still default to
    ``UNMEASURED_POST_L7_HANDOFF``; the spine seam uses
    ``MEASURED_POST_L7_HANDOFF``.
    """
    if not packet.complete():
        return UNMEASURED_POST_L7_HANDOFF
    return PostLevel7Handoff(
        screen=packet.screen,
        link_x=packet.link_x,
        link_y=packet.link_y,
        keys=packet.keys,
        bombs=packet.bombs,
        rupees=packet.rupees,
        heart_containers=packet.heart_containers,
        selected_item=packet.selected_item,
        whistle=packet.whistle,
        food=packet.food,
        rod=packet.rod,
        bow=packet.bow,
        arrows=packet.arrows,
        candle=CANDLE_RED if packet.candle is None else int(packet.candle),
        xy_tolerance=packet.xy_tolerance,
        evidence=packet.evidence,
        verified=packet.verified,
        route_eligible=packet.route_eligible,
    )


MEASURED_POST_L7_HANDOFF = handoff_from_overworld(MEASURED_POST_L7_EXIT)


@dataclass(frozen=True)
class BushBurnTarget:
    """Exact fire placement, promoted only after live RAM/visual evidence."""

    link_x: int | None = None
    link_y: int | None = None
    facing: str | None = None
    push_direction: str | None = None
    tolerance: int = 4
    evidence: str = "hypothesis"
    verified: bool = False
    route_eligible: bool = False

    def complete(self) -> bool:
        return (
            self.verified
            and self.link_x is not None
            and self.link_y is not None
            and self.facing in {"UP", "DOWN", "LEFT", "RIGHT"}
            and self.push_direction in {"UP", "DOWN", "LEFT", "RIGHT"}
        )


# The canonical controller receives no executable target by default: a burn
# only runs once someone hands it a verified BushBurnTarget.
UNVERIFIED_BUSH_BURN_TARGET = BushBurnTarget()

# Live walkable (assisted, no candle): left corridor x≈32–56 plus mid sand
# y≈88–96 east to x≈144.  Only open exit without candle: UP @ x≈48 → 0x5D.
WALKABLE_LEFT_X = (32, 56)
WALKABLE_SAND_Y = (88, 96)
WALKABLE_SAND_X_MAX = 144
OPEN_EXIT_UP_X = 48

# Every stand that opened the mode-16 mouth in the 5856-trial sweep
# (logs/level8_bush_burn_sweep.json "near_misses"): one secret tile, several
# approach angles (sweep opened mode 16; only (136,93) RIGHT was live-walked to 0x7E).
# Facing == push on every one of them.
MOUTH_STANDS = (
    (120, 93, "RIGHT", "RIGHT"),
    (128, 93, "RIGHT", "RIGHT"),
    (136, 93, "RIGHT", "RIGHT"),
    (160, 77, "DOWN", "DOWN"),
    (184, 93, "LEFT", "LEFT"),
    (192, 93, "LEFT", "LEFT"),
    (200, 93, "LEFT", "LEFT"),
)

# Swept-verified default: the one stand the entrance fixture actually replayed
# into live L8 play (Level8EntranceReconFixture.provenance.json, entry room
# 0x7E at (120, 205)).
VERIFIED_BUSH_X = 136
VERIFIED_BUSH_Y = 93
VERIFIED_FACING = "RIGHT"
VERIFIED_PUSH = "RIGHT"
VERIFIED_BUSH_AIM = (VERIFIED_BUSH_X, VERIFIED_BUSH_Y)

# Refuted belief (rr-u9js): stand past the sampled east limit at (144, 93),
# fire RIGHT, then push UP because "dungeon mouths are mode-16 UP".  The sweep
# burned the candle at (144, 93) on all four facings and both pushes and never
# saw a mouth; (144, 93) is not a mouth stand at all.
REFUTED_BUSH_AIM = (144, 93)
REFUTED_FACING = "RIGHT"
REFUTED_PUSH = "UP"

# Fixture-only live recon (rr-6o7.1): nes/zelda_i/scratch/level8_bush_burn_sweep.py
# ran a 5856-trial live sweep from Level8BushWithCandleFixture (candle
# owned+selected, triforce 0x7F, Link teleported to a documented-standable OW
# 0x6D tile -- NOT the measured natural post-L7 walk, so this is fixture-live,
# not route evidence). Firing the Red Candle at (136, 93) facing RIGHT and
# continuing RIGHT reproducibly transitions mode 5 -> 16 -> ... -> 5 with
# level==8, landing at live screen 0x7E, (120, 205), facing UP. Six other
# stands -- (120, 93)/(128, 93) facing+push RIGHT, (184, 93)/(192, 93)/
# (200, 93) facing+push LEFT, and (160, 77) facing+push DOWN -- opened the
# mode-16 mouth only: entry_room is null on every one of them, so a shared
# secret tile is an inference, NOT a reproduced entry. (136, 93) RIGHT/RIGHT
# is the sole stand ever carried into a room id, and the only one of the
# three RIGHT stands inside the walked sand channel, so it is also the only
# aim a natural approach can use. Captured as
# Level8EntranceReconFixture (see nes/zelda_i/scratch/capture_level8_entrance_fixture.py
# and its .provenance.json). The sweep never reached level 8 by pushing UP
# after mode==16 (entry_room is null on all seven mouth stands); the fixture
# reached it by continuing the same push_direction it fired with, so
# BurnLevel8BushController's ENTER phase now sends target.push_direction
# (rr-i6hq). Evidence is fixture-live: verified=True (it reliably
# reproduces), but route_eligible stays False since the real predecessor is
# still the unmeasured 0x42→0x6D walk, not a natural post-L7 approach.
LIVE_RECON_BUSH_BURN_TARGET = BushBurnTarget(
    link_x=VERIFIED_BUSH_X,
    link_y=VERIFIED_BUSH_Y,
    facing=VERIFIED_FACING,
    push_direction=VERIFIED_PUSH,
    tolerance=4,
    evidence="live_recon_fixture",
    verified=True,
    route_eligible=False,
)


class ApproachPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostLevel7ToBushController(OverworldPathController):
    """Measured L7 leave → L8 bush screen, with exact inventory preservation."""

    handoff: PostLevel7Handoff = UNMEASURED_POST_L7_HANDOFF
    hops: tuple[ScreenHop, ...] = ()
    phase: ApproachPhase = ApproachPhase.HOP
    max_frames: int = APPROACH_MAX_FRAMES
    require_sword: bool = True
    maze_waypoints: tuple[tuple[int, int], ...] = LEVEL8_5C_MAZE_WAYPOINTS
    maze_hop_pred: Any = None
    _env: Any = field(default=None, init=False, repr=False)
    _handoff_checked: bool = field(default=False, init=False, repr=False)
    _lattice_hop: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maze_hop_pred is None:
            self.maze_hop_pred = is_5c_maze_hop

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail_now(self, reason: str) -> FrameAction:
        self._set_phase(ApproachPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        if not self._handoff_checked or self._env is None:
            return False
        return (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == SCREEN_LEVEL8_BUSH
            and snap.triforce == POST_L7_TRIFORCE
            and read_u8(self._env.get_ram(), ADDR_CANDLE) == CANDLE_RED
        )

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._at_stop(snap):
            return self._finish("level8_bush_reached_from_post_l7")
        return self._fail_now("post_l7_path_exhausted_off_0x6d")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        ring = pond_42_north_strip_action(snap, swing=self._swing)
        if ring is not None:
            return ring
        extra = pond_reverse_to_l8_extra_hop_action(
            snap, hop, swing=self._swing
        )
        if extra is not None:
            return extra
        if self.stuck > self.stuck_threshold:
            self._lattice_hop = self.hop_index
        source = self.hops[self.hop_index - 1].target if self.hop_index else None
        if self._lattice_hop == self.hop_index and snap.screen == source:
            # Stalled on this hop: the ROM lattice to its exit band for the
            # rest of it. The idle below sat 34,960f in the 0x5C maze on the
            # power-on gathered spine.
            lo, hi = hop_exit_band(hop)
            step = ow_edge_band_step(self._env, snap, hop.direction, lo, hi)
            if step is not None:
                return self._swing(step, "post_l7_lattice")
        if self.stuck > self.stuck_threshold:
            return FrameAction(nes_idle_action(), "post_l7_path_stuck_wait")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self._handoff_checked:
            if self._env is None:
                return self._fail_now("entry_controller_env_not_bound")
            mismatch = self.handoff.mismatch(snap, self._env.get_ram())
            if mismatch is not None:
                return self._fail_now(mismatch)
            if not self.hops and snap.screen != SCREEN_LEVEL8_BUSH:
                return self._fail_now("post_l7_path_unmeasured")
            if self.hops and self.hops[-1].target != SCREEN_LEVEL8_BUSH:
                return self._fail_now("post_l7_path_does_not_end_0x6d")
            self._handoff_checked = True
            self.notes.append("post_l7_handoff_accepted")
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.handoff.evidence,
                "route_eligible": self.handoff.route_eligible,
                "failed": self.phase is ApproachPhase.FAILED,
                "writes": 0,
            }
        )
        return out


@dataclass
class SelectRedCandleController:
    """Select the already-owned Red Candle through the pause menu only."""

    max_frames: int = SELECT_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    selected_before: int | None = None
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(
            want=B_ITEM_CANDLE, name="candle", max_frames=self.max_frames
        )

    @property
    def phase(self):
        return self._select.phase

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._select.failed = True
        self._select.fail_reason = reason
        if reason not in self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self._fail("select_candle_timeout")
        if self._env is None:
            return self._fail("select_candle_env_not_bound")
        ram = self._env.get_ram()
        selected = read_u8(ram, ADDR_SELECTED_ITEM)
        if self.selected_before is None:
            self.selected_before = selected
            if (
                snap.level != 0
                or snap.mode != PLAY_MODE
                or snap.screen != SCREEN_LEVEL8_BUSH
                or snap.triforce != POST_L7_TRIFORCE
                or read_u8(ram, ADDR_CANDLE) != CANDLE_RED
            ):
                return self._fail("select_candle_entry_contract_mismatch")
        if snap.mode == 17:
            return self._fail("link_death")
        action = self._select.drive(snap)
        for note in self._select.notes:
            if note not in self.notes:
                self.notes.append(note)
        if self._select.failed:
            return self._fail(self._select.fail_reason or "select_candle_failed")
        if action is None:
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL8_BUSH
            ):
                self.success = True
                return FrameAction(nes_idle_action(), "done")
            return self._fail("pause_close_contract_mismatch")
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase.name,
            "frames": self.frames,
            "cursor_moves": self.cursor_moves,
            "normal_pause_input": True,
            "writes": 0,
            "notes": list(self.notes),
        }


class BurnPhase(Enum):
    AIM = auto()
    FIRE = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class BurnLevel8BushController:
    """Use Red Candle at a verified tile and require the live L8 transition."""

    target: BushBurnTarget = UNVERIFIED_BUSH_BURN_TARGET
    max_frames: int = BURN_MAX_FRAMES
    burn_budget: int = 800
    phase: BurnPhase = BurnPhase.AIM
    frames: int = 0
    burn_frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    candle_use_observed: bool = False
    observed_entry_room: int | None = None
    _env: Any = field(default=None, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _set_phase(self, phase: BurnPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(BurnPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self._fail("level8_entry_timeout")
        if self._env is None:
            return self._fail("burn_controller_env_not_bound")
        ram = self._env.get_ram()
        candle = read_u8(ram, ADDR_CANDLE)
        selected = read_u8(ram, ADDR_SELECTED_ITEM)
        candle_used = read_u8(ram, ADDR_CANDLE_USED)
        self.candle_use_observed = self.candle_use_observed or candle_used != 0

        if not self._validated:
            if not self.target.complete():
                return self._fail("bush_burn_target_unverified")
            if (
                snap.level != 0
                or snap.mode != PLAY_MODE
                or snap.screen != SCREEN_LEVEL8_BUSH
                or snap.triforce != POST_L7_TRIFORCE
                or candle != CANDLE_RED
                or selected != B_ITEM_CANDLE
                or candle_used != 0
            ):
                return self._fail("bush_burn_entry_contract_mismatch")
            self._validated = True
            self.notes.append("verified_burn_target_accepted")

        if snap.mode == 17:
            return self._fail("link_death")
        if snap.triforce != POST_L7_TRIFORCE or candle != CANDLE_RED:
            return self._fail("level8_entry_inventory_changed")
        if snap.level == LEVEL8:
            if not self.candle_use_observed:
                return self._fail("level8_entered_without_observed_candle_use")
            if snap.mode == PLAY_MODE:
                self.observed_entry_room = snap.screen
                self.success = True
                self._set_phase(BurnPhase.DONE, "level8_live_entry")
                return FrameAction(nes_idle_action(), "done")
            # l8_entry_burn4: level=8 mode=2 (black load) still had $EB=0x6D.
            # Wait for play. Do not trip left_bush_screen.
            self._set_phase(BurnPhase.ENTER, "level8_transition")
            return FrameAction(
                nes_action(str(self.target.push_direction)), "enter_level8_settle"
            )
        if self.burn_frames >= self.burn_budget:
            # Being controllable on 0x6D is approach evidence, never entry.
            return self._fail("burn_budget_exhausted_without_level8_entry")
        self.burn_frames += 1

        # rr-i6hq: the mouth does not swallow Link on UP. logs/
        # level8_bush_burn_sweep.json opened mode 16 at seven stands and logged
        # entry_room=null on every one; Level8EntranceReconFixture only reached
        # live L8 (0x7E, 111 frames past the push) by continuing the same
        # push_direction used to fire.  complete() already constrains this to a
        # cardinal, so UP is used here only when UP is the recorded push.
        enter = nes_action(str(self.target.push_direction))
        if snap.mode == 16:
            if not self.candle_use_observed:
                return self._fail("mouth_transition_without_candle_use")
            self._set_phase(BurnPhase.ENTER, "mouth_transition_observed")
            return FrameAction(enter, "enter_level8")
        if self.phase is BurnPhase.ENTER and snap.transitioning:
            return FrameAction(enter, "enter_level8_transition")
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.screen != SCREEN_LEVEL8_BUSH:
            return self._fail("left_bush_screen_without_level8_entry")
        if self.phase is BurnPhase.ENTER:
            return FrameAction(enter, "enter_level8")

        tx = int(self.target.link_x)
        ty = int(self.target.link_y)
        fire_tol = 2
        # Power-on leftover arrives 0x6D (48,61) from 0x5D south. RIGHT at
        # y=61/86 is trees (l8_entry_burn / burn2). Reach the aim before
        # east. Once FIRE, do not re-align: LEFT-correcting the RIGHT push
        # walks off the mouth (l8_entry_burn3, (142,93), candle used, no
        # mode 16).
        if self.phase is not BurnPhase.FIRE:
            if snap.link_y < ty - fire_tol:
                return FrameAction(nes_action("DOWN"), "bush_burn_drop_to_channel")
            if abs(snap.link_x - tx) > fire_tol:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < tx else "LEFT"),
                    "bush_burn_align_x",
                )
            if abs(snap.link_y - ty) > fire_tol:
                return FrameAction(
                    nes_action("DOWN" if snap.link_y < ty else "UP"),
                    "bush_burn_align_y",
                )
            self._set_phase(BurnPhase.FIRE)
        cycle = self.phase_frames % 36
        if cycle < 4:
            return FrameAction(nes_action(str(self.target.facing)), "bush_face")
        if cycle < 12:
            return FrameAction(nes_action("B"), "red_candle_fire")
        return FrameAction(
            nes_action(str(self.target.push_direction)), "push_revealed_mouth"
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase.name,
            "frames": self.frames,
            "burn": [self.burn_frames, self.burn_budget],
            "candle_use_observed": self.candle_use_observed,
            "observed_entry_room": self.observed_entry_room,
            "route_eligible": self.target.route_eligible,
            "writes": 0,
            "notes": list(self.notes),
        }


# The old 60R shop controller is deliberately absent from every public L8 hop.
BLUE_CANDLE_FALLBACK_ENABLED = False
BLUE_CANDLE_FALLBACK_ROUTE_ELIGIBLE = False


def make_post_l7_to_bush_controller(
    *,
    handoff: PostLevel7Handoff = UNMEASURED_POST_L7_HANDOFF,
    hops: tuple[ScreenHop, ...] = (),
) -> PostLevel7ToBushController:
    return PostLevel7ToBushController(handoff=handoff, hops=hops)


def make_select_red_candle_controller() -> SelectRedCandleController:
    return SelectRedCandleController()


def make_burn_level8_bush_controller(
    *, target: BushBurnTarget = UNVERIFIED_BUSH_BURN_TARGET
) -> BurnLevel8BushController:
    return BurnLevel8BushController(target=target)


class ReconBurnPhase(Enum):
    AIM = auto()
    FIRE = auto()
    ENTER = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class IsolatedBushReconController:
    """Fixture-live 0x6D burn trial. Budget exhaust on 0x6D is failure."""

    link_x: int = VERIFIED_BUSH_X
    link_y: int = VERIFIED_BUSH_Y
    facing: str = VERIFIED_FACING
    push_direction: str = VERIFIED_PUSH
    tolerance: int = 4
    max_frames: int = BURN_MAX_FRAMES
    burn_budget: int = 800
    phase: ReconBurnPhase = ReconBurnPhase.AIM
    frames: int = 0
    burn_frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    candle_use_observed: bool = False
    observed_entry_room: int | None = None
    evidence: str = "fixture-live"
    route_eligible: bool = False
    _env: Any = field(default=None, init=False, repr=False)
    _validated: bool = field(default=False, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _set_phase(self, phase: ReconBurnPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(ReconBurnPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            return self._fail("level8_entry_timeout")
        if self._env is None:
            return self._fail("bush_recon_env_not_bound")
        ram = self._env.get_ram()
        candle = read_u8(ram, ADDR_CANDLE)
        selected = read_u8(ram, ADDR_SELECTED_ITEM)
        candle_used = read_u8(ram, ADDR_CANDLE_USED)
        self.candle_use_observed = self.candle_use_observed or candle_used != 0

        if not self._validated:
            if (
                snap.level != 0
                or snap.mode != PLAY_MODE
                or snap.screen != SCREEN_LEVEL8_BUSH
            ):
                return self._fail("bush_recon_not_on_0x6d")
            if candle == 0:
                return self._fail("bush_recon_candle_unowned")
            if selected != B_ITEM_CANDLE:
                return self._fail("bush_recon_candle_not_selected")
            self._validated = True
            self.notes.append("fixture_live_bush_recipe_accepted")
            self.notes.append("refuted_aim_144_93_right_face_up_push")

        if snap.mode == 17:
            return self._fail("link_death")
        if snap.level == LEVEL8:
            if not self.candle_use_observed:
                return self._fail("level8_entered_without_observed_candle_use")
            if snap.mode == PLAY_MODE:
                self.observed_entry_room = snap.screen
                self.success = True
                self._set_phase(ReconBurnPhase.DONE, "level8_live_entry")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(ReconBurnPhase.ENTER, "level8_transition")
            return FrameAction(nes_action(self.push_direction), "enter_level8_settle")
        if self.burn_frames >= self.burn_budget:
            return self._fail("burn_budget_exhausted_without_level8_entry")
        self.burn_frames += 1

        # rr-i6hq: UP after mode 16 does not complete the transition here.  The
        # sweep opened the mouth at seven stands and recorded entry_room=null on
        # every one; the entrance fixture only reached live L8 by continuing the
        # push direction it fired with.
        enter = nes_action(self.push_direction)
        if snap.mode == 16:
            if not self.candle_use_observed:
                return self._fail("mouth_transition_without_candle_use")
            self._set_phase(ReconBurnPhase.ENTER, "mouth_transition_observed")
            return FrameAction(enter, "enter_level8")
        if self.phase is ReconBurnPhase.ENTER and snap.transitioning:
            return FrameAction(enter, "enter_level8_transition")
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.screen != SCREEN_LEVEL8_BUSH:
            return self._fail("left_bush_screen_without_level8_entry")
        if self.phase is ReconBurnPhase.ENTER:
            return FrameAction(enter, "enter_level8")

        if abs(snap.link_x - self.link_x) > self.tolerance:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < self.link_x else "LEFT"),
                "bush_burn_align_x",
            )
        if abs(snap.link_y - self.link_y) > self.tolerance:
            return FrameAction(
                nes_action("DOWN" if snap.link_y < self.link_y else "UP"),
                "bush_burn_align_y",
            )
        self._set_phase(ReconBurnPhase.FIRE)
        cycle = self.phase_frames % 36
        if cycle < 4:
            return FrameAction(nes_action(self.facing), "bush_face")
        if cycle < 12:
            return FrameAction(nes_action("B"), "red_candle_fire")
        return FrameAction(nes_action(self.push_direction), "push_revealed_mouth")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase.name,
            "frames": self.frames,
            "burn": [self.burn_frames, self.burn_budget],
            "aim": [self.link_x, self.link_y, self.facing, self.push_direction],
            "refuted_aim": [*REFUTED_BUSH_AIM, REFUTED_FACING, REFUTED_PUSH],
            "candle_use_observed": self.candle_use_observed,
            "observed_entry_room": self.observed_entry_room,
            "evidence": self.evidence,
            "route_eligible": self.route_eligible,
            "writes": 0,
            "triforce": POST_L7_TRIFORCE,
            "candle_red": CANDLE_RED,
            "notes": list(self.notes),
        }


def make_isolated_bush_recon_controller() -> IsolatedBushReconController:
    return IsolatedBushReconController()


__all__ = [
    "ADDR_CANDLE_USED",
    "APPROACH_MAX_FRAMES",
    "ApproachPhase",
    "B_ITEM_CANDLE",
    "BLUE_CANDLE_FALLBACK_ENABLED",
    "BLUE_CANDLE_FALLBACK_ROUTE_ELIGIBLE",
    "BURN_MAX_FRAMES",
    "BurnLevel8BushController",
    "BurnPhase",
    "BushBurnTarget",
    "CANDLE_RED",
    "IsolatedBushReconController",
    "LEVEL8",
    "LIVE_RECON_BUSH_BURN_TARGET",
    "MEASURED_POST_L7_HANDOFF",
    "MOUTH_STANDS",
    "OPEN_EXIT_UP_X",
    "POST_L7_TRIFORCE",
    "PostLevel7Handoff",
    "PostLevel7ToBushController",
    "REFUTED_BUSH_AIM",
    "REFUTED_FACING",
    "REFUTED_PUSH",
    "ReconBurnPhase",
    "SELECT_MAX_FRAMES",
    "SelectRedCandleController",
    "UNMEASURED_POST_L7_HANDOFF",
    "UNVERIFIED_BUSH_BURN_TARGET",
    "handoff_from_overworld",
    "VERIFIED_BUSH_AIM",
    "VERIFIED_BUSH_X",
    "VERIFIED_BUSH_Y",
    "VERIFIED_FACING",
    "VERIFIED_PUSH",
    "WALKABLE_LEFT_X",
    "WALKABLE_SAND_X_MAX",
    "WALKABLE_SAND_Y",
    "make_burn_level8_bush_controller",
    "make_isolated_bush_recon_controller",
    "make_post_l7_to_bush_controller",
    "make_select_red_candle_controller",
]
