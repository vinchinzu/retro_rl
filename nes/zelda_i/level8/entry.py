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
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
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
        for label, actual, expected in (
            ("keys", snap.keys, self.keys),
            ("bombs", snap.bombs, self.bombs),
            ("rupees", snap.rupees, self.rupees),
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


# The legacy recon aimed at (136, 93), but never opened the mouth.  Keep that
# belief in docs; the canonical controller receives no executable target.
UNVERIFIED_BUSH_BURN_TARGET = BushBurnTarget()


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
    _env: Any = field(default=None, init=False, repr=False)
    _handoff_checked: bool = field(default=False, init=False, repr=False)

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
        self, _snap: ZeldaSnapshot, _hop: ScreenHop
    ) -> FrameAction | None:
        # A new stuck position is evidence to inspect, not permission to jitter.
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


class SelectPhase(Enum):
    OPEN = auto()
    OPEN_SETTLE = auto()
    CYCLE = auto()
    CURSOR_SETTLE = auto()
    CLOSE = auto()
    CLOSE_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class SelectRedCandleController:
    """Select the already-owned Red Candle through the pause menu only."""

    max_frames: int = SELECT_MAX_FRAMES
    phase: SelectPhase = SelectPhase.OPEN
    frames: int = 0
    phase_frames: int = 0
    cursor_moves: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    selected_before: int | None = None
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _set_phase(self, phase: SelectPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(SelectPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
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
        if self.phase is SelectPhase.OPEN:
            if selected == B_ITEM_CANDLE:
                self.success = True
                self._set_phase(SelectPhase.DONE, "red_candle_already_selected")
                return FrameAction(nes_idle_action(), "done")
            self._set_phase(SelectPhase.OPEN_SETTLE, "pause_open")
            return FrameAction(nes_action("START"), "pause_open")
        if self.phase is SelectPhase.OPEN_SETTLE:
            if self.phase_frames >= 20:
                self._set_phase(SelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_settle")
        if self.phase is SelectPhase.CYCLE:
            if selected == B_ITEM_CANDLE:
                self._set_phase(SelectPhase.CLOSE, "candle_cursor_selected")
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= 8:
                return self._fail("candle_cursor_not_found")
            self.cursor_moves += 1
            self._set_phase(SelectPhase.CURSOR_SETTLE)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")
        if self.phase is SelectPhase.CURSOR_SETTLE:
            if self.phase_frames >= 8:
                self._set_phase(SelectPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_cursor_settle")
        if self.phase is SelectPhase.CLOSE:
            self._set_phase(SelectPhase.CLOSE_SETTLE, "pause_close")
            return FrameAction(nes_action("START"), "pause_close")
        if self.phase is SelectPhase.CLOSE_SETTLE:
            if self.phase_frames < 24:
                return FrameAction(nes_idle_action(), "pause_resume")
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and snap.screen == SCREEN_LEVEL8_BUSH
                and selected == B_ITEM_CANDLE
            ):
                self.success = True
                self._set_phase(SelectPhase.DONE, "red_candle_selected_naturally")
                return FrameAction(nes_idle_action(), "done")
            return self._fail("pause_close_contract_mismatch")
        return FrameAction(nes_idle_action(), "done")

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
        if snap.level == LEVEL8 and snap.mode == PLAY_MODE:
            if not self.candle_use_observed:
                return self._fail("level8_entered_without_observed_candle_use")
            self.observed_entry_room = snap.screen
            self.success = True
            self._set_phase(BurnPhase.DONE, "level8_live_entry")
            return FrameAction(nes_idle_action(), "done")
        if self.burn_frames >= self.burn_budget:
            # Being controllable on 0x6D is approach evidence, never entry.
            return self._fail("burn_budget_exhausted_without_level8_entry")
        self.burn_frames += 1

        if snap.mode == 16:
            if not self.candle_use_observed:
                return self._fail("mouth_transition_without_candle_use")
            self._set_phase(BurnPhase.ENTER, "mouth_transition_observed")
            return FrameAction(nes_action("UP"), "enter_level8")
        if self.phase is BurnPhase.ENTER and snap.transitioning:
            return FrameAction(nes_action("UP"), "enter_level8_transition")
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.screen != SCREEN_LEVEL8_BUSH:
            return self._fail("left_bush_screen_without_level8_entry")
        if self.phase is BurnPhase.ENTER:
            return FrameAction(nes_action("UP"), "enter_level8")

        tx = int(self.target.link_x)
        ty = int(self.target.link_y)
        if abs(snap.link_x - tx) > self.target.tolerance:
            return FrameAction(
                nes_action("RIGHT" if snap.link_x < tx else "LEFT"),
                "bush_burn_align_x",
            )
        if abs(snap.link_y - ty) > self.target.tolerance:
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
