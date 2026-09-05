"""Fail-closed natural Level 9 and write-free ending controllers.

The natural prefix controllers intentionally refuse to move until decoded ROM
topology and measured predecessor inventory are supplied.  The ending adapter
accepts controller input only; it never loads a fixture or writes inventory,
doors, rooms, progression, or capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level9.dungeon import (
    BOMBS_NOT_NATURAL,
    FULL_TRIFORCE,
    LEVEL9,
    MAGICAL_SWORD,
    MISSING_OLD_MAN_GATE,
    MISSING_POST_L8_LEFTOVER,
    MISSING_SILVER_ARROW_ROOM,
    MISSING_SPECTACLE_BOMB,
    MISSING_51_NORTH_WALK,
    PostLevel8Handoff,
    SILVER_ARROWS,
    TRIFORCE_NOT_FULL,
    UNMEASURED_POST_L8_HANDOFF,
    level9_credits_stop,
    level9_live_patra_stop,
)
from zelda_i.level9.ganon import (
    B_ITEM_ARROWS,
    MODE_ENDING,
    NORTH_DOOR,
    ROOM_GANON,
    ROOM_ZELDA,
    ganon_action,
    ganon_defeated,
    ganon_object,
    in_ganon_fight,
    in_zelda_room,
)
from zelda_i.level9.patra import final_patra_north_door_earned, patra_action
from zelda_i.dungeon.pause_select import PauseSelectController
from zelda_i.level9.path import final_patra_to_ganon_step
from zelda_i.ram import ADDR_SELECTED_ITEM, PLAY_MODE, ZeldaSnapshot, read_u8


@dataclass
class NaturalRouteUnavailableController:
    """One-frame fail-closed marker. Refuses without TF 0xFF / bombs; never writes."""

    chapter: str
    reason: str
    require_bombs: bool = False
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF
    max_frames: int = 1
    frames: int = 0
    success: bool = False
    failed: bool = False
    blocked_reason: str = ""

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.failed = True
        if snap.triforce != FULL_TRIFORCE:
            reason = TRIFORCE_NOT_FULL
        elif self.require_bombs and snap.bombs < 1:
            reason = BOMBS_NOT_NATURAL
        elif self.chapter == "level9_post_l8_overworld":
            reason = self.handoff.mismatch(snap) or self.reason
        else:
            reason = self.reason
        self.blocked_reason = reason
        return FrameAction(nes_idle_action(), reason)

    def report(self) -> dict[str, object]:
        return {
            "chapter": self.chapter,
            "evidence": "hypothesis",
            "route_eligible": False,
            "success": False,
            "failed": self.failed,
            "frames": self.frames,
            "reason": self.blocked_reason or self.reason,
            "missing_evidence": self.reason,
            "controller_memory_writes": 0,
            "progression_writes": 0,
            "capacity_writes": 0,
            "inventory_writes": 0,
            "triforce_writes": 0,
            "bomb_capacity_writes": 0,
            "room_writes": 0,
            "door_writes": 0,
        }


def make_post_l8_overworld_controller(
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> NaturalRouteUnavailableController:
    return NaturalRouteUnavailableController(
        "level9_post_l8_overworld",
        MISSING_POST_L8_LEFTOVER,
        require_bombs=True,
        handoff=handoff,
    )


def make_spectacle_rock_bomb_controller() -> NaturalRouteUnavailableController:
    return NaturalRouteUnavailableController(
        "level9_spectacle_rock_bomb",
        MISSING_SPECTACLE_BOMB,
        require_bombs=True,
    )


def make_old_man_tf_gate_controller() -> NaturalRouteUnavailableController:
    return NaturalRouteUnavailableController(
        "level9_old_man_tf_gate",
        MISSING_OLD_MAN_GATE,
        require_bombs=True,
    )


def make_silver_arrows_unavailable_controller() -> NaturalRouteUnavailableController:
    return NaturalRouteUnavailableController(
        "level9_natural_silver_arrows",
        MISSING_SILVER_ARROW_ROOM,
    )


def make_patra_join_unavailable_controller() -> NaturalRouteUnavailableController:
    return NaturalRouteUnavailableController(
        "level9_natural_patra_join",
        MISSING_51_NORTH_WALK,
    )


@dataclass
class _NaturalEndingController:
    """Shared diagnostics for controller-input-only ending stages."""

    max_frames: int
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def _action(self, action: list[int], reason: str) -> FrameAction:
        self.frames += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return FrameAction(action, reason)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self.notes.append(reason)
        return self._action(nes_idle_action(), reason)

    def report(self) -> dict[str, object]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "evidence": "fixture-live",
            "notes": list(self.notes),
            "reasons": dict(self.reasons),
            "fixture_loaded": False,
            "route_eligible": False,
            "controller_memory_writes": 0,
            "progression_writes": 0,
            "capacity_writes": 0,
            "inventory_writes": 0,
            "selected_item_writes": 0,
        }


@dataclass
class NaturalFinalPatraController(_NaturalEndingController):
    """Adapt the proven Patra policy only from the exact natural join state."""

    max_frames: int = 6000
    cooldown: int = 0
    start_checked: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not level9_live_patra_stop(snap):
                return self._fail("natural_live_patra_contract_miss")
        if final_patra_north_door_earned(snap):
            self.success = True
            return self._action(nes_idle_action(), "patra_north_door_earned")
        action, reason, self.cooldown = patra_action(
            snap,
            cooldown=self.cooldown,
        )
        return self._action(action, reason)


@dataclass
class NaturalSelectSilverArrowsController(_NaturalEndingController):
    """Select naturally owned Silver Arrows through the pause menu only."""

    max_frames: int = 240
    env: Any | None = field(default=None, repr=False)
    _contract_checked: bool = False
    _select: PauseSelectController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(want=B_ITEM_ARROWS, name="arrows")

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    @property
    def phase(self) -> str:
        return self._select.phase.name.lower()

    def bind_env(self, env: Any) -> None:
        self.env = env
        self._select.bind_env(env)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if self.env is None:
            return self._fail("environment_not_bound")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self._contract_checked:
            if not level9_live_patra_stop(snap):
                return self._fail("natural_live_patra_contract_miss")
            self._contract_checked = True
        action = self._select.drive(snap)
        for note in self._select.notes:
            if note not in self.notes:
                self.notes.append(note)
        if self._select.failed:
            return self._fail(self._select.fail_reason or "pause_select_failed")
        if action is None:
            if not level9_live_patra_stop(snap):
                return self._fail("patra_contract_lost_after_pause")
            self.success = True
            reason = (
                "silver_arrows_already_selected"
                if self._select.skipped
                else "silver_arrows_selected"
            )
            return self._action(nes_idle_action(), reason)
        return self._action(action.action, action.reason)

    def report(self) -> dict[str, object]:
        report = super().report()
        report.update(
            {
                "phase": self.phase,
                "cursor_moves": self.cursor_moves,
                "selection_method": "bounded_pause_menu_input",
                "selected_item_writes": 0,
            }
        )
        return report


@dataclass
class NaturalPatraToGanonController(_NaturalEndingController):
    max_frames: int = 900
    start_checked: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not final_patra_north_door_earned(snap):
                return self._fail("patra_north_door_not_earned")
        if in_ganon_fight(snap):
            self.success = True
            return self._action(nes_idle_action(), "ganon_arrived")
        frame = final_patra_to_ganon_step(snap)
        if frame.reason.startswith("unexpected_room"):
            return self._fail(frame.reason)
        return self._action(frame.action, frame.reason)


@dataclass
class NaturalGanonController(_NaturalEndingController):
    """Ganon policy with an earned-inventory gate and no B-slot fallback write."""

    max_frames: int = 7000
    cooldown: int = 0
    start_checked: bool = False
    env: Any | None = field(default=None, repr=False)

    def bind_env(self, env: Any) -> None:
        self.env = env

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if self.env is None:
            return self._fail("environment_not_bound")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            selected = read_u8(self.env.get_ram(), ADDR_SELECTED_ITEM)
            if not (
                in_ganon_fight(snap)
                and snap.triforce == FULL_TRIFORCE
                and snap.sword >= MAGICAL_SWORD
                and snap.bow > 0
                and snap.arrows == SILVER_ARROWS
                and selected == B_ITEM_ARROWS
            ):
                return self._fail("earned_ganon_inventory_or_b_slot_contract_miss")
        if ganon_defeated(self.env.get_ram()):
            self.success = True
            return self._action(nes_idle_action(), "ganon_defeated")
        action, reason, self.cooldown = ganon_action(
            snap,
            cooldown=self.cooldown,
        )
        return self._action(action, reason)


@dataclass
class NaturalPowerTriforceController(_NaturalEndingController):
    max_frames: int = 1400
    start_checked: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not (
                snap.level == LEVEL9
                and snap.screen == ROOM_GANON
                and snap.mode == PLAY_MODE
                and snap.triforce == FULL_TRIFORCE
            ):
                return self._fail("expected_ganon_room_after_defeat")
        if snap.cur_opened_doors & NORTH_DOOR:
            self.success = True
            return self._action(nes_idle_action(), "power_triforce_collected")
        boss = ganon_object(snap)
        if boss is None:
            return self._action(nes_idle_action(), "wait_power_triforce")
        if abs(snap.link_x - boss.x) > 4:
            direction = "RIGHT" if snap.link_x < boss.x else "LEFT"
        elif abs(snap.link_y - boss.y) > 4:
            direction = "DOWN" if snap.link_y < boss.y else "UP"
        else:
            return self._action(nes_idle_action(), "collect_power_triforce")
        return self._action(nes_action(direction), "approach_power_triforce")


@dataclass
class NaturalEnterZeldaController(_NaturalEndingController):
    max_frames: int = 1200
    start_checked: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not (
                snap.level == LEVEL9
                and snap.screen == ROOM_GANON
                and snap.mode == PLAY_MODE
                and snap.triforce == FULL_TRIFORCE
                and (snap.cur_opened_doors & NORTH_DOOR)
            ):
                return self._fail("ganon_north_door_not_earned")
        if in_zelda_room(snap):
            self.success = True
            return self._action(nes_idle_action(), "zelda_room_arrived")
        if snap.screen not in (ROOM_GANON, ROOM_ZELDA):
            return self._fail(f"unexpected_room_0x{snap.screen:02x}")
        if snap.screen == ROOM_GANON and abs(snap.link_x - 0x78) > 4:
            direction = "RIGHT" if snap.link_x < 0x78 else "LEFT"
            return self._action(nes_action(direction), "zelda_align_x")
        return self._action(nes_action("UP"), "zelda_push_north")


@dataclass
class NaturalRescueZeldaController(_NaturalEndingController):
    max_frames: int = 3500
    start_checked: bool = False

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not in_zelda_room(snap):
                return self._fail("expected_live_zelda_room")
        if level9_credits_stop(snap) or snap.mode == MODE_ENDING:
            self.success = True
            return self._action(nes_idle_action(), "ending_started")
        if snap.screen != ROOM_ZELDA:
            return self._fail(f"unexpected_room_0x{snap.screen:02x}")
        if snap.link_x < 0x70:
            direction = "RIGHT"
        elif snap.link_x > 0x80:
            direction = "LEFT"
        elif snap.link_y > 0x95:
            direction = "UP"
        elif snap.link_y < 0x95:
            direction = "DOWN"
        else:
            direction = "UP"
        buttons = (direction, "A") if self.frames % 12 == 0 else (direction,)
        return self._action(nes_action(*buttons), "clear_guard_fires")


@dataclass
class NaturalCreditsController(_NaturalEndingController):
    max_frames: int = 12000

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if level9_credits_stop(snap):
            self.success = True
            return self._action(nes_idle_action(), "credits_or_final_page")
        if snap.mode == 17:
            return self._fail("link_death")
        if snap.mode != MODE_ENDING:
            return self._fail(f"ending_mode_lost_{snap.mode}")
        return self._action(nes_idle_action(), "wait_credits")


__all__ = [
    "NaturalCreditsController",
    "NaturalEnterZeldaController",
    "NaturalFinalPatraController",
    "NaturalGanonController",
    "NaturalPatraToGanonController",
    "NaturalPowerTriforceController",
    "NaturalRescueZeldaController",
    "NaturalRouteUnavailableController",
    "NaturalSelectSilverArrowsController",
    "make_old_man_tf_gate_controller",
    "make_patra_join_unavailable_controller",
    "make_post_l8_overworld_controller",
    "make_silver_arrows_unavailable_controller",
    "make_spectacle_rock_bomb_controller",
]
