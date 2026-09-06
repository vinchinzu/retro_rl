"""Fail-closed natural Level 9 and write-free ending controllers.

The natural prefix controllers intentionally refuse to move until decoded ROM
topology and measured predecessor inventory are supplied.  The ending adapter
accepts controller input only; it never loads a fixture or writes inventory,
doors, rooms, progression, or capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.pause_select import PauseSelectController
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
    ROOM_LEVEL9_ENTRY,
    ROOM_OLD_MAN_TF,
    SILVER_ARROWS,
    TRIFORCE_NOT_FULL,
    UNMEASURED_POST_L8_HANDOFF,
    level9_credits_stop,
    level9_live_patra_stop,
)
from zelda_i.level9.overworld import (
    Level9PostL8OverworldController,
    Level9SpectacleRockBombController,
)
from zelda_i.level9.prefix import (
    Level9North76Controller, make_bomb_north_20_controller, make_bomb_north_65_controller,
    make_bomb_west_06_controller, make_cellar_60_controller, make_cellar_70_controller,
    make_cellar_75_controller, make_east_14_controller, make_east_15_controller,
    make_north_16_controller, make_north_76_controller, make_stairs_05_controller,
    make_stairs_55_controller, make_stairs_61_controller, make_west_62_controller,
    make_west_63_controller, make_west_66_controller,
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
from zelda_i.level9.path import final_patra_to_ganon_step
from zelda_i.level9.patra import final_patra_north_door_earned, patra_action
from zelda_i.level9.room51 import room51_to_41_step
from zelda_i.level9.stairs import (
    BOMB_WALL_04_WEST,
    BOMB_WALL_31_WEST,
    CELLAR_MODE,
    ROOM03,
    ROOM04,
    ROOM30,
    ROOM31,
    ROOM41,
    ROOM51,
    ROOM61,
    chase_sword_step,
    live_combat_objects,
    pushable_block,
    room03_stairs_step,
    room30_stairs_step,
    stair_transition_modes,
)
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
) -> Level9PostL8OverworldController:
    return Level9PostL8OverworldController(handoff=handoff)


def make_spectacle_rock_bomb_controller(
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> Level9SpectacleRockBombController:
    return Level9SpectacleRockBombController(handoff=handoff)


def make_old_man_tf_gate_controller(
    *, dest: int | None = ROOM_OLD_MAN_TF
) -> Level9North76Controller:
    return make_north_76_controller(dest=dest)


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
            "triforce_writes": 0,
            "selected_item_writes": 0,
        }


@dataclass
class NaturalSilverArrowsController(_NaturalEndingController):
    """Sequential controller connecting 0x76 through all 16 prefix hops to Silver Arrows 0x10.

    Traverses 16 natural hops without memory writes or state loads:
    0x76 -> 0x66 -> 0x65 -> 0x55 -> cellar 0x60 -> 0x14 -> 0x15 -> 0x16 ->
    0x06 -> 0x05 -> cellar 0x70 -> 0x63 -> 0x62 -> 0x61 -> cellar 0x75 ->
    0x20 -> 0x10.
    """

    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF
    max_frames: int = 1
    hop_i: int = 0
    start_checked: bool = False
    blocked_reason: str = ""
    _hops: tuple[Any, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        if self.handoff.complete():
            self.max_frames = 16000
        else:
            self.max_frames = 1
        if not self._hops:
            self._hops = (
                make_north_76_controller(), make_west_66_controller(),
                make_bomb_north_65_controller(), make_stairs_55_controller(),
                make_cellar_60_controller(), make_east_14_controller(),
                make_east_15_controller(), make_north_16_controller(),
                make_bomb_west_06_controller(), make_stairs_05_controller(),
                make_cellar_70_controller(), make_west_63_controller(),
                make_west_62_controller(), make_stairs_61_controller(),
                make_cellar_75_controller(), make_bomb_north_20_controller(),
            )

    def _fail(self, reason: str) -> FrameAction:
        self.blocked_reason = reason
        return super()._fail(reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if snap.triforce != FULL_TRIFORCE:
                return self._fail(TRIFORCE_NOT_FULL)
            if not self.handoff.complete():
                return self._fail(MISSING_SILVER_ARROW_ROOM)
            if not (
                snap.level == LEVEL9
                and snap.screen == ROOM_LEVEL9_ENTRY
                and snap.mode == PLAY_MODE
            ):
                return self._fail("natural_silver_arrows_predecessor_contract_miss")

        while self.hop_i < len(self._hops) and self._hops[self.hop_i].success:
            self.hop_i += 1

        if self.hop_i >= len(self._hops):
            self.success = True
            return self._action(nes_idle_action(), "silver_arrows_arrived")

        ctl = self._hops[self.hop_i]
        act = ctl.step(snap)
        if ctl.failed:
            return self._fail(ctl.notes[-1] if ctl.notes else f"hop_{self.hop_i}_failed")
        if ctl.success:
            self.hop_i += 1
            if self.hop_i >= len(self._hops):
                self.success = True
                return self._action(nes_idle_action(), "silver_arrows_arrived")
        return self._action(act.action, f"prefix_hop_{self.hop_i}_{act.reason}")

    def report(self) -> dict[str, object]:
        rep = super().report()
        rep.update({
            "chapter": "level9_natural_silver_arrows",
            "evidence": self.handoff.evidence,
            "route_eligible": self.handoff.route_eligible and self.success,
            "hop_i": self.hop_i,
            "total_hops": len(self._hops),
            "current_hop": getattr(self._hops[self.hop_i], "spec_id", f"hop_{self.hop_i}")
            if self.hop_i < len(self._hops)
            else "done",
            "missing_evidence": self.blocked_reason or None,
            "writes": 0,
        })
        return rep


def make_natural_silver_arrows_controller(
    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF,
) -> NaturalSilverArrowsController:
    return NaturalSilverArrowsController(handoff=handoff)


class PatraJoinPhase(Enum):
    SOUTH_10 = auto()
    CLEAR_20 = auto()
    NAV_BLOCK_20 = auto()
    PUSH_BLOCK_20 = auto()
    STAIRS_20 = auto()
    CELLAR_75 = auto()
    NAV_61 = auto()
    NAV_51 = auto()
    CLEAR_41 = auto()
    NORTH_41 = auto()
    CLEAR_31 = auto()
    NAV_BOMB_31 = auto()
    BOMB_31 = auto()
    CLEAR_30 = auto()
    STAIRS_30 = auto()
    CELLAR_67 = auto()
    CLEAR_04 = auto()
    NAV_BOMB_04 = auto()
    BOMB_04 = auto()
    CLEAR_03 = auto()
    STAIRS_03 = auto()
    CELLAR_77 = auto()
    WAIT_PATRA = auto()
    ARRIVED = auto()
    FAILED = auto()


NAV_BLOCK_20_WPS = ((176, 93), (176, 189), (96, 189), (96, 157))
NAV_61_WPS = ((48, 157), (48, 93), (120, 93), (120, 77))
NAV_BOMB_31_WPS = ((120, 189), (48, 189), (48, 141))


def cellar_west_to_east_step(snap: ZeldaSnapshot) -> FrameAction:
    x, y = int(snap.link_x), int(snap.link_y)
    if snap.mode != 9 or snap.transitioning:
        return FrameAction(nes_action("UP"), "cellar_exit_scroll")
    if y < 189 and x <= 64:
        return FrameAction(nes_action("DOWN"), "cellar_west_drop")
    if x < 192:
        return FrameAction(nes_action("RIGHT"), "cellar_floor_east")
    return FrameAction(nes_action("UP"), "cellar_east_climb")


def cellar_east_to_west_step(snap: ZeldaSnapshot) -> FrameAction:
    x, y = int(snap.link_x), int(snap.link_y)
    if snap.mode != 9 or snap.transitioning:
        return FrameAction(nes_action("UP"), "cellar_exit_scroll")
    if y < 189 and x >= 176:
        return FrameAction(nes_action("DOWN"), "cellar_east_drop")
    if x > 48:
        return FrameAction(nes_action("LEFT"), "cellar_floor_west")
    return FrameAction(nes_action("UP"), "cellar_west_climb")


@dataclass
class NaturalPatraJoinController(_NaturalEndingController):
    """One-frame sequential controller connecting Silver Arrows 0x10 to live Patra 0x52.

    Traverses 10 natural hops without memory writes or state loads:
    0x10 -> 0x20 -> cellar 0x75 -> 0x61 -> 0x51 -> 0x41 -> 0x31 -> 0x30 ->
    cellar 0x67 -> 0x04 -> 0x03 -> cellar 0x77 -> live Patra 0x52.
    """

    max_frames: int = 24000
    phase: PatraJoinPhase = PatraJoinPhase.SOUTH_10
    phase_frames: int = 0
    cooldown: int = 0
    wp_i: int = 0
    start_checked: bool = False
    _bomb_31: BombWallController = field(init=False, repr=False)
    _bomb_04: BombWallController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bomb_31 = BombWallController(
            wall=BOMB_WALL_31_WEST,
            level=LEVEL9,
            approach_tol=4,
            stand_tol=4,
            face_frames=4,
            step_back=6,
            wait_blast=100,
            wait_hold_face=False,
            require_bomb_consumed=True,
            max_frames=4000,
        )
        self._bomb_04 = BombWallController(
            wall=BOMB_WALL_04_WEST,
            level=LEVEL9,
            approach_tol=4,
            stand_tol=4,
            face_frames=4,
            step_back=6,
            wait_blast=100,
            wait_hold_face=False,
            require_bomb_consumed=True,
            max_frames=4000,
        )

    def _set_phase(self, phase: PatraJoinPhase) -> None:
        self.phase = phase
        self.phase_frames = 0
        self.wp_i = 0

    def _action(self, action: list[int], reason: str) -> FrameAction:
        self.phase_frames += 1
        return super()._action(action, reason)

    def _fail(self, reason: str) -> FrameAction:
        self.phase = PatraJoinPhase.FAILED
        return super()._fail(reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.success or self.failed:
            return self._action(nes_idle_action(), "done")
        if snap.mode == 17:
            return self._fail("link_death")
        if not self.start_checked:
            self.start_checked = True
            if not (
                snap.level == LEVEL9
                and snap.screen in (0x10, 0x20)
                and snap.mode == PLAY_MODE
                and snap.triforce == FULL_TRIFORCE
                and snap.bombs >= 1
            ):
                return self._fail("natural_patra_join_predecessor_contract_miss")

        # 1. SOUTH_10
        if self.phase == PatraJoinPhase.SOUTH_10:
            if snap.screen == 0x20 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_20)
            elif snap.screen == 0x10:
                if abs(snap.link_x - 120) > 2:
                    d = "LEFT" if snap.link_x > 120 else "RIGHT"
                    return self._action(nes_action(d), "south_10_align_x")
                return self._action(nes_action("DOWN"), "south_10_push_down")
            else:
                return self._action(nes_action("DOWN"), "south_10_scroll")

        # 2. CLEAR_20
        if self.phase == PatraJoinPhase.CLEAR_20:
            combat = live_combat_objects(snap)
            if len(combat) == 0 or self.phase_frames >= 2500:
                self._set_phase(PatraJoinPhase.NAV_BLOCK_20)
            else:
                act, self.cooldown = chase_sword_step(snap, self.cooldown)
                return self._action(act.action, "clear_20_combat")

        # 3. NAV_BLOCK_20
        if self.phase == PatraJoinPhase.NAV_BLOCK_20:
            if self.wp_i >= len(NAV_BLOCK_20_WPS):
                self._set_phase(PatraJoinPhase.PUSH_BLOCK_20)
            else:
                tx, ty = NAV_BLOCK_20_WPS[self.wp_i]
                dx, dy = tx - snap.link_x, ty - snap.link_y
                if abs(dx) <= 2 and abs(dy) <= 2:
                    self.wp_i += 1
                    if self.wp_i >= len(NAV_BLOCK_20_WPS):
                        self._set_phase(PatraJoinPhase.PUSH_BLOCK_20)
                        return self._action(nes_action("UP"), "nav_block_20_arrived")
                    tx, ty = NAV_BLOCK_20_WPS[self.wp_i]
                    dx, dy = tx - snap.link_x, ty - snap.link_y
                d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > 2 else ("DOWN" if dy > 0 else "UP")
                return self._action(nes_action(d), f"nav_block_20_wp{self.wp_i}")

        # 4. PUSH_BLOCK_20
        if self.phase == PatraJoinPhase.PUSH_BLOCK_20:
            block = pushable_block(snap)
            if self.phase_frames >= 50 or (block is not None and block.y <= 130):
                self._set_phase(PatraJoinPhase.STAIRS_20)
                return self._action(nes_idle_action(), "push_block_20_done")
            return self._action(nes_action("UP"), "push_block_20")

        # 5. STAIRS_20
        if self.phase == PatraJoinPhase.STAIRS_20:
            if snap.mode in (CELLAR_MODE, 10, 16) or stair_transition_modes(snap.mode):
                self._set_phase(PatraJoinPhase.CELLAR_75)
                return self._action(nes_idle_action(), "stairs_20_transition")
            dx, dy = 128 - snap.link_x, 141 - snap.link_y
            if abs(dx) > 1:
                d = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 1:
                d = "DOWN" if dy > 0 else "UP"
            else:
                d = "DOWN"
            return self._action(nes_action(d), "stairs_20_step")

        # 6. CELLAR_75
        if self.phase == PatraJoinPhase.CELLAR_75:
            if snap.screen == ROOM61 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.NAV_61)
                return self._action(nes_idle_action(), "cellar_75_arrived_61")
            act = cellar_west_to_east_step(snap)
            return self._action(act.action, act.reason)

        # 7. NAV_61
        if self.phase == PatraJoinPhase.NAV_61:
            if snap.screen == ROOM51 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.NAV_51)
                return self._action(nes_idle_action(), "nav_61_arrived_51")
            if snap.transitioning or snap.mode != PLAY_MODE:
                return self._action(nes_action("UP"), "nav_61_scroll")
            if self.wp_i < len(NAV_61_WPS):
                tx, ty = NAV_61_WPS[self.wp_i]
                dx, dy = tx - snap.link_x, ty - snap.link_y
                if abs(dx) <= 2 and abs(dy) <= 2:
                    self.wp_i += 1
                if self.wp_i < len(NAV_61_WPS):
                    tx, ty = NAV_61_WPS[self.wp_i]
                    dx, dy = tx - snap.link_x, ty - snap.link_y
                    d = ("RIGHT" if dx > 0 else "LEFT") if abs(dx) > 2 else ("DOWN" if dy > 0 else "UP")
                    return self._action(nes_action(d), f"nav_61_wp{self.wp_i}")
            return self._action(nes_action("UP"), "nav_61_push_up")

        # 8. NAV_51
        if self.phase == PatraJoinPhase.NAV_51:
            if snap.screen == ROOM41 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_41)
                return self._action(nes_idle_action(), "nav_51_arrived_41")
            act = room51_to_41_step(snap, self.phase_frames)
            return self._action(act.action, act.reason)

        # 9. CLEAR_41
        if self.phase == PatraJoinPhase.CLEAR_41:
            combat = live_combat_objects(snap)
            if len(combat) == 0 or self.phase_frames >= 1800:
                self._set_phase(PatraJoinPhase.NORTH_41)
                return self._action(nes_idle_action(), "clear_41_done")
            act, self.cooldown = chase_sword_step(snap, self.cooldown)
            return self._action(act.action, "clear_41_combat")

        # 10. NORTH_41
        if self.phase == PatraJoinPhase.NORTH_41:
            if snap.screen == ROOM31 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_31)
                return self._action(nes_idle_action(), "north_41_arrived_31")
            if snap.transitioning or snap.mode != PLAY_MODE:
                return self._action(nes_action("UP"), "north_41_scroll")
            if abs(snap.link_x - 120) > 2:
                d = "RIGHT" if snap.link_x < 120 else "LEFT"
                return self._action(nes_action(d), "north_41_align_x")
            return self._action(nes_action("UP"), "north_41_push_up")

        # 11. CLEAR_31
        if self.phase == PatraJoinPhase.CLEAR_31:
            combat = live_combat_objects(snap)
            if len(combat) == 0 or self.phase_frames >= 2000:
                self._set_phase(PatraJoinPhase.NAV_BOMB_31)
                return self._action(nes_idle_action(), "clear_31_done")
            act, self.cooldown = chase_sword_step(snap, self.cooldown)
            return self._action(act.action, "clear_31_combat")

        # 12. NAV_BOMB_31
        if self.phase == PatraJoinPhase.NAV_BOMB_31:
            if self.wp_i >= len(NAV_BOMB_31_WPS):
                self._set_phase(PatraJoinPhase.BOMB_31)
                return self._action(nes_idle_action(), "nav_bomb_31_stand")
            tx, ty = NAV_BOMB_31_WPS[self.wp_i]
            dx, dy = tx - snap.link_x, ty - snap.link_y
            if abs(dx) <= 2 and abs(dy) <= 2:
                self.wp_i += 1
                if self.wp_i >= len(NAV_BOMB_31_WPS):
                    self._set_phase(PatraJoinPhase.BOMB_31)
                    return self._action(nes_idle_action(), "nav_bomb_31_stand")
                tx, ty = NAV_BOMB_31_WPS[self.wp_i]
                dx, dy = tx - snap.link_x, ty - snap.link_y
            d = ("DOWN" if dy > 0 else "UP") if abs(dy) > 2 else ("RIGHT" if dx > 0 else "LEFT")
            return self._action(nes_action(d), f"nav_bomb_31_wp{self.wp_i}")

        # 13. BOMB_31
        if self.phase == PatraJoinPhase.BOMB_31:
            if snap.screen == ROOM30 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_30)
                return self._action(nes_idle_action(), "bomb_31_arrived_30")
            if snap.transitioning or snap.mode != PLAY_MODE:
                return self._action(nes_action("LEFT"), "bomb_31_scroll")
            act = self._bomb_31.step(snap)
            if self._bomb_31.success or self._bomb_31.phase in (BombWallPhase.DONE, BombWallPhase.PUSH):
                return self._action(nes_action("LEFT"), "bomb_31_push_left")
            return self._action(act.action, act.reason)

        # 14. CLEAR_30
        if self.phase == PatraJoinPhase.CLEAR_30:
            combat = live_combat_objects(snap)
            if len(combat) == 0 or self.phase_frames >= 2000:
                self._set_phase(PatraJoinPhase.STAIRS_30)
                return self._action(nes_idle_action(), "clear_30_done")
            act, self.cooldown = chase_sword_step(snap, self.cooldown)
            return self._action(act.action, "clear_30_combat")

        # 15. STAIRS_30
        if self.phase == PatraJoinPhase.STAIRS_30:
            if snap.mode in (CELLAR_MODE, 10, 16) or stair_transition_modes(snap.mode):
                self._set_phase(PatraJoinPhase.CELLAR_67)
                return self._action(nes_idle_action(), "stairs_30_transition")
            act = room30_stairs_step(snap)
            return self._action(act.action, act.reason)

        # 16. CELLAR_67
        if self.phase == PatraJoinPhase.CELLAR_67:
            if snap.screen == ROOM04 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_04)
                return self._action(nes_idle_action(), "cellar_67_arrived_04")
            act = cellar_west_to_east_step(snap)
            return self._action(act.action, act.reason)

        # 17. CLEAR_04
        if self.phase == PatraJoinPhase.CLEAR_04:
            combat = live_combat_objects(snap)
            if len(combat) == 0 or self.phase_frames >= 1200:
                self._set_phase(PatraJoinPhase.NAV_BOMB_04)
                return self._action(nes_idle_action(), "clear_04_done")
            act, self.cooldown = chase_sword_step(snap, self.cooldown)
            return self._action(act.action, "clear_04_combat")

        # 18. NAV_BOMB_04
        if self.phase == PatraJoinPhase.NAV_BOMB_04:
            x, y = snap.link_x, snap.link_y
            if y > 95 and x > 52:
                return self._action(nes_action("UP"), "nav_bomb_04_to_north_aisle")
            if x > 48:
                return self._action(nes_action("LEFT"), "nav_bomb_04_west_aisle")
            if y < 141:
                return self._action(nes_action("DOWN"), "nav_bomb_04_south_to_stand")
            self._set_phase(PatraJoinPhase.BOMB_04)
            return self._action(nes_idle_action(), "nav_bomb_04_stand")

        # 19. BOMB_04
        if self.phase == PatraJoinPhase.BOMB_04:
            if snap.screen == ROOM03 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.CLEAR_03)
                return self._action(nes_idle_action(), "bomb_04_arrived_03")
            if snap.transitioning or snap.mode != PLAY_MODE:
                return self._action(nes_action("LEFT"), "bomb_04_scroll")
            act = self._bomb_04.step(snap)
            if self._bomb_04.success or self._bomb_04.phase in (BombWallPhase.DONE, BombWallPhase.PUSH):
                return self._action(nes_action("LEFT"), "bomb_04_push_left")
            return self._action(act.action, act.reason)

        # 20. CLEAR_03
        if self.phase == PatraJoinPhase.CLEAR_03:
            combat = tuple(o for o in live_combat_objects(snap) if o.type_id != 0x2B)
            if len(combat) == 0:
                self._set_phase(PatraJoinPhase.STAIRS_03)
                return self._action(nes_idle_action(), "clear_03_done")
            act, self.cooldown = chase_sword_step(snap, self.cooldown, types=(0x13, 0x14, 0x17))
            return self._action(act.action, "clear_03_combat")

        # 21. STAIRS_03
        if self.phase == PatraJoinPhase.STAIRS_03:
            if snap.mode in (CELLAR_MODE, 10, 16) or stair_transition_modes(snap.mode):
                self._set_phase(PatraJoinPhase.CELLAR_77)
                return self._action(nes_idle_action(), "stairs_03_transition")
            likes = tuple(obj for obj in live_combat_objects(snap) if obj.type_id == 0x17)
            grabbed = any(
                abs(int(obj.x) - snap.link_x) <= 8 and abs(int(obj.y) - snap.link_y) <= 8
                for obj in likes
            )
            if grabbed:
                act, self.cooldown = chase_sword_step(snap, self.cooldown, types=(0x17,))
                return self._action(act.action, "room03_fight_like_like")
            act = room03_stairs_step(snap)
            return self._action(act.action, act.reason)

        # 22. CELLAR_77
        if self.phase == PatraJoinPhase.CELLAR_77:
            if snap.screen == 0x52 and snap.mode == PLAY_MODE and not snap.transitioning:
                self._set_phase(PatraJoinPhase.WAIT_PATRA)
                return self._action(nes_idle_action(), "cellar_77_arrived_patra")
            act = cellar_east_to_west_step(snap)
            return self._action(act.action, act.reason)

        # 23. WAIT_PATRA
        if self.phase == PatraJoinPhase.WAIT_PATRA:
            if level9_live_patra_stop(snap):
                self._set_phase(PatraJoinPhase.ARRIVED)
                self.success = True
                return self._action(nes_idle_action(), "live_patra_contract_met")
            if self.phase_frames >= 120:
                return self._fail("live_patra_contract_miss")
            return self._action(nes_idle_action(), "wait_patra_eyes_spawn")

        # 24. ARRIVED
        if self.phase == PatraJoinPhase.ARRIVED:
            self.success = True
            return self._action(nes_idle_action(), "done")

        return self._fail(f"unknown_phase_{self.phase}")

    def report(self) -> dict[str, object]:
        rep = super().report()
        rep["phase"] = self.phase.name
        return rep


def make_natural_patra_join_controller() -> NaturalPatraJoinController:
    return NaturalPatraJoinController()


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
    "Level9North76Controller", "Level9PostL8OverworldController",
    "Level9SpectacleRockBombController", "NaturalCreditsController",
    "NaturalEnterZeldaController", "NaturalFinalPatraController",
    "NaturalGanonController", "NaturalPatraJoinController",
    "NaturalPatraToGanonController", "NaturalPowerTriforceController",
    "NaturalRescueZeldaController", "NaturalRouteUnavailableController",
    "NaturalSelectSilverArrowsController", "NaturalSilverArrowsController",
    "PatraJoinPhase", "make_natural_patra_join_controller",
    "make_natural_silver_arrows_controller", "make_old_man_tf_gate_controller",
    "make_patra_join_unavailable_controller", "make_post_l8_overworld_controller",
    "make_silver_arrows_unavailable_controller", "make_spectacle_rock_bomb_controller",
]
