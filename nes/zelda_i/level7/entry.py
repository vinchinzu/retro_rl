"""Fail-closed post-L6 overworld handoff and natural Bait plan.

The start-``0x77`` pond walk stays recon-only.  Spine chapters refuse to move
until a measured leftover is supplied.  Do not treat OW ``0x22`` or the live
L6 prefix ``0x09`` as the L7 start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.anchors import SCREEN_LEVEL7_BAIT_SHOP_HYP, TF_BIT_L6
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

POST_L6_TRIFORCE = 0x3F
CANDLE_BLUE = 1
BAIT_COST = 60
BAIT_SHOP_SCREEN_HYP = SCREEN_LEVEL7_BAIT_SHOP_HYP  # 0x34
APPROACH_MAX_FRAMES = 40_000
BAIT_MAX_FRAMES = 1


@dataclass(frozen=True)
class PostLevel6Handoff:
    """Measured L6 leave required before L7 chapters may move.

    Every nullable value is part of the eventual handoff packet.  ``verified``
    stays false until the L6 owner reports the settled post-fanfare leftover.
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
    candle: int = CANDLE_BLUE
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
            return "post_l6_handoff_unmeasured"
        if snap.level != 0 or snap.mode != PLAY_MODE or snap.transitioning:
            return "post_l6_not_settled_overworld"
        if snap.screen != self.screen:
            return "post_l6_screen_mismatch"
        if abs(snap.link_x - int(self.link_x)) > self.xy_tolerance:
            return "post_l6_x_mismatch"
        if abs(snap.link_y - int(self.link_y)) > self.xy_tolerance:
            return "post_l6_y_mismatch"
        if snap.triforce != POST_L6_TRIFORCE or not (snap.triforce & TF_BIT_L6):
            return "post_l6_triforce_mismatch"
        if not snap.health_is_full or snap.heart_containers != self.heart_containers:
            return "post_l6_health_mismatch"
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
                return f"post_l6_{label}_mismatch"
        if int(read_u8(ram, ADDR_WHISTLE)) < 1:
            return "post_l6_whistle_required"
        if int(read_u8(ram, ADDR_ROD)) < 1:
            return "post_l6_rod_required"
        if int(read_u8(ram, ADDR_BOW)) < 1:
            return "post_l6_bow_required"
        return None


UNMEASURED_POST_L6_HANDOFF = PostLevel6Handoff()

# Live L6 residual is play 0x09 (56,109) TF 0x1F Rod=0 — not an L7 start.
CURRENT_L6_PREFIX_IS_NOT_L7_START = True


class ApproachPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostLevel6OverworldController(OverworldPathController):
    """Measured L6 leave → bait/pond approach.  Empty hops fail closed."""

    handoff: PostLevel6Handoff = UNMEASURED_POST_L6_HANDOFF
    hops: tuple[ScreenHop, ...] = ()
    phase: ApproachPhase = ApproachPhase.HOP
    max_frames: int = APPROACH_MAX_FRAMES
    require_sword: bool = True
    _env: Any = field(default=None, init=False, repr=False)
    _handoff_checked: bool = field(default=False, init=False, repr=False)

    @property
    def failed(self) -> bool:
        return self.phase is ApproachPhase.FAILED

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail_now(self, reason: str) -> FrameAction:
        self._set_phase(ApproachPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return self._fail_now("post_l6_path_exhausted_unmeasured")

    def _extra_hop_action(
        self, _snap: ZeldaSnapshot, _hop: ScreenHop
    ) -> FrameAction | None:
        if self.stuck > self.stuck_threshold:
            return FrameAction(nes_idle_action(), "post_l6_path_stuck_wait")
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if not self._handoff_checked:
            if self._env is None:
                return self._fail_now("entry_controller_env_not_bound")
            mismatch = self.handoff.mismatch(snap, self._env.get_ram())
            if mismatch is not None:
                return self._fail_now(mismatch)
            if not self.hops:
                return self._fail_now("post_l6_path_unmeasured")
            self._handoff_checked = True
            self.notes.append("post_l6_handoff_accepted")
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "evidence": self.handoff.evidence,
                "route_eligible": self.handoff.route_eligible,
                "failed": self.failed,
                "writes": 0,
            }
        )
        return out


@dataclass(frozen=True)
class BaitPurchasePlan:
    """Deterministic 60R + natural Food.  No rupee or ADDR_FOOD write."""

    shop_screen: int = BAIT_SHOP_SCREEN_HYP
    cost: int = BAIT_COST
    shop_cave_xy: tuple[int, int] | None = None
    farm_screens: tuple[int, ...] = ()
    shop_geometry_verified: bool = False
    farm_verified: bool = False
    evidence: str = "hypothesis"
    route_eligible: bool = False

    def can_pay(self, rupees: int) -> bool:
        return int(rupees) >= self.cost


UNVERIFIED_BAIT_PLAN = BaitPurchasePlan()


@dataclass
class NaturalBaitPurchaseController:
    """Refuse until shop geometry is live and Link already holds 60R."""

    plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
    max_frames: int = BAIT_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self._env is None:
            return self._fail("bait_controller_env_not_bound")
        ram = self._env.get_ram()
        rupees = int(read_u8(ram, ADDR_RUPEES))
        food = int(read_u8(ram, ADDR_FOOD))
        if food >= 1:
            return self._fail("bait_already_owned_room_unobserved")
        if not self.plan.can_pay(rupees):
            return self._fail("bait_need_60_rupees")
        if not self.plan.shop_geometry_verified or self.plan.shop_cave_xy is None:
            return self._fail("bait_shop_geometry_unobserved")
        if snap.screen != self.plan.shop_screen:
            return self._fail("bait_not_on_shop_screen")
        return self._fail("bait_purchase_policy_unobserved")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_bait_purchase",
            "shop_screen": hex(self.plan.shop_screen),
            "cost": self.plan.cost,
            "shop_geometry_verified": self.plan.shop_geometry_verified,
            "farm_screens": [hex(s) for s in self.plan.farm_screens],
            "evidence": self.plan.evidence,
            "route_eligible": self.plan.route_eligible,
            "writes": 0,
            "notes": list(self.notes),
        }


def make_post_l6_overworld_controller(
    *,
    handoff: PostLevel6Handoff = UNMEASURED_POST_L6_HANDOFF,
    hops: tuple[ScreenHop, ...] = (),
) -> PostLevel6OverworldController:
    return PostLevel6OverworldController(handoff=handoff, hops=hops)


def make_bait_purchase_controller(
    *, plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
) -> NaturalBaitPurchaseController:
    return NaturalBaitPurchaseController(plan=plan)


__all__ = [
    "APPROACH_MAX_FRAMES",
    "BAIT_COST",
    "BAIT_SHOP_SCREEN_HYP",
    "CANDLE_BLUE",
    "CURRENT_L6_PREFIX_IS_NOT_L7_START",
    "POST_L6_TRIFORCE",
    "UNMEASURED_POST_L6_HANDOFF",
    "UNVERIFIED_BAIT_PLAN",
    "BaitPurchasePlan",
    "NaturalBaitPurchaseController",
    "PostLevel6Handoff",
    "PostLevel6OverworldController",
    "make_bait_purchase_controller",
    "make_post_l6_overworld_controller",
]
