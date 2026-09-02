"""Fail-closed post-L6 overworld handoff and natural Bait plan.

The measured L6 fanfare leave is carried as the shared
``zelda_i.overworld.stitch.OverworldHandoff`` packet (``verified=False`` until
the L7 owner re-measures it with ``selected_item`` captured).  The spine
controller walks the fixture-live ``0x22 → 0x25`` bait prefix but only after the
handoff verifies; every hypothesis past ``0x25`` fails closed.  The natural Bait
buy refuses until shop geometry is live and Link already holds 60 rupees (no
``ADDR_FOOD`` / rupee write, ever).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.anchors import (
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
)
from zelda_i.level7.overworld import (
    POST_L6_TO_BAIT_HOPS,
    at_l6_cave_mouth,
    bait_24_east_action,
    bait_32_north_action,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.path import OverworldPathController
from zelda_i.overworld.stitch import (
    CUMULATIVE_TF,
    UNMEASURED_HANDOFF,
    OverworldHandoff,
)
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_RUPEES,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

POST_L6_TRIFORCE = CUMULATIVE_TF[6]  # 0x3F
BAIT_COST = 60
BAIT_SHOP_SCREEN_HYP = SCREEN_LEVEL7_BAIT_SHOP_HYP  # 0x34
APPROACH_MAX_FRAMES = 40_000
BAIT_MAX_FRAMES = 1
# Save-state name for the bait-walk recon script (scratch/run_bait_from_l6_exit).
POST_L6_EXIT_STATE = "Level6ExitOverworld"

# Measured post-L6 fanfare engine return: ``--through level6-exit`` 1/1
# (recordings/l6_exit_ow.json).  The shard fanfare auto-warps Link to OW 0x22 at
# the Dragon mouth tile (112,125), mode 5, not transitioning.  Screen 0x22
# confirms the bait/pond route; the position corrects the old poke fixture's
# (120,221).  ``verified`` stays False: the L7 owner must re-measure with
# ``selected_item`` captured and reconcile the 42R -> 60R Bait gap (a documented
# Survival rupee top-up, mirroring the bomb/key top-ups).
MEASURED_POST_L6_EXIT = OverworldHandoff(
    screen=SCREEN_LEVEL6_ENTRANCE,
    link_x=112,
    link_y=125,
    mode=PLAY_MODE,
    triforce=POST_L6_TRIFORCE,
    keys=2,
    bombs=8,
    rupees=42,
    heart_containers=8,
    selected_item=None,  # not captured by spine_final_fields this run
    whistle=1,
    food=0,
    rod=1,
    bow=1,
    arrows=1,
    candle=0,
    evidence="measured-level6-exit-1of1",
    verified=False,
    route_eligible=False,
)


class ApproachPhase(Enum):
    HOP = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class PostLevel6OverworldController(OverworldPathController):
    """Measured L6 leave -> fixture-live bait prefix.  Fails closed at 0x25.

    Refuses every frame until the shared ``OverworldHandoff`` verifies.  Once it
    does, walks ``POST_L6_TO_BAIT_HOPS`` (``0x22 -> 0x32 -> 0x33 -> 0x23 ->
    0x24 -> 0x25``, fixture-live 1/1) and then fails closed — there is no
    observed route past the ``0x25`` west mouth, and the bait shop ``0x34`` and
    pond ``0x42`` are still source hypotheses.
    """

    handoff: OverworldHandoff = UNMEASURED_HANDOFF
    hops: tuple[ScreenHop, ...] = POST_L6_TO_BAIT_HOPS
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
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        if hop.target == 0x33:
            act = bait_32_north_action(snap, swing=self._swing)
            if act is not None:
                return act
        if hop.target == 0x25:
            act = bait_24_east_action(snap, swing=self._swing)
            if act is not None:
                return act
        if self.stuck > self.stuck_threshold:
            return FrameAction(nes_idle_action(), "post_l6_path_stuck_wait")
        return None

    def _reentry_refusal(self, snap: ZeldaSnapshot) -> str | None:
        """Never walk back into the L6 dungeon mouth."""
        if snap.level == 6:
            return "l6_dungeon_enter"
        if at_l6_cave_mouth(snap):
            return "l6_cave_mouth"
        if snap.mode == 16 and snap.screen == SCREEN_LEVEL6_ENTRANCE:
            return "l6_cave_mouth_enter"
        if snap.in_cave:
            return "unexpected_cave"
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
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
        reentry = self._reentry_refusal(snap)
        if reentry is not None:
            return self._fail_now(reentry)
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
    handoff: OverworldHandoff = UNMEASURED_HANDOFF,
    hops: tuple[ScreenHop, ...] = POST_L6_TO_BAIT_HOPS,
) -> PostLevel6OverworldController:
    return PostLevel6OverworldController(handoff=handoff, hops=hops)


def make_bait_purchase_controller(
    *, plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
) -> NaturalBaitPurchaseController:
    return NaturalBaitPurchaseController(plan=plan)


__all__ = [
    "APPROACH_MAX_FRAMES",
    "BAIT_COST",
    "BAIT_MAX_FRAMES",
    "BAIT_SHOP_SCREEN_HYP",
    "MEASURED_POST_L6_EXIT",
    "POST_L6_EXIT_STATE",
    "POST_L6_TRIFORCE",
    "UNMEASURED_HANDOFF",
    "UNVERIFIED_BAIT_PLAN",
    "ApproachPhase",
    "BaitPurchasePlan",
    "NaturalBaitPurchaseController",
    "OverworldHandoff",
    "PostLevel6OverworldController",
    "make_bait_purchase_controller",
    "make_post_l6_overworld_controller",
]
