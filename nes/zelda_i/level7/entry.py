"""Fail-closed post-L6 overworld handoff and natural Bait plan.

The measured L6 fanfare leave is carried as the shared
``zelda_i.overworld.stitch.OverworldHandoff`` packet.  The spine controller
walks ``POST_L6_TO_POND_HOPS`` (greened prefix from ``0x22`` toward pond
``0x42``) after the handoff verifies.  ``POST_L6_TO_BAIT_HOPS`` is a dead
mountain-pocket spur — kept for bait-micro tests, not the default.  The
natural Bait buy refuses until shop geometry is live and Link already holds
60 rupees (no ``ADDR_FOOD`` / rupee write, ever).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_idle_action
from zelda_i.anchors import (
    SCREEN_LEVEL6_ENTRANCE,
    SCREEN_LEVEL7_BAIT_SHOP_HYP,
)
from zelda_i.dungeon.ops import poke_food
from zelda_i.level7.pond import (
    POST_L6_TO_POND_HOPS,
    ApproachPhase,
    PostLevel6OverworldController,
    make_post_l6_overworld_controller,
)
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

# Measured post-L6 fanfare engine return: ``--through level6-exit`` 2/2
# (recordings/l6_exit_ow.json 2026-09-02, recordings/l7p1_l6exit.json 2026-09-02).
# The shard fanfare auto-warps Link to OW 0x22 at the Dragon mouth tile
# (112,125), mode 5, not transitioning.  Both runs agree byte-for-byte on every
# measured field; the second run additionally captured the B-slot / Whistle /
# Food / Candle bytes (``spine_final_fields`` now takes ``ram``).
#
# ``verified=True``: every field below is the live settled RAM at the frame
# ``level7_post_l6_overworld`` begins.  ``route_eligible`` stays False until
# the greened ``POST_L6_TO_POND_HOPS`` prefix reaches pond ``0x42``.  The
# 60R Bait is paid from the wallet the gathering's hidden rupees leave
# after the Blue Ring; there is no rupee write.
# Re-measured 2026-09-22 on the gathered spine (White Sword, blue candle
# from the gathering, three overworld hearts). Keys/bombs/rupees are lower
# bounds; the wooden-sword arrival (8 containers, candle 0, bombs 8, 42R)
# is retired with the 18909f oracle.
MEASURED_POST_L6_EXIT = OverworldHandoff(
    screen=SCREEN_LEVEL6_ENTRANCE,
    link_x=112,
    link_y=125,
    mode=PLAY_MODE,
    triforce=POST_L6_TRIFORCE,
    keys=2,
    bombs=7,
    # No wallet floor: potion restocks spend it (last-heart run 30 left L6
    # with 29R) and the Bait stage checks its own price.
    rupees=0,
    heart_containers=11,
    selected_item=2,  # arrows still selected from the Gohma kill
    whistle=1,
    food=0,
    rod=1,
    bow=1,
    arrows=1,
    candle=1,
    evidence="measured-gathered-spine-2026-09-22",
    verified=True,
    route_eligible=False,
)


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


SURVIVAL_BAIT_FOOD = 1


@dataclass
class SurvivalBaitPurchaseController:
    """Survival-only Bait stand-in: disclose one ``ADDR_FOOD`` write, then pass.

    The natural L6 -> bait-shop overworld route is a mountain-locked pocket and
    still unmapped (bead ``rr-8t4.4``).  The Clean path keeps
    ``NaturalBaitPurchaseController`` fail-closed.  This Survival controller sets
    the owned Food byte (``$065D``) so ``level7-entry`` / L7-B can run, mirroring
    the disclosed rupee-count top-up (``SPINE_L7_RUPEE_RETOPUP``).  It never
    writes rupees, Whistle, a door, or a Triforce bit, and stays
    ``route_eligible=False``.
    """

    plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
    max_frames: int = BAIT_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    inventory_assist: dict[str, Any] | None = None
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        if not self.notes:
            self.notes.append(reason)
        return FrameAction(nes_idle_action(), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        self.frames += 1
        if self._env is None:
            return self._fail("survival_bait_env_not_bound")
        food = int(read_u8(self._env.get_ram(), ADDR_FOOD))
        if food < SURVIVAL_BAIT_FOOD:
            self.inventory_assist = poke_food(self._env, from_food=food)
            if int(self.inventory_assist.get("progression_writes") or 0):
                return self._fail("survival_bait_progression_write")
            if int(self.inventory_assist.get("food_writes") or 0) != 1:
                return self._fail("survival_bait_food_write_failed")
            food = int(read_u8(self._env.get_ram(), ADDR_FOOD))
        if food < SURVIVAL_BAIT_FOOD:
            return self._fail("survival_bait_food_not_set")
        self.success = True
        self.notes.append(f"survival_bait_food_set={food}")
        return FrameAction(nes_idle_action(), "survival_bait_food_fixture")

    def report(self) -> dict[str, Any]:
        writes = int((self.inventory_assist or {}).get("food_writes") or 0)
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_bait_purchase",
            "track": "survival_fixture",
            "shop_screen": hex(self.plan.shop_screen),
            "cost": self.plan.cost,
            "evidence": "survival-fixture",
            "route_eligible": False,
            "writes": writes,
            "inventory_assist": self.inventory_assist,
            "progression_writes": 0,
            "capacity_writes": 0,
            "notes": list(self.notes),
        }


def make_bait_purchase_controller(
    *, plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
) -> NaturalBaitPurchaseController:
    return NaturalBaitPurchaseController(plan=plan)


def make_survival_bait_purchase_controller(
    *, plan: BaitPurchasePlan = UNVERIFIED_BAIT_PLAN
) -> SurvivalBaitPurchaseController:
    return SurvivalBaitPurchaseController(plan=plan)


__all__ = [
    "APPROACH_MAX_FRAMES",
    "BAIT_COST",
    "BAIT_MAX_FRAMES",
    "BAIT_SHOP_SCREEN_HYP",
    "MEASURED_POST_L6_EXIT",
    "POST_L6_EXIT_STATE",
    "POST_L6_TO_POND_HOPS",
    "POST_L6_TRIFORCE",
    "SURVIVAL_BAIT_FOOD",
    "UNMEASURED_HANDOFF",
    "UNVERIFIED_BAIT_PLAN",
    "ApproachPhase",
    "BaitPurchasePlan",
    "NaturalBaitPurchaseController",
    "OverworldHandoff",
    "PostLevel6OverworldController",
    "SurvivalBaitPurchaseController",
    "make_bait_purchase_controller",
    "make_post_l6_overworld_controller",
    "make_survival_bait_purchase_controller",
]
