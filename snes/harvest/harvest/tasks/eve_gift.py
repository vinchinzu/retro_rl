"""Hand the married wife the item in hand, inside the remodeled house.

Group ``$43`` script 4 (Eve's indoor script, ``$B5938F``) takes a gift at an A
press while she is home: items 1-5 and 16-23 (crops, the egg 0x14, milk) add
+4, item 6/25/73 +8, 2/7 -8 and 9-12 -5. It runs at hours 6-10 and 15-17 (all
day in rain or Winter); 11-14 she is out and at 18 the script only takes the
-2 and parks. Each arm jumps back to the gift wait, so one gift a press. The
clock stops indoors, so an entry before 11:00 keeps the window open.

Measured on ``LN_y23_egg_home_probe`` (married, house 0x16, Y2 Winter D1):
from the door the x=98 column clears the table, Eve stands at (88,104), and
an A from (89,129) facing up showed text $03B2 and moved her 288 -> 292.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value
from harvest.planner.tasks.transitions import screen_faded_in
from harvest.tasks.nav import Point, get_pos_from_ram, make_action

HOUSE_L1_TILEMAP = 0x16
HOUSE_TILEMAPS = (HOUSE_L1_TILEMAP, 0x17)
WIFE_GIFTS_ENV = "HARVEST_WIFE_GIFTS"
EVE_WIFE_BIT = 0x0010
# Score is ``(hearts & 0x1FF) >> 4``: 31 at 496-511, and 512 wraps to 0. Child 1
# needs >= 450 and child 2 >=650 on a night with the max house (F4).
# Leave four extra hearts for the last day's failed/extra house approaches:
# the previous 502 policy ended at 494 in the 851-point continuation.
EVE_GIFT_TARGET = 506
EVE_FAMILY_TARGET = 654  # 650 plus the ordinary evening house-entry loss
# Two eggs a morning (+6 a day net) until child 1's 450: the level-2 house
# (no gift route yet) can arrive within ~95 days of the wedding.
EVE_GIFT_FAST_BELOW = 450
EVE_GIFT_LANE_X = 98
EVE_GIFT_STAND_DY = 24
# Opens at 6-10 and 15-17 outside rain and Winter (``$B59313``-``$B59329``).
EVE_GIFT_CLOSED_HOURS = frozenset({11, 12, 13, 14, 18})
EVE_SPAWN_WAIT_FRAMES = 180
EVE_GIFT_UNSTICK_FRAMES = 40    # still frames before the walk stops trusting the room walker
EVE_GIFT_X_SLACK = 6            # px off her column the gift press still lands (her sprite is 16 wide)


def wife_gift_target(ram: np.ndarray) -> int:
    from harvest.planner import run_strategy

    if run_strategy.active().court_girls and not int(read_ram_value(ram, "incubator_flags")) & 0x0008:
        return EVE_FAMILY_TARGET
    return EVE_GIFT_TARGET


def wife_settle_cap(ram: np.ndarray) -> int:
    """Highest current hearts that can settle to <=511 by the ending.

    Reserve the dated positive scenes before counting the remaining -2 nights.
    A late second child therefore requests negative gifts, rather than leaving
    650 hearts to wrap the final nine-bit affection score.
    """
    from harvest.planner.social_calendar import END_INDEX, date_index

    today = date_index(*(int(read_ram_value(ram, k)) for k in ("year", "season", "day")))
    remaining = max(0, END_INDEX - today)
    gains = 0
    for year in range(3):
        for season, day, gain in ((0, 23, 45), (3, 24, 30), (0, 1, 10), (2, 20, 10)):
            if today < date_index(year, season, day) < END_INDEX:
                gains += gain
    flags = int(read_ram_value(ram, "incubator_flags"))
    for bit, name in ((0x0004, "kid1_age"), (0x0008, "kid2_age")):
        age = int(read_ram_value(ram, name))
        if flags & bit and age < 60 and 60 - age <= remaining:
            gains += 50  # LaborMorningTask's positive answer, for either child
    return 511 + 2 * remaining - gains


HEART_DRAIN_ENV = "HARVEST_HEART_DRAIN"
# Every house entry at hour 18 takes exactly 2 hearts from a married wife and repeats (ROM probe on
# PR_y3d_Y3_Summer_D10: 554 -> 552 -> 550 -> 548 -> 546 -> 544, ~870 frames a cycle, the clock frozen at 18:xx).
# After the day's work that sheds the excess for zero game time, where a fish trip costs a whole morning.
HEART_DRAIN_PER_ENTRY = 2
HEART_DRAIN_MARGIN = 8  # end this far under the settle cap, as the last fish gift did
HEART_DRAIN_MAX_ENTRIES = 120


def heart_drain_enabled() -> bool:
    return os.environ.get(HEART_DRAIN_ENV, "1").strip().lower() not in {"0", "false", "no"}


def wife_heart_drain_entries(ram: np.ndarray) -> int:
    """Evening house entries that bring a wife above the settle cap down to ``cap - margin`` (0 when none)."""
    from harvest.planner import run_strategy
    from harvest.calendar_schedule import festival_on

    year, season, day = (int(read_ram_value(ram, k)) for k in ("year", "season", "day"))
    if not (run_strategy.active().court_girls
            and int(read_ram_value(ram, "incubator_flags")) & 0x0008 != 0  # child 2 flagged: hearts past 650 are spare
            and festival_on(year, season, day) is None):
        return 0
    excess = int(read_ram_value(ram, "eve_hearts")) - (wife_settle_cap(ram) - HEART_DRAIN_MARGIN)
    if excess <= 0:
        return 0
    return min(HEART_DRAIN_MAX_ENTRIES, -(-excess // HEART_DRAIN_PER_ENTRY))


def heart_drain_phases(entries: int, house_tilemap: int) -> list:
    """After the day's work: wait out the clock to 18:00, then enter and leave the house ``entries`` times.

    The last entry stays inside; the nightly sleep walk then finds the farmer already home.
    """
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE
    from harvest.planner.day_phase_chicken import WIFE_GIFT_PHASES
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec

    enter = [replace(p, params={**p.params, "target_tilemap": house_tilemap})
             if p.phase == "ENTER_HOUSE_FOR_GIFT" else p
             for p in WIFE_GIFT_PHASES if p.phase in ("NAV_HOUSE_FOR_GIFT", "ENTER_HOUSE_FOR_GIFT")]
    wait = PhaseSpec("HEART_DRAIN_WAIT", PhaseKind.WAIT_UNTIL_TIME,
                     {"target_hour": 18, "target_minute": 0, "timeout": 12_000}, failure_policy="optional")
    phases = [EXIT_TO_FARM_PHASE, wait]
    for index in range(entries):
        phases.extend(enter)
        if index != entries - 1:
            phases.append(EXIT_TO_FARM_PHASE)
    return phases


# Summer mornings reserved for the corn/tomato seed trip and the first sowing (see wife_fish_gift_wanted).
SUMMER_PLANTING_DAYS = 5


def wife_fish_gift_wanted(ram: np.ndarray) -> bool:
    from harvest.planner import run_strategy
    from harvest.calendar_schedule import festival_on
    from harvest.core.ram_catalog import read_ram_u16

    year, season, day = (int(read_ram_value(ram, k)) for k in ("year", "season", "day"))
    if season == 1 and day <= SUMMER_PLANTING_DAYS:
        # A fish trip is the whole morning. A corn ring planted by D3 gets six harvests, and the seed trip
        # is the first thing the morning plan drops: the continuation from the fixed Y2 winter shed Eve's
        # hearts on Summer D1-D3 and D5 and planted no corn or tomato at all (967 instead of ~985).
        return False
    if year >= 2 and season == 1 and day >= 27:
        if run_strategy.quota_harvest_due(ram):
            return False  # the final paid crop shipments have only these mornings left

    return (run_strategy.active().court_girls
            and int(read_ram_value(ram, "incubator_flags")) & 0x0008 != 0
            and int(read_ram_value(ram, "eve_hearts")) > wife_settle_cap(ram)
            and int(read_ram_value(ram, "hour")) < 11
            and not read_ram_u16(ram, 0x0196) & 0x001A
            and festival_on(year, season, day) is None)


def _gift_ceiling(ram: np.ndarray) -> int:
    target = wife_gift_target(ram)
    return target if target == EVE_FAMILY_TARGET else min(target, wife_settle_cap(ram))


def wife_gift_wanted(ram: np.ndarray) -> bool:
    """Eve is the wife, home in house 0x16, short of the window, and an egg is laid."""
    from harvest.core.animal_status import egg_available_today

    return (
        os.environ.get(WIFE_GIFTS_ENV, "1").strip().lower() not in {"0", "false", "no"}
        and int(read_ram_value(ram, "marriage_flags")) & EVE_WIFE_BIT != 0
        and int(read_ram_value(ram, "tilemap")) in HOUSE_TILEMAPS
        and int(read_ram_value(ram, "eve_hearts")) < _gift_ceiling(ram)
        and egg_available_today(ram)
    )


def apply_wife_gift(phases: list, ram: np.ndarray) -> list:
    """Put the egg chain ahead of a work morning that starts with the farm exit."""
    from harvest.core.animal_status import read_egg_available_flags
    from harvest.planner.day_phase_chicken import WIFE_GIFT_PHASES

    if not phases or phases[0].phase != "EXIT_TO_FARM":
        return phases
    if any(p.kind == "harvest_happiness" or (p.kind == "power_berry" and p.params.get("source") in {"harvest_festival", "egg_festival"}) for p in phases):
        return phases  # the dated trip needs the morning; the next egg is tomorrow
    if (int(read_ram_value(ram, "marriage_flags")) & EVE_WIFE_BIT
            and int(read_ram_value(ram, "tilemap")) in HOUSE_TILEMAPS
            and os.environ.get(WIFE_GIFTS_ENV, "1").strip().lower() not in {"0", "false", "no"}):
        if heart_drain_enabled():
            entries = wife_heart_drain_entries(ram)
            if entries > 0:
                return [*phases, *heart_drain_phases(entries, int(read_ram_value(ram, "tilemap")))]
        elif wife_fish_gift_wanted(ram):
            from harvest.planner.day_phase_types import PhaseSpec

            return [phases[0], PhaseSpec("GIVE_WIFE", "give_wife", {"fish": True},
                                        failure_policy="optional"), *phases]
    if not wife_gift_wanted(ram):
        return phases
    from harvest.planner.livestock_econ import flock_birds, hatch_allowed
    if flock_birds(ram) < 2 and hatch_allowed(ram):
        return phases  # restore the laying pair before spending its only egg
    eggs = bin(read_egg_available_flags(ram)).count("1")
    target = _gift_ceiling(ram)
    fast_below = target - 4 if target > EVE_GIFT_TARGET else EVE_GIFT_FAST_BELOW
    rounds = 2 if eggs >= 2 and int(read_ram_value(ram, "eve_hearts")) < fast_below else 1
    house = int(read_ram_value(ram, "tilemap"))
    gift_phases = [replace(p, params={**p.params, "target_tilemap": house})
                   if p.phase == "ENTER_HOUSE_FOR_GIFT" else p for p in WIFE_GIFT_PHASES]
    # Each round ends in the house, so each starts with the farm exit.
    return [*([phases[0], *gift_phases] * rounds), *phases]


def wife_sprite(ram: np.ndarray) -> Optional[Point]:
    for obj in game_objects(ram):
        if obj.kind == "npc_candidate":
            return Point(int(obj.pixel[0]), int(obj.pixel[1]))
    return None


class GiveWifeTask(Task):
    """Walk below the wife, hand over the item, confirm its expected heart delta."""

    name = "give_wife"

    def __init__(self, timeout: int = 1500, expected_delta: int = 4) -> None:
        self.timeout = timeout
        self.expected_delta = expected_delta
        self._step = 0
        self._hearts = 0
        self._leg = 0
        self._face = 0
        self._drops = 0
        self._failed: Optional[str] = None
        self._pressed_at: Optional[int] = None
        self._last = (-1, -1)
        self._still = 0

    def reset(self, world: WorldState) -> None:
        self._step = 0
        self._hearts = int(read_ram_value(world.ram, "eve_hearts"))
        self._leg = 0
        self._face = 0
        self._drops = 0
        self._failed = None
        self._pressed_at = None
        self._last, self._still = (-1, -1), 0

    def can_start(self, world: WorldState) -> bool:
        return int(read_ram_value(world.ram, "tilemap")) in HOUSE_TILEMAPS

    def _stand(self, wife: Point) -> list[tuple[Point, str]]:
        """Axis legs: to the x=98 lane, up it, then across below her."""
        below = wife.y + EVE_GIFT_STAND_DY
        return [
            (Point(EVE_GIFT_LANE_X, 200), "x"),
            (Point(EVE_GIFT_LANE_X, below), "y"),
            (Point(wife.x, below), "x"),
        ]

    def step(self, world: WorldState) -> TaskResult:
        if self._failed is None:
            result = self._step_inner(world)
            if result.status != TaskStatus.FAILURE:
                return result
            self._failed = result.reason or "gift failed"
        if int(read_ram_value(world.ram, "held_item")) == 0 or self._drops > 40:
            return TaskResult(status=TaskStatus.FAILURE, reason=self._failed)
        # A failed gift must not carry the egg into the day's tool work: A drops it.
        self._drops += 1
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(
            make_action(a=self._drops % 8 == 1)), reason=f"drop held item: {self._failed}")

    def _step_inner(self, world: WorldState) -> TaskResult:
        ram = world.ram
        self._step += 1
        hearts = int(read_ram_value(ram, "eve_hearts"))
        changed = hearts >= self._hearts + self.expected_delta if self.expected_delta > 0 else hearts <= self._hearts + self.expected_delta
        if self._pressed_at is not None and changed:
            if int(read_ram_value(ram, "input_lock")) == 1:
                return TaskResult(status=TaskStatus.SUCCESS, reason=f"eve {self._hearts}->{hearts}")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(
                make_action(a=self._step % 12 < 2)), reason="gift text")
        if self._step > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"gift timeout eve={hearts}")
        if int(read_ram_value(ram, "tilemap")) not in HOUSE_TILEMAPS:
            return TaskResult(status=TaskStatus.FAILURE, reason="not in a married house")
        if int(read_ram_value(ram, "hour")) in EVE_GIFT_CLOSED_HOURS:
            return TaskResult(status=TaskStatus.FAILURE, reason="wife script closed this hour")
        if self._pressed_at is not None:
            if self._step - self._pressed_at > 400:
                return TaskResult(status=TaskStatus.FAILURE, reason=f"no heart change eve={hearts}")
            # The add runs when the gift text closes.
            text_open = int(read_ram_value(ram, "input_lock")) != 1
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(
                make_action(a=text_open and self._step % 12 < 2)))
        if int(read_ram_value(ram, "held_item")) == 0:
            return TaskResult(status=TaskStatus.FAILURE, reason="nothing in hand")
        wife = wife_sprite(ram)
        # Her sprite spawns after the door's fade-in, a few dozen frames in.
        if wife is None or not screen_faded_in(ram):
            if self._step > EVE_SPAWN_WAIT_FRAMES:
                return TaskResult(status=TaskStatus.FAILURE, reason="wife not home")
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()))
        pos = get_pos_from_ram(ram)
        self._still = self._still + 1 if (pos.x, pos.y) == self._last else 0
        self._last = (pos.x, pos.y)
        if self._still > 90:
            others = [(o.kind, tuple(o.pixel)) for o in game_objects(ram) if not o.is_player]
            return TaskResult(status=TaskStatus.FAILURE,
                              reason=f"gift walk stuck at ({pos.x},{pos.y}) wife=({wife.x},{wife.y}) others={others}")
        legs = self._stand(wife)
        if int(read_ram_value(ram, "tilemap")) == 0x17 and self._leg < len(legs):
            from harvest.planner.tasks.inventory_exit import room_walk_action

            # The larger house has a table between the door and Eve. Read its
            # interior grid and walk round it rather than using the L1 lane.
            target = (wife.x, wife.y + EVE_GIFT_STAND_DY)
            dx, dy = target[0] - pos.x, target[1] - pos.y
            if max(abs(dx), abs(dy)) > 1:
                if self._still > EVE_GIFT_UNSTICK_FRAMES:
                    # The walker closes the last pixels sideways first. A child or the dog beside the stand blocks
                    # that sidestep for good (r47 Summer D21-D29: the farmer at (326,152), Eve at (328,120), an NPC at
                    # (339,152), no egg, -2 hearts a night, Eve 492 at the credits). Go up first, press from a near x.
                    if abs(dy) > 1:
                        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(up=dy < 0, down=dy > 0)))
                    if abs(dx) > EVE_GIFT_X_SLACK:
                        return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, target)))
                else:
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, target)))
            self._leg = len(legs)
        while self._leg < len(legs):
            target, axis = legs[self._leg]
            dx, dy = target.x - pos.x, target.y - pos.y
            if abs(dx) <= 1 and abs(dy) <= 1:
                self._leg += 1
                continue
            if (axis == "x" and abs(dx) > 1) or abs(dy) <= 1:
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(
                    make_action(left=dx < 0, right=dx > 0)))
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(
                make_action(up=dy < 0, down=dy > 0)))
        if self._face < 6:
            self._face += 1
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(up=True)))
        self._pressed_at = self._step
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(a=True)))


class FishWifeGiftTask(Task):
    """Keep one native catch, carry it home before evening and give Eve -8.

    One bounded attempt per day; failure leaves the normal work plan in charge.
    The fish is useful only after child 2's permanent flag has been retained.
    """

    name = "fish_wife_gift"

    def reset(self, world: WorldState) -> None:
        from harvest.tasks.power_berry import FishBerryTask

        self._stage = 0
        self._failed = None
        self._drops = 0
        self._child = FishBerryTask(keep_only=True, timeout=12_000)
        self._child.reset(world)

    def can_start(self, world: WorldState) -> bool:
        return wife_fish_gift_wanted(world.ram)

    def step(self, world: WorldState) -> TaskResult:
        if self._failed is None:
            result = self._step_inner(world)
            if result.status != TaskStatus.FAILURE:
                return result
            self._failed = result.reason
        if not int(read_ram_value(world.ram, "held_item")) or self._drops >= 40:
            return TaskResult(TaskStatus.FAILURE, reason=self._failed)
        self._drops += 1
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=self._drops % 8 == 1)),
                          reason=f"drop fish after failed wife trip: {self._failed}")

    def _step_inner(self, world: WorldState) -> TaskResult:
        from harvest.maps.map_config import mountain_downhill_escape, mountain_exit_then_farm
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        from harvest.maps.map_types import Waypoint
        from harvest.planner.tasks.transitions import DirectionalTransitionTask

        result = self._child.step(world)
        if result.status != TaskStatus.SUCCESS:
            return result
        self._stage += 1
        if self._stage == 1:
            pos = get_pos_from_ram(world.ram)
            hops = mountain_exit_then_farm(mountain_downhill_escape(pos.x, pos.y))
            self._child = MultiMapNavTask(waypoints=hops, timeout=8_000, allow_opportunistic_clear=False)
        elif self._stage == 2:
            self._child = MultiMapNavTask(waypoints=[Waypoint(0, p, radius=8)
                for p in ((456, 392), (136, 392), (136, 376))], timeout=4_000,
                allow_opportunistic_clear=False)
        elif self._stage == 3:
            hour = int(read_ram_value(world.ram, "hour"))
            if 11 <= hour < 15:
                self._stage -= 1
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()),
                                  reason="waiting outside for wife's 15:00 gift window")
            if hour >= 18:
                # Do not enter after the gift script has parked for the night.
                return TaskResult(TaskStatus.FAILURE, reason="fish gift missed wife's evening window")
            self._child = DirectionalTransitionTask(direction="up", origin_tilemap=0,
                target_tilemap=0x17 if int(read_ram_value(world.ram, "upgrade_flags")) & 0x80 else 0x16,
                timeout=2_500, settle_frames=20, stand_tile=(8, 23), door_align_px=136,
                overshoot_limit_px=328)
        elif self._stage == 4:
            self._child = GiveWifeTask(expected_delta=-8)
        else:
            return result
        self._child.reset(world)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
