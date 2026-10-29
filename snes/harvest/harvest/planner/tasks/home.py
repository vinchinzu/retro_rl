"""Home tasks: return, sleep, lumber, and carpenter orders.

Owns house errands and re-exports return/sleep implementations from
``home_return`` and ``home_sleep`` for the day planner.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness import Task, TaskResult, TaskStatus, WorldState

from harvest.planner.tasks.home_return import (
    HOUSE_DOOR_FRONT_PX,
    HOUSE_FRONT_PX,
    ReturnHomeTask,
)
from harvest.planner.tasks.home_sleep import (
    HOUSE_BED_STAND_PX,
    HOUSE_BED_STAND_TOLERANCE,
    HOUSE_SLEEP_TRANSITION_TILEMAP,
    GoToSleepTask,
)


@dataclass(frozen=True)
class _HouseOrder:
    """One carpenter quote. ``reserve`` is the wallet gate, ``gold`` is the debit."""

    wood: int
    reserve: int
    gold: int
    goal: str


_FIRST_HOUSE = _HouseOrder(250, 6000, 5000, "order")
_SECOND_HOUSE = _HouseOrder(500, 10000, 10000, "house2")
_WIFE = 0x0010
_HOUSE_1 = 0x0040
_HOUSE_2 = 0x0080


def _open_house_order(ram):
    """The quote still open, or None when no lumber or order should be scheduled."""
    from harvest.core.ram_catalog import read_ram_value as value

    upgrades = value(ram, "upgrade_flags")
    if upgrades & _HOUSE_2:
        return None
    married = bool(value(ram, "marriage_flags") & _WIFE)
    if married and upgrades & _HOUSE_1:
        return _SECOND_HOUSE
    if upgrades & _HOUSE_1:
        return None
    return _FIRST_HOUSE


def apply_house_phases(phases, ram):
    """Insert the open carpenter errand, or leave the day list unchanged."""
    from harvest.core.ram_catalog import read_ram_value as value
    from harvest.planner.day_phase_types import PhaseSpec
    from harvest.planner.livestock_econ import house_payment_allowed
    from harvest.planner.world_probe import WorldProbe

    probe = WorldProbe(ram=ram)
    if probe.is_storm() or probe.is_festival():
        return phases
    order = _open_house_order(ram)
    if order is None:
        return phases
    # Y1 Spring stays on crops. After that the quote can be paid in
    # Summer, Fall, or Winter, including a later year. Fall has no seed,
    # so the 10000G quote can still be open in Year 2. A Fall or Winter
    # first-house order does not arm the clock. Under two cows the first
    # house inserts neither HOUSE_WOOD nor HOUSE_UPGRADE; house-two lumber
    # is a different order and still gathers.
    if order is _FIRST_HOUSE and (
        value(ram, "eve_hearts") < 250
        or (value(ram, "year") == 0 and value(ram, "season") == 0)
        or value(ram, "num_cows") < 2
    ):
        return phases
    flags = value(ram, "marriage_flags")
    if flags & (0x20 | 0x80):
        return phases
    upgrade = PhaseSpec(
        "HOUSE_UPGRADE", "home_upgrade", {"goal": order.goal}, failure_policy="optional",
    )
    if flags & 0x40:
        return [upgrade] + list(phases)
    wood = value(ram, "stored_wood")
    if wood < order.wood and value(ram, "stamina") >= 16:
        extra = PhaseSpec(
            "HOUSE_WOOD", "house_wood", {"target": order.wood}, failure_policy="optional",
        )
    elif (wood >= order.wood and value(ram, "money") >= order.reserve
          and probe.weekday() not in (0, 6) and probe.clock().hour < 12
          and house_payment_allowed(ram)):
        extra = upgrade
    else:
        return phases
    # Dynamic crop planning follows the errand and sees the live inventory.
    at = 1 if phases and phases[0].phase == "EXIT_TO_FARM" else 0
    return list(phases[:at]) + [extra] + list(phases[at:])


class GatherHouseWoodTask(Task):
    """Fetch the axe and bank lumber, with a renewable mountain fallback.

    ``target`` is the quote. One session succeeds on that total or on the
    wood this outing actually banks. It does not invent the rest.
    """

    name = "gather_house_wood"

    def __init__(self, target: int = 250):
        self.target = int(target)
        self._asked_target = self.target

    def reset(self, world: WorldState) -> None:
        from harvest.core.ram_catalog import read_ram_value
        from harvest.planner.tasks.inventory import EnsureCarryToolTask
        self.target = self._asked_target
        order = _open_house_order(world.ram)
        if order is not None and order.wood > self.target:
            self.target = order.wood
        self._start = read_ram_value(world.ram, "stored_wood")
        self._steps = self._swings = 0
        self._phase = "axe"
        self._child = EnsureCarryToolTask(tool_id=4)
        self._child.reset(world)
        self.receipt = {}

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.ram_catalog import read_ram_value as value
        from harvest.core.tile_catalog import DebrisType
        from harvest.maps.map_config import Waypoint
        from harvest.maps.map_routes import ROUTES
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        from harvest.tasks.farm_clearer import FarmClearer
        from harvest.tasks.nav import get_pos_from_ram, make_action
        from retro_harness import ActionResult

        self._steps += 1
        ram = world.ram
        wood = value(ram, "stored_wood")
        session = self._start + 24
        gained = wood > self._start and self._phase == "chop"
        if wood >= min(self.target, session) or gained:
            self.receipt = {"start_wood": self._start, "end_wood": wood, "frames": self._steps}
            return TaskResult(TaskStatus.SUCCESS, reason=f"lumber +{wood - self._start}", meta=self.receipt)
        if self._steps > 22000:
            return TaskResult(TaskStatus.FAILURE, reason=f"lumber timeout phase={self._phase} wood={wood}")
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if self._phase == "axe":
                if result.status != TaskStatus.SUCCESS:
                    return result
                self._phase = "farm"
                self._child = FarmClearer(priority=[DebrisType.STUMP], fetch_tools=False,
                    timeout=9000, handoff="quota", quota={"stumps": 4})
            elif self._phase == "farm":
                if wood > self._start:
                    self.receipt = {"start_wood": self._start, "end_wood": wood, "frames": self._steps}
                    return TaskResult(TaskStatus.SUCCESS, reason=f"lumber +{wood - self._start}", meta=self.receipt)
                if value(ram, "stamina") < 16:
                    return TaskResult(TaskStatus.FAILURE, reason="lumber needs rested farmer")
                self._phase = "mountain"
                # Tile (20, 45) is the south exit. Due north is cliff tile
                # (20, 44). The spa hop steps off that lip before any east
                # BFS; the cabin corridor then runs east along y=696.
                hops = list(ROUTES["farm_to_mountain"]) + [
                    Waypoint(0x10, (328, 728), radius=20),
                    Waypoint(0x10, (328, 688), radius=16, run_direction="up", force_run=True),
                    Waypoint(0x10, (520, 696), radius=8, run_direction="right", force_run=True),
                    # The stump sits north of (704, 726). Reach that stand from
                    # the west so the axe faces up, instead of stopping on its
                    # north side at y=681.
                    Waypoint(0x10, (560, 726), radius=8, run_direction="down", force_run=True),
                    Waypoint(0x10, (704, 726), radius=4, run_direction="right", force_run=True),
                ]
                self._child = MultiMapNavTask(waypoints=hops, timeout=14000, initial_settle_frames=30)
            elif self._phase == "mountain":
                if result.status != TaskStatus.SUCCESS:
                    return result
                self._child = None
                self._phase = "chop"
                self._swings = 0
            if self._child is not None:
                self._child.reset(world)
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
        self._swings += 1
        if self._swings > 900:
            pos = get_pos_from_ram(ram)
            return TaskResult(TaskStatus.FAILURE, reason=f"mountain stump gave no wood at {pos}")
        # The base stump is above (704,726). Plant the facing before Y;
        # continued directional input resets the ROM's multi-hit counter.
        action = make_action(up=True) if self._swings == 1 else make_action(y=self._swings > 12 and self._swings % 48 < 20)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))


class HouseUpgradeTask(Task):
    """Request the carpenter at his cabin sign and verify the native payment."""

    name = "house_upgrade"

    def __init__(self, goal: str = "order"):
        self.goal = goal
        self._order = _FIRST_HOUSE

    def reset(self, world: WorldState) -> None:
        from harvest.core.ram_catalog import read_ram_value
        from harvest.planner.tasks.inventory import ExitToFarmTask
        self._start = {key: read_ram_value(world.ram, key) for key in ("money", "stored_wood")}
        self._steps = self._talk = 0
        self._intro = False
        self._intro_done = False
        self._reentry_wait = 0
        self._visit_due = bool(read_ram_value(world.ram, "marriage_flags") & 0x40)
        order = _open_house_order(world.ram)
        self._order = _SECOND_HOUSE if self.goal == _SECOND_HOUSE.goal or order is _SECOND_HOUSE else _FIRST_HOUSE
        self._phase = "exit"
        self._child = ExitToFarmTask()
        self._child.reset(world)
        self.receipt = {}

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.ram_catalog import read_ram_value as value
        from harvest.core.npc_catalog import current_dialogue_registers
        from harvest.maps.map_config import Waypoint
        from harvest.maps.map_routes import ROUTES, densify_waypoints
        from harvest.planner.tasks.multi_nav import MultiMapNavTask
        from harvest.planner.tasks.inventory_exit import room_walk_action
        from harvest.tasks.nav import get_pos_from_ram, make_action
        from retro_harness import ActionResult

        self._steps += 1
        ram = world.ram
        flags, upgrades = value(ram, "marriage_flags"), value(ram, "upgrade_flags")
        if self._phase == "return_wait":
            self._reentry_wait -= 1
            if self._reentry_wait <= 0:
                self._phase = "returning"
                self._child = ReturnHomeTask()
                self._child.reset(world)
        if not self._intro_done and self._phase == "sign" and current_dialogue_registers(ram)["text_id"] == 0x012A:
            self._intro = True
        if self._intro and self._phase == "sign" and value(ram, "input_lock") == 1:
            self._phase = "leave_intro"
        if self._phase == "leave_intro" and value(ram, "tilemap") == 0x10:
            self._phase = "enter"
            self._intro = False
            self._intro_done = True
            self._reentry_wait = 180
        if self._phase in ("cabin", "enter") and value(ram, "tilemap") == 0x2B:
            self._child = None
            self._phase = "sign"
        if flags & 0x80:
            paid = self._start["money"] - value(ram, "money")
            used = self._start["stored_wood"] - value(ram, "stored_wood")
            if paid != self._order.gold or used != self._order.wood:
                return TaskResult(TaskStatus.FAILURE, reason=f"unverified carpenter debit {paid}G/{used} wood")
            self.receipt = {"start": self._start, "money_spent": paid, "wood_spent": used, "frames": self._steps}
            if value(ram, "input_lock") == 1:
                ordered = f"house ordered: {self._order.gold}G and {self._order.wood} wood"
                return TaskResult(TaskStatus.SUCCESS, reason=ordered, meta=self.receipt)
        elif upgrades & _HOUSE_1 and self._order is _FIRST_HOUSE:
            return TaskResult(TaskStatus.SUCCESS, reason="first house complete")
        elif flags & 0x20 and value(ram, "input_lock") == 1 and self._phase not in ("returning", "return_wait"):
            if value(ram, "tilemap") == 0x2B:
                self._phase, self._child = "leave_request", None
            elif value(ram, "tilemap") == 0x10:
                self._phase = "return_wait"
                self._reentry_wait = 180
        if self._steps > 22000:
            pos = get_pos_from_ram(ram)
            text = current_dialogue_registers(ram)["text_id"]
            return TaskResult(TaskStatus.FAILURE, reason=f"carpenter timeout {self._phase} at {pos} text={text:#x}")
        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            if result.status != TaskStatus.SUCCESS:
                return result
            if self._phase == "returning":
                return TaskResult(TaskStatus.SUCCESS, reason="carpenter requested for tomorrow; returned home")
            if self._phase == "exit" and not flags & 0x80:
                if self._visit_due or flags & 0x40:
                    self._phase, self._child = "pay", None
                else:
                    short_money = self._start["money"] < self._order.reserve
                    short_wood = self._start["stored_wood"] < self._order.wood
                    if self.goal != "clock_arm" and (short_money or short_wood):
                        if self._order is _SECOND_HOUSE:
                            reason = "need 500 wood and 10000G"
                        else:
                            reason = "need 250 wood and 6000G including feather reserve"
                        return TaskResult(TaskStatus.FAILURE, reason=reason)
                    self._phase = "cabin"
                    hops = list(ROUTES["farm_to_mountain"]) + densify_waypoints([
                        Waypoint(0x10, (328, 696), radius=8),
                        Waypoint(0x10, (520, 696), radius=8),
                        Waypoint(0x10, (520, 568), radius=0),
                    ])
                    self._child = MultiMapNavTask(waypoints=hops, timeout=13000, initial_settle_frames=30)
                    self._child.reset(world)
            else:
                self._child = None
                self._phase = "enter" if self._phase == "cabin" else "pay"
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
        if value(ram, "input_lock") != 1:
            # Both the sign and cost menu use Yes on row zero.
            live = bool(int(ram[0x019B]) & 0x10)
            row = value(ram, "dialog_choice_result", raw=True)
            tap = self._steps % 12 < 3
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(up=live and row != 0 and tap, a=(not live or row == 0) and tap)))
        if self._phase == "return_wait":
            action = make_action()
        elif self._phase == "enter":
            self._reentry_wait = max(0, self._reentry_wait - 1)
            action = make_action() if self._reentry_wait else make_action(up=True, a=self._steps % 12 < 3)
        elif self._phase in ("leave_intro", "leave_request"):
            pos = get_pos_from_ram(ram)
            action = (room_walk_action(ram, (128, 184)) if pos.y < 180 or abs(pos.x - 128) > 3
                      else make_action(down=True, a=self._steps % 12 < 3))
        elif self._phase == "sign":
            if value(ram, "tilemap") != 0x2B:
                return TaskResult(TaskStatus.FAILURE, reason="carpenter cabin was not entered")
            pos = get_pos_from_ram(ram)
            if abs(pos.x - 152) > 2 or abs(pos.y - 160) > 2:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(room_walk_action(ram, (152, 160))))
            self._talk += 1
            action = make_action(up=self._talk < 3, a=self._talk > 8 and self._talk % 12 < 3)
        else:
            action = make_action(a=self._steps % 12 < 3)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(action))


@dataclass
class ReadyToGoHomeTask(Task):
    """Marker task: town/day work is done; planner should end the day.

    Success is the go-home flag. The day planner records it and advances into
    ``RETURN_HOME`` / ``GO_TO_SLEEP`` (or appends them if missing).
    """

    name: str = "ready_to_go_home"

    def reset(self, world: WorldState) -> None:
        return None

    def can_start(self, world: WorldState) -> bool:
        return True

    def step(self, world: WorldState) -> TaskResult:
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason="ready_to_go_home",
            checkpoint="ready_to_go_home",
            meta={"ready_to_go_home": True},
        )



__all__ = [
    "GatherHouseWoodTask",
    "HouseUpgradeTask",
    "apply_house_phases",
    "HOUSE_FRONT_PX",
    "HOUSE_DOOR_FRONT_PX",
    "HOUSE_BED_STAND_PX",
    "HOUSE_SLEEP_TRANSITION_TILEMAP",
    "HOUSE_BED_STAND_TOLERANCE",
    "ReadyToGoHomeTask",
    "ReturnHomeTask",
    "GoToSleepTask",
]
