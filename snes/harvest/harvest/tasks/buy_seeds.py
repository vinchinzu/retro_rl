"""Shelf buys closed by wallet and stock/ownership receipts.

Seeds use the flower shop (0x1C); brush/milker use the general store
(0x22); medicine uses the animal shop (0x24), delivered to the shed.
A menu or return to the farm alone never proves a purchase.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.carry import tool_in_carry_pair
from harvest.core.ram_catalog import read_ram_u8, read_ram_value
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, Tool, is_path_tilemap
from harvest.planner.livestock_econ import (
    SHED_ROW_2_ADDR,
    SHED_ROW_2_BRUSH_BIT,
    SHED_ROW_2_MILKER_BIT,
    SHOP_PRICE_G,
    TOOL_PRICE,
)
from harvest.maps.map_config import (
    SEGMENTS,
    farm_to_west_gate_waypoints,
    find_landmark,
    path_coords_leaked,
    segment_waypoints,
    slice_route_from_position,
)
from harvest.planner.day_plan_status import is_farm_tilemap, is_house_tilemap
from harvest.planner.tasks.inventory import ExitToFarmTask
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from harvest.tasks.primitives import drain_action_queue, press_a_sequence
from harvest.tasks.relationships import ErrandSocial

PATH_TILEMAP = 0x0C
TOWN_TILEMAP = 0x04
SHOP_TILEMAP = 0x1C
SHED_TILEMAP = 0x26
BARN_TILEMAP = 0x27
COOP_TILEMAP = 0x28

TOOL_SHOP_TILEMAP = 0x22

SHOP_NAV_SEGMENTS: Tuple[str, ...] = ("farm_to_path", "path_to_town_shop", "town_to_shop_door")
SHOP_BUY_SEGMENTS: Tuple[str, ...] = ("shop_to_counter",)
SHOP_RETURN_SEGMENTS: Tuple[str, ...] = ("shop_to_town", "town_shop_to_path", "path_to_farm")

@dataclass(frozen=True)
class ShopSpec:
    """One storefront: interior tilemap and the named route hops around it."""

    tilemap: int
    nav_segments: Tuple[str, ...]
    return_segments: Tuple[str, ...]  # shop -> town, town -> path, path -> farm
    town_segments: FrozenSet[str]  # hops that wait for a settled town read
    exit_min_y: int = 0  # town y the exit pixels must reach after the buy (the flip leaks mid-map coords)
    door_clear_y: int = 0  # >0: after the buy hold Down off the door while the town y is above this

FLOWER_SHOP = ShopSpec(
    SHOP_TILEMAP,
    SHOP_NAV_SEGMENTS,
    SHOP_RETURN_SEGMENTS,
    frozenset({"town_to_shop_door", "town_shop_to_path"}),
    door_clear_y=268,
)
TOOL_SHOP = ShopSpec(
    TOOL_SHOP_TILEMAP,
    ("farm_to_path", "path_to_town_shop", "town_to_tool_door"),
    ("tool_shop_to_town", "town_tool_shop_to_path", "path_to_farm"),
    frozenset({"town_to_tool_door", "town_tool_shop_to_path"}),
    exit_min_y=850,
)

ANIMAL_SHOP = ShopSpec(
    0x24, ("farm_to_path", "path_to_town_shop", "town_to_animal_door"),
    ("animal_shelf_to_town", "town_animal_shop_to_path", "path_to_farm"),
    frozenset({"town_to_animal_door", "town_animal_shop_to_path"}), exit_min_y=850,
)

@dataclass(frozen=True)
class ToolShelf:
    """A general-store shelf tool. ROM: ``DATA8_82CFB4`` item, ``DATA16_81BE0F`` shed bit.

    Shelf object F4 (14,21) is milker $4B, F6 (14,22) is brush $4C; both sit in
    the east wall, so the stand is x=217 facing right (A probes pos_x+7, and 216
    is the wall stop). Payment posts with the "thank you" box; the shed bit is
    ORed when the box closes.
    """

    tool_id: int
    price: int
    stand: Tuple[int, int]
    shed_bit: int  # $7F1F01 (shed row 2)
    counter_segment: str
    face: str = "right"
    shed_addr: int = SHED_ROW_2_ADDR

TOOL_SHED_ROW_ADDR = SHED_ROW_2_ADDR
TOOL_SHELVES: Dict[int, ToolShelf] = {
    int(Tool.SPRINKLER): ToolShelf(int(Tool.SPRINKLER), SHOP_PRICE_G["sprinkler"], (216, 342),
                                   0x10, "tool_shop_to_milker", "up", 0x11F02),
    9: ToolShelf(9, SHOP_PRICE_G["medicine"], (58, 134), 0x01, "animal_shop_to_medicine", "up"),
    int(Tool.BRUSH): ToolShelf(int(Tool.BRUSH), TOOL_PRICE[int(Tool.BRUSH)], (217, 358),
                               SHED_ROW_2_BRUSH_BIT, "tool_shop_to_brush"),
    int(Tool.MILKER): ToolShelf(int(Tool.MILKER), TOOL_PRICE[int(Tool.MILKER)], (217, 344),
                                SHED_ROW_2_MILKER_BIT, "tool_shop_to_milker"),
}
TOOL_STAND_X = 217

# buy_potato_seeds_d2: first A is at (182,342) tile (11,21) face up.
SEED_CLERK_PX = (182, 342)
# The shelf bags in the general store are the items on sale: A facing one
# streams its description, then the buy prompt (cursor 0 = buy). One stand per
# bag along y=342 (measured Summer D13: corn x=152 and tomato x=168 each took
# 300 G and added one bag; potato x=182 in spring).
SEED_STAND_X = {"turnip_seeds": 200, "potato_seeds": 182, "corn_seeds": 152, "tomato_seeds": 168, "grass_seeds": 216}
SEED_BAG_PRICE = {f"{kind}_seeds": SHOP_PRICE_G[f"{kind}_seed"] for kind in ("turnip", "potato", "corn", "tomato", "grass")}
SEED_STAND_Y = 342
SEED_CLERK_RADIUS = 10
SHOP_INTERIOR_SETTLE = 45
# The path->town warp leaves the pad dead until game_state 0x8000 (scene
# transition) clears, ~300f after arrival (measured D2). A nav started earlier
# presses into a dead pad, reads it as pixel-stuck and walks back out the gate.
TOWN_SETTLE_FRAMES = 480
# A pulses allowed per bag. Potato closes in a handful; the corn/tomato shelf
# bags stream a 3-page description first (~17 pulses measured Summer D13).
BUY_ATTEMPT_LIMIT = 28
ENTER_PULSE_LIMIT = 90
# An inbound town hop that ends back on the path (a villager on the east gate
# push-faced it out, Summer D27) walks back in this many times before failing.
TOWN_BOUNCE_RETRIES = 2
POTATO_BAG_PRICE = 200

def _tilemap(world: WorldState) -> int:
    ram = world.ram
    return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

def _needs_building_exit(tilemap: int) -> bool:
    return is_house_tilemap(tilemap) or tilemap in (SHED_TILEMAP, BARN_TILEMAP, COOP_TILEMAP)

def _stock(ram, field: str = "potato_seeds") -> int:
    try:
        return int(read_ram_value(ram, field) or 0)
    except Exception:
        return 0

def _money(ram) -> int:
    try:
        return int(read_ram_value(ram, "money") or 0)
    except Exception:
        return 0

def tool_owned(ram, tool_id: int) -> bool:
    """On the shed shelf (row-2 bit) or in the carry pair."""
    shelf = TOOL_SHELVES[int(tool_id)]
    if read_ram_u8(ram, shelf.shed_addr) & shelf.shed_bit:
        return True
    return tool_in_carry_pair(ram, int(tool_id))

def plan_tool_buys(ram, tool_ids: Sequence[int]) -> List[int]:
    """Shelf tools to buy this trip: unowned and affordable, in the order given.

    The wallet is spent down as it goes, so a short wallet buys the first tool
    it covers and never an item it cannot pay for.
    """
    money = _money(ram)
    plan: List[int] = []
    for tool_id in tool_ids:
        shelf = TOOL_SHELVES.get(int(tool_id))
        if shelf is None or int(tool_id) in plan or tool_owned(ram, int(tool_id)):
            continue
        if money < shelf.price:
            continue
        money -= shelf.price
        plan.append(int(tool_id))
    return plan

def shop_door_px() -> Tuple[int, int]:
    found = find_landmark("shop_door", tilemap_id=TOWN_TILEMAP)
    if found is None:
        return (37 * 16 + 8, 13 * 16 + 8)
    return found[1].target_px

def seed_stand_px(stock_field: str = "potato_seeds") -> Tuple[int, int]:
    """Shop-floor stand under the shelf bag for ``stock_field``."""
    return (SEED_STAND_X.get(stock_field, SEED_CLERK_PX[0]), SEED_STAND_Y)

def at_seed_clerk(
    ram,
    *,
    radius: int = SEED_CLERK_RADIUS,
    stand: Tuple[int, int] = SEED_CLERK_PX,
) -> bool:
    pos = get_pos_from_ram(ram)
    return (
        abs(int(pos.x) - stand[0]) <= radius
        and abs(int(pos.y) - stand[1]) <= radius
    )

def shop_interior_coords_settled(ram) -> bool:
    """Town door pixels linger on 0x1C for a few frames after the flip."""
    pos = get_pos_from_ram(ram)
    return int(pos.x) < 400 or int(pos.y) > 250

def town_coords_settled(ram) -> bool:
    """Path-left entry leaks ~(10,128) before the east-gate stand ~(756,422)."""
    from harvest.planner.tasks.transitions import screen_faded_in
    if ADDR_TILEMAP < len(ram):
        tilemap = int(ram[ADDR_TILEMAP])
        if tilemap != 0 and tilemap != TOWN_TILEMAP:
            return False
    pos = get_pos_from_ram(ram)
    if int(pos.x) <= 200 or int(pos.y) <= 100:
        return False
    return not (int(ram[0xD3]) & 0x80) and screen_faded_in(ram)

def first_shop_nav_segment(
    tilemap: int,
    segments: Sequence[str] = SHOP_NAV_SEGMENTS,
    shop_tilemap: int = SHOP_TILEMAP,
) -> Optional[str]:
    """Skip hops the live map has already finished."""
    if _needs_building_exit(tilemap):
        return None
    if is_farm_tilemap(tilemap):
        return "farm_to_path" if "farm_to_path" in segments else None
    if is_path_tilemap(tilemap):
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and is_path_tilemap(hops[0].tilemap) and name != "farm_to_path":
                return name
        return None
    if tilemap == TOWN_TILEMAP:
        for name in segments:
            hops = SEGMENTS.get(name, [])
            if hops and hops[0].tilemap == TOWN_TILEMAP:
                return name
        return None
    if tilemap == shop_tilemap:
        return None
    return segments[0] if segments else None

def first_shop_return_segment(
    tilemap: int,
    segments: Sequence[str] = SHOP_RETURN_SEGMENTS,
    shop_tilemap: int = SHOP_TILEMAP,
) -> Optional[str]:
    if tilemap == shop_tilemap:
        return segments[0]
    if tilemap == TOWN_TILEMAP:
        return segments[1]
    if is_path_tilemap(tilemap):
        return segments[2]
    return None

def purchase_closed(*, stock_before: int, stock_after: int, money_before: int, money_after: int) -> bool:
    return stock_after > stock_before and money_after < money_before

@dataclass
class BuyShelfItemTask(Task):
    """Farm/path/town → shop door → shelf buy → farm, RAM-closed.

    Seeds (default): flower shop 0x1C, receipt is stock up and wallet down.
    ``tool_ids``: general store 0x22, one trip buys each unowned affordable tool
    in order; receipt per tool is the wallet drop of exactly its price and the
    shed row-2 bit. Success needs every planned tool received.
    """

    name: str = "buy_seeds"
    timeout: int = 18_000
    nav_timeout: int = 6_000
    stock_field: str = "potato_seeds"
    bag_price: int = POTATO_BAG_PRICE
    # Buy several bags in one trip, capped by the wallet at reset().
    bags: int = 1
    # Seed crops bought after this one on the same visit, ``(stock_field, bags)``: the shelf is one row, a few steps apart.
    then: Tuple[Tuple[str, int], ...] = ()
    # General-store shelf tools (Tool ids) instead of a seed bag.
    tool_ids: Tuple[int, ...] = ()
    # The day composer shares this ledger across its shopping phases.
    social_visits: set = field(default_factory=set, repr=False)
    _social: ErrandSocial = field(default_factory=lambda: ErrandSocial(set()), init=False, repr=False)

    shop: ShopSpec = field(default=FLOWER_SHOP, init=False)
    _plan: List[int] = field(default_factory=list, init=False, repr=False)
    _cur_tool: int = field(default=0, init=False)
    _receipts: List[str] = field(default_factory=list, init=False, repr=False)
    _own_wait: int = field(default=0, init=False)
    _nothing: bool = field(default=False, init=False)
    _got: List[int] = field(default_factory=list, init=False, repr=False)
    _step_count: int = field(default=0, init=False)
    _child: Optional[Task] = field(default=None, init=False)
    _child_name: str = field(default="", init=False)
    _done_nav: set[str] = field(default_factory=set, init=False, repr=False)
    _phase: str = field(default="nav", init=False)
    _stock_before: int = field(default=0, init=False)
    _money_before: int = field(default=0, init=False)
    _seen_shop: bool = field(default=False, init=False)
    _bought: bool = field(default=False, init=False)
    _buy_attempts: int = field(default=0, init=False)
    _bags_seen: int = field(default=0, init=False)
    _attempts_at_last_bag: int = field(default=0, init=False)
    _enter_pulses: int = field(default=0, init=False)
    _interior_wait: int = field(default=0, init=False)
    _town_wait: int = field(default=0, init=False)
    _buy_queue: deque = field(default_factory=deque, init=False, repr=False)
    _last_reason: str = field(default="start", init=False)
    _path_settle: int = field(default=0, init=False)
    _align_steps: int = field(default=0, init=False)
    _town_bounces: int = field(default=0, init=False)
    _then: List[Tuple[str, int]] = field(default_factory=list, init=False, repr=False)
    _seed_receipts: List[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tool_ids:
            unknown = [t for t in self.tool_ids if int(t) not in TOOL_SHELVES]
            if unknown:
                raise ValueError(f"no general-store shelf for tool ids {unknown}")
            if 9 in self.tool_ids and tuple(self.tool_ids) != (9,):
                raise ValueError("medicine requires its own animal-shop trip")
            self.shop = ANIMAL_SHOP if 9 in self.tool_ids else TOOL_SHOP
            if self.name == "buy_seeds":
                self.name = "buy_tools"
        elif self.bag_price == POTATO_BAG_PRICE:
            self.bag_price = SEED_BAG_PRICE.get(self.stock_field, POTATO_BAG_PRICE)

    @property
    def stand_px(self) -> Tuple[int, int]:
        if self._cur_tool:
            return TOOL_SHELVES[self._cur_tool].stand
        return seed_stand_px(self.stock_field)

    @property
    def face(self) -> str:
        return TOOL_SHELVES[self._cur_tool].face if self._cur_tool else "up"

    @property
    def counter_segment(self) -> str:
        return TOOL_SHELVES[self._cur_tool].counter_segment if self._cur_tool else SHOP_BUY_SEGMENTS[0]

    def _held(self, ram) -> int:
        """Receipt counter: seed stock, or 1 once the shelf tool is owned."""
        if self._cur_tool:
            return int(tool_owned(ram, self._cur_tool))
        return _stock(ram, self.stock_field)

    def _activate_item(self, ram) -> None:
        """Snapshot the wallet and receipt counter for the item about to be bought."""
        if self._cur_tool:
            self.bag_price = TOOL_SHELVES[self._cur_tool].price
        self._stock_before = self._held(ram)
        self._money_before = _money(ram)
        self._bought = False
        self._buy_attempts = 0
        self._bags_seen = 0
        self._attempts_at_last_bag = 0
        self._own_wait = 0
        self._align_steps = 0
        self._buy_queue.clear()

    def reset(self, world: WorldState) -> None:
        self._social = ErrandSocial(visited=self.social_visits)
        self._align_steps = 0
        self._step_count = 0
        self._child = None
        self._child_name = ""
        self._done_nav.clear()
        self._phase = "nav"
        self._plan = plan_tool_buys(world.ram, self.tool_ids)
        self._nothing = bool(self.tool_ids) and not self._plan
        self._cur_tool = self._plan[0] if self._plan else 0
        self._receipts = []
        self._got = []
        self._then = [(str(f), int(n)) for f, n in self.then if int(n) > 0 and f in SEED_STAND_X]
        self._seed_receipts = []
        self._seen_shop = _tilemap(world) == self.shop.tilemap
        self._activate_item(world.ram)
        self._enter_pulses = 0
        self._interior_wait = 0
        self._town_wait = 0
        self._buy_queue.clear()
        self._last_reason = "start"
        self._path_settle = 0
        self._town_bounces = 0
        self._arm_nav(world)

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def phase_text(self) -> str:
        return self._child_name or self._phase

    def progress_snapshot(self) -> ProgressSnapshot:
        child = task_progress_snapshot(self._child) if self._child is not None else None
        return ProgressSnapshot(
            task_name=self.name,
            phase_text=self.phase_text,
            step_count=self._step_count,
            details=(
                ("stock_before", self._stock_before),
                ("money_before", self._money_before),
                ("tool", self._cur_tool),
                ("seen_shop", int(self._seen_shop)),
                ("bought", int(self._bought)),
            ),
            child=child,
        )

    def _purchase_ok(self, ram) -> bool:
        stock = self._held(ram)
        money = _money(ram)
        if self._cur_tool:
            # The wallet debit proves payment; the shed bit follows at box close.
            return self._money_before - money == int(self.bag_price)
        if purchase_closed(
            stock_before=self._stock_before,
            stock_after=stock,
            money_before=self._money_before,
            money_after=money,
        ):
            return True
        # Wallet posts inside 0x1C before the bag count. Treat a full
        # bag-price debit as the buy so we can leave and re-read stock.
        return money <= self._money_before - max(1, int(self.bag_price))

    def _bags_target(self) -> int:
        """Bags this trip wants, clamped to what the wallet held at reset."""
        if self._cur_tool:
            return 1
        affordable = self._money_before // max(1, int(self.bag_price))
        return max(1, min(int(self.bags), int(affordable)))

    def _bags_done(self, ram) -> int:
        """Purchases closed so far. Stock and wallet disagree mid-dialogue."""
        if self._cur_tool:
            return int(self._purchase_ok(ram))
        by_stock = _stock(ram, self.stock_field) - self._stock_before
        by_money = max(0, self._money_before - _money(ram) - self._social.gold_spent) // max(
            1, int(self.bag_price)
        )
        return max(0, int(by_stock), int(by_money))

    def _success(self, ram) -> TaskResult:
        if self.tool_ids:
            if not self._got:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason=f"no tool bought: {'; '.join(self._receipts)}",
                )
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason=f"bought tools {'; '.join(self._receipts)}",
            )
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                "bought " + " then ".join([*self._seed_receipts, self._seed_receipt(ram)])
                + (f" social={self._social.receipts}" if self._social.receipts else "")
            ),
        )

    def _seed_receipt(self, ram) -> str:
        return (
            f"{self.stock_field} {self._stock_before}->{_stock(ram, self.stock_field)} "
            f"money {self._money_before}->{_money(ram)} bags={self._bags_done(ram)}/{self._bags_target()}"
        )

    def _town_ready(self, ram) -> bool:
        """Town coords are real: past the gate leak and, after a buy, at the shop apron."""
        if not town_coords_settled(ram):
            return False
        return not (self._bought and int(get_pos_from_ram(ram).y) < self.shop.exit_min_y)

    def _nav_for(self, name: str, world: WorldState) -> MultiMapNavTask:
        pos = get_pos_from_ram(world.ram)
        tilemap = _tilemap(world)
        if name == "farm_to_path":
            hops = farm_to_west_gate_waypoints(int(pos.x), int(pos.y), tilemap=tilemap)
        else:
            hops = list(SEGMENTS.get(name, []))
            if not hops:
                hops = segment_waypoints(name)
        # Path-leak town pixels would slice to the shop door and walk off
        # the east gate. Keep the full east-to-door list until coords settle.
        # Inbound town hops start from the east gate; never slice against unsettled coords.
        if name in self.shop.town_segments and not self._bought:
            sliced = hops
        elif name in self.shop.town_segments and not self._town_ready(world.ram):
            sliced = hops
        else:
            sliced = slice_route_from_position(hops, pos.x, pos.y, tilemap=tilemap)
        timeout = self.nav_timeout
        if name in self.shop.town_segments:
            timeout = max(timeout, 9_000)
        task = MultiMapNavTask(
            name=f"seg_{name}",
            waypoints=sliced or hops,
            timeout=timeout,
            initial_settle_frames=12,
            allow_opportunistic_clear=not self._social.pending_gift,
        )
        task.reset(world)
        return task

    def _arm_nav(self, world: WorldState) -> None:
        tilemap = _tilemap(world)
        if _needs_building_exit(tilemap):
            self._child = ExitToFarmTask()
            self._child.reset(world)
            self._child_name = "exit_to_farm"
            self._last_reason = "leaving building"
            return
        if self._bought:
            if tilemap == TOWN_TILEMAP and not self._town_ready(world.ram):
                self._child = None
                self._child_name = "town_settle"
                self._last_reason = "town settle after shop"
                return
            pos = get_pos_from_ram(world.ram)
            if tilemap == TOWN_TILEMAP and int(pos.y) < self.shop.door_clear_y:
                # Doorface ~(601,232). Walk south off (37,14) before MultNav.
                self._child = None
                self._child_name = "leave_door"
                self._last_reason = "walk south off shop door"
                return
            nxt = first_shop_return_segment(
                tilemap, self.shop.return_segments, self.shop.tilemap
            )
            if nxt is None or nxt in self._done_nav:
                self._child = None
                self._child_name = "done"
                return
            self._child = self._nav_for(nxt, world)
            self._child_name = nxt
            self._last_reason = f"nav {nxt}"
            return
        if tilemap == self.shop.tilemap:
            self._seen_shop = True
            if not shop_interior_coords_settled(world.ram) and self._interior_wait < SHOP_INTERIOR_SETTLE:
                self._child = None
                self._child_name = "shop_settle"
                self._last_reason = "shop interior settle"
                return
            at_counter = at_seed_clerk(world.ram, stand=self.stand_px) or (
                not self._cur_tool and at_seed_clerk(world.ram, stand=SEED_CLERK_PX)
            )
            if at_counter:
                self._child = None
                self._child_name = "buy"
                self._phase = "buy"
                self._last_reason = "at clerk"
                return
            if self.counter_segment not in self._done_nav:
                self._child = self._nav_for(self.counter_segment, world)
                self._child_name = self.counter_segment
                self._last_reason = f"nav {self.counter_segment}"
                return
            self._child = None
            self._child_name = "buy"
            self._phase = "buy"
            return
        if tilemap == TOWN_TILEMAP and not town_coords_settled(world.ram) and self._town_wait < TOWN_SETTLE_FRAMES:
            self._child = None
            self._child_name = "town_settle"
            self._last_reason = "town east-gate settle"
            return
        remaining = [name for name in self.shop.nav_segments if name not in self._done_nav]
        nxt = first_shop_nav_segment(tilemap, remaining, self.shop.tilemap)
        if nxt is None:
            if is_farm_tilemap(tilemap):
                self._child = None
                self._child_name = "farm_exit_walk"
                self._last_reason = "farm exit walk"
                return
            if is_path_tilemap(tilemap):
                self._child = None
                self._child_name = "path_exit_walk"
                self._last_reason = "path exit walk"
                return
            self._child = None
            self._child_name = "enter"
            self._phase = "enter"
            self._last_reason = "at shop door"
            return
        self._child = self._nav_for(nxt, world)
        self._child_name = nxt
        self._last_reason = f"nav {nxt}"

    def _bounced_to_path(self, name: str, tilemap: int) -> bool:
        """Re-walk path -> town after an inbound town hop lands back on the path."""
        if (
            self._bought
            or name not in self.shop.town_segments
            or name not in self.shop.nav_segments
            or not is_path_tilemap(tilemap)
            or self._town_bounces >= TOWN_BOUNCE_RETRIES
        ):
            return False
        self._town_bounces += 1
        self._town_wait = 0
        self._done_nav.discard(name)
        self._done_nav.discard("path_to_town_shop")
        print(
            f"[BUY_SEEDS] {name} bounced to the path; walking back into town "
            f"({self._town_bounces}/{TOWN_BOUNCE_RETRIES})"
        )
        return True

    def _align_to_stand(self, world: WorldState) -> Optional[TaskResult]:
        """Walk along the shelf from the proven potato hop to this seed's stand.

        The shop nav is only proven to (182,342); the other shelf bags are a
        straight walk left/right along y=342 (measured Summer D13).
        """
        want, want_y = self.stand_px
        pos = get_pos_from_ram(world.ram)
        x, y = int(pos.x), int(pos.y)
        row_off = self._cur_tool and abs(y - want_y) > 2
        if abs(x - want) <= 1 and not row_off:
            return None
        self._align_steps += 1
        if self._align_steps > 240:
            return None  # let the buy loop fail with its own reason
        if row_off and self.face == "up":
            direction = "down" if y < want_y else "up"
        elif row_off:
            # The shelf column is solid: hugging the wall (x>=217) blocks the
            # row change, so back off to x<=215 before moving up or down.
            if x > want - 2:
                direction = "left"
            else:
                direction = "down" if y < want_y else "up"
        else:
            direction = "left" if x > want else "right"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(make_action(**{direction: True})),
            reason=f"align to {self.stock_field or self._cur_tool} stand {want, want_y} from {x, y}",
        )

    def _next_tool(self, ram) -> None:
        """Drop the finished or skipped tool and arm the next one the wallet covers.

        The next shelf is the adjacent stand, so it aligns in place (no re-nav).
        With nothing left, ``_bought`` hands the task to the walk home.
        """
        self._plan.pop(0)
        while self._plan:
            tool = self._plan[0]
            if _money(ram) >= TOOL_SHELVES[tool].price:
                self._cur_tool = tool
                self._activate_item(ram)
                self._done_nav.add(self.counter_segment)
                self._child = None
                self._child_name = "buy"
                self._phase = "buy"
                return
            self._receipts.append(f"{tool:#04x} skipped, wallet {_money(ram)}")
            self._plan.pop(0)
        self._bought = True

    def _next_seed(self, ram) -> None:
        """Book the finished crop and arm the next one; its shelf bag is a short walk along the same row."""
        self._seed_receipts.append(self._seed_receipt(ram))
        self.stock_field, self.bags = self._then.pop(0)
        self.bag_price = SEED_BAG_PRICE.get(self.stock_field, POTATO_BAG_PRICE)
        self._activate_item(ram)
        self._child = None
        self._child_name = "buy"
        self._phase = "buy"

    def _queue_buy(self, world: WorldState, *, lock: int) -> TaskResult:
        if self._buy_attempts == 0:
            self._stock_before = self._held(world.ram)
            # The wallet can move between reset and the shelf (the Y2+ peddler
            # takes 20,000G on the farm road), so the price delta is measured here.
            self._money_before = _money(world.ram)
            if self._cur_tool and self._money_before < TOOL_SHELVES[self._cur_tool].price:
                self._receipts.append(f"{self._cur_tool:#04x} skipped, wallet {self._money_before}")
                self._next_tool(world.ram)
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action()),
                    reason="shelf tool unaffordable at the shelf",
                )
        if self._buy_attempts >= BUY_ATTEMPT_LIMIT * self._bags_target() and self._seed_receipts:
            # A follow-on crop that will not close must not cost the crop already bought: walk home with it.
            print(f"[BUY] {self.stock_field} did not close after {self._buy_attempts} A presses; leaving with {self._seed_receipts}")
            self._then.clear()
            self._bought = True
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason="follow-on crop unverified")
        if self._buy_attempts >= BUY_ATTEMPT_LIMIT * self._bags_target():
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=(
                    f"shop buy unverified after {self._buy_attempts} A "
                    f"stock={self._stock_before} money={self._money_before}"
                ),
            )
        pos = get_pos_from_ram(world.ram)
        # Tape last step is Up onto (182,342), then A in place. Extra Up
        # walks off the clerk into the counter.
        if not self._cur_tool and abs(int(pos.y) - SEED_CLERK_PX[1]) > 2:
            face = "up" if int(pos.y) > SEED_CLERK_PX[1] else "down"
            self._buy_queue.extend(
                press_a_sequence(
                    face,
                    face_frames=min(8, abs(int(pos.y) - SEED_CLERK_PX[1])),
                    pre_press_settle_frames=12,
                    hold_frames=10,
                    settle_frames=36,
                    hold_face_with_a=False,
                )
            )
        else:
            self._buy_queue.extend(
                press_a_sequence(
                    self.face if lock == 1 else None,
                    face_frames=1 if lock == 1 else 0,
                    pre_press_settle_frames=20 if lock == 1 else 8,
                    hold_frames=8,
                    settle_frames=40 if lock != 2 else 28,
                )
            )
        self._buy_attempts += 1
        self._last_reason = f"buy attempt={self._buy_attempts} lock={lock}"
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(self._buy_queue.popleft()),
            reason=self._last_reason,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        social = self._social.step(self, world)
        if social is not None:
            return social
        if self._step_count > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"{self.name} timeout after {self._step_count}f phase={self.phase_text}",
            )

        if self._nothing:
            return TaskResult(
                status=TaskStatus.SUCCESS,
                reason="no shelf tool to buy (owned or wallet short)",
            )
        tilemap = _tilemap(world)
        if tilemap == self.shop.tilemap:
            if not self._seen_shop:
                self._town_wait = 0
            self._seen_shop = True
        elif (
            self._bought
            and tilemap == TOWN_TILEMAP
            and self._child_name == self.shop.return_segments[0]
        ):
            # Fresh settle after the shop→town flip (inbound wait already spent).
            self._town_wait = 0
        done = self._bags_done(world.ram) if self._seen_shop and self._buy_attempts else 0
        if done > self._bags_seen:
            self._bags_seen = done
            self._attempts_at_last_bag = self._buy_attempts
        if done >= self._bags_target() or (
            self._seen_shop and self._buy_attempts > 0 and self._bags_target() <= 1 and self._purchase_ok(world.ram)
        ):
            self._bought = True
        elif (
            done >= 1
            and self._buy_attempts - self._attempts_at_last_bag >= BUY_ATTEMPT_LIMIT
        ):
            # A second bag that will not close must not cost us the first one.
            # Bank what the wallet already paid for and walk home.
            print(
                f"[BUY] extra bag did not close after {BUY_ATTEMPT_LIMIT} A "
                f"presses; leaving with {done}/{self._bags_target()}"
            )
            self._bought = True
        lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
        if self._bought and tilemap == self.shop.tilemap and lock in {0, 2, 4}:
            queued = drain_action_queue(self._buy_queue)
            if queued is not None:
                return queued
            self._buy_queue.extend(
                press_a_sequence(None, face_frames=0, pre_press_settle_frames=6, hold_frames=6, settle_frames=18)
            )
            self._last_reason = "dismiss shop menu"
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(self._buy_queue.popleft()),
                reason=self._last_reason,
            )
        if self._plan and self._bought and tilemap == self.shop.tilemap and lock == 1:
            # Box closed: book the shed bit, then take the next shelf tool.
            if tool_owned(world.ram, self._cur_tool):
                self._receipts.append(
                    f"{self._cur_tool:#04x} money {self._money_before}->{_money(world.ram)}"
                )
                self._got.append(self._cur_tool)
                self._next_tool(world.ram)
            else:
                self._own_wait += 1
                if self._own_wait > 90:
                    return TaskResult(
                        status=TaskStatus.FAILURE,
                        reason=(
                            f"tool {self._cur_tool:#04x} paid {self._money_before}->"
                            f"{_money(world.ram)} but the shed bit never set"
                        ),
                    )
        if self._then and self._bought and tilemap == self.shop.tilemap and lock == 1 and not self._cur_tool:
            self._next_seed(world.ram)
        if (
            self._bought
            and self._seen_shop
            and is_farm_tilemap(tilemap)
            and not (self._cur_tool and self._plan)
        ):
            return self._success(world.ram)

        if self._child_name == "farm_exit_walk":
            pos = get_pos_from_ram(world.ram)
            if is_path_tilemap(tilemap) and not path_coords_leaked(pos.x, pos.y):
                self._done_nav.add("farm_to_path")
                self._arm_nav(world)
            else:
                self._last_reason = "farm exit walk"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)), reason=self._last_reason)

        if self._child_name == "path_exit_walk":
            if tilemap == TOWN_TILEMAP and town_coords_settled(world.ram):
                self._child = None
                self._arm_nav(world)
            else:
                self._last_reason = "path exit walk"
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(left=True, b=True)), reason=self._last_reason)

        if self._child_name == "leave_door":
            pos = get_pos_from_ram(world.ram)
            if int(pos.y) >= 268 or _tilemap(world) != TOWN_TILEMAP:
                self._arm_nav(world)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(down=True, b=True)), reason=self._last_reason)

        if self._child_name == "town_settle":
            self._town_wait += 1
            # Inbound may proceed after a cap (full east-gate list). After
            # shop, leak (138,468) walking toward (602,274) re-enters 0x1C.
            ready = self._town_ready(world.ram)
            inbound = not self._bought
            if ready or (inbound and self._town_wait >= TOWN_SETTLE_FRAMES):
                self._arm_nav(world)
            elif self._bought and self._town_wait > TOWN_SETTLE_FRAMES:
                # The town fade-in alone runs ~190 frames after the shop door; a 180-frame
                # cap failed the power-on D2 seed buy and stranded D2_FARM_CLEAR in town.
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="town coords never settled after shop exit",
                )
            action = make_action(down=True) if self._bought and self.shop.door_clear_y else make_action()
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action),
                reason=self._last_reason,
            )

        if self._child_name == "shop_settle":
            self._interior_wait += 1
            if shop_interior_coords_settled(world.ram) or self._interior_wait >= SHOP_INTERIOR_SETTLE:
                self._arm_nav(world)
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action()),
                reason=self._last_reason,
            )

        if self._phase == "enter" and tilemap != self.shop.tilemap:
            self._enter_pulses += 1
            if self._enter_pulses > ENTER_PULSE_LIMIT:
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="shop_door enter failed; MultNav did not reach 0x1C",
                )
            # Face-up A then walk up. Door is the landmark open face.
            if self._enter_pulses % 20 < 8:
                return TaskResult(
                    status=TaskStatus.RUNNING,
                    action=ActionResult(make_action(a=True)),
                    reason="shop_door A",
                )
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(up=True)),
                reason="shop_door walk up",
            )
        if self._phase == "enter" and tilemap == self.shop.tilemap:
            self._phase = "nav"
            self._enter_pulses = 0
            self._arm_nav(world)

        if self._phase == "buy" and not self._bought:
            lock = int(world.ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(world.ram) else 1
            queued = drain_action_queue(self._buy_queue)
            if queued is not None:
                return queued
            if lock in {0, 2, 4} and self._buy_attempts > 0:
                # Dialogue still up — keep A pulses. Buy is cursor 0.
                return self._queue_buy(world, lock=lock)
            if lock == 1 and self._buy_attempts == 0 and self._align_to_stand(world) is not None:
                return self._align_to_stand(world)
            if at_seed_clerk(world.ram, stand=self.stand_px) or self._buy_attempts == 0:
                return self._queue_buy(world, lock=lock)
            self._phase = "nav"
            self._arm_nav(world)

        pos = get_pos_from_ram(world.ram)
        if (
            not self._bought
            and is_path_tilemap(tilemap)
            and path_coords_leaked(pos.x, pos.y)
            and self._path_settle <= 180
        ):
            self._path_settle += 1
            self._child = None
            self._child_name = "path_settle"
            run = self._path_settle > 24
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(left=True, b=run)),
                reason="path coords settle",
            )
        if self._path_settle and not (
            is_path_tilemap(tilemap) and path_coords_leaked(pos.x, pos.y)
        ):
            self._path_settle = 0
            self._done_nav.add("farm_to_path")
            self._child = None
            self._arm_nav(world)

        if (
            self._child is not None
            and not self._bought
            and self._child_name in self.shop.nav_segments
        ):
            remaining = [name for name in self.shop.nav_segments if name not in self._done_nav]
            live = first_shop_nav_segment(tilemap, remaining, self.shop.tilemap)
            if (
                live is not None
                and live != self._child_name
                and self._child_name != "farm_to_path"
            ):
                finished = self._child_name
                self._done_nav.add(finished)
                self._child = None
                self._arm_nav(world)
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=f"{finished} clipped")

        if self._child is not None:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            name = self._child_name
            self._child = None
            if result.status == TaskStatus.FAILURE:
                if self._bounced_to_path(name, tilemap):
                    self._arm_nav(world)
                    return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=f"{name} bounced to the path; walking back in")
                return TaskResult(status=TaskStatus.FAILURE, reason=f"{name} failed: {result.reason}")
            if (
                name in self.shop.nav_segments
                or name == self.counter_segment
                or name in self.shop.return_segments
            ):
                self._done_nav.add(name)
            if name == self.counter_segment:
                self._phase = "buy"
            self._arm_nav(world)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=self._last_reason)

        if self._bought and not is_farm_tilemap(tilemap):
            self._arm_nav(world)
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action()), reason=self._last_reason)

        if not self._bought and tilemap == TOWN_TILEMAP:
            self._phase = "enter"
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(up=True)), reason="shop_door walk up")

        return TaskResult(
            status=TaskStatus.FAILURE,
            reason=f"shop unfinished tm=0x{tilemap:02X} seen={self._seen_shop} bought={self._bought}",
        )

# Seed-only callers predate the tool shelves.
BuySeedsTask = BuyShelfItemTask

__all__ = [
    "BuySeedsTask", "BuyShelfItemTask", "ANIMAL_SHOP", "FLOWER_SHOP", "ShopSpec", "TOOL_SHELVES",
    "TOOL_SHOP", "TOOL_SHOP_TILEMAP", "ToolShelf", "plan_tool_buys", "tool_owned",
    "PATH_TILEMAP", "POTATO_BAG_PRICE", "SEED_CLERK_PX", "SEED_BAG_PRICE",
    "SEED_STAND_X", "seed_stand_px", "SHOP_NAV_SEGMENTS", "SHOP_RETURN_SEGMENTS",
    "SHOP_TILEMAP", "TOWN_TILEMAP", "at_seed_clerk", "first_shop_nav_segment",
    "first_shop_return_segment", "purchase_closed", "shop_door_px", "town_coords_settled",
]
