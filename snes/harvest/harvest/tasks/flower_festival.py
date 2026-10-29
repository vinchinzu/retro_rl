"""Spring D23 Flower Festival: talk to everyone, then dance with Eve (+30).

Walk off the town's west edge before 15:00 and map ``0x08`` loads at (264,312)
with the clock set to 07:00 (docs/EVE_BAR.md). Talking to a girl pays +8 /
+12 / +15 by her hearts and clears her bit in ``$7F1F74``; the six villagers
clear bits 0-5. Once all eleven are clear the blue-haired woman at (120,248)
steps away from the alcove, the host there asks "Can we start dancing now?"
and "Let's dance" sets bit 11. The next girl talked to is the partner and gets
+30 at the end event, which then puts the farmer to bed: the next morning is
D24 06:00 in the house.

The festival ground has no walkability data. ``flower_festival_walk.json`` is
a lattice of 8-pixel hops probed with emulator save/restore
(``harvest.scripts.probe_walk_grid``); NPCs stand still, so their bodies are
already holes in it.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from harvest.core.npc_catalog import current_dialogue_registers
from harvest.core.ram_catalog import read_ram_u8, read_ram_u16, read_ram_value, field_spec
from harvest.core.tile_catalog import ADDR_TILEMAP
from harvest.maps.map_routes import segment_waypoints
from harvest.maps.map_types import Waypoint
from harvest.planner.tasks.multi_nav import MultiMapNavTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

TOWN_TILEMAP = 0x04
FESTIVAL_TILEMAP = 0x08
HOUSE_TILEMAP = 0x15
EDGE_X_MAX = 26  # town west strip; Left from y 128-150 loads the festival ground
EDGE_Y = (120, 156)
PATH_TILEMAP = 0x0C
FARM_TILEMAPS = (0x00, 0x01, 0x02, 0x03)
ACTION_ARROW = 9
ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_FLAGS_74 = 0x1F74  # $7F1F74 (live offset applies through read_ram_u16)
GRID_PATH = Path(__file__).resolve().parent.parent / "maps" / "flower_festival_walk.json"

UP, DOWN, LEFT, RIGHT = 1, 2, 4, 8
HOPS = {"up": (0, -1, UP), "down": (0, 1, DOWN), "left": (-1, 0, LEFT), "right": (1, 0, RIGHT)}

SETTLE_FRAMES = 240
ENTER_TIMEOUT = 400
ARRIVE_TOL = 2
STALL_FRAMES = 14
START_TIMEOUT = 70
QUIET_FRAMES = 70
STOP_TIMEOUT = 1500
DANCE_TIMEOUT = 9000


@dataclass(frozen=True)
class Stop:
    name: str
    npc: Tuple[int, int]  # where the NPC stands
    stand: Tuple[int, int]  # where the farmer talks from
    face: str
    bit: Optional[int] = None  # $7F1F74 bit this talk clears; None when it sets or gates one


# Order: Eve first (+12 at hearts 101-300), the other girls, the six villagers,
# the blue-haired woman, the host, then Eve again as the partner. Stands and
# bits were read off a full run (what-if pin, 2026-10-15).
STOPS: Tuple[Stop, ...] = (
    Stop("eve", (280, 344), (277, 326), "down", 10),
    Stop("maria", (328, 184), (310, 187), "right", 6),
    Stop("ann", (200, 296), (218, 293), "left", 7),
    Stop("nina", (232, 136), (229, 154), "up", 8),
    Stop("ellen", (312, 328), (294, 325), "right", 9),
    Stop("villager_184_136", (184, 136), (202, 139), "left", 0),
    Stop("villager_392_152", (392, 152), (361, 155), "right", 1),
    Stop("villager_184_216", (184, 216), (202, 213), "left", 2),
    Stop("villager_424_248", (424, 248), (406, 245), "right", 3),
    Stop("villager_248_248", (248, 248), (263, 251), "left", 4),
    Stop("villager_280_248", (280, 248), (262, 246), "right", 5),
    Stop("blue_woman", (120, 248), (138, 251), "left"),
)
HOST = Stop("host", (88, 232), (91, 249), "up")
HOST_APPROACH = (93, 250)  # straight left along the row once the blue woman has moved
PARTNER = Stop("eve_partner", (280, 344), (277, 326), "down")
ALL_TALKED_MASK = 0x07FF
HOST_BIT = 1 << 11
# Group 0x0E event 0 places this sprite at $B3F21D, pixel (392, 152).
# Text 0x0214 is the snow-flower seed (DATA16_B8BACE). Choice 0 takes
# text 0x0215, AddMoney $FFFF38 (−2,000G), and sets $7F1F02 bit 6.
# Choice 1 is text 0x0216 and leaves the wallet alone. The dance host
# stays on row 0.
SNOW_SEED_STOP = "villager_392_152"
_CHOICE_FLAG = 0x019B


class WalkGrid:
    """Lattice of verified 8-pixel hops with an A* over it."""

    def __init__(self, origin: Tuple[int, int], step: int, cells: Dict[Tuple[int, int], int]) -> None:
        self.origin, self.step, self.cells = origin, step, cells

    @classmethod
    def load(cls, path: Path = GRID_PATH) -> "WalkGrid":
        data = json.loads(path.read_text())
        cells = {(i, j): m for i, j, m in data["cells"]}
        return cls(tuple(data["origin"]), int(data["step"]), cells)

    def cell_of(self, px: Tuple[int, int]) -> Tuple[int, int]:
        return (
            round((px[0] - self.origin[0]) / self.step),
            round((px[1] - self.origin[1]) / self.step),
        )

    def center(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        return (self.origin[0] + cell[0] * self.step, self.origin[1] + cell[1] * self.step)

    def nearest(self, px: Tuple[int, int], radius: int = 3) -> Optional[Tuple[int, int]]:
        base = self.cell_of(px)
        best = None
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                cell = (base[0] + di, base[1] + dj)
                if cell in self.cells:
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, cell)
        return best[1] if best else None

    def path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        closed: frozenset = frozenset(),
    ) -> Optional[List[str]]:
        """Hop names from ``start`` to ``goal``, or None."""
        frontier = [(0, start)]
        cost = {start: 0}
        came: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        while frontier:
            _, cell = heapq.heappop(frontier)
            if cell == goal:
                hops: List[str] = []
                while cell != start:
                    cell, name = came[cell]
                    hops.append(name)
                return hops[::-1]
            mask = self.cells.get(cell, 0)
            for name, (di, dj, bit) in HOPS.items():
                nxt = (cell[0] + di, cell[1] + dj)
                if not mask & bit or nxt not in self.cells or (cell, name) in closed:
                    continue
                new = cost[cell] + 1
                if new < cost.get(nxt, 1 << 30):
                    cost[nxt] = new
                    came[nxt] = (cell, name)
                    h = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                    heapq.heappush(frontier, (new + h, nxt))
        return None


@dataclass
class FlowerFestivalTask(Task):
    """Take Eve through the Flower Festival. SUCCESS when the day rolls to D24."""

    name: str = "flower_festival"
    partner: str = "eve"
    timeout: int = 60000
    settle_frames: int = SETTLE_FRAMES

    _grid: Optional[WalkGrid] = field(default=None, init=False)
    _phase: str = field(default="to_edge", init=False)
    _frames: int = field(default=0, init=False)
    _steps: int = field(default=0, init=False)
    _stop_index: int = field(default=0, init=False)
    _stop: Optional[Stop] = field(default=None, init=False)
    _route: List[str] = field(default_factory=list, init=False)
    _hop: Optional[Tuple[str, Tuple[int, int]]] = field(default=None, init=False)
    _closed: set = field(default_factory=set, init=False)
    _replans: int = field(default=0, init=False)
    _attempts: int = field(default=0, init=False)
    _nav: Optional[Task] = field(default=None, init=False)
    _start_day: int = field(default=0, init=False)
    _last_pos: Optional[Tuple[int, int]] = field(default=None, init=False)
    _stall: int = field(default=0, init=False)
    _texts: List[int] = field(default_factory=list, init=False)
    _seen_lock: bool = field(default=False, init=False)
    _quiet: int = field(default=0, init=False)
    _talk_frames: int = field(default=0, init=False)
    _fail: Optional[str] = field(default=None, init=False)
    _flags_text: str = field(default="", init=False)
    _host_row_reached: bool = field(default=False, init=False)

    # ── RAM reads ────────────────────────────────────────────────────
    @staticmethod
    def _tilemap(ram: np.ndarray) -> int:
        return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

    @staticmethod
    def _pos(ram: np.ndarray) -> Tuple[int, int]:
        p = get_pos_from_ram(ram)
        return (p.x, p.y)

    @staticmethod
    def _flags(ram: np.ndarray) -> int:
        return read_ram_u16(ram, 0x11F74)

    # ── lifecycle ────────────────────────────────────────────────────
    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def progress_text(self) -> str:
        return f"{self._phase} stop={self._stop_index} flags={self._flags_text}"

    def reset(self, world: WorldState) -> None:
        if self._grid is None:
            self._grid = WalkGrid.load()
        ram = world.ram
        self._start_day = read_ram_value(ram, "day")
        self._steps = 0
        self._stop_index = 0
        self._fail = None
        self._nav = None
        # Inside the ground already: still wait, the entry scene may be mid-fade.
        self._begin("settle" if self._tilemap(ram) == FESTIVAL_TILEMAP else "to_edge")

    def _begin(self, phase: str) -> None:
        self._phase, self._frames = phase, 0
        self._stall, self._last_pos = 0, None

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _act(reason: str, **buttons: bool) -> TaskResult:
        return TaskResult(
            status=TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)), reason=reason
        )

    def _stalled(self, ram: np.ndarray) -> bool:
        here = self._pos(ram)
        self._stall = self._stall + 1 if here == self._last_pos else 0
        self._last_pos = here
        return self._stall >= STALL_FRAMES

    def _fail_now(self, reason: str) -> TaskResult:
        return TaskResult(status=TaskStatus.FAILURE, reason=reason)

    # ── step ─────────────────────────────────────────────────────────
    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        self._frames += 1
        ram = world.ram
        self._flags_text = f"{self._flags(ram):#06x}" if self._tilemap(ram) == FESTIVAL_TILEMAP else ""
        if self._steps > self.timeout:
            return self._fail_now(f"flower festival timeout in {self._phase}")
        if read_ram_value(ram, "day") != self._start_day:
            return TaskResult(status=TaskStatus.SUCCESS, reason="festival over, day rolled")
        return getattr(self, f"_phase_{self._phase}")(world)

    # ── getting in ───────────────────────────────────────────────────
    def _phase_to_edge(self, world: WorldState) -> TaskResult:
        ram = world.ram
        tilemap = self._tilemap(ram)
        if tilemap == FESTIVAL_TILEMAP:
            self._begin("settle")
            return self._act("in_festival")
        px = self._pos(ram)
        if tilemap == TOWN_TILEMAP and px[0] <= EDGE_X_MAX and EDGE_Y[0] <= px[1] <= EDGE_Y[1]:
            self._begin("enter")
            return self._act("at_edge")
        if self._nav is None:
            hops = self._hops_to_edge(tilemap)
            if hops is None:
                return self._fail_now(f"flower festival cannot start from tilemap=0x{tilemap:02X}")
            self._nav = MultiMapNavTask(waypoints=hops, timeout=30000, initial_settle_frames=30)
            self._nav.reset(world)
        result = self._nav.step(world)
        if result.status == TaskStatus.FAILURE:
            return self._fail_now(f"could not reach the town west edge: {result.reason}")
        if result.status == TaskStatus.SUCCESS:
            self._nav = None
            self._begin("enter")
            return self._act("at_edge")
        return TaskResult(status=TaskStatus.RUNNING, action=result.action, reason="walk_to_edge")

    @staticmethod
    def _hops_to_edge(tilemap: int) -> Optional[List[Waypoint]]:
        if tilemap in FARM_TILEMAPS:
            return segment_waypoints("farm_to_path", "path_to_town")
        if tilemap == PATH_TILEMAP:
            return segment_waypoints("path_to_town")
        if tilemap == TOWN_TILEMAP:  # out the south gate, up the path, in at the west edge
            return segment_waypoints("town_south_to_path", "path_to_town")
        return None

    def _phase_enter(self, world: WorldState) -> TaskResult:
        if self._tilemap(world.ram) == FESTIVAL_TILEMAP:
            self._begin("settle")
            return self._act("entered_festival")
        if self._frames > ENTER_TIMEOUT:
            return self._fail_now("the festival ground did not load (D23, before 15:00?)")
        return self._act("enter_festival", left=True)

    def _phase_settle(self, world: WorldState) -> TaskResult:
        if self._frames < self.settle_frames:
            return self._act("festival_settle")
        self._start_stop(0)
        return self._act("festival_ready")

    # ── stops ────────────────────────────────────────────────────────
    def _start_stop(self, index: int, stop: Optional[Stop] = None) -> None:
        self._stop_index = index
        self._stop = stop or STOPS[index]
        self._attempts = 0
        self._begin("walk")
        self._route, self._hop, self._closed, self._replans = [], None, set(), 0
        self._host_row_reached = False

    def _next_stop(self, ram: np.ndarray) -> TaskResult:
        index = self._stop_index + 1
        if index < len(STOPS):
            self._start_stop(index)
            return self._act("next_stop")
        flags = self._flags(ram)
        if flags & ALL_TALKED_MASK:
            return self._fail_now(f"villagers left unspoken, flags={flags:#06x}")
        self._start_stop(index, HOST)
        return self._act("to_host")

    def _phase_walk(self, world: WorldState) -> TaskResult:
        ram = world.ram
        if self._tilemap(ram) != FESTIVAL_TILEMAP:
            return self._fail_now("left the festival ground")
        stop = self._stop
        here = self._pos(ram)
        if stop is HOST and not self._host_row_reached:
            return self._host_walk(ram, here)
        if self._frames > STOP_TIMEOUT:
            return self._fail_now(f"could not reach {stop.name}")
        if self._hop is not None:
            return self._follow_hop(ram, here)
        if self._reached(here, stop.stand):
            self._begin("talk")
            self._reset_talk()
            return self._act("at_stand")
        grid = self._grid
        start, goal = grid.nearest(here), grid.nearest(stop.stand)
        if start is None or goal is None:
            return self._fine_align(ram, here, stop)
        route = grid.path(start, goal, frozenset(self._closed))
        if route is None:
            return self._fail_now(f"no festival route to {stop.name} from {here}")
        if not route:
            return self._fine_align(ram, here, stop)
        self._take_hop(route[0], start)
        return self._follow_hop(ram, here)

    def _host_walk(self, ram: np.ndarray, here: Tuple[int, int]) -> TaskResult:
        if self._frames > STOP_TIMEOUT:
            return self._fail_now("could not reach the host")
        if abs(here[1] - HOST_APPROACH[1]) > 3:
            return self._act("host_row", up=here[1] > HOST_APPROACH[1], down=here[1] < HOST_APPROACH[1])
        if here[0] > HOST_APPROACH[0] + 2:
            return self._act("host_walk", left=True)
        self._host_row_reached = True
        return self._act("host_row_done")

    @staticmethod
    def _reached(here: Tuple[int, int], target: Tuple[int, int], tol: int = 2) -> bool:
        return abs(here[0] - target[0]) <= tol and abs(here[1] - target[1]) <= tol

    def _take_hop(self, name: str, cell: Tuple[int, int]) -> None:
        di, dj, _ = HOPS[name]
        self._hop = (name, self._grid.center((cell[0] + di, cell[1] + dj)))
        self._stall, self._last_pos = 0, None

    def _follow_hop(self, ram: np.ndarray, here: Tuple[int, int]) -> TaskResult:
        name, target = self._hop
        axis = 0 if name in ("left", "right") else 1
        err = target[axis] - here[axis]
        if abs(err) <= ARRIVE_TOL:
            self._hop = None
            return self._act("hop_done")
        if self._stalled(ram):
            cell = self._grid.nearest(here)
            self._closed.add((cell, name))
            self._hop = None
            self._replans += 1
            if self._replans > 6:
                return self._fail_now(f"festival walk blocked near {here} going {name}")
            return self._act("hop_blocked")
        return self._act("hop", **{name: True})

    def _fine_align(self, ram: np.ndarray, here: Tuple[int, int], stop: Stop) -> TaskResult:
        dx, dy = stop.stand[0] - here[0], stop.stand[1] - here[1]
        if self._stalled(ram):  # pressed against something: talk from here
            dx = dy = 0
        if abs(dx) >= abs(dy) and abs(dx) > 1:
            return self._act("align", right=dx > 0, left=dx < 0)
        if abs(dy) > 1:
            return self._act("align", down=dy > 0, up=dy < 0)
        self._begin("talk")
        self._reset_talk()
        return self._act("at_stand")

    # ── talking ──────────────────────────────────────────────────────
    def _reset_talk(self) -> None:
        self._texts, self._seen_lock, self._quiet, self._talk_frames = [], False, 0, 0

    def _box_buttons(self, ram: np.ndarray) -> dict:
        """Page with A. The snow-seed seller's menu confirms on row 0."""
        pressed = self._talk_frames % 12 < 3
        if self._stop is not None and self._stop.name == SNOW_SEED_STOP:
            live = len(ram) > _CHOICE_FLAG and bool(int(ram[_CHOICE_FLAG]) & 0x10)
            if live:
                row = int(read_ram_value(ram, "dialog_choice_result", raw=True))
                if row != 1:
                    return {"down": pressed}
                return {"a": pressed}
        return {"a": pressed}

    def _phase_talk(self, world: WorldState) -> TaskResult:
        ram = world.ram
        if self._tilemap(ram) != FESTIVAL_TILEMAP:
            return self._after_talk(ram)
        d = current_dialogue_registers(ram)
        action = read_ram_u8(ram, ADDR_PLAYER_ACTION)
        self._talk_frames += 1
        if d["input_lock"] == 2:
            self._seen_lock, self._quiet = True, 0
            if not self._texts or self._texts[-1] != d["text_id"]:
                self._texts.append(d["text_id"])
        if not self._seen_lock:
            if self._talk_frames > START_TIMEOUT:
                return self._after_talk(ram)
            return self._act("talk", a=self._talk_frames % 30 < 3, **{self._stop.face: True})
        if d["input_lock"] == 2:
            return self._act("talk", **self._box_buttons(ram))
        self._quiet += 1
        if self._quiet > QUIET_FRAMES and action == 0:
            return self._after_talk(ram)
        return self._act("talk_close")

    def _after_talk(self, ram: np.ndarray) -> TaskResult:
        stop = self._stop
        flags = self._flags(ram)
        if stop is PARTNER:
            self._begin("dance")
            return self._act("dance")
        ok = bool(self._texts)
        if stop.bit is not None:
            ok = ok and not flags & (1 << stop.bit)
        elif stop is HOST:
            ok = ok and bool(flags & HOST_BIT)
        elif stop.name == "blue_woman":
            ok = ok and not flags & ALL_TALKED_MASK  # she steps aside once all eleven are clear
        if ok:
            if stop is HOST:
                self._start_stop(self._stop_index + 1, PARTNER)
                self._begin("to_partner")
                return self._act("to_partner")
            return self._next_stop(ram)
        self._attempts += 1
        if self._attempts >= 3:
            return self._fail_now(f"talking to {stop.name} did not register (flags={flags:#06x})")
        self._begin("walk")
        self._route, self._hop = [], None
        return self._act("retry_stop")

    # ── partner and dance ────────────────────────────────────────────
    def _phase_to_partner(self, world: WorldState) -> TaskResult:
        """Back out of the alcove along the row, then walk the lattice to Eve."""
        ram = world.ram
        here = self._pos(ram)
        if here[0] < STOPS[-1].stand[0]:
            return self._act("leave_alcove", right=True) if abs(here[1] - 250) <= 6 else self._act(
                "leave_alcove", down=here[1] < 250, up=here[1] > 250
            )
        self._start_stop(self._stop_index, PARTNER)
        return self._act("to_partner_walk")

    def _phase_dance(self, world: WorldState) -> TaskResult:
        ram = world.ram
        d = current_dialogue_registers(ram)
        if self._frames > DANCE_TIMEOUT:
            return self._fail_now("the dance never ended the day")
        if d["input_lock"] == 2 or read_ram_u8(ram, ADDR_PLAYER_ACTION) == ACTION_ARROW:
            return self._act("dance_text", a=self._frames % 12 < 3)
        return self._act("dance_wait")


__all__ = ["FlowerFestivalTask", "STOPS", "HOST", "PARTNER", "WalkGrid"]
