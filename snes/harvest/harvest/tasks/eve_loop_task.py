"""Eve at the bar: pump her hearts in one night.

The regular bar script (event group 5 event 0, Eve's CC script at $B3D213,
``docs/EVE_BAR.md``) pays per talk like this:

    hearts 0-119 or 201+   Spring juice question, "No thank you" +8 (half the
                           talks, never D22 or rain). Winter +2 per talk.
    hearts 120-149         nothing, "dead zone"
    hearts 150-200         "Sometimes" +8, every entry

After a +8 answer the script parks in a loop that only says "just kidding",
so each +8 costs one bar re-entry. Outdoors the hour caps at 18 and the bar
interior does not advance the clock, so entries are free at night: enter,
talk, leave, repeat. The first entry ever runs the first-meeting script with
Eve at the door side and no hearts; it is skipped by leaving and re-entering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from harvest.core.npc_catalog import (
    ROMANCE_HEART_THRESHOLDS,
    current_dialogue_registers,
    game_objects,
    romance_points_for_hearts,
)
from harvest.core.ram_catalog import field_spec, read_ram_u8, read_ram_u16, read_ram_value
from harvest.core.tile_catalog import ADDR_TILEMAP, is_path_tilemap
from harvest.maps.map_types import Waypoint
from harvest.planner.tasks.multi_nav import MultiMapNavTask
from harvest.tasks.nav import get_pos_from_ram, make_action
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

ADDR_PLAYER_ACTION = field_spec("player_action").address
ADDR_WEATHER_FLAGS = 0x0196  # live_offset=False; bit 0x02 rain, 0x08 snow

TOWN_TILEMAP = 0x04
BAR_TILEMAP = 0x1E
DOOR_PX = (152, 872)  # town pixel just below the bar door
DOOR_RADIUS = 8
NEAR_DOOR = 160  # closer than this the task steps straight at the door, farther it walks the town
EVE_SPRITES = range(0x0244, 0x024A)
FIRST_VISIT_MIN_X = 120  # first-meeting script seats Eve at (168, 440), regular at (56, 448)
STAND_DX, STAND_DY = 22, -2  # player stand relative to Eve, facing left

DEAD_ZONE = (120, 149)
CHOICE_BAND = (150, 200)
TEXT_JUICE = 0x00E6  # "Yes" +4 / "No thank you" +8
TRAP_TEXTS = frozenset({0x00E7, 0x00E8, 0x00F0, 0x00F1})  # answers that park the script
ACTION_ARROW = 9  # player_action while a text box waits on its arrow

SETTLE_FRAMES = 190  # fade-out, black, fade-in after the door; input is dead before this
OUTSIDE_FRAMES = 40
ENTER_TIMEOUT = 400
LEAVE_TIMEOUT = 900
APPROACH_TIMEOUT = 400
START_TIMEOUT = 90
QUIET_FRAMES = 40
CHOICE_TIMEOUT = 700
ENTRY_TIMEOUT = 6000

RAIN, SNOW = 0x0002, 0x0008
PATH_TILEMAP = 0x0C
FARM_TILEMAPS = frozenset({0x00, 0x01, 0x02, 0x03})


def _hops_to_bar(tilemap: int):
    """Farm or path waypoints that end on the bar door. None from anywhere else."""
    from harvest.maps.map_routes import segment_waypoints

    door = Waypoint(TOWN_TILEMAP, DOOR_PX, radius=DOOR_RADIUS)
    if tilemap in FARM_TILEMAPS:
        return segment_waypoints("farm_to_path", "path_to_town") + [door]
    if is_path_tilemap(tilemap):
        return segment_waypoints("path_to_town") + [door]
    return None


def bar_talk_outlook(hearts: int, *, season: int, day: int, weather_flags: int) -> str:
    """What one regular-bar talk with Eve can pay right now.

    ``choice`` +8 every entry, ``juice`` +8 on about half the talks (once per
    entry), ``winter`` +2 per talk, ``dead_zone`` and ``none`` pay nothing.
    Mirrors the script at $B3D2D4 branch for branch.
    """
    hearts = int(hearts)
    if DEAD_ZONE[0] <= hearts <= DEAD_ZONE[1]:
        return "dead_zone"
    if CHOICE_BAND[0] <= hearts <= CHOICE_BAND[1]:
        return "choice"
    if weather_flags & SNOW:
        return "winter" if day == 30 else "none"
    if season == 0:
        return "none" if day == 22 or weather_flags & RAIN else "juice"
    if season == 3:
        return "none" if day == 23 else "winter"
    return "none"


@dataclass
class EveTalkLoopTask(Task):
    """Enter the bar, talk to Eve until she pays, leave, repeat to a heart target."""

    name: str = "eve_talk_loop"
    target_hearts: int = 10  # tier 1-10, stops at that tier's top points
    target_points: Optional[int] = None  # explicit points; 250 is the proposal floor
    max_entries: int = 80
    max_empty_entries: int = 6
    max_talks_per_entry: int = 6
    timeout: int = 360000
    align_timeout: int = 1200
    settle_frames: int = SETTLE_FRAMES
    check_outlook: bool = True

    _phase: str = field(default="align", init=False)
    _phase_frames: int = field(default=0, init=False)
    _step_count: int = field(default=0, init=False)
    _target: int = field(default=999, init=False)
    _final: Optional[TaskResult] = field(default=None, init=False)
    _entries: int = field(default=0, init=False)
    _empty_entries: int = field(default=0, init=False)
    _first_visits: int = field(default=0, init=False)
    _entry_start_hearts: int = field(default=0, init=False)
    _entry_frames: int = field(default=0, init=False)
    _talks_this_entry: int = field(default=0, init=False)
    _eve: Optional[Tuple[int, int]] = field(default=None, init=False)
    _talk_frames: int = field(default=0, init=False)
    _talk_hearts0: int = field(default=0, init=False)
    _texts: List[int] = field(default_factory=list, init=False)
    _seen_lock: bool = field(default=False, init=False)
    _quiet: int = field(default=0, init=False)
    _menu_open: bool = field(default=False, init=False)
    _choice_frames: int = field(default=0, init=False)
    _start_misses: int = field(default=0, init=False)
    _stall: int = field(default=0, init=False)
    _last_pos: Optional[Tuple[int, int]] = field(default=None, init=False)
    _nav: Optional[Task] = field(default=None, init=False)
    _exit: Optional[Task] = field(default=None, init=False)

    # ── RAM reads ────────────────────────────────────────────────────
    @staticmethod
    def _hearts(ram: np.ndarray) -> int:
        return read_ram_u16(ram, field_spec("eve_hearts").address)

    @staticmethod
    def _tilemap(ram: np.ndarray) -> int:
        return int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0

    def _outlook(self, ram: np.ndarray) -> str:
        return bar_talk_outlook(
            self._hearts(ram),
            season=read_ram_value(ram, "season"),
            day=read_ram_value(ram, "day"),
            weather_flags=read_ram_u16(ram, ADDR_WEATHER_FLAGS, live_offset=False),
        )

    @staticmethod
    def _find_eve(ram: np.ndarray) -> Optional[Tuple[int, int]]:
        for obj in game_objects(ram):
            if not obj.is_player and obj.sprite_table_idx in EVE_SPRITES:
                return obj.pixel
        return None

    # ── lifecycle ────────────────────────────────────────────────────
    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def progress_text(self) -> str:
        return f"{self._phase} entries={self._entries} target={self._target}"

    def reset(self, world: WorldState) -> None:
        self._target = (
            int(self.target_points)
            if self.target_points is not None
            else romance_points_for_hearts(self.target_hearts)
        )
        tilemap = self._tilemap(world.ram)
        inside = tilemap == BAR_TILEMAP
        if inside:
            self._set_phase("settle")
            self._phase_frames = self.settle_frames  # already inside: the scene is live
        elif tilemap == TOWN_TILEMAP:
            self._set_phase("align")
        else:
            self._set_phase("to_town")
        self._step_count = self._entries = self._empty_entries = self._first_visits = 0
        self._final = None
        self._nav = self._exit = None
        self._reset_entry(world.ram)

    def _reset_entry(self, ram: np.ndarray) -> None:
        self._entry_start_hearts = self._hearts(ram)
        self._entry_frames = self._talks_this_entry = self._start_misses = 0
        self._eve = None
        self._reset_talk(ram)

    def _reset_talk(self, ram: np.ndarray) -> None:
        self._talk_frames = 0
        self._talk_hearts0 = self._hearts(ram)
        self._texts = []
        self._seen_lock = False
        self._quiet = self._choice_frames = 0
        self._menu_open = False

    def _set_phase(self, phase: str) -> None:
        self._phase, self._phase_frames = phase, 0
        self._stall, self._last_pos = 0, None

    # ── actions ──────────────────────────────────────────────────────
    @staticmethod
    def _act(reason: str, **buttons: bool) -> TaskResult:
        return TaskResult(
            status=TaskStatus.RUNNING, action=ActionResult(make_action(**buttons)), reason=reason
        )

    def _finish(self, status: TaskStatus, reason: str) -> TaskResult:
        return TaskResult(status=status, reason=reason)

    def _stop(self, ram: np.ndarray, status: TaskStatus, reason: str) -> TaskResult:
        """End the task, walking out of the bar first when inside it."""
        self._final = self._finish(status, reason)
        if self._tilemap(ram) == BAR_TILEMAP:
            self._set_phase("leave")
            return self._act("leaving_bar")
        return self._final

    def _stalled(self, ram: np.ndarray) -> bool:
        pos = get_pos_from_ram(ram)
        here = (pos.x, pos.y)
        self._stall = self._stall + 1 if here == self._last_pos else 0
        self._last_pos = here
        return self._stall >= 12

    # ── step ─────────────────────────────────────────────────────────
    def step(self, world: WorldState) -> TaskResult:
        self._step_count += 1
        self._phase_frames += 1
        ram = world.ram
        if self._step_count > self.timeout:
            return self._finish(TaskStatus.FAILURE, "eve talk loop timeout")
        tilemap = self._tilemap(ram)
        hearts = self._hearts(ram)

        if self._phase != "leave" and self._final is None:
            if hearts >= self._target:
                return self._stop(
                    ram,
                    TaskStatus.SUCCESS,
                    f"eve hearts {hearts}/{self._target} entries={self._entries}",
                )
            if self.check_outlook and self._phase in ("to_town", "align", "enter", "settle", "approach"):
                outlook = self._outlook(ram)
                if outlook in ("dead_zone", "none"):
                    return self._stop(
                        ram,
                        TaskStatus.FAILURE,
                        f"eve {outlook} at hearts={hearts}/{self._target}: "
                        "bar talk pays nothing (docs/EVE_BAR.md)",
                    )
            if self._entries >= self.max_entries:
                return self._stop(ram, TaskStatus.FAILURE, f"eve entry cap hearts={hearts}/{self._target}")
            if self._empty_entries >= self.max_empty_entries:
                return self._stop(
                    ram, TaskStatus.FAILURE, f"eve paid nothing in {self._empty_entries} entries"
                )
            if tilemap == BAR_TILEMAP:
                self._entry_frames += 1
                if self._entry_frames > ENTRY_TIMEOUT:
                    return self._stop(ram, TaskStatus.FAILURE, "eve entry stuck")

        handler = getattr(self, f"_phase_{self._phase}")
        return handler(world, tilemap, hearts)

    # ── walk to town, then the door ──────────────────────────────────
    def _phase_to_town(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap in (TOWN_TILEMAP, BAR_TILEMAP):
            self._nav = self._exit = None
            self._set_phase("settle" if tilemap == BAR_TILEMAP else "align")
            return self._act("town_reached")
        if self._exit is not None:
            result = self._exit.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._exit = None
            if result.status != TaskStatus.SUCCESS:
                return self._finish(TaskStatus.FAILURE, f"eve could not leave for town: {result.reason}")
            return self._act("left_for_town")
        if self._nav is None:
            from harvest.planner.day_plan_status import is_house_tilemap
            from harvest.planner.tasks.inventory import ExitToFarmTask

            if is_house_tilemap(tilemap):
                self._exit = ExitToFarmTask()
                self._exit.reset(world)
                return self._act("leave_for_town")
            hops = _hops_to_bar(tilemap)
            if not hops:
                pos = get_pos_from_ram(world.ram)
                return self._finish(
                    TaskStatus.FAILURE,
                    f"expected Eve loop origin tilemap=0x{TOWN_TILEMAP:02X} near={DOOR_PX}, "
                    f"got tilemap=0x{tilemap:02X} pos=({pos.x},{pos.y})",
                )
            self._nav = MultiMapNavTask(waypoints=hops, timeout=20000, initial_settle_frames=30)
            self._nav.reset(world)
        result = self._nav.step(world)
        if result.status == TaskStatus.FAILURE:
            return self._finish(TaskStatus.FAILURE, f"eve could not reach town: {result.reason}")
        if result.status == TaskStatus.SUCCESS:
            self._nav = None
            self._set_phase("align")
            return self._act("town_reached")
        return TaskResult(status=TaskStatus.RUNNING, action=result.action, reason="walk_to_town")

    # ── outside the door ─────────────────────────────────────────────
    def _phase_align(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        ram = world.ram
        if tilemap == BAR_TILEMAP:
            self._set_phase("settle")
            return self._act("inside_bar")
        pos = get_pos_from_ram(ram)
        if tilemap == TOWN_TILEMAP:
            dx, dy = DOOR_PX[0] - pos.x, DOOR_PX[1] - pos.y
            if abs(dx) <= DOOR_RADIUS and abs(dy) <= DOOR_RADIUS:
                self._set_phase("enter")
                return self._act("origin_ready")
            if self._nav is not None or max(abs(dx), abs(dy)) > NEAR_DOOR:
                return self._walk_to_door(world)  # sticky: the walk owns the last stretch too
            if self._phase_frames <= self.align_timeout:
                if pos.y < 700 and self._phase_frames < 120:
                    return self._act("align_outside")
                if abs(dx) > DOOR_RADIUS:
                    return self._act("align_outside", right=dx > 0, left=dx < 0)
                return self._act("align_outside", down=dy > 0, up=dy < 0)
        return self._finish(
            TaskStatus.FAILURE,
            f"expected Eve loop origin tilemap=0x{TOWN_TILEMAP:02X} near={DOOR_PX}, "
            f"got tilemap=0x{tilemap:02X} pos=({pos.x},{pos.y})",
        )

    def _walk_to_door(self, world: WorldState) -> TaskResult:
        if self._nav is None:
            self._nav = MultiMapNavTask(
                waypoints=[Waypoint(TOWN_TILEMAP, DOOR_PX, radius=DOOR_RADIUS)],
                timeout=20000,
                initial_settle_frames=30,
            )
            self._nav.reset(world)
        result = self._nav.step(world)
        if result.status == TaskStatus.FAILURE:
            return self._finish(TaskStatus.FAILURE, f"eve could not reach the bar door: {result.reason}")
        if result.status == TaskStatus.SUCCESS:
            self._nav = None
            self._set_phase("enter")
            return self._act("origin_ready")
        return TaskResult(status=TaskStatus.RUNNING, action=result.action, reason="walk_to_bar")

    def _phase_enter(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap == BAR_TILEMAP:
            self._entries += 1
            self._reset_entry(world.ram)
            self._set_phase("settle")
            return self._act("entered_bar")
        if self._phase_frames > ENTER_TIMEOUT:
            return self._finish(TaskStatus.FAILURE, "eve bar door did not open (closed today?)")
        return self._act("enter_bar", up=True, b=True)

    def _phase_outside(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if self._phase_frames >= OUTSIDE_FRAMES:
            self._set_phase("enter")
        return self._act("outside_bar")

    # ── inside the bar ───────────────────────────────────────────────
    def _phase_settle(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap != BAR_TILEMAP:
            self._set_phase("align")
            return self._act("outside_bar")
        if self._phase_frames < self.settle_frames:
            return self._act("bar_settle")
        self._eve = self._find_eve(world.ram)
        if self._eve is None:
            self._empty_entries += 1
            self._set_phase("leave")
            return self._act("eve_absent")
        if self._eve[0] > FIRST_VISIT_MIN_X:
            self._first_visits += 1
            if self._first_visits > 2:
                return self._stop(
                    world.ram, TaskStatus.FAILURE, "eve stays at the first-meeting spot"
                )
            self._set_phase("leave")
            return self._act("first_visit_skip")
        self._set_phase("approach")
        return self._act("bar_ready")

    def _phase_approach(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap != BAR_TILEMAP:
            self._set_phase("align")
            return self._act("outside_bar")
        ram = world.ram
        eve = self._find_eve(ram) or self._eve
        sx, sy = eve[0] + STAND_DX, eve[1] + STAND_DY
        pos = get_pos_from_ram(ram)
        aligned = abs(pos.y - sy) <= 4
        if aligned and pos.x <= sx + 3:
            return self._begin_talk(ram)
        stalled = self._stalled(ram)
        if stalled and aligned and pos.x < eve[0] + 40:  # pressed against her
            return self._begin_talk(ram)
        if stalled or self._phase_frames > APPROACH_TIMEOUT:
            self._empty_entries += 1
            self._set_phase("leave")
            return self._act("approach_failed")
        if not aligned:
            return self._act("approach_eve", up=pos.y > sy, down=pos.y < sy)
        return self._act("approach_eve", left=True)

    def _begin_talk(self, ram: np.ndarray) -> TaskResult:
        self._set_phase("talk")
        self._reset_talk(ram)
        return self._act("talk_ready", left=True)

    def _phase_talk(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap != BAR_TILEMAP:
            self._set_phase("align")
            return self._act("outside_bar")
        ram = world.ram
        d = current_dialogue_registers(ram)
        lock, action = d["input_lock"], read_ram_u8(ram, ADDR_PLAYER_ACTION)
        self._talk_frames += 1
        if lock == 2:
            self._seen_lock, self._quiet = True, 0
            if not self._texts or self._texts[-1] != d["text_id"]:
                self._texts.append(d["text_id"])
                self._menu_open, self._choice_frames = False, 0

        if not self._seen_lock:  # A not taken yet: press, wait, retry
            if self._talk_frames > START_TIMEOUT:
                self._start_misses += 1
                self._talk_frames = 0
                if self._start_misses >= 3:
                    self._empty_entries += 1
                    self._set_phase("leave")
                    return self._act("talk_never_started")
            return self._act("talk", a=self._talk_frames % 30 < 3, left=True)

        if self._texts and self._texts[-1] == TEXT_JUICE:
            return self._answer_juice(d, action)
        if lock == 2:
            return self._act("talk", a=self._talk_frames % 12 < 3)

        self._quiet += 1
        if self._quiet > QUIET_FRAMES and action == 0:
            return self._talk_done(ram)
        return self._act("talk_close")

    def _answer_juice(self, d: dict, action: int) -> TaskResult:
        """Wait out the text, take the arrow, then Down to "No thank you" and A."""
        self._quiet = 0
        self._choice_frames += 1
        if self._choice_frames > CHOICE_TIMEOUT:
            return self._act("choice_default", a=self._choice_frames % 12 < 3)
        if self._menu_open:
            if d["choice_result"] == 1:
                return self._act("choose_no_thank_you", a=True)
            return self._act("choose_down", down=self._choice_frames % 8 < 4)
        if action == ACTION_ARROW:
            self._menu_open = True
            return self._act("choose_open", a=True)
        return self._act("choose_wait")

    def _talk_done(self, ram: np.ndarray) -> TaskResult:
        hearts = self._hearts(ram)
        gain = hearts - self._talk_hearts0
        trapped = any(t in TRAP_TEXTS for t in self._texts)
        texts = ",".join(f"{t:#06x}" for t in self._texts)
        self._talks_this_entry += 1
        if gain > 0 and trapped:
            print(f"[EVE] entry={self._entries} +{gain} hearts={hearts}/{self._target} texts={texts}", flush=True)
            self._set_phase("leave")
            return self._act("heart_gain")
        if gain > 0:  # winter +2: the script is not parked, keep talking
            self._reset_talk(ram)
            return self._act("talk_again")
        if self._talks_this_entry >= self.max_talks_per_entry:
            if hearts == self._entry_start_hearts:
                self._empty_entries += 1
            self._set_phase("leave")
            return self._act("entry_exhausted")
        self._reset_talk(ram)
        return self._act("talk_next")

    # ── leaving ──────────────────────────────────────────────────────
    def _phase_leave(self, world: WorldState, tilemap: int, hearts: int) -> TaskResult:
        if tilemap != BAR_TILEMAP:
            if self._final is not None:
                return self._final
            self._set_phase("outside")
            return self._act("outside_bar")
        if self._phase_frames > LEAVE_TIMEOUT:
            return self._finish(TaskStatus.FAILURE, "eve could not leave the bar")
        if current_dialogue_registers(world.ram)["input_lock"] != 1:
            return self._act("leave_dialog", a=self._phase_frames % 12 < 3)
        pos = get_pos_from_ram(world.ram)
        if pos.x < 138:
            return self._act("leave", right=True, b=True)
        if pos.y < 456 and abs(pos.x - 144) > 6:
            return self._act("leave", right=pos.x < 144, left=pos.x > 144, b=True)
        return self._act("leave", down=True, b=True)


__all__ = [
    "BAR_TILEMAP",
    "EveTalkLoopTask",
    "ROMANCE_HEART_THRESHOLDS",
    "bar_talk_outlook",
    "romance_points_for_hearts",
]
