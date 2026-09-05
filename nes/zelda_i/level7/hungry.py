"""Level 7 Hungry Goriya: pause-select Bait, natural Food 1→0, UP to MAP 0x18.

Live room ``$EB=0x28``.  Equip already-owned Bait (B-slot 6) through the
pause menu — START / idle 20 / RIGHT / idle 8 / START close / idle 24, same
shape as ``dungeon.pause_select.PauseSelectController``.  Snapshot has no
``selected_item``; read ``ADDR_SELECTED_ITEM`` after ``bind_env``.  Do not
poke selected-item or Food.

Then align ``x=120``, walk UP toward the NPC (object 0x36 at (120,128) on
the 0x28 pin), tap B until Food falls 1→0, then UP out the opened north
door.  Success is dest play ``0x18`` plus that falling edge measured from
this controller's first frame — a pin that already has Food 0 never greens.

No RAM writes.  OccupancyWalker is banned.  ``route_eligible=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.hop_controller import HopController, dungeon_align_then_push
from zelda_i.level7.graph import HUNGRY_GORIYA, LEVEL7_ROOM_BY_ID
from zelda_i.level7.path import (
    DOOR_Y_TOL,
    ENTRY_SCREEN,
    NORTH_X_TOL,
    _goriya_fight,
    live_goriyas,
)
from zelda_i.dungeon.pause_select import B_SLOT_BAIT, PauseSelectController
from zelda_i.ram import (
    ADDR_FOOD,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

__all__ = [
    "DEST",
    "DOOR_X",
    "FEED_Y",
    "FOOD_B_SLOT",
    "HungryPhase",
    "LEVEL7",
    "Level7HungryGoriyaController",
    "ROOM",
    "ROOM_38",
    "Room38UpController",
    "make_level7_hungry_goriya_controller",
    "north_of_room38_ram_id",
    "room_38_up_step",
]

LEVEL7 = 7
DEATH_MODE = 17
ROOM = 0x28
DEST = 0x18
FOOD_B_SLOT = B_SLOT_BAIT
DOOR_X = 120
FEED_Y = 141
ALIGN = 4
HUNGRY_MAX_FRAMES = 2400
B_PERIOD = 16
B_WINDOW = 6
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


class HungryPhase(Enum):
    SELECT = auto()
    APPROACH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level7HungryGoriyaController:
    """Feed the Hungry Goriya in live ``0x28`` and leave north to MAP ``0x18``."""

    max_frames: int = HUNGRY_MAX_FRAMES
    phase: HungryPhase = HungryPhase.SELECT
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    initial_food: int | None = None
    last_food: int | None = None
    food_consumed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(want=FOOD_B_SLOT, name="bait")

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: HungryPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(HungryPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _approach(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        if abs(x - DOOR_X) > ALIGN:
            btn = "LEFT" if x > DOOR_X else "RIGHT"
            return FrameAction(nes_action(btn), "hungry_align_x")
        if not self.food_consumed:
            if y > FEED_Y:
                return FrameAction(nes_action("UP"), "hungry_approach")
            if self.phase_frames % B_PERIOD < B_WINDOW:
                return FrameAction(nes_action("B"), "hungry_feed")
            return FrameAction(nes_action("UP"), "hungry_feed_up")
        return FrameAction(nes_action("UP"), "hungry_exit_north")

    def _run_select(self, snap: ZeldaSnapshot) -> FrameAction:
        action = self._select.drive(snap)
        for note in self._select.notes:
            self._note(note)
        if self._select.failed:
            return self._fail(self._select.fail_reason)
        if action is None:
            if (
                snap.level != LEVEL7
                or snap.mode != PLAY_MODE
                or int(snap.screen) != ROOM
            ):
                return self._fail("pause_close_contract_mismatch")
            self._set_phase(HungryPhase.APPROACH, "bait_ready")
            return self._approach(snap)
        return action

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("hungry_goriya_env_not_bound")
        ram = self._env.get_ram()
        food = int(read_u8(ram, ADDR_FOOD))
        if self.initial_food is None:
            self.initial_food = food
            self._note(f"food_in_{food}")
            if food < 1:
                return self._fail("hungry_goriya_requires_food")
        if self.last_food is not None and self.last_food >= 1 and food == 0:
            self.food_consumed = True
            self._note("food_consumed_naturally")
        self.last_food = food
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        if self.frames > self.max_frames:
            return self._fail("budget_exhausted")
        dest_settled = (
            snap.level == LEVEL7
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and int(snap.screen) == DEST
        )
        if dest_settled and self.food_consumed:
            self.success = True
            self._set_phase(HungryPhase.DONE, "fed_and_map_0x18")
            return FrameAction(nes_idle_action(), "done")
        if snap.transitioning or snap.mode in WAIT_MODES:
            if self.food_consumed:
                return FrameAction(nes_action("UP"), "hungry_scroll")
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.level != LEVEL7:
                return self._fail(f"left_level_{snap.level}")
            if int(snap.screen) != ROOM:
                if int(snap.screen) == DEST:
                    return self._fail("dest_without_food_consume")
                return self._fail(
                    f"left_room_L{snap.level}_0x{snap.screen:02x}"
                )
        if self.phase is HungryPhase.SELECT:
            return self._run_select(snap)
        return self._approach(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_hungry_goriya",
            "room": f"0x{ROOM:02X}",
            "dest": f"0x{DEST:02X}",
            "food_consumed": self.food_consumed,
            "initial_food": self.initial_food,
            "phase": self.phase.name,
            "cursor_moves": self.cursor_moves,
            "normal_pause_input": True,
            "writes": 0,
            "route_eligible": False,
            "evidence": "fixture-live",
            "notes": list(self.notes),
        }


def make_level7_hungry_goriya_controller() -> Level7HungryGoriyaController:
    """Fresh 0x28 feed-and-leave controller (never share instances)."""
    return Level7HungryGoriyaController()


# 0x38 (GORIYA_PRE_HUNGRY): entry (208,141) E mouth, diamond floor. The y=149
# interior row blocks UP at x=120/104/88/200. Rise the east mouth pocket
# x=208 to y=93, cross to x=120, KEY-UP (keys 4->3) to live $EB=0x28.
# 2/2 (38_up_v6/v7). Compass room_item 0x0f stays uncollected.
ROOM_38 = 0x38
ROOM_38_EAST_POCKET_X = 208
ROOM_38_NORTH_X = 120
ROOM_38_TOP_BAND_Y = 93
ROOM38_UP_MAX_FRAMES = 8000


def north_of_room38_ram_id() -> int | None:
    """Live ``$EB`` of the room north of ``0x38`` (HUNGRY_GORIYA), or None."""
    return LEVEL7_ROOM_BY_ID[HUNGRY_GORIYA].ram_id


def room_38_up_step(
    snap: ZeldaSnapshot,
    *,
    dest: int | None = None,
    saw_goriya: bool = False,
    frames: int = 0,
) -> FrameAction:
    """One frame of 0x38 kill-clear → east-pocket rise → KEY north door.

    Interior ``y=149`` is a diamond wall (UP blocked at x=120/104/88/200).
    Recollect the east mouth pocket ``x=208``, rise to ``y=93``, cross to
    ``x=120``, push UP. The key consume is natural.
    """
    if snap.level != LEVEL7:
        return FrameAction(nes_idle_action(), "wait_level7")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "up38_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if dest is not None and snap.screen == dest:
        return FrameAction(nes_idle_action(), "up38_arrived")
    if snap.screen != ROOM_38:
        return FrameAction(nes_idle_action(), f"unexpected_room_0x{snap.screen:02x}")

    live = live_goriyas(snap)
    if live:
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "goriya_missing")
        return _goriya_fight(snap, target, frames=frames)
    if not saw_goriya:
        return FrameAction(nes_idle_action(), "spawn_wait")

    x, y = int(snap.link_x), int(snap.link_y)
    if y > ROOM_38_TOP_BAND_Y + 12 and x < ROOM_38_EAST_POCKET_X - NORTH_X_TOL:
        return FrameAction(nes_action("RIGHT"), "up38_pocket")
    if y > ROOM_38_TOP_BAND_Y + DOOR_Y_TOL:
        return FrameAction(nes_action("UP"), "up38_rise")
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=ROOM_38_NORTH_X,
        x_tol=NORTH_X_TOL,
        reason="up38",
    )


@dataclass(kw_only=True)
class Room38UpController(HopController):
    """0x38 (GORIYA_PRE_HUNGRY) east mouth: kill-clear, east-pocket rise,
    KEY-UP to live dest 0x28 (HUNGRY_GORIYA).  2/2 (38_up_v6/v7).
    Recon-wired only.
    """

    spec_id: str = "level7_room38_up"
    max_frames: int = ROOM38_UP_MAX_FRAMES
    require_level: int = LEVEL7
    done_reason: str = "left_0x38_north"
    dest: int | None = field(default_factory=north_of_room38_ram_id)
    saw_goriya: bool = False

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if (
            snap.level != LEVEL7
            or snap.mode != PLAY_MODE
            or snap.transitioning
            or snap.screen in {ENTRY_SCREEN, 0x49, 0x39, ROOM_38}
        ):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"arrived_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_saw={int(self.saw_goriya)}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        return FrameAction(nes_action("UP"), "up38_scroll")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if live_goriyas(snap):
            self.saw_goriya = True
        action = room_38_up_step(
            snap, dest=self.dest, saw_goriya=self.saw_goriya, frames=self.frames
        )
        if action.reason.startswith("unexpected_room"):
            if snap.screen == 0x39:
                return self.mark_fail("east_backtrack")
            return self.mark_fail(action.reason)
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "stage_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "door": "UP",
        }
