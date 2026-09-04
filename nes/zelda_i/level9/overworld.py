"""Level 9 (Death Mountain) overworld anchors and entry capability gates.

Gated by full Triforce (``ADDR_TRIFORCE == 0xFF``) for interior progress.
Bomb-rock OW screen can be mapped earlier.  Spectacle Rock and the entrance
room are live; the natural interior route remains future work.

See ``docs/LEVEL9_ROUTE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level9.ganon import (
    ADDR_GANON_OBJ_PHASE_BASE,
    GANON_BROWN_STATE,
    OBJ_GANON,
    ROOM_GANON,
    credits_rolling,
    final_ending_screen,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_MAGIC_KEY,
    ADDR_RING,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

from zelda_i.anchors import FULL_TRIFORCE, SCREEN_LEVEL9_ROCK_HYP

SOURCE_HYPOTHESIS = True
SCREEN_LEVEL9_POTION_NEAR_HYP = 0x04  # one left (source)
LEVEL9 = 9
ROOM_LEVEL9_ENTRY = 0x76
B_ITEM_BOMBS = 1
# Source / Data Crystal style values (confirm live).
RING_RED_PLANNED = 2
ARROWS_SILVER_PLANNED = 2

LEVEL9_ROCK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT"),
    # Fixture-live 2026-09-03: the unaligned north push stalls at
    # 0x78 (16,109).  The north mouth is the established x=48 column.
    ScreenHop(0x68, "UP", align_x=48),
    # Fixture-live 2026-09-03 (rr-sz8.5): unaligned UP stalls; a
    # column sweep from Level9OverworldReconFixture found the 0x68
    # north mouth at x=48 (reproduced twice, independent runs).
    ScreenHop(0x58, "UP", align_x=48),
    # Fixture-live 2026-09-03 (rr-sz8.5): 0x58 north mouth at x=112.
    ScreenHop(0x48, "UP", align_x=112),
    # Fixture-live 2026-09-03 (rr-sz8.5): 0x48 north mouth at x=128.
    ScreenHop(0x38, "UP", align_x=128),
    # v3 crossed the y=141 bridge but the west-edge x=48 north push halted at
    # (48,133).  Offline screenshot geometry places the north mouth in the
    # central sandy corridor; use x=120 for the next falsifiable trial.
    ScreenHop(0x28, "UP", align_x=120),
    # Fixture-live 2026-09-03 (rr-sz8.5): 0x28 west mouth at y=102.
    ScreenHop(0x27, "LEFT", align_y=102),
    # Fixture-live 2026-09-03 (rr-sz8.5): the apparent full-width wall was
    # only the east mountain pocket.  Drop to y~=133, walk west to x~=144,
    # then climb the central mouth.  The fixture controller below owns the
    # required multi-axis waypoint; align_x documents the live mouth.
    ScreenHop(0x17, "UP", align_x=144),
    # Raft crossing: approach the north edge from x~=64.
    ScreenHop(0x07, "UP", align_x=64),
    # Row-zero west corridor.  Re-align at y~=141 on both screens because
    # overworld enemies can displace Link during the walk.
    ScreenHop(0x06, "LEFT", align_y=141),
    ScreenHop(SCREEN_LEVEL9_ROCK_HYP, "LEFT", align_y=141),
)


class FixtureEntryPhase(Enum):
    """Fixture-only phases for the disclosed 0x77 -> Level 9 entry trial."""

    EAST_77 = auto()
    NORTH_78 = auto()
    NORTH_68 = auto()
    ALIGN_58_Y = auto()
    ALIGN_58_X = auto()
    NORTH_58 = auto()
    INLAND_48 = auto()
    ALIGN_48_X = auto()
    NORTH_48 = auto()
    INLAND_38 = auto()
    ALIGN_38_X = auto()
    NORTH_38 = auto()
    ALIGN_28_Y = auto()
    WEST_28 = auto()
    DROP_27 = auto()
    ALIGN_27_X = auto()
    NORTH_27 = auto()
    CLIMB_17 = auto()
    ALIGN_17_X = auto()
    NORTH_17 = auto()
    ALIGN_07_Y = auto()
    WEST_07 = auto()
    WEST_06 = auto()
    ROCK_OBSERVE = auto()
    PAUSE_OPEN = auto()
    PAUSE_OPEN_WAIT = auto()
    PAUSE_SELECT = auto()
    PAUSE_CURSOR_WAIT = auto()
    PAUSE_CLOSE_WAIT = auto()
    ROCK_TOP_Y = auto()
    ROCK_GAP_X = auto()
    ROCK_BOTTOM_Y = auto()
    ROCK_LEFT_X = auto()
    ROCK_FACE_UP = auto()
    ROCK_FIRE = auto()
    ROCK_BLAST_WAIT = auto()
    ROCK_ENTER = auto()
    DUNGEON_SETTLE = auto()
    DONE = auto()
    FAILED = auto()


_MOVE_PHASES = frozenset(
    {
        FixtureEntryPhase.EAST_77,
        FixtureEntryPhase.NORTH_78,
        FixtureEntryPhase.NORTH_68,
        FixtureEntryPhase.ALIGN_58_Y,
        FixtureEntryPhase.ALIGN_58_X,
        FixtureEntryPhase.NORTH_58,
        FixtureEntryPhase.INLAND_48,
        FixtureEntryPhase.ALIGN_48_X,
        FixtureEntryPhase.NORTH_48,
        FixtureEntryPhase.INLAND_38,
        FixtureEntryPhase.ALIGN_38_X,
        FixtureEntryPhase.NORTH_38,
        FixtureEntryPhase.ALIGN_28_Y,
        FixtureEntryPhase.WEST_28,
        FixtureEntryPhase.DROP_27,
        FixtureEntryPhase.ALIGN_27_X,
        FixtureEntryPhase.NORTH_27,
        FixtureEntryPhase.CLIMB_17,
        FixtureEntryPhase.ALIGN_17_X,
        FixtureEntryPhase.NORTH_17,
        FixtureEntryPhase.ALIGN_07_Y,
        FixtureEntryPhase.WEST_07,
        FixtureEntryPhase.WEST_06,
        FixtureEntryPhase.ROCK_TOP_Y,
        FixtureEntryPhase.ROCK_GAP_X,
        FixtureEntryPhase.ROCK_BOTTOM_Y,
        FixtureEntryPhase.ROCK_LEFT_X,
        FixtureEntryPhase.ROCK_ENTER,
    }
)

_TRANSITION_HOLD = {
    FixtureEntryPhase.EAST_77: "RIGHT",
    FixtureEntryPhase.NORTH_78: "UP",
    FixtureEntryPhase.NORTH_68: "UP",
    FixtureEntryPhase.NORTH_58: "UP",
    FixtureEntryPhase.NORTH_48: "UP",
    FixtureEntryPhase.NORTH_38: "UP",
    FixtureEntryPhase.WEST_28: "LEFT",
    FixtureEntryPhase.NORTH_27: "UP",
    FixtureEntryPhase.NORTH_17: "UP",
    FixtureEntryPhase.WEST_07: "LEFT",
    FixtureEntryPhase.WEST_06: "LEFT",
    FixtureEntryPhase.ROCK_ENTER: "UP",
}


@dataclass
class Level9FixtureEntryController:
    """Replay the disclosed fixture path and bomb the left Spectacle Rock.

    This is deliberately not the natural ``level9-entry`` controller.  It
    starts from ``Level9OverworldReconFixture``, inherits that fixture's
    disclosed inventory pokes, uses only controller input at runtime, and
    remains ``route_eligible=false`` until the measured post-L8 leave exists.
    """

    phase: FixtureEntryPhase = FixtureEntryPhase.EAST_77
    frames: int = 0
    phase_frames: int = 0
    max_frames: int = 12_000
    max_stuck: int = 180
    success: bool = False
    failed: bool = False
    failure: str = ""
    notes: list[str] = field(default_factory=list)
    route_screens: list[int] = field(default_factory=list)
    selected_before: int | None = None
    selected_after: int | None = None
    bombs_before: int | None = None
    bombs_after: int | None = None
    cursor_moves: int = 0
    b_presses: int = 0
    rock_observe_frames: int = 0
    blast_wait_frames: int = 0
    dungeon_settle_frames: int = 0
    blocked_cell: dict[str, int | str] | None = None
    last_snap: ZeldaSnapshot | None = field(default=None, repr=False)
    _env: Any = field(default=None, init=False, repr=False)
    _start_checked: bool = field(default=False, init=False, repr=False)
    _last_pose: tuple[int, int, int] | None = field(default=None, init=False, repr=False)
    _stuck: int = field(default=0, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _selected(self) -> int:
        assert self._env is not None
        return read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM)

    def _magic_key(self) -> int:
        assert self._env is not None
        return read_u8(self._env.get_ram(), ADDR_MAGIC_KEY)

    def _set_phase(self, phase: FixtureEntryPhase, note: str = "") -> None:
        if phase is self.phase:
            return
        self.phase = phase
        self.phase_frames = 0
        self._stuck = 0
        self._last_pose = None
        if note:
            self.notes.append(note)

    def _fail(self, reason: str, snap: ZeldaSnapshot) -> FrameAction:
        self.failed = True
        self.failure = reason
        if self.phase in _MOVE_PHASES:
            self.blocked_cell = {
                "phase": self.phase.name,
                "screen": int(snap.screen),
                "x": int(snap.link_x),
                "y": int(snap.link_y),
            }
        self._set_phase(FixtureEntryPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    @staticmethod
    def _move(direction: str, reason: str) -> FrameAction:
        return FrameAction(nes_action(direction), reason)

    def _axis(
        self,
        snap: ZeldaSnapshot,
        *,
        axis: str,
        target: int,
        tolerance: int,
        reason: str,
    ) -> FrameAction | None:
        value = snap.link_x if axis == "x" else snap.link_y
        if abs(int(value) - target) <= tolerance:
            return None
        if axis == "x":
            direction = "LEFT" if value > target else "RIGHT"
        else:
            direction = "UP" if value > target else "DOWN"
        return self._move(direction, reason)

    def _check_start(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self._start_checked:
            return None
        if self._env is None:
            return self._fail("fixture_entry_env_not_bound", snap)
        if not (
            snap.level == 0
            and snap.mode == PLAY_MODE
            and snap.screen == 0x77
            and snap.triforce == FULL_TRIFORCE
            and snap.bombs > 0
            and self._magic_key() == 1
        ):
            return self._fail("fixture_entry_start_contract_mismatch", snap)
        self._start_checked = True
        self.bombs_before = int(snap.bombs)
        self.selected_before = self._selected()
        self.route_screens.append(int(snap.screen))
        self.notes.append("disclosed_fixture_start_accepted")
        return None

    def _sync_settled_screen(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Advance only on settled RAM; transition screen bytes can oscillate."""
        if snap.mode != PLAY_MODE or snap.transitioning:
            return None
        expected: dict[FixtureEntryPhase, tuple[int, FixtureEntryPhase, str]] = {
            FixtureEntryPhase.EAST_77: (0x78, FixtureEntryPhase.NORTH_78, "settled_0x78"),
            FixtureEntryPhase.NORTH_78: (0x68, FixtureEntryPhase.NORTH_68, "settled_0x68"),
            FixtureEntryPhase.NORTH_68: (0x58, FixtureEntryPhase.ALIGN_58_Y, "settled_0x58"),
            FixtureEntryPhase.NORTH_58: (0x48, FixtureEntryPhase.INLAND_48, "settled_0x48"),
            FixtureEntryPhase.NORTH_48: (0x38, FixtureEntryPhase.INLAND_38, "settled_0x38"),
            FixtureEntryPhase.NORTH_38: (0x28, FixtureEntryPhase.ALIGN_28_Y, "settled_0x28"),
            FixtureEntryPhase.WEST_28: (0x27, FixtureEntryPhase.DROP_27, "settled_0x27"),
            FixtureEntryPhase.NORTH_27: (0x17, FixtureEntryPhase.CLIMB_17, "settled_0x17"),
            FixtureEntryPhase.NORTH_17: (0x07, FixtureEntryPhase.ALIGN_07_Y, "settled_0x07"),
            FixtureEntryPhase.WEST_07: (0x06, FixtureEntryPhase.WEST_06, "settled_0x06"),
            FixtureEntryPhase.WEST_06: (0x05, FixtureEntryPhase.ROCK_OBSERVE, "settled_0x05"),
        }
        row = expected.get(self.phase)
        if row is None or int(snap.screen) != row[0]:
            return None
        self.route_screens.append(int(snap.screen))
        self._set_phase(row[1], row[2])
        return FrameAction(nes_idle_action(), row[2])

    def _track_stuck(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.phase not in _MOVE_PHASES or snap.mode != PLAY_MODE:
            self._last_pose = None
            self._stuck = 0
            return None
        pose = (int(snap.screen), int(snap.link_x), int(snap.link_y))
        self._stuck = self._stuck + 1 if pose == self._last_pose else 0
        self._last_pose = pose
        if self._stuck > self.max_stuck:
            return self._fail("fixture_route_occupancy_miss_halt", snap)
        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.last_snap = snap
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self.frames >= self.max_frames:
            return self._fail("fixture_entry_timeout", snap)
        if snap.mode == 17:
            return self._fail("link_death", snap)
        start = self._check_start(snap)
        if start is not None:
            return start

        # Stop controller input as soon as the dungeon loader owns Link.  This
        # prevents the same trial from walking north into the Old Man room.
        if snap.level == LEVEL9:
            if snap.mode == PLAY_MODE and not snap.transitioning:
                if snap.screen != ROOM_LEVEL9_ENTRY:
                    return self._fail("wrong_level9_entry_room", snap)
                if self.phase is not FixtureEntryPhase.DUNGEON_SETTLE:
                    self.route_screens.append(int(snap.screen))
                    self._set_phase(FixtureEntryPhase.DUNGEON_SETTLE, "settled_l9_0x76")
                self.dungeon_settle_frames += 1
                if self.dungeon_settle_frames >= 24:
                    expected_bombs = int(self.bombs_before or 0) - 1
                    if int(snap.bombs) != expected_bombs:
                        return self._fail("bomb_delta_not_exactly_one", snap)
                    if self._selected() != B_ITEM_BOMBS:
                        return self._fail("bomb_selection_not_preserved", snap)
                    self.bombs_after = int(snap.bombs)
                    self.selected_after = self._selected()
                    self.success = True
                    self._set_phase(FixtureEntryPhase.DONE, "fixture_entry_0x76")
                    return FrameAction(nes_idle_action(), "done")
                return FrameAction(nes_idle_action(), "dungeon_0x76_settle")
            return FrameAction(nes_idle_action(), "dungeon_loader_wait")
        if snap.level != 0 and snap.mode == PLAY_MODE:
            return self._fail("unexpected_dungeon_level", snap)

        synced = self._sync_settled_screen(snap)
        if synced is not None:
            return synced

        if snap.transitioning:
            hold = _TRANSITION_HOLD.get(self.phase)
            if hold is None:
                return FrameAction(nes_idle_action(), "transition_wait")
            return self._move(hold, f"{self.phase.name.lower()}_transition")

        # Pause selection deliberately reads the game's cursor result; it
        # never assigns ADDR_SELECTED_ITEM.
        if self.phase is FixtureEntryPhase.ROCK_OBSERVE:
            self.rock_observe_frames += 1
            if self.rock_observe_frames < 30:
                return FrameAction(nes_idle_action(), "spectacle_rock_screenshot_hold")
            self._set_phase(FixtureEntryPhase.PAUSE_OPEN, "rock_screenshot_observed")
        if self.phase is FixtureEntryPhase.PAUSE_OPEN:
            if self._selected() == B_ITEM_BOMBS:
                self.selected_after = B_ITEM_BOMBS
                self._set_phase(FixtureEntryPhase.ROCK_TOP_Y, "bombs_already_selected")
            else:
                self._set_phase(FixtureEntryPhase.PAUSE_OPEN_WAIT, "pause_open")
                return FrameAction(nes_action("START"), "pause_open")
        if self.phase is FixtureEntryPhase.PAUSE_OPEN_WAIT:
            if self.phase_frames < 40:
                return FrameAction(nes_idle_action(), "pause_open_settle")
            self._set_phase(FixtureEntryPhase.PAUSE_SELECT)
        if self.phase is FixtureEntryPhase.PAUSE_SELECT:
            if self._selected() == B_ITEM_BOMBS:
                self.selected_after = B_ITEM_BOMBS
                self._set_phase(FixtureEntryPhase.PAUSE_CLOSE_WAIT, "pause_close")
                return FrameAction(nes_action("START"), "pause_close")
            if self.cursor_moves >= 8:
                return self._fail("bomb_cursor_not_found", snap)
            self.cursor_moves += 1
            self._set_phase(FixtureEntryPhase.PAUSE_CURSOR_WAIT)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")
        if self.phase is FixtureEntryPhase.PAUSE_CURSOR_WAIT:
            if self.phase_frames < 8:
                return FrameAction(nes_idle_action(), "pause_cursor_settle")
            self._set_phase(FixtureEntryPhase.PAUSE_SELECT)
            return FrameAction(nes_idle_action(), "pause_cursor_observe")
        if self.phase is FixtureEntryPhase.PAUSE_CLOSE_WAIT:
            if self.phase_frames < 40:
                return FrameAction(nes_idle_action(), "pause_resume")
            if self._selected() != B_ITEM_BOMBS:
                return self._fail("bomb_selection_lost", snap)
            self._set_phase(FixtureEntryPhase.ROCK_TOP_Y, "bombs_selected_by_pause")

        stuck = self._track_stuck(snap)
        if stuck is not None:
            return stuck

        if self.phase is FixtureEntryPhase.EAST_77:
            action = self._axis(snap, axis="y", target=140, tolerance=4, reason="start_align_y140")
            return action or self._move("RIGHT", "start_east_0x78")
        if self.phase is FixtureEntryPhase.NORTH_78:
            action = self._axis(snap, axis="x", target=48, tolerance=4, reason="screen78_align_x48")
            return action or self._move("UP", "screen78_north_0x68")
        if self.phase is FixtureEntryPhase.NORTH_68:
            action = self._axis(snap, axis="x", target=48, tolerance=4, reason="screen68_align_x48")
            return action or self._move("UP", "screen68_north_0x58")
        if self.phase is FixtureEntryPhase.ALIGN_58_Y:
            action = self._axis(snap, axis="y", target=157, tolerance=4, reason="screen58_bush_y157")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ALIGN_58_X)
        if self.phase is FixtureEntryPhase.ALIGN_58_X:
            # The first full replay accepted x=106 under a loose tolerance and
            # then halted against the bush at (104,149).  The live north mouth
            # is the exact x~=112 lane; do not re-probe the blocked loose cell.
            action = self._axis(snap, axis="x", target=112, tolerance=1, reason="screen58_bush_x112")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.NORTH_58, "screen58_north_lane")
        if self.phase is FixtureEntryPhase.NORTH_58:
            return self._move("UP", "screen58_north_0x48")
        if self.phase is FixtureEntryPhase.INLAND_48:
            if snap.link_y > 189:
                return self._move("UP", "screen48_inland")
            self._set_phase(FixtureEntryPhase.ALIGN_48_X)
        if self.phase is FixtureEntryPhase.ALIGN_48_X:
            action = self._axis(snap, axis="x", target=128, tolerance=4, reason="screen48_align_x128")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.NORTH_48, "screen48_north_lane")
        if self.phase is FixtureEntryPhase.NORTH_48:
            return self._move("UP", "screen48_north_0x38")
        if self.phase is FixtureEntryPhase.INLAND_38:
            # v2 tried to align west at y=189, slid to (112,205), and halted
            # against water.  The live screenshot shows the horizontal bridge
            # at y~=141; climb to it before crossing west toward x=48.
            if snap.link_y > 141:
                return self._move("UP", "screen38_bridge_y141")
            self._set_phase(FixtureEntryPhase.ALIGN_38_X)
        if self.phase is FixtureEntryPhase.ALIGN_38_X:
            if abs(snap.link_y - 141) > 4:
                return self._move(
                    "UP" if snap.link_y > 141 else "DOWN",
                    "screen38_bridge_realign_y141",
                )
            action = self._axis(snap, axis="x", target=120, tolerance=4, reason="screen38_align_x120")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.NORTH_38, "screen38_north_lane")
        if self.phase is FixtureEntryPhase.NORTH_38:
            if (
                snap.screen == 0x38
                and abs(snap.link_x - 48) <= 4
                and abs(snap.link_y - 133) <= 4
            ):
                return self._fail("known_blocked_0x38_x48_y133_replan", snap)
            return self._move("UP", "screen38_north_0x28")
        if self.phase is FixtureEntryPhase.ALIGN_28_Y:
            action = self._axis(snap, axis="y", target=102, tolerance=4, reason="screen28_align_y102")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.WEST_28, "screen28_west_lane")
        if self.phase is FixtureEntryPhase.WEST_28:
            return self._move("LEFT", "screen28_west_0x27")
        if self.phase is FixtureEntryPhase.DROP_27:
            action = self._axis(snap, axis="y", target=133, tolerance=4, reason="screen27_drop_below_mountain")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ALIGN_27_X)
        if self.phase is FixtureEntryPhase.ALIGN_27_X:
            action = self._axis(snap, axis="x", target=144, tolerance=4, reason="screen27_central_mouth_x144")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.NORTH_27, "screen27_central_mouth")
        if self.phase is FixtureEntryPhase.NORTH_27:
            return self._move("UP", "screen27_north_0x17")
        if self.phase is FixtureEntryPhase.CLIMB_17:
            action = self._axis(snap, axis="y", target=133, tolerance=4, reason="screen17_climb_y133")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ALIGN_17_X)
        if self.phase is FixtureEntryPhase.ALIGN_17_X:
            action = self._axis(snap, axis="x", target=64, tolerance=4, reason="screen17_raft_x64")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.NORTH_17, "screen17_raft_lane")
        if self.phase is FixtureEntryPhase.NORTH_17:
            return self._move("UP", "screen17_raft_north_0x07")
        if self.phase is FixtureEntryPhase.ALIGN_07_Y:
            action = self._axis(snap, axis="y", target=141, tolerance=4, reason="screen07_west_y141")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.WEST_07, "screen07_west_lane")
        if self.phase is FixtureEntryPhase.WEST_07:
            return self._move("LEFT", "screen07_west_0x06")
        if self.phase is FixtureEntryPhase.WEST_06:
            action = self._axis(snap, axis="y", target=141, tolerance=4, reason="screen06_realign_y141")
            return action or self._move("LEFT", "screen06_west_0x05")
        if self.phase is FixtureEntryPhase.ROCK_TOP_Y:
            action = self._axis(snap, axis="y", target=93, tolerance=4, reason="rock_east_column_to_top")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ROCK_GAP_X)
        if self.phase is FixtureEntryPhase.ROCK_GAP_X:
            action = self._axis(snap, axis="x", target=120, tolerance=4, reason="rock_top_to_center_gap")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ROCK_BOTTOM_Y)
        if self.phase is FixtureEntryPhase.ROCK_BOTTOM_Y:
            action = self._axis(snap, axis="y", target=173, tolerance=4, reason="rock_center_gap_to_south")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ROCK_LEFT_X)
        if self.phase is FixtureEntryPhase.ROCK_LEFT_X:
            action = self._axis(snap, axis="x", target=72, tolerance=4, reason="left_rock_bomb_stand")
            if action is not None:
                return action
            self._set_phase(FixtureEntryPhase.ROCK_FACE_UP)
        if self.phase is FixtureEntryPhase.ROCK_FACE_UP:
            self._set_phase(FixtureEntryPhase.ROCK_FIRE, "left_rock_faced_up")
            return self._move("UP", "left_rock_face_up")
        if self.phase is FixtureEntryPhase.ROCK_FIRE:
            if snap.bombs != self.bombs_before:
                return self._fail("bomb_count_changed_before_fire", snap)
            self.b_presses += 1
            self._set_phase(FixtureEntryPhase.ROCK_BLAST_WAIT, "one_bomb_below_left_rock")
            return FrameAction(nes_action("B"), "left_spectacle_rock_bomb")
        if self.phase is FixtureEntryPhase.ROCK_BLAST_WAIT:
            self.blast_wait_frames += 1
            if snap.bombs != self.bombs_before:
                self.bombs_after = int(snap.bombs)
            if self.blast_wait_frames < 180:
                return FrameAction(nes_idle_action(), "left_rock_blast_wait")
            if self.b_presses != 1 or self.bombs_after != int(self.bombs_before or 0) - 1:
                return self._fail("bomb_was_not_consumed_exactly_once", snap)
            self._set_phase(FixtureEntryPhase.ROCK_ENTER, "left_rock_blast_complete")
        if self.phase is FixtureEntryPhase.ROCK_ENTER:
            if self.phase_frames > 500:
                return self._fail("left_rock_mouth_did_not_enter", snap)
            action = self._axis(snap, axis="x", target=72, tolerance=4, reason="left_rock_mouth_realign")
            return action or self._move("UP", "enter_left_spectacle_rock")
        return self._fail("unknown_fixture_entry_phase", snap)

    def report(self) -> dict[str, Any]:
        leftover: dict[str, int | list[int]] = {}
        if self.last_snap is not None:
            leftover = {
                "level": int(self.last_snap.level),
                "room": int(self.last_snap.screen),
                "screen": int(self.last_snap.screen),
                "mode": int(self.last_snap.mode),
                "x": int(self.last_snap.link_x),
                "y": int(self.last_snap.link_y),
                "xy": [int(self.last_snap.link_x), int(self.last_snap.link_y)],
                "keys": int(self.last_snap.keys),
                "bombs": int(self.last_snap.bombs),
                "health": int(self.last_snap.health),
                "triforce": int(self.last_snap.triforce),
            }
        return {
            "success": self.success,
            "failed": self.failed,
            "failure": self.failure or None,
            "phase": self.phase.name,
            "frames": self.frames,
            "fixture_only": True,
            "natural_entry": False,
            "route_eligible": False,
            "normal_pause_input": True,
            "route_screens": [f"0x{screen:02X}" for screen in self.route_screens],
            "selected_item": {
                "before": self.selected_before,
                "after": self.selected_after,
                "delta": (
                    None
                    if self.selected_before is None or self.selected_after is None
                    else self.selected_after - self.selected_before
                ),
                "cursor_moves": self.cursor_moves,
                "writes": 0,
            },
            "bombs": {
                "before": self.bombs_before,
                "after": self.bombs_after,
                "delta": (
                    None
                    if self.bombs_before is None or self.bombs_after is None
                    else self.bombs_after - self.bombs_before
                ),
                "b_presses": self.b_presses,
            },
            "blocked_cell": self.blocked_cell,
            "notes": list(self.notes),
            "leftover": leftover,
            "controller_memory_writes": 0,
            "position_writes": 0,
            "inventory_writes": 0,
            "progression_writes": 0,
            "capacity_writes": 0,
            "selected_item_writes": 0,
            "room_writes": 0,
            "door_writes": 0,
        }
def has_full_triforce(ram) -> bool:
    return read_u8(ram, ADDR_TRIFORCE) == FULL_TRIFORCE


def triforce_bits(ram) -> int:
    return int(read_u8(ram, ADDR_TRIFORCE))


def has_red_ring(ram) -> bool:
    return read_u8(ram, ADDR_RING) >= RING_RED_PLANNED


def has_silver_arrows(ram) -> bool:
    return read_u8(ram, ADDR_ARROWS) >= ARROWS_SILVER_PLANNED


def required_caps_for_entry() -> frozenset[str]:
    """Full TF for Old Man; bombs for rock (source)."""
    return frozenset({"full_triforce", "bombs"})


def required_caps_for_ganon() -> frozenset[str]:
    return frozenset({"full_triforce", "silver_arrows"})


def missing_entry_caps(ram, *, rock_only: bool = False) -> list[str]:
    """Caps missing for entry attempt.

    ``rock_only``: map bomb-rock OW without requiring full TF.
    """
    missing: list[str] = []
    if not rock_only and not has_full_triforce(ram):
        missing.append("full_triforce")
    # Bombs checked by caller snap.bombs if desired; not hard-fail here.
    return missing


def on_level9_rock_hyp(snap: ZeldaSnapshot) -> bool:
    return (
        snap.level == 0
        and snap.mode == PLAY_MODE
        and snap.screen == SCREEN_LEVEL9_ROCK_HYP
    )


def level9_dungeon_play(snap: ZeldaSnapshot) -> bool:
    return snap.level == LEVEL9 and snap.mode == PLAY_MODE


def level9_entry_stop(snap: ZeldaSnapshot) -> bool:
    return level9_dungeon_play(snap) and snap.screen == ROOM_LEVEL9_ENTRY


def level9_overworld_stop(snap: ZeldaSnapshot) -> bool:
    return on_level9_rock_hyp(snap)


def level9_ending_stop(snap: ZeldaSnapshot) -> bool:
    """True once the update loop reaches rolling credits or its final page."""
    return credits_rolling(snap) or final_ending_screen(snap)


def level9_ganon_planning_notes() -> dict[str, Any]:
    return {
        "policy": "stun Ganon (sword) until brown, then Silver Arrow on B",
        "silver_arrows_ram": hex(ADDR_ARROWS),
        "silver_arrows_value_planned": ARROWS_SILVER_PLANNED,
        "object_type_id": OBJ_GANON,
        "brown_state_ram": hex(0x00AC),
        "brown_state_initial": GANON_BROWN_STATE,
        "dying_phase_base": hex(ADDR_GANON_OBJ_PHASE_BASE),
        "live_verified": True,
    }


def planning_report() -> dict[str, Any]:
    return {
        "level": LEVEL9,
        "name": "Death Mountain",
        "status": "backward_recon_live_natural_route_pending",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "required_entry_caps": sorted(required_caps_for_entry()),
        "required_ganon_caps": sorted(required_caps_for_ganon()),
        "full_triforce": FULL_TRIFORCE,
        "ram": {
            "triforce": hex(ADDR_TRIFORCE),
            "ring": hex(ADDR_RING),
            "arrows": hex(ADDR_ARROWS),
        },
        "screens_hypothesized": {
            "bomb_rock": hex(SCREEN_LEVEL9_ROCK_HYP),
            "potion_near": hex(SCREEN_LEVEL9_POTION_NEAR_HYP),
        },
        "rock_hops_from_start": [
            {"target": hex(h.target), "dir": h.direction} for h in LEVEL9_ROCK_HOPS
        ],
        "ganon": level9_ganon_planning_notes(),
        "ending_stop": "mode=0x13, updating!=0, submode=3 credits or 4 final",
        "live": {
            "rock_screen": SCREEN_LEVEL9_ROCK_HYP,
            "entry_room": ROOM_LEVEL9_ENTRY,
            "red_ring_room": None,
            "silver_arrow_room": 0x10,
            "silver_arrow_evidence": "hypothesis",
            "ganon_room": ROOM_GANON,
        },
        "natural_entry": {
            "post_l8_leftover": "unmeasured",
            "start_based_rock_hops": "fixture-live; not the cumulative leave",
            "route_eligible": False,
        },
        "docs": "nes/zelda_i/docs/LEVEL9_ROUTE.md",
    }
