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
from zelda_i.combat import BOMB_DROP_OBJECT_TYPE, BOMB_DROP_STATES
from zelda_i.dungeon.hop_controller import ow_edge_band_step, room_step
from zelda_i.dungeon.pause_select import PauseSelectController
from zelda_i.level9.dungeon import (
    BOMBS_NOT_NATURAL,
    MISSING_SPECTACLE_BOMB,
    PostLevel8Handoff,
    TRIFORCE_NOT_FULL,
    UNMEASURED_POST_L8_HANDOFF,
)
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops
from zelda_i.overworld.common import scoop_floor_drop
from zelda_i.overworld.path import OverworldPathController, PathNavPhase
from zelda_i.ram import (
    ADDR_MAGIC_KEY,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

from zelda_i.anchors import FULL_TRIFORCE, SCREEN_LEVEL9_ROCK_HYP

SOURCE_HYPOTHESIS, SCREEN_LEVEL9_POTION_NEAR_HYP, LEVEL9 = True, 0x04, 9
ROOM_LEVEL9_ENTRY, B_ITEM_BOMBS = 0x76, 1
RING_RED_PLANNED, ARROWS_SILVER_PLANNED = 2, 2

LEVEL9_ROCK_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x78, "RIGHT"),
    ScreenHop(0x68, "UP", align_x=48),  # 0x78 unaligned north stalls (16,109)
    ScreenHop(0x58, "UP", align_x=48),
    ScreenHop(0x48, "UP", align_x=112),
    ScreenHop(0x38, "UP", align_x=128),
    ScreenHop(0x28, "UP", align_x=120),  # x=48 west-edge north is blocked
    ScreenHop(0x27, "LEFT", align_y=102),
    ScreenHop(0x17, "UP", align_x=144),  # drop y~133, west to central mouth
    ScreenHop(0x07, "UP", align_x=64),
    ScreenHop(0x06, "LEFT", align_y=141),
    ScreenHop(SCREEN_LEVEL9_ROCK_HYP, "LEFT", align_y=141),
)

REVERSE_5C_MAZE_WAYPOINTS: tuple[tuple[int, int], ...] = ((192, 132), (192, 92), (16, 92))

# Reverse overworld connector from post-L8 leave (0x6D) to 0x58, joining LEVEL9_ROCK_HOPS
POST_L8_TO_LEVEL9_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x5D, "UP", align_x=48),
    ScreenHop(0x5C, "LEFT", align_y=132),
    ScreenHop(0x5B, "LEFT", align_y=92),
    ScreenHop(0x5A, "LEFT", align_y=93),
    ScreenHop(0x59, "LEFT", align_y=140),
    ScreenHop(0x58, "LEFT", align_y=155),
    ScreenHop(0x48, "UP", align_x=112),
    ScreenHop(0x38, "UP", align_x=128),
    ScreenHop(0x28, "UP", align_x=120),
    ScreenHop(0x27, "LEFT", align_y=102),
    ScreenHop(0x17, "UP", align_x=144),
    ScreenHop(0x07, "UP", align_x=64),
    ScreenHop(0x06, "LEFT", align_y=141),
    ScreenHop(SCREEN_LEVEL9_ROCK_HYP, "LEFT", align_y=141),
)
POST_L8_TO_LEVEL9_SCREENS: tuple[int, ...] = path_screens_from_hops(
    0x6D, POST_L8_TO_LEVEL9_HOPS
)

# Post-L8 route via 0x4A bomb shop (rr-ps7.5) to purchase 4 bombs for Level 9.
POST_L8_TO_BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x5D, "UP", align_x=48),
    ScreenHop(0x5C, "LEFT", align_y=132),
    ScreenHop(0x5B, "LEFT", align_y=92),
    ScreenHop(0x5A, "LEFT", align_y=93),
    ScreenHop(0x59, "LEFT", align_y=140),
    ScreenHop(0x49, "UP", align_x=112),
    ScreenHop(0x4A, "RIGHT", align_y=141),
)

POST_L8_FROM_BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = (
    ScreenHop(0x49, "LEFT", align_y=141),
    ScreenHop(0x59, "DOWN", align_x=112),
    ScreenHop(0x58, "LEFT", align_y=155),
    ScreenHop(0x48, "UP", align_x=112),
    ScreenHop(0x38, "UP", align_x=128),
    ScreenHop(0x28, "UP", align_x=120),
    ScreenHop(0x27, "LEFT", align_y=102),
    ScreenHop(0x17, "UP", align_x=144),
    ScreenHop(0x07, "UP", align_x=64),
    ScreenHop(0x06, "LEFT", align_y=141),
    ScreenHop(SCREEN_LEVEL9_ROCK_HYP, "LEFT", align_y=141),
)

POST_L8_VIA_BOMB_SHOP_HOPS: tuple[ScreenHop, ...] = (
    *POST_L8_TO_BOMB_SHOP_HOPS,
    *POST_L8_FROM_BOMB_SHOP_HOPS,
)


class FixtureEntryPhase(Enum):
    """Fixture-only phases for the disclosed 0x77 -> Level 9 entry trial."""
    EAST_77, NORTH_78, NORTH_68 = auto(), auto(), auto()
    ALIGN_58_Y, ALIGN_58_X, NORTH_58 = auto(), auto(), auto()
    INLAND_48, ALIGN_48_X, NORTH_48 = auto(), auto(), auto()
    INLAND_38, ALIGN_38_X, NORTH_38 = auto(), auto(), auto()
    ALIGN_28_Y, WEST_28, DROP_27 = auto(), auto(), auto()
    ALIGN_27_X, NORTH_27, CLIMB_17 = auto(), auto(), auto()
    ALIGN_17_X, NORTH_17, ALIGN_07_Y = auto(), auto(), auto()
    WEST_07, WEST_06 = auto(), auto()
    PAUSE_OPEN, ROCK_TOP_Y, ROCK_GAP_X = auto(), auto(), auto()
    ROCK_BOTTOM_Y, ROCK_LEFT_X, ROCK_FACE_UP = auto(), auto(), auto()
    ROCK_FIRE, ROCK_BLAST_WAIT, ROCK_ENTER = auto(), auto(), auto()
    DUNGEON_SETTLE, DONE, FAILED = auto(), auto(), auto()


_NON_MOVE = frozenset({
    FixtureEntryPhase.PAUSE_OPEN,
    FixtureEntryPhase.ROCK_FACE_UP, FixtureEntryPhase.ROCK_FIRE,
    FixtureEntryPhase.ROCK_BLAST_WAIT, FixtureEntryPhase.DUNGEON_SETTLE,
    FixtureEntryPhase.DONE, FixtureEntryPhase.FAILED,
})
_MOVE_PHASES = frozenset(p for p in FixtureEntryPhase if p not in _NON_MOVE)

_TRANSITION_HOLD = {
    FixtureEntryPhase.EAST_77: "RIGHT",
    FixtureEntryPhase.NORTH_78: "UP", FixtureEntryPhase.NORTH_68: "UP",
    FixtureEntryPhase.NORTH_58: "UP", FixtureEntryPhase.NORTH_48: "UP",
    FixtureEntryPhase.NORTH_38: "UP", FixtureEntryPhase.WEST_28: "LEFT",
    FixtureEntryPhase.NORTH_27: "UP", FixtureEntryPhase.NORTH_17: "UP",
    FixtureEntryPhase.WEST_07: "LEFT", FixtureEntryPhase.WEST_06: "LEFT",
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
    b_presses: int = 0
    blast_wait_frames: int = 0
    dungeon_settle_frames: int = 0
    blocked_cell: dict[str, int | str] | None = None
    last_snap: ZeldaSnapshot | None = field(default=None, repr=False)
    _env: Any = field(default=None, init=False, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)
    _start_checked: bool = field(default=False, init=False, repr=False)
    _last_pose: tuple[int, int, int] | None = field(default=None, init=False, repr=False)
    _stuck: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(want=B_ITEM_BOMBS, name="bombs")

    @property
    def cursor_moves(self) -> int:
        return self._select.cursor_moves

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

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
            FixtureEntryPhase.WEST_06: (0x05, FixtureEntryPhase.PAUSE_OPEN, "settled_0x05"),
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
        # never assigns ADDR_SELECTED_ITEM. RIGHT only counts when $0656
        # actually changes (shared PauseSelectController).
        if self.phase is FixtureEntryPhase.PAUSE_OPEN:
            driven = self._select.drive(snap)
            for note in self._select.notes:
                if note not in self.notes:
                    self.notes.append(note)
            if self._select.failed:
                return self._fail(
                    self._select.fail_reason or "pause_select_failed", snap
                )
            if driven is not None:
                return driven
            self.selected_after = B_ITEM_BOMBS
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
            # x=106 under a loose tol halted at the (104,149) bush. Lane is 112.
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
            # y=189 west-align slid into water at (112,205). Bridge is y~141.
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


@dataclass
class Level9PostL8OverworldController(OverworldPathController):
    """Walk from post-L8 leave (OW 0x6D) to Spectacle Rock (OW 0x05).

    Refuses without full Triforce (0xFF), natural bombs (>0), and complete
    measured post-L8 handoff. Never writes RAM.
    """

    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF
    hops: tuple[ScreenHop, ...] = POST_L8_TO_LEVEL9_HOPS
    max_frames: int = 12_000
    reverse_maze_waypoints: tuple[tuple[int, int], ...] = REVERSE_5C_MAZE_WAYPOINTS
    reverse_maze_wp_index: int = 0
    stop_screen: int = SCREEN_LEVEL9_ROCK_HYP
    bomb_goal: int = 0
    check_handoff: bool = True
    _handoff_checked: bool = field(default=False, init=False, repr=False)
    failed: bool = field(default=False, init=False, repr=False)
    blocked_reason: str = field(default="", init=False, repr=False)
    _env: Any = field(default=None, init=False, repr=False)
    # rr-mzxn follow-on: 0x58's arrival band (y~141) has an obstacle blocking
    # LEFT somewhere around x=155-224 that isn't present a few px south
    # (y>=149, confirmed clear all the way to x=112). Re-checking a y
    # threshold every frame ping-pongs once the walk is already past it (a
    # single UP tap can dip back under almost any fixed threshold), so track
    # "cleared" once instead of re-testing y forever.
    _cleared_58_south_wall: bool = field(default=False, init=False, repr=False)
    # 0x38's "realign to y=141" branch pressed DOWN whenever y<137, which
    # actively undoes an UP press that already overshot north past the
    # hazard band (observed reaching y~105 before being walked back to
    # ~134) -- same "latch, don't re-test" fix as _cleared_58_south_wall.
    _cleared_38_bridge: bool = field(default=False, init=False, repr=False)
    _cleared_27_gap: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.hops:
            self.hops = POST_L8_TO_LEVEL9_HOPS
        if self.check_handoff and not self.handoff.complete():
            self.max_frames = 1

    def bind_env(self, env: Any) -> None:
        self._env = env

    def reset(self) -> None:
        super().reset()
        self.reverse_maze_wp_index = 0
        self._handoff_checked = False
        self.failed = False
        self.blocked_reason = ""
        self._cleared_58_south_wall = False
        self._cleared_38_bridge = False
        self._cleared_27_gap = False

    def _fail_now(self, reason: str) -> FrameAction:
        self.failed = True
        self.blocked_reason = reason
        self._set_phase(PathNavPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _at_stop(self, snap: ZeldaSnapshot) -> bool:
        return (
            self.hop_index >= len(self.hops)
            and snap.level == 0
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == self.stop_screen
        )

    def _after_hops(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._at_stop(snap):
            note = (
                "level9_spectacle_rock_reached"
                if self.stop_screen == SCREEN_LEVEL9_ROCK_HYP
                else "level9_post_l8_overworld_reached"
            )
            return self._finish(note)
        return self._fail_now(f"post_l8_path_exhausted_off_{self.stop_screen:#04x}")

    def _extra_hop_action(
        self, snap: ZeldaSnapshot, hop: ScreenHop
    ) -> FrameAction | None:
        if snap.screen == 0x5D and self.bomb_goal and snap.bombs < self.bomb_goal:
            # Blue Moblins on the first leg can drop a natural four-bomb
            # pack. Bank it before the west scroll; the 0x4A shop then skips
            # only when the unchanged eight-bomb target is already met.
            pickup = scoop_floor_drop(
                snap,
                types=(BOMB_DROP_OBJECT_TYPE,),
                states=BOMB_DROP_STATES,
                travel_dir=hop.direction,
                radius=48,
                reason="5d_scoop_bomb",
                want=True,
            )
            if pickup is not None:
                return pickup
        if snap.screen == 0x6D and hop.target == 0x5D:
            if abs(snap.link_x - 48) > 4:
                btn = "LEFT" if snap.link_x > 48 else "RIGHT"
                return self._swing(btn, "6d_walk_left_x48")
            return self._swing("UP", "6d_north_0x5d")

        if snap.screen == 0x5D and hop.target == 0x5C:
            if abs(snap.link_y - 132) > 4:
                btn = "UP" if snap.link_y > 132 else "DOWN"
                return self._swing(btn, "5d_align_y132")
            return self._swing("LEFT", "5d_west_0x5c")

        if snap.screen == 0x5C and hop.target == 0x5B:
            if self.reverse_maze_wp_index == 0:
                if snap.link_x <= 192 and abs(snap.link_y - 132) <= 6:
                    self.reverse_maze_wp_index = 1
                else:
                    return self._swing("LEFT", "5c_reverse_maze_wp0")
            if self.reverse_maze_wp_index == 1:
                if snap.link_y <= 92 and abs(snap.link_x - 192) <= 6:
                    self.reverse_maze_wp_index = 2
                else:
                    if abs(snap.link_x - 192) > 2:
                        btn = "RIGHT" if snap.link_x < 192 else "LEFT"
                        return self._swing(btn, "5c_reverse_maze_align_x192")
                    return self._swing("UP", "5c_reverse_maze_wp1")
            if self.reverse_maze_wp_index == 2:
                if snap.link_x <= 16:
                    self.reverse_maze_wp_index = 3
                else:
                    if abs(snap.link_y - 92) > 2:
                        btn = "DOWN" if snap.link_y < 92 else "UP"
                        return self._swing(btn, "5c_reverse_maze_align_y92")
                    return self._swing("LEFT", "5c_reverse_maze_wp2")
            return self._swing("LEFT", "5c_reverse_maze_exit")

        if snap.screen == 0x5B and hop.target == 0x5A:
            if abs(snap.link_y - 93) > 4:
                btn = "DOWN" if snap.link_y < 93 else "UP"
                return self._swing(btn, "5b_highway_realign_y93")
            return self._swing("LEFT", "5b_highway_left_0x5a")

        if snap.screen == 0x5A and hop.target == 0x59:
            if snap.link_y < 140:
                return self._swing("DOWN", "5a_step_down_y140")
            if snap.link_y > 145:
                # A hit can knock Link below the west passage. At (112,205),
                # blindly pressing LEFT holds against the south wall forever.
                step = ow_edge_band_step(None, snap, "LEFT", 137, 145)
                if step is not None:
                    return self._swing(step, "5a_recover_west_lattice")
            return self._swing("LEFT", "5a_walk_left_0x59")

        if snap.screen == 0x59 and hop.target == 0x58:
            # 0x59 is a cross: the x 112-128 corridor from 0x49 (y 61-221)
            # and the y 117-157 band out west. The hand walk below overshot
            # down the corridor and held LEFT against its wall at
            # (112,165..205) for 6500 frames under Zora fire (9 hearts, real
            # C10 resume CL9A). The lattice to the band's west edge covers
            # the 0x5A arrival (y=141) and the 0x49 one alike.
            step = ow_edge_band_step(None, snap, "LEFT", 137, 145)
            if step is not None:
                return self._swing(step, "59_west_lattice")
            # The direct walk arrives from 0x5A at y=141. The bomb-shop
            # return arrives from 0x49 at (112,61), above the west passage;
            # first descend through the center opening. Do not descend to
            # y=155 here: the east arrival has a wall immediately below it.
            if snap.link_y < 137:
                return self._swing("DOWN", "59_descend_to_west_passage")
            return self._swing("LEFT", "59_walk_left_0x58")

        if snap.screen == 0x59 and hop.target == 0x49:
            if abs(snap.link_x - 112) > 4:
                btn = "LEFT" if snap.link_x > 112 else "RIGHT"
                return self._swing(btn, "59_align_x112")
            return self._swing("UP", "59_north_0x49")

        if snap.screen == 0x49 and hop.target == 0x4A:
            if abs(snap.link_y - 141) > 4:
                btn = "UP" if snap.link_y > 141 else "DOWN"
                return self._swing(btn, "49_align_y141")
            return self._swing("RIGHT", "49_east_0x4a")

        if snap.screen == 0x4A and hop.target == 0x49:
            if abs(snap.link_y - 141) > 4:
                btn = "UP" if snap.link_y > 141 else "DOWN"
                return self._swing(btn, "4a_align_y141")
            return self._swing("LEFT", "4a_west_0x49")

        if snap.screen == 0x49 and hop.target == 0x59:
            if abs(snap.link_x - 112) > 4:
                btn = "LEFT" if snap.link_x > 112 else "RIGHT"
                return self._swing(btn, "49_align_x112")
            return self._swing("DOWN", "49_south_0x59")

        if snap.screen == 0x58 and hop.target == 0x48:
            # Arrival band y~141 has an obstacle blocking LEFT somewhere in
            # x=155-224 (confirmed empirically); y>=149 is clear all the way
            # to x=112, and x=112 is then a clear vertical corridor up to
            # 0x48. Descend once (latched, not re-checked -- see
            # ``_cleared_58_south_wall``) before the leftward x-align.
            if not self._cleared_58_south_wall:
                if snap.link_y < 149:
                    return self._swing("DOWN", "58_step_down_clear_wall")
                self._cleared_58_south_wall = True
            if snap.link_x > 112 + 4:
                return self._swing("LEFT", "58_walk_left_x112")
            if snap.link_x < 112 - 4:
                return self._swing("RIGHT", "58_realign_x112")
            return self._swing("UP", "58_north_0x48")

        if snap.screen == 0x48 and hop.target == 0x38:
            if snap.link_y > 189:
                return self._swing("UP", "48_inland")
            if abs(snap.link_x - 128) > 4:
                btn = "LEFT" if snap.link_x > 128 else "RIGHT"
                return self._swing(btn, "48_align_x128")
            return self._swing("UP", "48_north_0x38")

        if snap.screen == 0x38 and hop.target == 0x28:
            if abs(snap.link_x - 48) <= 4 and abs(snap.link_y - 133) <= 4:
                return self._fail_now("known_blocked_0x38_x48_y133_replan")
            # ROM lattice first, as on 0x17: the x=112..143 cut is the only
            # way north, and once the bridge latch set, a knock off x=120
            # pressed UP into rock for 10387f (natural_credits_47r).
            step = ow_edge_band_step(None, snap, "UP", 112, 136)
            if step is not None:
                return self._swing(step, "38_lattice")
            if not self._cleared_38_bridge:
                if abs(snap.link_x - 120) > 4:
                    btn = "LEFT" if snap.link_x > 120 else "RIGHT"
                    return self._swing(btn, "38_align_x120")
                if snap.link_y > 141:
                    return self._swing("UP", "38_bridge_y141")
                # First time we read y<=141 (however far below -- a real UP
                # press here routinely overshoots to ~105, not a clean stop
                # at 141), latch and never look back. The old "realign to
                # exactly 141" branch below this point re-pressed DOWN on
                # every future frame with y<137, which actively walks Link
                # back south and erases the progress it just made -- that
                # was the real bug, not just a re-checked threshold (see
                # rr-sz8.5 probe notes).
                self._cleared_38_bridge = True
            return self._swing("UP", "38_north_0x28")

        if snap.screen == 0x27 and hop.target == 0x17:
            # ROM lattice first: a PolicyGuard detour left Link east of the
            # x=144 mouth after the latch below, and UP pressed the mountain
            # from x 176-224 until the 12000f cap (C11Evalo6).
            step = ow_edge_band_step(None, snap, "UP", 140, 148)
            if step is not None:
                return self._swing(step, "27_lattice")
            # Same latch bug as 0x38 (see _cleared_38_bridge): the final
            # "UP" commit below routinely overshoots y<133, and this
            # DOWN-pressing check re-fires on every later frame with
            # y<133, walking Link back and undoing the northward progress
            # it just made. Latch once, never re-test.
            if not self._cleared_27_gap:
                if snap.link_y < 133:
                    return self._swing("DOWN", "27_drop_below_mountain")
                if abs(snap.link_x - 144) > 4:
                    btn = "LEFT" if snap.link_x > 144 else "RIGHT"
                    return self._swing(btn, "27_central_mouth_x144")
                self._cleared_27_gap = True
            return self._swing("UP", "27_north_0x17")

        if snap.screen == 0x17 and hop.target == 0x07:
            # ROM lattice first: the x=64 dock column is the only way north.
            # The cardinals below spent 12000f at (128,141) on the power-on
            # gathered spine.
            step = ow_edge_band_step(None, snap, "UP", 60, 68)
            if step is not None:
                return self._swing(step, "17_lattice")
            # East of the x~96 water strip the lattice has no route (the
            # stepladder crosses it). Cross on the y=133 row itself: drifting
            # to y~128 pressed LEFT into the mountain lip for 12000f.
            if abs(snap.link_y - 133) > 2:
                btn = "UP" if snap.link_y > 133 else "DOWN"
                return self._swing(btn, "17_climb_y133")
            if abs(snap.link_x - 64) > 4:
                btn = "LEFT" if snap.link_x > 64 else "RIGHT"
                return self._swing(btn, "17_raft_x64")
            return self._swing("UP", "17_raft_north_0x07")

        if snap.screen == 0x07 and hop.target == 0x06:
            step = ow_edge_band_step(None, snap, "LEFT", 137, 145)
            if step is not None:
                return self._swing(step, "07_lattice")
            if abs(snap.link_y - 141) > 4:
                btn = "UP" if snap.link_y > 141 else "DOWN"
                return self._swing(btn, "07_west_y141")
            return self._swing("LEFT", "07_west_0x06")

        if snap.screen == 0x06 and hop.target == SCREEN_LEVEL9_ROCK_HYP:
            step = ow_edge_band_step(None, snap, "LEFT", 137, 145)
            if step is not None:
                return self._swing(step, "06_lattice")
            if abs(snap.link_y - 141) > 4:
                btn = "UP" if snap.link_y > 141 else "DOWN"
                return self._swing(btn, "06_realign_y141")
            return self._swing("LEFT", "06_west_0x05")

        return None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.failed or self.phase is PathNavPhase.FAILED:
            return FrameAction(nes_idle_action(), self.blocked_reason or "failed")
        if self.check_handoff and not self._handoff_checked:
            mismatch = self.handoff.mismatch(snap)
            if mismatch is not None:
                return self._fail_now(mismatch)
            self._handoff_checked = True
            self.notes.append("post_l8_handoff_accepted")
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        out = super().report()
        out.update(
            {
                "chapter": "level9_post_l8_overworld",
                "evidence": self.handoff.evidence,
                "route_eligible": self.handoff.route_eligible and self.success,
                "failed": self.failed or (self.phase is PathNavPhase.FAILED),
                "reason": self.blocked_reason or (self.notes[-1] if self.notes else None),
                "missing_evidence": self.blocked_reason or None,
                "controller_memory_writes": 0, "progression_writes": 0,
                "capacity_writes": 0, "inventory_writes": 0, "triforce_writes": 0,
                "bomb_capacity_writes": 0, "room_writes": 0, "door_writes": 0,
                "writes": 0,
            }
        )
        return out


def has_full_triforce(ram) -> bool:
    return read_u8(ram, ADDR_TRIFORCE) == FULL_TRIFORCE


def triforce_bits(ram) -> int:
    return int(read_u8(ram, ADDR_TRIFORCE))


# Left Spectacle Rock secret: tile object 0x63 in slot 11; $80 in its state
# is "revealed" (OW secrets are tile objects).
ROCK_SECRET_TYPE = 0x63
ROCK_MAX_BOMBS = 3


LYNEL_BEAM_TYPE = 0x57


def _beam_on_row(snap: ZeldaSnapshot, reach: int = 64) -> bool:
    return any(
        int(o.type_id) == LYNEL_BEAM_TYPE
        and abs(int(o.y) - int(snap.link_y)) <= 8
        and abs(int(o.x) - int(snap.link_x)) <= reach
        for o in snap.objects
    )


FACING_UP = 0x08
# A surfaced red Leever at (74,173) knocked Link off the stand on the B frame.
_NOT_BODIES = frozenset({0x57, 0x5F, 0x60, 0x63, 0x64})


def _body_near(snap: ZeldaSnapshot, reach: int = 24) -> bool:
    lx, ly = int(snap.link_x), int(snap.link_y)
    return any(
        o.slot >= 1
        and int(o.type_id) not in (0, 0xFF)
        and int(o.type_id) not in _NOT_BODIES
        and int(o.hp) > 0
        and max(abs(int(o.x) - lx), abs(int(o.y) - ly)) <= reach
        for o in snap.objects
    )


def _rock_secret_unrevealed(snap: ZeldaSnapshot) -> bool:
    obj = snap.object_in_slot(11)
    return obj is not None and int(obj.type_id) == ROCK_SECRET_TYPE and not int(obj.state) & 0x80


class SpectacleRockBombPhase(Enum):
    """Phases for bombing left Spectacle Rock and entering Level 9 room 0x76."""
    ALIGN_216_X, PAUSE_OPEN, ROCK_TOP_Y = auto(), auto(), auto()
    ROCK_GAP_X, ROCK_BOTTOM_Y, ROCK_LEFT_X = auto(), auto(), auto()
    ROCK_FACE_UP, ROCK_FIRE, ROCK_BLAST_WAIT = auto(), auto(), auto()
    ROCK_ENTER, DUNGEON_SETTLE, DONE, FAILED = auto(), auto(), auto(), auto()


@dataclass
class Level9SpectacleRockBombController:
    """Bomb the left Spectacle Rock on OW 0x05 and enter Level 9 room 0x76.

    Takes over on OW 0x05 at (240, 141) from Level9PostL8OverworldController.
    Refuses without full Triforce (0xFF), natural bombs (>0), and complete
    measured post-L8 handoff.
    """

    handoff: PostLevel8Handoff = UNMEASURED_POST_L8_HANDOFF
    max_frames: int = 1
    phase: SpectacleRockBombPhase = SpectacleRockBombPhase.ALIGN_216_X
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    failed: bool = False
    failure: str = ""
    notes: list[str] = field(default_factory=list)
    bombs_before: int | None = None
    bombs_after: int | None = None
    b_presses: int = 0
    blast_wait_frames: int = 0
    dungeon_settle_frames: int = 0
    _handoff_checked: bool = False
    _env: Any = field(default=None, repr=False)
    _select: PauseSelectController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._select = PauseSelectController(want=B_ITEM_BOMBS, name="bombs")
        if not self.handoff.complete():
            self.max_frames = 1
        else:
            self.max_frames = 4000

    def bind_env(self, env: Any) -> None:
        self._env = env
        self._select.bind_env(env)

    def _selected(self) -> int:
        if self._env is not None:
            return read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM)
        return int(self.handoff.selected_item or B_ITEM_BOMBS)

    def _set_phase(self, phase: SpectacleRockBombPhase, note: str = "") -> None:
        if phase is self.phase:
            return
        self.phase = phase
        self.phase_frames = 0
        if note:
            self.notes.append(note)

    def _action(self, action: list[int], reason: str) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        return FrameAction(action, reason)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self.failure = reason
        self.phase = SpectacleRockBombPhase.FAILED
        self.notes.append(reason)
        return self._action(nes_idle_action(), reason)

    def _axis(
        self,
        snap: ZeldaSnapshot,
        *,
        axis: str,
        target: int,
        tolerance: int,
        reason: str,
    ) -> FrameAction | None:
        value = getattr(snap, f"link_{axis}")
        delta = target - value
        if abs(delta) <= tolerance:
            return None
        btn = "RIGHT" if delta > 0 else "LEFT"
        if axis == "y":
            btn = "DOWN" if delta > 0 else "UP"
        return self._action(nes_action(btn), reason)

    def _goto(
        self, snap: ZeldaSnapshot, goal: tuple[int, int], tol: int, reason: str
    ) -> FrameAction | None:
        """One ROM-lattice press toward ``goal``, or None once within ``tol``."""
        step = room_step(snap, goal, tol=tol, env=self._env)
        return None if step is None else self._action(nes_action(step), reason)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.failed or self.phase is SpectacleRockBombPhase.FAILED:
            return FrameAction(nes_idle_action(), self.failure or "failed")
        if self.success or self.phase is SpectacleRockBombPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.frames >= self.max_frames:
            return self._fail("spectacle_rock_bomb_timeout")

        if not self._handoff_checked:
            self._handoff_checked = True
            if snap.triforce != FULL_TRIFORCE:
                return self._fail(TRIFORCE_NOT_FULL)
            if snap.bombs < 1:
                return self._fail(BOMBS_NOT_NATURAL)
            if not self.handoff.complete():
                return self._fail(MISSING_SPECTACLE_BOMB)
            if snap.level != 0 or snap.screen != SCREEN_LEVEL9_ROCK_HYP:
                return self._fail("not_on_spectacle_rock_0x05")
            self.bombs_before = int(snap.bombs)
            self.notes.append("spectacle_rock_handoff_accepted")
            if self._selected() != B_ITEM_BOMBS:
                self._set_phase(SpectacleRockBombPhase.PAUSE_OPEN, "pause_select_bombs")

        if snap.mode == 17:
            return self._fail("link_death")

        if snap.level == LEVEL9:
            if snap.mode == PLAY_MODE and not snap.transitioning:
                if snap.screen != ROOM_LEVEL9_ENTRY:
                    return self._fail(f"wrong_level9_entry_room_0x{snap.screen:02X}")
                if self.phase is not SpectacleRockBombPhase.DUNGEON_SETTLE:
                    self._set_phase(SpectacleRockBombPhase.DUNGEON_SETTLE, "settled_l9_0x76")
                self.dungeon_settle_frames += 1
                if self.dungeon_settle_frames >= 24:
                    expected_bombs = int(self.bombs_before or 0) - self.b_presses
                    if int(snap.bombs) != expected_bombs:
                        return self._fail("bomb_delta_not_the_presses")
                    self.bombs_after = int(snap.bombs)
                    self.success = True
                    self._set_phase(SpectacleRockBombPhase.DONE, "natural_entry_0x76")
                    return self._action(nes_idle_action(), "done")
                return self._action(nes_idle_action(), "dungeon_0x76_settle")
            return self._action(nes_idle_action(), "dungeon_loader_wait")

        if snap.transitioning:
            return self._action(nes_idle_action(), "transition_wait")

        if self.phase is SpectacleRockBombPhase.PAUSE_OPEN:
            driven = self._select.drive(snap)
            if self._select.failed:
                return self._fail(self._select.fail_reason or "pause_select_failed")
            if driven is not None:
                return self._action(driven.action, driven.reason)
            self._set_phase(SpectacleRockBombPhase.ALIGN_216_X, "bombs_selected")

        # The walk to the stand is four lattice legs (``_goto``): east column,
        # top row, centre gap, south row. Blind one-axis presses pinned Link
        # in the (64,77) nook for 3200 frames once a dodge moved him off the
        # line they were tuned on.
        if self.phase is SpectacleRockBombPhase.ALIGN_216_X:
            self._set_phase(SpectacleRockBombPhase.ROCK_TOP_Y, "rock_walk_lattice")

        if self.phase is SpectacleRockBombPhase.ROCK_TOP_Y:
            act = self._goto(snap, (216, 93), 4, "rock_climb_top_y93")
            if act is not None:
                return act
            self._set_phase(SpectacleRockBombPhase.ROCK_GAP_X, "rock_top_y93_reached")

        if self.phase is SpectacleRockBombPhase.ROCK_GAP_X:
            act = self._goto(snap, (120, 93), 4, "rock_top_to_center_gap_x120")
            if act is not None:
                return act
            self._set_phase(SpectacleRockBombPhase.ROCK_BOTTOM_Y, "rock_center_gap_reached")

        if self.phase is SpectacleRockBombPhase.ROCK_BOTTOM_Y:
            act = self._goto(snap, (120, 173), 4, "rock_center_gap_to_south_y173")
            if act is not None:
                return act
            self._set_phase(SpectacleRockBombPhase.ROCK_LEFT_X, "rock_south_reached")

        if self.phase is SpectacleRockBombPhase.ROCK_LEFT_X:
            # Exact stand: the fire check needs x within 2 of 80, and a 4 px
            # walk tolerance swapped the two at x=84.
            act = self._goto(snap, (80, 173), 1, "rock_south_to_left_stand_x80")
            if act is not None:
                return act
            self._set_phase(SpectacleRockBombPhase.ROCK_FACE_UP, "rock_left_stand_reached")

        if self.phase is SpectacleRockBombPhase.ROCK_FACE_UP:
            # Only on the stand: a Lynel beam knocked Link 4 px off x=80 on
            # the B frame and the bomb missed the secret (power-on gathered
            # spine).
            # The turn press walks Link up toward the rock; a bomb from as high
            # as y=163 still lands on it.
            if abs(snap.link_x - 80) > 2 or not 163 <= snap.link_y <= 175:
                self._set_phase(SpectacleRockBombPhase.ROCK_LEFT_X, "rock_stand_lost")
                return self._action(nes_idle_action(), "rock_stand_lost")
            # Lynel sword beams run along y=173; one landed on the B frame on
            # all three tries. Hold the bomb until the row is clear.
            if _beam_on_row(snap) or _body_near(snap):
                return self._action(nes_action("UP"), "rock_hold_threat")
            # B drops the bomb the way Link faces: a one-frame UP had not
            # turned him yet and the bomb went east of the rock.
            if int(snap.facing) != FACING_UP:
                return self._action(nes_action("UP"), "left_rock_face_up")
            self._set_phase(SpectacleRockBombPhase.ROCK_FIRE, "left_rock_faced_up")
            return self._action(nes_action("UP"), "left_rock_face_up")

        if self.phase is SpectacleRockBombPhase.ROCK_FIRE:
            self.b_presses += 1
            self.blast_wait_frames = 0
            self._set_phase(SpectacleRockBombPhase.ROCK_BLAST_WAIT, "one_bomb_below_left_rock")
            return self._action(nes_action("B"), "left_spectacle_rock_bomb")

        if self.phase is SpectacleRockBombPhase.ROCK_BLAST_WAIT:
            self.blast_wait_frames += 1
            if snap.bombs != self.bombs_before:
                self.bombs_after = int(snap.bombs)
            if self.blast_wait_frames < 180:
                return self._action(nes_idle_action(), "left_rock_blast_wait")
            if self.bombs_after != int(self.bombs_before or 0) - self.b_presses:
                return self._fail("bomb_was_not_consumed_once_per_press")
            self._set_phase(SpectacleRockBombPhase.ROCK_ENTER, "left_rock_blast_complete")

        if self.phase is SpectacleRockBombPhase.ROCK_ENTER:
            if _rock_secret_unrevealed(snap) and self.b_presses < ROCK_MAX_BOMBS:
                self._set_phase(SpectacleRockBombPhase.ROCK_LEFT_X, "rock_secret_missed_retry")
                return self._action(nes_idle_action(), "rock_retry")
            if self.phase_frames > 1200:
                return self._fail("left_rock_mouth_did_not_enter")
            ax_x = self._axis(snap, axis="x", target=80, tolerance=4, reason="left_rock_mouth_realign")
            if ax_x is not None:
                return ax_x
            # The rock entrance triggers at y=157. A knockback during blast
            # wait can push Link north of the entrance (e.g. y=149); walk
            # down into the opening instead of pushing up into the wall.
            if snap.link_y < 157:
                if _body_near(snap) and (self.phase_frames % 16 in (0, 1)):
                    return self._action(nes_action("DOWN", "A"), "enter_left_spectacle_rock_slash_down")
                return self._action(nes_action("DOWN"), "enter_left_spectacle_rock_from_north")
            if _body_near(snap) and (self.phase_frames % 16 in (0, 1)):
                return self._action(nes_action("UP", "A"), "enter_left_spectacle_rock_slash")
            return self._action(nes_action("UP"), "enter_left_spectacle_rock")

        return self._fail("unknown_spectacle_rock_bomb_phase")

    def report(self) -> dict[str, Any]:
        return {
            "chapter": "level9_spectacle_rock_bomb",
            "evidence": self.handoff.evidence,
            "route_eligible": self.handoff.route_eligible and self.success,
            "success": self.success,
            "failed": self.failed,
            "reason": self.failure or (self.notes[-1] if self.notes else None),
            "missing_evidence": self.failure or None,
            "frames": self.frames,
            "phase": self.phase.name,
            "bombs_before": self.bombs_before,
            "bombs_after": self.bombs_after,
            "b_presses": self.b_presses,
            "controller_memory_writes": 0, "progression_writes": 0,
            "capacity_writes": 0, "inventory_writes": 0, "position_writes": 0,
            "selected_item_writes": 0, "triforce_writes": 0,
            "bomb_capacity_writes": 0, "room_writes": 0, "door_writes": 0,
            "writes": 0,
        }
