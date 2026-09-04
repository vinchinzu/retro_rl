"""Level 7 Demon pond: pause-select Recorder, whistle-drain, stairs into 0x79.

Live recipe (``scratch/probe_l7_pond_drain.py`` drain_v2, ``LEVEL7_ROUTE.md``):
OW ``$EB=0x42`` south shore ``(128,221)`` → stand ``(128,189)`` → 12×B + idle
~240 for the song → stairs ``(96,132)`` tile 114 → L7 play ``0x79``
``(120,205)``.  Requires already-owned Whistle (``ADDR_WHISTLE`` / ``$065C``
>= 1).  Never poke whistle or ``$0656`` selected_item.

Pause-select B-slot 5 through the menu — START / idle 20 / RIGHT / idle 8 /
START close / idle 24, same shape as ``level7.hungry`` / ``level7.digdogger``.
Snapshot has no ``selected_item``; read ``ADDR_SELECTED_ITEM`` after
``bind_env``.

OccupancyWalker is banned (overworld south shore sits outside dungeon bounds).
Waypoint micro toward the live stairs cell; seek probe candidates if that
cell does not trigger; halt on the first occupancy miss (do not batch).
No RAM writes.  ``route_eligible=False``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_u8,
)

__all__ = [
    "BLOW_STAND",
    "DEST",
    "DEST_XY",
    "LEVEL7",
    "POND_SCREEN",
    "POND_STAIR_TILE",
    "SOUTH_SHORE",
    "STAIR_CANDIDATES",
    "STAIRS_XY",
    "WHISTLE_B_SLOT",
    "Level7PondDrainController",
    "PondPhase",
    "make_pond_drain_controller",
]

LEVEL7 = 7
DEATH_MODE = 17
POND_SCREEN = 0x42
DEST = 0x79
DEST_XY = (120, 205)
SOUTH_SHORE = (128, 221)
BLOW_STAND = (128, 189)
STAIRS_XY = (96, 132)
POND_STAIR_TILE = 114
WHISTLE_B_SLOT = 5
# drain_v2 first cell is STAIRS_XY; remaining cells copy probe STAIR_CANDIDATES.
STAIR_CANDIDATES: tuple[tuple[int, int], ...] = (
    STAIRS_XY,
    (104, 128),
    (96, 128),
    (112, 128),
    (104, 136),
    (96, 136),
    (112, 136),
    (104, 120),
    (88, 128),
    (120, 128),
)
POND_MAX_FRAMES = 8000
SELECT_MAX_FRAMES = 240
STAND_SETTLE_FRAMES = 8
OPEN_SETTLE_FRAMES = 20
CURSOR_SETTLE_FRAMES = 8
CLOSE_SETTLE_FRAMES = 24
MAX_CURSOR_MOVES = 8
BLOW_PRESSES = 12
BLOW_WAIT_FRAMES = 240
ARRIVE_TOL = 3
STAIR_DWELL_FRAMES = 12
STUCK_FRAMES = 16
WAIT_MODES = (2, 3, 4, 6, 7, 9, 10, 11, 16)


class PondPhase(Enum):
    OPEN = auto()
    OPEN_SETTLE = auto()
    CYCLE = auto()
    CURSOR_SETTLE = auto()
    CLOSE = auto()
    CLOSE_SETTLE = auto()
    WALK = auto()
    STAND_SETTLE = auto()
    BLOW = auto()
    BLOW_WAIT = auto()
    STAIRS = auto()
    DONE = auto()
    FAILED = auto()


_SELECT_PHASES = (
    PondPhase.OPEN,
    PondPhase.OPEN_SETTLE,
    PondPhase.CYCLE,
    PondPhase.CURSOR_SETTLE,
    PondPhase.CLOSE,
    PondPhase.CLOSE_SETTLE,
)


def _toward(
    xy: tuple[int, int], dest: tuple[int, int], *, tol: int
) -> str | None:
    """Larger-delta cardinal, matching probe ``_seek`` / stair walk."""
    x, y = xy
    tx, ty = dest
    dx, dy = tx - x, ty - y
    if abs(dx) <= tol and abs(dy) <= tol:
        return None
    if abs(dx) >= abs(dy) and abs(dx) > tol:
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class Level7PondDrainController:
    """Drain live OW ``0x42`` with owned Whistle and enter play ``0x79``."""

    max_frames: int = POND_MAX_FRAMES
    phase: PondPhase = PondPhase.OPEN
    frames: int = 0
    phase_frames: int = 0
    cursor_moves: int = 0
    blow_presses: int = 0
    stair_index: int = 0
    success: bool = False
    failed: bool = False
    blew: bool = False
    screen_in: int | None = None
    level_in: int | None = None
    leftover: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)
    _stuck: int = field(default=0, init=False, repr=False)
    _dwell: int = field(default=0, init=False, repr=False)
    _last_xy: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: PondPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(PondPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, snap: ZeldaSnapshot, reason: str = "entered_0x79") -> FrameAction:
        self.success = True
        self.leftover = {
            "level": int(snap.level),
            "screen": int(snap.screen),
            "mode": int(snap.mode),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
        }
        self._set_phase(PondPhase.DONE, reason)
        return FrameAction(nes_idle_action(), reason)

    def _begin_blow(self, note: str) -> FrameAction:
        self.blow_presses = 1
        self._set_phase(PondPhase.BLOW, note)
        return FrameAction(nes_action("B"), "whistle_blow")

    def _walk_to_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        direction = _toward(
            (int(snap.link_x), int(snap.link_y)), BLOW_STAND, tol=ARRIVE_TOL
        )
        if direction is not None:
            axis = "y" if direction in ("UP", "DOWN") else "x"
            return FrameAction(nes_action(direction), f"stand_{axis}")
        self._set_phase(PondPhase.STAND_SETTLE, "at_stand")
        return FrameAction(nes_idle_action(), "stand_arrive")

    def _stairs_cell(self) -> tuple[int, int]:
        return STAIR_CANDIDATES[min(self.stair_index, len(STAIR_CANDIDATES) - 1)]

    def _advance_stair_candidate(self) -> None:
        missed = self._stairs_cell()
        self._note(f"stairs_miss_{missed[0]}_{missed[1]}")
        self.stair_index += 1
        self._dwell = 0
        self._stuck = 0
        self._last_xy = None

    def _walk_stairs(self, snap: ZeldaSnapshot) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        dest = self._stairs_cell()
        direction = _toward(xy, dest, tol=ARRIVE_TOL)
        if direction is None:
            self._dwell += 1
            if self._dwell >= STAIR_DWELL_FRAMES:
                if self.stair_index + 1 >= len(STAIR_CANDIDATES):
                    return self._fail("stairs_not_found")
                self._advance_stair_candidate()
            return FrameAction(nes_action("UP"), "stairs_step")
        if self._last_xy == xy:
            self._stuck += 1
            if self._stuck >= STUCK_FRAMES:
                return self._fail("occupancy_miss")
        else:
            self._stuck = 0
        self._last_xy = xy
        tag = "stairs_seek" if self.stair_index else "stairs"
        axis = "y" if direction in ("UP", "DOWN") else "x"
        return FrameAction(nes_action(direction), f"{tag}_{axis}")

    def _after_select(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is PondPhase.WALK:
            return self._walk_to_stand(snap)
        if self.phase is PondPhase.STAND_SETTLE:
            if self.phase_frames >= STAND_SETTLE_FRAMES:
                return self._begin_blow("whistle_ready")
            return FrameAction(nes_idle_action(), "stand_settle")
        if self.phase is PondPhase.BLOW:
            if self.blow_presses >= BLOW_PRESSES:
                self._set_phase(PondPhase.BLOW_WAIT, "whistle_wait")
                return FrameAction(nes_idle_action(), "whistle_wait")
            self.blow_presses += 1
            return FrameAction(nes_action("B"), "whistle_blow")
        if self.phase is PondPhase.BLOW_WAIT:
            if self.phase_frames >= BLOW_WAIT_FRAMES:
                self.blew = True
                self._set_phase(PondPhase.STAIRS, "seek_stairs")
                return self._walk_stairs(snap)
            return FrameAction(nes_idle_action(), "whistle_wait")
        if self.phase is PondPhase.STAIRS:
            return self._walk_stairs(snap)
        return FrameAction(nes_idle_action(), "done")

    def _select(self, snap: ZeldaSnapshot, selected: int) -> FrameAction:
        if self.frames >= SELECT_MAX_FRAMES:
            return self._fail("select_recorder_timeout")
        if self.phase is PondPhase.OPEN:
            if selected == WHISTLE_B_SLOT:
                self._set_phase(PondPhase.WALK, "recorder_already_selected")
                return self._after_select(snap)
            self._set_phase(PondPhase.OPEN_SETTLE, "pause_open")
            return FrameAction(nes_action("START"), "pause_open")
        if self.phase is PondPhase.OPEN_SETTLE:
            if self.phase_frames >= OPEN_SETTLE_FRAMES:
                self._set_phase(PondPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_settle")
        if self.phase is PondPhase.CYCLE:
            if selected == WHISTLE_B_SLOT:
                self._set_phase(PondPhase.CLOSE, "recorder_cursor_selected")
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= MAX_CURSOR_MOVES:
                return self._fail("recorder_cursor_not_found")
            self.cursor_moves += 1
            self._set_phase(PondPhase.CURSOR_SETTLE)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")
        if self.phase is PondPhase.CURSOR_SETTLE:
            if self.phase_frames >= CURSOR_SETTLE_FRAMES:
                self._set_phase(PondPhase.CYCLE)
            return FrameAction(nes_idle_action(), "pause_cursor_settle")
        if self.phase is PondPhase.CLOSE:
            self._set_phase(PondPhase.CLOSE_SETTLE, "pause_close")
            return FrameAction(nes_action("START"), "pause_close")
        if self.phase is PondPhase.CLOSE_SETTLE:
            if self.phase_frames < CLOSE_SETTLE_FRAMES:
                return FrameAction(nes_idle_action(), "pause_resume")
            if (
                snap.level == 0
                and snap.mode == PLAY_MODE
                and int(snap.screen) == POND_SCREEN
                and selected == WHISTLE_B_SLOT
            ):
                self._set_phase(PondPhase.WALK, "recorder_selected_naturally")
                return self._after_select(snap)
            return self._fail("pause_close_contract_mismatch")
        return FrameAction(nes_idle_action(), "select")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success or self.failed:
            return FrameAction(nes_idle_action(), "done")
        if self._env is None:
            return self._fail("pond_drain_env_not_bound")
        ram = self._env.get_ram()
        whistle = int(read_u8(ram, ADDR_WHISTLE))
        selected = int(read_u8(ram, ADDR_SELECTED_ITEM))
        if self.screen_in is None:
            self.screen_in = int(snap.screen)
            self.level_in = int(snap.level)
            self._note(f"screen_in_L{snap.level}_0x{snap.screen:02x}")
        if whistle < 1:
            return self._fail("pond_requires_whistle")
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
        if dest_settled:
            started_elsewhere = not (
                self.level_in == LEVEL7 and self.screen_in == DEST
            )
            if started_elsewhere:
                return self._finish(snap)
            return self._fail("already_in_entry_without_drain")
        if snap.transitioning or snap.mode in WAIT_MODES:
            if self.phase is PondPhase.STAIRS:
                return FrameAction(nes_idle_action(), "stairs_scroll")
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.level == 0:
                if int(snap.screen) != POND_SCREEN:
                    return self._fail(f"left_ow_0x{snap.screen:02x}")
            elif snap.level != LEVEL7:
                return self._fail(f"left_level_{snap.level}")
            elif int(snap.screen) != DEST:
                return self._fail(
                    f"left_room_L{snap.level}_0x{snap.screen:02x}"
                )
        if self.phase in _SELECT_PHASES:
            return self._select(snap, selected)
        return self._after_select(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_pond_drain_entry",
            "pond_screen": f"0x{POND_SCREEN:02X}",
            "dest": f"0x{DEST:02X}",
            "blow_stand": list(BLOW_STAND),
            "stairs_xy": list(STAIRS_XY),
            "stair_tile": POND_STAIR_TILE,
            "stair_index": self.stair_index,
            "blew": self.blew,
            "phase": self.phase.name,
            "cursor_moves": self.cursor_moves,
            "blow_presses": self.blow_presses,
            "screen_in": self.screen_in,
            "level_in": self.level_in,
            "leftover": dict(self.leftover) if self.leftover else None,
            "normal_pause_input": True,
            "writes": 0,
            "route_eligible": False,
            "evidence": "fixture-live",
            "notes": list(self.notes),
        }


def make_pond_drain_controller() -> Level7PondDrainController:
    """Fresh 0x42 whistle-drain + 0x79 entry controller (never share instances)."""
    return Level7PondDrainController()
