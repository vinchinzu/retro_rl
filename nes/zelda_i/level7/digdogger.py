"""Level 7 room 0x1C: whistle-shrink FORCED_DIGDOGGER, sword, north 0x0C.

Live recipe (probe ``scratch/probe_l7_forced_digdogger.py``, tags
``1c_wh_v3``/``v4``): west mouth ``(16,141)``, stand ``(120,141)``, pause-select
recorder B-slot 5 (cycle past Red Candle=4; no ``$0656`` poke), 12×B until
type ``0x38`` → ``0x18``, sword the shrunk bodies, UP to play ``0x0C``.

Do not call ``level5.boss_path.fight_digdogger`` (it mutates env). OccupancyWalker
is banned. No RAM writes. ``route_eligible`` stays False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.dungeon.behaviors import (
    DIGDOGGER_SHRUNK_TYPE,
    DIGDOGGER_TYPE,
    EnemyKind,
    engagement_hint,
)
from zelda_i.dungeon.hop_controller import axis_dir, dungeon_align_then_push
from zelda_i.ram import ADDR_SELECTED_ITEM, PLAY_MODE, ZeldaSnapshot, read_u8

__all__ = [
    "DEST",
    "LEVEL7",
    "ROOM",
    "WHISTLE_B_SLOT",
    "WHISTLE_STAND",
    "DigdoggerPhase",
    "Level7ForcedDigdoggerController",
    "make_level7_forced_digdogger_controller",
]

LEVEL7 = 7
ROOM = 0x1C
DEST = 0x0C
DEATH_MODE = 17
WHISTLE_B_SLOT = 5
WHISTLE_STAND = (120, 141)
NORTH_DOOR = (120, 93)
ARRIVE_TOL = 3
STAND_SETTLE_FRAMES = 8
OPEN_SETTLE_FRAMES = 20
CURSOR_SETTLE_FRAMES = 8
CLOSE_SETTLE_FRAMES = 24
MAX_CURSOR_MOVES = 8
BLOW_PRESSES = 12
BLOW_WAIT_FRAMES = 240
BLOW_ATTEMPTS = 4
EMPTY_SWORD_FRAMES = 30
DIGDOGGER_MAX_FRAMES = 16000
_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
_SCROLL_MODES = (2, 3, 4, 6, 7, 10, 16)


class DigdoggerPhase(Enum):
    WALK = auto()
    STAND_SETTLE = auto()
    SELECT_OPEN = auto()
    SELECT_OPEN_SETTLE = auto()
    SELECT_CYCLE = auto()
    SELECT_CURSOR_SETTLE = auto()
    SELECT_CLOSE = auto()
    SELECT_CLOSE_SETTLE = auto()
    BLOW = auto()
    BLOW_WAIT = auto()
    SWORD = auto()
    EXIT = auto()
    DONE = auto()
    FAILED = auto()


def _boss_types(snap: ZeldaSnapshot) -> set[int]:
    return {
        int(obj.type_id) & 0xFF
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12
        and (int(obj.type_id) & 0xFF) in (DIGDOGGER_TYPE, DIGDOGGER_SHRUNK_TYPE)
    }


def _large(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12 and (int(obj.type_id) & 0xFF) == DIGDOGGER_TYPE
    ]


def _shrunk_live(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= int(obj.slot) <= 12
        and (int(obj.type_id) & 0xFF) == DIGDOGGER_SHRUNK_TYPE
        and int(obj.hp) > 0
    ]


def _live_digdogger(snap: ZeldaSnapshot) -> bool:
    return bool(_large(snap) or _shrunk_live(snap))


@dataclass
class Level7ForcedDigdoggerController:
    """Kill live ``$EB=0x1C`` Digdogger and leave north into play ``0x0C``."""

    max_frames: int = DIGDOGGER_MAX_FRAMES
    frames: int = 0
    phase_frames: int = 0
    phase: DigdoggerPhase = DigdoggerPhase.WALK
    success: bool = False
    failed: bool = False
    saw_boss: bool = False
    saw_large: bool = False
    shrunk: bool = False
    killed: bool = False
    cursor_moves: int = 0
    blow_presses: int = 0
    blow_attempts: int = 0
    sword_frames: int = 0
    empty_sword_frames: int = 0
    selected_before: int | None = None
    leftover: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)
    _env: Any = field(default=None, init=False, repr=False)

    def bind_env(self, env: Any) -> None:
        self._env = env

    def _note(self, note: str) -> None:
        if note not in self.notes:
            self.notes.append(note)

    def _set_phase(self, phase: DigdoggerPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self._note(note)

    def _selected(self) -> int | None:
        if self._env is None:
            return None
        return read_u8(self._env.get_ram(), ADDR_SELECTED_ITEM)

    def _fail(self, reason: str) -> FrameAction:
        self.failed = True
        self._set_phase(DigdoggerPhase.FAILED, reason)
        return FrameAction(nes_idle_action(), reason)

    def _finish(self, snap: ZeldaSnapshot, reason: str = "dest_0x0c") -> FrameAction:
        self.success = True
        self.killed = True
        self.leftover = {
            "level": int(snap.level),
            "screen": int(snap.screen),
            "mode": int(snap.mode),
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "triforce": int(snap.triforce),
        }
        self._set_phase(DigdoggerPhase.DONE, reason)
        return FrameAction(nes_idle_action(), reason)

    def _observe(self, snap: ZeldaSnapshot) -> None:
        types = _boss_types(snap)
        if DIGDOGGER_TYPE in types:
            self.saw_boss = True
            self.saw_large = True
        if DIGDOGGER_SHRUNK_TYPE in types:
            self.saw_boss = True
            self.shrunk = True

    def _walk_to(
        self, snap: ZeldaSnapshot, dest: tuple[int, int], reason: str
    ) -> FrameAction | None:
        direction = axis_dir(
            (int(snap.link_x), int(snap.link_y)), dest, y_first=True, tol=ARRIVE_TOL
        )
        if direction is None:
            return None
        axis = "y" if direction in ("UP", "DOWN") else "x"
        return FrameAction(nes_action(direction), f"{reason}_{axis}")

    def _begin_blow(self, note: str) -> FrameAction:
        self.blow_attempts += 1
        self.blow_presses = 1
        self._set_phase(DigdoggerPhase.BLOW, note)
        return FrameAction(nes_action("B"), "whistle_blow")

    def _after_stand(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        selected = self._selected()
        if selected is None:
            return self._fail("digdogger_env_not_bound")
        if self.selected_before is None:
            self.selected_before = selected
        if selected == WHISTLE_B_SLOT:
            return self._begin_blow("recorder_already_selected")
        self._set_phase(DigdoggerPhase.SELECT_OPEN_SETTLE, "pause_open")
        return FrameAction(nes_action("START"), "pause_open")

    def _select(self, snap: ZeldaSnapshot) -> FrameAction:
        selected = self._selected()
        if selected is None:
            return self._fail("digdogger_env_not_bound")
        if self.phase is DigdoggerPhase.SELECT_OPEN:
            if selected == WHISTLE_B_SLOT:
                return self._begin_blow("recorder_already_selected")
            self._set_phase(DigdoggerPhase.SELECT_OPEN_SETTLE, "pause_open")
            return FrameAction(nes_action("START"), "pause_open")
        if self.phase is DigdoggerPhase.SELECT_OPEN_SETTLE:
            if self.phase_frames >= OPEN_SETTLE_FRAMES:
                self._set_phase(DigdoggerPhase.SELECT_CYCLE)
            return FrameAction(nes_idle_action(), "pause_settle")
        if self.phase is DigdoggerPhase.SELECT_CYCLE:
            if selected == WHISTLE_B_SLOT:
                self._set_phase(DigdoggerPhase.SELECT_CLOSE, "recorder_cursor_selected")
                return FrameAction(nes_idle_action(), "cursor_ready")
            if self.cursor_moves >= MAX_CURSOR_MOVES:
                return self._fail("recorder_cursor_not_found")
            self.cursor_moves += 1
            self._set_phase(DigdoggerPhase.SELECT_CURSOR_SETTLE)
            return FrameAction(nes_action("RIGHT"), "pause_next_item")
        if self.phase is DigdoggerPhase.SELECT_CURSOR_SETTLE:
            if self.phase_frames >= CURSOR_SETTLE_FRAMES:
                self._set_phase(DigdoggerPhase.SELECT_CYCLE)
            return FrameAction(nes_idle_action(), "pause_cursor_settle")
        if self.phase is DigdoggerPhase.SELECT_CLOSE:
            self._set_phase(DigdoggerPhase.SELECT_CLOSE_SETTLE, "pause_close")
            return FrameAction(nes_action("START"), "pause_close")
        if self.phase is DigdoggerPhase.SELECT_CLOSE_SETTLE:
            if self.phase_frames < CLOSE_SETTLE_FRAMES:
                return FrameAction(nes_idle_action(), "pause_resume")
            if (
                snap.level == LEVEL7
                and snap.mode == PLAY_MODE
                and int(snap.screen) == ROOM
                and selected == WHISTLE_B_SLOT
            ):
                return self._begin_blow("recorder_selected_naturally")
            return self._fail("pause_close_contract_mismatch")
        return FrameAction(nes_idle_action(), "select")

    def _sword(self, snap: ZeldaSnapshot) -> FrameAction:
        self.sword_frames += 1
        live = _shrunk_live(snap)
        if not live:
            if _large(snap):
                return self._begin_blow("large_still_present")
            self.empty_sword_frames += 1
            if self.empty_sword_frames >= EMPTY_SWORD_FRAMES:
                self.killed = True
                self._set_phase(DigdoggerPhase.EXIT, "digdogger_killed")
                return self._exit_north(snap)
            return FrameAction(nes_idle_action(), "wait_shrunk_slots")
        self.empty_sword_frames = 0
        target = nearest_enemy(snap.link_x, snap.link_y, live)
        if target is None:
            return FrameAction(nes_idle_action(), "sword_idle")
        hint = engagement_hint(EnemyKind.DIGDOGGER, snap, target)
        if should_swing_at(
            snap.link_x, snap.link_y, hint.face, (target,), hint=hint
        ) and (self.sword_frames % 8) < 4:
            return FrameAction(nes_action(hint.face, "A"), "sword_swing")
        if hint.retreat:
            back = _OPPOSITE.get(hint.face, "DOWN")
            return FrameAction(nes_action(back), "sword_retreat")
        return FrameAction(nes_action(hint.face), "sword_chase")

    def _exit_north(self, snap: ZeldaSnapshot) -> FrameAction:
        # Hold UP on/above the door plane. Walking DOWN to y=93 oscillates
        # in the mouth (live leftover sat at (120,89) for the full budget).
        return dungeon_align_then_push(
            snap,
            push_dir="UP",
            target_x=NORTH_DOOR[0],
            x_tol=ARRIVE_TOL,
            reason="north",
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == DEATH_MODE:
            return self._fail("death")
        self._observe(snap)

        dest_play = (
            int(snap.level) == LEVEL7
            and int(snap.screen) == DEST
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        )
        if dest_play:
            if self.saw_boss and self.shrunk and not _live_digdogger(snap):
                return self._finish(snap)
            return self._fail("dest_without_kill_edge")
        if self.frames >= self.max_frames:
            return self._fail("budget_exhausted")

        if snap.transitioning or snap.mode in _SCROLL_MODES:
            if self.phase is DigdoggerPhase.EXIT or self.killed:
                return FrameAction(nes_action("UP"), "north_scroll")
            return FrameAction(nes_idle_action(), "wait_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if int(snap.level) != LEVEL7:
            return self._fail(f"left_level_{snap.level}")
        if int(snap.screen) != ROOM:
            return self._fail(f"left_room_L{snap.level}_0x{snap.screen:02x}")

        if self.phase is DigdoggerPhase.WALK:
            walked = self._walk_to(snap, WHISTLE_STAND, "stand")
            if walked is not None:
                return walked
            self._set_phase(DigdoggerPhase.STAND_SETTLE, "at_stand")
            return FrameAction(nes_idle_action(), "stand_arrive")
        if self.phase is DigdoggerPhase.STAND_SETTLE:
            if self.phase_frames >= STAND_SETTLE_FRAMES:
                return self._after_stand(snap)
            return FrameAction(nes_idle_action(), "stand_settle")
        if self.phase in (
            DigdoggerPhase.SELECT_OPEN,
            DigdoggerPhase.SELECT_OPEN_SETTLE,
            DigdoggerPhase.SELECT_CYCLE,
            DigdoggerPhase.SELECT_CURSOR_SETTLE,
            DigdoggerPhase.SELECT_CLOSE,
            DigdoggerPhase.SELECT_CLOSE_SETTLE,
        ):
            return self._select(snap)
        if self.phase is DigdoggerPhase.BLOW:
            if _shrunk_live(snap):
                self._set_phase(DigdoggerPhase.SWORD, "shrunk_during_blow")
                return self._sword(snap)
            if self.blow_presses >= BLOW_PRESSES:
                self._set_phase(DigdoggerPhase.BLOW_WAIT, "whistle_wait")
                return FrameAction(nes_idle_action(), "whistle_wait")
            self.blow_presses += 1
            return FrameAction(nes_action("B"), "whistle_blow")
        if self.phase is DigdoggerPhase.BLOW_WAIT:
            if _shrunk_live(snap):
                self._set_phase(DigdoggerPhase.SWORD, "shrunk")
                return self._sword(snap)
            if self.phase_frames >= BLOW_WAIT_FRAMES:
                if _large(snap):
                    if self.blow_attempts >= BLOW_ATTEMPTS:
                        return self._fail("whistle_did_not_shrink")
                    walked = self._walk_to(snap, WHISTLE_STAND, "restand")
                    if walked is not None:
                        return walked
                    return self._begin_blow("whistle_retry")
                self.shrunk = True
                self.killed = True
                self._set_phase(DigdoggerPhase.EXIT, "large_gone_after_blow")
                return self._exit_north(snap)
            return FrameAction(nes_idle_action(), "whistle_wait")
        if self.phase is DigdoggerPhase.SWORD:
            return self._sword(snap)
        if self.phase is DigdoggerPhase.EXIT:
            return self._exit_north(snap)
        return FrameAction(nes_idle_action(), "done")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "spec_id": "level7_forced_digdogger",
            "live_room": f"0x{ROOM:02X}",
            "dest": f"0x{DEST:02X}",
            "phase": self.phase.name,
            "saw_boss": self.saw_boss,
            "shrunk": self.shrunk,
            "killed": self.killed,
            "cursor_moves": self.cursor_moves,
            "blow_attempts": self.blow_attempts,
            "selected_before": self.selected_before,
            "leftover": dict(self.leftover) if self.leftover else None,
            "writes": 0,
            "normal_pause_input": True,
            "route_eligible": False,
            "notes": list(self.notes),
        }


def make_level7_forced_digdogger_controller() -> Level7ForcedDigdoggerController:
    """Fresh 0x1C whistle-shrink + north 0x0C controller (never share instances)."""
    return Level7ForcedDigdoggerController()
