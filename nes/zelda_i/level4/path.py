"""Level 4 multi-room path controllers (entry, bomb wall, door micros).

Room specs and stop predicates remain in ``level4_dungeon``. Maze token
controllers live in ``level4_maze_path``; 0x30-0x60 in ``level4_stepladder``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon.engine import (
    DungeonPhase,
    GenericDungeonRoomController,
)
from zelda_i.level4.dungeon import (
    BOMB_61_NORTH_FACE,
    BOMB_61_NORTH_STAND,
    BOMB_61_OPENS_TO,
    KEY_61_EAST_Y,
    KEY_61_EAST_Y_TOL,
    LEFT_51_Y,
    LEVEL4,
    ROOM_12_SPEC,
    ROOM_31_SPEC,
    ROOM_32_SPEC,
    ROOM_40_SPEC,
    ROOM_50_SPEC,
    ROOM_51_SPEC,
    ROOM_61_SPEC,
    ROOM_62_SPEC,
    ROOM_L4_COMPASS_62,
    ROOM_L4_ENTRY,
    ROOM_L4_KEESE_KEY_51,
    ROOM_L4_VIRES_50,
    ROOM_L4_VIRES_61,
    VIRE_OBJECT_TYPE,
    VIRE_SPLIT_KEESE_TYPE,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


class BombWall61North:
    """Geometry stand for ``BombWallController``: 0x61 bomb-UP → 0x51."""

    room = ROOM_L4_VIRES_61
    stand = BOMB_61_NORTH_STAND
    face = BOMB_61_NORTH_FACE
    opens_to = BOMB_61_OPENS_TO


def _need_clear_61(snap: ZeldaSnapshot) -> bool:
    """True while any Vire/split type is present (HP may be 0 on first spawn frames)."""
    return any(
        1 <= o.slot <= 12
        and o.type_id in (VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE)
        for o in snap.objects
    )


def make_bomb_61_north_controller(
    *, clear_vires: bool = True
) -> BombWallController:
    """0x61 → bomb north → 0x51. Optionally clear Vires first."""
    return BombWallController(
        wall=BombWall61North(),
        level=LEVEL4,
        clear_spec=ROOM_61_SPEC if clear_vires else None,
        clear_when=_need_clear_61 if clear_vires else None,
        face_frames=6,
        step_back=0,
        wait_blast=BOMB_N_WAIT_BLAST,
        require_bomb_consumed=False,
        wait_hold_face=True,
        max_frames=20000,
    )


def make_room_61_clear_controller() -> GenericDungeonRoomController:
    """Fight-only or entry+fight clear of 0x61 Vires."""
    return GenericDungeonRoomController(ROOM_61_SPEC)


def make_room_51_key_controller() -> GenericDungeonRoomController:
    """Clear 0x51 Keese + collect key (FIXED_INVENTORY keys)."""
    return GenericDungeonRoomController(ROOM_51_SPEC)


def make_room_50_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x50 Vires (north exit → 0x40 after clear; rr-xc3x)."""
    return GenericDungeonRoomController(ROOM_50_SPEC)


def make_room_62_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x62 Vires (compass maze; pickup / exits residual)."""
    return GenericDungeonRoomController(ROOM_62_SPEC)


def make_room_40_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x40 Zols+gels (key pickup residual)."""
    return GenericDungeonRoomController(ROOM_40_SPEC)


def make_room_31_clear_controller() -> GenericDungeonRoomController:
    """Clear 5× Vire on 0x31 maze (rr-resv)."""
    return GenericDungeonRoomController(ROOM_31_SPEC)


def make_room_32_clear_controller() -> GenericDungeonRoomController:
    """Clear 2× Zol + 2× LikeLike on 0x32 (ignore 0x2b/0x68; rr-tib8)."""
    return GenericDungeonRoomController(ROOM_32_SPEC)


def make_room_12_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x12 Vires (settle_all_dead=0; ignore block 0x68; rr-rvae)."""
    return GenericDungeonRoomController(ROOM_12_SPEC)


# --- 0x51 free LEFT → 0x50 (rr-2ysf pocket) ---


class Left50Phase(Enum):
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Left50Controller:
    """From 0x51 (post-key): align y≈141, push LEFT into 0x50 play-ready."""

    max_frames: int = 2500
    phase: Left50Phase = Left50Phase.ALIGN
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Left50Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Left50Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Left50Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Left50Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_50
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(Left50Phase.DONE, "entered_0x50")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("LEFT"), "scroll_left")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if snap.screen not in (ROOM_L4_KEESE_KEY_51, ROOM_L4_VIRES_50):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if abs(snap.link_y - LEFT_51_Y) > KEY_61_EAST_Y_TOL:
            return FrameAction(
                nes_action("UP" if snap.link_y > LEFT_51_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(Left50Phase.PUSH, "push_left")
        return FrameAction(nes_action("LEFT"), "push_left")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_left_0x50",
            "target_room": f"0x{ROOM_L4_VIRES_50:02x}",
        }


def make_left_50_controller() -> Level4Left50Controller:
    return Level4Left50Controller()


# --- 0x61 KEY-RIGHT → 0x62 compass maze (rr-2ysf) ---


class KeyRight62Phase(Enum):
    CLEAR = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4KeyRight62Controller:
    """From 0x61 with ≥1 key: optional clear Vires, then KEY-RIGHT into 0x62.

    Live (rr-2ysf): hold RIGHT @ y≈141; keys 1→0; enter vestibule ~(16,141).
    """

    clear_vires: bool = True
    max_frames: int = 25000
    phase: KeyRight62Phase = KeyRight62Phase.CLEAR
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    keys_before: int | None = None
    notes: list[str] = field(default_factory=list)
    _clear: GenericDungeonRoomController | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.clear_vires:
            self._clear = GenericDungeonRoomController(ROOM_61_SPEC)
            self._clear.phase = DungeonPhase.FIGHT
        else:
            self.phase = KeyRight62Phase.ALIGN

    def _set_phase(self, phase: KeyRight62Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(KeyRight62Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is KeyRight62Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is KeyRight62Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_COMPASS_62
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(KeyRight62Phase.DONE, "entered_0x62")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("RIGHT"), "scroll_right")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is KeyRight62Phase.CLEAR:
            if snap.screen != ROOM_L4_VIRES_61:
                return self._fail(f"clear_wrong_room_0x{snap.screen:02x}")
            assert self._clear is not None
            live = any(
                1 <= o.slot <= 12
                and o.type_id in (VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE)
                for o in snap.objects
            )
            if not live and snap.room_all_dead >= 10:
                self.keys_before = snap.keys
                self._set_phase(KeyRight62Phase.ALIGN, "cleared_0x61")
            else:
                return self._clear.step(snap)

        if snap.screen not in (ROOM_L4_VIRES_61, ROOM_L4_COMPASS_62):
            return self._fail(f"wrong_room_0x{snap.screen:02x}")

        if self.keys_before is None:
            self.keys_before = snap.keys
        # keys may drop to 0 mid-push after the lock consumes; keep holding RIGHT.

        if abs(snap.link_y - KEY_61_EAST_Y) > KEY_61_EAST_Y_TOL:
            self._set_phase(KeyRight62Phase.ALIGN, "align_y")
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN"),
                "align_y",
            )
        self._set_phase(KeyRight62Phase.PUSH, "push_key_right")
        return FrameAction(nes_action("RIGHT"), "push_key_right")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "keys_before": self.keys_before,
            "segment": "level4_key_right_0x62",
            "target_room": f"0x{ROOM_L4_COMPASS_62:02x}",
            "clear": self._clear.report() if self._clear is not None else None,
        }


def make_key_right_62_controller(*, clear_vires: bool = True) -> Level4KeyRight62Controller:
    return Level4KeyRight62Controller(clear_vires=clear_vires)


# --- 0x71 empty entry → UP → 0x61 (rr-zchy) ---


class EntryUpPhase(Enum):
    SETTLE = auto()
    ALIGN = auto()
    PUSH = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4EntryUpController:
    """Empty 0x71 mouth: center x≈120, push UP into 0x61 play-ready."""

    max_frames: int = 2500
    phase: EntryUpPhase = EntryUpPhase.SETTLE
    frames: int = 0
    phase_frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: EntryUpPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(EntryUpPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is EntryUpPhase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is EntryUpPhase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_61
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(EntryUpPhase.DONE, "entered_0x61")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")
        if snap.transitioning or snap.mode in (4, 6, 7):
            return FrameAction(nes_action("UP"), "scroll_up")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if self.phase is EntryUpPhase.SETTLE:
            if snap.screen != ROOM_L4_ENTRY:
                return self._fail(f"wrong_room_0x{snap.screen:02x}")
            self._set_phase(EntryUpPhase.ALIGN, "align_x")

        if self.phase is EntryUpPhase.ALIGN:
            if abs(snap.link_x - 120) > 4:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                    "align_x",
                )
            self._set_phase(EntryUpPhase.PUSH, "push_up")
            return FrameAction(nes_action("UP"), "push_up")

        if self.phase is EntryUpPhase.PUSH:
            if abs(snap.link_x - 120) > 6:
                return FrameAction(
                    nes_action("RIGHT" if snap.link_x < 120 else "LEFT"),
                    "re_align_x",
                )
            return FrameAction(nes_action("UP"), "push_up")

        return FrameAction(nes_idle_action(), "idle")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "segment": "level4_entry_up_0x71",
            "target_room": f"0x{ROOM_L4_VIRES_61:02x}",
        }


def make_entry_up_controller() -> Level4EntryUpController:
    return Level4EntryUpController()
