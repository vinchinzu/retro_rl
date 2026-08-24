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
from zelda_i.bomb_wall_path import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon import (
    DungeonPhase,
    GenericDungeonRoomController,
)
from zelda_i.level4_dungeon import (
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


def planning_interior_report() -> dict:
    """Machine-readable live interior facts for probes / docs."""
    from zelda_i.level4_dungeon import (
        BOMB_21_NORTH_STAND,
        COMPASS_PICKUP_XY,
        GEL_SPLIT_OBJECT_TYPE,
        GLEEOK_FIREBALL_TYPE,
        GLEEOK_HEAD_OBJECT_TYPE,
        GLEEOK_OBJECT_TYPE,
        KEY_30_EAST_Y,
        KEY_40_PICKUP_XY,
        KEY_61_OPENS_TO,
        LADDER_60_PICKUP_XY,
        LEVEL4_COMPASS_BIT,
        LEVEL4_MAP_BIT,
        LEVEL4_TRIFORCE_BIT,
        MAP_21_PICKUP_XY,
        MAZE_31_EAST_X_MIN,
        MAZE_31_EAST_Y,
        PUSH_12_BLOCK_FROM,
        PUSH_12_BLOCK_TO,
        PUSH_12_DIR,
        PUSH_12_STAND,
        PUSH_32_DIR,
        PUSH_32_STAND,
        ROOM_ITEM_COMPASS,
        ROOM_ITEM_HEART_CONTAINER,
        ROOM_ITEM_MAP,
        ROOM_ITEM_SMALL_KEY,
        ROOM_ITEM_STEPLADDER,
        ROOM_L4_BUBBLES_00,
        ROOM_L4_EAST_31,
        ROOM_L4_EAST_32,
        ROOM_L4_GLEEOK_13,
        ROOM_L4_KEY_01,
        ROOM_L4_MANHANDLA_10,
        ROOM_L4_MAP_21,
        ROOM_L4_MID_11,
        ROOM_L4_NORTH_30,
        ROOM_L4_STEPLADDER,
        ROOM_L4_TRAPS_02,
        ROOM_L4_TRIFORCE,
        ROOM_L4_VIRES_12,
        ROOM_L4_WATER_NORTH_20,
        ROOM_L4_ZOLS_40,
        STAIRS_32_APPROACH,
    )
    from zelda_i.level4_maze_path import (
        MAP_21_HOLD,
        MAP_21_SAMPLE_PATH,
        MAZE_50_HOLD,
        MAZE_50_LONG_UP,
        MAZE_50_TO_NORTH,
        MAZE_62_RETURN_WEST,
        MAZE_62_TO_COMPASS,
        MAZE_IN_HOLD,
        MAZE_OUT_HOLD,
        PATH_12_TO_GLEEOK,
        RIGHT_12_HOLD,
    )
    from zelda_i.level4_stepladder import (
        EXIT_60_HOLD,
        MAZE_31_HOLD,
        MAZE_60_HOLD,
        MAZE_60_TO_LADDER,
        POST_LADDER_ITEM_SETTLE,
        WEST_31_SAMPLE_PATH,
    )
    return {
        "level": LEVEL4,
        "bead": "rr-5lu",
        "tip": "rr-rvae",
        "track": "assisted_map_first_pass",
        "status": "gleeok_tf08_dual_green",
        "date": "2026-08-10",
        "entry_room": hex(ROOM_L4_ENTRY),
        "live_graph": {
            hex(ROOM_L4_ENTRY): {"UP": hex(ROOM_L4_VIRES_61)},
            hex(ROOM_L4_VIRES_61): {
                "BOMB_UP": hex(ROOM_L4_KEESE_KEY_51),
                "KEY_RIGHT": hex(ROOM_L4_COMPASS_62),
                "RIGHT_reenter": hex(ROOM_L4_COMPASS_62),
                "DOWN": hex(ROOM_L4_ENTRY),
                "enemies": {"0x12": 3, "split": "0x1c"},
            },
            hex(ROOM_L4_KEESE_KEY_51): {
                "LEFT": hex(ROOM_L4_VIRES_50),
                "DOWN": hex(ROOM_L4_VIRES_61),
                "UP": "sealed",
                "RIGHT": "sealed",
                "enemies": {"0x1b": 8},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
                "note": "UP/RIGHT not key doors (poke keys no consume)",
            },
            hex(ROOM_L4_VIRES_50): {
                "enemies": {"0x12": 5},
                "RIGHT": hex(ROOM_L4_KEESE_KEY_51),
                "UP_scripted": hex(ROOM_L4_ZOLS_40),
                "note": "north via MAZE_50_TO_NORTH hold6 + long UP (rr-xc3x)",
            },
            hex(ROOM_L4_COMPASS_62): {
                "enemies": {"0x12": 5},
                "room_item": hex(ROOM_ITEM_COMPASS),
                "LEFT": hex(ROOM_L4_VIRES_61),
                "compass_bit": hex(LEVEL4_COMPASS_BIT),
                "pickup_xy": list(COMPASS_PICKUP_XY),
                "note": "dark_maze_compass_live_return_west_no_bomb_exit",
            },
            hex(ROOM_L4_ZOLS_40): {
                "enemies": {"0x13": 5, "split": "0x14"},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
                "key_pickup_xy": list(KEY_40_PICKUP_XY),
                "DOWN": hex(ROOM_L4_VIRES_50),
                "UP": hex(ROOM_L4_NORTH_30),
                "LEFT": "sealed",
                "RIGHT": "sealed",
                "note": "first outside early component; clear+key then free UP (rr-q8eq)",
            },
            hex(ROOM_L4_NORTH_30): {
                "enemies": {"0x12": 3, "0x2b": 2},
                "DOWN": hex(ROOM_L4_ZOLS_40),
                "KEY_RIGHT": hex(ROOM_L4_EAST_31),
                "UP": "sealed",
                "LEFT": "sealed",
                "RIGHT_free": "sealed",
                "note": (
                    "clear Vires ignore invuln 0x2b (rr-n1wn); walkable y≥128; "
                    "KEY-RIGHT@y141 → 0x31 (5× Vire)"
                ),
            },
            hex(ROOM_L4_EAST_31): {
                "enemies": {"0x12": 5},
                "LEFT": hex(ROOM_L4_NORTH_30),
                "RIGHT_after_clear": hex(ROOM_L4_EAST_32),
                "UP": "sealed",
                "note": (
                    "maze interior (rr-resv); clear opens doors 2→3 (R free); "
                    "hold4 BFS east band → RIGHT → 0x32; N free sealed"
                ),
            },
            hex(ROOM_L4_EAST_32): {
                "enemies": {"0x13": 2, "0x17": 2, "0x2b": 2, "0x68": 1},
                "LEFT": hex(ROOM_L4_EAST_31),
                "push_left_stairs": hex(ROOM_L4_STEPLADDER),
                "note": (
                    "live free-RIGHT of cleared 0x31 (rr-resv/rr-tib8); "
                    "clear Zol+LikeLike (ignore 0x2b/0x68); push left block "
                    "→ mode-9 0x60 Stepladder"
                ),
            },
            hex(ROOM_L4_STEPLADDER): {
                "mode": 9,
                "room_item": hex(ROOM_ITEM_STEPLADDER),
                "pickup_xy": list(LADDER_60_PICKUP_XY),
                "enemies": {"0x1b": 4},
                "note": "stairs basement under 0x32; ADDR_LADDER on touch (rr-tib8)",
                "exit": {
                    "to": hex(ROOM_L4_EAST_32),
                    "hold": EXIT_60_HOLD,
                    "settle_idle": POST_LADDER_ITEM_SETTLE,
                    "note": (
                        "continuous: item freeze 150f; reverse-dock waypoints "
                        "(no BFS) → 0x32 play leftover (192,189); isolated "
                        "EXIT_60_SAMPLE_PATH is not this tape"
                    ),
                },
            },
            "post_ladder_0x32": {
                "checkpoint": "Level4PostLadder",
                "ladder": 1,
                "LEFT_bfs": hex(ROOM_L4_EAST_31),
                "note": (
                    "rr-05fz: free LEFT around pushed 0x68 block (WEST_31_SAMPLE_PATH); "
                    "backtrack 0x31→0x30; KEY-UP needs keys≥1 (rr-rvae map)"
                ),
            },
            hex(ROOM_L4_NORTH_30) + "_post_ladder": {
                "KEY_UP_with_ladder_key": hex(ROOM_L4_WATER_NORTH_20),
                "note": (
                    "rr-rvae: water tiles walkable with ladder; KEY-UP consumes 1 key → 0x20; "
                    "free N without key still sealed; KEY-UP 0x31 → 0x21 south pocket isolated"
                ),
            },
            hex(ROOM_L4_WATER_NORTH_20): {
                "enemies": {"0x12": 5},
                "DOWN": hex(ROOM_L4_NORTH_30),
                "UP": hex(0x10),
                "RIGHT_after_clear": hex(ROOM_L4_MAP_21),
                "note": (
                    "clear Vires (+split 0x1c); door bit R may stay 0 — push x≈208 y≈141 RIGHT → 0x21"
                ),
            },
            hex(ROOM_L4_MAP_21): {
                "enemies": {"0x15": 5},
                "room_item": hex(ROOM_ITEM_MAP),
                "LEFT": hex(ROOM_L4_WATER_NORTH_20),
                "BOMB_UP": hex(ROOM_L4_MID_11),
                "map_bit": hex(LEVEL4_MAP_BIT),
                "pickup_xy": list(MAP_21_PICKUP_XY),
                "bomb_up_stand": list(BOMB_21_NORTH_STAND),
                "note": (
                    "rr-rvae assisted dual 2/2: gel thrash expands maze then hold6 BFS "
                    "MAP_21_SAMPLE_PATH → ADDR_MAP|0x08 @~(208,181); south KEY-UP pocket "
                    "from 0x31 is wall-isolated (x≤176); BOMB_UP@(120,105) → 0x11"
                ),
            },
            hex(ROOM_L4_MID_11): {
                "enemies": {"0x35": "multi"},
                "DOWN": hex(ROOM_L4_MAP_21),
                "BOMB_UP": hex(ROOM_L4_KEY_01),
                "RIGHT": hex(ROOM_L4_VIRES_12),
                "LEFT": hex(ROOM_L4_MANHANDLA_10),
                "note": (
                    "type 0x35 cluster stays live; north is BOMB_UP@(120,105) "
                    "not free (v1 leftover (120,93)); RIGHT 0x12 / LEFT 0x10"
                ),
            },
            hex(ROOM_L4_KEY_01): {
                "enemies": {"0x1b": 8},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
                "DOWN": hex(ROOM_L4_MID_11),
                "LEFT": hex(ROOM_L4_BUBBLES_00),
                "RIGHT": hex(ROOM_L4_TRAPS_02),
                "note": (
                    "rr-rvae: Keese clear + RoomItemId 0x19 → keys≥1 natural key for "
                    "map KEY-UP residual; free links to 0x00/0x02"
                ),
            },
            hex(ROOM_L4_VIRES_12): {
                "enemies": {"0x12": 5, "0x68": 1},
                "LEFT": hex(ROOM_L4_MID_11),
                "UP": hex(ROOM_L4_TRAPS_02),
                "RIGHT_after_push_block": hex(ROOM_L4_GLEEOK_13),
                "push_block": {
                    "stand": list(PUSH_12_STAND),
                    "dir": PUSH_12_DIR,
                    "from": list(PUSH_12_BLOCK_FROM),
                    "to": list(PUSH_12_BLOCK_TO),
                    "doors_after": 3,
                },
                "path_hold": RIGHT_12_HOLD,
                "path_len": len(PATH_12_TO_GLEEOK),
                "note": (
                    "rr-rvae dual-green: clear 5× Vire; push 0x68 LEFT "
                    "(96,144)→(80,144) opens R bit doors 2→3; hold4 "
                    "PATH_12_TO_GLEEOK plen31 → 0x13 (naive y141 hold-RIGHT fails)"
                ),
            },
            hex(ROOM_L4_TRAPS_02): {
                "enemies": {"0x49": 6},
                "DOWN": hex(ROOM_L4_VIRES_12),
                "note": "blade traps only; no other free exits (rr-rvae)",
            },
            hex(ROOM_L4_GLEEOK_13): {
                "enemies": {"0x43": "gleeok", "head": "0x46", "fireball": "0x56"},
                "room_item": hex(ROOM_ITEM_HEART_CONTAINER),
                "LEFT": hex(ROOM_L4_VIRES_12),
                "UP_after_clear": hex(ROOM_L4_TRIFORCE),
                "checkpoint": "Level4GleeokEnter",
                "checkpoint_complete": "Level4Complete",
                "note": (
                    "rr-rvae dual-green: melee Gleeok 0x43 HP≈160 + head 0x46; "
                    "HC 0x1a; UP → 0x03 TF bit 0x08 (~4.3k f dual)"
                ),
            },
            hex(ROOM_L4_TRIFORCE): {
                "DOWN": hex(ROOM_L4_GLEEOK_13),
                "tf_bit": hex(LEVEL4_TRIFORCE_BIT),
                "checkpoint": "Level4Complete",
                "note": "rr-rvae dual-green TF pickup @~(120,141) → mode 18",
            },
            hex(ROOM_L4_MANHANDLA_10): {
                "enemies": {"0x3c": "manhandla"},
                "DOWN": hex(ROOM_L4_WATER_NORTH_20),
                "UP": hex(ROOM_L4_BUBBLES_00),
                "RIGHT": hex(ROOM_L4_MID_11),
                "note": "optional side boss; free UP → 0x00 bubbles dead-end",
            },
            hex(ROOM_L4_BUBBLES_00): {
                "enemies": {"0x40": 2, "0x4e": 1},
                "DOWN": hex(ROOM_L4_MANHANDLA_10),
                "note": "dead-end north of Manhandla (rr-rvae)",
            },
        },
        "post_compass": {
            "bead": "rr-o0nn",
            "expand": "rr-tib8",
            "start": "Level4Compass",
            "early_component": [
                hex(ROOM_L4_ENTRY),
                hex(ROOM_L4_VIRES_61),
                hex(ROOM_L4_KEESE_KEY_51),
                hex(ROOM_L4_VIRES_50),
                hex(ROOM_L4_COMPASS_62),
            ],
            "first_outside": hex(ROOM_L4_ZOLS_40),
            "next_outside": hex(ROOM_L4_NORTH_30),
            "keys_at_compass": 0,
            "ladder": 1,
            "evidence": [
                "recordings/l4_xc3x_breakthrough.json",
                "recordings/l4_q8eq_40_dense_bfs.json",
                "recordings/l4_q8eq_key40_key_40.json",
                "recordings/l4_n1wn_clear30_clear_30.json",
                "recordings/l4_resv_31_bfs.json",
                "recordings/l4_resv_room32_recon.json",
                "recordings/l4_tib8_clear32_clear_32.json",
                "recordings/l4_tib8_stepladder_stepladder.json",
            ],
            "blocked": [
                "0x51 UP/RIGHT sealed (not key)",
                "0x62 bomb exits none",
                "0x40 LEFT/RIGHT sealed",
                "0x31 N free sealed (maze)",
                "0x32 free N/E/W sealed (only LEFT + stairs)",
                "no Vire key-farm drops (8 cycles)",
            ],
            "opened": [
                "0x50 north scripted → 0x40 (Zols + key 0x19)",
                "0x40 clear+key → free UP → 0x30",
                "0x30 clear Vires (ignore 0x2b)",
                "0x30 KEY-RIGHT @y141 → 0x31 (5× Vire)",
                "0x31 clear Vires → free RIGHT → 0x32",
                "0x32 clear Zol+LikeLike → push left → 0x60 ADDR_LADDER",
            ],
        },
        "bomb_61_north": {
            "stand": list(BOMB_61_NORTH_STAND),
            "face": BOMB_61_NORTH_FACE,
            "opens_to": hex(BOMB_61_OPENS_TO),
        },
        "key_61_east": {
            "y": KEY_61_EAST_Y,
            "opens_to": hex(KEY_61_OPENS_TO),
            "key_cost": 1,
        },
        "maze_62": {
            "in_hold": MAZE_IN_HOLD,
            "out_hold": MAZE_OUT_HOLD,
            "to_compass": list(MAZE_62_TO_COMPASS),
            "return_west": list(MAZE_62_RETURN_WEST),
            "pickup_xy": list(COMPASS_PICKUP_XY),
        },
        "maze_50_north": {
            "hold": MAZE_50_HOLD,
            "long_up": MAZE_50_LONG_UP,
            "path": list(MAZE_50_TO_NORTH),
            "opens_to": hex(ROOM_L4_ZOLS_40),
        },
        "segments": {
            "entry_up": "rr-zchy",
            "clear_vires_61": "rr-yr77",
            "bomb_up_51": "rr-h278",
            "keese_key_51": "rr-wqdu",
            "clear_50": "rr-2ysf",
            "key_right_62": "rr-2ysf",
            "clear_62": "rr-2ysf",
            "compass_62": "rr-9so0",
            "north_40": "rr-xc3x",
            "key_40": "rr-q8eq",
            "north_30": "rr-q8eq",
            "clear_30": "rr-n1wn",
            "key_right_31": "rr-n1wn",
            "clear_31": "rr-resv",
            "east_32": "rr-resv",
            "clear_32": "rr-tib8",
            "stepladder": "rr-tib8",
            "stepladder_path": "rr-tib8",
        },
        "key_40": {
            "pickup_xy": list(KEY_40_PICKUP_XY),
            "gel_split": hex(GEL_SPLIT_OBJECT_TYPE),
            "opens_north": hex(ROOM_L4_NORTH_30),
        },
        "clear_30": {
            "enemies": {"0x12": 3, "ignore": "0x2b"},
            "settle_all_dead": 0,
            "walkable_y_min": 128,
            "checkpoint": "Level4Room30Cleared",
        },
        "key_right_31": {
            "y": KEY_30_EAST_Y,
            "opens_to": hex(ROOM_L4_EAST_31),
            "key_cost": 1,
            "checkpoint": "Level4Room31",
        },
        "clear_31": {
            "enemies": {"0x12": 5},
            "settle_all_dead": 0,
            "doors_after_clear": 3,
            "checkpoint": "Level4Room31Cleared",
        },
        "east_32": {
            "hold": MAZE_31_HOLD,
            "east_x_min": MAZE_31_EAST_X_MIN,
            "east_y": MAZE_31_EAST_Y,
            "opens_to": hex(ROOM_L4_EAST_32),
            "checkpoint": "Level4Room32",
        },
        "clear_32": {
            "enemies": {"0x13": 2, "0x17": 2, "ignore": ["0x2b", "0x68"]},
            "settle_all_dead": 0,
            "checkpoint": "Level4Room32Cleared",
        },
        "stepladder": {
            "push_stand": list(PUSH_32_STAND),
            "push_dir": PUSH_32_DIR,
            "stairs_approach": list(STAIRS_32_APPROACH),
            "stairs_room": hex(ROOM_L4_STEPLADDER),
            "mode": 9,
            "room_item": hex(ROOM_ITEM_STEPLADDER),
            "pickup_xy": list(LADDER_60_PICKUP_XY),
            "path_hold": MAZE_60_HOLD,
            "path_len": len(MAZE_60_TO_LADDER),
            "checkpoint": "Level4Stepladder",
        },
        "map_21": {
            "bead": "rr-rvae",
            "room": hex(ROOM_L4_MAP_21),
            "room_item": hex(ROOM_ITEM_MAP),
            "map_bit": hex(LEVEL4_MAP_BIT),
            "pickup_xy": list(MAP_21_PICKUP_XY),
            "hold": MAP_21_HOLD,
            "sample_path": list(MAP_21_SAMPLE_PATH),
            "via": [hex(ROOM_L4_NORTH_30), hex(ROOM_L4_WATER_NORTH_20)],
            "key_cost": 1,
            "checkpoint": "Level4Map",
            "track": "assisted_first_pass",
            "evidence": "recordings/l4_rvae_map_final.json",
        },
        "right_13": {
            "bead": "rr-rvae",
            "from": hex(ROOM_L4_VIRES_12),
            "to": hex(ROOM_L4_GLEEOK_13),
            "push_stand": list(PUSH_12_STAND),
            "push_dir": PUSH_12_DIR,
            "block_from": list(PUSH_12_BLOCK_FROM),
            "block_to": list(PUSH_12_BLOCK_TO),
            "path_hold": RIGHT_12_HOLD,
            "path": list(PATH_12_TO_GLEEOK),
            "path_len": len(PATH_12_TO_GLEEOK),
            "checkpoint_cleared": "Level4Room12Cleared",
            "checkpoint_enter": "Level4GleeokEnter",
            "track": "assisted_first_pass",
            "dual_green": True,
            "evidence": "recordings/l4_rvae_right13_dual.json",
        },
        "gleeok_tf": {
            "bead": "rr-rvae",
            "from": "Level4GleeokEnter",
            "room": hex(ROOM_L4_GLEEOK_13),
            "boss_type": hex(GLEEOK_OBJECT_TYPE),
            "head_type": hex(GLEEOK_HEAD_OBJECT_TYPE),
            "fireball": hex(GLEEOK_FIREBALL_TYPE),
            "hc": hex(ROOM_ITEM_HEART_CONTAINER),
            "tf_room": hex(ROOM_L4_TRIFORCE),
            "tf_bit": hex(LEVEL4_TRIFORCE_BIT),
            "policy": "melee_A_prefer_heads_then_body",
            "checkpoint": "Level4Complete",
            "track": "assisted_first_pass",
            "dual_green": True,
            "evidence": "recordings/l4_rvae_gleeok_tf_dual.json",
            "runner": "scripts/run_level4_gleeok.py",
            "module": "level4_boss_combat.Level4GleeokFightController",
        },
        "not_yet": [
            "rr-05fz CLOSED: skip-compass NaturalKey checkpoint + map_21 --no-key-poke dual",
            "rr-05fz CLOSED: continuous PostLadderNaturalKey → TF dual (assisted; not Clean)",
            "Clean promote",
        ],
    }


