"""Level 4 (Snake) dungeon room specs and live anchors.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Interior recon (assisted/pure, rr-5lu / rr-2ysf 2026-08-09/10) —
**no walkthrough hardcodes** beyond live IDs.

Live path from ``Level4Entrance`` (room **0x71**)::

    0x71 entry (empty combat) --UP@x≈120--> 0x61
    0x61: 3× Vire type ``0x12`` (HP 64) → wooden sword splits to type ``0x1c``
    0x61 --BOMB_UP stand≈(120,105) face UP--> 0x51
    0x51: 8× Keese type ``0x1b`` (TYPE-only) + RoomItemId ``0x19`` key
    0x51 --LEFT @ y≈141--> 0x50 (5× Vire ``0x12``)  **dead-end pocket**
    0x51 --DOWN @ x≈120--> 0x61
    0x61 --KEY-RIGHT @ y≈141 (keys 1→0)--> 0x62
    0x62: 5× Vire + RoomItemId ``0x16`` Compass (dark maze)
    0x62 --maze compass + return LEFT--> 0x61 (ADDR_COMPASS bit 0x08)
    **Post-compass (rr-o0nn live):** component closed at
    {0x71, 0x61, 0x51, 0x50, 0x62}. From Level4Compass: free/BOMB UP→0x51,
    RIGHT re-enter 0x62 (no key), LEFT 0x51→0x50. 0x51 UP+RIGHT sealed
    (key poke does not consume). 0x50 bomb denser N/no new exit. 0x62 bomb
    exits none. No Vire key-farm drops. ADDR_LADDER residual.

Not Clean STATUS. Stepladder / Gleeok / TF ``0x08`` still residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.bomb_wall_path import BOMB_N_WAIT_BLAST, BombWallController
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    KEESE_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    register_room_spec,
)
from zelda_i.level4_overworld import LEVEL4, LEVEL4_ENTRY_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

# --- Live L4 room anchors (rr-5lu / rr-2ysf 2026-08-09/10) ---
ROOM_L4_ENTRY = LEVEL4_ENTRY_ROOM  # 0x71 — empty combat mouth
ROOM_L4_VIRES_61 = 0x61  # north of entry; 3× Vire 0x12
ROOM_L4_KEESE_KEY_51 = 0x51  # bomb-N of 0x61; 8× Keese + key 0x19
ROOM_L4_VIRES_50 = 0x50  # west of 0x51; 5× Vire 0x12 (dead-end pocket)
ROOM_L4_COMPASS_62 = 0x62  # KEY-RIGHT of 0x61; 5× Vire + compass 0x16 dark maze

VIRE_OBJECT_TYPE = 0x12  # live on 0x61/0x50/0x62; HP 64; sword splits → 0x1c
VIRE_SPLIT_KEESE_TYPE = 0x1C  # live split residual from Vire (not standard 0x1B)
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_COMPASS = 0x16  # live room item on 0x62
ROOM_ITEM_NONE = 0x03
LEVEL4_COMPASS_BIT = 0x08  # ADDR_COMPASS bit for dungeon level 4

# Bomb-north wall 0x61 → 0x51 (live stand ≈ y105, face UP).
BOMB_61_NORTH_STAND = (120, 105)
BOMB_61_NORTH_FACE = "UP"
BOMB_61_OPENS_TO = ROOM_L4_KEESE_KEY_51

# Key-east door 0x61 → 0x62 (live: y≈141 hold RIGHT; keys 1→0).
KEY_61_EAST_Y = 141
KEY_61_EAST_Y_TOL = 4
KEY_61_OPENS_TO = ROOM_L4_COMPASS_62

# Free LEFT 0x51 → 0x50.
LEFT_51_Y = 141

# Dark-maze 0x62 compass (rr-9so0 live BFS from Level4Room62Cleared).
# Hold each token for MAZE_IN_HOLD / MAZE_OUT_HOLD frames. Pickup ~ (136,132).
# After compass, corridor path back to west vestibule then LEFT → 0x61 play.
MAZE_IN_HOLD = 6
MAZE_OUT_HOLD = 4
MAZE_62_TO_COMPASS: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "RIGHT",
    "UP",
    "UP",
    "UP",
    "RIGHT",
    "UP",
    "UP",
    "UP",
)
MAZE_62_RETURN_WEST: tuple[str, ...] = (
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "DOWN",
    "DOWN",
    "DOWN",
    "LEFT",
    "DOWN",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "LEFT",
    "UP",
    "LEFT",
    "UP",
    "UP",
    "UP",
    "LEFT",
)
COMPASS_PICKUP_XY = (136, 132)

_PATROL_MID: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
)


def level4_room_ready(snap: ZeldaSnapshot, room: int) -> bool:
    return (
        snap.level == LEVEL4
        and snap.mode == PLAY_MODE
        and snap.screen == room
        and not snap.transitioning
    )


def level4_entry_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_ENTRY)


def level4_room_61_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_VIRES_61)


def level4_room_51_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_KEESE_KEY_51)


def level4_room_61_cleared(ram: np.ndarray) -> bool:
    """0x61 with no live Vire/split and RoomAllDead settle."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_61):
        return False
    live = ROOM_61_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_room_51_key_success(ram: np.ndarray) -> bool:
    """Keese clear + at least one key collected in room 0x51."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_KEESE_KEY_51):
        return False
    keese = ROOM_51_SPEC.live_enemies(snap)
    return len(keese) == 0 and snap.room_all_dead >= 20 and snap.keys >= 1


def level4_room_50_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_VIRES_50)


def level4_room_50_cleared(ram: np.ndarray) -> bool:
    """0x50 with no live Vire/split and RoomAllDead settle."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_VIRES_50):
        return False
    live = ROOM_50_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_room_62_ready(ram: np.ndarray) -> bool:
    return level4_room_ready(read_snapshot(ram), ROOM_L4_COMPASS_62)


def level4_room_62_cleared(ram: np.ndarray) -> bool:
    """0x62 Vires cleared (compass pickup residual)."""
    snap = read_snapshot(ram)
    if not level4_room_ready(snap, ROOM_L4_COMPASS_62):
        return False
    live = ROOM_62_SPEC.live_enemies(snap)
    return len(live) == 0 and snap.room_all_dead >= 20


def level4_compass_collected(ram: np.ndarray) -> bool:
    """L4 compass inventory bit set (ADDR_COMPASS & 0x08)."""
    snap = read_snapshot(ram)
    return bool(snap.compass & LEVEL4_COMPASS_BIT)


def level4_compass_route_success(ram: np.ndarray) -> bool:
    """Compass bit set and back on 0x61 play-ready (maze return complete)."""
    snap = read_snapshot(ram)
    return (
        bool(snap.compass & LEVEL4_COMPASS_BIT)
        and level4_room_ready(snap, ROOM_L4_VIRES_61)
    )


# --- Specs (assisted geometry; not Clean promote) ---
ROOM_71_SPEC = DungeonRoomSpec(
    spec_id="level4_room71_entry",
    source_room=ROOM_L4_ENTRY,
    room_id=ROOM_L4_ENTRY,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(),
    expected_enemy_count=0,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(patrol=((120, 150),)),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(DoorRoute("UP", ((120, 150), (120, 93))),),
    max_frames=2000,
    level=LEVEL4,
)

ROOM_61_SPEC = DungeonRoomSpec(
    spec_id="level4_room61_vires",
    source_room=ROOM_L4_ENTRY,
    room_id=ROOM_L4_VIRES_61,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=3,  # Vires; split increases count mid-fight
    alive_rule=AliveRule.TYPE_AND_HP,  # Vire uses HP
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),  # split 0x1c HP stays 0
    object_slot_max=12,  # splits land in slots 10–11+
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(),  # bomb wall, not free door
    max_frames=16000,
    level=LEVEL4,
)

ROOM_51_SPEC = DungeonRoomSpec(
    spec_id="level4_room51_keese_key",
    source_room=ROOM_L4_VIRES_61,
    room_id=ROOM_L4_KEESE_KEY_51,
    entry=DoorRoute("UP", ((120, 205), (120, 150))),
    enemy_types=(KEESE_OBJECT_TYPE,),
    expected_enemy_count=8,
    alive_rule=AliveRule.TYPE,  # Keese HP stays 0 while alive
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=56,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        # Live pickup ~ (136,149) after clear; dense mid-room hunt.
        target=(136, 149),
        waypoints=(
            (128, 141),
            (136, 149),
            (120, 149),
            (144, 141),
            (112, 141),
            (128, 157),
            (128, 125),
            (96, 157),
            (160, 157),
            (96, 125),
            (160, 125),
            (80, 141),
            (176, 141),
            (120, 173),
            (120, 109),
        ),
    ),
    room_item_id=ROOM_ITEM_SMALL_KEY,
    exit_routes=(DoorRoute("LEFT", ((120, 141), (40, 141))),),
    max_frames=16000,
    level=LEVEL4,
)

ROOM_50_SPEC = DungeonRoomSpec(
    spec_id="level4_room50_vires",
    source_room=ROOM_L4_KEESE_KEY_51,
    room_id=ROOM_L4_VIRES_50,
    entry=DoorRoute("LEFT", ((224, 141), (180, 141))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_NONE,
    exit_routes=(DoorRoute("RIGHT", ((120, 141), (220, 141))),),
    max_frames=20000,
    level=LEVEL4,
)

ROOM_62_SPEC = DungeonRoomSpec(
    spec_id="level4_room62_vires_compass",
    source_room=ROOM_L4_VIRES_61,
    room_id=ROOM_L4_COMPASS_62,
    entry=DoorRoute("RIGHT", ((16, 141), (48, 141))),
    enemy_types=(VIRE_OBJECT_TYPE, VIRE_SPLIT_KEESE_TYPE),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    type_only_enemy_types=(VIRE_SPLIT_KEESE_TYPE,),
    object_slot_max=12,
    combat=CombatTuning(
        patrol=_PATROL_MID,
        engage_distance=72,
        attack_phase=0,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    # Compass is a bitfield — clear-only here; pickup residual (rr-2ysf maze).
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=ROOM_ITEM_COMPASS,
    exit_routes=(DoorRoute("LEFT", ((48, 141), (16, 141))),),
    max_frames=20000,
    level=LEVEL4,
)

register_room_spec(ROOM_71_SPEC)
register_room_spec(ROOM_61_SPEC)
register_room_spec(ROOM_51_SPEC)
register_room_spec(ROOM_50_SPEC)
register_room_spec(ROOM_62_SPEC)


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
    """Clear 0x50 pocket Vires (dead-end; stepladder is KEY-RIGHT 0x62)."""
    return GenericDungeonRoomController(ROOM_50_SPEC)


def make_room_62_clear_controller() -> GenericDungeonRoomController:
    """Clear 0x62 Vires (compass maze; pickup / exits residual)."""
    return GenericDungeonRoomController(ROOM_62_SPEC)


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


# --- 0x62 dark maze: compass + return LEFT → 0x61 (rr-9so0) ---


class Compass62Phase(Enum):
    MAZE_IN = auto()
    MAZE_OUT = auto()
    EXIT_LEFT = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level4Compass62Controller:
    """From cleared 0x62: maze to compass pickup, return west, LEFT to 0x61.

    Live (rr-9so0): hold scripted dirs (BFS) — open seek fails on maze walls.
    Success: ``ADDR_COMPASS & 0x08`` and play-ready on 0x61.
    """

    max_frames: int = 12000
    phase: Compass62Phase = Compass62Phase.MAZE_IN
    frames: int = 0
    phase_frames: int = 0
    path_index: int = 0
    hold_left: int = 0
    success: bool = False
    compass_at_frame: int | None = None
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Compass62Phase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            self.phase_frames = 0
            self.path_index = 0
            self.hold_left = 0
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Compass62Phase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1

        if self.phase is Compass62Phase.DONE:
            return FrameAction(nes_idle_action(), "done")
        if self.phase is Compass62Phase.FAILED:
            return FrameAction(nes_idle_action(), "failed")
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")

        if (
            bool(snap.compass & LEVEL4_COMPASS_BIT)
            and snap.level == LEVEL4
            and snap.screen == ROOM_L4_VIRES_61
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self._set_phase(Compass62Phase.DONE, "compass_and_0x61")
            return FrameAction(nes_idle_action(), "done")

        if snap.level != LEVEL4:
            return FrameAction(nes_idle_action(), "wait_level4")

        # Scroll through door transitions while exiting west.
        if snap.transitioning or snap.mode in (4, 6, 7):
            if self.phase is Compass62Phase.EXIT_LEFT or (
                snap.screen in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61)
            ):
                return FrameAction(nes_action("LEFT"), "scroll_left")
            return FrameAction(nes_idle_action(), f"wait_scroll_{snap.mode}")

        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if bool(snap.compass & LEVEL4_COMPASS_BIT) and self.compass_at_frame is None:
            self.compass_at_frame = self.frames
            self.notes.append(f"compass_bit_f{self.frames}")

        if self.phase is Compass62Phase.MAZE_IN:
            if snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_in_wrong_room_0x{snap.screen:02x}")
            if bool(snap.compass & LEVEL4_COMPASS_BIT):
                self._set_phase(Compass62Phase.MAZE_OUT, "got_compass")
                # fall through to MAZE_OUT this frame
            else:
                if self.path_index >= len(MAZE_62_TO_COMPASS):
                    return self._fail("maze_in_path_exhausted_no_compass")
                direction = MAZE_62_TO_COMPASS[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_IN_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_in_{direction}")

        if self.phase is Compass62Phase.MAZE_OUT:
            if snap.screen == ROOM_L4_VIRES_61:
                self._set_phase(Compass62Phase.EXIT_LEFT, "already_0x61")
            elif snap.screen != ROOM_L4_COMPASS_62:
                return self._fail(f"maze_out_wrong_room_0x{snap.screen:02x}")
            elif self.path_index >= len(MAZE_62_RETURN_WEST):
                self._set_phase(Compass62Phase.EXIT_LEFT, "return_path_done")
            else:
                direction = MAZE_62_RETURN_WEST[self.path_index]
                self.hold_left += 1
                if self.hold_left >= MAZE_OUT_HOLD:
                    self.path_index += 1
                    self.hold_left = 0
                return FrameAction(nes_action(direction), f"maze_out_{direction}")

        # EXIT_LEFT: push west door / finish settle on 0x61
        if snap.screen == ROOM_L4_VIRES_61 and snap.mode == PLAY_MODE:
            if bool(snap.compass & LEVEL4_COMPASS_BIT) and not snap.transitioning:
                self.success = True
                self._set_phase(Compass62Phase.DONE, "settled_0x61")
                return FrameAction(nes_idle_action(), "done")
        if snap.screen not in (ROOM_L4_COMPASS_62, ROOM_L4_VIRES_61):
            return self._fail(f"exit_wrong_room_0x{snap.screen:02x}")
        # Align y≈141 when still in 0x62 vestibule, then LEFT.
        if snap.screen == ROOM_L4_COMPASS_62 and abs(snap.link_y - KEY_61_EAST_Y) > 8:
            return FrameAction(
                nes_action("UP" if snap.link_y > KEY_61_EAST_Y else "DOWN"),
                "align_exit_y",
            )
        return FrameAction(nes_action("LEFT"), "exit_left")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "notes": list(self.notes),
            "compass_at_frame": self.compass_at_frame,
            "path_index": self.path_index,
            "segment": "level4_compass_0x62",
            "maze_in": list(MAZE_62_TO_COMPASS),
            "maze_out": list(MAZE_62_RETURN_WEST),
            "pickup_xy": list(COMPASS_PICKUP_XY),
        }


def make_compass_62_controller() -> Level4Compass62Controller:
    return Level4Compass62Controller()


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
    return {
        "level": LEVEL4,
        "bead": "rr-5lu",
        "tip": "rr-o0nn",
        "track": "assisted",
        "status": "interior_compass_live_stepladder_residual",
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
                "note": "dead_end_pocket_no_bomb_exit",
            },
            hex(ROOM_L4_COMPASS_62): {
                "enemies": {"0x12": 5},
                "room_item": hex(ROOM_ITEM_COMPASS),
                "LEFT": hex(ROOM_L4_VIRES_61),
                "compass_bit": hex(LEVEL4_COMPASS_BIT),
                "pickup_xy": list(COMPASS_PICKUP_XY),
                "note": "dark_maze_compass_live_return_west_no_bomb_exit",
            },
        },
        "post_compass": {
            "bead": "rr-o0nn",
            "start": "Level4Compass",
            "component": [
                hex(ROOM_L4_ENTRY),
                hex(ROOM_L4_VIRES_61),
                hex(ROOM_L4_KEESE_KEY_51),
                hex(ROOM_L4_VIRES_50),
                hex(ROOM_L4_COMPASS_62),
            ],
            "keys_at_compass": 0,
            "ladder": 0,
            "evidence": [
                "recordings/l4_o0nn_focus.json",
                "recordings/l4_o0nn_prod.json",
                "recordings/l4_o0nn_bombs.json",
                "recordings/l4_o0nn_keypoke.json",
            ],
            "blocked": [
                "0x51 UP/RIGHT sealed (not key)",
                "0x50 bomb denser N no open",
                "0x62 bomb exits none",
                "no Vire key-farm drops (8 cycles)",
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
        "segments": {
            "entry_up": "rr-zchy",
            "clear_vires_61": "rr-yr77",
            "bomb_up_51": "rr-h278",
            "keese_key_51": "rr-wqdu",
            "clear_50": "rr-2ysf",
            "key_right_62": "rr-2ysf",
            "clear_62": "rr-2ysf",
            "compass_62": "rr-9so0",
            "stepladder_path": "rr-o0nn",
        },
        "not_yet": [
            "stepladder room / ADDR_LADDER",
            "room outside closed post-compass component",
            "Gleeok boss type",
            "TF bit 0x08 natural",
            "Clean promote",
        ],
    }


# Re-export for scripts that want phase types
__all__ = [
    "BOMB_61_NORTH_FACE",
    "BOMB_61_NORTH_STAND",
    "BOMB_61_OPENS_TO",
    "BombWall61North",
    "COMPASS_PICKUP_XY",
    "Compass62Phase",
    "DungeonPhase",
    "EntryUpPhase",
    "KEY_61_EAST_Y",
    "KEY_61_OPENS_TO",
    "KeyRight62Phase",
    "LEVEL4_COMPASS_BIT",
    "Left50Phase",
    "Level4Compass62Controller",
    "Level4EntryUpController",
    "Level4KeyRight62Controller",
    "Level4Left50Controller",
    "MAZE_62_RETURN_WEST",
    "MAZE_62_TO_COMPASS",
    "MAZE_IN_HOLD",
    "MAZE_OUT_HOLD",
    "ROOM_50_SPEC",
    "ROOM_51_SPEC",
    "ROOM_61_SPEC",
    "ROOM_62_SPEC",
    "ROOM_71_SPEC",
    "ROOM_ITEM_COMPASS",
    "ROOM_L4_COMPASS_62",
    "ROOM_L4_ENTRY",
    "ROOM_L4_KEESE_KEY_51",
    "ROOM_L4_VIRES_50",
    "ROOM_L4_VIRES_61",
    "VIRE_OBJECT_TYPE",
    "VIRE_SPLIT_KEESE_TYPE",
    "level4_compass_collected",
    "level4_compass_route_success",
    "level4_entry_ready",
    "level4_room_50_cleared",
    "level4_room_50_ready",
    "level4_room_51_key_success",
    "level4_room_51_ready",
    "level4_room_61_cleared",
    "level4_room_61_ready",
    "level4_room_62_cleared",
    "level4_room_62_ready",
    "level4_room_ready",
    "make_bomb_61_north_controller",
    "make_compass_62_controller",
    "make_entry_up_controller",
    "make_key_right_62_controller",
    "make_left_50_controller",
    "make_room_50_clear_controller",
    "make_room_51_key_controller",
    "make_room_61_clear_controller",
    "make_room_62_clear_controller",
    "planning_interior_report",
]
