"""Level 4 (Snake) dungeon room specs and live anchors.

Uses ``dungeon.DungeonRoomSpec`` / ``GenericDungeonRoomController`` read-only.
Interior recon (assisted, rr-5lu 2026-08-09) — **no walkthrough hardcodes**.

Live path from ``Level4Entrance`` (room **0x71**)::

    0x71 entry (empty combat) --UP@x≈120--> 0x61
    0x61: 3× Vire type ``0x12`` (HP 64) → wooden sword splits to type ``0x1c``
    0x61 --BOMB_UP stand≈(120,105) face UP--> 0x51
    0x51: 8× Keese type ``0x1b`` (TYPE-only) + RoomItemId ``0x19`` key
    0x51 --LEFT @ y≈141--> 0x50 (5× Vire ``0x12``)  **tip residual**

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

# --- Live L4 room anchors (assisted recon rr-5lu 2026-08-09) ---
ROOM_L4_ENTRY = LEVEL4_ENTRY_ROOM  # 0x71 — empty combat mouth
ROOM_L4_VIRES_61 = 0x61  # north of entry; 3× Vire 0x12
ROOM_L4_KEESE_KEY_51 = 0x51  # bomb-N of 0x61; 8× Keese + key 0x19
ROOM_L4_VIRES_50 = 0x50  # west of 0x51; 5× Vire 0x12 (LIVE exit only)

VIRE_OBJECT_TYPE = 0x12  # live on 0x61/0x50; HP 64; splits on sword hit
VIRE_SPLIT_KEESE_TYPE = 0x1C  # live split residual from Vire (not standard 0x1B)
ROOM_ITEM_SMALL_KEY = 0x19
ROOM_ITEM_NONE = 0x03

# Bomb-north wall 0x61 → 0x51 (live stand ≈ y105, face UP).
BOMB_61_NORTH_STAND = (120, 105)
BOMB_61_NORTH_FACE = "UP"
BOMB_61_OPENS_TO = ROOM_L4_KEESE_KEY_51

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

register_room_spec(ROOM_71_SPEC)
register_room_spec(ROOM_61_SPEC)
register_room_spec(ROOM_51_SPEC)


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
        "track": "assisted",
        "status": "interior_partial",
        "date": "2026-08-09",
        "entry_room": hex(ROOM_L4_ENTRY),
        "live_graph": {
            hex(ROOM_L4_ENTRY): {"UP": hex(ROOM_L4_VIRES_61)},
            hex(ROOM_L4_VIRES_61): {
                "BOMB_UP": hex(ROOM_L4_KEESE_KEY_51),
                "KEY_RIGHT": "0x62",
                "enemies": {"0x12": 3, "split": "0x1c"},
            },
            hex(ROOM_L4_KEESE_KEY_51): {
                "LEFT": hex(ROOM_L4_VIRES_50),
                "enemies": {"0x1b": 8},
                "room_item": hex(ROOM_ITEM_SMALL_KEY),
            },
            hex(ROOM_L4_VIRES_50): {
                "enemies": {"0x12": 5},
                "note": "side_pocket; RIGHT seals after full clear",
            },
            "0x62": {
                "name": "compass_dark_maze",
                "room_item": "0x16",
                "entry": "KEY_RIGHT from 0x61",
                "maze": "DOWN then RIGHT from vestibule",
                "note": "stepladder residual rr-2ysf",
            },
        },
        "bomb_61_north": {
            "stand": list(BOMB_61_NORTH_STAND),
            "face": BOMB_61_NORTH_FACE,
            "opens_to": hex(BOMB_61_OPENS_TO),
        },
        "segments": {
            "entry_up": "rr-zchy",
            "clear_vires_61": "rr-yr77",
            "bomb_up_51": "rr-h278",
            "keese_key_51": "rr-wqdu",
            "stepladder_path": "rr-2ysf",
        },
        "not_yet": [
            "stepladder room / ADDR_LADDER",
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
    "DungeonPhase",
    "EntryUpPhase",
    "Level4EntryUpController",
    "ROOM_51_SPEC",
    "ROOM_61_SPEC",
    "ROOM_71_SPEC",
    "ROOM_L4_ENTRY",
    "ROOM_L4_KEESE_KEY_51",
    "ROOM_L4_VIRES_50",
    "ROOM_L4_VIRES_61",
    "VIRE_OBJECT_TYPE",
    "VIRE_SPLIT_KEESE_TYPE",
    "level4_entry_ready",
    "level4_room_51_key_success",
    "level4_room_51_ready",
    "level4_room_61_cleared",
    "level4_room_61_ready",
    "level4_room_ready",
    "make_bomb_61_north_controller",
    "make_entry_up_controller",
    "make_room_51_key_controller",
    "make_room_61_clear_controller",
    "planning_interior_report",
]
