"""Survival-spine L6 from L5 TF settle through blade-trap room 0x48.

Do not poke Rod / doors / keys. Do not grant Whistle. Isolated BFS banned.
Ignore object types 0x2b / 0x68. 0x58 north is free UP after Keese clear.
0x38 / Rod / Gohma / TF 0x20 remain residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import LEVEL6_ENTRY_ROOM, TF_BIT_L5
from zelda_i.level6_dungeon import (
    LEVEL6_COMPASS_BIT,
    ROOM_58_SPEC,
    ROOM_68_SPEC,
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
    make_compass_68_controller,
    make_east_key_controller,
    make_keese_58_controller,
    make_west_wizzrobe_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_COMPASS_ROOM,
    LEVEL6_EAST_KEY_ROOM,
    LEVEL6_KEESE_ROOM,
    LEVEL6_TRAPS_ROOM,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_SETTLE_MAX_FRAMES,
    Level6EntryRightController,
    Level6WestKeyDoorController,
    PostL5TriforceSettleController,
    make_post_l5_level6_controller,
)
from zelda_i.level6_path import (
    NORTH_68_MAX_FRAMES,
    Level6North68Controller,
    make_north_48_controller,
    make_north_58_controller,
)
from zelda_i.ram import (
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

__all__ = [
    "L6_STOPS",
    "L6_THROUGH",
    "Level6North68Controller",
    "Level6Return79Controller",
    "continue_level6_spine",
    "level6_clear68_stages",
    "level6_clear68_success",
    "level6_compass_stages",
    "level6_compass_success",
    "level6_clear58_stages",
    "level6_clear58_success",
    "level6_keese_stages",
    "level6_keese_success",
    "level6_room48_stages",
    "level6_room48_success",
    "level6_east_key_stages",
    "level6_east_key_success",
    "level6_entry_stages",
    "level6_entry_success",
    "level6_west_stages",
    "level6_west_success",
]

L6_THROUGH: tuple[str, ...] = (
    "level6-entry",
    "level6-east-key",
    "level6-west",
    "level6-compass",
    "level6-clear68",
    "level6-keese",
    "level6-clear58",
    "level6-room48",
)
L6_STOPS: dict[str, str] = {
    "level6-entry": "level6_entry_0x79",
    "level6-east-key": "level6_east_key_0x7a",
    "level6-west": "level6_west_0x78",
    "level6-compass": "level6_compass_0x68",
    "level6-clear68": "level6_clear_0x68",
    "level6-keese": "level6_keese_0x58",
    "level6-clear58": "level6_clear_0x58",
    "level6-room48": "level6_room_0x48",
}


def level6_entry_stages():
    """After L5 TF: idle the fanfare on 0x0B, then Lost Hills LEFT into L6."""
    return (
        (
            "settle_l5_tf",
            PostL5TriforceSettleController(),
            POST_L5_SETTLE_MAX_FRAMES,
        ),
        (
            "enter_level6",
            make_post_l5_level6_controller(),
            POST_L5_PATH_MAX_FRAMES,
        ),
    )


def level6_entry_success(snap: ZeldaSnapshot, *, whistle: int) -> bool:
    """Room-ready Dragon entry 0x79 with L5 inventory. Do not grant Whistle."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_ENTRY_ROOM
        and not snap.transitioning
        and bool(snap.triforce & TF_BIT_L5)
        and whistle >= 1
        and snap.raft > 0
        and snap.ladder > 0
    )


def level6_east_key_stages():
    """Entry 0x79 leftover → wall-first RIGHT → 0x7a key. No door pokes."""
    right = Level6EntryRightController()
    fight = make_east_key_controller()
    return (
        ("level6_right_0x7a", right, right.max_frames),
        ("level6_east_key_0x7a", fight, ROOM_7A_SPEC.max_frames),
    )


def level6_east_key_success(snap: ZeldaSnapshot, *, keys_before: int) -> bool:
    """Cleared 0x7a with a natural key pickup. Do not UP to Old Man 0x6a."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_EAST_KEY_ROOM
        and not snap.transitioning
        and snap.keys > keys_before
        and not ROOM_7A_SPEC.live_enemies(snap)
        and bool(snap.triforce & TF_BIT_L5)
    )


@dataclass
class Level6Return79Controller:
    """Free LEFT 0x7a → 0x79. Never UP (Old Man wastes the key)."""

    max_frames: int = 4000
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level6_return_0x79",
        }

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            return FrameAction(nes_idle_action(), "timeout")
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return FrameAction(nes_idle_action(), "link_death")
        if (
            snap.level == LEVEL6
            and snap.screen == LEVEL6_ENTRY_ROOM
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            self.success = True
            self.notes.append("arrived_79")
            return FrameAction(nes_idle_action(), "arrived_79")
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            return FrameAction(nes_action("LEFT"), "return_scroll")
        if snap.level != LEVEL6 or snap.screen != LEVEL6_EAST_KEY_ROOM:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        if abs(snap.link_y - 141) > 4:
            btn = "DOWN" if snap.link_y < 141 else "UP"
            return FrameAction(nes_action(btn), "return_ay")
        return FrameAction(nes_action("LEFT"), "return_left")


def level6_west_stages():
    """0x7a leftover → free 0x79 → key-LEFT 0x78 → clear. No 0x68."""
    back = Level6Return79Controller()
    door = Level6WestKeyDoorController()
    fight = make_west_wizzrobe_controller()
    return (
        ("level6_return_0x79", back, back.max_frames),
        ("level6_west_key_0x78", door, door.max_frames),
        ("level6_west_clear_0x78", fight, ROOM_78_SPEC.max_frames),
    )


def level6_west_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x78. Do not enter compass 0x68."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == ROOM_78_SPEC.room_id
        and not snap.transitioning
        and not ROOM_78_SPEC.live_enemies(snap)
        and bool(snap.triforce & TF_BIT_L5)
    )


def level6_compass_stages():
    """0x78 leftover → occupancy UP into compass room 0x68. No fight."""
    north = Level6North68Controller()
    return (
        ("level6_north_0x68", north, NORTH_68_MAX_FRAMES),
    )


def level6_compass_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x68. Compass pickup / Zol clear residual."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_COMPASS_ROOM
        and not snap.transitioning
        and bool(snap.triforce & TF_BIT_L5)
    )


def level6_clear68_stages():
    """0x68 leftover → occupancy Zol clear + compass bit. Ignore 0x2b/0x68."""
    fight = make_compass_68_controller()
    return (
        ("level6_clear_0x68", fight, ROOM_68_SPEC.max_frames),
    )


def level6_clear68_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x68 with L6 compass bit. Do not poke ADDR_COMPASS."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_COMPASS_ROOM
        and not snap.transitioning
        and (snap.compass & LEVEL6_COMPASS_BIT) != 0
        and not ROOM_68_SPEC.live_enemies(snap)
        and bool(snap.triforce & TF_BIT_L5)
    )


def level6_keese_stages():
    """0x68 leftover → occupancy UP into Keese 0x58. No fight."""
    north = make_north_58_controller()
    return (
        ("level6_north_0x58", north, NORTH_68_MAX_FRAMES),
    )


def level6_keese_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x58. Keese clear / key drop residual."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_KEESE_ROOM
        and not snap.transitioning
        and bool(snap.triforce & TF_BIT_L5)
    )


def level6_clear58_stages():
    """0x58 leftover → occupancy Keese clear. Key drop residual. No pokes."""
    fight = make_keese_58_controller()
    return (
        ("level6_clear_0x58", fight, ROOM_58_SPEC.max_frames),
    )


def level6_clear58_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready empty 0x58. Do not require key inventory."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_KEESE_ROOM
        and not snap.transitioning
        and not ROOM_58_SPEC.live_enemies(snap)
        and bool(snap.triforce & TF_BIT_L5)
    )


def level6_room48_stages():
    """0x58 leftover → occupancy long-UP into 0x48. No door poke."""
    north = make_north_48_controller()
    return (
        ("level6_north_0x48", north, north.max_frames),
    )


def level6_room48_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready 0x48. Blade-trap run residual."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and snap.screen == LEVEL6_TRAPS_ROOM
        and not snap.transitioning
        and bool(snap.triforce & TF_BIT_L5)
    )


def continue_level6_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L6 suffix after L5 TF. Mutates ``run``; caller returns it."""
    if not run_stages(
        env,
        run,
        level6_entry_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    run.success = level6_entry_success(snap, whistle=whistle)
    if not run.success:
        run.failed_stage = "level6_entry_0x79"
        return
    if through == "level6-entry":
        return

    keys_before = int(snap.keys)
    if not run_stages(
        env,
        run,
        level6_east_key_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_east_key_success(snap, keys_before=keys_before)
    if not run.success:
        run.failed_stage = "level6_east_key_0x7a"
        return
    if through == "level6-east-key":
        return

    if not run_stages(
        env,
        run,
        level6_west_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_west_success(snap)
    if not run.success:
        run.failed_stage = "level6_west_0x78"
        return
    if through == "level6-west":
        return

    if not run_stages(
        env,
        run,
        level6_compass_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_compass_success(snap)
    if not run.success:
        run.failed_stage = "level6_compass_0x68"
        return
    if through == "level6-compass":
        return

    if not run_stages(
        env,
        run,
        level6_clear68_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_clear68_success(snap)
    if not run.success:
        run.failed_stage = "level6_clear_0x68"
        return
    if through == "level6-clear68":
        return

    if not run_stages(
        env,
        run,
        level6_keese_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_keese_success(snap)
    if not run.success:
        run.failed_stage = "level6_keese_0x58"
        return
    if through == "level6-keese":
        return

    if not run_stages(
        env,
        run,
        level6_clear58_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_clear58_success(snap)
    if not run.success:
        run.failed_stage = "level6_clear_0x58"
        return
    if through == "level6-clear58":
        return

    if not run_stages(
        env,
        run,
        level6_room48_stages(),
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    ):
        return
    snap = read_snapshot(env.get_ram())
    run.success = level6_room48_success(snap)
    if not run.success:
        run.failed_stage = "level6_room_0x48"
