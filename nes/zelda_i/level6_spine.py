"""Survival-spine L6 from L5 TF settle through west wizzrobes 0x78.

Do not poke Rod / doors / keys. Do not grant Whistle. Isolated BFS banned.
Ignore object types 0x2b / 0x68. Compass 0x68 and Rod/Gohma remain residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.anchors import LEVEL6_ENTRY_ROOM, TF_BIT_L5
from zelda_i.level6_dungeon import (
    ROOM_78_SPEC,
    ROOM_7A_SPEC,
    make_east_key_controller,
    make_west_wizzrobe_controller,
)
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_EAST_KEY_ROOM,
    POST_L5_PATH_MAX_FRAMES,
    POST_L5_SETTLE_MAX_FRAMES,
    Level6EntryRightController,
    Level6WestKeyDoorController,
    PostL5TriforceSettleController,
    make_post_l5_level6_controller,
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
    "Level6Return79Controller",
    "continue_level6_spine",
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
)
L6_STOPS: dict[str, str] = {
    "level6-entry": "level6_entry_0x79",
    "level6-east-key": "level6_east_key_0x7a",
    "level6-west": "level6_west_0x78",
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
