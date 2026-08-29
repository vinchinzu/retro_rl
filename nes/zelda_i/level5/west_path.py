"""Level 5 west-door hops: 0x27 → 0x26 → 0x25 → 0x24.

Room specs and stop predicates remain in ``level5_dungeon``.
Import from ``zelda_i.level5.path`` (public facade).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5.dungeon import (
    LEVEL_5,
    ROOM_L5_NORTH_27,
    ROOM_L5_WEST_24,
    ROOM_L5_WEST_25,
    ROOM_L5_WEST_26,
)
from zelda_i.level5.path import EAST_DOOR_CHANNEL_Y, EAST_DOOR_WALL_X, walk_axis
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

SOUTH_PINCH_Y = 189
WEST_DOOR_X = 32


def level5_west26_from_27_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x27 west key door: off ladder, south y=189, west mouth.

    Start is often the 0x27 ladder (120,141). RIGHT from there stalls.
    Leave south first, then east wall if free, then y=189 past the x=160
    pinch, then west door at y=141.
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "west26_wait_level5")
    if snap.screen == ROOM_L5_WEST_26 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "west26_arrived")
    if snap.screen != ROOM_L5_NORTH_27:
        return FrameAction(
            nes_idle_action(), f"west26_unexpected_room_0x{snap.screen:02x}"
        )
    if snap.transitioning or snap.mode != PLAY_MODE:
        return FrameAction(nes_action("LEFT"), "west26_west_scroll")

    on_ladder_col = abs(snap.link_x - 120) <= 10
    # Off the mid ladder before any east/west at y=141.
    if on_ladder_col and snap.link_y < SOUTH_PINCH_Y - 3:
        return FrameAction(nes_action("DOWN"), "west26_off_ladder")
    # South band first (probe continues here even if east wall stalls).
    if snap.link_y < SOUTH_PINCH_Y - 3:
        return FrameAction(nes_action("DOWN"), "west26_south_band")
    # Optional east wall once already south — skip if already west of pinch.
    if snap.link_x < 160 and snap.link_x < EAST_DOOR_WALL_X - 2 and snap.link_y >= SOUTH_PINCH_Y - 6:
        # Already on south band; go west, do not climb back to east wall.
        pass
    elif snap.link_x >= 160 and snap.link_x < EAST_DOOR_WALL_X - 2 and snap.link_y < 170:
        return FrameAction(nes_action("RIGHT"), "west26_east_wall")
    if snap.link_x > WEST_DOOR_X + 4:
        if abs(snap.link_y - SOUTH_PINCH_Y) > 4 and snap.link_x > 48:
            direction = "DOWN" if snap.link_y < SOUTH_PINCH_Y else "UP"
            return FrameAction(nes_action(direction), "west26_hold_south_y")
        return FrameAction(nes_action("LEFT"), "west26_south_west")
    # West column: UP from y≈185 at x=32 is blocked. Step to x≈48, rise, then door.
    if snap.link_y > EAST_DOOR_CHANNEL_Y + 6:
        if snap.link_x < 46:
            return FrameAction(nes_action("RIGHT"), "west26_clear_sw_block")
        return FrameAction(nes_action("UP"), "west26_rise_to_door")
    if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
        direction = "DOWN" if snap.link_y < EAST_DOOR_CHANNEL_Y else "UP"
        return FrameAction(nes_action(direction), "west26_align_door_y")
    if snap.link_x > WEST_DOOR_X + 1:
        return FrameAction(nes_action("LEFT"), "west26_to_mouth")
    return FrameAction(nes_action("LEFT"), "west26_unlock_26")


@dataclass
class Level5West26From27Controller:
    """Walk 0x27 → west key door → 0x26. No combat. No pokes."""

    max_frames: int = 4000
    settle_frames: int = 30
    frames: int = 0
    settle_left: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    last_room: int = -1

    def report(self) -> dict:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": "level5_west26_from_cleared27",
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
        if snap.screen != self.last_room:
            self.notes.append(
                f"room_0x{snap.screen:02x}_f{self.frames}_xy={snap.link_x},{snap.link_y}_k={snap.keys}"
            )
            self.last_room = snap.screen
        if (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_WEST_26
            and snap.mode == PLAY_MODE
        ):
            if self.settle_left <= 0 and "settling_26" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_26")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_26")
            self.success = True
            self.notes.append("arrived_26")
            return FrameAction(nes_idle_action(), "arrived_26")
        return level5_west26_from_27_step(snap)




def walk_west_from_27(env, assist, total: list[int]) -> dict:
    """Proven 0x27 leave: east wall, south y=189, west key door → 0x26."""
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y]}]
    for axis, tgt in (("x", 208), ("y", 189), ("x", 32), ("y", 141), ("x", 32)):
        ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
        snap = _rs(env.get_ram())
        log.append({"step": f"{axis}:{tgt}", "ok": ok, "xy": [snap.link_x, snap.link_y], "room": snap.screen})
    # Align and push through the key door.
    for _ in range(24):
        snap = _rs(env.get_ram())
        if abs(snap.link_x - 32) <= 2 and abs(snap.link_y - 141) <= 2:
            break
        if abs(snap.link_y - 141) > 2:
            env.step(nes_action("DOWN" if snap.link_y < 141 else "UP"))
        else:
            env.step(nes_action("LEFT" if snap.link_x > 32 else "RIGHT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    room0 = _rs(env.get_ram()).screen
    for _ in range(220):
        snap = _rs(env.get_ram())
        if snap.screen != room0:
            break
        env.step(nes_action("LEFT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(36):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = _rs(env.get_ram())
    return {
        "path": "east_wall_south189_west_door",
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_WEST_26 and snap.mode == PLAY_MODE,
    }


def make_west26_from_27_controller() -> Level5West26From27Controller:
    return Level5West26From27Controller()


WEST26_TO_25_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 32), ("y", 141))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
)

WEST25_TO_24_PATHS = (
    ("y141_west", (("y", 141), ("x", 32))),
    ("south189_west", (("y", 189), ("x", 32), ("y", 141))),
    ("north109_west", (("y", 109), ("x", 80), ("y", 141), ("x", 32))),
    ("east208_south189_west", (("x", 208), ("y", 189), ("x", 32), ("y", 141))),
    ("south173_west64", (("y", 173), ("x", 64), ("y", 141), ("x", 32))),
)


def _align_door(env, assist, total: list[int], tx: int, ty: int = 141, frames: int = 24) -> list[int]:
    from zelda_i.ram import read_snapshot as _rs

    for _ in range(frames):
        snap = _rs(env.get_ram())
        if abs(snap.link_x - tx) <= 2 and abs(snap.link_y - ty) <= 2:
            break
        if abs(snap.link_y - ty) > 2:
            env.step(nes_action("DOWN" if snap.link_y < ty else "UP"))
        else:
            env.step(nes_action("LEFT" if snap.link_x > tx else "RIGHT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = _rs(env.get_ram())
    return [snap.link_x, snap.link_y]


def _push_left(env, assist, total: list[int], frames: int = 220) -> None:
    from zelda_i.ram import read_snapshot as _rs

    room0 = _rs(env.get_ram()).screen
    for _ in range(frames):
        snap = _rs(env.get_ram())
        if snap.screen != room0:
            break
        env.step(nes_action("LEFT"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def _idle(env, assist, total: list[int], frames: int = 36) -> None:
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def walk_west_from_26(env, assist, total: list[int]) -> dict:
    """Proven 0x26 leave: y=141 then west open door → 0x25. Moat/C-block fallbacks."""
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    room0 = snap.screen
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y], "room": snap.screen}]
    used = None
    for name, steps in WEST26_TO_25_PATHS:
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "step": f"{name}:{axis}:{tgt}",
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "room": snap.screen,
                }
            )
        _align_door(env, assist, total, 32, 141)
        snap = _rs(env.get_ram())
        if abs(snap.link_x - 32) <= 6 and abs(snap.link_y - 141) <= 4:
            used = name
            break
    _align_door(env, assist, total, 32, 141, frames=32)
    _push_left(env, assist, total, frames=220)
    _idle(env, assist, total, 36)
    snap = _rs(env.get_ram())
    return {
        "path": used or "y141_west",
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_WEST_25
            and snap.mode == PLAY_MODE
        ),
    }


def walk_west_from_25(env, assist, total: list[int]) -> dict:
    """Proven 0x25 leave: y=141 then west key door → 0x24. Door only; no Digdogger."""
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    room0 = snap.screen
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y], "room": snap.screen}]
    used = None
    for name, steps in WEST25_TO_24_PATHS:
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=500)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "step": f"{name}:{axis}:{tgt}",
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "room": snap.screen,
                }
            )
        _align_door(env, assist, total, 32, 141)
        snap = _rs(env.get_ram())
        if abs(snap.link_x - 32) <= 8 and abs(snap.link_y - 141) <= 8:
            used = name
            break
    _align_door(env, assist, total, 32, 141, frames=28)
    _push_left(env, assist, total, frames=240)
    _idle(env, assist, total, 36)
    snap = _rs(env.get_ram())
    return {
        "path": used or "y141_west",
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "fought_digdogger": False,
        "success": (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_WEST_24
            and snap.mode == PLAY_MODE
        ),
    }

