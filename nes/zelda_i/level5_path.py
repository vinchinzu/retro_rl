"""Level 5 multi-room path policy.

East key: 0x66 → 0x76 east key door → 0x77.
Post-east-key: 0x77 → 0x76 → 0x66 free UP → 0x56.
Cleared27 west key: 0x27 east wall → south y≈189 around x=160 pinch → west door → 0x26.
Cleared26 west open: 0x26 y=141 (moat/C-block fallbacks) → 0x25.
Cleared25 west key: 0x25 y=141 → 0x24 door only (no Digdogger).
Whistle: bomb-west 0x65→0x64 center stairs → cellar 0x07 other mouth →
0x06 key-west → 0x05 clear+block stairs → 0x04 Recorder → left mouth back to 0x05.
Whistle65 east: walk the existing 0x66-west bomb hole (diamond y=109 then east);
bomb EAST only if sealed. Then 0x66 UP → 0x56 → 0x57 north (do not clear Zols) → 0x47 → 0x37
→ 0x27 → 0x26 → 0x25 → 0x24 Digdogger. Do not try UP from 0x65.

Room specs and stop predicates remain in ``level5_dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_NORTH_27,
    ROOM_L5_NORTH_56,
    ROOM_L5_POLS_77,
    ROOM_L5_WEST_24,
    ROOM_L5_WEST_25,
    ROOM_L5_WEST_26,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

EAST_DOOR_APPROACH_Y = 157
EAST_DOOR_CHANNEL_Y = 141
# Live statue blocks UP at x=200 (stuck 200,149). Start y-slide at wall x≈208.
EAST_DOOR_WALL_X = 208

_ISOLATED_ROOM_77 = "L5_Room_77"


def should_force_keys_zero(
    start_state: str,
    *,
    keep_keys: bool = False,
    force_keys_zero: bool = False,
) -> bool:
    """True only for isolated ``L5_Room_77`` combat, unless explicitly overridden."""
    if keep_keys:
        return False
    if force_keys_zero:
        return True
    return start_state == _ISOLATED_ROOM_77


def level5_east_key_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x66→0x76→key door→0x77 navigation policy.

    Room 0x66 supplies the key.  Return through its south door, leave the
    0x76 north/south mouth, approach the east wall on y≈157, then move to the
    door channel y≈141 without stepping back into the center statues.
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "east_key_wait_level5")
    if snap.screen == ROOM_L5_POLS_77 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "east_key_arrived")

    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning:
            return FrameAction(nes_action("DOWN"), "east_key_south_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_66")
        # The fixed key can be collected while Link is still standing on the
        # Stepladder across the horizontal river. Finish the crossing before
        # horizontal alignment; sideways input is locked on the ladder tile.
        if snap.link_y < 141:
            return FrameAction(nes_action("DOWN"), "east_key_finish_ladder")
        if abs(snap.link_x - 120) > 4:
            direction = "LEFT" if snap.link_x > 120 else "RIGHT"
            return FrameAction(nes_action(direction), "east_key_align_south_x")
        return FrameAction(nes_action("DOWN"), "east_key_return_76")

    if snap.screen == ROOM_L5_ENTRY:
        if snap.transitioning:
            return FrameAction(nes_action("RIGHT"), "east_key_east_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_76")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "east_key_leave_south_mouth")
        if snap.link_x < EAST_DOOR_WALL_X - 4:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                direction = "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                return FrameAction(nes_action(direction), "east_key_align_approach_y")
            return FrameAction(nes_action("RIGHT"), "east_key_approach_wall")
        if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
            direction = "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
            return FrameAction(nes_action(direction), "east_key_align_channel_y")
        return FrameAction(nes_action("RIGHT"), "east_key_unlock_77")

    return FrameAction(
        nes_idle_action(), f"east_key_unexpected_room_0x{snap.screen:02x}"
    )


NORTH_DOOR_X = 120
WEST_LEAVE_EAST_X = 140


def level5_west65_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x77→0x76→0x66 free UP→0x56 navigation policy.

    Source route after the east key: return west, north through 0x66, then UP
    into the next dark room (live 0x56). Reuse the 0x76 statue bypass (y≈157).
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "west65_wait_level5")
    if snap.screen == ROOM_L5_NORTH_56 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "west65_arrived_56")

    if snap.screen == ROOM_L5_POLS_77:
        if snap.transitioning:
            return FrameAction(nes_action("LEFT"), "west65_west_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_77")
        # Two 2x3 block clusters pinch y=141 around x≈96. Pass south (y≈173)
        # then drop to the west-door channel once past the left cluster.
        if snap.link_x > 48:
            if abs(snap.link_y - 173) > 3:
                direction = "UP" if snap.link_y > 173 else "DOWN"
                return FrameAction(nes_action(direction), "west65_align_77_south_y")
            return FrameAction(nes_action("LEFT"), "west65_pass_77_blocks")
        if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
            direction = "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
            return FrameAction(nes_action(direction), "west65_align_77_y")
        return FrameAction(nes_action("LEFT"), "west65_return_76")

    if snap.screen == ROOM_L5_ENTRY:
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_76")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "west65_leave_south_mouth")
        # East doorway (x≈224,y≈141) blocks DOWN. Step out to the wall
        # before the statue-bypass y-slide.
        if snap.link_x > EAST_DOOR_WALL_X:
            return FrameAction(nes_action("LEFT"), "west65_leave_east_mouth")
        if snap.link_x > WEST_LEAVE_EAST_X:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                direction = "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                return FrameAction(nes_action(direction), "west65_align_approach_y")
            return FrameAction(nes_action("LEFT"), "west65_leave_east_door")
        if abs(snap.link_x - NORTH_DOOR_X) > 4:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "west65_align_north_x")
        return FrameAction(nes_action("UP"), "west65_enter_66")

    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning or snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "west65_wait_66")
        if snap.link_y < 141 and snap.link_x < 80:
            return FrameAction(nes_action("DOWN"), "west65_finish_ladder")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "west65_leave_66_south")
        if abs(snap.link_x - NORTH_DOOR_X) > 4:
            direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
            return FrameAction(nes_action(direction), "west65_align_66_north_x")
        return FrameAction(nes_action("UP"), "west65_enter_56")

    return FrameAction(
        nes_idle_action(), f"west65_unexpected_room_0x{snap.screen:02x}"
    )


@dataclass
class Level5West65Controller:
    """Walk 0x77→0x76→0x66 free UP→0x56; stop on arrival.

    No combat. Success = room-ready 0x56. Does not poke keys or doors.
    """

    max_frames: int = 8000
    settle_frames: int = 45
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
            "spec_id": "level5_north56_from_east_key",
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
            and snap.screen == ROOM_L5_NORTH_56
            and snap.mode == PLAY_MODE
        ):
            if self.settle_left <= 0 and "settling_56" not in self.notes:
                self.settle_left = self.settle_frames
                self.notes.append("settling_56")
            if self.settle_left > 0:
                self.settle_left -= 1
                if self.settle_left > 0:
                    return FrameAction(nes_idle_action(), "settle_56")
            self.success = True
            self.notes.append("arrived_56")
            return FrameAction(nes_idle_action(), "arrived_56")

        return level5_west65_step(snap)


def make_west65_controller() -> Level5West65Controller:
    return Level5West65Controller()



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



def walk_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 500) -> bool:
    """Step one axis toward target. Used by the 0x27 west-key leave."""
    last = None
    stall = 0
    read_snapshot = __import__("zelda_i.ram", fromlist=["read_snapshot"]).read_snapshot
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        env.step(action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 40:
                return False
        else:
            stall = 0
        last = pos
    return False


def _cellar_walk_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 700) -> bool:
    """Axis walk that survives recorder fanfare and aborts on a real 0x04 leave."""
    last = None
    stall = 0
    read_snapshot = __import__("zelda_i.ram", fromlist=["read_snapshot"]).read_snapshot
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != ROOM_L5_WHISTLE_ITEM:
            return True
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        _step(env, assist, total, action)
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 160:
                return False
        else:
            stall = 0
        last = pos
    return False


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



ROOM_L5_BLUE_64 = 0x64
ROOM_L5_CELLAR_07 = 0x07
ROOM_L5_PASSAGE_06 = 0x06
ROOM_L5_WHISTLE_05 = 0x05
ROOM_L5_WHISTLE_ITEM = 0x04
BLUE_DARKNUT_TYPE = 0x0C
BOMB_WEST_STAND = (40, 141)
CENTER_STAIRS = (120, 141)
CELLAR_MODES = (9, 10, 11, 16)
# 0x06 diamond: 0x68 rests (96,144). Push UP → (96,128). Stairs stand (96,133).
# Center 0x70–0x73 tiles are decorative and do not warp. South key is 0x16, not return.
ROOM_06_BLOCK_X = 96
ROOM_06_BLOCK_REST_Y = 144
ROOM_06_BLOCK_PUSHED_Y = 128
ROOM_06_STAIRS_X = 96
ROOM_06_STAIRS_Y = 133


def _step(env, assist, total: list[int], action) -> None:
    env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])


def select_b_item_menu(env, assist, total: list[int], want: int) -> dict:
    """Pause-cycle B items. want=1 bombs, want=5 recorder. No RAM poke."""
    from zelda_i.ram import ADDR_SELECTED_ITEM, read_u8

    selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    seen = [selected0]
    if selected0 == want:
        return {"used": False, "selected": selected0, "seen": seen}
    _step(env, assist, total, nes_action("START"))
    idle = __import__("zelda_i.dungeon_ops", fromlist=["idle"]).idle
    idle(env, assist, total, 20)
    chosen = selected0
    for _ in range(8):
        _step(env, assist, total, nes_action("RIGHT"))
        idle(env, assist, total, 8)
        cur = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        seen.append(cur)
        if cur == want:
            chosen = cur
            break
    _step(env, assist, total, nes_action("START"))
    idle(env, assist, total, 24)
    return {
        "used": True,
        "selected_before": selected0,
        "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "seen": seen,
        "preferred": chosen,
    }


def bomb_west_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the west wall of cleared 0x65. One bomb. Dest must become 0x64.

    Live 0x65 has a center diamond: y=109 then x=32 then y=141, not y=141 first.
    Hold LEFT through the west scroll even while SCREEN still reads 0x65.
    """
    from zelda_i.dungeon_ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=400)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=200)
    goto(env, assist, total, 32, 141, tol=3, max_f=300)
    for _ in range(8):
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 1)
    snap = _rs(env.get_ram())
    bombs0 = int(snap.bombs)
    room0 = int(snap.screen)
    _step(env, assist, total, nes_action("LEFT", "B"))
    for _ in range(16):
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 100)
    for _ in range(360):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64:
            break
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 24)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64:
            break
        if snap.mode in (6, 7, 4, 16):
            _step(env, assist, total, nes_action("LEFT"))
        else:
            _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "bomb_west_from_65",
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "bombs_spent": bombs0 - int(snap.bombs),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": (
            snap.level == LEVEL_5
            and snap.screen == ROOM_L5_BLUE_64
            and snap.mode == PLAY_MODE
        ),
    }


def _in_cellar(snap) -> bool:
    return snap.mode in CELLAR_MODES


def bomb_east_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the east wall of cleared 0x65. One bomb. Dest must become 0x66.

    North shutter is one-way (0x55 S=open / 0x65 N=shutter). Diamond: y=109
    then east, not y=141 first.
    """
    from zelda_i.dungeon_ops import goto, idle
    from zelda_i.ram import read_snapshot as _rs

    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 208, max_f=500)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 224, max_f=200)
    goto(env, assist, total, 224, 141, tol=3, max_f=300)
    for _ in range(8):
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 1)
    snap = _rs(env.get_ram())
    bombs0 = int(snap.bombs)
    room0 = int(snap.screen)
    _step(env, assist, total, nes_action("RIGHT", "B"))
    for _ in range(16):
        _step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 100)
    for _ in range(360):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == 0x66:
            break
        _step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 24)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == 0x66:
            break
        if snap.mode in (6, 7, 4, 16):
            _step(env, assist, total, nes_action("RIGHT"))
        else:
            _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "bomb_east_from_65",
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "bombs_spent": bombs0 - int(snap.bombs),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == 0x66 and snap.mode == PLAY_MODE,
    }


def take_center_stairs_64(env, assist, total: list[int]) -> dict:
    """Walk the south (then north) gap onto visible center stairs in 0x64.

    Do not hunt the east bomb hole. Success = cellar/stairs mode or room 0x07.
    """
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import read_snapshot as _rs

    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen}

    def done(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.level == LEVEL_5 and snap.screen == ROOM_L5_CELLAR_07

    paths = (
        (("y", 189), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 96), ("y", 149), ("x", 120), ("y", 141)),
        (("y", 189), ("x", 64), ("y", 141), ("x", 120)),
        (("y", 93), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 120), ("y", 141)),
        (("y", 173), ("x", 120), ("y", 141), ("x", 120)),
    )
    for name_i, steps in enumerate(paths):
        if done(_rs(env.get_ram())):
            break
        if _rs(env.get_ram()).screen != ROOM_L5_BLUE_64:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=360)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "path": name_i,
                    "step": f"{axis}:{tgt}",
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if done(snap):
                break
        idle(env, assist, total, 20)
        snap = _rs(env.get_ram())
        if done(snap):
            break
        # Nudge onto the tile; never hold LEFT (east bomb hole → 0x65).
        for direction in ("UP", "DOWN", "RIGHT"):
            for _ in range(16):
                snap = _rs(env.get_ram())
                if done(snap) or snap.screen != ROOM_L5_BLUE_64:
                    break
                _step(env, assist, total, nes_action(direction))
            idle(env, assist, total, 8)
            if done(_rs(env.get_ram())):
                break
        if done(_rs(env.get_ram())):
            break

    for _ in range(200):
        snap = _rs(env.get_ram())
        if done(snap) or (snap.mode == PLAY_MODE and snap.screen != ROOM_L5_BLUE_64):
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "south_gap_center_stairs",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "cellar": _in_cellar(snap),
        "success": done(snap) and snap.screen != 0x65,
    }


# Live L5 cellar 0x07: left mouth spawn ~(48,93); floor y=189; right climb x=192.
L5_CELLAR_FLOOR_Y = 189
L5_CELLAR_LEFT_X = 48
L5_CELLAR_RIGHT_X = 192


def cellar_other_mouth(env, assist, total: list[int]) -> dict:
    """From L5 cellar 0x07, take the opposite mouth to room 0x06. No pokes."""
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    # Stair-enter sits at (128,141) ~90f, then remaps to a ladder (48,93) or (192,93).
    for _ in range(180):
        snap = _rs(env.get_ram())
        if snap.mode in CELLAR_MODES and (snap.link_x <= 64 or snap.link_x >= 176):
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}
    # Left column is the 0x64 return. Floor-cross to x=192 then UP → 0x06.
    if snap.link_x <= 128:
        side = "right"
        tx = L5_CELLAR_RIGHT_X
    else:
        side = "left"
        tx = L5_CELLAR_LEFT_X
    walk_axis(env, assist, total, "y", L5_CELLAR_FLOOR_Y, max_f=400)
    walk_axis(env, assist, total, "x", tx, max_f=500)
    room0 = _rs(env.get_ram()).screen
    push_dir(env, assist, total, "UP", frames=200)
    idle(env, assist, total, 20)
    for _ in range(240):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen != room0:
            break
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_PASSAGE_06:
            break
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "cellar_other_mouth",
        "start": start,
        "chose_side": side,
        "target_x": tx,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE,
    }


def key_west_to(env, assist, total: list[int], expect: int) -> dict:
    """Spend a key at the west door. No door/key poke."""
    from zelda_i.dungeon_ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    room0 = int(snap.screen)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=500)
    goto(env, assist, total, 32, 141, tol=3, max_f=300)
    push_dir(env, assist, total, "LEFT", frames=240)
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    if snap.screen != room0:
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE:
                break
            _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "key_west",
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == expect and snap.mode == PLAY_MODE,
    }


def fight_blue_darknuts(env, assist, total: list[int], room: int, expected: int, source: int) -> dict:
    """Reuse GenericDungeonRoomController + ROOM_5B_SPEC / ROOM_59 combat."""
    from dataclasses import replace

    from zelda_i.dungeon import (
        DoorRoute,
        DungeonPhase,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )
    from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
    from zelda_i.ram import read_snapshot as _rs

    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_blue_darknuts",
        source_room=source,
        room_id=room,
        entry=DoorRoute("LEFT", ((224, 141),)),
        enemy_types=(BLUE_DARKNUT_TYPE, 0x0B),
        expected_enemy_count=expected,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
        max_frames=28000,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == room:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n})
        action = ctl.step(snap)
        _step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = _rs(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in (BLUE_DARKNUT_TYPE, 0x0B) and o.hp > 0
    ] if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "progress": progress,
        "spec_id": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
    }


def push_block_stairs(env, assist, total: list[int], room: int) -> dict:
    """Push 0x68 then stand on revealed stairs. Never treat a door exit as stairs."""
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.level9_stairs import BLOCK_STAIRS_X, BLOCK_STAIRS_Y, PUSHABLE_BLOCK
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    blocks = [
        o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
    ]
    log = []
    dest = None

    def left_ok(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.screen != room and snap.mode in (*CELLAR_MODES, PLAY_MODE) and snap.screen != 0x65

    targets = [(b.x, b.y) for b in blocks] + [
        (96, 144),
        (112, 144),
        (80, 144),
        (120, 144),
        (128, 144),
    ]
    seen = set()
    for tx, ty in targets:
        key = (tx // 8, ty // 8)
        if key in seen:
            continue
        seen.add(key)
        snap = _rs(env.get_ram())
        if left_ok(snap):
            dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
            break
        walk_axis(env, assist, total, "y", ty, max_f=280)
        walk_axis(env, assist, total, "x", tx + 16, max_f=280)
        rec = {"stand": [tx, ty], "dirs": []}
        for direction in ("LEFT", "UP", "DOWN", "RIGHT"):
            push_dir(env, assist, total, direction, frames=90)
            idle(env, assist, total, 8)
            snap = _rs(env.get_ram())
            rec["dirs"].append(
                {
                    "dir": direction,
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        log.append(rec)
        if dest is not None:
            break
        for sx, sy in ((tx, ty), (BLOCK_STAIRS_X, BLOCK_STAIRS_Y), CENTER_STAIRS, (120, 125)):
            walk_axis(env, assist, total, "y", sy, max_f=200)
            walk_axis(env, assist, total, "x", sx, max_f=200)
            idle(env, assist, total, 10)
            snap = _rs(env.get_ram())
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        if dest is not None:
            break
    snap = _rs(env.get_ram())
    return {
        "blocks_seen": [{"slot": b.slot, "x": b.x, "y": b.y} for b in blocks],
        "dest": dest,
        "end": {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]},
        "log": log,
        "success": dest is not None,
    }



def take_whistle_04(env, assist, total: list[int]) -> dict:
    """Cellar 0x04: floor y=189, short ladder x=176, left on y=141 to the Recorder."""
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    walk_axis(env, assist, total, "y", 189, max_f=400)
    walk_axis(env, assist, total, "x", 176, max_f=400)
    for _ in range(80):
        snap = _rs(env.get_ram())
        if int(read_u8(env.get_ram(), ADDR_WHISTLE)) > w0:
            break
        if snap.link_y <= 141 and abs(snap.link_x - 176) <= 4:
            break
        _step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 8)
    walk_axis(env, assist, total, "y", 141, max_f=200)
    walk_axis(env, assist, total, "x", 128, max_f=300)
    idle(env, assist, total, 12)
    w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    if w1 <= w0:
        walk_axis(env, assist, total, "x", 144, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=200)
        idle(env, assist, total, 10)
        w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    snap = _rs(env.get_ram())
    return {
        "in": w0,
        "out": w1,
        "got": w1 > w0,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "mode": snap.mode,
    }


def hunt_whistle(env, assist, total: list[int]) -> dict:
    """Walk item stands until ADDR_WHISTLE becomes 1.

    Room 0x04 is a side-scroll item cellar: top-down stands stay on the
    floor (y=189). Use take_whistle_04 (right short ladder -> y=141).
    """
    from zelda_i.dungeon_ops import idle
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    w0 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    snap0 = _rs(env.get_ram())
    room0 = snap0.screen
    hits = []
    if room0 == ROOM_L5_WHISTLE_ITEM or snap0.mode in CELLAR_MODES:
        cellar = take_whistle_04(env, assist, total)
        hits.append({"via": "take_whistle_04", "xy": cellar.get("xy"), "value": cellar.get("out")})
        if cellar.get("got"):
            return {"in": w0, "out": cellar["out"], "got": True, "hits": hits, "via": "take_whistle_04"}
    stands = (
        (120, 141),
        (136, 141),
        (104, 141),
        (120, 125),
        (120, 157),
        (80, 141),
        (160, 141),
        (120, 109),
        (64, 117),
        (176, 117),
        (96, 165),
        (144, 165),
    )
    for tx, ty in stands:
        snap = _rs(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        idle(env, assist, total, 10)
        w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        snap = _rs(env.get_ram())
        hits.append({"stand": [tx, ty], "xy": [snap.link_x, snap.link_y], "value": w1})
        if w1 > w0:
            break
    w1 = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    return {"in": w0, "out": w1, "got": w1 > w0, "hits": hits}


# Live 0x04 item cellar: isolated recorder alcove ~y=141, x≈112–176.
# Short ladder at x=176 drops to pit y=189. Left mouth stairs at x=48
# (spawn 48,65) return to play 0x05. Do not walk left on the alcove —
# the platform does not connect to the left column.
WHISTLE_04_LADDER_X = 176
WHISTLE_04_PIT_Y = 189
WHISTLE_04_MOUTH_X = 48
WHISTLE_04_MOUTH_Y = 65


def exit_whistle_04(env, assist, total: list[int]) -> dict:
    """Leave cellar 0x04: alcove x=176 DOWN → pit y=189 → left mouth x=48 UP → 0x05.

    Failed probes walked LEFT/UP on the recorder alcove (y=141). That platform
    does not connect to the left column. Drop the short ladder first.
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "room": snap.screen,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }
    log = [dict(start, tag="start")]

    def left_ok(s) -> bool:
        return s.mode == PLAY_MODE and s.screen != ROOM_L5_WHISTLE_ITEM

    def rec(tag: str) -> None:
        s = _rs(env.get_ram())
        log.append(
            {
                "tag": tag,
                "xy": [s.link_x, s.link_y],
                "mode": s.mode,
                "room": s.screen,
                "tile": int(s.colliding_tile),
            }
        )

    # Recorder item-get holds Link overhead. Tap RIGHT (toward the ladder)
    # until RAM x/y changes — idle-only is not enough; walk_axis aborts at 40.
    idle(env, assist, total, 40)
    thawed = False
    for n in range(10):
        x0 = _rs(env.get_ram()).link_x
        y0 = _rs(env.get_ram()).link_y
        for i in range(24):
            _step(env, assist, total, nes_action("RIGHT"))
            snap = _rs(env.get_ram())
            if snap.link_x != x0 or snap.link_y != y0:
                thawed = True
                log.append({"tag": "thawed", "burst": n, "f": i, "xy": [snap.link_x, snap.link_y]})
                break
        if thawed:
            break
        idle(env, assist, total, 32)
    rec("unstick")

    def drop_ladder() -> None:
        _cellar_walk_axis(env, assist, total, "y", 141, max_f=240)
        _cellar_walk_axis(env, assist, total, "x", WHISTLE_04_LADDER_X, max_f=700)
        rec("ladder")
        for _ in range(280):
            snap = _rs(env.get_ram())
            if left_ok(snap) or snap.link_y >= WHISTLE_04_PIT_Y - 2:
                break
            _step(env, assist, total, nes_action("DOWN"))
        idle(env, assist, total, 8)
        _cellar_walk_axis(env, assist, total, "y", WHISTLE_04_PIT_Y, max_f=400)
        rec("pit")

    # Alcove (y≈141) only drops at the short ladder x=176.
    for attempt in range(3):
        snap = _rs(env.get_ram())
        if left_ok(snap) or snap.link_y >= 170:
            break
        drop_ladder()
        snap = _rs(env.get_ram())
        if snap.link_y < 170 and abs(snap.link_x - WHISTLE_04_LADDER_X) > 4:
            log.append({"tag": f"retry_ladder_{attempt}", "xy": [snap.link_x, snap.link_y]})

    snap = _rs(env.get_ram())
    # Live RAM: only the pit (y>=170) connects to the left mouth. Do not
    # walk LEFT on the alcove — that stalls at x≈112, y=141.
    if not left_ok(snap) and snap.link_y >= 170:
        _cellar_walk_axis(env, assist, total, "x", WHISTLE_04_MOUTH_X, max_f=700)
        rec("left_col")
        # Hold UP from the pit. Do not walk_axis to y=65 — that overshoots
        # into 0x05's north door after the mouth fires.
        push_dir(env, assist, total, "UP", frames=280)
        idle(env, assist, total, 12)
        for _ in range(280):
            snap = _rs(env.get_ram())
            if left_ok(snap):
                break
            if abs(snap.link_x - WHISTLE_04_MOUTH_X) > 4:
                _step(
                    env,
                    assist,
                    total,
                    nes_action("LEFT" if snap.link_x > WHISTLE_04_MOUTH_X else "RIGHT"),
                )
            else:
                _step(env, assist, total, nes_action("UP"))
        idle(env, assist, total, 20)
    rec("after_up")
    snap = _rs(env.get_ram())
    return {
        "path": "alcove_ladder176_pit189_left48",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "success": (
            snap.level == LEVEL_5
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_L5_WHISTLE_05
        ),
        "left_cellar": left_ok(snap),
        "thawed": thawed,
    }



leave_whistle_cellar = exit_whistle_04
walk_out_of_04 = exit_whistle_04


def take_center_stairs_06(env, assist, total: list[int]) -> dict:
    """0x06: north-around, push left 0x68 north, idle on (120,141) tile 0x71.

    (96,133) is the outbound cellar *spawn*, not the inbound warp. Live
    Whistle05 tape: after 0x68 (96,144)->(96,128), stand/hunt (120,141) /
    (128,141) tile 0x71 -> mode 9 room 0x07. Do not walk south to 0x16.
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen}

    def done(s) -> bool:
        if s.mode == PLAY_MODE and s.screen == 0x16:
            return False
        if _in_cellar(s):
            return True
        return s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07

    def rec(tag, **extra):
        s = _rs(env.get_ram())
        blocks = [
            {"x": o.x, "y": o.y}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id == 0x68
        ]
        row = {
            "tag": tag,
            "xy": [s.link_x, s.link_y],
            "mode": s.mode,
            "room": s.screen,
            "tile": int(s.colliding_tile),
            "blocks": blocks,
            "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            **extra,
        }
        log.append(row)
        return s

    if snap.link_x < 48:
        walk_axis(env, assist, total, "x", 48, max_f=200)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    rec("north_wall")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "x", 64, max_f=300)
        rec("nw")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "y", 160, max_f=400)
        rec("left_south")
        walk_axis(env, assist, total, "x", 96, max_f=300)
        rec("under_block")
        push_dir(env, assist, total, "UP", frames=140)
        idle(env, assist, total, 12)
        rec("pushed")

    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "y", 141, max_f=300)
        rec("y141")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "x", 120, max_f=300)
        rec("stand_120_141")
    # CheckWarp only fires while standing still on 0x71.
    for i in range(400):
        s = _rs(env.get_ram())
        if done(s):
            rec("warped", f=i)
            break
        _step(env, assist, total, nes_idle_action())

    if not done(_rs(env.get_ram())):
        for tx, ty in ((120, 141), (128, 141), (112, 141)):
            if done(_rs(env.get_ram())) or _rs(env.get_ram()).screen != ROOM_L5_PASSAGE_06:
                break
            walk_axis(env, assist, total, "y", ty, max_f=160)
            walk_axis(env, assist, total, "x", tx, max_f=160)
            rec("hunt", tgt=[tx, ty])
            for _ in range(300):
                if done(_rs(env.get_ram())):
                    break
                _step(env, assist, total, nes_idle_action())
            if done(_rs(env.get_ram())):
                break

    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    return {
        "path": "north_around_push68_idle_120_141",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "cellar": _in_cellar(snap),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "success": done(snap) and snap.screen != 0x16,
        "south_key_drop": snap.screen == 0x16,
    }


def _raw_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 700) -> bool:
    """Hold one axis toward target. walk_axis bails on cellar mode 9 (stall 40)."""
    last = None
    stall = 0
    from zelda_i.ram import read_snapshot as _rs

    for _ in range(max_f):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_BLUE_64:
            return True
        if axis == "x":
            if abs(snap.link_x - target) <= 1:
                return True
            action = nes_action("RIGHT" if snap.link_x < target else "LEFT")
        else:
            if abs(snap.link_y - target) <= 1:
                return True
            action = nes_action("DOWN" if snap.link_y < target else "UP")
        _step(env, assist, total, action)
        snap2 = _rs(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 180:
                return False
        else:
            stall = 0
        last = pos
    return False


def cellar_to_64(env, assist, total: list[int]) -> dict:
    """From L5 cellar 0x07 (0x06-stairs spawn ~128,141), left mouth → 0x64.

    walk_axis bails on cellar mode 9. Proven live: raw-step to x=192 without
    climbing, drop to pit y=189, floor-cross x=48, climb y=61, UP.
    y=165/x=192/y=61/UP is the 0x06 mouth (do not use it from this spawn).
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}

    def rec(tag):
        s = _rs(env.get_ram())
        log.append({"tag": tag, "xy": [s.link_x, s.link_y], "mode": s.mode, "room": s.screen, "tile": int(s.colliding_tile)})
        return s

    def left_ok(s) -> bool:
        return s.mode == PLAY_MODE and s.screen == ROOM_L5_BLUE_64

    # Reach the right drop without climbing the 0x06 mouth (y<120).
    if snap.link_x >= 80 and snap.link_y < 170:
        for _ in range(480):
            s = _rs(env.get_ram())
            if left_ok(s):
                break
            if abs(s.link_x - L5_CELLAR_RIGHT_X) <= 2 and s.link_y >= 180:
                break
            if s.link_y < 120:
                _step(env, assist, total, nes_action("DOWN"))
            elif abs(s.link_x - L5_CELLAR_RIGHT_X) > 2:
                _step(env, assist, total, nes_action("RIGHT" if s.link_x < L5_CELLAR_RIGHT_X else "LEFT"))
            else:
                _step(env, assist, total, nes_action("DOWN"))
        rec("right_col")
    # Pit floor first. y=165 on the right ladder cannot cross left.
    _raw_axis(env, assist, total, "y", L5_CELLAR_FLOOR_Y, max_f=500)
    rec("pit")
    _raw_axis(env, assist, total, "x", L5_CELLAR_LEFT_X, max_f=700)
    rec("left_col")
    _raw_axis(env, assist, total, "y", 61, max_f=500)
    rec("left_climb")
    push_dir(env, assist, total, "UP", frames=260)
    idle(env, assist, total, 12)
    for _ in range(280):
        snap = _rs(env.get_ram())
        if left_ok(snap):
            break
        if snap.mode == PLAY_MODE and snap.screen not in (ROOM_L5_CELLAR_07,):
            break
        if abs(snap.link_x - L5_CELLAR_LEFT_X) > 4:
            _step(env, assist, total, nes_action("LEFT" if snap.link_x > L5_CELLAR_LEFT_X else "RIGHT"))
        else:
            _step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 16)
    rec("after_up")
    snap = _rs(env.get_ram())
    return {
        "path": "raw_x192_pit189_x48_y61_up",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_BLUE_64 and snap.mode == PLAY_MODE,
    }



def walk_east_from_05(env, assist, total: list[int]) -> dict:
    """Cleared 0x05 east (already unlocked) → 0x06."""
    from zelda_i.dungeon_ops import goto, idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 224, max_f=500)
    goto(env, assist, total, 224, 141, tol=3, max_f=300)
    push_dir(env, assist, total, "RIGHT", frames=240)
    idle(env, assist, total, 16)
    snap = _rs(env.get_ram())
    if snap.screen != ROOM_L5_PASSAGE_06:
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_PASSAGE_06:
                break
            _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    return {
        "path": "east_from_05",
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE,
    }


def take_block_stairs_06(env, assist, total: list[int]) -> dict:
    """0x06: north-around diamond, push 0x68 UP, idle on (96,133) → cellar 0x07.

    Do not walk south (key door 0x16). Do not hunt diamond-center tiles —
    those 0x70–0x73 stands do not warp. Live: block (96,144)→(96,128),
    stairs stand (96,133) (outbound cellar spawn).
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "room": snap.screen,
        "keys": int(snap.keys),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }
    log = [dict(start, tag="start")]

    def done(s) -> bool:
        if s.mode == PLAY_MODE and s.screen == 0x16:
            return False
        if _in_cellar(s):
            return True
        return s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07

    def rec(tag: str) -> None:
        s = _rs(env.get_ram())
        blocks = [
            {"x": o.x, "y": o.y}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id == 0x68
        ]
        log.append(
            {
                "tag": tag,
                "xy": [s.link_x, s.link_y],
                "mode": s.mode,
                "room": s.screen,
                "tile": int(s.colliding_tile),
                "blocks": blocks,
                "keys": int(s.keys),
            }
        )

    if snap.link_x < 48:
        walk_axis(env, assist, total, "x", 48, max_f=240)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    rec("north_wall")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "x", 64, max_f=300)
        rec("nw")
    if not done(_rs(env.get_ram())):
        walk_axis(env, assist, total, "y", 160, max_f=400)
        rec("left_south")
        walk_axis(env, assist, total, "x", ROOM_06_BLOCK_X, max_f=300)
        rec("under_block")
        push_dir(env, assist, total, "UP", frames=140)
        idle(env, assist, total, 8)
        rec("pushed")

    # Live warp: after block is north at (96,128), walk RIGHT from (112,141)
    # onto (128,141). Standing on (96,133) does not warp.
    if not done(_rs(env.get_ram())) and _rs(env.get_ram()).screen == ROOM_L5_PASSAGE_06:
        walk_axis(env, assist, total, "y", 141, max_f=240)
        walk_axis(env, assist, total, "x", 112, max_f=240)
        rec("pre_stair_112_141")
        walk_axis(env, assist, total, "x", 128, max_f=240)
        rec("walk_128_141")
        for direction in ("RIGHT", "UP", "DOWN"):
            if done(_rs(env.get_ram())):
                break
            for _ in range(24):
                snap = _rs(env.get_ram())
                if done(snap) or snap.screen != ROOM_L5_PASSAGE_06:
                    break
                if snap.link_y > 196:
                    _step(env, assist, total, nes_action("UP"))
                    continue
                _step(env, assist, total, nes_action(direction))
            rec(f"nudge_{direction}")

    idle(env, assist, total, 12)
    snap = _rs(env.get_ram())
    cellar = _in_cellar(snap)
    ok = (
        (cellar or (snap.level == LEVEL_5 and snap.screen == ROOM_L5_CELLAR_07))
        and snap.screen != 0x16
    )
    return {
        "path": "push68_right_128_141",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "cellar": cellar,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(snap.keys),
        "success": ok,
        "south_key_drop": snap.screen == 0x16,
    }


def cellar_07_to_64(env, assist, total: list[int]) -> dict:
    """From 0x06-spawned cellar 0x07: right-drop x=192, left-climb x=48 → 0x64."""
    from zelda_i.ram import ADDR_WHISTLE, read_snapshot as _rs, read_u8

    rec = cellar_to_64(env, assist, total)
    snap = _rs(env.get_ram())
    rec["whistle"] = int(read_u8(env.get_ram(), ADDR_WHISTLE))
    rec["keys"] = int(snap.keys)
    return rec


def walk_east_from_64(env, assist, total: list[int]) -> dict:
    """0x64 east bomb-hole → 0x65. North-around the diamond; never (120,141).

    Spawn after cellar is ~(96,157) on the stairs column. Cardinal east/south
    into the diamond drops back into 0x07. Leave west to x=64, north wall
    y=93, east to x=208, then door y=141 RIGHT. South-band fallback if needed.
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}

    def in_cellar(s) -> bool:
        return s.mode in CELLAR_MODES or (s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07)

    def go(steps) -> bool:
        for axis, tgt in steps:
            s = _rs(env.get_ram())
            if in_cellar(s):
                return False
            if s.mode == PLAY_MODE and s.screen == 0x65:
                return True
            walk_axis(env, assist, total, axis, tgt, max_f=450)
            s = _rs(env.get_ram())
            if in_cellar(s):
                return False
            if s.mode == PLAY_MODE and s.screen == 0x65:
                return True
        return not in_cellar(_rs(env.get_ram()))

    # North-around first (same leave used to take 0x06/0x64 stairs).
    ok = go((("x", 64), ("y", 93), ("x", 208), ("y", 141), ("x", 224)))
    via = "north93_east_bombhole"
    if not ok or in_cellar(_rs(env.get_ram())):
        # Still on 0x64: south-band fallback, stay off x=120.
        if _rs(env.get_ram()).screen == ROOM_L5_BLUE_64 and not in_cellar(_rs(env.get_ram())):
            ok = go((("x", 64), ("y", 189), ("x", 208), ("y", 141), ("x", 224)))
            via = "south189_east_bombhole"

    if _rs(env.get_ram()).screen == ROOM_L5_BLUE_64 and not in_cellar(_rs(env.get_ram())):
        push_dir(env, assist, total, "RIGHT", frames=240)
        idle(env, assist, total, 16)
        for _ in range(240):
            snap = _rs(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == 0x65:
                break
            if in_cellar(snap):
                break
            if snap.mode in (6, 7, 4, 16):
                _step(env, assist, total, nes_action("RIGHT"))
            else:
                _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)

    snap = _rs(env.get_ram())
    return {
        "path": via,
        "start": start,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == 0x65 and snap.mode == PLAY_MODE,
    }




EAST65_TO_66_PATHS = (
    ("north109_east", (("y", 109), ("x", 208), ("y", 141), ("x", 224))),
    ("south189_east", (("y", 189), ("x", 208), ("y", 141), ("x", 224))),
    ("y141_east", (("y", 141), ("x", 224))),
)


def walk_east_from_65(env, assist, total: list[int]) -> dict:
    """0x65 east through the existing 0x66-west bomb hole → 0x66.

    Live 0x65 has a center diamond: y=109 then east, not y=141 first.
    North shutter is one-way from 0x55 — do not try UP.
    If the hole is sealed from this side, bomb EAST (same wall).
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "mode": snap.mode,
        "bombs": int(snap.bombs),
    }
    log = [dict(start, tag="start")]
    used = None
    bombed = False

    def at_66(s) -> bool:
        return s.level == LEVEL_5 and s.screen == 0x66 and s.mode == PLAY_MODE

    # Sitting on the closed north shutter ~(120,93): drop to the diamond band.
    if snap.screen == 0x65 and snap.link_y < 109:
        walk_axis(env, assist, total, "y", 109, max_f=300)
        snap = _rs(env.get_ram())
        log.append({"tag": "leave_north_shutter", "xy": [snap.link_x, snap.link_y]})

    for name, steps in EAST65_TO_66_PATHS:
        if at_66(_rs(env.get_ram())):
            used = name
            break
        if _rs(env.get_ram()).screen != 0x65:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=450)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "step": f"{name}:{axis}:{tgt}",
                    "xy": [snap.link_x, snap.link_y],
                    "room": snap.screen,
                    "mode": snap.mode,
                }
            )
            if at_66(snap):
                used = name
                break
        if used:
            break
        snap = _rs(env.get_ram())
        if snap.screen == 0x65 and snap.link_x >= 200:
            push_dir(env, assist, total, "RIGHT", frames=240)
            idle(env, assist, total, 16)
            for _ in range(240):
                snap = _rs(env.get_ram())
                if at_66(snap):
                    used = name
                    break
                if snap.mode in (6, 7, 4, 16):
                    _step(env, assist, total, nes_action("RIGHT"))
                else:
                    _step(env, assist, total, nes_idle_action())
            idle(env, assist, total, 12)
        if at_66(_rs(env.get_ram())):
            used = name
            break

    if not at_66(_rs(env.get_ram())) and _rs(env.get_ram()).screen == 0x65:
        bomb = bomb_east_from_65(env, assist, total)
        bombed = True
        log.append({"step": "bomb_east_fallback", **{k: bomb[k] for k in bomb if k != "menu"}})
        if bomb.get("success"):
            used = "bomb_east_from_65"

    snap = _rs(env.get_ram())
    return {
        "path": used or "walk_east_sealed",
        "start": start,
        "log": log,
        "bombed": bombed,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": at_66(snap),
    }



ROOM_L5_EAST_ZOLS = 0x57
ROOM_L5_NORTH_GIBDOS = 0x47
# 0x57 foes_item statue after a Zol clear. Blocks the north channel at y≈125.
STATUE_5F_TYPE = 0x5F
STATUE_5F_X = 128
STATUE_5F_Y = 128
DIAMOND_NORTH_Y = 109
NORTH_DOOR_Y = 93


def walk_north_from_57(env, assist, total: list[int]) -> dict:
    """0x57 east Zols → 0x47 north Gibdos. ROM N=open.

    Do **not** clear the Zols first. Secret ``foes_item`` drops statue ``0x5f``
    at (128, 128) and Link cannot get north of y≈125 after that.

    Try in this order (L5 Forward walked the north door):
    1. key-north — x=120, y=93, push UP (key spends if the bit is locked)
    2. push the center 0x5f
    3. diamond north-around — y=109 band, then door column
    """
    from zelda_i.dungeon_ops import idle, push_dir
    from zelda_i.ram import read_snapshot as _rs

    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    log = [{"step": "start", "xy": [snap.link_x, snap.link_y], "doors": int(snap.cur_opened_doors), "keys": keys0}]

    def at_47(s) -> bool:
        return s.level == LEVEL_5 and s.screen == ROOM_L5_NORTH_GIBDOS and s.mode == PLAY_MODE

    def push_up(frames: int = 280) -> None:
        push_dir(env, assist, total, "UP", frames=frames)
        idle(env, assist, total, 16)
        for _ in range(240):
            s = _rs(env.get_ram())
            if at_47(s) or s.screen != ROOM_L5_EAST_ZOLS:
                break
            if s.mode in (6, 7, 4, 16):
                _step(env, assist, total, nes_action("UP"))
            else:
                _step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 8)

    # 1. key-north
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", NORTH_DOOR_X, max_f=500)
    walk_axis(env, assist, total, "y", NORTH_DOOR_Y, max_f=400)
    snap = _rs(env.get_ram())
    log.append({"step": "key_north_stand", "xy": [snap.link_x, snap.link_y], "doors": int(snap.cur_opened_doors)})
    push_up(280)
    snap = _rs(env.get_ram())
    log.append({"step": "key_north", "xy": [snap.link_x, snap.link_y], "room": snap.screen, "keys": int(snap.keys)})
    if at_47(snap):
        return {
            "path": "key_north",
            "log": log,
            "keys_in": keys0,
            "keys_out": int(snap.keys),
            "key_spent": int(snap.keys) < keys0,
            "dest": snap.screen,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "success": True,
        }

    # 2. push center 0x5f
    statue = None
    for o in snap.objects:
        if 1 <= o.slot <= 12 and o.type_id == STATUE_5F_TYPE:
            statue = o
            break
    if statue is not None or snap.link_y > 120:
        walk_axis(env, assist, total, "y", STATUE_5F_Y + 16, max_f=300)
        walk_axis(env, assist, total, "x", STATUE_5F_X, max_f=300)
        walk_axis(env, assist, total, "y", STATUE_5F_Y, max_f=200)
        push_dir(env, assist, total, "UP", frames=80)
        idle(env, assist, total, 8)
        snap = _rs(env.get_ram())
        log.append({"step": "push_5f", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        walk_axis(env, assist, total, "x", NORTH_DOOR_X, max_f=300)
        walk_axis(env, assist, total, "y", NORTH_DOOR_Y, max_f=300)
        push_up(240)
        snap = _rs(env.get_ram())
        log.append({"step": "after_push_5f", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        if at_47(snap):
            return {
                "path": "push_5f",
                "log": log,
                "keys_in": keys0,
                "keys_out": int(snap.keys),
                "key_spent": int(snap.keys) < keys0,
                "dest": snap.screen,
                "xy": [snap.link_x, snap.link_y],
                "mode": snap.mode,
                "success": True,
            }

    # 3. diamond north-around
    for name, steps in (
        ("west109", (("y", DIAMOND_NORTH_Y), ("x", 96), ("x", NORTH_DOOR_X), ("y", NORTH_DOOR_Y))),
        ("east109", (("y", DIAMOND_NORTH_Y), ("x", 160), ("x", NORTH_DOOR_X), ("y", NORTH_DOOR_Y))),
    ):
        if at_47(_rs(env.get_ram())):
            break
        if _rs(env.get_ram()).screen != ROOM_L5_EAST_ZOLS:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=400)
            snap = _rs(env.get_ram())
            log.append({"step": f"{name}:{axis}:{tgt}", "xy": [snap.link_x, snap.link_y], "room": snap.screen})
            if at_47(snap):
                break
        push_up(280)
        snap = _rs(env.get_ram())
        log.append({"step": name, "xy": [snap.link_x, snap.link_y], "room": snap.screen})
        if at_47(snap):
            return {
                "path": f"diamond_{name}",
                "log": log,
                "keys_in": keys0,
                "keys_out": int(snap.keys),
                "key_spent": int(snap.keys) < keys0,
                "dest": snap.screen,
                "xy": [snap.link_x, snap.link_y],
                "mode": snap.mode,
                "success": True,
            }

    snap = _rs(env.get_ram())
    return {
        "path": "north_blocked",
        "log": log,
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": at_47(snap),
    }




__all__ = [
    "EAST_DOOR_APPROACH_Y",
    "EAST_DOOR_CHANNEL_Y",
    "EAST_DOOR_WALL_X",
    "NORTH_DOOR_X",
    "SOUTH_PINCH_Y",
    "WEST_DOOR_X",
    "WEST_LEAVE_EAST_X",
    "Level5West26From27Controller",
    "Level5West65Controller",
    "level5_east_key_step",
    "level5_west26_from_27_step",
    "level5_west65_step",
    "make_west26_from_27_controller",
    "walk_west_from_27",
    "walk_west_from_26",
    "walk_west_from_25",
    "walk_axis",
    "WEST26_TO_25_PATHS",
    "WEST25_TO_24_PATHS",
    "make_west65_controller",
    "should_force_keys_zero",
    "ROOM_L5_BLUE_64",
    "ROOM_L5_CELLAR_07",
    "ROOM_L5_PASSAGE_06",
    "ROOM_L5_WHISTLE_05",
    "ROOM_L5_WHISTLE_ITEM",
    "BLUE_DARKNUT_TYPE",
    "BOMB_WEST_STAND",
    "CENTER_STAIRS",
    "CELLAR_MODES",
    "bomb_west_from_65",
    "bomb_east_from_65",
    "take_center_stairs_64",
    "cellar_other_mouth",
    "key_west_to",
    "fight_blue_darknuts",
    "push_block_stairs",
    "hunt_whistle",
    "take_whistle_04",
    "select_b_item_menu",
    "exit_whistle_04",
    "leave_whistle_cellar",
    "walk_out_of_04",
    "take_center_stairs_06",
    "_raw_axis",
    "cellar_to_64",
    "walk_east_from_05",
    "walk_east_from_64",
    "walk_north_from_57",
    "ROOM_L5_EAST_ZOLS",
    "ROOM_L5_NORTH_GIBDOS",
    "STATUE_5F_TYPE",
    "STATUE_5F_X",
    "STATUE_5F_Y",
    "DIAMOND_NORTH_Y",
    "NORTH_DOOR_Y",
    "walk_east_from_65",
    "EAST65_TO_66_PATHS",
    "take_block_stairs_06",
    "cellar_07_to_64",
    "WHISTLE_04_LADDER_X",
    "WHISTLE_04_PIT_Y",
    "WHISTLE_04_MOUTH_X",
    "WHISTLE_04_MOUTH_Y",
    "ROOM_06_BLOCK_X",
    "ROOM_06_BLOCK_REST_Y",
    "ROOM_06_BLOCK_PUSHED_Y",
    "ROOM_06_STAIRS_X",
    "ROOM_06_STAIRS_Y",
]
