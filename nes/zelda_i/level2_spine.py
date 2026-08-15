"""Level 2 Survival-spine path: live 0x7d through Magical Boomerang 0x4f.

Room specs and stop predicates stay in ``level2_dungeon``. This module owns
the continuous-spine orchestration and the named nav controllers that
``GenericDungeonRoomController`` + spec waypoints cannot do (backtrack,
west-entry 0x6e, diamond-east key door).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import GenericDungeonRoomController, RewardKind, RewardSpec
from zelda_i.level2_bomb_path import (
    make_bomb_north_controller,
    make_boom_bomb_north_controller,
)
from zelda_i.level2_dungeon import (
    ROOM_4F_SPEC,
    ROOM_6C_SPEC,
    ROOM_6D_SPEC,
    ROOM_6E_SPEC,
    ROOM_6F_SPEC,
    ROOM_7E_SPEC,
    ROOM_L2_COMPASS,
    ROOM_L2_EAST_KEY,
    ROOM_L2_EAST_OF_ROPES,
    ROOM_L2_ENTRY,
    ROOM_L2_ROPES,
    ROOM_L2_WEST_KEY,
)
from zelda_i.nav_common import (
    DIAMOND_BAND_6E,
    DIAMOND_BAND_7D,
    diamond_east_phase,
    track_stuck,
    unstick_wiggle,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

DOOR_X = 120
DOOR_Y = 141
# Live 2026-08-15: ALIGN_TOL=6 pushed RIGHT at y=135 into the 0x6c wall and
# timed out at (128, 133). East/west mouths need |y-141|≤2 (L3 uses ≤4).
ALIGN_TOL = 2
STUCK_THRESHOLD = 24
BACKTRACK_7D_MAX_FRAMES = 3000
ENTER_6E_WEST_MAX_FRAMES = 4000
ENTER_6F_KEY_MAX_FRAMES = 4000
# Mid-room box used to leave the 0x7d east alcove after 0x7e LEFT.
_MID_X = (70, 180)
_MID_Y = (110, 175)

Hop = tuple[int, str, int]

# Isolated 7e walks onto the key mid-fight from keys=0. The spine already
# holds the west key, so Generic idles at |delta|≤5 on target (136,141) and
# misses a 5px-off floor key (live timeout at (141,141), keys still 1).
ROOM_7E_SPINE_SPEC = replace(
    ROOM_7E_SPEC,
    spec_id="level2_room7e_east_key_spine",
    reward=RewardSpec(
        kind=RewardKind.FIXED_INVENTORY,
        inventory_field="keys",
        target=(136, 141),
        waypoints=(
            (136, 141),
            (120, 141),
            (136, 125),
            (152, 141),
            (136, 157),
            (104, 141),
            (168, 141),
        ),
        reward_while_live=True,
    ),
)


class Level2NavPhase(Enum):
    WALK = auto()
    DONE = auto()
    FAILED = auto()


def level2_through_success(snap: ZeldaSnapshot) -> bool:
    """``through=level2`` stop: Magical Boomerang owned, not merely Moon 0x7d."""
    return int(snap.magical_boomerang) != 0


def _in_mid_room(snap: ZeldaSnapshot) -> bool:
    return _MID_X[0] <= snap.link_x <= _MID_X[1] and _MID_Y[0] <= snap.link_y <= _MID_Y[1]


def _align_then_push(snap: ZeldaSnapshot, direction: str) -> FrameAction:
    """Door-column align, then hold the cardinal through the doorway."""
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - DOOR_Y) > ALIGN_TOL:
            return FrameAction(
                nes_action("DOWN" if snap.link_y < DOOR_Y else "UP"),
                "align_door_y",
            )
        return FrameAction(nes_action(direction), "push_door")
    if abs(snap.link_x - DOOR_X) > ALIGN_TOL:
        return FrameAction(
            nes_action("RIGHT" if snap.link_x < DOOR_X else "LEFT"),
            "align_door_x",
        )
    return FrameAction(nes_action(direction), "push_door")


@dataclass
class Level2RoomWalkController:
    """Ordered dungeon door hops. Optional diamond-free on one source room."""

    dest_room: int
    hops: tuple[Hop, ...]
    max_frames: int
    diamond_free_room: int | None = None
    phase: Level2NavPhase = Level2NavPhase.WALK
    frames: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)
    _last_dir: str = "RIGHT"
    _stuck: int = 0
    _last_x: int = -1
    _last_y: int = -1
    _last_screen: int = -1

    def _set_phase(self, phase: Level2NavPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Level2NavPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self._stuck, self._last_x, self._last_y, self._last_screen = track_stuck(
            snap,
            last_x=self._last_x,
            last_y=self._last_y,
            last_screen=self._last_screen,
            stuck=self._stuck,
        )
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.screen == self.dest_room and snap.mode == PLAY_MODE:
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "arrived")
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return FrameAction(nes_action(self._last_dir), "room_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

        if (
            self.diamond_free_room is not None
            and snap.screen == self.diamond_free_room
            and not _in_mid_room(snap)
        ):
            # East alcove of 0x7d: LEFT on y=141 hits the diamond (live timeout
            # at (208,141)). Drop to band y=157 first, then LEFT into mid.
            if snap.screen == ROOM_L2_ENTRY and snap.link_x >= 176:
                if abs(snap.link_y - DIAMOND_BAND_7D) > 4:
                    self._last_dir = "DOWN" if snap.link_y < DIAMOND_BAND_7D else "UP"
                    return FrameAction(nes_action(self._last_dir), "alcove_band")
                self._last_dir = "LEFT"
                return FrameAction(nes_action("LEFT"), "alcove_left")
            action, _ = diamond_east_phase(snap, phase="free", band_y=157)
            self._last_dir = "LEFT"
            return action

        if self._stuck > STUCK_THRESHOLD:
            action, self._stuck = unstick_wiggle(self._stuck, reason="door_unstick")
            return action

        for from_room, direction, _to_room in self.hops:
            if snap.screen == from_room:
                self._last_dir = direction
                return _align_then_push(snap, direction)
        return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "dest_room": self.dest_room,
            "notes": list(self.notes),
        }


@dataclass
class Level2BacktrackTo7dController(Level2RoomWalkController):
    """0x6c RIGHT → 0x6d DOWN → entry 0x7d (after west key)."""

    dest_room: int = ROOM_L2_ENTRY
    hops: tuple[Hop, ...] = (
        (ROOM_L2_WEST_KEY, "RIGHT", ROOM_L2_ROPES),
        (ROOM_L2_ROPES, "DOWN", ROOM_L2_ENTRY),
    )
    max_frames: int = BACKTRACK_7D_MAX_FRAMES


@dataclass
class Level2WestEnter6eController(Level2RoomWalkController):
    """0x7e UP → 0x6e. West-via-0x7d timed out in the east diamond alcove.

    South mouth of 0x6e can stick ~y=181; keep UP until play y≤175.
    """

    dest_room: int = ROOM_L2_EAST_OF_ROPES
    hops: tuple[Hop, ...] = (
        (ROOM_L2_EAST_KEY, "UP", ROOM_L2_EAST_OF_ROPES),
    )
    max_frames: int = ENTER_6E_WEST_MAX_FRAMES
    diamond_free_room: int | None = None


@dataclass
class Level2Enter6fKeyController:
    """0x6e key-RIGHT via diamond-east (band y≈113). No LEFT on the final push.

    Fails honestly when keys==0 (do not poke). Consumes one key.
    """

    dest_room: int = ROOM_L2_COMPASS
    from_room: int = ROOM_L2_EAST_OF_ROPES
    band_y: int = DIAMOND_BAND_6E
    require_keys: int = 1
    max_frames: int = ENTER_6F_KEY_MAX_FRAMES
    phase: Level2NavPhase = Level2NavPhase.WALK
    frames: int = 0
    diamond_phase: str = "free"
    cycle: int = 0
    success: bool = False
    notes: list[str] = field(default_factory=list)

    def _set_phase(self, phase: Level2NavPhase, note: str = "") -> None:
        if phase is not self.phase:
            self.phase = phase
            if note:
                self.notes.append(note)

    def _fail(self, note: str) -> FrameAction:
        self._set_phase(Level2NavPhase.FAILED, note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if snap.mode == 17:
            return self._fail("link_death")
        if self.frames >= self.max_frames:
            return self._fail("timeout")
        if snap.screen == self.dest_room and snap.mode == PLAY_MODE:
            self.success = True
            self._set_phase(Level2NavPhase.DONE, "key_door_entered")
            return FrameAction(nes_idle_action(), "done")
        if snap.mode == 8:
            return FrameAction(nes_idle_action(), "hurt_freeze")
        if snap.transitioning:
            return FrameAction(nes_action("RIGHT"), "room_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.from_room:
            return FrameAction(nes_idle_action(), f"wait_room_0x{snap.screen:02x}")
        if snap.keys < self.require_keys:
            return self._fail("no_keys")

        action, nxt = diamond_east_phase(
            snap,
            phase=self.diamond_phase,
            band_y=self.band_y,
            cycle=self.cycle,
        )
        if nxt == self.diamond_phase:
            self.cycle += 1
        else:
            self.diamond_phase = nxt
            self.cycle = 0
        return action

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "diamond_phase": self.diamond_phase,
            "frames": self.frames,
            "notes": list(self.notes),
        }


def level2_to_boom_stages():
    """Controller table: live 0x7d through Magical Boomerang 0x4f.

    Bomb stages fail with ``no_bombs`` when inventory is 0; sibling
    ``rr-4d53.2.2`` owns the farm. Do not poke bombs, keys, or doors.
    """
    bomb_6f = make_bomb_north_controller()
    bomb_5f = make_boom_bomb_north_controller()
    return (
        ("clear6d", GenericDungeonRoomController(ROOM_6D_SPEC), ROOM_6D_SPEC.max_frames),
        (
            "clear6c_key",
            GenericDungeonRoomController(ROOM_6C_SPEC),
            ROOM_6C_SPEC.max_frames,
        ),
        (
            "backtrack_7d",
            Level2BacktrackTo7dController(),
            BACKTRACK_7D_MAX_FRAMES,
        ),
        (
            "clear7e_key",
            GenericDungeonRoomController(ROOM_7E_SPINE_SPEC),
            ROOM_7E_SPINE_SPEC.max_frames,
        ),
        (
            "enter_6e_west",
            Level2WestEnter6eController(),
            ENTER_6E_WEST_MAX_FRAMES,
        ),
        ("clear6e", GenericDungeonRoomController(ROOM_6E_SPEC), ROOM_6E_SPEC.max_frames),
        (
            "enter_6f_key",
            Level2Enter6fKeyController(),
            ENTER_6F_KEY_MAX_FRAMES,
        ),
        (
            "clear6f_compass",
            GenericDungeonRoomController(ROOM_6F_SPEC),
            ROOM_6F_SPEC.max_frames,
        ),
        ("bomb_north_6f", bomb_6f, bomb_6f.max_frames),
        ("bomb_north_5f", bomb_5f, bomb_5f.max_frames),
        (
            "clear4f_boom",
            GenericDungeonRoomController(ROOM_4F_SPEC),
            ROOM_4F_SPEC.max_frames,
        ),
    )


__all__ = [
    "BACKTRACK_7D_MAX_FRAMES",
    "ENTER_6E_WEST_MAX_FRAMES",
    "ENTER_6F_KEY_MAX_FRAMES",
    "Level2BacktrackTo7dController",
    "Level2Enter6fKeyController",
    "Level2NavPhase",
    "Level2RoomWalkController",
    "Level2WestEnter6eController",
    "ROOM_7E_SPINE_SPEC",
    "level2_through_success",
    "level2_to_boom_stages",
]
