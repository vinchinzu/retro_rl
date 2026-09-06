"""Fixture-live Level 9 Magical-Key prefix dest hops.

0x76 leftover → north door UP. Dest is RAM (hyp 0x66 Old Man TF gate).
Natural-spine factories in ``natural_path`` stay fail-closed.
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
)
from zelda_i.combat import should_swing_at
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.level9.patra import patra_action
from zelda_i.level9.dungeon import LEVEL9, ROOM_LEVEL9_ENTRY, ROOM_OLD_MAN_TF, ROOM_RED_RING_HYP
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

NORTH_DOOR = DOOR_TARGETS["UP"]  # (120, 93)
WEST_DOOR = DOOR_TARGETS["LEFT"]  # (32, 141)
EAST_DOOR = DOOR_TARGETS["RIGHT"]  # (224, 141)
NORTH_ORIGIN = ROOM_LEVEL9_ENTRY  # 0x76
NORTH_DEST_HYP = ROOM_OLD_MAN_TF  # 0x66
NORTH_DEST_POSE = (120, 205)  # live P1/P2 south mouth
WEST_ORIGIN = ROOM_OLD_MAN_TF  # 0x66
WEST_DEST_HYP = 0x65
WEST_DEST_POSE = (224, 141)  # live W1 east mouth
BOMB_NORTH_ORIGIN = 0x65
BOMB_NORTH_DEST_HYP = 0x55
BOMB_NORTH_DEST_POSE = (120, 189)  # live south mouth of room 0x55
BOMB_NORTH_STAND = (120, 93)
BOMB_NORTH_APPROACH = ((208, 141), (208, 93), (120, 93))
STAIRS_55_ORIGIN = 0x55
STAIRS_55_DEST_HYP = 0x60
STAIRS_55_DEST_POSE = (192, 93)  # right ladder in cellar 0x60
STAIRS_55_PUSH_X = 96
STAIRS_55_PUSH_BLOCK_Y = 128
STAIRS_55_STAIR_X = 128
STAIRS_55_STAIR_Y = 141
STAIRS_55_STAIR_TILE = (128, 141)
CELLAR_60_ORIGIN = 0x60
CELLAR_60_SOURCE_RETURN = 0x55
CELLAR_60_DEST_HYP = 0x14
CELLAR_60_DEST_POSE = (96, 157)  # live C1/C2 staircase arrival
CELLAR_60_WEST_X = 48
CELLAR_60_EAST_X = 192
CELLAR_60_FLOOR_Y = 189
CELLAR_60_MOUTH_Y = 93
CELLAR_70_ORIGIN = 0x70
CELLAR_70_SOURCE_RETURN = 0x05
CELLAR_70_DEST_HYP = 0x63
CELLAR_70_START_POSE = (192, 93)
CELLAR_70_DEST_POSE = (160, 157)  # live staircase emergence in room 0x63
CELLAR_70_WEST_X = 48
CELLAR_70_EAST_X = 192
CELLAR_70_FLOOR_Y = 189
CELLAR_70_MOUTH_Y = 93
EAST_14_ORIGIN = 0x14
EAST_14_DEST_HYP = 0x15
EAST_14_DEST_POSE = (16, 141)  # live E1/E2 west mouth arrival
EAST_14_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (176, 157),
    (176, 93),
    (32, 93),
    (32, 189),
    (208, 189),
    (208, 141),
    (224, 141),
)
EAST_15_ORIGIN = 0x15
EAST_15_DEST_HYP = 0x16
EAST_15_START_POSE = (16, 141)
EAST_15_DEST_POSE = (32, 141)  # live E15 west mouth arrival
STAIRS_TILES = range(0x70, 0x74)
RED_RING = ROOM_RED_RING_HYP  # 0x07
_DOOR_TOL = 4
_SAMPLE_PERIOD = 12
_MAX_FRAMES = 4000
_DEBUG_05 = bool(os.environ.get("L9_DEBUG_05"))

def is_north_neighbor(origin: int, dest: int) -> bool:
    """Same column, one dungeon row north (``$EB - 0x10``)."""
    return dest == origin - 0x10

def is_west_neighbor(origin: int, dest: int) -> bool:
    """Same row, one dungeon column west (``$EB - 1``)."""
    return dest == origin - 1

def is_east_neighbor(origin: int, dest: int) -> bool:
    """Same row, one dungeon column east (``$EB + 1``)."""
    return dest == origin + 1

def north_76_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align to 120, then UP. No occupancy, no sword, no LEFT/RIGHT push."""
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=NORTH_DOOR[0],
        x_tol=_DOOR_TOL,
        reason="north_76",
    )

def west_66_step(snap: ZeldaSnapshot) -> FrameAction:
    """y-align to 141, then LEFT. No occupancy, no sword, no UP."""
    return dungeon_align_then_push(
        snap,
        push_dir="LEFT",
        target_y=WEST_DOOR[1],
        door_plane=WEST_DOOR[0],
        y_tol=_DOOR_TOL,
        reason="west_66",
    )

NORTH_16_ORIGIN = 0x16
NORTH_16_DEST_HYP = 0x06
NORTH_16_START_POSE = (16, 141)
NORTH_16_DEST_POSE = (120, 205)  # south mouth of room 0x06
BOMB_WEST_06_ORIGIN = 0x06
BOMB_WEST_06_DEST_HYP = 0x05
BOMB_WEST_06_START_POSE = (120, 205)
BOMB_WEST_06_DEST_POSE = (208, 141)  # east mouth of room 0x05
BOMB_WEST_06_STAND = (48, 141)
BOMB_WEST_06_APPROACH: tuple[tuple[int, int], ...] = (
    (120, 189),
    (48, 189),
    (48, 141),
)
STAIRS_05_ORIGIN = 0x05
STAIRS_05_DEST_HYP = 0x70
STAIRS_05_START_POSE = (208, 173)
STAIRS_05_DEST_POSE = (192, 93)  # right ladder in cellar 0x70
STAIRS_05_PUSH_X = 96
STAIRS_05_PUSH_BLOCK_Y = 128
STAIRS_05_STAIR_X = 208
STAIRS_05_STAIR_Y = 96
WEST_63_ORIGIN = 0x63
WEST_63_DEST_HYP = 0x62
WEST_63_START_POSE = (160, 157)
WEST_63_DEST_POSE = (224, 141)  # east mouth of room 0x62
WEST_62_ORIGIN = 0x62
WEST_62_DEST_HYP = 0x61
WEST_62_START_POSE = (224, 141)
WEST_62_DEST_POSE = (224, 141)  # east mouth of room 0x61
STAIRS_61_ORIGIN = 0x61
STAIRS_61_DEST_HYP = 0x75
STAIRS_61_START_POSE = (224, 141)
STAIRS_61_DEST_POSE = (192, 93)  # right ladder in cellar 0x75
STAIRS_61_PUSH_X = 96
STAIRS_61_PUSH_BLOCK_Y = 128
STAIRS_61_STAIR_X = 128
STAIRS_61_STAIR_Y = 141
CELLAR_75_ORIGIN = 0x75
CELLAR_75_SOURCE_RETURN = 0x61
CELLAR_75_DEST_HYP = 0x20
CELLAR_75_START_POSE = (192, 93)
CELLAR_75_DEST_POSE = (96, 157)  # live staircase emergence in room 0x20
CELLAR_75_WEST_X = 48
CELLAR_75_EAST_X = 192
CELLAR_75_FLOOR_Y = 189
CELLAR_75_MOUTH_Y = 93
BOMB_NORTH_20_ORIGIN = 0x20
BOMB_NORTH_20_DEST_HYP = 0x10
BOMB_NORTH_20_START_POSE = (96, 157)
BOMB_NORTH_20_DEST_POSE = (120, 189)  # south mouth of room 0x10
BOMB_NORTH_20_STAND = (120, 93)
BOMB_NORTH_20_APPROACH: tuple[tuple[int, int], ...] = (
    (96, 189),
    (176, 189),
    (176, 93),
    (120, 93),
)

def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "triforce": int(snap.triforce),
    }

@dataclass(kw_only=True)
class Level9PrefixHopController(HopController):
    """Shared base for Level 9 prefix hop controllers."""

    max_frames: int = _MAX_FRAMES
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    origin: int = 0
    dest_hyp: int = 0
    door_dir: str = "UP"

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action(self.door_dir), f"{self.door_dir.lower()}_scroll")

    def valid_dest(self, screen: int) -> bool:
        return screen == self.dest_hyp

    def invalid_dest_note(self, screen: int) -> str:
        return f"unexpected_dest_0x{screen:02x}"

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, self.origin):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return self.valid_dest(snap.screen)

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.origin
        ):
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if not self.valid_dest(snap.screen):
                return self.mark_fail(self.invalid_dest_note(snap.screen))
        return None

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "dest_hyp": self.dest_hyp,
            "door": self.door_dir,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "leftover": dict(self.leftover),
        }

@dataclass(kw_only=True)
class Level9North76Controller(Level9PrefixHopController):
    """0x76 leftover → north door UP. Dest is RAM; fail non-north / 0x07."""

    spec_id: str = "level9_north_76"
    done_reason: str = "left_0x76_north"
    origin: int = NORTH_ORIGIN
    dest_hyp: int = NORTH_DEST_HYP
    door_dir: str = "UP"

    def valid_dest(self, screen: int) -> bool:
        return is_north_neighbor(self.origin, screen)

    def invalid_dest_note(self, screen: int) -> str:
        return f"not_north_neighbor_0x{screen:02x}"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("UP"), "north_settle")
        return north_76_step(snap)

@dataclass(kw_only=True)
class Level9WestDoorHopController(Level9PrefixHopController):
    """Shared base for west door hops (0x66 -> 0x65, 0x63 -> 0x62, 0x62 -> 0x61)."""

    door_dir: str = "LEFT"

    def valid_dest(self, screen: int) -> bool:
        return is_west_neighbor(self.origin, screen)

    def invalid_dest_note(self, screen: int) -> str:
        return f"not_west_neighbor_0x{screen:02x}"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("LEFT"), "west_settle")
        return dungeon_align_then_push(
            snap,
            push_dir="LEFT",
            target_y=WEST_DOOR[1],
            door_plane=WEST_DOOR[0],
            y_tol=_DOOR_TOL,
            reason=self.spec_id,
        )

@dataclass(kw_only=True)
class Level9West66Controller(Level9WestDoorHopController):
    """0x66 leftover → west shutter LEFT. Dest is RAM; fail non-west / 0x07."""

    spec_id: str = "level9_west_66"
    done_reason: str = "left_0x66_west"
    origin: int = WEST_ORIGIN
    dest_hyp: int = WEST_DEST_HYP

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("LEFT"), "west_settle")
        return west_66_step(snap)

@dataclass(kw_only=True)
class Level9West63Controller(Level9WestDoorHopController):
    """0x63 leftover -> west key door LEFT -> play 0x62 (8 Keese)."""

    spec_id: str = "level9_west_63"
    done_reason: str = "settled_play_0x62"
    origin: int = WEST_63_ORIGIN
    dest_hyp: int = WEST_63_DEST_HYP

@dataclass(kw_only=True)
class Level9West62Controller(Level9WestDoorHopController):
    """0x62 leftover -> open west door LEFT -> play 0x61 (Patra room)."""

    spec_id: str = "level9_west_62"
    done_reason: str = "settled_play_0x61"
    origin: int = WEST_62_ORIGIN
    dest_hyp: int = WEST_62_DEST_HYP

def make_north_76_controller(*, dest: int | None = None) -> Level9North76Controller:
    return Level9North76Controller(dest=dest)

def make_west_66_controller(*, dest: int | None = None) -> Level9West66Controller:
    return Level9West66Controller(dest=dest)

def make_west_63_controller(*, dest: int | None = None) -> Level9West63Controller:
    return Level9West63Controller(dest=dest)

def make_west_62_controller(*, dest: int | None = None) -> Level9West62Controller:
    return Level9West62Controller(dest=dest)

@dataclass(kw_only=True)
class Level9BombWallHopController(Level9PrefixHopController):
    """Shared base for bomb wall dest hop controllers."""

    _bomb_wall: BombWallController = field(init=False, repr=False)
    stand_coords: tuple[int, int] = (0, 0)
    approach_waypoints: tuple[tuple[int, int], ...] = ()

    def init_bomb_wall(self) -> None:
        wall = SimpleNamespace(
            room=self.origin,
            stand=self.stand_coords,
            face=self.door_dir,
            opens_to=self.dest_hyp if self.dest is None else self.dest,
        )
        self._bomb_wall = BombWallController(
            wall=wall,
            level=self.require_level or LEVEL9,
            approach_waypoints=self.approach_waypoints,
            approach_tol=_DOOR_TOL,
            stand_tol=_DOOR_TOL,
            face_frames=4,
            step_back=6,
            wait_blast=100,
            wait_hold_face=False,
            require_bomb_consumed=True,
            max_frames=self.max_frames,
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action(self.door_dir), f"{self.door_dir.lower()}_settle")
        act = self._bomb_wall.step(snap)
        self.notes.extend(n for n in self._bomb_wall.notes if n not in self.notes)
        if self._bomb_wall.phase == BombWallPhase.FAILED:
            return self.mark_fail(
                self._bomb_wall.notes[-1] if self._bomb_wall.notes else "bomb_wall_failed"
            )
        return act

    def report(self) -> dict[str, Any]:
        rep = super().report()
        bomb_report = self._bomb_wall.report()
        rep.update({
            "stand": list(self.stand_coords),
            "bombs_before_place": bomb_report.get("bombs_before_place"),
            "bombs_after_place": bomb_report.get("bombs_after_place"),
        })
        return rep

@dataclass(kw_only=True)
class Level9BombNorth65Controller(Level9BombWallHopController):
    """0x65 leftover -> bomb north wall UP. Dest is RAM (hyp 0x55 Lanmola)."""

    spec_id: str = "level9_bomb_north_65"
    done_reason: str = "left_0x65_bomb_north"
    origin: int = BOMB_NORTH_ORIGIN
    dest_hyp: int = BOMB_NORTH_DEST_HYP
    door_dir: str = "UP"
    stand_coords: tuple[int, int] = BOMB_NORTH_STAND
    approach_waypoints: tuple[tuple[int, int], ...] = BOMB_NORTH_APPROACH

    def __post_init__(self) -> None:
        self.init_bomb_wall()

    def valid_dest(self, screen: int) -> bool:
        return is_north_neighbor(self.origin, screen)

    def invalid_dest_note(self, screen: int) -> str:
        return f"not_north_neighbor_0x{screen:02x}"

def make_bomb_north_65_controller(*, dest: int | None = None) -> Level9BombNorth65Controller:
    return Level9BombNorth65Controller(dest=dest)

@dataclass(kw_only=True)
class Level9BombNorth20Controller(Level9BombWallHopController):
    """0x20 leftover -> approach north stand (120, 93) -> bomb UP -> play 0x10."""

    spec_id: str = "level9_bomb_north_20"
    done_reason: str = "settled_play_0x10"
    origin: int = BOMB_NORTH_20_ORIGIN
    dest_hyp: int = BOMB_NORTH_20_DEST_HYP
    door_dir: str = "UP"
    stand_coords: tuple[int, int] = BOMB_NORTH_20_STAND
    approach_waypoints: tuple[tuple[int, int], ...] = BOMB_NORTH_20_APPROACH

    def __post_init__(self) -> None:
        self.init_bomb_wall()

    def valid_dest(self, screen: int) -> bool:
        return is_north_neighbor(self.origin, screen)

    def invalid_dest_note(self, screen: int) -> str:
        return f"not_north_neighbor_0x{screen:02x}"

def make_bomb_north_20_controller(*, dest: int | None = None) -> Level9BombNorth20Controller:
    return Level9BombNorth20Controller(dest=dest)

@dataclass(kw_only=True)
class Level9StairsHopController(Level9PrefixHopController):
    """Base controller for underworld cellar stairs hops (0x55 -> 0x60, 0x05 -> 0x70)."""

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PASSAGE_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, self.origin):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen == self.dest_hyp

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"cellar_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_idle_action(), "cellar_enter_scroll")

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.origin
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        if (
            snap.mode == PASSAGE_MODE
            and not snap.transitioning
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_cellar_0x{snap.screen:02x}")
        return None

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep.pop("door", None)
        return rep

@dataclass(kw_only=True)
class Level9Stairs55Controller(Level9StairsHopController):
    """0x55 leftover -> clear Lanmola -> push block UP -> stairs -> cellar 0x60."""

    spec_id: str = "level9_stairs_55"
    done_reason: str = "settled_cellar_0x60"
    origin: int = STAIRS_55_ORIGIN
    dest_hyp: int = STAIRS_55_DEST_HYP
    _cleared: bool = False
    _pushed: bool = False

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")

        live_segs = [o for o in snap.objects if o.type_id == 0x3A and o.hp > 0]
        if live_segs:
            self._cleared = False
            nearest = min(
                live_segs,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            dist = abs(dx) + abs(dy)
            if dist < 45:
                if abs(dx) > abs(dy):
                    d = "RIGHT" if dx > 0 else "LEFT"
                else:
                    d = "DOWN" if dy > 0 else "UP"
                return FrameAction(
                    nes_action(d, "A") if self.frames % 5 == 0 else nes_action(d),
                    "lanmola_slash",
                )
            return FrameAction(
                nes_action("UP", "A") if self.frames % 8 == 0 else nes_idle_action(),
                "lanmola_wait",
            )

        self._cleared = True

        block = next(
            (o for o in snap.objects if o.type_id == 0x68 or o.slot == 11),
            None,
        )
        if block is not None and block.y > STAIRS_55_PUSH_BLOCK_Y:
            self._pushed = False
            if snap.link_y < 185 and snap.link_x != STAIRS_55_PUSH_X:
                return FrameAction(nes_action("DOWN"), "recenter_y")
            if snap.link_x > STAIRS_55_PUSH_X:
                return FrameAction(nes_action("LEFT"), "align_push_x")
            if snap.link_x < STAIRS_55_PUSH_X:
                return FrameAction(nes_action("RIGHT"), "align_push_x")
            return FrameAction(nes_action("UP"), "push_block_up")

        self._pushed = True

        if snap.link_x < STAIRS_55_STAIR_X:
            return FrameAction(nes_action("RIGHT"), "walk_stair_x")
        if snap.link_y < STAIRS_55_STAIR_Y:
            return FrameAction(nes_action("DOWN"), "walk_stair_y")
        if snap.link_y > STAIRS_55_STAIR_Y:
            return FrameAction(nes_action("UP"), "walk_stair_y")
        return FrameAction(nes_idle_action(), "stand_on_stairs")

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep.update({
            "cleared": self._cleared,
            "pushed": self._pushed,
            "stairs": list(STAIRS_55_STAIR_TILE),
        })
        return rep

def make_stairs_55_controller(*, dest: int | None = None) -> Level9Stairs55Controller:
    return Level9Stairs55Controller(dest=dest)

def cellar_60_step(snap: ZeldaSnapshot) -> FrameAction:
    """DOWN right column to floor, LEFT along floor to x=48, UP west ladder."""
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    on_west = abs(x - CELLAR_60_WEST_X) <= _DOOR_TOL
    on_floor = y >= CELLAR_60_FLOOR_Y - _DOOR_TOL

    if on_floor:
        if x > CELLAR_60_WEST_X + _DOOR_TOL:
            return FrameAction(nes_action("LEFT"), "cellar_floor_west")
        if x < CELLAR_60_WEST_X - _DOOR_TOL:
            return FrameAction(nes_action("RIGHT"), "cellar_floor_east")
        return FrameAction(nes_action("UP"), "cellar_west_climb")

    if on_west:
        if y > CELLAR_60_MOUTH_Y + _DOOR_TOL:
            return FrameAction(nes_action("UP"), "cellar_west_up")
        if tile in STAIRS_TILES:
            return FrameAction(nes_idle_action(), "cellar_exit_warp")
        return FrameAction(nes_action("UP"), "cellar_west_lip")

    if x < CELLAR_60_EAST_X - _DOOR_TOL:
        return FrameAction(nes_action("RIGHT"), "cellar_to_east")
    return FrameAction(nes_action("DOWN"), "cellar_east_drop")

@dataclass(kw_only=True)
class Level9CellarHopController(Level9PrefixHopController):
    """Shared base for underworld cellar traversal hops (0x60 -> 0x14, 0x70 -> 0x63)."""

    door_dir: str = "STAIRS"
    source_return: int = 0
    on_floor: bool = False

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (RED_RING, self.origin, self.source_return):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen == self.dest_hyp

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_idle_action(), "cellar_exit_scroll")

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.screen == RED_RING:
            return self.mark_fail("red_ring_0x07")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.screen == self.source_return:
                return self.mark_fail(f"returned_source_0x{self.source_return:02x}")
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            if snap.screen != self.dest_hyp:
                return self.mark_fail(f"unexpected_dest_0x{snap.screen:02x}")
        if (
            snap.mode == PASSAGE_MODE
            and not snap.transitioning
            and snap.screen != self.origin
        ):
            return self.mark_fail(f"unexpected_cellar_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode == PLAY_MODE and not snap.transitioning:
            return FrameAction(nes_idle_action(), "wait_play_settle")
        if snap.mode != PASSAGE_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")

        if snap.link_y >= CELLAR_60_FLOOR_Y - _DOOR_TOL:
            self.on_floor = True

        act = cellar_60_step(snap)
        if act.reason.endswith(("_up", "_climb", "_lip")) and snap.link_x >= 128:
            return self.mark_fail("up_on_source_ladder")
        return act

@dataclass(kw_only=True)
class Level9Cellar60Controller(Level9CellarHopController):
    """0x60 cellar right ladder leftover -> floor LEFT -> west ladder UP -> play 0x14."""

    spec_id: str = "level9_cellar_60"
    done_reason: str = "emerged_play_0x14"
    origin: int = CELLAR_60_ORIGIN
    dest_hyp: int = CELLAR_60_DEST_HYP
    source_return: int = CELLAR_60_SOURCE_RETURN

def make_cellar_60_controller(*, dest: int | None = None) -> Level9Cellar60Controller:
    return Level9Cellar60Controller(dest=dest)

@dataclass(kw_only=True)
class Level9Cellar70Controller(Level9CellarHopController):
    """0x70 cellar right ladder leftover -> floor LEFT -> west ladder UP -> play 0x63."""

    spec_id: str = "level9_cellar_70"
    done_reason: str = "emerged_play_0x63"
    origin: int = CELLAR_70_ORIGIN
    dest_hyp: int = CELLAR_70_DEST_HYP
    source_return: int = CELLAR_70_SOURCE_RETURN

def make_cellar_70_controller(*, dest: int | None = None) -> Level9Cellar70Controller:
    return Level9Cellar70Controller(dest=dest)

@dataclass(kw_only=True)
class Level9Cellar75Controller(Level9CellarHopController):
    """0x75 cellar right ladder leftover -> floor LEFT -> west ladder UP -> play 0x20."""

    spec_id: str = "level9_cellar_75"
    done_reason: str = "emerged_play_0x20"
    origin: int = CELLAR_75_ORIGIN
    dest_hyp: int = CELLAR_75_DEST_HYP
    source_return: int = CELLAR_75_SOURCE_RETURN

def make_cellar_75_controller(*, dest: int | None = None) -> Level9Cellar75Controller:
    return Level9Cellar75Controller(dest=dest)

@dataclass(kw_only=True)
class Level9East14Controller(Level9PrefixHopController):
    """0x14 leftover -> perimeter walk -> east door RIGHT -> play 0x15."""

    spec_id: str = "level9_east_14"
    done_reason: str = "left_0x14_east"
    origin: int = EAST_14_ORIGIN
    dest_hyp: int = EAST_14_DEST_HYP
    door_dir: str = "RIGHT"
    _wp_index: int = 0

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("RIGHT"), "east_settle")

        for obj in snap.objects:
            if obj.type_id == 0x17 and obj.hp > 0:
                dist = abs(obj.x - snap.link_x) + abs(obj.y - snap.link_y)
                if dist <= 24 and self.frames % 4 == 0:
                    dx = obj.x - snap.link_x
                    dy = obj.y - snap.link_y
                    d = "RIGHT" if abs(dx) > abs(dy) and dx > 0 else (
                        "LEFT" if abs(dx) > abs(dy) else ("DOWN" if dy > 0 else "UP")
                    )
                    return FrameAction(nes_action(d, "A"), "like_like_slash")

        if self._wp_index < len(EAST_14_WAYPOINTS):
            tx, ty = EAST_14_WAYPOINTS[self._wp_index]
            dx = tx - snap.link_x
            dy = ty - snap.link_y
            if abs(dx) <= _DOOR_TOL and abs(dy) <= _DOOR_TOL:
                self._wp_index += 1
                return FrameAction(nes_idle_action(), f"reach_wp_{self._wp_index}")

            if abs(dx) > 2:
                btn = "RIGHT" if dx > 0 else "LEFT"
                return FrameAction(
                    nes_action(btn, "A") if self.frames % 16 == 0 else nes_action(btn),
                    f"walk_wp_{self._wp_index}_x",
                )
            btn = "DOWN" if dy > 0 else "UP"
            return FrameAction(
                nes_action(btn, "A") if self.frames % 16 == 0 else nes_action(btn),
                f"walk_wp_{self._wp_index}_y",
            )

        return FrameAction(nes_action("RIGHT"), "push_east_door")

def make_east_14_controller(*, dest: int | None = None) -> Level9East14Controller:
    return Level9East14Controller(dest=dest)

def east_15_step(snap: ZeldaSnapshot) -> FrameAction:
    """y-align to 141, then RIGHT to door."""
    return dungeon_align_then_push(
        snap,
        push_dir="RIGHT",
        target_y=EAST_DOOR[1],
        door_plane=EAST_DOOR[0],
        y_tol=_DOOR_TOL,
        reason="east_15",
    )

@dataclass(kw_only=True)
class Level9East15Controller(Level9PrefixHopController):
    """0x15 leftover -> hold RIGHT -> play 0x16."""

    spec_id: str = "level9_east_15"
    done_reason: str = "left_0x15_east"
    origin: int = EAST_15_ORIGIN
    dest_hyp: int = EAST_15_DEST_HYP
    door_dir: str = "RIGHT"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("RIGHT"), "east_settle")

        if self.frames % 8 == 0:
            return FrameAction(nes_action("RIGHT", "A"), "east_15_slash")
        return east_15_step(snap)

def make_east_15_controller(*, dest: int | None = None) -> Level9East15Controller:
    return Level9East15Controller(dest=dest)

def north_16_step(snap: ZeldaSnapshot) -> FrameAction:
    """x-align to 120, then UP to door."""
    return dungeon_align_then_push(
        snap,
        push_dir="UP",
        target_x=NORTH_DOOR[0],
        x_tol=_DOOR_TOL,
        reason="north_16",
    )

@dataclass(kw_only=True)
class Level9North16Controller(Level9PrefixHopController):
    """0x16 leftover -> x-align 120 -> hold UP -> play 0x06 (Old Man hint)."""

    spec_id: str = "level9_north_16"
    done_reason: str = "left_0x16_north"
    origin: int = NORTH_16_ORIGIN
    dest_hyp: int = NORTH_16_DEST_HYP
    door_dir: str = "UP"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_action("UP"), "north_settle")
        return north_16_step(snap)

def make_north_16_controller(*, dest: int | None = None) -> Level9North16Controller:
    return Level9North16Controller(dest=dest)

@dataclass(kw_only=True)
class Level9BombWest06Controller(Level9BombWallHopController):
    """0x06 leftover -> approach west stand (48, 141) -> bomb LEFT -> play 0x05."""

    spec_id: str = "level9_bomb_west_06"
    done_reason: str = "left_0x06_bomb_west"
    origin: int = BOMB_WEST_06_ORIGIN
    dest_hyp: int = BOMB_WEST_06_DEST_HYP
    door_dir: str = "LEFT"
    stand_coords: tuple[int, int] = BOMB_WEST_06_STAND
    approach_waypoints: tuple[tuple[int, int], ...] = BOMB_WEST_06_APPROACH

    def __post_init__(self) -> None:
        self.init_bomb_wall()

    def valid_dest(self, screen: int) -> bool:
        return is_west_neighbor(self.origin, screen)

    def invalid_dest_note(self, screen: int) -> str:
        return f"not_west_neighbor_0x{screen:02x}"

def make_bomb_west_06_controller(*, dest: int | None = None) -> Level9BombWest06Controller:
    return Level9BombWest06Controller(dest=dest)

@dataclass(kw_only=True)
class Level9Stairs05Controller(Level9StairsHopController):
    """0x05 leftover -> clear/avoid foes -> push block (96, 144) UP -> stairs -> cellar 0x70."""

    spec_id: str = "level9_stairs_05"
    done_reason: str = "settled_cellar_0x70"
    origin: int = STAIRS_05_ORIGIN
    dest_hyp: int = STAIRS_05_DEST_HYP
    # Power-on evidence (rr-sz8.6, 2026-09-06): room 0x05 has 5 live blue/orange
    # Wizzrobes (type 0x23/0x24). A blind chase-and-mash-A policy landed 0
    # kills in 12000f (never actually checked the sword hitbox), and ignoring
    # them entirely got Link knocked back to nearly the same spot forever
    # (UnlimitedHealthAssist prevents death, not knockback). Fix: proper
    # should_swing_at-gated combat (only swing when the hitbox actually
    # overlaps) with a backstep-when-stuck-too-close fallback, ported from
    # level6.wizzrobe.Level6EastKeyController -- verified live (fast-iteration
    # pin L9Room05EntryReal) to clear all 5 in ~1700f, well inside budget.
    max_frames: int = 12_000
    _cleared: bool = False
    _pushed: bool = False
    _wizz_prev_count: int = -1
    _wizz_last_progress_frame: int = 0
    _wizz_backstep_frames: int = 0
    _recentered_push_y: bool = False

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")

        block = next(
            (o for o in snap.objects if o.type_id == 0x68 or o.slot == 11),
            None,
        )

        live_wizz = [
            o for o in snap.objects
            if o.type_id in (0x23, 0x24) and o.hp > 0
        ]
        if _DEBUG_05 and self.frames % 25 == 0:
            print(
                f"[stairs05 dbg] f{self.frames} xy=({snap.link_x},{snap.link_y}) "
                f"mode={snap.mode} ium={snap.is_updating_mode} trans={snap.transitioning} "
                f"health={snap.health:#x} sword={snap.sword} "
                f"block={(block.x, block.y, block.type_id, block.slot, block.hp) if block else None} "
                f"wizz={[(w.x, w.y, w.hp, w.state) for w in live_wizz]} "
                f"objs={[(o.slot, o.type_id, o.x, o.y, o.hp, o.state) for o in snap.objects]}",
                flush=True,
            )

        # Full-clear before touching the block: nothing left to knock Link
        # off the push stand-off once this branch is done.
        if live_wizz:
            n_live = len(live_wizz)
            if self._wizz_prev_count < 0:
                self._wizz_prev_count = n_live
                self._wizz_last_progress_frame = self.frames
            elif n_live < self._wizz_prev_count:
                self._wizz_prev_count = n_live
                self._wizz_last_progress_frame = self.frames
                self._wizz_backstep_frames = 0

            nearest = min(
                live_wizz,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            stuck_close = (
                dist < 16 and (self.frames - self._wizz_last_progress_frame) > 100
            )
            if stuck_close or self._wizz_backstep_frames > 0:
                if self._wizz_backstep_frames <= 0:
                    self._wizz_backstep_frames = 24
                self._wizz_backstep_frames -= 1
                if self._wizz_backstep_frames == 0:
                    self._wizz_last_progress_frame = self.frames
                dx = nearest.x - snap.link_x
                dy = nearest.y - snap.link_y
                if abs(dx) >= abs(dy):
                    d = "LEFT" if dx >= 0 else "RIGHT"
                else:
                    d = "UP" if dy >= 0 else "DOWN"
                return FrameAction(nes_action(d), "wizzrobe_backstep")

            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) > abs(dy):
                direction = "RIGHT" if dx > 0 else "LEFT"
            else:
                direction = "DOWN" if dy > 0 else "UP"
            if should_swing_at(snap.link_x, snap.link_y, direction, live_wizz):
                return FrameAction(nes_action(direction, "A"), "wizzrobe_engage_slash")
            return FrameAction(nes_action(direction), "wizzrobe_engage")

        if block is not None and block.y > STAIRS_05_PUSH_BLOCK_Y:
            self._pushed = False
            if not self._recentered_push_y:
                if snap.link_y < 165:
                    return FrameAction(nes_action("DOWN"), "recenter_y")
                self._recentered_push_y = True
            if abs(snap.link_x - STAIRS_05_PUSH_X) > 4:
                d = "LEFT" if snap.link_x > STAIRS_05_PUSH_X else "RIGHT"
                return FrameAction(nes_action(d), "align_push_x")
            return FrameAction(nes_action("UP"), "push_block_up")

        self._pushed = True

        if snap.link_x < STAIRS_05_STAIR_X:
            if snap.link_y < 173:
                return FrameAction(nes_action("DOWN"), "walk_stair_south_aisle")
            return FrameAction(nes_action("RIGHT"), "walk_stair_x")
        if snap.link_y > STAIRS_05_STAIR_Y:
            return FrameAction(nes_action("UP"), "walk_stair_y")
        return FrameAction(nes_action("UP"), "stand_on_stairs")

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep.update({
            "pushed": self._pushed,
            "stairs": [STAIRS_05_STAIR_X, STAIRS_05_STAIR_Y],
        })
        return rep

def make_stairs_05_controller(*, dest: int | None = None) -> Level9Stairs05Controller:
    return Level9Stairs05Controller(dest=dest)

@dataclass(kw_only=True)
class Level9Stairs61Controller(Level9StairsHopController):
    """0x61 leftover -> clear Patra -> push block (96, 144) UP -> stairs (128, 141) -> cellar 0x75."""

    spec_id: str = "level9_stairs_61"
    done_reason: str = "settled_cellar_0x75"
    origin: int = STAIRS_61_ORIGIN
    dest_hyp: int = STAIRS_61_DEST_HYP
    # Power-on evidence (rr-sz8.6/.7, 2026-09-06): this "other Patra" room has
    # block/wall geometry the final-Patra room (0x52) doesn't, so the proven
    # south-stand-and-pulse policy lands hits much slower here (~1 eye per
    # ~4000f against the real power-on pin L9Room61EntryReal, vs ~180f/eye in
    # 0x52) -- budget generously (same lesson as stairs_05) rather than
    # re-tune the policy for speed.
    max_frames: int = 20_000
    _cleared: bool = False
    _pushed: bool = False
    _stage: int = 0
    _patra_cooldown: int = 0
    _stuck_xy: tuple[int, int] | None = None
    _stuck_frames: int = 0
    _stuck_escape_frames: int = 0
    _escape_dir: str = "UP"
    _patra_seen: bool = False

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")

        if self._stage == 0:
            patra_eyes = [o for o in snap.objects if o.type_id == 0x25 and o.hp > 0]
            patra_body = next((o for o in snap.objects if o.type_id == 0x47 and o.hp > 0), None)
            if patra_eyes or patra_body is not None:
                self._patra_seen = True
            # Root cause (rr-sz8.6/.7, 2026-09-06): the real hop-transition
            # snapshot lands on the exact frame the room loads, before Patra
            # has spawned (body registers frame 1, eyes 2 frames later --
            # same spawn race LEVEL9_ROUTE.md documents for room 0x52's
            # WAIT_PATRA phase). Reading "no live eyes/body" on that very
            # first frame falsely looked like an already-cleared room, so
            # this jumped straight to the block push while Patra was still
            # fully alive -- confirmed via a corrected power-on pin
            # (L9Room61EntryReal, captured at the true hop-transition frame
            # instead of 60 frames late) reproducing the exact live failure
            # (stuck push-looping at (32,93) with all 8 eyes alive). Require
            # having actually observed Patra at least once before trusting
            # a "cleared" reading.
            if self._patra_seen and not patra_eyes and patra_body is None:
                self._cleared = True
                self._stage = 1
            else:
                # Same south-stand-and-pulse policy proven live for the final
                # Patra (room 0x52, patra.py): distance-gated mash-A here
                # landed 0 hits in 387f against the real power-on pin
                # (L9Room61EntryReal) -- never checked the sword hitbox, same
                # bug class fixed for stairs_05's Wizzrobes.
                #
                # Room 0x61's block/wall geometry (absent in 0x52) can pin
                # Link against an obstacle while patra_action keeps re-issuing
                # the same axis-align command every frame (RNG-dependent --
                # a different eye/body trajectory than the one used to tune
                # this can wedge Link somewhere the south-stand target can't
                # reach directly). Detect a long no-progress stall and step
                # toward the room's open center to break free before
                # resuming the proven policy, rather than let it loop forever.
                if self._stuck_escape_frames > 0:
                    self._stuck_escape_frames -= 1
                    return FrameAction(nes_action(self._escape_dir), "patra_stuck_escape")
                xy = (int(snap.link_x), int(snap.link_y))
                if xy == self._stuck_xy:
                    self._stuck_frames += 1
                else:
                    self._stuck_xy = xy
                    self._stuck_frames = 0
                if self._stuck_frames > 90:
                    dx, dy = 120 - snap.link_x, 133 - snap.link_y
                    if abs(dx) >= abs(dy):
                        self._escape_dir = "RIGHT" if dx > 0 else "LEFT"
                    else:
                        self._escape_dir = "DOWN" if dy > 0 else "UP"
                    self._stuck_escape_frames = 20
                    self._stuck_frames = 0
                    return FrameAction(nes_action(self._escape_dir), "patra_stuck_escape")
                action, reason, self._patra_cooldown = patra_action(
                    snap, cooldown=self._patra_cooldown,
                )
                return FrameAction(action, reason)

        if self._stage == 1:
            if snap.link_x <= 64:
                self._stage = 2
            else:
                return FrameAction(nes_action("LEFT"), "nav_west_aisle")

        if self._stage == 2:
            if snap.link_y >= 189:
                self._stage = 3
            else:
                return FrameAction(nes_action("DOWN"), "nav_south_aisle")

        if self._stage == 3:
            if snap.link_x >= STAIRS_61_PUSH_X:
                self._stage = 4
            else:
                return FrameAction(nes_action("RIGHT"), "nav_push_align")

        if self._stage == 4:
            block = next((o for o in snap.objects if o.slot == 11), None)
            if block and block.y <= STAIRS_61_PUSH_BLOCK_Y:
                self._pushed = True
                self._stage = 5
            else:
                return FrameAction(nes_action("UP"), "push_block_up")

        if self._stage == 5:
            if snap.link_x >= STAIRS_61_STAIR_X:
                self._stage = 6
            else:
                return FrameAction(nes_action("RIGHT"), "walk_stair_x")

        return FrameAction(nes_action("UP"), "stand_on_stairs")

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep.update({"pushed": self._pushed, "stairs": [STAIRS_61_STAIR_X, STAIRS_61_STAIR_Y]})
        return rep

def make_stairs_61_controller(*, dest: int | None = None) -> Level9Stairs61Controller:
    return Level9Stairs61Controller(dest=dest)

__all__ = [
    "BOMB_NORTH_APPROACH", "BOMB_NORTH_DEST_HYP", "BOMB_NORTH_DEST_POSE",
    "BOMB_NORTH_ORIGIN", "BOMB_NORTH_STAND",
    "BOMB_NORTH_20_APPROACH", "BOMB_NORTH_20_DEST_HYP", "BOMB_NORTH_20_DEST_POSE",
    "BOMB_NORTH_20_ORIGIN", "BOMB_NORTH_20_STAND", "BOMB_NORTH_20_START_POSE",
    "BOMB_WEST_06_APPROACH", "BOMB_WEST_06_DEST_HYP", "BOMB_WEST_06_DEST_POSE",
    "BOMB_WEST_06_ORIGIN", "BOMB_WEST_06_STAND", "BOMB_WEST_06_START_POSE",
    "CELLAR_60_DEST_HYP", "CELLAR_60_DEST_POSE", "CELLAR_60_EAST_X",
    "CELLAR_60_FLOOR_Y", "CELLAR_60_MOUTH_Y", "CELLAR_60_ORIGIN",
    "CELLAR_60_SOURCE_RETURN", "CELLAR_60_WEST_X",
    "CELLAR_70_DEST_HYP", "CELLAR_70_DEST_POSE", "CELLAR_70_EAST_X",
    "CELLAR_70_FLOOR_Y", "CELLAR_70_MOUTH_Y", "CELLAR_70_ORIGIN",
    "CELLAR_70_SOURCE_RETURN", "CELLAR_70_WEST_X",
    "CELLAR_75_DEST_HYP", "CELLAR_75_DEST_POSE", "CELLAR_75_EAST_X",
    "CELLAR_75_FLOOR_Y", "CELLAR_75_MOUTH_Y", "CELLAR_75_ORIGIN",
    "CELLAR_75_SOURCE_RETURN", "CELLAR_75_WEST_X",
    "EAST_14_DEST_HYP", "EAST_14_DEST_POSE", "EAST_14_ORIGIN", "EAST_14_WAYPOINTS",
    "EAST_15_DEST_HYP", "EAST_15_DEST_POSE", "EAST_15_ORIGIN", "EAST_15_START_POSE",
    "NORTH_16_DEST_HYP", "NORTH_16_DEST_POSE", "NORTH_16_ORIGIN", "NORTH_16_START_POSE",
    "NORTH_DEST_HYP", "NORTH_DEST_POSE", "NORTH_DOOR", "NORTH_ORIGIN", "RED_RING",
    "STAIRS_05_DEST_HYP", "STAIRS_05_DEST_POSE", "STAIRS_05_ORIGIN",
    "STAIRS_05_PUSH_BLOCK_Y", "STAIRS_05_PUSH_X", "STAIRS_05_STAIR_X",
    "STAIRS_05_STAIR_Y", "STAIRS_05_START_POSE",
    "STAIRS_55_DEST_HYP", "STAIRS_55_DEST_POSE", "STAIRS_55_ORIGIN",
    "STAIRS_55_PUSH_BLOCK_Y", "STAIRS_55_PUSH_X", "STAIRS_55_STAIR_TILE",
    "STAIRS_55_STAIR_X", "STAIRS_55_STAIR_Y", "STAIRS_TILES",
    "STAIRS_61_DEST_HYP", "STAIRS_61_DEST_POSE", "STAIRS_61_ORIGIN",
    "STAIRS_61_PUSH_BLOCK_Y", "STAIRS_61_PUSH_X", "STAIRS_61_STAIR_X",
    "STAIRS_61_STAIR_Y", "STAIRS_61_START_POSE",
    "WEST_62_DEST_HYP", "WEST_62_DEST_POSE", "WEST_62_ORIGIN", "WEST_62_START_POSE",
    "WEST_63_DEST_HYP", "WEST_63_DEST_POSE", "WEST_63_ORIGIN", "WEST_63_START_POSE",
    "WEST_DEST_HYP", "WEST_DEST_POSE", "WEST_DOOR", "WEST_ORIGIN",
    "Level9BombNorth20Controller", "Level9BombNorth65Controller", "Level9BombWest06Controller",
    "Level9Cellar60Controller", "Level9Cellar70Controller", "Level9Cellar75Controller",
    "Level9East14Controller", "Level9East15Controller", "Level9North16Controller",
    "Level9North76Controller", "Level9PrefixHopController",
    "Level9Stairs05Controller", "Level9Stairs55Controller", "Level9Stairs61Controller",
    "Level9West62Controller", "Level9West63Controller", "Level9West66Controller",
    "cellar_60_step", "east_15_step", "is_east_neighbor", "is_north_neighbor", "is_west_neighbor",
    "make_bomb_north_20_controller", "make_bomb_north_65_controller", "make_bomb_west_06_controller",
    "make_cellar_60_controller", "make_cellar_70_controller", "make_cellar_75_controller",
    "make_east_14_controller", "make_east_15_controller", "make_north_16_controller",
    "make_north_76_controller", "make_stairs_05_controller", "make_stairs_55_controller",
    "make_stairs_61_controller", "make_west_62_controller", "make_west_63_controller",
    "make_west_66_controller", "north_16_step", "north_76_step", "west_66_step",
]

