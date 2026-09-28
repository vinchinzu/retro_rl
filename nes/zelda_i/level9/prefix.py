"""Fixture-live Level 9 Magical-Key prefix dest hops.

0x76 leftover → north door UP. Dest is RAM (hyp 0x66 Old Man TF gate).
Natural-spine factories in ``natural_path`` stay fail-closed.
"""

from __future__ import annotations

from zelda_i.dungeon.passage import passage_step

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.hop_controller import (
    HopController,
    WAIT_SCROLL_B,
    dungeon_align_then_push,
    lattice_door_step,
    stairs_step,
)
from zelda_i.dungeon.ops import DOOR_TARGETS
from zelda_i.level9.patra import (
    PATRA_ROOM_FULL,
    PATRA_STAND_DY,
    PatraBlade,
    patra_action,
    patra_body,
    patra_eyes,
)
from zelda_i.level9.dungeon import LEVEL9, ROOM_LEVEL9_ENTRY, ROOM_OLD_MAN_TF, ROOM_RED_RING_HYP, SILVER_ARROWS
from zelda_i.dungeon.engine import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.walk import live_env

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
ROOM_10_ORIGIN = 0x10
STAIRS_05_DEST_HYP = 0x70
STAIRS_05_START_POSE = (208, 173)
STAIRS_05_DEST_POSE = (192, 93)  # right ladder in cellar 0x70
STAIRS_05_PUSH_X = 96
STAIRS_05_PUSH_BLOCK_Y = 128
STAIRS_05_STAIR_X = 208
STAIRS_05_STAIR_Y = 96
# bomb_west_06 drops Link at (208,141) -- standing *in* the hole -- and two
# Wizzrobes camp inside the east wall at x=224, so they are always the nearest
# target and the chase drags him straight back out. Retreat west of this line
# once, before engaging at all.
STAIRS_05_DOOR_ROW_Y = 141
STAIRS_05_DOOR_ROW_TOL = 16
STAIRS_05_OFF_ROW_Y = 173   # the pose the proven clear was tuned from
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
    # A door hop may opt in: a boxed hand walk (a knock onto a moat lip)
    # then falls back to the ROM-lattice route to this hop's door. Not for a
    # Like-Like room (0x14): an engulfed Link reads as boxed, and the lattice
    # press replaced the slash that frees him (4000-frame timeout).
    door_hop: bool = False

    def __post_init__(self) -> None:
        if self.door_hop and self.exit_dir is None:
            self.exit_dir = self.door_dir

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
    return passage_step(snap, east_x=CELLAR_60_EAST_X, align=_DOOR_TOL, mouth_y=CELLAR_60_MOUTH_Y)

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
    _stuck_xy: tuple[int, int] | None = None
    _stuck_frames: int = 0
    _cross_axis_frames: int = 0

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
                self._stuck_xy = None
                self._stuck_frames = 0
                self._cross_axis_frames = 0
                return FrameAction(nes_idle_action(), f"reach_wp_{self._wp_index}")

            # The walk is x-first, which has no way out of a wall. A Like Like
            # bump that leaves Link a few pixels off the y=93 lane puts him
            # LEFT into stone with dx still large, and he held it for 3,373
            # frames at (176,101) until the chapter timed out (live power-on,
            # rr-sz8.7). On a no-progress stall, work the *other* axis toward
            # the same waypoint for a moment, then resume.
            xy = (int(snap.link_x), int(snap.link_y))
            if xy == self._stuck_xy:
                self._stuck_frames += 1
            else:
                self._stuck_xy = xy
                self._stuck_frames = 0
            if self._cross_axis_frames > 0:
                self._cross_axis_frames -= 1
                if abs(dy) > 2:
                    btn = "DOWN" if dy > 0 else "UP"
                elif abs(dx) > 2:
                    btn = "RIGHT" if dx > 0 else "LEFT"
                else:
                    btn = "RIGHT"
                return FrameAction(nes_action(btn), f"walk_wp_{self._wp_index}_unstick")
            if self._stuck_frames > 60:
                self._stuck_frames = 0
                self._cross_axis_frames = 24
                btn = ("DOWN" if dy > 0 else "UP") if abs(dy) > 2 else (
                    "RIGHT" if dx > 0 else "LEFT")
                return FrameAction(nes_action(btn), f"walk_wp_{self._wp_index}_unstick")

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
    door_hop: bool = True

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

@dataclass
class RoomFight:
    """The generic room engine as one phase of a bespoke hop controller.

    Built on first use and rebuilt after a failed engine (a timeout or a
    knock out of the room), so the hop keeps fighting on its own gate. The
    rebuild keeps the wave census: a fresh engine in a room already all dead
    would wait on ``expected_enemy_count`` forever.
    """

    spec: DungeonRoomSpec
    _ctl: GenericDungeonRoomController | None = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        """The engine cleared the room and swept its drops and room item."""
        return self._ctl is not None and self._ctl.success

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self._ctl is None or self._ctl.phase is DungeonPhase.FAILED:
            seen = self._ctl.max_live_enemies if self._ctl is not None else 0
            self._ctl = GenericDungeonRoomController(spec=self.spec)
            self._ctl.max_live_enemies = seen
            env = live_env.current()
            if env is not None:
                self._ctl.bind_env(env)
        return self._ctl.step(snap)


# 0x10's five Wizzrobes (three 0x2B traps never clear). Patrol the south
# corridor Link enters on: scored over 12 RNG offsets from a Blue Ring
# power-on pin, it beat the middle band (1463f / 11.6h) at 703f / 9.2h, and
# the threat evader lost 5 of 12 to a knock out of the room.
ROOM_10_WIZZROBES_SPEC = DungeonRoomSpec(
    spec_id="level9_room10_wizzrobes",
    source_room=0x20,
    room_id=ROOM_10_ORIGIN,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(0x23, 0x24),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE_AND_HP,
    combat=CombatTuning(
        # Off the door column x=120: a hit there knocks Link out south.
        patrol=((72, 189), (168, 189)),
        engage_distance=48,
        attack_phase=2,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
        occupancy_patrol=True,
        occupancy_from_tilemap=True,
        # x=32 is the only bridge between this room's horizontal bands.
        occupancy_bounds=(32, 208, 93, 189),
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=1),
    max_frames=12000,
    level=LEVEL9,
)

# 0x05's five Wizzrobes, entered from the east bomb hole (208,141). They are
# always inside engage range, so the patrol never runs (three patrols tied).
ROOM_05_WIZZROBES_SPEC = replace(
    ROOM_10_WIZZROBES_SPEC,
    spec_id="level9_room05_wizzrobes",
    source_room=0x06,
    room_id=0x05,
    entry=DoorRoute("LEFT", ((208, 141),)),
    combat=replace(ROOM_10_WIZZROBES_SPEC.combat, patrol=((176, 173), (128, 181)), occupancy_bounds=None),
)


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
    # (UnlimitedHealthAssist prevents death, not knockback). The clear now
    # runs on the generic engine (``ROOM_05_WIZZROBES_SPEC``).
    max_frames: int = 12_000
    _cleared: bool = False
    _pushed: bool = False
    _fight: RoomFight = field(
        default_factory=lambda: RoomFight(ROOM_05_WIZZROBES_SPEC), repr=False
    )
    _recentered_push_y: bool = False
    _cleared_east_band: bool = False
    _recenter_frames: int = 0
    _recenter_stuck_y: int = -1
    _recenter_stuck_frames: int = 0
    _recenter_escape_frames: int = 0
    _recenter_escape_dir: str = "LEFT"

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.origin:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")

        # Step off the door row before anything else, and latch it.
        # bomb_west_06 now drops Link at (208,141), standing *in* the hole it
        # blew, with two Wizzrobes camped inside the east wall at (224,141) --
        # always the nearest target, so engaging from there chases him
        # straight back out and the hop fails `unexpected_play_0x06`. Dropping
        # to y=173 restores the pose the proven 5-Wizzrobe clear was measured
        # from (STAIRS_05_START_POSE) without changing the fight itself.
        if not self._cleared_east_band:
            if abs(snap.link_y - STAIRS_05_DOOR_ROW_Y) < STAIRS_05_DOOR_ROW_TOL:
                return FrameAction(nes_action("DOWN"), "leave_east_doorway")
            self._cleared_east_band = True

        block = next(
            (o for o in snap.objects if o.type_id == 0x68 or o.slot == 11),
            None,
        )

        # Full clear before touching the block, on the generic engine and
        # the ROM's all-dead flag. The chase it replaces (also copied into
        # 0x10) read a Wizzrobe teleport gap (hp 0) as a clear; scored over
        # 12 offsets the engine clears in 670f / 6.9h.
        if not snap.room_all_dead:
            return self._fight.step(snap)

        if block is not None and block.y > STAIRS_05_PUSH_BLOCK_Y:
            self._pushed = False
            if not self._recentered_push_y:
                # Getting south of the block is a bare DOWN hold, which has no
                # way out of a wall: from (144,125) it burned 10,597 frames and
                # timed the chapter out (live power-on, rr-sz8.7). Escape
                # sideways on no progress, and give up on the recenter
                # entirely rather than spend the whole budget on it -- the push
                # legs below re-derive from position every frame.
                self._recenter_frames += 1
                if snap.link_y >= 165 or self._recenter_frames > 1200:
                    self._recentered_push_y = True
                else:
                    y = int(snap.link_y)
                    if y == self._recenter_stuck_y:
                        self._recenter_stuck_frames += 1
                    else:
                        self._recenter_stuck_y = y
                        self._recenter_stuck_frames = 0
                    if self._recenter_escape_frames > 0:
                        self._recenter_escape_frames -= 1
                        return FrameAction(
                            nes_action(self._recenter_escape_dir), "recenter_y_escape"
                        )
                    if self._recenter_stuck_frames > 60:
                        self._recenter_stuck_frames = 0
                        self._recenter_escape_frames = 24
                        self._recenter_escape_dir = (
                            "LEFT" if snap.link_x > STAIRS_05_PUSH_X else "RIGHT"
                        )
                        return FrameAction(
                            nes_action(self._recenter_escape_dir), "recenter_y_escape"
                        )
                    return FrameAction(nes_action("DOWN"), "recenter_y")
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

# Room 0x10 holds no floor item at all: its ROM room-attribute item byte reads
# 0x03 (none) and the live `room_item_id` agrees on every entry. What it does
# hold is secret code 5, `block_reveals_stairs` -- the same gating the proven
# 0x05 / 0x30 push-stairs hops use. Pushing the 0x68 block east exposes a
# staircase in the room's north-east cell, and that staircase drops into
# Level 9 cellar 0x4F, whose ROM item byte is 0x09 = Silver Arrow and whose
# two stair mouths both read back to 0x10. Verified live from the
# L9Room10EntryReal pin (rr-sz8.6, 2026-09-06): clear -> push -> stairs ->
# cellar pickup -> return lands `arrows == 2` back in 0x10 in ~5200 frames.
ROOM_10_CELLAR = 0x4F
ROOM_10_SOUTH = 0x20
ROOM_10_BLOCK_TYPE = 0x68
# Statue bands at y~112 and y~176 block every column except the west lane
# x=32, so vertical travel is routed through it; the north band row y=93 and
# the middle band rows y=125-165 are open the full width.
ROOM_10_WEST_X = 32
ROOM_10_NORTH_Y = 93
ROOM_10_MID_Y = 141
ROOM_10_STAIR_X = 208  # revealed stair cell (208, 96)
ROOM_10_PUSH_STAND = (176, 141)  # west of the block, on its row
ROOM_10_PUSHED_X = 200  # block x once shoved clear of (192, 144)
# Cellar 0x4F: stairs spit Link at the west shaft, the floor corridor runs at
# y=189, a second shaft at x=176 climbs into the upper chamber, and the
# Silver Arrow sprite sits at (128, 141) inside it.
CELLAR_4F_FLOOR_Y = 189
CELLAR_4F_SHAFT_X = 176
CELLAR_4F_CHAMBER_Y = 141
CELLAR_4F_ITEM_X = 128
CELLAR_4F_EXIT_X = 48
ROOM_10_SOUTH_Y = 189  # the row the 0x20 bomb-hole doorway sits on
ROOM_10_MOUTH_X = 120  # doorway column back down into 0x20
_R10_TOL = 3


def room10_lane_step(link_x: int, link_y: int, row_y: int) -> str | None:
    """Direction to reach room 0x10's band row ``row_y``, or None once on it.

    The statue bands at y~112 and y~176 block every column except the west
    lane x=32, so all vertical travel in this room is routed through it. The
    lane check is gated on "still need vertical travel": testing x against the
    lane unconditionally would fight the horizontal leg that follows and
    ping-pong forever.
    """
    if abs(link_y - row_y) <= _R10_TOL:
        return None
    if abs(link_x - ROOM_10_WEST_X) > _R10_TOL:
        return "LEFT" if link_x > ROOM_10_WEST_X else "RIGHT"
    return "UP" if link_y > row_y else "DOWN"


@dataclass(kw_only=True)
class Level9Room10SilverArrowsController(HopController):
    """0x10 leftover -> clear Wizzrobes -> push the 0x68 east -> revealed
    stairs -> cellar 0x4F Silver Arrows -> back up into 0x10.

    ``arrived`` is the inventory rising edge back in 0x10, not a dest hop:
    both of cellar 0x4F's mouths return here, so the join that follows still
    starts from a 0x10 leftover.
    """

    spec_id: str = "level9_room10_silver_arrows"
    done_reason: str = "silver_arrows_collected"
    max_frames: int = 16_000
    require_level: int = LEVEL9
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    _fight: RoomFight = field(
        default_factory=lambda: RoomFight(ROOM_10_WIZZROBES_SPEC), repr=False
    )
    _pushed: bool = False
    _on_floor: bool = False

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return (
            snap.screen == ROOM_10_ORIGIN
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.arrows >= SILVER_ARROWS
        )

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.screen == ROOM_10_CELLAR or snap.mode == PASSAGE_MODE:
            return self._cellar_policy(snap)
        if snap.screen == ROOM_10_SOUTH:
            # A Wizzrobe hit knocks Link out through 0x10's open south door;
            # idling in 0x20 cost 201 hearts on the Blue Ring L9 resume.
            step = lattice_door_step(None, snap, "UP")
            return FrameAction(nes_action(step or "UP"), "room10_reenter")
        if snap.screen != ROOM_10_ORIGIN:
            return FrameAction(nes_idle_action(), f"unexpected_screen_0x{snap.screen:02x}")
        return self._room_policy(snap)

    # -- room 0x10 ---------------------------------------------------------
    def _room_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        block = next(
            (o for o in snap.objects if o.type_id == ROOM_10_BLOCK_TYPE), None
        )
        if block is not None and block.x >= ROOM_10_PUSHED_X:
            self._pushed = True
        if not self._pushed:
            if not snap.room_all_dead:
                return self._wizzrobe_combat(snap)
            return self._push_block(snap)
        return self._walk_to_stairs(snap)

    def _wizzrobe_combat(self, snap: ZeldaSnapshot) -> FrameAction:
        # The generic engine (tile-seeded walker, beams, bottom-corridor
        # patrol). The greedy chase it replaces spent 13740 of 16000 frames
        # walking into the statue bands and took 110 hearts (Blue Ring
        # resume); the engine clears in 703f / 9.2h mean over 12 offsets.
        return self._fight.step(snap)

    def _push_block(self, snap: ZeldaSnapshot) -> FrameAction:
        stand_x, stand_y = ROOM_10_PUSH_STAND
        blocked = self._route_to_row(snap, stand_y, "room10_push")
        if blocked is not None:
            return blocked
        if abs(snap.link_x - stand_x) > _R10_TOL:
            d = "LEFT" if snap.link_x > stand_x else "RIGHT"
            return FrameAction(nes_action(d), "room10_push_align_x")
        return FrameAction(nes_action("RIGHT"), "room10_push_block_east")

    def _walk_to_stairs(self, snap: ZeldaSnapshot) -> FrameAction:
        step = stairs_step(None, snap)
        if step is not None:
            return FrameAction(nes_action(step), "room10_stairs_lattice")
        blocked = self._route_to_row(snap, ROOM_10_NORTH_Y, "room10_stairs")
        if blocked is not None:
            return blocked
        if snap.link_x != ROOM_10_STAIR_X:
            d = "LEFT" if snap.link_x > ROOM_10_STAIR_X else "RIGHT"
            return FrameAction(nes_action(d), "room10_stairs_east")
        return FrameAction(nes_idle_action(), "room10_stand_on_stairs")

    def _route_to_row(
        self, snap: ZeldaSnapshot, row_y: int, reason: str
    ) -> FrameAction | None:
        """Cross the statue bands via the west lane. None once y is on ``row_y``."""
        d = room10_lane_step(int(snap.link_x), int(snap.link_y), row_y)
        if d is None:
            return None
        leg = "west_lane" if abs(snap.link_x - ROOM_10_WEST_X) > _R10_TOL else "lane_travel"
        return FrameAction(nes_action(d), f"{reason}_{leg}")

    # -- cellar 0x4F -------------------------------------------------------
    def _cellar_policy(self, snap: ZeldaSnapshot) -> FrameAction:
        """Drop to the floor corridor, climb the x=176 shaft into the chamber,
        take the arrows, then reverse out through the west exit shaft.

        Every leg re-derives itself from the live position instead of latching
        a phase: the cellar Keese knock Link sideways, and a latched leg that
        holds one cardinal into a wall stalls forever (observed at (208,141)).
        The one thing position alone cannot tell us is whether Link has been
        down to the corridor yet -- the chamber and the entry shaft are both
        above it -- so that single fact is latched.
        """
        x, y = int(snap.link_x), int(snap.link_y)

        if snap.arrows >= SILVER_ARROWS:
            # The west exit column and the chamber are both above the floor
            # corridor, so height alone cannot tell them apart -- split on x
            # first, or standing at the top of the exit shaft reads as "still
            # in the chamber" and holds RIGHT into the wall forever.
            if x <= CELLAR_4F_EXIT_X + _R10_TOL:
                # West of the ladder column is floor only; UP there climbs
                # nothing (a Keese knock left Link at (32,189) pressing UP
                # for 13349f on one RNG offset).
                if x < CELLAR_4F_EXIT_X - _R10_TOL:
                    return FrameAction(nes_action("RIGHT"), "cellar4f_exit_align")
                return FrameAction(nes_action("UP"), "cellar4f_climb_out")
            if y < CELLAR_4F_FLOOR_Y - _R10_TOL:
                if abs(x - CELLAR_4F_SHAFT_X) > _R10_TOL:
                    d = "RIGHT" if x < CELLAR_4F_SHAFT_X else "LEFT"
                    return FrameAction(nes_action(d), "cellar4f_back_shaft")
                return FrameAction(nes_action("DOWN"), "cellar4f_back_drop")
            return FrameAction(nes_action("LEFT"), "cellar4f_to_exit")

        if not self._on_floor:
            if y < CELLAR_4F_FLOOR_Y - _R10_TOL:
                return FrameAction(nes_action("DOWN"), "cellar4f_drop")
            self._on_floor = True
        if y > CELLAR_4F_CHAMBER_Y + _R10_TOL:
            if abs(x - CELLAR_4F_SHAFT_X) > _R10_TOL:
                if y < CELLAR_4F_FLOOR_Y - _R10_TOL:
                    return FrameAction(nes_action("DOWN"), "cellar4f_regain_floor")
                d = "RIGHT" if x < CELLAR_4F_SHAFT_X else "LEFT"
                return FrameAction(nes_action(d), "cellar4f_to_shaft")
            return FrameAction(nes_action("UP"), "cellar4f_climb")
        if x > CELLAR_4F_ITEM_X + _R10_TOL:
            return FrameAction(nes_action("LEFT"), "cellar4f_to_item")
        if x < CELLAR_4F_ITEM_X - _R10_TOL:
            return FrameAction(nes_action("RIGHT"), "cellar4f_to_item")
        return FrameAction(nes_idle_action(), "cellar4f_wait_item")

    def report(self) -> dict[str, Any]:
        rep = super().report()
        rep.update({
            "cellar": ROOM_10_CELLAR,
            "pushed": self._pushed,
            "on_cellar_floor": self._on_floor,
        })
        return rep


def make_room10_silver_arrows_controller() -> Level9Room10SilverArrowsController:
    return Level9Room10SilverArrowsController()

# 0x61 after Patra: the engine as a sweep (no census to wait on) takes the
# room's key at (208,96) and any floor drops before the block and stairs.
# The ledger listed that key untaken on every Blue Ring power-on (rr-qb6w).
ROOM_61_SWEEP_SPEC = replace(
    ROOM_10_WIZZROBES_SPEC,
    spec_id="level9_room61_sweep",
    source_room=0x62,
    room_id=0x61,
    entry=DoorRoute("LEFT", ((224, 141),)),
    enemy_types=(0x47, 0x25),
    expected_enemy_count=0,
    max_frames=900,
)


@dataclass(kw_only=True)
class Level9Stairs61Controller(Level9StairsHopController):
    """0x61 leftover -> clear Patra -> push block (96, 144) UP -> stairs (128, 141) -> cellar 0x75."""

    spec_id: str = "level9_stairs_61"
    done_reason: str = "settled_cellar_0x75"
    origin: int = STAIRS_61_ORIGIN
    dest_hyp: int = STAIRS_61_DEST_HYP
    patra_stand_dy: int = PATRA_STAND_DY
    max_frames: int = 20_000
    _cleared: bool = False
    _pushed: bool = False
    _stage: int = 0
    _patra_cooldown: int = 0
    _patra_melee: bool = False
    _blade: PatraBlade = field(
        default_factory=lambda: PatraBlade(box=PATRA_ROOM_FULL), repr=False
    )
    _stuck_xy: tuple[int, int] | None = None
    _stuck_frames: int = 0
    _stuck_escape_frames: int = 0
    _escape_dir: str = "UP"
    _patra_seen: bool = False
    _sweep: RoomFight = field(
        default_factory=lambda: RoomFight(ROOM_61_SWEEP_SPEC), repr=False
    )

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
            # The transition precedes the body/eye spawn; no objects alone
            # cannot prove a clear until Patra has actually been observed.
            if self._patra_seen and not patra_eyes and patra_body is None:
                self._cleared = True
                self._stage = 1
            else:
                # Below full health the sword beam cannot fire.  Keep the
                # melee arm once entered so a mid-fight heart pickup cannot
                # move Link between two unrelated stands.
                self._patra_melee |= not snap.health_is_full
                env = live_env.current()
                if self._patra_melee and env is not None:
                    action, reason = self._blade.step(snap, env)
                    return FrameAction(action, reason)
                # Same south-stand-and-pulse policy proven live for the final
                # Patra (room 0x52, patra.py): distance-gated mash-A here
                # landed 0 hits in 387f against the real power-on pin
                # (L9Room61EntryReal) -- never checked the sword hitbox, same
                # bug class fixed for stairs_05's Wizzrobes.
                #
                # Room 0x61's blocks can pin a walk to the stand. Count only
                # frames the policy walks and Link does not move: standing on
                # the lane is the policy, and an escape on any still frame
                # broke the stand ~340 frames a fight.
                if self._stuck_escape_frames > 0:
                    self._stuck_escape_frames -= 1
                    return FrameAction(nes_action(self._escape_dir), "patra_stuck_escape")
                action, reason, cooldown = patra_action(
                    snap, cooldown=self._patra_cooldown, stand_dy=self.patra_stand_dy,
                    room=PATRA_ROOM_FULL,
                )
                xy = (int(snap.link_x), int(snap.link_y))
                if reason.startswith("align") and xy == self._stuck_xy:
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
                self._patra_cooldown = cooldown
                return FrameAction(action, reason)

        if self._stage == 1 and not self._sweep.done and (
            self._sweep._ctl is None or self._sweep._ctl.phase is not DungeonPhase.FAILED
        ):
            return self._sweep.step(snap)

        if self._stage >= 1:
            # ROM block secret + stair tile first. The west-aisle cardinals
            # below held LEFT at (144,173) for 13726f on the power-on
            # gathered spine; they stay as the fallback.
            step = stairs_step(None, snap)
            if step is not None:
                return FrameAction(nes_action(step), "rom_stairs")

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
    "CELLAR_4F_EXIT_X", "CELLAR_4F_FLOOR_Y", "CELLAR_4F_ITEM_X",
    "CELLAR_4F_SHAFT_X", "ROOM_10_CELLAR", "ROOM_10_ORIGIN",
    "ROOM_10_PUSH_STAND", "ROOM_10_STAIR_X", "ROOM_10_WEST_X",
    "WEST_62_DEST_HYP", "WEST_62_DEST_POSE", "WEST_62_ORIGIN", "WEST_62_START_POSE",
    "WEST_63_DEST_HYP", "WEST_63_DEST_POSE", "WEST_63_ORIGIN", "WEST_63_START_POSE",
    "WEST_DEST_HYP", "WEST_DEST_POSE", "WEST_DOOR", "WEST_ORIGIN",
    "Level9BombNorth20Controller", "Level9BombNorth65Controller", "Level9BombWest06Controller",
    "Level9Cellar60Controller", "Level9Cellar70Controller", "Level9Cellar75Controller",
    "Level9East14Controller", "Level9East15Controller", "Level9North16Controller",
    "Level9North76Controller", "Level9PrefixHopController", "Level9Room10SilverArrowsController",
    "Level9Stairs05Controller", "Level9Stairs55Controller", "Level9Stairs61Controller",
    "Level9West62Controller", "Level9West63Controller", "Level9West66Controller",
    "ROOM_10_MOUTH_X", "ROOM_10_SOUTH_Y",
    "cellar_60_step", "east_15_step", "is_east_neighbor", "is_north_neighbor", "is_west_neighbor",
    "make_bomb_north_20_controller", "make_bomb_north_65_controller", "make_bomb_west_06_controller",
    "make_cellar_60_controller", "make_cellar_70_controller", "make_cellar_75_controller",
    "make_east_14_controller", "make_east_15_controller", "make_north_16_controller",
    "make_north_76_controller", "make_room10_silver_arrows_controller",
    "make_stairs_05_controller", "make_stairs_55_controller",
    "make_stairs_61_controller", "make_west_62_controller", "make_west_63_controller",
    "make_west_66_controller", "north_16_step", "north_76_step", "room10_lane_step",
    "west_66_step",
]
