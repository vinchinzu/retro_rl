"""Level 7 nose cellar 0x7B: B-side spawn, floor-cross west to play 0x29.

ROM AttrA=play 0x29 (left, x=$30) AttrB=play 0x0D (right, x=$C0). InitMode9
from 0x0D spawns the right/source ladder. CheckSubroom UP at Y<$40 and
X>=$80 returns to 0x0D — that is the dead "return-only" miss. Far side is
the left ladder. Inverse of ``cellar_cross_dir`` (which always targets
east_x then UP). OccupancyWalker is banned. No RAM writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import CELLAR_MODE, HopController, WAIT_SCROLL_B
from zelda_i.level7.stairs import (
    CELLAR_LADDER_LEFT_X,
    CELLAR_LADDER_RIGHT_X,
    CHECKSUBROOM_SPLIT_X,
    NOSE_CELLAR_ROM,
    PRE_BOSS_ROM,
    TIP_OF_NOSE_ROM,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "ALIGN",
    "CELLAR_CROSS_MAX_FRAMES",
    "CELLAR_ROOM",
    "DEST_ROOM",
    "EAST_X",
    "EXIT_STAIRS",
    "FLOOR_Y",
    "LEVEL7",
    "MOUTH_Y",
    "PIT_LEDGE_Y",
    "PIT_TILE",
    "RAM_CLAIM",
    "SOURCE_ROOM",
    "SPAWN_XY",
    "STAIRS_TILES",
    "WEST_X",
    "Level7NoseCellarCrossController",
    "make_nose_cellar_cross_controller",
    "nose_cellar_cross_step",
    "nose_cellar_cross_success",
]

LEVEL7 = 7
CELLAR_ROOM = NOSE_CELLAR_ROM  # 0x7B
SOURCE_ROOM = TIP_OF_NOSE_ROM  # 0x0D
DEST_ROOM = PRE_BOSS_ROM  # 0x29
ALIGN = 4
WEST_X = CELLAR_LADDER_LEFT_X  # 0x30 = 48
EAST_X = CELLAR_LADDER_RIGHT_X  # 0xC0 = 192
FLOOR_Y = 189
MOUTH_Y = 93
EXIT_STAIRS = (WEST_X, MOUTH_Y)
SPAWN_XY = (EAST_X, MOUTH_Y)
PIT_TILE = 250
PIT_LEDGE_Y = 141
STAIRS_TILES = range(0x70, 0x74)
CELLAR_CROSS_MAX_FRAMES = 4000
CELLAR_PLAY_MODES = (CELLAR_MODE, 11)
_SAMPLE_PERIOD = 12

# Written before the first live trial. Miss if dest is 0x0D.
RAM_CLAIM = (
    "From Level7Interior0DNoseCellarReconFixture (mode-9 cellar 0x7B, right "
    "ladder x=$C0), DOWN to floor y=189, LEFT to x=48 ($30), UP left ladder. "
    "First settled play $EB is 0x29 (ROM AttrA). Miss if dest is 0x0D "
    "(CheckSubroom AttrB / UP on the source ladder). Never UP at x>=$80."
)


def nose_cellar_cross_step(snap: ZeldaSnapshot) -> FrameAction:
    """DOWN the east/source column, floor LEFT to x=48, UP west. Never source UP.

    ``cellar_cross_dir`` always targets east_x then UP (L6 A→B). This is the
    inverse B→A. Do not LEFT at y=141 if colliding_tile is the L8 pit 250 —
    go RIGHT to the east column and drop there.
    """
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    on_west = abs(x - WEST_X) <= ALIGN
    on_floor = y >= FLOOR_Y - ALIGN
    if on_floor:
        if x > WEST_X + ALIGN:
            return FrameAction(nes_action("LEFT"), "cellar_floor_west")
        if x < WEST_X - ALIGN:
            return FrameAction(nes_action("RIGHT"), "cellar_floor_east")
        return FrameAction(nes_action("UP"), "cellar_west_climb")
    if on_west:
        if y > MOUTH_Y + ALIGN:
            return FrameAction(nes_action("UP"), "cellar_west_up")
        if tile in STAIRS_TILES:
            return FrameAction(nes_idle_action(), "cellar_exit_warp")
        return FrameAction(nes_action("UP"), "cellar_west_lip")
    # Mid-height, not west. Never UP: x>=$80 is CheckSubroom AttrB → 0x0D.
    if tile == PIT_TILE and x < EAST_X - ALIGN:
        return FrameAction(nes_action("RIGHT"), "cellar_pit_to_east")
    if x < EAST_X - ALIGN:
        return FrameAction(nes_action("RIGHT"), "cellar_to_east")
    return FrameAction(nes_action("DOWN"), "cellar_east_drop")


def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "candle": int(snap.candle),
        "triforce": int(snap.triforce),
    }


def nose_cellar_cross_success(snap: ZeldaSnapshot) -> bool:
    """Exact AttrA endpoint 0x29. Source return 0x0D is a failure."""
    return (
        snap.level == LEVEL7
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == DEST_ROOM
    )


@dataclass(kw_only=True)
class Level7NoseCellarCrossController(HopController):
    """0x7B right-spawn leftover, DOWN, floor LEFT, west-ladder UP. Fixture-live."""

    spec_id: str = "level7_nose_cellar_0x7b"
    max_frames: int = CELLAR_CROSS_MAX_FRAMES
    require_level: int = LEVEL7
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "emerged_0x29"
    dest: int | None = DEST_ROOM
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    writes: int = 0
    arrival_seen: bool = False
    on_floor: bool = False

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if self.dest is None:
            return (
                snap.mode == PLAY_MODE
                and not snap.transitioning
                and snap.screen != CELLAR_ROOM
            )
        return nose_cellar_cross_success(snap) and snap.screen == self.dest

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = _leftover(snap)
        if force or self.frames <= 2 or self.frames % _SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "on_floor": self.on_floor,
                    "arrival_seen": self.arrival_seen,
                }
            )
        return action

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        x, y = int(snap.link_x), int(snap.link_y)
        # L8 trap is LEFT at y=141. Floor y=189 and the west ladder can
        # report tile 250 without being that pit (C1 leftover (48,189)).
        if (
            int(snap.colliding_tile) == PIT_TILE
            and abs(y - PIT_LEDGE_Y) <= ALIGN + 4
            and abs(x - WEST_X) > ALIGN
            and y < FLOOR_Y - ALIGN
        ):
            return self.mark_fail("pit_tile_250")
        if snap.mode == PLAY_MODE and not snap.transitioning:
            if snap.screen == SOURCE_ROOM:
                return self.mark_fail("returned_source_0x0d")
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"wrong_play_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), "wait_dest")
        if snap.mode not in CELLAR_PLAY_MODES and snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL7:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.mode == PASSAGE_MODE and snap.screen != CELLAR_ROOM:
            return self.mark_fail("unexpected_cellar")
        if not self.arrival_seen:
            if abs(x - EAST_X) <= ALIGN + 8 and y >= MOUTH_Y - ALIGN:
                self.arrival_seen = True
                self.notes.append(f"b_side_spawn_{x}_{y}")
            else:
                return FrameAction(nes_idle_action(), "passage_init_wait")
        if y >= FLOOR_Y - ALIGN:
            self.on_floor = True
        act = nose_cellar_cross_step(snap)
        if act.reason.endswith(("_up", "_climb", "_lip")) and x >= CHECKSUBROOM_SPLIT_X:
            return self.mark_fail("up_on_source_ladder")
        return act

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "spec_id": self.spec_id,
            "room": CELLAR_ROOM,
            "dest": self.dest,
            "dest_screen": self.dest,
            "policy": RAM_CLAIM,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "STAIRS",
            "leftover": dict(self.leftover),
            "arrival_seen": self.arrival_seen,
            "on_floor": self.on_floor,
        }


def make_nose_cellar_cross_controller(
    *, dest: int | None = DEST_ROOM
) -> Level7NoseCellarCrossController:
    """Cross cellar 0x7B from the 0x0D B-side spawn. Do not climb source UP."""
    return Level7NoseCellarCrossController(dest=dest)
