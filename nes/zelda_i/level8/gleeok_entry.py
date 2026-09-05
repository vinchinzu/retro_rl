"""Play 0x4C leftover → bomb the north wall. Dest is RAM (hyp 0x3C).

Do not walk the centre stairs (return to cellar 0x2F). Do not fight.
OccupancyWalker banned. Pause-select already-owned bombs onto B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.dungeon.pause_select import B_SLOT_BOMBS
from zelda_i.level8.passage import CELLAR_ROOM, SOURCE_ROOM
from zelda_i.level8.path import GLEEOK_HYP
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "BOMB_NORTH_APPROACH_4C",
    "BOMB_NORTH_STAND",
    "DEST",
    "DEST_HYP",
    "DEST_POSE",
    "RAM_CLAIM",
    "ORIGIN",
    "ORIGIN_POSE",
    "STAIRS_TILES",
    "UP_BIT",
    "Level8BombNorth4CController",
    "make_bomb_north_4c_controller",
]

LEVEL8 = 8
ORIGIN = 0x4C
ORIGIN_POSE = (112, 125)
DEST_HYP = 0x3C  # confirmed live N7; RAM body 0x45 HP160
DEST = 0x3C  # live $EB from 0x4C bomb-N; south mouth
DEST_POSE = (120, 189)  # live N7 arrival
BOMB_NORTH_STAND = (120, 93)  # N6 live north-wall pose; 0x3E (120,105) overshoots into the alcove here
# N3: LEFT at y=117 reaches (64,117) then boxes LEFT into tile 178
# hunting x=48. First wp is that live tile; then south-east around.
BOMB_NORTH_APPROACH_4C: tuple[tuple[int, int], ...] = (
    (64, 117),
    (64, 157),
    (176, 157),
    (176, 109),
    (120, 109),
)
STAIRS_TILES = range(0x70, 0x74)
UP_BIT = 0x08
_SAMPLE_PERIOD = 12
_MAX_FRAMES = 8000

RAM_CLAIM = (
    "From play 0x4C leftover (112,125), do not stand on the centre stairs. "
    "Walk the diamond-maze perimeter to the north-wall bomb stand (120,93) "
    "via (120,109), one bomb facing UP, first settled play $EB is RAM (hyp "
    "0x3C, NOT 0x2F, NOT 0x3F). Bombs 6->5, keys 8->8, MK 1, TF 0x7F. "
    "OccupancyWalker banned."
)


class BombWall4CNorth:
    """0x4C north wall. opens_to is hyp until live dest is locked."""

    room = ORIGIN
    stand = BOMB_NORTH_STAND
    face = "UP"

    def __init__(self, opens_to: int) -> None:
        self.opens_to = int(opens_to)


def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "magic_key": int(getattr(snap, "magic_key", 0)),
        "triforce": int(snap.triforce),
        "doors": int(snap.cur_opened_doors),
    }


@dataclass(kw_only=True)
class Level8BombNorth4CController(HopController):
    """0x4C leftover → bomb-N. Dest is RAM; fail 0x2F / 0x3F. No sword."""

    spec_id: str = "level8_bomb_north_4c"
    max_frames: int = _MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x4c_north"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0
    _env: Any = field(default=None, init=False, repr=False)
    _wall: BombWallController | None = field(default=None, init=False, repr=False)
    _doors0: int | None = field(default=None, init=False)
    _bombs0: int | None = field(default=None, init=False)

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def bind_env(self, env: Any) -> None:
        self._env = env
        if self._wall is not None:
            self._wall.bind_env(env)

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if snap.screen in (CELLAR_ROOM, SOURCE_ROOM, ORIGIN):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_action("UP"), "north_scroll")

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == PASSAGE_MODE or snap.screen == CELLAR_ROOM:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if snap.screen == SOURCE_ROOM:
            return self.mark_fail("returned_0x3f")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == ORIGIN
            and int(snap.colliding_tile) in STAIRS_TILES
        ):
            return self.mark_fail("centre_stairs")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == ORIGIN
            and self._bombs0 is not None
            and int(snap.bombs) >= int(self._bombs0)
            and int(snap.cur_opened_doors) & UP_BIT
            and not (int(self._doors0 or 0) & UP_BIT)
        ):
            return self.mark_fail("north_kill_clear_not_bomb")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != ORIGIN
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if self._doors0 is None:
            self._doors0 = int(snap.cur_opened_doors)
            self._bombs0 = int(snap.bombs)
        if snap.bombs <= 0:
            return self.mark_fail("no_bombs")
        if self._wall is None:
            opens = int(self.dest) if self.dest is not None else GLEEOK_HYP
            self._wall = BombWallController(
                wall=BombWall4CNorth(opens_to=opens),
                level=LEVEL8,
                approach_waypoints=BOMB_NORTH_APPROACH_4C,
                max_frames=self.max_frames,
                require_bomb_consumed=True,
                select_item=B_SLOT_BOMBS,
            )
            if self._env is not None:
                self._wall.bind_env(self._env)
        action = self._wall.step(snap)
        if self._wall.phase is BombWallPhase.FAILED:
            note = (
                self._wall.notes[-1]
                if self._wall.notes
                else "bomb_north_failed"
            )
            return self.mark_fail(note)
        if self._wall.success and not self.arrived(snap):
            return FrameAction(nes_idle_action(), "wall_done")
        return action

    def report(self) -> dict[str, Any]:
        wall = None if self._wall is None else self._wall.report()
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "policy": RAM_CLAIM,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "UP",
            "gate": "bomb_north",
            "leftover": dict(self.leftover),
            "bomb_wall": wall,
        }


def make_bomb_north_4c_controller(
    *, dest: int | None = DEST,
) -> Level8BombNorth4CController:
    return Level8BombNorth4CController(dest=dest)
