"""Level 6 play 0x38 east/west door census after dated bomb-south.

Leftover (120,93) north mouth. Reclear of 0x38 is false (no killable
respawn; leftover invuln 0x2B + block 0x68). Cardinal DOWN at (120,189)
tile 170 doors=0 mask=0 is dated. v1 stand (120,181) DOWN and v2
(120,173) DOWN both consumed a bomb (8→7) and did not open the wall.

v3: OccupancyWalker to east mouth then west mouth; short cardinal push
each; record tile/doors/mask. Isolated BFS banned. Ignore 0x2B. Do not
push 0x68. No bomb. No KEY-UP 0x09. No CheckWarp. No reclear. Halt at
first occupancy no-path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon_ops import apply_owned_inventory
from zelda_i.level2_puzzles import DOOR_UP, BombWall
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_TRAPS_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
    LEVEL6_WIZZROBE_38_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "BOMB38_SOUTH_MAX_FRAMES",
    "BOMB_38_SOUTH_STAND",
    "BOMB_WALL_38_SOUTH",
    "CENSUS_PUSH_FRAMES",
    "EAST_DOOR",
    "SOUTH_DOOR_Y",
    "WEST_DOOR",
    "Level6Bomb38SouthController",
    "level6_bomb38_south_stages",
    "level6_bomb38_south_success",
    "make_bomb38_south_controller",
    "select_bombs_owned",
]

# Dated v1/v2 bomb stands. Do not reuse. v3 is east/west census.
BOMB_38_SOUTH_STAND = (120, 173)
SOUTH_DOOR_X = 120
SOUTH_DOOR_Y = 189
EAST_DOOR = (208, 141)
WEST_DOOR = (32, 141)
DOOR_TOL = 4
WEST_SPAWN_XMIN = 16
CENSUS_PUSH_FRAMES = 90
BOMB38_SOUTH_MAX_FRAMES = 8000
BOMB38_SOUTH_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)

BOMB_WALL_38_SOUTH = BombWall(
    room=LEVEL6_WIZZROBE_38_ROOM,
    stand=BOMB_38_SOUTH_STAND,
    face="DOWN",
    opens_to=LEVEL6_TRAPS_ROOM,
    opened_door_bit=DOOR_UP,
    live=False,
    notes=(
        "v1 (120,181) and v2 (120,173) DOWN consumed no-open. "
        "v3 occupancy east (208,141) then west (32,141) census."
    ),
)


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


def select_bombs_owned(env: Any, run: Any) -> None:
    """B-slot bombs (already owned). Unused on v3 census — no count top-up."""
    extra = apply_owned_inventory(env, select_bomb=True)
    prev = getattr(run, "inventory_assist", None)
    if prev is None:
        run.inventory_assist = extra
        return
    merged = dict(prev)
    merged["writes"] = list(prev.get("writes") or []) + list(
        extra.get("writes") or []
    )
    merged["notes"] = list(prev.get("notes") or []) + list(extra.get("notes") or [])
    merged["select_bomb"] = True
    run.inventory_assist = merged


@dataclass
class Level6Bomb38SouthController:
    """Occupancy east then west door census. Never bomb. Never tile-170 hold."""

    spec_id: str = "level6_bomb_south_0x38"
    room: int = LEVEL6_WIZZROBE_38_ROOM
    dest: int = LEVEL6_TRAPS_ROOM
    goal: tuple[int, int] = EAST_DOOR
    max_frames: int = BOMB38_SOUTH_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    phase: str = "east_path"
    push_frames: int = 0
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    east_census: dict[str, int] = field(default_factory=dict)
    west_census: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _door_snap(self, snap: ZeldaSnapshot) -> dict[str, int]:
        return {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "tile": int(snap.colliding_tile),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "bombs": int(snap.bombs),
            "keys": int(snap.keys),
        }

    def _at(self, snap: ZeldaSnapshot, dest: tuple[int, int]) -> bool:
        tx, ty = dest
        return abs(snap.link_x - tx) <= DOOR_TOL and abs(snap.link_y - ty) <= DOOR_TOL

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "rod": self._rod(snap),
            "bow": self._bow(snap),
            "arrows": self._arrows(snap),
            "tile": int(snap.colliding_tile),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        if force or self.frames <= 2 or self.frames % BOMB38_SOUTH_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "rod": self._rod(snap),
                    "keys": int(snap.keys),
                    "bombs": int(snap.bombs),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                    "phase": self.phase,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.keys >= 0 and int(snap.keys) < self.keys:
            self.notes.append(
                f"key_spent_38_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
            )
        self.keys = int(snap.keys)
        note = (
            f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_rod={self._rod(snap)}_tf={snap.triforce:02x}"
            f"_keys={int(snap.keys)}_bombs={int(snap.bombs)}"
        )
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"), force=True
        )

    def _record_census(self, snap: ZeldaSnapshot, side: str) -> None:
        info = self._door_snap(snap)
        if side == "east":
            self.east_census = info
        else:
            self.west_census = info
        note = (
            f"census_{side}_{info['x']}_{info['y']}_tile={info['tile']}"
            f"_doors={info['cur_opened_doors']}_mask={info['open_doorway_mask']}"
        )
        if note not in self.notes:
            self.notes.append(note)

    def _set_goal(self, dest: tuple[int, int]) -> None:
        self.goal = dest
        self.walker.path = None
        self.walker.goal = dest
        self.walker.last_dir = None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={self._rod(snap)}_keys={int(snap.keys)}"
                    f"_bombs={int(snap.bombs)}_phase={self.phase}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if snap.mode == CELLAR_MODE:
            return self._fail(
                snap,
                f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.level == 0:
            return self._fail(
                snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}_{snap.screen:02x}")
        if (
            snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self._rod(snap) != 0
        ):
            if snap.screen == LEVEL6_ROD_WIZZ_ROOM:
                return self._fail(
                    snap,
                    f"key_up_09_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}",
                )
            if snap.screen == LEVEL6_WIZZROBE_28_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_28_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen != self.dest:
                side = "east" if self.phase.startswith("east") else "west"
                return self._fail(
                    snap,
                    f"census_{side}_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            face = "RIGHT" if self.phase.startswith("east") else "LEFT"
            return FrameAction(nes_action(face), "side_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            face = "RIGHT" if self.phase.startswith("east") else "LEFT"
            return FrameAction(nes_action(face), "side_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        if xy[1] >= SOUTH_DOOR_Y:
            self.walker.last_dir = None
            return self._emit(snap, FrameAction(nes_action("UP"), "mouth_back"))

        if self.phase == "east_path" and self._at(snap, EAST_DOOR):
            self.phase = "east_push"
            self.push_frames = 0
            self._record_census(snap, "east")
        if self.phase == "west_path" and self._at(snap, WEST_DOOR):
            self.phase = "west_push"
            self.push_frames = 0
            self._record_census(snap, "west")

        if self.phase == "east_push":
            if not self._at(snap, EAST_DOOR):
                self.phase = "east_path"
                self._set_goal(EAST_DOOR)
            else:
                self.push_frames += 1
                if self.push_frames >= CENSUS_PUSH_FRAMES:
                    self.phase = "west_path"
                    self._set_goal(WEST_DOOR)
                    self.notes.append("east_no_scroll")
                else:
                    self.walker.last_dir = None
                    return self._emit(
                        snap, FrameAction(nes_action("RIGHT"), "east_push")
                    )

        if self.phase == "west_push":
            if not self._at(snap, WEST_DOOR):
                self.phase = "west_path"
                self._set_goal(WEST_DOOR)
            else:
                self.push_frames += 1
                if self.push_frames >= CENSUS_PUSH_FRAMES:
                    return self._fail(
                        snap,
                        "census_sealed_east_west"
                        f"_e={self.east_census.get('tile')}"
                        f"_w={self.west_census.get('tile')}"
                        f"_doors={snap.cur_opened_doors}_mask={snap.open_doorway_mask}",
                    )
                self.walker.last_dir = None
                return self._emit(snap, FrameAction(nes_action("LEFT"), "west_push"))

        dest = EAST_DOOR if self.phase.startswith("east") else WEST_DOOR
        self.goal = dest
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        just_missed = self.walker.misses > misses_before
        if just_missed and (self.walker.misses <= 8 or self.frames % 60 == 0):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and xy[1] <= NORTH_BAND_Y:
            return self._fail(
                snap, f"occupancy_halt_north_{xy[0]}_{xy[1]}"
            )
        if direction is None:
            return self._fail(
                snap, f"occupancy_halt_{self.phase}_{xy[0]}_{xy[1]}"
            )
        reason = "to_east" if self.phase.startswith("east") else "to_west"
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "v3 occupancy east (208,141) RIGHT census then west (32,141) "
                "LEFT census; no bomb; no hold DOWN at y=189 tile 170; "
                "ignore 0x2B; do not push 0x68; halt occupancy no-path; "
                "no reclear; no KEY-UP 0x09; no CheckWarp; v1/v2 dated "
                "consume-no-open at (120,181) and (120,173)"
            ),
            "leftover": dict(self.leftover),
            "east_census": dict(self.east_census),
            "west_census": dict(self.west_census),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "dest": self.dest,
            "goal": self.goal,
            "keys": self.keys,
            "phase": self.phase,
        }


def make_bomb38_south_controller() -> Level6Bomb38SouthController:
    """Occupancy east/west door census of 0x38. No bomb. No count poke."""
    return Level6Bomb38SouthController()


def level6_bomb38_south_stages():
    """Play 0x38 leftover (120,93) → east/west census. Stop still 0x48."""
    ctl = make_bomb38_south_controller()
    return (
        ("level6_bomb_south_0x38", ctl, ctl.max_frames),
    )


def level6_bomb38_south_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 0x48 with ADDR_ROD. Enter-stop; traps may stay live."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen == LEVEL6_TRAPS_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
