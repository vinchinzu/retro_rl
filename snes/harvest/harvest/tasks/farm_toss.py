"""Reactive held-debris toss for the west plant pocket.

The Day 2 pocket can put the player at (13,27) with weeds immediately east
and south.  A fixed south throw or fence run therefore pins against the exact
notch that must be freed for planting.  Prefer a RAM-confirmed open adjacent
tile for face+A; retain the fence-jump action only as a no-open-side fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.animal_status import read_held_item
from harvest.core.task_progress import ProgressSnapshot
from harvest.core.tile_catalog import FARM_WALKABLE, STALE_TILE_IDS, TILE_TO_DEBRIS
from harvest.tasks.nav import TILE_SIZE, get_pos_from_ram, get_tile_at, make_action

HELD_WEED = 0x09
HELD_STONE = 0x0D
HELD_BUSH = 0x29
POCKET_LIFT_HELD: frozenset[int] = frozenset({HELD_WEED, HELD_STONE, HELD_BUSH})
HELD_FENCE = 0x0F  # a lifted fence post (tile 0x05)
# Debris the pond takes (the Day 2 clear dumps stones and fence posts there).
POND_TOSS_HELD: frozenset[int] = POCKET_LIFT_HELD | {HELD_FENCE}
# Held ids the pond must never take: a crop, an egg or milk (the bin pays), fodder, a hen.
KEEP_HELD: frozenset[int] = frozenset({0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x1A, 0x25})


def pond_takes(held: int) -> bool:
    """Debris, or anything unrecognised that the yard refused (Y2 Summer D18, sweep t5f_s54:
    held 0x57 after a forced-failure day, every yard tile 0xA8 refused it, 8 rescues)."""
    held = int(held)
    return held in POND_TOSS_HELD or (held != 0 and held not in KEEP_HELD)

# Tape drop stands (reference only). The skill runs straight south, not to these.
POCKET_DROP_COLUMNS: Tuple[int, ...] = (14, 15, 16)
POCKET_DROP_Y = 32
FENCE_WALL_Y = 31

Tile = Tuple[int, int]

_FACE_DELTAS = {
    "up": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
    "down": (0, 1),
}


def is_pocket_lift_held(held: int) -> bool:
    return int(held) in POCKET_LIFT_HELD


def needs_south_fence_drop(player_tile: Tile, held: int) -> bool:
    """True when a pocket lift is still in hands north of / on the fence."""
    if not is_pocket_lift_held(held):
        return False
    return int(player_tile[1]) <= FENCE_WALL_Y


def nearest_pocket_drop(player_tile: Tile) -> Tile:
    col = min(POCKET_DROP_COLUMNS, key=lambda c: abs(c - int(player_tile[0])))
    return (col, POCKET_DROP_Y)


def pocket_no_toss_tiles() -> set:
    """3x3 ring, hoe stands, and the y=30 west lip — never land debris here."""
    try:
        from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
        from harvest.tasks.crop_geometry import hoe_plan, plot_tiles
    except Exception:
        return set()
    cx, cy = WEST_POCKET_PLANT_CENTER
    tiles = set(plot_tiles((cx, cy), include_center=True))
    for target, stand, _face in hoe_plan((cx, cy)):
        tiles.add(target)
        tiles.add(stand)
    for x in range(11, 16):
        tiles.add((x, 30))
    return tiles


def open_toss_face(ram, player_tile: Tile, blocked=()) -> Optional[str]:
    """Choose an adjacent, loaded ground tile that can accept a throw.

    North of the fence, prefer north/sideways so the crop notch to the south
    stays free.  ``FARM_WALKABLE`` includes weeds, so debris must be excluded
    explicitly.  Hoe stands and the lift origin are never valid landings —
    tossing onto the cell just lifted is the (11,28) false-success.
    """
    faces = (
        ("up", "left", "right", "down")
        if int(player_tile[1]) <= FENCE_WALL_Y
        else ("down", "left", "right", "up")
    )
    forbidden = pocket_no_toss_tiles() | {tuple(tile) for tile in blocked}
    for face in faces:
        dx, dy = _FACE_DELTAS[face]
        dest = (int(player_tile[0]) + dx, int(player_tile[1]) + dy)
        if dest in forbidden:
            continue
        tile_id = int(get_tile_at(ram, dest[0], dest[1]))
        if (
            tile_id in FARM_WALKABLE
            and tile_id not in STALE_TILE_IDS
            and tile_id not in TILE_TO_DEBRIS
        ):
            return face
    return None


def evaluate_lift_verify(ram, origin: Tile) -> str:
    """``blocked`` still debris, ``carrying`` in hands, ``cleared`` gone."""
    from harvest.core.tile_catalog import TILE_TO_DEBRIS as debris_ids

    tile_id = int(get_tile_at(ram, origin[0], origin[1]))
    if debris_ids.get(tile_id) is not None:
        return "blocked"
    if int(read_held_item(ram)):
        return "carrying"
    return "cleared"


def _plot_no_toss_tiles() -> set:
    return pocket_no_toss_tiles()


def fence_jump_action(
    player_tile: Tile,
    held: int,
    *,
    last_y: Optional[int] = None,
    stasis: int = 0,
) -> Optional[np.ndarray]:
    """One reactive frame. ``None`` when hands are empty.

    Primary: B+Down through the fence. Lateral only after south stasis
    (blocked by a stone on the column) — never a planned east detour.
    """
    if not held:
        return None
    y = int(player_tile[1])
    x = int(player_tile[0])
    if y <= FENCE_WALL_Y:
        if stasis >= 40 and last_y is not None and last_y <= FENCE_WALL_Y:
            if x < 15:
                return make_action(right=True, b=True)
            if x > 15:
                return make_action(left=True, b=True)
        return make_action(down=True, b=True)
    return None


def toss_pulse_action(pulse: int, *, face: str = "down") -> np.ndarray:
    """Face an open tile, settle, then A with no d-pad.

    Holding the face on throw walks onto the landing (live D2: UP from
    (11,30) re-drops the (11,29) stone onto (11,28)).
    """
    phase = int(pulse) % 40
    if phase < 2:
        return make_action(**{face: True})
    if phase < 8:
        return make_action()
    if phase < 24:
        return make_action(a=True)
    return make_action()


@dataclass
class FenceJumpTossSkill(Task):
    """Toss toward open ground; fence-jump only when boxed in."""

    name: str = "fence_jump_toss"
    timeout: int = 300
    blocked: frozenset = field(default_factory=frozenset)

    _steps: int = field(default=0, init=False)
    _toss_pulse: int = field(default=0, init=False)
    _last_y: Optional[int] = field(default=None, init=False)
    _last_tile: Optional[Tile] = field(default=None, init=False)
    _stasis: int = field(default=0, init=False)
    _toss_face: Optional[str] = field(default=None, init=False)
    _toss_dest: Optional[Tile] = field(default=None, init=False)
    _rejected: set = field(default_factory=set, init=False)

    def reset(self, world: WorldState) -> None:
        self._steps = 0
        self._toss_pulse = 0
        self._last_y = None
        self._last_tile = None
        self._stasis = 0
        self._toss_face = None
        self._toss_dest = None
        self._rejected = set()

    def can_start(self, world: WorldState) -> bool:
        return True

    def progress_snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            task_name=self.name,
            phase_text="open_toss" if self._toss_face is not None else "escape",
            step_count=self._steps,
        )

    def step(self, world: WorldState) -> TaskResult:
        self._steps += 1
        held = int(read_held_item(world.ram))
        pos = get_pos_from_ram(world.ram)
        tile = (pos.x // TILE_SIZE, pos.y // TILE_SIZE)
        if not held:
            return TaskResult(status=TaskStatus.SUCCESS, reason="hands empty")
        if self._steps > self.timeout:
            return TaskResult(
                status=TaskStatus.FAILURE,
                reason=f"fence-jump toss timeout held=0x{held:02X} tile={tile}",
            )
        if self._last_tile == tile:
            self._stasis += 1
        else:
            self._stasis = 0
            self._last_tile = tile
            self._last_y = tile[1]

        # One pulse per face. A pulse that is still holding did not land, so
        # that tile is not tried again. North-yard dirt often looks open and
        # refuses the stone; retrying it burns the fence-jump budget.
        if (
            self._toss_face is not None
            and self._toss_pulse > 0
            and self._toss_pulse % 40 == 0
        ):
            if self._toss_dest is not None:
                self._rejected.add(self._toss_dest)
            self._toss_face = None
            self._toss_dest = None
        if self._toss_face is None:
            blocked = frozenset(self.blocked) | self._rejected
            self._toss_face = open_toss_face(world.ram, tile, blocked=blocked)
            if self._toss_face is not None:
                dx, dy = _FACE_DELTAS[self._toss_face]
                self._toss_dest = (int(tile[0]) + dx, int(tile[1]) + dy)
        if self._toss_face is not None:
            action = toss_pulse_action(self._toss_pulse, face=self._toss_face)
            face = self._toss_face
            self._toss_pulse += 1
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(action),
                reason=f"toss held debris toward open {face}",
            )

        # West pond lip has no safe adjacent landing (origin / plot / water).
        # Carry east to open ground before jumping the y=31 fence.
        if int(tile[1]) <= FENCE_WALL_Y and int(tile[0]) < 16 and self._stasis < 24:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(make_action(right=True, b=True)),
                reason="carry east to open toss",
            )

        # Fully boxed-in fallback.  This is deliberately bounded by the skill
        # timeout; callers can skip the target instead of sitting for 1000f+.
        jump = fence_jump_action(
            tile, held, last_y=self._last_y, stasis=self._stasis
        )
        if jump is not None:
            return TaskResult(
                status=TaskStatus.RUNNING,
                action=ActionResult(jump),
                reason="run south jump fence",
            )
        action = toss_pulse_action(self._toss_pulse)
        self._toss_pulse += 1
        return TaskResult(
            status=TaskStatus.RUNNING,
            action=ActionResult(action),
            reason="toss south of fence",
        )


def in_place_toss_actions(*, face: str = "down") -> list:
    return [
        *[make_action(**{face: True}) for _ in range(2)],
        *[make_action() for _ in range(2)],
        *[make_action(a=True) for _ in range(12)],
        *[make_action() for _ in range(12)],
    ]


def _fence_jump_timeout(ram) -> int:
    """Pocket tosses are short. A north-yard stone still has to reach y=31."""
    if ram is None:
        return 300
    pos = get_pos_from_ram(ram)
    tile_y = int(pos.y) // TILE_SIZE
    if tile_y > FENCE_WALL_Y:
        return 300
    return max(300, 80 + (FENCE_WALL_Y - tile_y) * 20)


def start_fence_jump_skill(
    *,
    frame: int = 0,
    ram=None,
    blocked=(),
) -> FenceJumpTossSkill:
    skill = FenceJumpTossSkill(
        timeout=_fence_jump_timeout(ram),
        blocked=frozenset(tuple(tile) for tile in blocked),
    )
    skill.reset(
        WorldState(
            frame=frame,
            ram=ram if ram is not None else np.zeros(1, dtype=np.uint8),
            info={},
            obs=None,
        )
    )
    return skill


def step_fence_jump_skill(
    skill: Optional[FenceJumpTossSkill],
    ram,
    *,
    frame: int = 0,
) -> Tuple[Optional[FenceJumpTossSkill], Optional[np.ndarray]]:
    """Advance the toss skill. Returns ``(skill_or_none, action_or_none)``."""
    if skill is None:
        return None, None
    world = WorldState(frame=frame, ram=ram, info={}, obs=None)
    result = skill.step(world)
    if result.status == TaskStatus.RUNNING:
        action = result.action.action if result.action is not None else make_action()
        return skill, action
    return None, None


def held_toss_actions(
    player_tile: Tile,
    held: int,
    *,
    face: str = "down",
) -> Tuple[str, list]:
    """Back-compat for unit tests. Prefer :class:`FenceJumpTossSkill`."""
    if needs_south_fence_drop(player_tile, held):
        return "pocket_south", [make_action(down=True, b=True)]
    actions = [
        make_action(**{face: True}),
        make_action(),
        make_action(a=True),
        make_action(),
    ]
    return "in_place", actions


__all__ = [
    "FENCE_WALL_Y",
    "HELD_FENCE",
    "HELD_BUSH",
    "HELD_STONE",
    "HELD_WEED",
    "POCKET_DROP_COLUMNS",
    "POCKET_DROP_Y",
    "POCKET_LIFT_HELD",
    "KEEP_HELD",
    "POND_TOSS_HELD",
    "pond_takes",
    "FenceJumpTossSkill",
    "evaluate_lift_verify",
    "fence_jump_action",
    "in_place_toss_actions",
    "is_pocket_lift_held",
    "start_fence_jump_skill",
    "step_fence_jump_skill",
    "nearest_pocket_drop",
    "needs_south_fence_drop",
    "open_toss_face",
    "pocket_no_toss_tiles",
    "toss_pulse_action",
    "held_toss_actions",
]
