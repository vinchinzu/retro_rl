"""Static editor collision for generalist occupancy and door potential.

Live bank-$7F clipdata is not mapped on the practice core (WRAM block is
$7E only). Editor room JSON is the same 16px grid the catalog already uses.
Door tiles (clip 9) stay walkable so occupancy does not hide the Join.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from super_metroid.paths import REPO_DIR

TILE_PX = 16
CLIP_AIR = 0
CLIP_DOOR = 9


class CollisionDependencyError(RuntimeError):
    """The editor collision required by the contractor is unavailable."""


def editor_rooms_dir() -> Path | None:
    """``SUPER_METROID_EDITOR_NAV`` file/dir, else the sibling snes_editor export."""

    configured = os.environ.get("SUPER_METROID_EDITOR_NAV")
    candidates: list[Path] = []
    if configured:
        nav = Path(configured).expanduser()
        candidates.append(nav.parent / "rooms" if nav.is_file() else nav / "rooms")
        candidates.append(nav)
    else:
        candidates.append(
            REPO_DIR.parent
            / "snes_editor"
            / "super_metroid_rl"
            / "super_metroid_editor"
            / "export"
            / "sm_nav"
            / "rooms"
        )
    for path in candidates:
        if path.is_dir():
            return path
    return None


@dataclass(frozen=True)
class RoomSolid:
    """One room's 16px collision grid. 0/9 are walkable; other clips occupy."""

    room_id: int
    width: int
    height: int
    collision: tuple[tuple[int, ...], ...]
    doors: tuple[tuple[int, int], ...]

    def clip_at(self, wx: int, wy: int) -> int | None:
        bx, by = int(wx) // TILE_PX, int(wy) // TILE_PX
        if not (0 <= bx < self.width and 0 <= by < self.height):
            return None
        return int(self.collision[by][bx])

    def is_solid(self, wx: int, wy: int) -> bool:
        clip = self.clip_at(wx, wy)
        if clip is None:
            return True
        return clip not in (CLIP_AIR, CLIP_DOOR)

    def nearest_door(self, sx: int, sy: int) -> tuple[int, int] | None:
        if not self.doors:
            return None
        best = self.doors[0]
        best_d = (best[0] - sx) ** 2 + (best[1] - sy) ** 2
        for door in self.doors[1:]:
            dist = (door[0] - sx) ** 2 + (door[1] - sy) ** 2
            if dist < best_d:
                best, best_d = door, dist
        return best


def room_solid_from_collision(
    room_id: int, collision: list[list[int]] | tuple[tuple[int, ...], ...]
) -> RoomSolid:
    try:
        rows = tuple(tuple(int(cell) for cell in row) for row in collision)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid collision cells for room 0x{room_id:04X}") from exc
    if not rows or not rows[0]:
        raise ValueError(f"empty collision for room 0x{room_id:04X}")
    height = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"ragged collision for room 0x{room_id:04X}")
    doors: list[tuple[int, int]] = []
    for by, row in enumerate(rows):
        for bx, clip in enumerate(row):
            if clip == CLIP_DOOR:
                doors.append((bx * TILE_PX + TILE_PX // 2, by * TILE_PX + TILE_PX // 2))
    return RoomSolid(
        room_id=int(room_id),
        width=width,
        height=height,
        collision=rows,
        doors=tuple(doors),
    )


@lru_cache(maxsize=256)
def load_room_solid(room_id: int, root: Path | None = None) -> RoomSolid | None:
    directory = root if root is not None else editor_rooms_dir()
    if directory is None:
        return None
    path = Path(directory) / f"room_{int(room_id):04X}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    grid = payload.get("collision")
    if not isinstance(grid, list) or not grid:
        return None
    return room_solid_from_collision(int(room_id), grid)


def require_row_solids(
    rows: Iterable[Any], *, root: Path | None = None
) -> dict[int, RoomSolid]:
    """Load collision for every start and Goal room in a curriculum."""

    selected_rows = tuple(rows)
    room_ids = sorted(
        {
            int(room_id)
            for row in selected_rows
            for room_id in (getattr(row, "room_id", 0), getattr(row, "goal_room_id", 0))
            if int(room_id or 0) > 0
        }
    )
    cross_room_starts = {
        int(getattr(row, "room_id", 0) or 0)
        for row in selected_rows
        if int(getattr(row, "room_id", 0) or 0)
        != int(getattr(row, "goal_room_id", 0) or 0)
    }
    directory = Path(root) if root is not None else editor_rooms_dir()
    room_names = ", ".join(f"0x{room_id:04X}" for room_id in room_ids)
    if directory is None:
        configured = os.environ.get("SUPER_METROID_EDITOR_NAV")
        dependency = configured or "the sibling snes_editor sm_nav/rooms export"
        raise CollisionDependencyError(
            "editor collision directory unavailable for curriculum rooms "
            f"{room_names or '(none)'}; checked {dependency!r}. Set "
            "SUPER_METROID_EDITOR_NAV to the sm_nav export or its rooms directory"
        )

    solids: dict[int, RoomSolid] = {}
    missing: list[int] = []
    invalid: list[tuple[int, str]] = []
    for room_id in room_ids:
        try:
            solid = load_room_solid(room_id, directory)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append((room_id, str(exc)))
            continue
        if solid is None:
            missing.append(room_id)
        else:
            solids[room_id] = solid
    doorless = sorted(
        room_id
        for room_id in cross_room_starts
        if room_id in solids and not solids[room_id].doors
    )
    if missing or invalid or doorless:
        details: list[str] = []
        if missing:
            details.append(
                "missing " + ", ".join(f"0x{room_id:04X}" for room_id in missing)
            )
        if invalid:
            details.append(
                "invalid "
                + ", ".join(
                    f"0x{room_id:04X} ({reason})" for room_id, reason in invalid
                )
            )
        if doorless:
            details.append(
                "cross-room start "
                + ", ".join(f"0x{room_id:04X}" for room_id in doorless)
                + " has no clip-9 door"
            )
        raise CollisionDependencyError(
            f"editor collision unavailable in {directory}: {'; '.join(details)}. "
            "Set SUPER_METROID_EDITOR_NAV to an sm_nav export containing every "
            "selected curriculum room"
        )
    return solids


def potential_xy(
    state: Any, goal: Any, solid: RoomSolid | None
) -> tuple[int, int]:
    """Dense-reward target: Join xy in-room, else nearest door in this room."""

    gx = int(getattr(goal, "x", 0) or 0)
    gy = int(getattr(goal, "y", 0) or 0)
    room = int(getattr(state, "room_id", getattr(state, "room", 0)) or 0)
    if room == int(getattr(goal, "room_id", 0) or 0):
        return gx, gy
    if solid is None:
        raise CollisionDependencyError(
            f"editor collision missing for cross-room potential in room 0x{room:04X}"
        )
    if int(solid.room_id) != room:
        raise CollisionDependencyError(
            "editor collision room mismatch for cross-room potential: "
            f"state=0x{room:04X}, collision=0x{int(solid.room_id):04X}"
        )
    door = solid.nearest_door(
        int(getattr(state, "samus_x", getattr(state, "x", 0)) or 0),
        int(getattr(state, "samus_y", getattr(state, "y", 0)) or 0),
    )
    if door is None:
        raise CollisionDependencyError(
            f"cross-room potential in room 0x{room:04X} has no clip-9 door"
        )
    return door


__all__ = [
    "CLIP_AIR",
    "CLIP_DOOR",
    "CollisionDependencyError",
    "TILE_PX",
    "RoomSolid",
    "editor_rooms_dir",
    "load_room_solid",
    "potential_xy",
    "require_row_solids",
    "room_solid_from_collision",
]
