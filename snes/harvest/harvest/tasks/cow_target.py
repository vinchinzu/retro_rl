"""Cow targeting, pin memory, and stand selection for CowChoresTask."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.core.animal_probe import cow_slot_snapshots
from harvest.core.animal_status import (
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_TALKED_FLAG,
    read_cow_daily_flags,
)
from harvest.tasks.animal_navigation import align_to_pixel
from harvest.tasks.cow_geometry import (
    COW_TALK_FACE,
    COW_TALK_STAND,
    cow_body_tile,
    cow_interact_pixel,
    cow_push_escape_tile,
    face_for_cow_at_stand,
    geometric_fallback_stands,
    is_adjacent_to_cow_tile,
    preferred_cow_stands,
    stand_blocked,
    stand_in_bounds,
)
from harvest.tasks.nav import make_action


class CowTargetMixin:
    """Resolve target cow tile/pixel and approach stands."""

    def _target_cow_tile(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._target_cow_slot is None:
            return None
        for row in cow_slot_snapshots(ram, require_barn=True):
            if int(row.get("slot", -1)) != self._target_cow_slot:
                continue
            tile = row.get("tile")
            if not isinstance(tile, list) or len(tile) != 2:
                return None
            return int(tile[0]), int(tile[1])
        return None

    def _target_cow_pixel(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        if self._target_cow_slot is None:
            return None
        for row in cow_slot_snapshots(ram, require_barn=True):
            if int(row.get("slot", -1)) != self._target_cow_slot:
                continue
            pixel = row.get("pixel")
            if not isinstance(pixel, list) or len(pixel) != 2:
                return None
            return int(pixel[0]), int(pixel[1])
        return None

    def _target_cow_body_tile(self, ram: np.ndarray) -> Optional[Tuple[int, int]]:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return None
        return cow_body_tile(tile)

    def _is_adjacent_to_target_cow(self, ram: np.ndarray, stand: Tuple[int, int], face: str) -> bool:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return False
        return is_adjacent_to_cow_tile(stand, face, tile)

    def _remember_current_pin(self) -> None:
        if self._target_cow_slot is None:
            return
        self._recent_pin_slot = self._target_cow_slot
        self._recent_pin_stand = self._navigator.current_tile
        self._recent_pin_face = self._talk_face

    def _recent_pin_milk_face(self, ram: np.ndarray, stand: Optional[Tuple[int, int]] = None) -> Optional[str]:
        if self._target_cow_slot is None or self._recent_pin_slot != self._target_cow_slot:
            return None
        stand = stand or self._navigator.current_tile
        if self._recent_pin_stand != stand:
            return None
        if self._recent_pin_face not in ("left", "right"):
            return None
        if self._is_adjacent_to_target_cow(ram, stand, self._recent_pin_face):
            return self._recent_pin_face
        tile = self._target_cow_tile(ram)
        if tile is None:
            return None
        facing = self._facing_tile(stand, self._recent_pin_face)
        # After a successful brush/talk, the cow can idle one horizontal body
        # tile away before the milker is selected. Reuse only that proven pin;
        # do not make this a general brush/talk adjacency rule.
        if facing[1] == tile[1] and abs(facing[0] - tile[0]) == 1:
            flags = read_cow_daily_flags(ram, self._target_cow_slot)
            if flags & (COW_DAILY_BRUSHED_FLAG | COW_DAILY_TALKED_FLAG):
                return self._recent_pin_face
        return None

    def _face_for_target_cow(self, ram: np.ndarray, stand: Optional[Tuple[int, int]] = None) -> str:
        stand = stand or self._talk_stand
        return face_for_cow_at_stand(
            stand,
            self._target_cow_tile(ram),
            default_face=COW_TALK_FACE,
            talk_stand=self._talk_stand,
            talk_face=self._talk_face,
        )

    def _cow_interact_pixel(self, ram: np.ndarray, *, tool: bool) -> Optional[Tuple[int, int]]:
        pixel = self._target_cow_pixel(ram)
        if pixel is None:
            return None
        return cow_interact_pixel(
            pixel,
            self._talk_face,
            tool=tool,
            cow_tile=self._target_cow_tile(ram),
        )

    def _at_cow_interact_pixel(self, ram: np.ndarray, *, tool: bool, tolerance: int = 1) -> bool:
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is None:
            return False
        if tool and self._talk_face in ("left", "right"):
            return (
                target[0] == self._navigator.current_pos.x
                and target[1] == self._navigator.current_pos.y
            )
        return (
            abs(target[0] - self._navigator.current_pos.x) <= tolerance
            and abs(target[1] - self._navigator.current_pos.y) <= tolerance
        )

    def _align_to_cow_interact_pixel(self, ram: np.ndarray, *, tool: bool) -> Optional[np.ndarray]:
        target = self._cow_interact_pixel(ram, tool=tool)
        if target is None:
            return None
        if tool and self._talk_face in ("left", "right"):
            dx = target[0] - self._navigator.current_pos.x
            dy = target[1] - self._navigator.current_pos.y
            if dx != 0:
                return make_action(right=dx > 0, left=dx < 0)
            if dy != 0:
                return make_action(down=dy > 0, up=dy < 0)
            return None
        return align_to_pixel(
            (self._navigator.current_pos.x, self._navigator.current_pos.y),
            target,
            tolerance=1,
        )

    def _candidate_cow_stands(self, ram: np.ndarray) -> list[Tuple[Tuple[int, int], str]]:
        tile = self._target_cow_tile(ram)
        if tile is None:
            return [(COW_TALK_STAND, COW_TALK_FACE)]

        cx, cy = tile
        preferred: list[Tuple[Tuple[int, int], str]] = []
        current = self._navigator.current_tile
        current_face = self._face_for_target_cow(ram, current)
        if self._is_adjacent_to_target_cow(ram, current, current_face):
            preferred.append((current, current_face))
        preferred.extend(preferred_cow_stands(cx, cy))

        candidates: list[Tuple[Tuple[int, int], str]] = []
        scored: list[Tuple[Tuple[int, int, int], Tuple[Tuple[int, int], str]]] = []
        seen: set[Tuple[int, int]] = set()
        cow_tiles = self._cow_tiles(ram)
        for index, (stand, face) in enumerate(preferred):
            sx, sy = stand
            if stand in seen:
                continue
            seen.add(stand)
            if not stand_in_bounds(stand):
                continue
            if stand_blocked(stand, cow_tiles):
                continue
            if not self._pathfinder.is_walkable(ram, sx, sy, current_pos=self._navigator.current_tile):
                continue
            if self._find_path_around_cows(ram, self._navigator.current_tile, stand) is None:
                continue
            candidates.append((stand, face))
            # Wall-side cows already prefer body-right stands in `preferred`;
            # escape-pin scoring would re-rank head-on (1, cy) first and that
            # stand often fails to start talk/brush dialog.
            if cx <= 4:
                pin_penalty = 0
            else:
                pin_penalty = 0 if self._cow_escape_blocked(ram, tile, stand, face, cow_tiles) else 1
            current = self._navigator.current_tile
            distance = abs(sx - current[0]) + abs(sy - current[1])
            scored.append(((pin_penalty, index, distance), (stand, face)))
        if scored:
            return [item for _score, item in sorted(scored, key=lambda row: row[0])]
        if candidates:
            return candidates
        # Path checks can fail while cows shuffle; still aim at a geometric
        # side stand instead of snapping to the default talk tile across barn.
        loose: list[Tuple[Tuple[int, int], Tuple[Tuple[int, int], str]]] = []
        for index, (stand, face) in enumerate(preferred):
            sx, sy = stand
            if not stand_in_bounds(stand):
                continue
            if stand_blocked(stand, cow_tiles):
                continue
            if not self._pathfinder.is_walkable(
                ram, sx, sy, current_pos=self._navigator.current_tile
            ):
                continue
            current = self._navigator.current_tile
            distance = abs(sx - current[0]) + abs(sy - current[1])
            loose.append(((index, distance), (stand, face)))
        if loose:
            return [item for _score, item in sorted(loose, key=lambda row: row[0])]
        # Absolute geometric fallback — never snap to the default talk stand
        # when we still know where the target cow is.
        current = self._navigator.current_tile
        return geometric_fallback_stands(
            cx,
            cy,
            cow_tiles,
            current=current,
            current_face=self._face_for_target_cow(ram, current),
        )

    def _cow_escape_blocked(
        self,
        ram: np.ndarray,
        cow_tile: Tuple[int, int],
        stand: Tuple[int, int],
        face: str,
        cow_tiles: set[Tuple[int, int]],
    ) -> bool:
        escape = cow_push_escape_tile(cow_tile, stand, face)
        if escape is None:
            return False
        if not stand_in_bounds(escape):
            return True
        other_cow_tiles = set(cow_tiles)
        other_cow_tiles.discard(cow_tile)
        other_cow_tiles.discard(cow_body_tile(cow_tile))
        if escape in other_cow_tiles:
            return True
        return not self._pathfinder.is_walkable(
            ram, escape[0], escape[1], current_pos=self._navigator.current_tile
        )
