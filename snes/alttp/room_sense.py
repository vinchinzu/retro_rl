"""Room sensing: enemy-aware sprite boxes, edge detection, and map overlays.

Used by main-hall (and later B1) navigation. Operates on observed RAM + the
current render frame — no progression writes.

**Map authority:** measured room geometry lives in ``alttp/maps/room_XX.json``
and is loaded via :func:`alttp.room_map.load_room_map`. Do not redeclare coords
in segments.

Coordinate conventions:

- **World / room** — ``AlttpSnapshot.link_x/y`` and sprite X/Y (16-bit).
- **Screen** — ``sx = world_x - camera_x``, ``sy = world_y - camera_y`` on the
  256×224 SNES framebuffer (HUD occupies the top ~32 px).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from alttp.primitives import (
    CASTLE_HOSTILE_TYPES,
    SPRITE_HEART,
    SPRITE_SMALL_KEY,
    SPRITE_ZELDA,
    SpriteSnapshot,
    active_sprites,
)
from alttp.ram import AlttpSnapshot
from alttp.room_map import (  # re-export for callers / tests
    MAPS_DIR,
    ClearPolicy,
    KnownDoor,
    RoomMap,
    RoomMapPoint,
    list_room_maps,
    load_room_map,
    room_map_path,
    save_room_map,
)
from alttp.startup import snapshot_env

__all__ = [
    "MAPS_DIR",
    "SPRITE_KIND_HOSTILE",
    "SPRITE_KIND_NPC",
    "SPRITE_KIND_OTHER",
    "SPRITE_KIND_PICKUP",
    "ClearPolicy",
    "KnownDoor",
    "RoomEdge",
    "RoomMap",
    "RoomMapPoint",
    "SpriteBox",
    "box_for_sprite",
    "classify_sprite",
    "detect_edge",
    "draw_room_overlay",
    "enemy_boxes",
    "find_edges_by_push",
    "list_room_maps",
    "load_room_map",
    "nearest_enemy_box",
    "overlay_from_env",
    "path_blocked_by_enemies",
    "room_map_path",
    "save_room_map",
    "sprite_boxes",
    "world_to_screen",
]

# Approximate body boxes (half-width, half-height). Feet sit near RAM position.
_DEFAULT_HALF = (8, 12)
_HOSTILE_HALF = (10, 14)
_PICKUP_HALF = (6, 6)
_NPC_HALF = (8, 14)

SPRITE_KIND_HOSTILE = "hostile"
SPRITE_KIND_PICKUP = "pickup"
SPRITE_KIND_NPC = "npc"
SPRITE_KIND_OTHER = "other"

_PICKUP_TYPES = frozenset({SPRITE_HEART, SPRITE_SMALL_KEY})
_NPC_TYPES = frozenset({SPRITE_ZELDA})


@dataclass(frozen=True)
class SpriteBox:
    """Axis-aligned box around one active sprite (room coordinates)."""

    slot: int
    sprite_type: int
    kind: str
    x: int
    y: int
    x0: int
    y0: int
    x1: int
    y1: int
    hp: int = 0
    state: int = 0

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= int(x) <= self.x1 and self.y0 <= int(y) <= self.y1

    def intersects(self, other: SpriteBox) -> bool:
        return not (
            self.x1 < other.x0
            or self.x0 > other.x1
            or self.y1 < other.y0
            or self.y0 > other.y1
        )

    def inflate(self, pad: int) -> SpriteBox:
        p = int(pad)
        return SpriteBox(
            slot=self.slot,
            sprite_type=self.sprite_type,
            kind=self.kind,
            x=self.x,
            y=self.y,
            x0=self.x0 - p,
            y0=self.y0 - p,
            x1=self.x1 + p,
            y1=self.y1 + p,
            hp=self.hp,
            state=self.state,
        )

    def distance_to_point(self, x: int, y: int) -> int:
        dx = 0 if self.x0 <= x <= self.x1 else min(abs(x - self.x0), abs(x - self.x1))
        dy = 0 if self.y0 <= y <= self.y1 else min(abs(y - self.y0), abs(y - self.y1))
        return dx + dy

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "spriteType": self.sprite_type,
            "kind": self.kind,
            "xy": [self.x, self.y],
            "box": [self.x0, self.y0, self.x1, self.y1],
            "hp": self.hp,
            "state": self.state,
        }


@dataclass(frozen=True)
class RoomEdge:
    """Observed transition out of an expected room (runtime)."""

    direction: str  # UP | DOWN | LEFT | RIGHT | unknown
    from_room: int
    to_room: int
    from_xy: tuple[int, int]
    to_xy: tuple[int, int]
    outdoors: bool = False
    frames: int = 0
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "fromRoom": self.from_room,
            "toRoom": self.to_room,
            "fromXy": list(self.from_xy),
            "toXy": list(self.to_xy),
            "outdoors": self.outdoors,
            "frames": self.frames,
            "label": self.label,
        }


def classify_sprite(sprite_type: int) -> str:
    t = int(sprite_type)
    if t in CASTLE_HOSTILE_TYPES:
        return SPRITE_KIND_HOSTILE
    if t in _PICKUP_TYPES:
        return SPRITE_KIND_PICKUP
    if t in _NPC_TYPES:
        return SPRITE_KIND_NPC
    return SPRITE_KIND_OTHER


def _half_extents(kind: str, sprite_type: int) -> tuple[int, int]:
    del sprite_type  # reserved for per-type tuning
    if kind == SPRITE_KIND_HOSTILE:
        return _HOSTILE_HALF
    if kind == SPRITE_KIND_PICKUP:
        return _PICKUP_HALF
    if kind == SPRITE_KIND_NPC:
        return _NPC_HALF
    return _DEFAULT_HALF


def box_for_sprite(sprite: SpriteSnapshot) -> SpriteBox:
    kind = classify_sprite(sprite.sprite_type)
    hw, hh = _half_extents(kind, sprite.sprite_type)
    # RAM position ≈ feet; box extends upward.
    return SpriteBox(
        slot=sprite.slot,
        sprite_type=sprite.sprite_type,
        kind=kind,
        x=sprite.x,
        y=sprite.y,
        x0=sprite.x - hw,
        y0=sprite.y - 2 * hh,
        x1=sprite.x + hw,
        y1=sprite.y + 4,
        hp=sprite.hp,
        state=sprite.state,
    )


def sprite_boxes(
    env: object,
    *,
    kinds: Iterable[str] | None = None,
    max_distance: int | None = None,
    link_xy: tuple[int, int] | None = None,
) -> tuple[SpriteBox, ...]:
    """Build AABB boxes for active sprites, optionally filtered."""
    snap = snapshot_env(env)
    lx, ly = link_xy if link_xy is not None else (snap.link_x, snap.link_y)
    wanted = None if kinds is None else frozenset(kinds)
    out: list[SpriteBox] = []
    for sp in active_sprites(env):
        box = box_for_sprite(sp)
        if wanted is not None and box.kind not in wanted:
            continue
        if max_distance is not None:
            if abs(box.x - lx) + abs(box.y - ly) > int(max_distance):
                continue
        out.append(box)
    return tuple(out)


def enemy_boxes(
    env: object,
    *,
    max_distance: int | None = 220,
    pad: int = 0,
) -> tuple[SpriteBox, ...]:
    """Hostile sprite boxes near Link (enemy-aware planning input)."""
    boxes = sprite_boxes(
        env,
        kinds=(SPRITE_KIND_HOSTILE,),
        max_distance=max_distance,
    )
    if pad:
        return tuple(b.inflate(pad) for b in boxes)
    return boxes


def nearest_enemy_box(
    env: object,
    *,
    max_distance: int = 220,
) -> SpriteBox | None:
    boxes = enemy_boxes(env, max_distance=max_distance)
    if not boxes:
        return None
    snap = snapshot_env(env)
    return min(boxes, key=lambda b: b.distance_to_point(snap.link_x, snap.link_y))


def path_blocked_by_enemies(
    env: object,
    target_x: int,
    target_y: int,
    *,
    pad: int = 12,
    max_distance: int = 180,
) -> SpriteBox | None:
    """Return the first hostile box that intersects the Link→target segment."""
    snap = snapshot_env(env)
    x0, y0 = snap.link_x, snap.link_y
    boxes = enemy_boxes(env, max_distance=max_distance, pad=pad)
    if not boxes:
        return None
    steps = max(abs(target_x - x0), abs(target_y - y0), 1)
    steps = min(steps, 64)
    for i in range(steps + 1):
        t = i / steps
        x = int(x0 + (target_x - x0) * t)
        y = int(y0 + (target_y - y0) * t)
        for box in boxes:
            if box.contains(x, y):
                return box
    return None


def world_to_screen(
    snapshot: AlttpSnapshot,
    x: int,
    y: int,
) -> tuple[int, int]:
    """Project room coords onto the current 256×224 framebuffer."""
    return int(x) - int(snapshot.camera_x), int(y) - int(snapshot.camera_y)


def detect_edge(
    before: AlttpSnapshot,
    after: AlttpSnapshot,
    *,
    expected_room: int,
    frames: int = 0,
    label: str = "",
    preferred_direction: str | None = None,
) -> RoomEdge | None:
    """If ``after`` left ``expected_room`` (or went outdoors), describe the edge.

    Direction is inferred from link delta. Pass ``preferred_direction`` when
    the caller already knows the door (avoids outdoor teleport ambiguity).
    """
    if not before.indoors or before.room_base_id != expected_room:
        return None
    left = (not after.indoors) or after.room_base_id != expected_room
    if not left:
        return None
    if preferred_direction:
        direction = preferred_direction.upper()
    else:
        dx = after.link_x - before.link_x
        dy = after.link_y - before.link_y
        if not after.indoors and abs(dx) + abs(dy) > 200:
            # Outdoor teleport: world coords jump; use preferred or DOWN default
            # for south exits (common castle pattern). Caller should pass preferred.
            direction = "DOWN"
        elif abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
    return RoomEdge(
        direction=direction,
        from_room=expected_room,
        to_room=0 if not after.indoors else after.room_base_id,
        from_xy=(before.link_x, before.link_y),
        to_xy=(after.link_x, after.link_y),
        outdoors=not after.indoors,
        frames=frames,
        label=label,
    )


def find_edges_by_push(
    env: object,
    *,
    room: int,
    directions: Sequence[str] = ("UP", "DOWN", "LEFT", "RIGHT"),
    max_frames_per_dir: int = 360,
    step_size: int = 4,
) -> list[RoomEdge]:
    """From the current pose, push each cardinal until stuck or room exit.

    Destructive and **sequential** (each ray continues from prior stop).
    For independent rays, reload a save-state between directions.
    """
    from alttp.startup import action_for, step_frames

    found: list[RoomEdge] = []
    start = snapshot_env(env)
    if not start.indoors or start.room_base_id != room:
        return found

    for direction in directions:
        before = snapshot_env(env)
        if before.room_base_id != room or not before.indoors:
            break
        prev_xy = (before.link_x, before.link_y)
        stuck = 0
        frames = 0
        while frames < max_frames_per_dir:
            step_frames(env, action_for(direction), step_size)
            frames += step_size
            after = snapshot_env(env)
            edge = detect_edge(
                before,
                after,
                expected_room=room,
                frames=frames,
                label=direction,
                preferred_direction=direction,
            )
            if edge is not None:
                found.append(edge)
                break
            xy = (after.link_x, after.link_y)
            if xy == prev_xy:
                stuck += 1
            else:
                stuck = 0
                prev_xy = xy
            if stuck >= 12:
                break
            before = after
    return found


def draw_room_overlay(
    frame: np.ndarray,
    snapshot: AlttpSnapshot,
    *,
    boxes: Sequence[SpriteBox] = (),
    points: Sequence[RoomMapPoint] = (),
    link_box: bool = True,
    title: str = "",
) -> np.ndarray:
    """Copy RGB frame and draw sprite boxes + map points (requires Pillow)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(frame).copy())
    draw = ImageDraw.Draw(img)
    kind_color = {
        SPRITE_KIND_HOSTILE: (255, 48, 48),
        SPRITE_KIND_PICKUP: (80, 200, 255),
        SPRITE_KIND_NPC: (255, 220, 40),
        SPRITE_KIND_OTHER: (180, 180, 180),
    }

    def _rect(
        x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], width: int = 1
    ) -> None:
        s0 = world_to_screen(snapshot, x0, y0)
        s1 = world_to_screen(snapshot, x1, y1)
        draw.rectangle([s0[0], s0[1], s1[0], s1[1]], outline=color, width=width)

    if link_box:
        lx, ly = snapshot.link_x, snapshot.link_y
        _rect(lx - 8, ly - 16, lx + 8, ly + 2, (40, 255, 80), 2)

    for box in boxes:
        color = kind_color.get(box.kind, (200, 200, 200))
        _rect(box.x0, box.y0, box.x1, box.y1, color, 2)

    for pt in points:
        sx, sy = world_to_screen(snapshot, pt.x, pt.y)
        r = 3
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=(255, 255, 0), width=1)
        draw.text((sx + 4, sy - 6), pt.label[:12], fill=(255, 255, 0))

    if title:
        draw.text((4, 210), title[:40], fill=(255, 255, 0))

    return np.asarray(img)


def overlay_from_env(
    env: object,
    *,
    include_enemies: bool = True,
    include_all_sprites: bool = False,
    points: Sequence[RoomMapPoint] = (),
    title: str = "",
) -> np.ndarray:
    """Render env frame with current enemy-aware boxes."""
    snap = snapshot_env(env)
    frame = np.asarray(env.render())  # type: ignore[attr-defined]
    if include_all_sprites:
        boxes = sprite_boxes(env)
    elif include_enemies:
        boxes = enemy_boxes(env, max_distance=None)
    else:
        boxes = ()
    return draw_room_overlay(
        frame,
        snap,
        boxes=boxes,
        points=points,
        title=title or f"room 0x{snap.room_base_id:02X} ({snap.link_x},{snap.link_y})",
    )
