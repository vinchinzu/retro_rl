"""Room-local Samus coords → *area-map* pixels (pixel-accurate).

``mapX`` / ``mapY`` in the room graph are **per-area**, not a single global
Zebes plane. Area basemaps under ``maps/legacy/<area>.png`` are sized exactly
to that area's map extent (same convention as snes_editor
``navigation/trace_renderer.py``)::

    area_x = (map_x - area_min_map_x) * 256 + samus_x
    area_y = (map_y - area_min_map_y) * 256 + samus_y

Do **not** place different areas into one coordinate system using raw mapX —
paths will look like random spaghetti (the ScriptersWar full PNG mismatch).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from super_metroid.paths import FULL_ROOM_GRAPH_PATH, MAPS_DIR

MAP_SCREEN_PX = 256

# Area basemap filenames (maps/legacy or maps/viewer/areas after prepare).
AREA_MAP_FILES = {
    "Crateria": "crateria.png",
    "Brinstar": "brinstar.png",
    "Norfair": "norfair.png",
    "Wrecked Ship": "wrecked_ship.png",
    "Maridia": "maridia.png",
    "Tourian": "tourian.png",
    "Ceres": "ceres.png",
}

AREA_SLUG = {
    "Crateria": "crateria",
    "Brinstar": "brinstar",
    "Norfair": "norfair",
    "Wrecked Ship": "wrecked_ship",
    "Maridia": "maridia",
    "Tourian": "tourian",
    "Ceres": "ceres",
}


def area_slug(area: str) -> str:
    if area in AREA_SLUG:
        return AREA_SLUG[area]
    return re.sub(r"[^a-z0-9]+", "_", area.lower()).strip("_")


@dataclass(frozen=True)
class RoomPlacement:
    """Map placement for one room (area-local map squares)."""

    room_id: int
    name: str
    area: str
    map_x: int
    map_y: int
    width_screens: int
    height_screens: int

    @property
    def width_px(self) -> int:
        return self.width_screens * MAP_SCREEN_PX

    @property
    def height_px(self) -> int:
        return self.height_screens * MAP_SCREEN_PX

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "name": self.name,
            "area": self.area,
            "map_x": self.map_x,
            "map_y": self.map_y,
            "width_screens": self.width_screens,
            "height_screens": self.height_screens,
            "width_px": self.width_px,
            "height_px": self.height_px,
        }


@dataclass(frozen=True)
class AreaBounds:
    """Pixel bounds of one area basemap."""

    area: str
    min_map_x: int
    min_map_y: int
    max_map_x: int  # exclusive (mapX + widthScreens max)
    max_map_y: int
    width_px: int
    height_px: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "area": self.area,
            "slug": area_slug(self.area),
            "min_map_x": self.min_map_x,
            "min_map_y": self.min_map_y,
            "max_map_x": self.max_map_x,
            "max_map_y": self.max_map_y,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "map_file": AREA_MAP_FILES.get(self.area),
        }


@dataclass(frozen=True)
class MapPoint:
    """One Samus center sample in *area basemap* pixel space."""

    frame: int
    room_id: int
    area: str
    x: int  # room-local
    y: int
    ax: float  # area map pixel
    ay: float
    pose: int | None = None
    phase: str | None = None

    def to_dict(self, *, compact: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "f": self.frame,
            "r": self.room_id,
            "a": area_slug(self.area),
            "ax": round(self.ax, 2),
            "ay": round(self.ay, 2),
        }
        if not compact:
            d["x"] = self.x
            d["y"] = self.y
            d["area"] = self.area
            if self.pose is not None:
                d["pose"] = self.pose
            if self.phase is not None:
                d["phase"] = self.phase
        return d


# Back-compat alias used by older imports / tests.
WorldPoint = MapPoint


def load_room_index(graph_path: Path | None = None) -> dict[int, RoomPlacement]:
    path = Path(graph_path) if graph_path else FULL_ROOM_GRAPH_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Room graph not found: {path}. Generate with "
            "scripts/export/room_problems.py or restore maps/full_room_graph.json"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    index: dict[int, RoomPlacement] = {}
    for raw in data.get("rooms") or []:
        rid = int(raw["roomId"])
        index[rid] = RoomPlacement(
            room_id=rid,
            name=str(raw.get("name") or raw.get("handle") or f"room_{rid:04X}"),
            area=str(raw.get("area") or ""),
            map_x=int(raw["mapX"]),
            map_y=int(raw["mapY"]),
            width_screens=int(raw.get("widthScreens") or 1),
            height_screens=int(raw.get("heightScreens") or 1),
        )
    return index


def area_bounds(rooms: Mapping[int, RoomPlacement] | Iterable[RoomPlacement]) -> dict[str, AreaBounds]:
    """Per-area min map square and basemap pixel size."""
    by_area: dict[str, list[RoomPlacement]] = {}
    seq = rooms.values() if isinstance(rooms, Mapping) else rooms
    for room in seq:
        if not room.area:
            continue
        by_area.setdefault(room.area, []).append(room)
    out: dict[str, AreaBounds] = {}
    for area, rs in by_area.items():
        min_x = min(r.map_x for r in rs)
        min_y = min(r.map_y for r in rs)
        max_x = max(r.map_x + r.width_screens for r in rs)
        max_y = max(r.map_y + r.height_screens for r in rs)
        out[area] = AreaBounds(
            area=area,
            min_map_x=min_x,
            min_map_y=min_y,
            max_map_x=max_x,
            max_map_y=max_y,
            width_px=(max_x - min_x) * MAP_SCREEN_PX,
            height_px=(max_y - min_y) * MAP_SCREEN_PX,
        )
    return out


def to_area(
    room: RoomPlacement,
    bounds: AreaBounds,
    samus_x: float | int,
    samus_y: float | int,
    *,
    x_sub: int | None = None,
    y_sub: int | None = None,
) -> tuple[float, float]:
    """Room-local Samus → area basemap pixels."""
    fx = float(samus_x)
    fy = float(samus_y)
    if x_sub is not None:
        fx += (int(x_sub) & 0xFFFF) / 65536.0
    if y_sub is not None:
        fy += (int(y_sub) & 0xFFFF) / 65536.0
    ax = (room.map_x - bounds.min_map_x) * MAP_SCREEN_PX + fx
    ay = (room.map_y - bounds.min_map_y) * MAP_SCREEN_PX + fy
    return ax, ay


# Deprecated name: previously meant fake global scripterswar coords.
def to_world(
    room: RoomPlacement,
    samus_x: float | int,
    samus_y: float | int,
    *,
    x_sub: int | None = None,
    y_sub: int | None = None,
    bounds: AreaBounds | None = None,
) -> tuple[float, float]:
    """Area-map pixels (requires *bounds* for correct origin)."""
    if bounds is None:
        # Absolute map squares only — wrong across areas; kept for tests of
        # arithmetic. Prefer to_area() with real bounds.
        fx = float(samus_x)
        fy = float(samus_y)
        if x_sub is not None:
            fx += (int(x_sub) & 0xFFFF) / 65536.0
        if y_sub is not None:
            fy += (int(y_sub) & 0xFFFF) / 65536.0
        return room.map_x * MAP_SCREEN_PX + fx, room.map_y * MAP_SCREEN_PX + fy
    return to_area(room, bounds, samus_x, samus_y, x_sub=x_sub, y_sub=y_sub)


def point_from_sample(
    rooms: Mapping[int, RoomPlacement],
    bounds_by_area: Mapping[str, AreaBounds],
    *,
    room_id: int,
    x: int | float,
    y: int | float,
    frame: int = 0,
    x_sub: int | None = None,
    y_sub: int | None = None,
    pose: int | None = None,
    phase: str | None = None,
    skip_offmap: bool = True,
) -> MapPoint | None:
    if skip_offmap and (float(x) > 60_000 or float(y) > 60_000):
        return None
    room = rooms.get(int(room_id))
    if room is None or not room.area:
        return None
    bounds = bounds_by_area.get(room.area)
    if bounds is None:
        return None
    ax, ay = to_area(room, bounds, x, y, x_sub=x_sub, y_sub=y_sub)
    return MapPoint(
        frame=int(frame),
        room_id=int(room_id),
        area=room.area,
        x=int(x),
        y=int(y),
        ax=ax,
        ay=ay,
        pose=pose,
        phase=phase,
    )


def rooms_geojson_for_area(
    rooms: Iterable[RoomPlacement],
    bounds: AreaBounds,
) -> dict[str, Any]:
    """GeoJSON room rectangles in area basemap pixels ([x,y] for Leaflet)."""
    features = []
    for room in rooms:
        if room.area != bounds.area:
            continue
        x0 = (room.map_x - bounds.min_map_x) * MAP_SCREEN_PX
        y0 = (room.map_y - bounds.min_map_y) * MAP_SCREEN_PX
        x1 = x0 + room.width_px
        y1 = y0 + room.height_px
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        props = room.to_dict()
        props["ax0"] = x0
        props["ay0"] = y0
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def default_viewer_asset_dir() -> Path:
    return MAPS_DIR / "viewer"


# Legacy constant removed from API surface; keep number for any external refs.
WORLD_SIZE_PX = 0  # unused — area basemaps only
