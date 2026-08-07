"""Area-basemap Super Metroid path viewer (pixel-accurate CoG traces).

Uses per-area maps (``maps/legacy/crateria.png``, …) with::

    area_x = (mapX - area_min_map_x) * 256 + samus_x

Paths are split into same-room short-step segments — never drawn straight
across the map between rooms.
"""

from __future__ import annotations

from super_metroid.map_viewer.coords import (
    MAP_SCREEN_PX,
    MapPoint,
    RoomPlacement,
    area_bounds,
    load_room_index,
    to_area,
)
from super_metroid.map_viewer.paths import WorldPath, export_path, load_path_source

__all__ = [
    "MAP_SCREEN_PX",
    "MapPoint",
    "RoomPlacement",
    "WorldPath",
    "area_bounds",
    "export_path",
    "load_path_source",
    "load_room_index",
    "to_area",
]
