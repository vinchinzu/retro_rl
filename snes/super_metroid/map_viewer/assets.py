"""Prepare *area* basemaps + room geojson for the static viewer.

Uses ``maps/legacy/<area>.png`` which already match graph mapX/mapY extents
pixel-for-pixel (unlike the ScriptersWar full-tile montage).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from super_metroid.map_viewer.coords import (
    AREA_MAP_FILES,
    area_bounds,
    area_slug,
    default_viewer_asset_dir,
    load_room_index,
    rooms_geojson_for_area,
)
from super_metroid.paths import MAPS_DIR

LEGACY_MAP_DIR = MAPS_DIR / "legacy"
PACKAGE_VIEWER_DIR = Path(__file__).resolve().parent / "static"
# snes_editor sibling fallback
EDITOR_MAP_DIR = (
    MAPS_DIR.parent.parent.parent.parent
    / "snes_editor"
    / "super_metroid_rl"
    / "maps"
)


def _find_area_png(filename: str) -> Path | None:
    for root in (LEGACY_MAP_DIR, MAPS_DIR, EDITOR_MAP_DIR):
        cand = root / filename
        if cand.is_file():
            return cand
    return None


def prepare_areas(
    *,
    out_dir: Path | None = None,
    force: bool = False,
    max_edge: int = 0,
) -> dict[str, Any]:
    """Copy area basemaps 1:1 + write per-area geojson + basemap.json.

    Default keeps full resolution so path ``ax/ay`` stay pixel-identical to
    ``maps/legacy/*.png``. Pass ``max_edge`` only if you intentionally accept
    a scaled basemap (viewer multiplies path coords by ``pixel_scale``).
    """
    from PIL import Image

    out_dir = Path(out_dir) if out_dir else default_viewer_asset_dir()
    areas_dir = out_dir / "areas"
    rooms_dir = out_dir / "rooms"
    areas_dir.mkdir(parents=True, exist_ok=True)
    rooms_dir.mkdir(parents=True, exist_ok=True)

    rooms = load_room_index()
    bounds = area_bounds(rooms)
    area_meta: list[dict[str, Any]] = []

    for area, b in sorted(bounds.items(), key=lambda kv: kv[0]):
        filename = AREA_MAP_FILES.get(area)
        if not filename:
            continue
        src = _find_area_png(filename)
        if src is None:
            continue
        slug = area_slug(area)
        dest = areas_dir / f"{slug}.png"
        scale = 1.0
        if force or not dest.is_file():
            # Prefer hardlink/copy for 1:1 pixel fidelity (no re-encode).
            if max_edge and max_edge > 0:
                im = Image.open(src).convert("RGBA")
                w, h = im.size
                long_edge = max(w, h)
                if long_edge > max_edge:
                    scale = max_edge / long_edge
                    im = im.resize(
                        (int(w * scale), int(h * scale)),
                        Image.Resampling.LANCZOS,
                    )
                    im.save(dest, optimize=True)
                else:
                    shutil.copy2(src, dest)
            else:
                shutil.copy2(src, dest)
        with Image.open(dest) as im:
            w, h = im.size
        if b.width_px > 0:
            scale = w / b.width_px

        geo = rooms_geojson_for_area(rooms.values(), b)
        if abs(scale - 1.0) > 1e-6:
            for feat in geo["features"]:
                ring = feat["geometry"]["coordinates"][0]
                feat["geometry"]["coordinates"][0] = [
                    [c[0] * scale, c[1] * scale] for c in ring
                ]
                feat["properties"]["ax0"] = feat["properties"]["ax0"] * scale
                feat["properties"]["ay0"] = feat["properties"]["ay0"] * scale

        geo_path = rooms_dir / f"{slug}.geojson"
        geo_path.write_text(json.dumps(geo), encoding="utf-8")

        meta = b.to_dict()
        meta.update(
            {
                "file": f"areas/{slug}.png",
                "rooms_file": f"rooms/{slug}.geojson",
                "pixel_scale": scale,
                "display_width": w,
                "display_height": h,
                "source": str(src),
                "size_ok": abs(w - b.width_px) <= 2 and abs(h - b.height_px) <= 2,
            }
        )
        area_meta.append(meta)

    basemap = {
        "schema": "super_metroid_area_basemap_v2",
        "coord_note": (
            "area_x = (mapX - area_min_map_x)*256 + samus_x; "
            "paths store ax/ay already in area basemap pixels "
            "(multiply by pixel_scale when rendering if basemap was downscaled)"
        ),
        "areas": area_meta,
        "default_area": "crateria",
    }
    basemap_path = out_dir / "basemap.json"
    basemap_path.write_text(json.dumps(basemap, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": str(out_dir), "areas": len(area_meta), "basemap": str(basemap_path)}


def sync_static_viewer(out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else default_viewer_asset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css"):
        src = PACKAGE_VIEWER_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing viewer static asset: {src}")
        shutil.copy2(src, out_dir / name)
    return out_dir / "index.html"


def prepare_all(
    *,
    overview_size: int = 0,
    force: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Build area basemaps, rooms geojson, and sync static UI.

    ``overview_size`` 0 = full-res copy (pixel-accurate). Positive = max long edge.
    """
    out_dir = Path(out_dir) if out_dir else default_viewer_asset_dir()
    area_result = prepare_areas(out_dir=out_dir, force=force, max_edge=overview_size)
    index_html = sync_static_viewer(out_dir)
    return {**area_result, "index": str(index_html)}
