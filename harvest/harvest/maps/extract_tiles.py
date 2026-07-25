#!/usr/bin/env python3
"""Reference map comparison utilities for Harvest Moon SNES.

Renders maps from ROM + save-state data (no emulator needed) and compares them
against reference images for alignment validation.

Usage:
    uv run python extract_tiles.py --compare-reference ranch --compare-state Y1_After_Buy_Potato --compare-dir debug_alignment/reference_compare
    uv run python extract_tiles.py --export-reference-png town --export-reference-output debug_alignment/reference_exports/town.png
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np

from harvest.paths import DEBUG_ALIGNMENT_DIR, MAPS_DIR, PROJECT_DIR
from harvest.runtime.rom_tools import (
    HarvestMoonRom,
    build_metatile_atlas,
    parse_save_state,
    read_metatile_grid,
    read_tilemap_id,
    render_full_map,
    resolve_state_path,
)

SCRIPT_DIR = PROJECT_DIR

REFERENCE_MAP_SOURCES = {
    "ranch": DEBUG_ALIGNMENT_DIR / "reference_ranch_map.png",
    "town": MAPS_DIR / "harvest-moon-town-snes-map.webp",
    "mountain": MAPS_DIR / "harvest-moon-mountain-spring-clear-snes-map.webp",
    "mountain_spring_clear": MAPS_DIR / "harvest-moon-mountain-spring-clear-snes-map.webp",
}

TILE = 16
MAP_W = 64
SW, SH = 256, 224
MAP_PX_W = MAP_W * TILE
MAP_PX_H = MAP_W * TILE


@dataclass(frozen=True)
class ReferenceMapReport:
    state_name: str
    reference_source: str
    tilemap_id: int
    best_x: int
    best_y: int
    chunk_size: int
    compared_chunks: int
    mean_rgb_error: float
    mean_structural_error: float
    max_structural_error: float


def camera_offset(px: int, py: int) -> tuple[int, int]:
    """Calculate the top-left viewport pixel offset from player position."""
    cx = max(0, min(px - SW // 2, MAP_PX_W - SW))
    cy = max(0, min(py - SH // 2, MAP_PX_H - SH))
    return cx, cy


def mean_rgb_error(atlas_patch: np.ndarray, obs_patch: np.ndarray) -> float:
    return float(np.abs(atlas_patch.astype(np.int16) - obs_patch.astype(np.int16)).mean())


def _gray(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    return 0.299 * img_f[..., 0] + 0.587 * img_f[..., 1] + 0.114 * img_f[..., 2]


def structural_tile_error(atlas_patch: np.ndarray, obs_patch: np.ndarray) -> float:
    """Compare tile structure while ignoring palette/brightness drift."""
    atlas_gray = _gray(atlas_patch)
    obs_gray = _gray(obs_patch)
    atlas_gray = atlas_gray - atlas_gray.mean()
    obs_gray = obs_gray - obs_gray.mean()
    atlas_std = float(atlas_gray.std())
    obs_std = float(obs_gray.std())
    if atlas_std > 1e-3:
        atlas_gray /= atlas_std
    if obs_std > 1e-3:
        obs_gray /= obs_std
    return float(np.abs(atlas_gray - obs_gray).mean())


def _write_ppm(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.ascontiguousarray(img.astype(np.uint8))
    with path.open("wb") as handle:
        handle.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
        handle.write(rgb.tobytes())


def save_rgb_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".ppm":
        _write_ppm(path, img)
        return
    if suffix != ".png":
        raise ValueError(f"Unsupported image suffix for {path}")

    from PySide6.QtGui import QImage

    rgb = np.ascontiguousarray(img.astype(np.uint8))
    qimg = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.shape[1] * 3,
        QImage.Format.Format_RGB888,
    )
    if not qimg.save(str(path), "PNG"):
        raise RuntimeError(f"Could not save PNG: {path}")


def resolve_reference_source(source: str) -> str:
    alias_path = REFERENCE_MAP_SOURCES.get(source.lower())
    if alias_path is not None:
        if not alias_path.exists():
            raise RuntimeError(f"Reference asset not found for {source}: {alias_path}")
        return str(alias_path)
    return source


def load_rgb_image(source: str) -> np.ndarray:
    """Load an RGB image from disk, a direct image URL, or a simple HTML page URL."""
    from PySide6.QtGui import QImage

    source = resolve_reference_source(source)
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        image_bytes = _load_remote_image_bytes(source)
        qimg = QImage.fromData(image_bytes)
    else:
        qimg = QImage(source)
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    if qimg.isNull():
        raise RuntimeError(f"Could not load image: {source}")
    width = qimg.width()
    height = qimg.height()
    arr = np.frombuffer(qimg.bits(), dtype=np.uint8)
    arr = arr.reshape((height, qimg.bytesPerLine()))[:, : width * 3]
    return arr.reshape((height, width, 3)).copy()


def export_reference_png(reference_source: str, output_path: Path) -> Path:
    save_rgb_image(output_path, load_rgb_image(reference_source))
    return output_path


def _slug_reference_name(reference_source: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", reference_source).strip("_") or "reference_map"


def _load_remote_image_bytes(source: str) -> bytes:
    def _fetch(url: str) -> tuple[bytes, str, str]:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": url,
            },
        )
        try:
            with urlopen(req) as response:
                content_type = response.headers.get_content_type()
                final_url = response.geturl()
                return response.read(), content_type, final_url
        except HTTPError as exc:
            parsed_url = urlparse(url)
            if parsed_url.netloc.endswith("vgmaps.de") and parsed_url.path.endswith("/maps/view.php"):
                raise RuntimeError(
                    "VGMaps page URLs on vgmaps.de are Cloudflare-protected from CLI fetches; "
                    "use a direct image URL such as https://www.vgmaps.com/atlas/SuperNES/HarvestMoon-Ranch.png"
                ) from exc
            raise RuntimeError(f"Could not fetch {url}: HTTP {exc.code}") from exc

    data, content_type, final_url = _fetch(source)
    if content_type.startswith("image/"):
        return data

    html = data.decode("utf-8", errors="ignore")
    candidates = re.findall(r"""(?:src|href)=["']([^"']+\.(?:png|webp))["']""", html, flags=re.IGNORECASE)
    if not candidates:
        raise RuntimeError(f"Could not find an image link in {source}")

    preferred = None
    for candidate in candidates:
        lower = candidate.lower()
        if "harvest" in lower and "ranch" in lower:
            preferred = candidate
            break
    image_url = urljoin(final_url, preferred or candidates[0])
    image_bytes, image_type, _ = _fetch(image_url)
    if not image_type.startswith("image/"):
        raise RuntimeError(f"Resolved non-image URL from {source}: {image_url}")
    return image_bytes


def find_best_reference_crop(rendered: np.ndarray, reference: np.ndarray) -> tuple[int, int, float]:
    """Find the best crop of the rendered map that aligns to the reference image."""
    render_h, render_w = rendered.shape[:2]
    ref_h, ref_w = reference.shape[:2]
    if ref_h > render_h or ref_w > render_w:
        raise ValueError(
            f"Reference image {reference.shape[:2]} is larger than rendered map {rendered.shape[:2]}"
        )

    ref_gray = _gray(reference)
    ref_gray = (ref_gray - ref_gray.mean()) / max(float(ref_gray.std()), 1e-6)
    best_score: float | None = None
    best_x = 0
    best_y = 0
    for y in range(render_h - ref_h + 1):
        for x in range(render_w - ref_w + 1):
            crop = _gray(rendered[y : y + ref_h, x : x + ref_w])
            crop = (crop - crop.mean()) / max(float(crop.std()), 1e-6)
            score = float(np.abs(crop - ref_gray).mean())
            if best_score is None or score < best_score:
                best_score = score
                best_x = x
                best_y = y
    assert best_score is not None
    return best_x, best_y, best_score


def compare_rendered_map_to_reference(
    rendered: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 128,
) -> tuple[ReferenceMapReport, np.ndarray, list[dict[str, object]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    best_x, best_y, _ = find_best_reference_crop(rendered, reference)
    ref_h, ref_w = reference.shape[:2]
    aligned = rendered[best_y : best_y + ref_h, best_x : best_x + ref_w]

    heatmap = np.zeros((ref_h, ref_w), dtype=np.uint8)
    rgb_errors: list[float] = []
    structural_errors: list[float] = []
    rows: list[dict[str, object]] = []

    for chunk_y in range(0, ref_h, chunk_size):
        for chunk_x in range(0, ref_w, chunk_size):
            y1 = min(chunk_y + chunk_size, ref_h)
            x1 = min(chunk_x + chunk_size, ref_w)
            render_chunk = aligned[chunk_y:y1, chunk_x:x1]
            ref_chunk = reference[chunk_y:y1, chunk_x:x1]
            rgb_error = mean_rgb_error(render_chunk, ref_chunk)
            structural_error = structural_tile_error(render_chunk, ref_chunk)
            heatmap[chunk_y:y1, chunk_x:x1] = min(255, int(structural_error * 96))
            rgb_errors.append(rgb_error)
            structural_errors.append(structural_error)
            rows.append(
                {
                    "chunk_x": chunk_x,
                    "chunk_y": chunk_y,
                    "width": x1 - chunk_x,
                    "height": y1 - chunk_y,
                    "rgb_error": rgb_error,
                    "structural_error": structural_error,
                }
            )

    heatmap_rgb = np.repeat(heatmap[:, :, None], 3, axis=2)
    debug_img = np.concatenate([aligned, reference, heatmap_rgb], axis=1)
    report = ReferenceMapReport(
        state_name="",
        reference_source="",
        tilemap_id=0,
        best_x=best_x,
        best_y=best_y,
        chunk_size=chunk_size,
        compared_chunks=len(rows),
        mean_rgb_error=float(np.mean(rgb_errors)) if rgb_errors else 0.0,
        mean_structural_error=float(np.mean(structural_errors)) if structural_errors else 0.0,
        max_structural_error=float(np.max(structural_errors)) if structural_errors else 0.0,
    )
    return report, debug_img, rows


def export_reference_chunk_table(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "state_name",
        "reference_source",
        "tilemap_id",
        "best_x",
        "best_y",
        "chunk_x",
        "chunk_y",
        "width",
        "height",
        "rgb_error",
        "structural_error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def compare_state_map_reference(
    state_name: str,
    reference_source: str,
    *,
    output_dir: Path | None = None,
    chunk_size: int = 128,
) -> tuple[ReferenceMapReport, list[dict[str, object]]]:
    """Compare a ROM-rendered map to a reference image (no emulator needed)."""
    rom = HarvestMoonRom.load()
    state = parse_save_state(resolve_state_path(state_name))
    tilemap_id = read_tilemap_id(state.ram)
    scene = rom.read_map_scene(tilemap_id)
    atlas = build_metatile_atlas(state, scene.graphic_preset.bg12nba)
    grid = read_metatile_grid(state.ram)
    rendered = render_full_map(atlas, grid)

    reference = load_rgb_image(reference_source)
    base_report, debug_img, rows = compare_rendered_map_to_reference(
        rendered,
        reference,
        chunk_size=chunk_size,
    )
    report = ReferenceMapReport(
        state_name=state_name,
        reference_source=reference_source,
        tilemap_id=tilemap_id,
        best_x=base_report.best_x,
        best_y=base_report.best_y,
        chunk_size=base_report.chunk_size,
        compared_chunks=base_report.compared_chunks,
        mean_rgb_error=base_report.mean_rgb_error,
        mean_structural_error=base_report.mean_structural_error,
        max_structural_error=base_report.max_structural_error,
    )
    for row in rows:
        row["state_name"] = state_name
        row["reference_source"] = reference_source
        row["tilemap_id"] = tilemap_id
        row["best_x"] = report.best_x
        row["best_y"] = report.best_y

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_rgb_image(output_dir / f"{state_name}_rendered_map.png", rendered)
        save_rgb_image(output_dir / f"{state_name}_reference_compare.png", debug_img)
        export_reference_chunk_table(rows, output_dir / f"{state_name}_reference_chunks.csv")

    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Moon map reference comparison")
    parser.add_argument("--compare-reference", default=None, help="Reference image path or alias for full-map comparison")
    parser.add_argument("--compare-state", default=None, help="State name for full-map reference comparison")
    parser.add_argument("--compare-dir", default=None, help="Optional directory for reference comparison exports")
    parser.add_argument("--compare-chunk-size", type=int, default=128, help="Chunk size in pixels for reference comparison CSV/heatmap")
    parser.add_argument("--export-reference-png", default=None, help="Reference map alias/path/URL to export as a clean PNG")
    parser.add_argument("--export-reference-output", default=None, help="Output path for --export-reference-png")
    args = parser.parse_args()

    if args.compare_reference:
        compare_state = args.compare_state or "Y1_After_Buy_Potato"
        compare_dir = Path(args.compare_dir) if args.compare_dir else SCRIPT_DIR / "debug_alignment" / "reference_compare"
        try:
            compare_report, _compare_rows = compare_state_map_reference(
                compare_state,
                args.compare_reference,
                output_dir=compare_dir,
                chunk_size=args.compare_chunk_size,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print("\nReference map comparison:")
        print(
            f"  {compare_report.state_name}: tilemap=0x{compare_report.tilemap_id:02X}"
            f" crop=({compare_report.best_x},{compare_report.best_y})"
            f" chunks={compare_report.compared_chunks}"
            f" rgb_mean={compare_report.mean_rgb_error:.1f}"
            f" structural_mean={compare_report.mean_structural_error:.3f}"
            f" structural_max={compare_report.max_structural_error:.3f}"
        )
        print(f"  wrote reference comparison outputs to {compare_dir}")

    if args.export_reference_png:
        default_name = _slug_reference_name(args.export_reference_png)
        output_path = Path(args.export_reference_output) if args.export_reference_output else SCRIPT_DIR / "debug_alignment" / "reference_exports" / f"{default_name}.png"
        try:
            written = export_reference_png(args.export_reference_png, output_path)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        print("\nReference PNG export:")
        print(f"  wrote {written}")

    if not args.compare_reference and not args.export_reference_png:
        parser.print_help()


if __name__ == "__main__":
    main()
