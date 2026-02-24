"""Render Samus position trace overlaid on area map PNGs.

Reads a trace JSON (from watch --actions), loads room metadata from
/tmp/sm_export/, and composites the trail onto the area map image.

Usage:
    from super_metroid_rl.navigation.trace_renderer import render_trace_on_map
    render_trace_on_map("trace.json", "crateria", "output.png")

Or via CLI:
    uv run python -m platformer_common -l sm_parlor_descent trace-map \\
        --trace trace.json --area crateria -o output.png
"""

from __future__ import annotations

import json
from pathlib import Path

from super_metroid_rl.navigation.map_data import (
    DEFAULT_EXPORT_DIR,
    NavNode,
    load_nav_graph,
)

# Area name normalization
AREA_NAMES = {
    0: "crateria",
    1: "brinstar",
    2: "norfair",
    3: "wrecked_ship",
    4: "maridia",
    5: "tourian",
    6: "ceres",
}
AREA_NAME_TO_ID = {v: k for k, v in AREA_NAMES.items()}
# Accept common aliases
AREA_NAME_TO_ID.update({
    "wrecked ship": 3,
    "wreckedship": 3,
    "wrecked-ship": 3,
})

# Annotation tag colors (RGB)
TAG_COLORS = {
    "ledge_hit": (255, 80, 80),
    "bad_path": (255, 165, 0),
    "slow": (255, 255, 0),
    "good": (80, 255, 80),
    "other": (160, 160, 160),
}

# Map PNG filename lookup (area_name -> filename in maps/)
AREA_MAP_FILES = {
    "crateria": "crateria.png",
    "brinstar": "brinstar.png",
    "norfair": "norfair.png",
    "wrecked_ship": "wrecked_ship.png",
    "maridia": "maridia.png",
    "tourian": "tourian.png",
    "ceres": "ceres.png",
}


def _load_nodes(export_dir: Path) -> dict[int, NavNode]:
    """Load nav graph nodes keyed by room_id."""
    nav_path = export_dir / "nav_graph.json"
    if not nav_path.exists():
        raise FileNotFoundError(f"Nav graph not found: {nav_path}")
    nodes, _ = load_nav_graph(nav_path)
    return {n.room_id: n for n in nodes}


def _area_for_room(node: NavNode) -> str:
    """Get normalized area name for a node."""
    return AREA_NAMES.get(node.area, node.area_name.lower().replace(" ", "_"))


def detect_area(trace_data: dict, nodes: dict[int, NavNode]) -> str | None:
    """Auto-detect the primary area from trace room IDs."""
    from collections import Counter

    area_counts: Counter[str] = Counter()
    for pt in trace_data.get("trace", []):
        room_id = pt.get("room_id", 0)
        node = nodes.get(room_id)
        if node:
            area_counts[_area_for_room(node)] += 1
    if area_counts:
        return area_counts.most_common(1)[0][0]
    return None


def render_trace_on_map(
    trace_path: str | Path,
    area_name: str,
    output_path: str | Path,
    map_dir: str | Path | None = None,
    export_dir: str | Path = DEFAULT_EXPORT_DIR,
) -> Path:
    """Render position trace overlaid on an area map PNG.

    Args:
        trace_path: Path to trace JSON file.
        area_name: Area name (crateria, brinstar, etc.).
        output_path: Where to write the output PNG.
        map_dir: Directory containing area map PNGs.
        export_dir: SM export directory with nav_graph.json.

    Returns:
        Path to the written output file.
    """
    from PIL import Image, ImageDraw, ImageFont

    trace_path = Path(trace_path)
    output_path = Path(output_path)
    export_dir = Path(export_dir)

    if map_dir is None:
        map_dir = Path(__file__).resolve().parent.parent / "maps"
    else:
        map_dir = Path(map_dir)

    # Load trace
    trace_data = json.loads(trace_path.read_text())
    trace_points = trace_data.get("trace", [])
    if not trace_points:
        raise ValueError(f"No trace points in {trace_path}")

    # Load nodes
    nodes = _load_nodes(export_dir)

    # Normalize area name
    area_key = area_name.lower().replace(" ", "_").replace("-", "_")
    if area_key not in AREA_MAP_FILES:
        raise ValueError(
            f"Unknown area '{area_name}'. Known: {', '.join(AREA_MAP_FILES)}"
        )

    # Load area map
    map_file = map_dir / AREA_MAP_FILES[area_key]
    if not map_file.exists():
        raise FileNotFoundError(f"Area map not found: {map_file}")
    base_img = Image.open(map_file).convert("RGBA")

    # Find area bounds (min mapX/mapY for this area)
    area_id = AREA_NAME_TO_ID.get(area_key)
    area_nodes = [n for n in nodes.values() if n.area == area_id]
    if not area_nodes:
        raise ValueError(f"No rooms found for area '{area_name}' (id={area_id})")

    min_map_x = min(n.map_x for n in area_nodes)
    min_map_y = min(n.map_y for n in area_nodes)

    # Convert trace points to area-map pixel coordinates
    converted: list[tuple[int, int, int]] = []  # (area_px, area_py, frame)
    for pt in trace_points:
        room_id = pt.get("room_id", 0)
        node = nodes.get(room_id)
        if not node:
            continue
        # Only include points for rooms in this area
        if node.area != area_id:
            continue

        px = pt.get("x", 0)
        py = pt.get("y", 0)
        area_px = (node.map_x - min_map_x) * 256 + px
        area_py = (node.map_y - min_map_y) * 256 + py
        converted.append((area_px, area_py, pt.get("frame", 0)))

    if not converted:
        raise ValueError(
            f"No trace points fall within area '{area_name}'. "
            f"Check --area or trace room IDs."
        )

    # Create overlay
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw trail as connected line segments with time gradient (blue -> red)
    n_pts = len(converted)
    if n_pts >= 2:
        for i in range(n_pts - 1):
            t = i / max(n_pts - 1, 1)
            # Blue (0,100,255) -> Red (255,50,50)
            r = int(0 + t * 255)
            g = int(100 - t * 50)
            b = int(255 - t * 205)
            x1, y1, _ = converted[i]
            x2, y2, _ = converted[i + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, 180), width=2)

    # Draw center of gravity
    cog = trace_data.get("center_of_gravity")
    if cog:
        cog_room = cog.get("room_id", 0)
        cog_node = nodes.get(cog_room)
        if cog_node and cog_node.area == area_id:
            cog_ax = (cog_node.map_x - min_map_x) * 256 + cog["x"]
            cog_ay = (cog_node.map_y - min_map_y) * 256 + cog["y"]
            r = 8
            # Black outline
            draw.ellipse(
                [(cog_ax - r - 1, cog_ay - r - 1), (cog_ax + r + 1, cog_ay + r + 1)],
                fill=(0, 0, 0, 200),
            )
            # White fill
            draw.ellipse(
                [(cog_ax - r, cog_ay - r), (cog_ax + r, cog_ay + r)],
                fill=(255, 255, 255, 220),
            )

    # Draw annotation markers
    annotations = trace_data.get("annotations", [])
    # Build frame -> trace-point lookup for positioning annotations
    frame_to_pos: dict[int, tuple[int, int]] = {}
    for ax, ay, frame in converted:
        frame_to_pos[frame] = (ax, ay)

    for ann in annotations:
        frame = ann.get("frame", -1)
        pos = frame_to_pos.get(frame)
        if not pos:
            continue
        tags = ann.get("tags", [])
        color = TAG_COLORS.get(tags[0], (160, 160, 160)) if tags else (160, 160, 160)
        ax, ay = pos
        r = 6
        draw.ellipse(
            [(ax - r, ay - r), (ax + r, ay + r)],
            fill=(*color, 220),
            outline=(0, 0, 0, 180),
            width=1,
        )

    # Legend box (top-left)
    total_frames = trace_data.get("total_frames", len(trace_points))
    level_name = trace_data.get("level", "?")
    legend_lines = [
        f"{level_name}",
        f"frames: {total_frames} ({total_frames / 60:.1f}s)",
        f"trace pts: {n_pts} in {area_name}",
    ]
    if cog:
        legend_lines.append(f"CoG: ({cog['x']:.0f}, {cog['y']:.0f})")

    # Draw legend background
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

    line_h = 16
    legend_w = max(font.getlength(line) for line in legend_lines) + 12
    legend_h = len(legend_lines) * line_h + 8
    draw.rectangle([(4, 4), (4 + legend_w, 4 + legend_h)], fill=(0, 0, 0, 160))
    for i, line in enumerate(legend_lines):
        draw.text((10, 8 + i * line_h), line, fill=(255, 255, 255, 240), font=font)

    # Composite overlay onto base
    result = Image.alpha_composite(base_img, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"Saved trace overlay: {output_path}")
    print(f"  {n_pts} points in {area_name}, {len(annotations)} annotations")
    return output_path
