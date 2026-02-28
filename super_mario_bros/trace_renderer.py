"""Render Mario position trace overlaid on level map PNGs.

Reads a trace JSON (from watch --actions) and composites the trail onto
the level map image rendered by the SMB ROM editor CLI.

SMB levels are linear, so player_x maps directly to the map X coordinate.
Y is 0 at screen top, ~208 at bottom, matching the map PNG height.

Usage:
    from super_mario_bros.trace_renderer import render_smb_trace
    render_smb_trace("trace.json", "smb_1_1", "output.png")

Or via CLI:
    uv run python -m platformer_common -l smb_1_1 trace-map \\
        --trace trace.json -o output.png
"""

from __future__ import annotations

import json
from pathlib import Path

# Map dir for pre-rendered level PNGs
MAPS_DIR = Path(__file__).resolve().parent / "maps"

# Level ID -> map filename
LEVEL_MAP_FILES = {
    "smb_1_1": "smb_1_1.png",
    "smb_1_2": "smb_1_2.png",
    "smb_4_1": "smb_4_1.png",
    "smb_4_2": "smb_4_2.png",
    "smb_8_1": "smb_8_1.png",
    "smb_8_2": "smb_8_2.png",
    "smb_8_3": "smb_8_3.png",
    "smb_8_4": "smb_8_4.png",
}


def render_smb_trace(
    trace_path: str | Path,
    level_id: str,
    output_path: str | Path,
    map_dir: str | Path | None = None,
    speed_coloring: bool = True,
) -> Path:
    """Render position trace overlaid on a level map PNG.

    Args:
        trace_path: Path to trace JSON file.
        level_id: Level config ID (e.g. "smb_1_1").
        output_path: Where to write the output PNG.
        map_dir: Directory containing level map PNGs.
        speed_coloring: Color path by speed (blue=slow, red=fast).

    Returns:
        Path to the written output file.
    """
    from PIL import Image, ImageDraw, ImageFont

    trace_path = Path(trace_path)
    output_path = Path(output_path)
    if map_dir is None:
        map_dir = MAPS_DIR
    else:
        map_dir = Path(map_dir)

    # Load trace
    trace_data = json.loads(trace_path.read_text())
    trace_points = trace_data.get("trace", [])
    if not trace_points:
        raise ValueError(f"No trace points in {trace_path}")

    # Find map file
    map_key = level_id
    # Also try matching with aliases
    map_file = None
    for key, fname in LEVEL_MAP_FILES.items():
        if key == map_key or key in map_key:
            candidate = map_dir / fname
            if candidate.exists():
                map_file = candidate
                break

    if map_file is None:
        # Try auto-naming
        auto = map_dir / f"{level_id}.png"
        if auto.exists():
            map_file = auto
        else:
            available = [f for f in map_dir.glob("*.png")]
            raise FileNotFoundError(
                f"No map for '{level_id}' in {map_dir}. "
                f"Available: {[f.name for f in available]}"
            )

    base_img = Image.open(map_file).convert("RGBA")
    map_w, map_h = base_img.size

    # Convert trace points to map pixel coordinates.
    # SMB levels are linear: player_x maps directly to map X.
    # Player_y: NES screen is 240 pixels, map is 208 pixels (13 tile rows × 16px).
    # Adjust Y: map shows tiles 0-12 (y 0-208), player sprites can be 0-240.
    converted: list[tuple[int, int, int, float]] = []  # (px, py, frame, speed_x)
    for pt in trace_points:
        px = pt.get("x", 0)
        py = pt.get("y", 0)
        speed_x = pt.get("speed_x", 0)

        # Clamp to map bounds
        px = max(0, min(px, map_w - 1))
        # Map Y: SMB map height is 208px (13 tiles).
        # Player Y from RAM is sprite position (0-240).
        # Offset: map starts at tile row 0, but player y=0 is screen top.
        # Direct mapping works since map = screen height minus status bar.
        py = max(0, min(py, map_h - 1))

        converted.append((px, py, pt.get("frame", 0), speed_x))

    if not converted:
        raise ValueError("No convertible trace points")

    # Create overlay
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n_pts = len(converted)
    if n_pts >= 2:
        # Compute speed stats for color scaling
        speeds = [abs(c[3]) for c in converted]
        max_speed = max(speeds) if speeds else 1
        if max_speed == 0:
            max_speed = 1

        for i in range(n_pts - 1):
            x1, y1, _, s1 = converted[i]
            x2, y2, _, _ = converted[i + 1]

            # Skip large jumps (area transitions, door warps)
            if abs(x2 - x1) > 256:
                continue

            if speed_coloring:
                # Color by speed: blue (0) -> green (mid) -> red (fast)
                t = min(abs(s1) / max_speed, 1.0) if max_speed > 0 else 0
                if t < 0.5:
                    # Blue -> Green
                    s = t * 2
                    r, g, b = int(30 * (1 - s)), int(100 + 155 * s), int(255 * (1 - s))
                else:
                    # Green -> Red
                    s = (t - 0.5) * 2
                    r, g, b = int(255 * s), int(255 * (1 - s)), 0
            else:
                # Time gradient: blue -> red
                t = i / max(n_pts - 1, 1)
                r = int(t * 255)
                g = int(100 - t * 50)
                b = int(255 - t * 205)

            draw.line([(x1, y1), (x2, y2)], fill=(r, g, b, 200), width=2)

    # Draw center of gravity
    cog = trace_data.get("center_of_gravity")
    if cog:
        cog_x = int(cog.get("x", 0))
        cog_y = int(cog.get("y", 0))
        cog_x = max(0, min(cog_x, map_w - 1))
        cog_y = max(0, min(cog_y, map_h - 1))
        radius = 8
        draw.ellipse(
            [(cog_x - radius - 1, cog_y - radius - 1),
             (cog_x + radius + 1, cog_y + radius + 1)],
            fill=(0, 0, 0, 200),
        )
        draw.ellipse(
            [(cog_x - radius, cog_y - radius),
             (cog_x + radius, cog_y + radius)],
            fill=(255, 255, 255, 220),
        )

    # Speed statistics for legend
    speed_data = [pt.get("speed_x", 0) for pt in trace_points]
    avg_speed = sum(speed_data) / len(speed_data) if speed_data else 0
    max_fwd = max(speed_data) if speed_data else 0
    # Count frames where Mario is stationary
    stall_frames = sum(1 for s in speed_data if s == 0)
    stall_pct = stall_frames / len(speed_data) * 100 if speed_data else 0

    # Legend box
    total_frames = trace_data.get("total_frames", len(trace_points))
    level_name = trace_data.get("level", level_id)
    legend_lines = [
        f"{level_name}",
        f"frames: {total_frames} ({total_frames / 60:.1f}s)",
        f"trace pts: {n_pts}",
        f"avg speed: {avg_speed:.1f} px/f",
        f"max speed: {max_fwd} px/f",
        f"stall: {stall_frames}f ({stall_pct:.0f}%)",
    ]
    if cog:
        legend_lines.append(f"CoG: ({cog.get('x', 0):.0f}, {cog.get('y', 0):.0f})")

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    line_h = 16
    legend_w = max(font.getlength(line) for line in legend_lines) + 12
    legend_h = len(legend_lines) * line_h + 8
    draw.rectangle([(4, 4), (4 + legend_w, 4 + legend_h)], fill=(0, 0, 0, 180))
    for i, line in enumerate(legend_lines):
        draw.text((10, 8 + i * line_h), line, fill=(255, 255, 255, 240), font=font)

    # Composite overlay onto base
    result = Image.alpha_composite(base_img, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(output_path))
    print(f"Saved trace overlay: {output_path}")
    print(f"  {n_pts} points, avg speed={avg_speed:.1f} px/f, stall={stall_pct:.0f}%")
    return output_path
