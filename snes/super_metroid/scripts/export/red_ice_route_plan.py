#!/usr/bin/env python3
"""Render deterministic PNGs for the Red Tower Ice checkpoint plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from super_metroid.paths import GAME_DIR

PLAN = GAME_DIR / "routes" / "kpdr" / "data" / "red_tower_ice_checkpoint_plan.json"
ROOM_IMAGE = (
    GAME_DIR
    / "refs"
    / "sm-json-data"
    / "region"
    / "brinstar"
    / "roomDiagrams"
    / "red_RedTower_66.png"
)
REF_DIR = GAME_DIR / "docs" / "tasks" / "refs"
CONTEXT_IMAGE = REF_DIR / "red_tower_bottom_ice_reference.png"
FULL_OUTPUT = REF_DIR / "red_tower_ice_checkpoint_plan.png"
EDGE_OUTPUT = REF_DIR / "red_tower_ice_first_edge.png"

GREEN = "#35e67a"
BLUE = "#49b8ff"
GRAY = "#aab2bd"
YELLOW = "#ffd65a"
RED = "#ff626e"
INK = "#ecf2f8"
PANEL = "#101722"
PANEL_2 = "#182433"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/liberation/{name}", size)


def _status_color(status: str) -> str:
    if status.startswith("verified"):
        return GREEN
    if status.startswith("observed"):
        return BLUE
    return GRAY


def _center(checkpoint: dict) -> tuple[int, int]:
    x0, x1 = checkpoint["x"]
    y0, y1 = checkpoint["y"]
    # Room diagram is 247 px wide for 256 px of WRAM room coordinates.
    return round(((x0 + x1) / 2) * 247 / 256), round((y0 + y1) / 2)


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    points: Iterable[tuple[int, int]],
    *,
    fill: str,
    width: int = 3,
    dash: int = 12,
) -> None:
    values = list(points)
    for start, end in zip(values, values[1:]):
        x0, y0 = start
        x1, y1 = end
        length = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))
        for offset in range(0, length, dash * 2):
            a = offset / length
            b = min(length, offset + dash) / length
            draw.line(
                (
                    round(x0 + (x1 - x0) * a),
                    round(y0 + (y1 - y0) * a),
                    round(x0 + (x1 - x0) * b),
                    round(y0 + (y1 - y0) * b),
                ),
                fill=fill,
                width=width,
            )


def _arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str,
    width: int = 5,
) -> None:
    draw.line(points, fill=fill, width=width, joint="curve")
    if len(points) < 2:
        return
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    dx, dy = x1 - x0, y1 - y0
    mag = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / mag, dy / mag
    px, py = -uy, ux
    size = max(10, width * 3)
    left = (round(x1 - ux * size + px * size * 0.55), round(y1 - uy * size + py * size * 0.55))
    right = (round(x1 - ux * size - px * size * 0.55), round(y1 - uy * size - py * size * 0.55))
    draw.polygon([(x1, y1), left, right], fill=fill)


def render_full(plan: dict, output: Path) -> Path:
    room = Image.open(ROOM_IMAGE).convert("RGB")
    canvas = Image.new("RGB", (1120, room.height), PANEL)
    canvas.paste(room, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30, bold=True)
    label_font = _font(18, bold=True)
    small_font = _font(15)

    checkpoints = plan["checkpoints"]
    by_id = {row["id"]: row for row in checkpoints}
    ordered = sorted(checkpoints, key=lambda row: sum(row["y"]) / 2)

    # Planned main route, then the verified first edge over it.
    route_points = [_center(row) for row in reversed(ordered)]
    _dashed_line(draw, route_points, fill=GRAY, width=3, dash=10)
    verified_arc = [(208, 2443), (92, 2443), (208, 2245), (155, 2104), (82, 2351)]
    _arrow(draw, verified_arc, fill=GREEN, width=6)

    draw.rectangle((247, 0, 1119, 82), fill="#0a1019")
    draw.text((276, 14), "RED TOWER ICE CLIMB — CHECKPOINT TREE", font=title_font, fill=INK)
    draw.text(
        (278, 52),
        "green = emulator-verified edge · blue = observed seat · gray = planned",
        font=small_font,
        fill=GRAY,
    )

    last_label_y = -100
    for index, row in enumerate(ordered, start=1):
        px, py = _center(row)
        color = _status_color(row["status"])
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=color, outline="#071018", width=2)
        label_y = max(py - 11, last_label_y + 25)
        label_y = min(label_y, room.height - 26)
        last_label_y = label_y
        draw.line((px + 9, py, 274, label_y + 9), fill=color, width=2)
        tag = f"C{index:02d}"
        draw.rounded_rectangle((282, label_y, 1085, label_y + 23), 5, fill=PANEL_2)
        draw.text((292, label_y + 2), tag, font=small_font, fill=color)
        status = row["status"].replace("_", " ")
        text = f"{row['id']}  y≈{round(sum(row['y']) / 2)}  · {row['kind']} · {status}"
        draw.text((342, label_y + 2), text, font=small_font, fill=INK)

    # Recovery funnels are deliberately separate from the optimistic path.
    recovery = (
        (by_id["lower_ripper_4"], by_id["bottom_floor"]),
        (by_id["mid_platform_6"], by_id["mid_floor"]),
        (by_id["upper_ripper_4"], by_id["thin_seat"]),
    )
    for source, target in recovery:
        sx, sy = _center(source)
        tx, ty = _center(target)
        _dashed_line(draw, [(sx - 18, sy), (24, sy), (24, ty), (tx - 18, ty)], fill=RED, width=2, dash=8)

    draw.rounded_rectangle((690, 104, 1085, 290), 14, fill="#0b111a", outline=GREEN, width=2)
    draw.text((712, 122), "VERIFIED EDGE 01", font=label_font, fill=GREEN)
    lines = (
        "bottom_floor → lower_ripper_1",
        "live X track → Ice shot → runup",
        "consecutive WJ 20/4/8 + 14/2/6",
        "steer to frozen support; settle 8f",
        "2 exact natural runs; 31 patrol phases total",
        "230–414 policy frames; 408–636 FPS",
    )
    for line_index, line in enumerate(lines):
        draw.text((712, 154 + line_index * 21), line, font=small_font, fill=INK)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def render_first_edge(plan: dict, output: Path) -> Path:
    context = Image.open(CONTEXT_IMAGE).convert("RGB")
    canvas = Image.new("RGB", (1120, context.height), PANEL)
    canvas.paste(context, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(29, bold=True)
    label_font = _font(19, bold=True)
    body_font = _font(17)
    small_font = _font(14)

    # This is a visual context crop, not the coordinate-scale room diagram.
    start = (343, 700)
    fire = (205, 704)
    target = (238, 369)
    wall = (355, 120)
    _arrow(draw, [start, (285, 704), fire], fill=BLUE, width=6)
    _arrow(draw, [fire, (216, 560), target], fill=YELLOW, width=5)
    _dashed_line(draw, [fire, (345, 520), wall, (300, 25)], fill=GREEN, width=6, dash=14)
    _dashed_line(draw, [(260, 2), (145, 80), target], fill=GREEN, width=6, dash=14)
    for number, point, color in ((1, start, BLUE), (2, fire, YELLOW), (3, wall, GREEN), (4, target, GREEN)):
        x, y = point
        draw.ellipse((x - 17, y - 17, x + 17, y + 17), fill="#071018", outline=color, width=4)
        draw.text((x - 6, y - 12), str(number), font=label_font, fill=color)

    draw.rectangle((426, 0, 1119, 78), fill="#0a1019")
    draw.text((458, 14), "EDGE 01 — FLOOR TO FROZEN RIPPER", font=title_font, fill=INK)
    draw.text((460, 49), "Red Tower 0xA253 · Hi-Jump + Ice · human handoff after success", font=small_font, fill=GRAY)

    steps = (
        ("1", BLUE, "Natural entry", "Bottom floor ~(216,2443); keep clear of both doors."),
        ("2", YELLOW, "Phase-aware freeze", "Track the lowest Ripper's live X. Fire only in x=92..145."),
        ("3", GREEN, "Run + double WJ", "Right-wall spin, then 20/4/8 + 14/2/6 timing spans."),
        ("4", GREEN, "Stable checkpoint", "Land y=2351 on frozen support; require grounded + timer ≥30."),
    )
    y = 112
    for number, color, heading, body in steps:
        draw.ellipse((461, y, 493, y + 32), fill="#071018", outline=color, width=3)
        draw.text((472, y + 5), number, font=body_font, fill=color)
        draw.text((510, y), heading, font=label_font, fill=color)
        draw.text((510, y + 28), body, font=body_font, fill=INK)
        y += 104

    draw.rounded_rectangle((458, 548, 1082, 742), 14, fill=PANEL_2, outline=GREEN, width=2)
    draw.text((480, 568), "VERIFICATION / NON-CLAIM", font=label_font, fill=GREEN)
    report_lines = (
        "PASS  dual exact from post_ice_bat_to_red_pure",
        "PASS  31 patrol phases total (0..240f, step 8)",
        "PASS  230..414f at 408..636 FPS; lands x≈85..156",
        "OPEN  edge 02 (lower_ripper_1 → lower_ripper_2) is still planned",
        "OPEN  this PNG does not claim Red Tower → Hellway GREEN",
    )
    for line_index, line in enumerate(report_lines):
        fill = GREEN if line.startswith("PASS") else RED
        draw.text((482, 605 + line_index * 26), line, font=body_font, fill=fill)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-output", type=Path, default=FULL_OUTPUT)
    parser.add_argument("--edge-output", type=Path, default=EDGE_OUTPUT)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    print(render_full(plan, args.full_output))
    print(render_first_edge(plan, args.edge_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
