"""In-video RTA split panel (top-left) for SMB warps any% recordings.

Tracks exit-detect cum times (first frame of each post-exit stage, then axe)
and freezes both the panel and the speedrun clock on ``oper_mode=2`` (Axe).

Labels match the warps route: 1-1, 1-2, 4-1, 4-2, 8-1…8-4 / AXE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from smb.timing import NTSC_FPS, format_time_mmss

# Zero-indexed (world, level) → short HUD label for warps any%.
_WARP_STAGE_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "1-1",
    (0, 1): "1-2",
    (3, 0): "4-1",
    (3, 1): "4-2",
    (7, 0): "8-1",
    (7, 1): "8-2",
    (7, 2): "8-3",
    (7, 3): "8-4",
}

# Ordered expected stages for the warps showcase panel (fixed row order).
WARP_PANEL_ORDER: tuple[str, ...] = (
    "1-1",
    "1-2",
    "4-1",
    "4-2",
    "8-1",
    "8-2",
    "8-3",
    "8-4",
)

# Legal next labels on the warps route. Intermediate load flicker (1-3 after
# 1-2, 4-3 after 4-2, blackout garbage) is ignored so the panel only locks
# real route exits.
WARP_SUCCESSORS: dict[str, frozenset[str]] = {
    "1-1": frozenset({"1-2"}),
    "1-2": frozenset({"4-1"}),  # underground warp → World 4
    "4-1": frozenset({"4-2"}),
    "4-2": frozenset({"8-1"}),  # vine/top-clip warp → World 8
    "8-1": frozenset({"8-2"}),
    "8-2": frozenset({"8-3"}),
    "8-3": frozenset({"8-4"}),
    "8-4": frozenset(),  # ends on Axe, not a level change
}


def stage_label(world: int, level: int) -> str:
    """Human stage label; falls back to ``W-L`` for non-warp stages."""
    key = (int(world), int(level))
    if key in _WARP_STAGE_LABELS:
        return _WARP_STAGE_LABELS[key]
    return f"{int(world) + 1}-{int(level) + 1}"


def _format_hud_time(frames: int, fps: float) -> str:
    """Compact ``M:SS.cc`` (centiseconds) for the panel."""
    if frames < 0:
        frames = 0
    total_cs = int(round(frames * 100 / fps)) if fps > 0 else 0
    minutes, rem = divmod(total_cs, 60 * 100)
    secs, cs = divmod(rem, 100)
    if minutes >= 10:
        return f"{minutes:02d}:{secs:02d}.{cs:02d}"
    return f"{minutes}:{secs:02d}.{cs:02d}"


@dataclass
class RtaSplitTracker:
    """Live exit-detect splits for the recording HUD.

    Clock is caller-supplied (usually ``video.timer_frames``). On Axe
    (8-4 + ``oper_mode=2``) the tracker freezes: further ``observe`` calls
    leave completed splits and final time unchanged.
    """

    fps: float = NTSC_FPS
    panel_order: tuple[str, ...] = WARP_PANEL_ORDER
    completed: list[dict[str, Any]] = field(default_factory=list)
    current_label: str | None = None
    frozen: bool = False
    freeze_frame: int | None = None
    _seen_playing: bool = False
    _prev_key: tuple[int, int] | None = None

    def observe(self, snap: Any, *, clock_frames: int) -> bool:
        """Update from a post-step snapshot. Returns True if a split just locked."""
        if self.frozen:
            return False

        world = int(getattr(snap, "world"))
        level = int(getattr(snap, "level"))
        oper = int(getattr(snap, "oper_mode"))
        key = (world, level)

        # Axe / ending: lock final 8-4 split and freeze.
        if world == 7 and level == 3 and oper == 2:
            return self._freeze_ending(clock_frames=clock_frames)

        if oper != 1:
            return False

        # Ignore garbage pre-play / blackout until first real stage.
        if not self._seen_playing:
            if key not in _WARP_STAGE_LABELS and stage_label(world, level) not in self.panel_order:
                # Still allow non-warp full runs: any world 0–7 level 0–3.
                if not (0 <= world <= 7 and 0 <= level <= 3):
                    return False
            self._seen_playing = True
            self._prev_key = key
            self.current_label = stage_label(world, level)
            return False

        assert self._prev_key is not None
        if key == self._prev_key:
            return False

        # Level change → previous stage exit only if successor is on-route.
        # Skip load-screen flicker (e.g. 1-2→1-3 before warp settles on 4-1).
        prev_label = stage_label(*self._prev_key)
        new_label = stage_label(world, level)
        allowed = WARP_SUCCESSORS.get(prev_label)
        if allowed is not None and new_label not in allowed:
            return False

        self._lock_split(prev_label, clock_frames=clock_frames, kind="exit")
        self._prev_key = key
        self.current_label = new_label
        return True

    def _freeze_ending(self, *, clock_frames: int) -> bool:
        label = self.current_label or "8-4"
        # Prefer 8-4 as the final segment name even if current was blank.
        if self._prev_key == (7, 3) or label == "8-4":
            label = "8-4"
        already = {row["label"] for row in self.completed}
        if label not in already:
            self._lock_split(label, clock_frames=clock_frames, kind="axe")
        self.frozen = True
        self.freeze_frame = int(clock_frames)
        self.current_label = None
        return True

    def _lock_split(self, label: str, *, clock_frames: int, kind: str) -> None:
        prev_cum = int(self.completed[-1]["cum_frames"]) if self.completed else 0
        self.completed.append(
            {
                "label": label,
                "cum_frames": int(clock_frames),
                "seg_frames": int(clock_frames) - prev_cum,
                "cum_time": format_time_mmss(int(clock_frames), self.fps),
                "seg_time": format_time_mmss(int(clock_frames) - prev_cum, self.fps),
                "kind": kind,
            }
        )

    def lines(
        self,
        *,
        clock_frames: int,
        max_rows: int | None = None,
    ) -> list[str]:
        """Text rows for the overlay (header + stage rows + total)."""
        done = {row["label"]: row for row in self.completed}
        order = list(self.panel_order)
        # Include any unexpected completed labels after known order.
        for row in self.completed:
            if row["label"] not in order:
                order.append(row["label"])

        rows: list[str] = ["RTA"]
        for label in order:
            if label in done:
                t = _format_hud_time(int(done[label]["cum_frames"]), self.fps)
                mark = " *" if done[label].get("kind") == "axe" else ""
                rows.append(f"{label}  {t}{mark}")
            elif self.current_label == label and not self.frozen:
                t = _format_hud_time(int(clock_frames), self.fps)
                rows.append(f"{label}  {t}  ")
            else:
                rows.append(f"{label}  --:--.--")

        display_clock = (
            int(self.freeze_frame)
            if self.frozen and self.freeze_frame is not None
            else int(clock_frames)
        )
        total = _format_hud_time(display_clock, self.fps)
        if self.frozen:
            rows.append(f"AXE {total}")
        else:
            rows.append(f"TOT {total}")

        if max_rows is not None and max_rows > 0:
            # Keep header + last (max_rows-1) body lines.
            if len(rows) > max_rows:
                rows = [rows[0]] + rows[-(max_rows - 1) :]
        return rows

    def report(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "freeze_frame": self.freeze_frame,
            "fps": self.fps,
            "completed": list(self.completed),
            "current_label": self.current_label,
        }


def draw_rta_split_panel(
    obs: np.ndarray,
    lines: Sequence[str],
    *,
    x: int = 2,
    y: int = 2,
    pad: int = 2,
    bg: tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.55,
    fg: tuple[int, int, int] = (240, 248, 255),
    header_fg: tuple[int, int, int] = (103, 232, 164),
    axe_fg: tuple[int, int, int] = (255, 214, 102),
    current_fg: tuple[int, int, int] = (255, 255, 180),
) -> np.ndarray:
    """Composite a semi-transparent top-left text panel onto ``obs`` (RGB)."""
    if not lines:
        return np.asarray(obs, dtype=np.uint8)

    rgb = np.asarray(obs, dtype=np.uint8).copy()
    h, w = rgb.shape[:2]
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=8)

    line_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_w = max((b[2] - b[0] for b in line_sizes), default=0)
    line_h = max((b[3] - b[1] for b in line_sizes), default=8) + 1
    box_w = text_w + pad * 2
    box_h = line_h * len(lines) + pad * 2
    x1 = max(0, min(x, w - 1))
    y1 = max(0, min(y, h - 1))
    x2 = max(x1 + 1, min(x1 + box_w, w))
    y2 = max(y1 + 1, min(y1 + box_h, h))

    alpha = int(round(255 * max(0.0, min(1.0, bg_alpha))))
    draw.rectangle((x1, y1, x2, y2), fill=(*bg, alpha))

    cy = y1 + pad
    for i, line in enumerate(lines):
        if i == 0:
            color = header_fg
        elif line.startswith("AXE"):
            color = axe_fg
        elif line.rstrip().endswith("  ") and "--:--.--" not in line and "*" not in line:
            # Live current row (trailing spaces marker from lines()).
            color = current_fg
        else:
            color = fg
        draw.text((x1 + pad, cy), line.rstrip(), fill=color, font=font)
        cy += line_h

    return np.asarray(image.convert("RGB"), dtype=np.uint8)
