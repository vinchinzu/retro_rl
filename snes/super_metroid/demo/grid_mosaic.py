"""ffmpeg xstack mosaic + on-frame labels for the room-grid demo."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NTSC_FPS = 60
DEFAULT_SECONDS = 30
DEFAULT_COLS = 4
DEFAULT_ROWS = 4
CELL_W = 256
CELL_H = 224


def label_frame(obs: np.ndarray, text: str) -> np.ndarray:
    """Draw a 12px name bar on the top of one RGB frame."""

    rgb = np.asarray(obs, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bar_h = 12
    draw.rectangle((0, 0, image.width, bar_h), fill=(5, 10, 18))
    draw.text((3, 1), text, fill=(255, 214, 102), font=font)
    return np.asarray(image)


def xstack_filter(
    n: int,
    *,
    cols: int,
    rows: int,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    fps: int = NTSC_FPS,
) -> str:
    """Build the scale + xstack filter for ``n`` same-size inputs."""

    if n != cols * rows:
        raise ValueError(f"need {cols * rows} clips, got {n}")
    parts = [
        f"[{i}:v]scale={cell_w}:{cell_h}:flags=neighbor,setsar=1,fps={fps}[v{i}]"
        for i in range(n)
    ]
    stacked = "".join(f"[v{i}]" for i in range(n))
    parts.append(
        f"{stacked}xstack=inputs={n}:grid={cols}x{rows}:fill=black[out]"
    )
    return ";".join(parts)


def composite_grid(
    clips: Sequence[Path],
    output: Path,
    *,
    cols: int = DEFAULT_COLS,
    rows: int = DEFAULT_ROWS,
    seconds: float = DEFAULT_SECONDS,
    cell_w: int = CELL_W,
    cell_h: int = CELL_H,
    fps: int = NTSC_FPS,
    crf: int = 18,
) -> Path:
    """Loop/trim each clip to ``seconds`` and xstack into one silent mp4."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to composite the room grid")
    n = cols * rows
    paths = [Path(p) for p in clips]
    if len(paths) != n:
        raise ValueError(f"composite_grid expected {n} clips, got {len(paths)}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing tile clip: {path}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command: list[str] = [ffmpeg, "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(
            ["-stream_loop", "-1", "-t", f"{seconds:g}", "-i", str(path)]
        )
    command.extend(
        [
            "-filter_complex",
            xstack_filter(
                n, cols=cols, rows=rows, cell_w=cell_w, cell_h=cell_h, fps=fps
            ),
            "-map",
            "[out]",
            "-t",
            f"{seconds:g}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            str(output),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg xstack failed ({result.returncode}): {stderr or 'no stderr'}"
        )
    return output
