#!/usr/bin/env python3
"""Extract a short clip from a YouTube URL (yt-dlp + ffmpeg).

```bash
# 10s from t=2846 (47:26) — Alcatraz path reference
uv run python super_metroid/scripts/tools/yt_clip.py \\
  "https://youtu.be/2jGRpPCbzg4?t=2846" \\
  --start 2846 --duration 10 \\
  -o super_metroid/recordings/alcatraz_ref_t2846_10s.mp4

# Parse start from URL ?t= / &t=
uv run python super_metroid/scripts/tools/yt_clip.py \\
  "https://youtu.be/2jGRpPCbzg4?t=2846" --duration 10

# Also dump PNG keyframes every N seconds
uv run python super_metroid/scripts/tools/yt_clip.py URL --start 2846 -d 10 \\
  --keyframes 2 --keyframe-dir super_metroid/debug/spore/alcatraz_ref
```
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"missing dependency: {name} (install and retry)")
    return path


def parse_start_from_url(url: str) -> float | None:
    """Return seconds from ?t= / &t= / #t= (supports 1h2m3s and bare seconds)."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    raw = None
    if "t" in qs:
        raw = qs["t"][0]
    elif "start" in qs:
        raw = qs["start"][0]
    elif parsed.fragment.startswith("t="):
        raw = parsed.fragment[2:]
    if raw is None:
        return None
    raw = raw.rstrip("s")
    if raw.isdigit():
        return float(raw)
    # 1h2m3s / 2m3s / 90s
    m = re.fullmatch(
        r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s?)?",
        raw,
    )
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g or 0) for g in m.groups())
    return float(h * 3600 + mi * 60 + s)


def stem_from_url(url: str, start: float, duration: float) -> str:
    m = re.search(r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{6,})", url)
    vid = m.group(1) if m else "clip"
    return f"{vid}_t{int(start)}_{int(duration)}s"


def extract_clip(
    url: str,
    *,
    start: float,
    duration: float,
    out: Path,
    height: int = 720,
    crf: int = 18,
) -> Path:
    yt_dlp = _which("yt-dlp")
    ffmpeg = _which("ffmpeg")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    end = start + duration
    # Section download avoids fetching the whole VOD.
    tmp = out.with_suffix(".ytdl.mkv")
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--no-update",
        "-f",
        f"bv*[height<={height}]+ba/b[height<={height}]/b",
        "--download-sections",
        f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o",
        str(tmp),
        url,
    ]
    print(f"[yt_clip] download {start}s..{end}s → {tmp.name}", flush=True)
    subprocess.run(cmd, check=True)
    if not tmp.is_file():
        # yt-dlp may pick another extension
        cands = list(tmp.parent.glob(tmp.stem + ".*"))
        if not cands:
            raise SystemExit("yt-dlp produced no file")
        tmp = cands[0]
    print(f"[yt_clip] remux → {out}", flush=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return out


def dump_keyframes(video: Path, dest: Path, *, every_s: float = 2.0) -> list[Path]:
    ffmpeg = _which("ffmpeg")
    dest.mkdir(parents=True, exist_ok=True)
    pattern = dest / "kf_%02d.png"
    # fps=1/every_s selects one frame per every_s seconds
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps=1/{every_s}",
            "-q:v",
            "2",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    paths = sorted(dest.glob("kf_*.png"))
    print(f"[yt_clip] keyframes: {len(paths)} → {dest}", flush=True)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="YouTube URL (t= in query used if --start omitted)")
    ap.add_argument("--start", type=float, default=None, help="Start seconds")
    ap.add_argument("-d", "--duration", type=float, default=10.0, help="Clip length (s)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument(
        "--keyframes",
        type=float,
        default=0.0,
        help="If >0, dump a PNG every N seconds of the clip",
    )
    ap.add_argument("--keyframe-dir", type=Path, default=None)
    args = ap.parse_args()

    start = args.start
    if start is None:
        start = parse_start_from_url(args.url)
    if start is None:
        raise SystemExit("need --start or URL with ?t= / &t=")

    out = args.output
    if out is None:
        out = Path("super_metroid/recordings") / f"{stem_from_url(args.url, start, args.duration)}.mp4"

    path = extract_clip(
        args.url,
        start=start,
        duration=args.duration,
        out=out,
        height=args.height,
        crf=args.crf,
    )
    print(f"[yt_clip] wrote {path} ({args.duration}s from t={start})", flush=True)

    if args.keyframes and args.keyframes > 0:
        kf_dir = args.keyframe_dir or path.with_suffix("").parent / (path.stem + "_kf")
        dump_keyframes(path, kf_dir, every_s=args.keyframes)

    sys.exit(0)


if __name__ == "__main__":
    main()
