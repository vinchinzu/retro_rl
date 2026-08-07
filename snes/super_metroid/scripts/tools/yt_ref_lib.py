"""Shared helpers for Super Metroid YouTube reference workspaces.

Gitignored tree::

    snes/super_metroid/refs/yt_reference/<video_id>/
      video/<id>.mp4
      layout.json
      segments/*.json
      frames/
      inputs/
      chunks/<name>/

Tracked CLIs live under ``scripts/tools/yt_ref.py`` (and thin wrappers).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from super_metroid.paths import YT_DEFAULT_REF_ID, YT_REFERENCE_DIR
except ImportError:  # script path without package context
    _GAME = Path(__file__).resolve().parents[2]
    YT_REFERENCE_DIR = _GAME / "refs" / "yt_reference"
    YT_DEFAULT_REF_ID = "TFsGVxQReMw"

DEFAULT_REF_ID = YT_DEFAULT_REF_ID
DEFAULT_URL = f"https://youtu.be/{DEFAULT_REF_ID}"

# Kentroid KPDR stream layout (1920x1080) — Input Display faces, game right.
KENTROID_LAYOUT_TEMPLATE: dict = {
    "video_id": DEFAULT_REF_ID,
    "source_resolution": [1920, 1080],
    "fps": 60.0,
    "notes": (
        "Kentroid KPDR layout: left LiveSplit + Input Display + webcam; "
        "right = game. Button faces light green (most) or orange/red (A). "
        "Sample faces only (not history bars above)."
    ),
    "vod_to_igt_offset_s": 14.0,
    "regions": {
        "game_monitor": {"box": [545, 16, 1359, 1048]},
        "livesplit": {"box": [16, 16, 512, 460]},
        "timer": {"box": [40, 16, 280, 50]},
        "controller": {
            "box": [40, 600, 490, 120],
            "style": "input_display_snes_faces",
        },
        "webcam": {"box": [16, 720, 512, 340]},
    },
    "buttons": {
        "order": [
            "Left",
            "Up",
            "Right",
            "Down",
            "A",
            "B",
            "X",
            "Y",
            "L",
            "R",
            "Select",
            "Start",
        ],
        "retro_harness_order": [
            "B",
            "Y",
            "Select",
            "Start",
            "Up",
            "Down",
            "Left",
            "Right",
            "A",
            "X",
            "L",
            "R",
        ],
        "face_row": {
            "cy": 652,
            "half_h": 8,
            "half_w": 10,
            "x0": 44.0,
            "spacing": 41.5,
            "lit_green_score": 40.0,
            "lit_red_score": 40.0,
        },
        "hotspots": {
            "Left": {"cx": 44, "cy": 652, "w": 20, "h": 16},
            "Up": {"cx": 86, "cy": 652, "w": 20, "h": 16},
            "Right": {"cx": 127, "cy": 652, "w": 20, "h": 16},
            "Down": {"cx": 168, "cy": 652, "w": 20, "h": 16},
            "A": {"cx": 210, "cy": 652, "w": 20, "h": 16},
            "B": {"cx": 252, "cy": 652, "w": 20, "h": 16},
            "X": {"cx": 293, "cy": 652, "w": 20, "h": 16},
            "Y": {"cx": 334, "cy": 652, "w": 20, "h": 16},
            "L": {"cx": 376, "cy": 652, "w": 20, "h": 16},
            "R": {"cx": 418, "cy": 652, "w": 20, "h": 16},
            "Select": {"cx": 459, "cy": 652, "w": 20, "h": 16},
            "Start": {"cx": 500, "cy": 652, "w": 20, "h": 16},
        },
    },
    "probe_frames": [],
}


def which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"missing dependency: {name}")
    return path


def video_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if "youtu.be" in (parsed.netloc or ""):
        vid = parsed.path.strip("/").split("/")[0]
        if vid:
            return vid
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        return qs["v"][0]
    m = re.search(r"(?:youtu\.be/|v=|shorts/)([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    raise SystemExit(f"cannot parse YouTube video id from: {url}")


def parse_time_token(raw: str | float | int) -> float:
    """Parse seconds from float/int or H:MM:SS / M:SS / 1h2m3s / bare seconds."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().rstrip("s")
    if not s:
        raise ValueError("empty time")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?", s)
    if m and any(m.groups()):
        h, mi, sec = m.group(1), m.group(2), m.group(3)
        return float(h or 0) * 3600 + float(mi or 0) * 60 + float(sec or 0)
    parts = s.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    raise ValueError(f"unrecognized time: {raw!r}")


@dataclass(frozen=True)
class RefWorkspace:
    """Paths for one VOD reference id."""

    video_id: str
    root: Path

    @classmethod
    def resolve(cls, ref: str | None = None) -> RefWorkspace:
        """``ref`` may be video id, URL, or path under yt_reference/."""
        if ref is None or ref == "":
            vid = DEFAULT_REF_ID
        elif Path(ref).is_dir() and (Path(ref) / "layout.json").exists():
            root = Path(ref).resolve()
            return cls(video_id=root.name, root=root)
        elif "youtube" in ref or "youtu.be" in ref or ref.startswith("http"):
            vid = video_id_from_url(ref)
        else:
            vid = ref.strip().strip("/")
        root = YT_REFERENCE_DIR / vid
        return cls(video_id=vid, root=root)

    @property
    def video_dir(self) -> Path:
        return self.root / "video"

    @property
    def video_path(self) -> Path:
        for ext in (".mp4", ".mkv", ".webm"):
            p = self.video_dir / f"{self.video_id}{ext}"
            if p.is_file():
                return p
        # any large media in video/
        if self.video_dir.is_dir():
            cands = sorted(
                [
                    p
                    for p in self.video_dir.iterdir()
                    if p.suffix.lower() in {".mp4", ".mkv", ".webm"} and p.stat().st_size > 1_000_000
                ],
                key=lambda p: p.stat().st_size,
                reverse=True,
            )
            if cands:
                return cands[0]
        return self.video_dir / f"{self.video_id}.mp4"

    @property
    def layout_path(self) -> Path:
        return self.root / "layout.json"

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def chunks_dir(self) -> Path:
        return self.root / "chunks"

    def ensure_dirs(self) -> None:
        for d in (
            self.root,
            self.video_dir,
            self.segments_dir,
            self.frames_dir,
            self.inputs_dir,
            self.chunks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def load_layout(self) -> dict:
        if not self.layout_path.is_file():
            raise SystemExit(
                f"missing layout: {self.layout_path}\n"
                f"  Run: uv run python snes/super_metroid/scripts/tools/yt_ref.py fetch "
                f"--ref {self.video_id} --template-layout"
            )
        return json.loads(self.layout_path.read_text())

    def default_segments_path(self) -> Path:
        preferred = self.segments_dir / "kpdr_paths.json"
        if preferred.is_file():
            return preferred
        any_json = sorted(self.segments_dir.glob("*.json"))
        if any_json:
            return any_json[0]
        return preferred

    def load_segments(self, path: Path | None = None) -> dict:
        p = path or self.default_segments_path()
        if not p.is_file():
            raise SystemExit(f"missing segments file: {p}")
        return json.loads(p.read_text())

    def resolve_segment(
        self, segment_id: str, segments_path: Path | None = None
    ) -> tuple[float, float, dict]:
        data = self.load_segments(segments_path)
        for seg in data.get("segments", []):
            if seg.get("id") == segment_id:
                s, e = seg.get("vod_start_s"), seg.get("vod_end_s")
                if s is None or e is None:
                    raise SystemExit(f"segment {segment_id!r} missing vod_start_s/vod_end_s")
                return float(s), float(e), seg
        known = [s.get("id") for s in data.get("segments", [])]
        raise SystemExit(f"segment id not found: {segment_id!r} (known: {known})")

    def status(self) -> dict:
        vid = self.video_path if self.video_path.is_file() else None
        return {
            "video_id": self.video_id,
            "root": str(self.root),
            "video": str(vid) if vid else None,
            "video_mb": round(vid.stat().st_size / 1e6, 1) if vid else None,
            "layout": self.layout_path.is_file(),
            "segments": sorted(p.name for p in self.segments_dir.glob("*.json"))
            if self.segments_dir.is_dir()
            else [],
            "chunks": sorted(p.name for p in self.chunks_dir.iterdir() if p.is_dir())
            if self.chunks_dir.is_dir()
            else [],
        }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def button_centers(layout: dict) -> list[tuple[str, int, int]]:
    face = layout["buttons"]["face_row"]
    order = layout["buttons"]["order"]
    x0 = float(face["x0"])
    sp = float(face["spacing"])
    cy = int(face["cy"])
    return [(name, int(round(x0 + i * sp)), cy) for i, name in enumerate(order)]


def is_button_lit(
    frame,
    cx: int,
    cy: int,
    *,
    half_w: int,
    half_h: int,
    g_thresh: float,
    r_thresh: float,
) -> tuple[bool, float, float]:
    import numpy as np

    y1, y2 = cy - half_h, cy + half_h
    x1, x2 = cx - half_w, cx + half_w
    h, w = frame.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    patch = frame[y1:y2, x1:x2].astype(np.float32)
    if patch.size == 0:
        return False, 0.0, 0.0
    mean = patch.mean(axis=(0, 1))
    g = float(mean[1] - max(mean[0], mean[2]))
    r = float(mean[2] - max(mean[0], mean[1]))
    return (g > g_thresh or r > r_thresh), g, r


def harness_vector(lit: dict[str, bool], harness_order: list[str]) -> list[int]:
    return [1 if lit.get(name, False) else 0 for name in harness_order]


def crop_box(frame, box: list[int] | tuple[int, ...]):
    """box = [x, y, w, h]"""
    x, y, w, h = (int(v) for v in box)
    return frame[y : y + h, x : x + w]


def game_box(layout: dict) -> list[int] | None:
    reg = layout.get("regions", {}).get("game_monitor") or {}
    return reg.get("box")


def gold_spark_score(bgr_region) -> float:
    """Fraction of gold-ish pixels (shinespark afterimages)."""
    import numpy as np

    if bgr_region is None or bgr_region.size == 0:
        return 0.0
    img = bgr_region.astype(np.float32)
    gold = (img[:, :, 2] > 180) & (img[:, :, 1] > 150) & (img[:, :, 0] < 120)
    return float(gold.mean())


def hold_intervals(press_events: list[dict]) -> list[dict]:
    """Convert edge list into hold intervals {button, start_s, end_s, dur_s}."""
    down_t: dict[str, float] = {}
    holds: list[dict] = []
    for e in press_events:
        b, edge, t = e["button"], e["edge"], float(e["vod_s"])
        if edge == "down":
            down_t[b] = t
        elif edge == "up" and b in down_t:
            s = down_t.pop(b)
            holds.append(
                {
                    "button": b,
                    "start_s": round(s, 4),
                    "end_s": round(t, 4),
                    "dur_s": round(t - s, 4),
                }
            )
    return holds


def duty_cycle(frames: list[dict], button_order: list[str]) -> dict[str, float]:
    if not frames:
        return {}
    counts = {n: 0 for n in button_order}
    for rec in frames:
        for n in rec.get("lit") or []:
            counts[n] = counts.get(n, 0) + 1
    n = len(frames)
    return {k: round(v / n, 3) for k, v in counts.items() if v}


def fetch_video(
    ws: RefWorkspace,
    url: str | None = None,
    *,
    template_layout: bool = True,
    skip_download: bool = False,
) -> Path:
    """Download best quality into workspace; scaffold dirs + layout."""
    ws.ensure_dirs()
    url = url or f"https://youtu.be/{ws.video_id}"
    if template_layout and not ws.layout_path.is_file():
        layout = dict(KENTROID_LAYOUT_TEMPLATE)
        layout["video_id"] = ws.video_id
        write_json(ws.layout_path, layout)
        print(f"[yt_ref] wrote template layout → {ws.layout_path}", flush=True)
    elif not ws.layout_path.is_file():
        write_json(ws.layout_path, {"video_id": ws.video_id, "buttons": {}, "regions": {}})
        print(f"[yt_ref] wrote empty layout stub → {ws.layout_path}", flush=True)

    readme = ws.root / "README.md"
    if not readme.is_file():
        readme.write_text(
            f"# YT reference `{ws.video_id}`\n\n"
            f"URL: {url}\n\n"
            "Gitignored workspace. Tools: "
            "`uv run python snes/super_metroid/scripts/tools/yt_ref.py --help`\n"
        )

    if skip_download:
        return ws.video_path

    yt_dlp = which("yt-dlp")
    out_tmpl = str(ws.video_dir / f"{ws.video_id}.%(ext)s")
    cmd = [
        yt_dlp,
        "--no-playlist",
        "--no-update",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--write-info-json",
        "--write-description",
        "-o",
        out_tmpl,
        url,
    ]
    print(f"[yt_ref] fetch {url} → {ws.video_dir}", flush=True)
    subprocess.run(cmd, check=True)
    if not ws.video_path.is_file():
        raise SystemExit("download finished but video file not found")
    print(f"[yt_ref] video → {ws.video_path} ({ws.video_path.stat().st_size / 1e9:.2f} GiB)", flush=True)
    return ws.video_path


def extract_buttons(
    video: Path,
    layout: dict,
    *,
    start_s: float,
    end_s: float,
    stride: int = 1,
) -> dict:
    """Frame-sample Input Display faces → samples + edges."""
    import cv2

    face = layout["buttons"]["face_row"]
    half_w = int(face["half_w"])
    half_h = int(face["half_h"])
    g_thresh = float(face["lit_green_score"])
    r_thresh = float(face["lit_red_score"])
    centers = button_centers(layout)
    harness_order = layout["buttons"].get(
        "retro_harness_order",
        ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
    )
    fps_hint = float(layout.get("fps") or 60.0)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    v_fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    start_f = int(start_s * v_fps)
    end_f = int(end_s * v_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    frames_out: list[dict] = []
    press_events: list[dict] = []
    prev_lit: dict[str, bool] | None = None
    fidx = start_f
    while fidx < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        if (fidx - start_f) % stride != 0:
            fidx += 1
            continue
        lit_map: dict[str, bool] = {}
        for name, cx, cy in centers:
            lit, _g, _r = is_button_lit(
                frame,
                cx,
                cy,
                half_w=half_w,
                half_h=half_h,
                g_thresh=g_thresh,
                r_thresh=r_thresh,
            )
            lit_map[name] = lit
        t = fidx / v_fps
        rec = {
            "frame": fidx,
            "vod_s": round(t, 4),
            "lit": [n for n, v in lit_map.items() if v],
            "harness": harness_vector(lit_map, harness_order),
        }
        frames_out.append(rec)
        if prev_lit is not None:
            for name, now in lit_map.items():
                if now and not prev_lit.get(name, False):
                    press_events.append(
                        {"vod_s": round(t, 4), "frame": fidx, "button": name, "edge": "down"}
                    )
                elif not now and prev_lit.get(name, False):
                    press_events.append(
                        {"vod_s": round(t, 4), "frame": fidx, "button": name, "edge": "up"}
                    )
        prev_lit = lit_map
        fidx += 1
    cap.release()

    return {
        "video": str(video),
        "start_s": start_s,
        "end_s": end_s,
        "fps": v_fps,
        "stride": stride,
        "button_order": [n for n, _, _ in centers],
        "harness_order": harness_order,
        "n_samples": len(frames_out),
        "press_events": press_events,
        "frames": frames_out,
        "notes": "Rough Input Display sample — not SNES-frame TAS quality.",
    }


def write_extract_outputs(result: dict, out_stem: Path) -> dict[str, Path]:
    import csv

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_stem.with_suffix(".json") if out_stem.suffix != ".json" else out_stem
    write_json(json_path, result)
    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "vod_s", "buttons", *result["harness_order"]])
        for rec in result["frames"]:
            w.writerow(
                [
                    rec["frame"],
                    rec["vod_s"],
                    "+".join(rec["lit"]) if rec["lit"] else "",
                    *rec["harness"],
                ]
            )
    edges_path = json_path.with_name(json_path.stem + "_edges.json")
    write_json(
        edges_path,
        {
            "start_s": result["start_s"],
            "end_s": result["end_s"],
            "n_events": len(result["press_events"]),
            "press_events": result["press_events"],
        },
    )
    return {"json": json_path, "csv": csv_path, "edges": edges_path}


def dump_frames(
    video: Path,
    dest: Path,
    *,
    start_s: float,
    end_s: float,
    every_s: float = 1.0,
    layout: dict | None = None,
    game_only: bool = True,
    prefix: str = "t",
) -> list[Path]:
    import cv2

    dest.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    box = game_box(layout) if (layout and game_only) else None
    paths: list[Path] = []
    t = start_s
    while t < end_s - 1e-9:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        if box:
            frame = crop_box(frame, box)
        # filename: t01338p50.jpg for 1338.5
        stem = f"{prefix}{int(t):05d}p{int(round((t % 1) * 100)):02d}"
        path = dest / f"{stem}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
        t += every_s
    cap.release()
    return paths


def scan_spark(
    video: Path,
    layout: dict,
    *,
    start_s: float,
    end_s: float,
    step_s: float = 0.1,
    threshold: float = 0.01,
) -> list[dict]:
    """Sample gold afterimage score over a window (shinespark detector)."""
    import cv2

    box = game_box(layout)
    if not box:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    hits: list[dict] = []
    t = start_s
    while t < end_s - 1e-9:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        region = crop_box(frame, box)
        # center band — avoid HUD
        h = region.shape[0]
        mid = region[h // 5 : 4 * h // 5]
        score = gold_spark_score(mid)
        if score >= threshold:
            hits.append({"vod_s": round(t, 3), "gold_frac": round(score, 5)})
        t += step_s
    cap.release()
    return hits


def analyze_extract(
    result: dict,
    *,
    spark_hits: list[dict] | None = None,
    min_hold_s: float = 0.1,
) -> dict:
    holds = hold_intervals(result.get("press_events") or [])
    long_holds = [h for h in holds if h["dur_s"] >= min_hold_s]
    duty = duty_cycle(result.get("frames") or [], result.get("button_order") or [])
    out: dict = {
        "start_s": result.get("start_s"),
        "end_s": result.get("end_s"),
        "n_samples": result.get("n_samples"),
        "n_events": len(result.get("press_events") or []),
        "duty_cycle": duty,
        "holds": long_holds,
        "holds_all": holds,
        "notes": (
            "Rough timing from Input Display. Not SNES-frame perfect. "
            "Review before pure controllers."
        ),
    }
    if spark_hits is not None:
        out["spark_hits"] = spark_hits
        if spark_hits:
            out["spark_window_s"] = [
                spark_hits[0]["vod_s"],
                spark_hits[-1]["vod_s"],
            ]
    # Relative to window start for easy controller seeds
    t0 = float(result.get("start_s") or 0.0)
    out["holds_rel"] = [
        {
            **h,
            "start_rel_s": round(h["start_s"] - t0, 4),
            "end_rel_s": round(h["end_s"] - t0, 4),
        }
        for h in long_holds
    ]
    return out


