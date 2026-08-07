#!/usr/bin/env python3
"""YouTube KPDR reference workspace tools (gitignored under refs/yt_reference/).

Default VOD: Kentroid KPDR ``TFsGVxQReMw`` (omit ``--ref``).

```bash
# Status / list workspaces
uv run python snes/super_metroid/scripts/tools/yt_ref.py list
uv run python snes/super_metroid/scripts/tools/yt_ref.py status

# Fetch (or re-scaffold) the default VOD
uv run python snes/super_metroid/scripts/tools/yt_ref.py fetch
uv run python snes/super_metroid/scripts/tools/yt_ref.py fetch --skip-download  # dirs+layout only

# Chunk workflow: buttons + frames + hold/spark analysis → chunks/<name>/
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \\
  --start 1338 --end 1351 --name moat_shinespark --spark

# Named segment from segments/kpdr_paths.json
uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \\
  --segment-id k2_spazer --name k2_spazer --every 2 --stride 2

# Pieces (same defaults)
uv run python snes/super_metroid/scripts/tools/yt_ref.py extract --start 200 --end 210 -o smoke
uv run python snes/super_metroid/scripts/tools/yt_ref.py frames --start 1338 --end 1351 --every 0.5
uv run python snes/super_metroid/scripts/tools/yt_ref.py analyze \\
  --inputs-stem snes/super_metroid/refs/yt_reference/TFsGVxQReMw/inputs/kihunter_moat_probe
```

Chunk outputs (under ``refs/yt_reference/<id>/chunks/<name>/``)::

    buttons.json  buttons.csv  buttons_edges.json
    analyze.json  frames/*.jpg  meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
_TOOLS = Path(__file__).resolve().parent
for _p in (ROOT, _SNES, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import YT_DEFAULT_REF_ID, YT_REFERENCE_DIR  # noqa: E402
import yt_ref_lib as lib  # noqa: E402


def _add_ref_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--ref",
        default=YT_DEFAULT_REF_ID,
        help=f"Video id, URL, or workspace path (default: {YT_DEFAULT_REF_ID})",
    )


def _resolve_window(ws: lib.RefWorkspace, args: argparse.Namespace) -> tuple[float, float, dict | None]:
    if getattr(args, "segment_id", None):
        segs = Path(args.segments) if getattr(args, "segments", None) else None
        return ws.resolve_segment(args.segment_id, segs)
    if args.start is None or args.end is None:
        raise SystemExit("need --start/--end or --segment-id")
    start = lib.parse_time_token(args.start)
    end = lib.parse_time_token(args.end)
    if end <= start:
        raise SystemExit("end must be > start")
    return start, end, None


def cmd_list(_args: argparse.Namespace) -> int:
    YT_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    roots = sorted(p for p in YT_REFERENCE_DIR.iterdir() if p.is_dir()) if YT_REFERENCE_DIR.is_dir() else []
    if not roots:
        print(f"(empty) {YT_REFERENCE_DIR}")
        return 0
    for root in roots:
        ws = lib.RefWorkspace(video_id=root.name, root=root)
        st = ws.status()
        mark = "*" if st["video_id"] == YT_DEFAULT_REF_ID else " "
        vid = f"{st['video_mb']}MB" if st["video_mb"] else "no-video"
        print(
            f"{mark} {st['video_id']:14s}  layout={st['layout']}  "
            f"{vid:10s}  segs={len(st['segments'])}  chunks={len(st['chunks'])}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ws = lib.RefWorkspace.resolve(args.ref)
    st = ws.status()
    print(json.dumps(st, indent=2))
    if st.get("segments"):
        try:
            data = ws.load_segments()
            print("\nsegments:")
            for seg in data.get("segments", []):
                s, e = seg.get("vod_start_s"), seg.get("vod_end_s")
                flag = "ok" if s is not None and e is not None else "unmarked"
                print(f"  [{flag:8s}] {seg.get('id'):28s}  {s} → {e}  {seg.get('label', '')}")
        except SystemExit:
            pass
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    url = args.url
    if url:
        vid = lib.video_id_from_url(url)
        ws = lib.RefWorkspace.resolve(vid)
    else:
        ws = lib.RefWorkspace.resolve(args.ref)
        url = f"https://youtu.be/{ws.video_id}"
    lib.fetch_video(
        ws,
        url,
        template_layout=not args.no_template_layout,
        skip_download=args.skip_download,
    )
    print(json.dumps(ws.status(), indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    ws = lib.RefWorkspace.resolve(args.ref)
    layout = ws.load_layout()
    start, end, seg = _resolve_window(ws, args)
    video = ws.video_path
    if not video.is_file():
        raise SystemExit(f"missing video: {video} (run yt_ref.py fetch)")
    result = lib.extract_buttons(video, layout, start_s=start, end_s=end, stride=max(1, args.stride))
    if seg:
        result["segment"] = {
            "id": seg.get("id"),
            "label": seg.get("label"),
            "project_tip": seg.get("project_tip"),
        }
    if args.out:
        out = Path(args.out)
        if not out.is_absolute() and out.parent == Path("."):
            out = ws.inputs_dir / out.name
    else:
        name = args.segment_id or f"t{int(start)}_{int(end)}"
        out = ws.inputs_dir / name
    paths = lib.write_extract_outputs(result, out)
    duty = lib.duty_cycle(result["frames"], result["button_order"])
    print(f"[yt_ref] extract samples={result['n_samples']} edges={len(result['press_events'])}", flush=True)
    print(f"[yt_ref] → {paths['json']}", flush=True)
    print(f"[yt_ref] duty_cycle={duty}", flush=True)
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    ws = lib.RefWorkspace.resolve(args.ref)
    layout = ws.load_layout() if not args.full_frame else None
    start, end, _seg = _resolve_window(ws, args)
    video = ws.video_path
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")
    dest = Path(args.out) if args.out else ws.frames_dir / f"t{int(start)}_{int(end)}"
    paths = lib.dump_frames(
        video,
        dest,
        start_s=start,
        end_s=end,
        every_s=args.every,
        layout=layout if layout else ws.load_layout(),
        game_only=not args.full_frame,
    )
    print(f"[yt_ref] frames={len(paths)} every={args.every}s → {dest}", flush=True)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    ws = lib.RefWorkspace.resolve(args.ref)
    stem = Path(args.inputs_stem)
    if not stem.suffix:
        json_path = stem.with_suffix(".json") if stem.exists() or stem.parent != Path(".") else ws.inputs_dir / f"{stem.name}.json"
        if not json_path.is_file() and (ws.inputs_dir / f"{stem.name}.json").is_file():
            json_path = ws.inputs_dir / f"{stem.name}.json"
    else:
        json_path = stem
    if not json_path.is_file():
        # try edges companion
        alt = ws.inputs_dir / Path(args.inputs_stem).name
        if not str(alt).endswith(".json"):
            alt = alt.with_suffix(".json")
        if alt.is_file():
            json_path = alt
    if not json_path.is_file():
        raise SystemExit(f"missing extract json: {json_path}")
    result = json.loads(json_path.read_text())
    spark_hits = None
    if args.spark:
        layout = ws.load_layout()
        video = ws.video_path
        spark_hits = lib.scan_spark(
            video,
            layout,
            start_s=float(result["start_s"]),
            end_s=float(result["end_s"]),
            step_s=args.spark_step,
        )
    analysis = lib.analyze_extract(result, spark_hits=spark_hits, min_hold_s=args.min_hold)
    out = Path(args.out) if args.out else json_path.with_name(json_path.stem + "_analyze.json")
    lib.write_json(out, analysis)
    print(f"[yt_ref] analyze → {out}", flush=True)
    print(f"[yt_ref] duty={analysis['duty_cycle']}", flush=True)
    print(f"[yt_ref] long_holds (≥{args.min_hold}s): {len(analysis['holds'])}", flush=True)
    for h in analysis["holds"][:20]:
        print(f"  {h['button']:8s}  {h['start_s']:8.3f} → {h['end_s']:8.3f}  ({h['dur_s']:.3f}s)")
    if analysis.get("spark_window_s"):
        print(f"[yt_ref] spark_window={analysis['spark_window_s']} hits={len(analysis['spark_hits'])}", flush=True)
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """One-shot: extract + frames + analyze into chunks/<name>/."""
    ws = lib.RefWorkspace.resolve(args.ref)
    layout = ws.load_layout()
    start, end, seg = _resolve_window(ws, args)
    video = ws.video_path
    if not video.is_file():
        raise SystemExit(f"missing video: {video} (run yt_ref.py fetch)")

    name = args.name
    if not name:
        if seg:
            name = str(seg.get("id"))
        else:
            name = f"t{int(start)}_{int(end)}"
    # sanitize
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    chunk_dir = ws.chunks_dir / name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    print(f"[yt_ref] chunk {name!r}  vod {start} → {end}  → {chunk_dir}", flush=True)

    result = lib.extract_buttons(
        video, layout, start_s=start, end_s=end, stride=max(1, args.stride)
    )
    if seg:
        result["segment"] = {
            "id": seg.get("id"),
            "label": seg.get("label"),
            "project_tip": seg.get("project_tip"),
        }
    paths = lib.write_extract_outputs(result, chunk_dir / "buttons")

    frame_paths = lib.dump_frames(
        video,
        chunk_dir / "frames",
        start_s=start,
        end_s=end,
        every_s=args.every,
        layout=layout,
        game_only=not args.full_frame,
    )

    spark_hits = None
    if args.spark:
        spark_hits = lib.scan_spark(
            video,
            layout,
            start_s=start,
            end_s=end,
            step_s=args.spark_step,
        )
    analysis = lib.analyze_extract(result, spark_hits=spark_hits, min_hold_s=args.min_hold)
    lib.write_json(chunk_dir / "analyze.json", analysis)

    meta = {
        "name": name,
        "video_id": ws.video_id,
        "vod_start_s": start,
        "vod_end_s": end,
        "duration_s": round(end - start, 3),
        "segment": result.get("segment"),
        "stride": args.stride,
        "frame_every_s": args.every,
        "n_samples": result["n_samples"],
        "n_events": len(result["press_events"]),
        "n_frames": len(frame_paths),
        "duty_cycle": analysis["duty_cycle"],
        "spark_window_s": analysis.get("spark_window_s"),
        "paths": {
            "buttons": str(paths["json"]),
            "edges": str(paths["edges"]),
            "analyze": str(chunk_dir / "analyze.json"),
            "frames": str(chunk_dir / "frames"),
        },
        "quality_note": (
            "Rough Input Display timing (~60 Hz). Good for phase seeds; "
            "not SNES-frame TAS. Human-review before pure controllers."
        ),
    }
    lib.write_json(chunk_dir / "meta.json", meta)

    # Human-readable one-pager
    lines = [
        f"# chunk `{name}`",
        "",
        f"- VOD: `{ws.video_id}` `{start}` → `{end}` ({end - start:.1f}s)",
        f"- samples: {result['n_samples']}  edges: {len(result['press_events'])}  frames: {len(frame_paths)}",
        f"- duty: `{analysis['duty_cycle']}`",
    ]
    if analysis.get("spark_window_s"):
        lines.append(f"- spark gold window: {analysis['spark_window_s']}")
    lines += ["", "## Long holds", ""]
    for h in analysis["holds"][:40]:
        lines.append(
            f"- **{h['button']}** `{h['start_s']:.3f}` → `{h['end_s']:.3f}` "
            f"({h['dur_s']:.3f}s, rel {h['start_s'] - start:+.3f})"
        )
    lines += ["", "## Edge timeline", ""]
    for e in result["press_events"][:80]:
        lines.append(f"- `{e['vod_s']:.3f}` {e['edge']:4s} {e['button']}")
    if len(result["press_events"]) > 80:
        lines.append(f"- … +{len(result['press_events']) - 80} more (see buttons_edges.json)")
    (chunk_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(f"[yt_ref] duty={analysis['duty_cycle']}", flush=True)
    if analysis.get("spark_window_s"):
        print(f"[yt_ref] spark_window={analysis['spark_window_s']}", flush=True)
    print(f"[yt_ref] wrote {chunk_dir}/meta.json + SUMMARY.md", flush=True)
    return 0


def cmd_segments(args: argparse.Namespace) -> int:
    """List or print a segment window."""
    ws = lib.RefWorkspace.resolve(args.ref)
    data = ws.load_segments(Path(args.segments) if args.segments else None)
    if args.segment_id:
        s, e, seg = ws.resolve_segment(args.segment_id, Path(args.segments) if args.segments else None)
        print(json.dumps({"vod_start_s": s, "vod_end_s": e, "segment": seg}, indent=2))
        return 0
    for seg in data.get("segments", []):
        print(
            f"{seg.get('id', '?'):28s}  {seg.get('vod_start_s')} → {seg.get('vod_end_s')}  "
            f"{seg.get('label', '')}"
        )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="List yt_reference workspaces")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", help="Show one workspace status + segments")
    _add_ref_arg(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("fetch", help="Download VOD + scaffold workspace")
    _add_ref_arg(p)
    p.add_argument("--url", default=None, help="YouTube URL (sets ref id from URL)")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument(
        "--no-template-layout",
        action="store_true",
        help="Do not copy Kentroid button layout template",
    )
    p.set_defaults(func=cmd_fetch)

    def _window_args(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--start", default=None, help="VOD start (sec or M:SS / H:MM:SS)")
        pp.add_argument("--end", default=None, help="VOD end")
        pp.add_argument("--segment-id", default=None)
        pp.add_argument("--segments", default=None, help="Override segments JSON path")

    p = sub.add_parser("extract", help="Button extract for a window")
    _add_ref_arg(p)
    _window_args(p)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("-o", "--out", default=None, help="Output stem (default: inputs/…)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("frames", help="Dump game keyframes for a window")
    _add_ref_arg(p)
    _window_args(p)
    p.add_argument("--every", type=float, default=1.0, help="Seconds between frames")
    p.add_argument("--full-frame", action="store_true", help="Full 1080p, not game crop")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=cmd_frames)

    p = sub.add_parser("analyze", help="Holds/duty/(optional spark) from an extract JSON")
    _add_ref_arg(p)
    p.add_argument("--inputs-stem", required=True, help="Path or name under inputs/")
    p.add_argument("--spark", action="store_true", help="Scan gold afterimages")
    p.add_argument("--spark-step", type=float, default=0.1)
    p.add_argument("--min-hold", type=float, default=0.1)
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("chunk", help="Extract + frames + analyze → chunks/<name>/")
    _add_ref_arg(p)
    _window_args(p)
    p.add_argument("--name", default=None, help="Chunk folder name")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--every", type=float, default=1.0, help="Keyframe interval (s)")
    p.add_argument("--spark", action="store_true", help="Gold shinespark scan")
    p.add_argument("--spark-step", type=float, default=0.1)
    p.add_argument("--min-hold", type=float, default=0.1)
    p.add_argument("--full-frame", action="store_true")
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("segments", help="List marked segments")
    _add_ref_arg(p)
    p.add_argument("--segment-id", default=None)
    p.add_argument("--segments", default=None)
    p.set_defaults(func=cmd_segments)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
