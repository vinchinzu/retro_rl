"""CLI for area-basemap CoG viewer.

::

    uv run python -m super_metroid.map_viewer serve --open

    uv run python -m super_metroid.map_viewer export-path \\
      tasks/parlor_left_human.json --id parlor_human
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import webbrowser
from functools import partial
from pathlib import Path

from super_metroid.map_viewer.assets import prepare_all
from super_metroid.map_viewer.coords import (
    area_bounds,
    default_viewer_asset_dir,
    load_room_index,
    to_area,
)
from super_metroid.map_viewer.paths import (
    DEFAULT_COLORS,
    discover_default_sources,
    export_catalog,
    export_path,
    load_path_source,
)
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR


def _cmd_prepare(args: argparse.Namespace) -> int:
    result = prepare_all(
        overview_size=args.size,
        force=args.force,
        out_dir=Path(args.out) if args.out else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_export_path(args: argparse.Namespace) -> int:
    rooms = load_room_index(Path(args.graph) if args.graph else None)
    bounds = area_bounds(rooms)
    src = Path(args.source)
    if not src.is_absolute():
        for base in (Path.cwd(), GAME_DIR, RECORDINGS_DIR):
            cand = base / src
            if cand.exists():
                src = cand
                break
    wp = load_path_source(
        src,
        rooms,
        bounds,
        stride=args.stride,
        max_points=args.max_points,
        path_id=args.id,
        label=args.label,
        color=args.color,
        kind=args.kind,
        max_step_px=args.max_step,
    )
    out = (
        Path(args.output)
        if args.output
        else default_viewer_asset_dir() / "paths" / f"{wp.id}.json"
    )
    export_path(wp, out, compact=not args.full)
    paths_dir = out.parent
    index_path = paths_dir / "index.json"
    entries = []
    if index_path.is_file():
        try:
            entries = list(json.loads(index_path.read_text()).get("paths") or [])
        except json.JSONDecodeError:
            entries = []
    entries = [e for e in entries if e.get("id") != wp.id]
    from super_metroid.map_viewer.coords import area_slug

    entries.append(
        {
            "id": wp.id,
            "label": wp.label,
            "kind": wp.kind,
            "color": wp.color,
            "point_count": len(wp.points),
            "segment_count": len(wp.segments),
            "marker_count": len(wp.markers),
            "primary_area": wp.primary_area,
            "primary_area_slug": area_slug(wp.primary_area) if wp.primary_area else "",
            "file": out.name,
            "source": wp.source,
            "meta": wp.meta,
        }
    )
    index_path.write_text(
        json.dumps(
            {"schema": "super_metroid_path_catalog_v2", "paths": entries},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {out}  points={len(wp.points)} segments={len(wp.segments)} "
        f"markers={len(wp.markers)} area={wp.primary_area} kind={wp.kind}"
    )
    return 0


def _cmd_export_defaults(args: argparse.Namespace) -> int:
    rooms = load_room_index()
    bounds = area_bounds(rooms)
    out_dir = (
        Path(args.out) if args.out else default_viewer_asset_dir() / "paths"
    )
    sources = [Path(s) for s in (args.sources or [])] or discover_default_sources()
    paths = []
    for i, src in enumerate(sources):
        try:
            stride = args.stride
            if src.name == "series.jsonl" and src.stat().st_size > 800_000:
                stride = max(stride, 4)
            wp = load_path_source(
                src,
                rooms,
                bounds,
                stride=stride,
                max_points=args.max_points,
                color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                max_step_px=args.max_step,
            )
            if not wp.points and not wp.markers:
                print(f"skip empty: {src}")
                continue
            # Dense paths need real segments
            if wp.kind != "continuous_sparse" and not wp.segments:
                print(f"skip no segments: {src}")
                continue
            paths.append(wp)
            print(
                f"  + {wp.id}: {len(wp.segments)} segs / {len(wp.points)} pts "
                f"area={wp.primary_area} from {src}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {src}: {exc}")
    if not paths:
        print("No paths exported.")
        return 1
    index = export_catalog(paths, out_dir)
    print(f"Catalog: {index} ({len(paths)} paths)")
    return 0


def _cmd_where(args: argparse.Namespace) -> int:
    rooms = load_room_index()
    bounds = area_bounds(rooms)
    rid = int(args.room, 0)
    room = rooms.get(rid)
    if room is None:
        print(f"Unknown room {rid:#x}", file=sys.stderr)
        return 1
    b = bounds[room.area]
    ax, ay = to_area(room, b, args.x, args.y, x_sub=args.x_sub, y_sub=args.y_sub)
    print(
        json.dumps(
            {
                "room": room.to_dict(),
                "area_bounds": b.to_dict(),
                "local": {"x": args.x, "y": args.y},
                "area_px": {"x": ax, "y": ay},
            },
            indent=2,
        )
    )
    return 0


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        if self.command != "GET" or not str(args[0]).startswith("200"):
            super().log_message(fmt, *args)


def _cmd_serve(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else default_viewer_asset_dir()
    print("Preparing area basemaps + static UI…")
    prepare_all(overview_size=args.size, force=args.force, out_dir=out)
    paths_dir = out / "paths"
    if not (paths_dir / "index.json").is_file() or args.export_defaults:
        print("Exporting default dense path(s)…")
        _cmd_export_defaults(
            argparse.Namespace(
                out=str(paths_dir),
                sources=None,
                stride=args.stride,
                max_points=args.max_points,
                max_step=args.max_step,
            )
        )
    handler = partial(_QuietHandler, directory=str(out))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving {out}")
        print(f"Open {url}")
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m super_metroid.map_viewer",
        description="Pixel-aligned area map + Samus CoG path overlays",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Copy area basemaps + rooms geojson + UI")
    prep.add_argument(
        "--size",
        type=int,
        default=0,
        help="Max long-edge for area PNGs (0 = full-res 1:1 copy, pixel-accurate)",
    )
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--out")
    prep.set_defaults(func=_cmd_prepare)

    exp = sub.add_parser("export-path", help="Export one source as segmented path JSON")
    exp.add_argument("source")
    exp.add_argument("--id")
    exp.add_argument("--label")
    exp.add_argument("--color", default=None)
    exp.add_argument("--kind", default=None)
    exp.add_argument("--stride", type=int, default=1)
    exp.add_argument("--max-points", type=int, default=None)
    exp.add_argument("--max-step", type=float, default=48.0, help="Max px between line joints")
    exp.add_argument("--output", "-o")
    exp.add_argument("--graph")
    exp.add_argument("--full", action="store_true")
    exp.set_defaults(func=_cmd_export_path)

    defs = sub.add_parser("export-defaults", help="Export 1–2 dense default paths")
    defs.add_argument("--out")
    defs.add_argument("--sources", nargs="*")
    defs.add_argument("--stride", type=int, default=2)
    defs.add_argument("--max-points", type=int, default=30_000)
    defs.add_argument("--max-step", type=float, default=48.0)
    defs.set_defaults(func=_cmd_export_defaults)

    where = sub.add_parser("where", help="Room-local → area basemap px")
    where.add_argument("room")
    where.add_argument("x", type=int)
    where.add_argument("y", type=int)
    where.add_argument("--x-sub", type=int, default=None)
    where.add_argument("--y-sub", type=int, default=None)
    where.set_defaults(func=_cmd_where)

    serve = sub.add_parser("serve", help="Prepare + export defaults + HTTP server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--size",
        type=int,
        default=0,
        help="Max long-edge (0 = full-res area maps)",
    )
    serve.add_argument("--force", action="store_true")
    serve.add_argument("--out")
    serve.add_argument("--stride", type=int, default=2)
    serve.add_argument("--max-points", type=int, default=30_000)
    serve.add_argument("--max-step", type=float, default=48.0)
    serve.add_argument("--export-defaults", action="store_true")
    serve.add_argument("--open", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
