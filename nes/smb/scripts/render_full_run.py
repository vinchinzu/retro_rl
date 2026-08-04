#!/usr/bin/env python3
"""Render an SMB multi-exit showcase video.

Default source is a **verified single-session playthrough** (successful final
attempt per exit, death-checked in the emulator) — not a naive legal stitch.

Examples::

    # Warp any% real playthrough (auto-select best verified session)
    uv run python -m smb.scripts.render_full_run --route warp

    # Pin a completed full-game practice session
    uv run python -m smb.scripts.render_full_run --route warp --session 20260429_172649

    # Dry-run plan only
    uv run python -m smb.scripts.render_full_run --route warp --plan-only

    # Old legal-stitch / optimizer sources
    uv run python -m smb.scripts.render_full_run --route warp --source legal_stitch
    uv run python -m smb.scripts.render_full_run --route warp --source optimizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smb.full_run import build_stitch_plan, render_stitch_plan, write_plan_manifest
from smb.paths import FULLGAME_REPLAYS_DIR, GAME_DIR
from smb.routes import get_route, list_routes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        "-r",
        default="warp",
        help="Route id/alias (warp, all_exits, smb_warp_any_percent, ...)",
    )
    parser.add_argument(
        "--source",
        choices=("playthrough", "legal_stitch", "optimizer"),
        default="playthrough",
        help="Clip source (default: verified single-session playthrough)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Practice session id for --source playthrough (default: auto)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output MP4 path (default under smb/recordings/fullgame_replays/)",
    )
    parser.add_argument("--scale", type=int, default=3, help="Pixel scale (default 3)")
    parser.add_argument(
        "--title-frames",
        type=int,
        default=None,
        help="Interstitial frames between exits (default: 24 playthrough / 90 other)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve clips and write manifest; do not encode video",
    )
    parser.add_argument(
        "--list-routes",
        action="store_true",
        help="List registered exit routes and exit",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Error if any route exit lacks a source (default: skip missing)",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Playthrough: do not require emulator death-check (not recommended)",
    )
    args = parser.parse_args(argv)

    if args.list_routes:
        for route in list_routes():
            print(
                f"{route.route_id:<28} {len(route.exits):>3} exits  "
                f"{route.display_name}"
            )
        return 0

    route = get_route(args.route)
    plan = build_stitch_plan(
        route,
        source=args.source,
        skip_missing=not args.fail_on_missing,
        session_id=args.session,
        require_verified=not args.allow_unverified,
    )

    default_name = f"{route.route_id}_{args.source}.mp4"
    if args.source == "playthrough" and plan.clips:
        sid = plan.clips[0].session_id
        if sid and all(c.session_id == sid for c in plan.clips):
            default_name = f"{route.route_id}_playthrough_{sid}.mp4"
    output = args.output or (FULLGAME_REPLAYS_DIR / default_name)

    print(f"Route: {route.display_name} ({route.route_id})")
    print(f"Source: {plan.source_kind}")
    print(f"Exits on route: {len(route.exits)}  resolved clips: {len(plan.clips)}")
    print(
        f"Play frames: {plan.total_play_frames} "
        f"({plan.total_play_frames / 60.0:.1f}s @ 60fps)"
    )
    for clip in plan.clips:
        print(
            f"  {clip.exit.exit_id:>4}  {clip.frames:5d}f  "
            f"{clip.source_kind}  {clip.state_path.name}"
            + (f"  session={clip.session_id}" if clip.session_id else "")
            + (f"  branch={clip.branch_id}" if clip.branch_id else "")
        )
    if plan.missing:
        print(f"Missing: {', '.join(plan.missing)}")
    for note in plan.notes:
        print(f"  note: {note}")

    if args.plan_only:
        manifest = Path(output).with_suffix(".json")
        write_plan_manifest(plan, manifest)
        print(f"Wrote plan manifest: {manifest}")
        return 0

    if not plan.clips:
        print("No clips to render.", file=sys.stderr)
        return 1

    render_stitch_plan(
        plan,
        output,
        scale=args.scale,
        title_card_frames=args.title_frames,
        game_dir=GAME_DIR,
        abort_on_death=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
