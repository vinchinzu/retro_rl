#!/usr/bin/env python3
"""Door-warp full-game route skeleton (boss fights skipped).

Examples:

```bash
uv run python snes/super_metroid/scripts/probe/route.py list
uv run python snes/super_metroid/scripts/probe/route.py phantoon-to-ridley
uv run python snes/super_metroid/scripts/probe/route.py ridley-to-mb
uv run python snes/super_metroid/scripts/probe/route.py late-full
uv run python snes/super_metroid/scripts/probe/route.py full
uv run python snes/super_metroid/scripts/probe/route.py full --stop-after morph_ball
uv run python snes/super_metroid/scripts/probe/route.py full-hybrid
uv run python snes/super_metroid/scripts/probe/route.py leg landing_site morph_ball
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.dev.phantoon_dev import (  # noqa: E402
    PHANTOON_ENTRY_STATE,
    capture_phantoon_entry,
)
from super_metroid.dev.route_dev import (  # noqa: E402
    DEFAULT_CONTINUOUS_PREFIX_VIDEO,
    DEFAULT_FRAMES_PER_ROOM,
    DEFAULT_HYBRID_REPORT,
    DEFAULT_HYBRID_VIDEO,
    DEFAULT_TOUR_REPORT,
    DEFAULT_TOUR_VIDEO,
    FULL_LEG_ORDER,
    LATE_LEG_ORDER,
    PHANTOON_ENTRY,
    default_full_source_state,
    grant_route_loadout,
    load_full_hops,
    load_late_hops,
    mark_all_major_bosses,
    run_full_route,
    run_hybrid_full_route,
    run_late_route,
    run_leg,
    run_phantoon_to_ridley,
    run_ridley_to_mother_brain,
    summarize_full_graph_legs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show completion legs and hop coverage")
    sub.add_parser("phantoon-to-ridley", help="Warp Phantoon → Ridley (skip fights)")
    sub.add_parser("ridley-to-mb", help="Warp Ridley → Mother Brain (skip fights)")
    full_late = sub.add_parser("late-full", help="Warp Phantoon → Landing Site finish")
    full_late.add_argument(
        "--stop-after",
        choices=[t for _, t in LATE_LEG_ORDER],
        default=None,
        help="Stop after reaching this anchor",
    )
    full_late.add_argument(
        "--source",
        type=Path,
        default=PHANTOON_ENTRY,
        help="Starting save state (default: Phantoon entry)",
    )

    def _add_full_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--stop-after",
            choices=[t for _, t in FULL_LEG_ORDER],
            default=None,
            help="Stop after reaching this anchor",
        )
        p.add_argument(
            "--start-from-leg",
            choices=sorted({s for s, _ in FULL_LEG_ORDER}),
            default=None,
            help="Skip legs until this source anchor",
        )
        p.add_argument(
            "--source",
            type=Path,
            default=None,
            help="Starting save state (default: natural_post_spore or phantoon)",
        )
        p.add_argument(
            "--no-checkpoints",
            action="store_true",
            help="Do not write dev_route_anchor_*.state files",
        )
        p.add_argument(
            "--video",
            type=Path,
            default=None,
            help="Optional H.264 path (room-tour settle frames only)",
        )
        p.add_argument(
            "--frames-per-room",
            type=int,
            default=DEFAULT_FRAMES_PER_ROOM,
            help=f"Idle frames captured per hop when --video is set (default {DEFAULT_FRAMES_PER_ROOM})",
        )
        p.add_argument(
            "--report",
            type=Path,
            default=None,
            help="Optional JSON report path",
        )

    full = sub.add_parser(
        "full",
        help="Warp all 22 completion legs Ceres → Landing Site finish",
    )
    _add_full_args(full)

    full_tour = sub.add_parser(
        "full-tour",
        help="Full 22-leg tour with default video/report under recordings/",
    )
    _add_full_args(full_tour)

    full_hybrid = sub.add_parser(
        "full-hybrid",
        help=(
            "Hybrid full-route video: continuous supers prefix + "
            "door-warp tour from Super room → Landing Site (bosses skipped)"
        ),
    )
    _add_full_args(full_hybrid)
    full_hybrid.set_defaults(
        video=DEFAULT_HYBRID_VIDEO,
        report=DEFAULT_HYBRID_REPORT,
        start_from_leg="spore_spawn_supers",
    )
    full_hybrid.add_argument(
        "--splice-prefix",
        type=Path,
        default=DEFAULT_CONTINUOUS_PREFIX_VIDEO,
        help=(
            "Continuous prefix mp4 to prepend (default: "
            f"{DEFAULT_CONTINUOUS_PREFIX_VIDEO})"
        ),
    )
    full_hybrid.add_argument(
        "--no-prefix",
        action="store_true",
        help="Record warp suffix only (no continuous-prefix splice)",
    )
    full_hybrid.add_argument(
        "--tour-video",
        type=Path,
        default=None,
        help="Optional path for the warp-tour-only intermediate mp4",
    )

    leg = sub.add_parser("leg", help="Run a single named completion leg")
    leg.add_argument("source")
    leg.add_argument("target")
    leg.add_argument("--source-state", type=Path, default=None)

    ensure = sub.add_parser(
        "ensure-phantoon-entry",
        help="Capture Phantoon entry if missing",
    )

    args = parser.parse_args()

    if args.command == "list":
        full_hops = load_full_hops()
        late_hops = load_late_hops()
        result = {
            "fullLegOrder": [
                {"source": s, "target": t, "key": f"{s}__{t}"}
                for s, t in FULL_LEG_ORDER
            ],
            "lateLegOrder": [
                {"source": s, "target": t, "key": f"{s}__{t}"}
                for s, t in LATE_LEG_ORDER
            ],
            "completionLegs": summarize_full_graph_legs(),
            "fullHopTableKeys": sorted(full_hops.keys()),
            "hopTableKeys": sorted(late_hops.keys()),
            "fullHopCount": sum(len(v) for v in full_hops.values()),
            "nullDoorHops": sum(
                1
                for chain in full_hops.values()
                for h in chain
                if h.get("door") is None
            ),
            "developmentOnly": True,
        }
    elif args.command == "ensure-phantoon-entry":
        if PHANTOON_ENTRY_STATE.exists():
            result = {"exists": True, "path": str(PHANTOON_ENTRY_STATE)}
        else:
            result = capture_phantoon_entry(output=PHANTOON_ENTRY_STATE)
    elif args.command == "phantoon-to-ridley":
        if not PHANTOON_ENTRY.exists():
            capture_phantoon_entry(output=PHANTOON_ENTRY)
        result = run_phantoon_to_ridley()
    elif args.command == "ridley-to-mb":
        result = run_ridley_to_mother_brain()
    elif args.command == "late-full":
        if not args.source.exists() and args.source == PHANTOON_ENTRY:
            capture_phantoon_entry(output=PHANTOON_ENTRY)
        result = run_late_route(
            source_state=args.source,
            stop_after=args.stop_after,
        )
    elif args.command in ("full", "full-tour"):
        video = args.video
        report = args.report
        if args.command == "full-tour":
            if video is None:
                video = DEFAULT_TOUR_VIDEO
            if report is None:
                report = DEFAULT_TOUR_REPORT
        result = run_full_route(
            source_state=args.source,
            stop_after=args.stop_after,
            start_from_leg=args.start_from_leg,
            save_checkpoints=not args.no_checkpoints,
            video_path=video,
            frames_per_room=args.frames_per_room,
            report_path=report,
        )
    elif args.command == "full-hybrid":
        video = args.video if args.video is not None else DEFAULT_HYBRID_VIDEO
        report = args.report if args.report is not None else DEFAULT_HYBRID_REPORT
        prefix = None if args.no_prefix else args.splice_prefix
        result = run_hybrid_full_route(
            source_state=args.source,
            stop_after=args.stop_after,
            start_from_leg=args.start_from_leg,
            save_checkpoints=not args.no_checkpoints,
            video_path=video,
            frames_per_room=args.frames_per_room,
            report_path=report,
            splice_prefix=prefix,
            tour_video_path=args.tour_video,
        )
    elif args.command == "leg":
        env = make_dev_env()
        try:
            src = args.source_state or default_full_source_state()
            if not src.exists():
                capture_phantoon_entry(output=PHANTOON_ENTRY)
                src = PHANTOON_ENTRY
            boot_from_state(env, src)
            grant_route_loadout(env)
            mark_all_major_bosses(env)
            result = run_leg(env, args.source, args.target)
        finally:
            env.close()
    else:
        raise SystemExit(f"unknown command {args.command}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
