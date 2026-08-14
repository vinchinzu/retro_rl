"""Compact room-engine CLI (SM-style small context for agents).

Usage:
  uv run python alttp/scripts/room_engine.py list
  uv run python alttp/scripts/room_engine.py show room_61
  uv run python alttp/scripts/room_engine.py show room_61 --json
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \\
      --edge west_to_0x60 --state CastleMain --overlay

Agents: prefer ``show`` over reading segment source. Geometry lives in maps/*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

def cmd_list(_: argparse.Namespace) -> int:
    from alttp.room_sense import list_room_maps, load_room_map

    maps = list_room_maps()
    if not maps:
        print("No maps in alttp/maps/")
        return 1
    for mid in maps:
        m = load_room_map(mid)
        doors = ", ".join(f"{d.label}({d.direction}→{d.role})" for d in m.doors)
        print(f"{mid:12} 0x{m.room_base_id:02X}  {m.name}")
        print(f"             doors: {doors}")
        if m.source_state:
            print(f"             state: {m.source_state}")
    return 0

def cmd_show(args: argparse.Namespace) -> int:
    from alttp.room_sense import load_room_map

    m = load_room_map(args.map_id)
    if args.json:
        print(json.dumps(m.compact_summary(), indent=2))
        return 0
    s = m.compact_summary()
    print(f"map {s['mapId']}  room {s['roomHex']}  {s['name']}")
    if s.get("sourceState"):
        print(f"sourceState: {s['sourceState']}")
    print("points:")
    for label, xy in s["points"].items():
        print(f"  {label:22} ({xy[0]}, {xy[1]})")
    print("doors:")
    for d in s["doors"]:
        print(
            f"  {d['label']:22} {d['dir']:5} → {d['to']}  "
            f"role={d['role']}  path={d['path']}"
        )
    print("clearPolicy:", s["clear"])
    for n in s.get("notes") or []:
        print(f"  note: {n}")
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    _configure_headless()
    from alttp.opening_route.room_engine import run_room_edge
    from alttp.paths import RECORDINGS_DIR
    from alttp.room_sense import load_room_map, overlay_from_env
    from alttp.startup import build_boot_env, snapshot_env
    from alttp.primitives import settle_control
    import numpy as np
    from PIL import Image

    room_map = load_room_map(args.map_id)
    door = room_map.door(args.edge)
    if door is None:
        known = [d.label for d in room_map.doors]
        print(f"unknown edge {args.edge!r}; known: {known}", file=sys.stderr)
        return 2

    state = args.state or room_map.source_state or "CastleMain"
    out_dir = RECORDINGS_DIR / "probe_room_engine"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_boot_env(state)
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        if args.overlay:
            img = overlay_from_env(
                env,
                include_enemies=True,
                points=room_map.points,
                title=f"{args.map_id} start",
            )
            start_path = out_dir / f"{args.map_id}_{args.edge}_start.png"
            Image.fromarray(img).save(start_path)
            print(f"Wrote {start_path}")

        result = run_room_edge(
            env,
            args.map_id,
            args.edge,
            clear=not args.no_clear,
            source="state_load_dev",
        )

        shot = Path(args.screenshot) if args.screenshot else (
            RECORDINGS_DIR / f"{args.map_id}_{args.edge}.png"
        )
        shot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(env.render())).save(shot)  # type: ignore[attr-defined]
        print(f"Wrote {shot}")

        if args.overlay:
            img = overlay_from_env(
                env,
                include_enemies=True,
                points=room_map.points,
                title=f"{args.map_id} end",
            )
            end_path = out_dir / f"{args.map_id}_{args.edge}_end.png"
            Image.fromarray(img).save(end_path)
            print(f"Wrote {end_path}")
    finally:
        env.close()  # type: ignore[attr-defined]

    report = result.to_report(f"room_engine:{args.map_id}:{args.edge}")
    report["roomMap"] = room_map.compact_summary()
    report["cli"] = {
        "mapId": args.map_id,
        "edge": args.edge,
        "state": state,
    }
    json_out = Path(args.json_out) if args.json_out else (
        RECORDINGS_DIR / f"{args.map_id}_{args.edge}.json"
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {json_out}")
    print(
        f"ok={result.ok} phase={result.phase} frames={result.frames} "
        f"blocker={result.blocker!r}"
    )
    snap = result.snapshot
    print(
        f"final room=0x{snap.room_base_id:02X} xy=({snap.link_x},{snap.link_y}) "
        f"indoors={snap.indoors}"
    )

    # Isolated edge: ok=True means door destination reached.
    return 0 if result.ok else 1

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List maps/*.json")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Compact map summary (agent context)")
    p_show.add_argument("map_id", help="e.g. room_61 or 0x61")
    p_show.add_argument("--json", action="store_true", help="Machine-readable")
    p_show.set_defaults(func=cmd_show)

    p_run = sub.add_parser("run", help="Clear + exit one door from a save-state")
    p_run.add_argument("map_id", help="e.g. room_61")
    p_run.add_argument("--edge", required=True, help="Door label from map")
    p_run.add_argument("--state", default="", help="Save state (default: map sourceState)")
    p_run.add_argument("--no-clear", action="store_true")
    p_run.add_argument("--overlay", action="store_true")
    p_run.add_argument("--json-out", type=Path, default=None)
    p_run.add_argument("--screenshot", type=Path, default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
