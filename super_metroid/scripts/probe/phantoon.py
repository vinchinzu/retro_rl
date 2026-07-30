#!/usr/bin/env python3
"""Capture and probe Power Bombs → ship route → Phantoon development states.

Examples:

```bash
uv run python super_metroid/scripts/probe/phantoon.py collect-pb
# Skip pure Pink PB / GHZ / Noob (dev bridge) and land Red Tower with PB:
uv run python super_metroid/scripts/probe/phantoon.py skip-to-red
uv run python super_metroid/scripts/probe/phantoon.py skip-to-ghz
uv run python super_metroid/scripts/probe/phantoon.py capture-entry
uv run python super_metroid/scripts/probe/phantoon.py fight --frames 6000
uv run python super_metroid/scripts/probe/phantoon.py ship-route
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.dev.phantoon_dev import (  # noqa: E402
    GHZ_PLAYABLE_STATE,
    PHANTOON_ENTRY_STATE,
    RED_TOWER_POST_PB_STATE,
    RED_TOWER_STATE,
    capture_phantoon_entry,
    collect_power_bombs,
    door_warp_ship_route,
    grant_power_bombs_dev,
    run_phantoon_fight,
    skip_to_ghz,
    skip_to_red_tower_post_pb,
)
from super_metroid.ram import parse_env_state  # noqa: E402


def _ship_route(*, source: Path, grant_pbs: bool) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        for _ in range(3):
            state = parse_env_state(env)
            assist.apply(env.data, state)
            env.step(idle_action())
        if grant_pbs and parse_env_state(env).max_power_bombs <= 0:
            grant_power_bombs_dev(env)
        hops = door_warp_ship_route(env, place_free=True, save_hops=True)
        final = parse_env_state(env)
        return {
            "hops": hops,
            "final": {
                "roomIdHex": f"0x{final.room_id:04X}",
                "powerBombs": f"{final.power_bombs}/{final.max_power_bombs}",
            },
            "developmentOnly": True,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("collect-pb", help="Door-warp collect Pink Brinstar Power Bombs")

    skip_ghz = sub.add_parser(
        "skip-to-ghz",
        help="Dev skip pure Pink PB: grant PB, warp to GHZ playable start",
    )
    skip_ghz.add_argument("--output", type=Path, default=GHZ_PLAYABLE_STATE)

    skip_red = sub.add_parser(
        "skip-to-red",
        help="Dev skip pure PB/GHZ/Noob: grant PB, warp chain to Red Tower",
    )
    skip_red.add_argument("--output", type=Path, default=RED_TOWER_POST_PB_STATE)

    cap = sub.add_parser("capture-entry", help="Warp Red Tower → Phantoon entry")
    cap.add_argument("--source", type=Path, default=RED_TOWER_POST_PB_STATE)
    cap.add_argument("--output", type=Path, default=PHANTOON_ENTRY_STATE)
    cap.add_argument("--no-grant-pbs", action="store_true")

    fight = sub.add_parser("fight", help="Spray-fight Phantoon from entry state")
    fight.add_argument("--frames", type=int, default=6000)
    fight.add_argument("--source", type=Path, default=PHANTOON_ENTRY_STATE)

    route = sub.add_parser("ship-route", help="Door-warp ship route hops only")
    route.add_argument("--source", type=Path, default=RED_TOWER_POST_PB_STATE)
    route.add_argument("--no-grant-pbs", action="store_true")

    args = parser.parse_args()
    if args.command == "collect-pb":
        result = collect_power_bombs()
    elif args.command == "skip-to-ghz":
        result = skip_to_ghz(output=args.output)
    elif args.command == "skip-to-red":
        result = skip_to_red_tower_post_pb(output=args.output)
    elif args.command == "capture-entry":
        result = capture_phantoon_entry(
            source=args.source,
            output=args.output,
            grant_pbs=not args.no_grant_pbs,
        )
    elif args.command == "fight":
        result = run_phantoon_fight(
            source=args.source,
            max_frames=args.frames,
            capture_if_missing=True,
        )
    elif args.command == "ship-route":
        result = _ship_route(source=args.source, grant_pbs=not args.no_grant_pbs)
    else:
        raise SystemExit(f"unknown command {args.command}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
