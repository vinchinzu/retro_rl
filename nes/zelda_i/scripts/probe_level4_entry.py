"""Level 4 entry probe — plan dry-run, or delegate live to run_level4_entry.

Examples::

    uv run python nes/zelda_i/scripts/probe_level4_entry.py --plan-only
    uv run python nes/zelda_i/scripts/probe_level4_entry.py --infinite-life \\
        --from-state Level3Complete --trials 2 --save-state
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zelda_i.level4_overworld import (
    LEVEL4_TRIFORCE_BIT,
    planning_report,
)


def _print_plan() -> int:
    report = planning_report()
    print("=== Level 4 entry — LIVE OW path (assisted recon rr-0fx) ===")
    print(json.dumps(report, indent=2))
    print()
    print("Required entry cap: raft @", report["ram"]["raft"])
    print("Dungeon item: stepladder @", report["ram"]["ladder"])
    print("Triforce bit:", hex(LEVEL4_TRIFORCE_BIT))
    print("Dock:", report["screens"]["dock"])
    print("Island:", report["screens"]["island_or_door"])
    print("Entry room:", report["screens"]["entry_room"])
    print()
    print("Live runner: scripts/run_level4_entry.py --infinite-life --trials 2")
    print("Do not poke ADDR_RAFT; not Clean STATUS.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-only",
        action="store_true",
        default=False,
        help="Print live/planning summary; do not boot emu",
    )
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (forwarded to run_level4_entry)",
    )
    p.add_argument(
        "--save-state",
        action="store_true",
        help="Save OW_L4Dock / Level4Entrance on success",
    )
    p.add_argument("--from-state", default="Level3Complete")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--tag", default="l4_entry")
    p.add_argument("--dock-only", action="store_true")
    p.add_argument("--door-only", action="store_true")
    args = p.parse_args()

    if args.plan_only or not (
        args.infinite_life or args.save_state or args.trials > 1
    ):
        return _print_plan()

    # Live path: durable runner (scripts/ is not a package — load by path).
    import importlib.util

    runner_path = Path(__file__).resolve().parent / "run_level4_entry.py"
    spec = importlib.util.spec_from_file_location("run_level4_entry", runner_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    argv = [
        "--from-state",
        args.from_state,
        "--trials",
        str(args.trials),
        "--tag",
        args.tag,
    ]
    if args.infinite_life:
        argv.append("--infinite-life")
    if args.save_state:
        argv.append("--save-state")
    if args.dock_only:
        argv.append("--dock-only")
    if args.door_only:
        argv.append("--door-only")
    return mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
