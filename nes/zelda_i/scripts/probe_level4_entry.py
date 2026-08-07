"""Level 4 entry probe — planning dry-run by default; live refuses without Raft.

Examples::

    uv run python zelda_i/scripts/probe_level4_entry.py --plan-only
    uv run python zelda_i/scripts/probe_level4_entry.py --infinite-life --save-state
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
    LEVEL4,
    LEVEL4_TRIFORCE_BIT,
    has_raft,
    missing_entry_caps,
    planning_report,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, ADDR_RAFT, read_snapshot, read_u8


def _print_plan() -> int:
    report = planning_report()
    print("=== Level 4 entry — PLANNING (source hypotheses, not live) ===")
    print(json.dumps(report, indent=2))
    print()
    print("Required entry cap: raft @", report["ram"]["raft"])
    print("Dungeon item: stepladder @", report["ram"]["ladder"])
    print("Triforce bit:", hex(LEVEL4_TRIFORCE_BIT))
    print("Hypothesized dock:", report["screens_hypothesized"]["dock"])
    print("Hypothesized island:", report["screens_hypothesized"]["island_or_door"])
    print()
    print("Live entry requires real ADDR_RAFT from L3 (no Clean poke).")
    print("Optional: walk dock without raft and save OW_L4Dock only.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan-only",
        action="store_true",
        default=False,
        help="Print required caps and hypothesized screens; do not boot emu",
    )
    p.add_argument(
        "--allow-missing-caps",
        action="store_true",
        help="Allow live walk toward dock without Raft (map only; no entry claim)",
    )
    p.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist if a live nav path is implemented later",
    )
    p.add_argument(
        "--save-state",
        action="store_true",
        help="Save OW_L4Dock / Level4Entrance when live path exists",
    )
    p.add_argument("--from-state", default=None, help="Start state name (future)")
    p.add_argument("--tag", default="l4_entry", help="Recording tag prefix")
    args = p.parse_args()

    if args.plan_only or not args.allow_missing_caps and not args.from_state:
        # Default safe path: always print plan. Live nav not implemented yet.
        rc = _print_plan()
        if args.plan_only:
            return rc
        # Fall through only if user asked for live flags without plan-only.
        if not (args.infinite_life or args.save_state or args.from_state):
            print(
                "\nNo live hop controller yet. Re-run with --plan-only, "
                "or implement dock walk then re-enable live mode."
            )
            return 0

    # Live path scaffold: boot only if we can load RAM and check caps.
    try:
        from retro_harness.env import make_env
        from retro_harness.segment_runner import configure_headless, write_json_report
        from zelda_i.paths import GAME
    except ImportError as exc:
        print("Cannot live-probe (import failed):", exc)
        return _print_plan()

    configure_headless()
    env = make_env(GAME, state=args.from_state or "Level1")
    obs, _ = env.reset()
    ram = env.get_ram()
    snap = read_snapshot(ram)
    missing = missing_entry_caps(ram)

    report = {
        "tag": args.tag,
        "plan": planning_report(),
        "start": {
            "mode": snap.mode,
            "level": snap.level,
            "screen": hex(snap.screen),
            "raft": int(read_u8(ram, ADDR_RAFT)),
            "ladder": int(read_u8(ram, ADDR_LADDER)),
            "triforce": snap.triforce,
        },
        "missing_entry_caps": missing,
        "allow_missing_caps": args.allow_missing_caps,
        "live_nav_implemented": False,
        "success": False,
        "notes": [
            "level4_overworld hop controller not implemented; refuse full entry",
        ],
    }

    if missing and not args.allow_missing_caps:
        report["notes"].append(f"refused: missing {missing}")
        print(json.dumps(report, indent=2))
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        write_json_report(report, RECORDINGS_DIR / f"{args.tag}_refused.json")
        env.close()
        return 2

    if not missing:
        report["notes"].append(
            "raft present; still no automated dock→island nav in this stub"
        )
    else:
        report["notes"].append(
            "dock-map mode: raft missing; do not claim Level4Entrance"
        )

    print(json.dumps(report, indent=2))
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(report, RECORDINGS_DIR / f"{args.tag}_stub.json")
    env.close()
    # Exit 0 for honest stub documentation; 2 only on hard refuse without allow.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
