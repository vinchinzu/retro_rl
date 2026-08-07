"""Level 7 entry probe — planning dry-run by default; live refuses without Whistle.

Examples::

    uv run python zelda_i/scripts/probe_level7_entry.py --plan-only
    uv run python zelda_i/scripts/probe_level7_entry.py --infinite-life --save-state
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

from zelda_i.level7_overworld import (
    LEVEL7_TRIFORCE_BIT,
    has_food,
    has_whistle,
    missing_entry_caps,
    planning_report,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_FOOD, ADDR_WHISTLE, read_snapshot, read_u8


def _print_plan() -> int:
    report = planning_report()
    print("=== Level 7 entry — PLANNING (source hypotheses, not live) ===")
    print(json.dumps(report, indent=2))
    print()
    print("Required entry cap: whistle @", report["ram"]["whistle"])
    print("Mid-dungeon cap: food/bait @", report["ram"]["food"])
    print("Triforce bit:", hex(LEVEL7_TRIFORCE_BIT))
    print("Hypothesized bait shop:", report["screens_hypothesized"]["bait_shop"])
    print("Hypothesized pond:", report["screens_hypothesized"]["pond"])
    print()
    print("Live entry requires real ADDR_WHISTLE from L5 (no Clean poke).")
    print("Optional: walk pond screen without whistle and save OW_L7Pond only.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-only", action="store_true", default=False)
    p.add_argument(
        "--allow-missing-caps",
        action="store_true",
        help="Allow pond-map walk without Whistle (no entry claim)",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--from-state", default=None)
    p.add_argument("--tag", default="l7_entry")
    args = p.parse_args()

    if args.plan_only or not (
        args.infinite_life or args.save_state or args.from_state or args.allow_missing_caps
    ):
        rc = _print_plan()
        if args.plan_only or not (
            args.infinite_life or args.save_state or args.from_state
        ):
            if not args.plan_only:
                print(
                    "\nNo live hop controller yet. Re-run with --plan-only, "
                    "or implement pond walk then re-enable live mode."
                )
            return rc

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
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "food": int(read_u8(ram, ADDR_FOOD)),
            "triforce": snap.triforce,
        },
        "missing_entry_caps": missing,
        "has_food": has_food(ram),
        "allow_missing_caps": args.allow_missing_caps,
        "live_nav_implemented": False,
        "success": False,
        "notes": [
            "level7_overworld hop controller not implemented; refuse full entry",
        ],
    }

    if missing and not args.allow_missing_caps:
        report["notes"].append(f"refused: missing {missing}")
        print(json.dumps(report, indent=2))
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        write_json_report(report, RECORDINGS_DIR / f"{args.tag}_refused.json")
        env.close()
        return 2

    if not has_food(ram):
        report["notes"].append(
            "food/bait missing — pond entry may work; hungry Goriya will block clear"
        )

    print(json.dumps(report, indent=2))
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(report, RECORDINGS_DIR / f"{args.tag}_stub.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
