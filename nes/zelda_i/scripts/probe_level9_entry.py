"""Level 9 entry probe — planning dry-run by default; live refuses without full TF.

Examples::

    uv run python zelda_i/scripts/probe_level9_entry.py --plan-only
    uv run python zelda_i/scripts/probe_level9_entry.py --rock-only --infinite-life
    uv run python zelda_i/scripts/probe_level9_entry.py --infinite-life --save-state
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

from zelda_i.level9_overworld import (
    FULL_TRIFORCE,
    has_full_triforce,
    has_silver_arrows,
    missing_entry_caps,
    planning_report,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_ARROWS, ADDR_RING, ADDR_TRIFORCE, read_snapshot, read_u8


def _print_plan() -> int:
    report = planning_report()
    print("=== Level 9 entry — PLANNING (source hypotheses, not live) ===")
    print(json.dumps(report, indent=2))
    print()
    print("Required entry: full triforce ==", hex(FULL_TRIFORCE), "@", report["ram"]["triforce"])
    print("Ganon finish: silver arrows @", report["ram"]["arrows"])
    print("Hypothesized bomb rock:", report["screens_hypothesized"]["bomb_rock"])
    print()
    print("OW rock can be mapped without TF (--rock-only).")
    print("Interior Old Man blocks without all 8 shards (no Clean poke).")
    print("Ending/credits stop is an unverified stub.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-only", action="store_true", default=False)
    p.add_argument(
        "--rock-only",
        action="store_true",
        help="Map Spectacle Rock OW only; do not require full triforce",
    )
    p.add_argument(
        "--allow-missing-caps",
        action="store_true",
        help="Alias for --rock-only style mapping without full TF",
    )
    p.add_argument("--infinite-life", action="store_true")
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--from-state", default=None)
    p.add_argument("--tag", default="l9_entry")
    args = p.parse_args()

    rock_only = args.rock_only or args.allow_missing_caps

    if args.plan_only or not (
        args.infinite_life or args.save_state or args.from_state or rock_only
    ):
        rc = _print_plan()
        if args.plan_only or not (
            args.infinite_life or args.save_state or args.from_state
        ):
            if not args.plan_only:
                print(
                    "\nNo live hop controller yet. Re-run with --plan-only, "
                    "or implement rock walk then re-enable live mode."
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
    missing = missing_entry_caps(ram, rock_only=rock_only)

    report = {
        "tag": args.tag,
        "plan": planning_report(),
        "rock_only": rock_only,
        "start": {
            "mode": snap.mode,
            "level": snap.level,
            "screen": hex(snap.screen),
            "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
            "full_triforce": has_full_triforce(ram),
            "ring": int(read_u8(ram, ADDR_RING)),
            "arrows": int(read_u8(ram, ADDR_ARROWS)),
            "bombs": snap.bombs,
        },
        "missing_entry_caps": missing,
        "has_silver_arrows": has_silver_arrows(ram),
        "live_nav_implemented": False,
        "success": False,
        "notes": [
            "level9_overworld hop controller not implemented; refuse full entry",
            "ending stop stub always False",
        ],
    }

    if missing and not rock_only:
        report["notes"].append(f"refused: missing {missing}")
        print(json.dumps(report, indent=2))
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        write_json_report(report, RECORDINGS_DIR / f"{args.tag}_refused.json")
        env.close()
        return 2

    if rock_only:
        report["notes"].append(
            "rock-only mode: may map OW_L9Rock; do not claim Level9Entrance "
            "without full TF + bomb cave settle"
        )

    print(json.dumps(report, indent=2))
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(report, RECORDINGS_DIR / f"{args.tag}_stub.json")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
