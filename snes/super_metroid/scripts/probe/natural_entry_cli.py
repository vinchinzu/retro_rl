#!/usr/bin/env python3
"""Multi-boss natural-entry capture CLI (one shared entry, not per-boss sprawl).

Development infrastructure only — not continuous evidence. Capture records
room + pose + door settle without progression / boss-bit forges.

Library API: ``super_metroid.combat.natural_entry`` (capture helpers only).

```bash
# Catalog + requirements
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py list
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py describe phantoon

# Bomb Torizo: continuous power-on prefix (slow)
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py capture-natural bomb_torizo

# Non-BT bosses: settle capture from a doorway / predecessor save
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  kraid --from-state entry --mode room_entry
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  phantoon --from-state path/to/phantoon_entry.state --mode room_entry
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  botwoon --from-state path/to/botwoon_entry.state --mode room_entry

# Plan only (no emulator)
uv run python snes/super_metroid/scripts/probe/natural_entry_cli.py capture-natural \\
  phantoon --plan-only
```

Bomb Torizo back-compat remains on ``bomb_torizo_combat.py capture-natural``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get("_SNES_IMPORT_ROOT", ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.combat.natural_entry import (  # noqa: E402
    CAPTURE_MODES,
    describe_capture_target,
    list_capture_targets,
    normalize_boss_id,
    run_capture_natural,
)


def add_capture_natural_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared ``capture-natural`` flags to a subparser."""
    parser.add_argument(
        "boss",
        help=(
            "Catalog boss id or alias "
            "(bomb_torizo, kraid, phantoon, botwoon, draygon, ...)"
        ),
    )
    parser.add_argument(
        "--from-state",
        type=str,
        default=None,
        help=(
            "Source save path or known alias (e.g. entry for kraid). "
            "Required for bosses without a continuous power-on prefix."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=CAPTURE_MODES,
        default=None,
        help="Capture predicate (default: active for BT, room_entry otherwise)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Scratch .state path (default: scratch/natural_<boss>_active.state)",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Provenance JSON path (default: next to state)",
    )
    parser.add_argument(
        "--max-prefix-frames",
        type=int,
        default=60_000,
        help="Max frames for continuous power-on prefix (BT)",
    )
    parser.add_argument(
        "--max-source-frames",
        type=int,
        default=3_000,
        help="Max idle frames when capturing from --from-state",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print capture plan JSON without running the emulator",
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Full multi-boss natural-entry CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Multi-boss natural-entry capture (development infrastructure). "
            "Records room + pose + door settle without progression writes. "
            "Not continuous evidence."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser(
        "list",
        help="List catalog bosses and capture requirements",
    )
    p_list.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON (default: short table-ish JSON list)",
    )
    p_list.set_defaults(func=_cli_list)

    p_cap = sub.add_parser(
        "capture-natural",
        help=(
            "Capture natural room/pose/door settle for a boss "
            "(continuous prefix or --from-state)"
        ),
    )
    add_capture_natural_arguments(p_cap)
    p_cap.set_defaults(func=_cli_capture_natural)

    p_desc = sub.add_parser(
        "describe",
        help="Describe capture plan for one boss (no emulator)",
    )
    p_desc.add_argument("boss", help="Boss id or alias")
    p_desc.set_defaults(func=_cli_describe)

    return parser


def _cli_list(args: argparse.Namespace) -> int:
    targets = list_capture_targets()
    if args.json:
        print(json.dumps(targets, indent=2))
    else:
        rows = [
            {
                "bossId": t["bossId"],
                "roomIdHex": t["roomIdHex"],
                "continuousPrefix": t["continuousPrefix"],
                "requiresFromState": t["requiresFromState"],
                "defaultMode": t["defaultMode"],
                "continuousStatus": t["continuousStatus"],
            }
            for t in targets
        ]
        print(json.dumps(rows, indent=2))
    return 0


def _cli_describe(args: argparse.Namespace) -> int:
    try:
        target = describe_capture_target(args.boss)
    except KeyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(target, indent=2))
    return 0


def _cli_capture_natural(args: argparse.Namespace) -> int:
    try:
        bid = normalize_boss_id(args.boss)
    except KeyError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 2

    if args.plan_only:
        plan = describe_capture_target(bid)
        plan["requestedMode"] = args.mode or plan["defaultMode"]
        plan["fromState"] = args.from_state
        plan["command"] = "capture-natural"
        plan["planOnly"] = True
        print(json.dumps(plan, indent=2))
        # plan-only is green when catalog resolves (including non-BT).
        return 0

    result = run_capture_natural(
        bid,
        from_state=args.from_state,
        mode=args.mode,
        output=args.output,
        provenance_path=args.provenance,
        max_prefix_frames=args.max_prefix_frames,
        max_source_frames=args.max_source_frames,
    )
    payload = result.to_dict()
    payload["command"] = "capture-natural"
    text = json.dumps(payload, indent=2)
    print(text)
    if args.report is not None:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n", encoding="utf-8")
    if result.success:
        return 0
    # missing_from_state is a usage error (2); other failures are 1.
    if result.outcome.startswith("missing_from_state"):
        return 2
    return 1


def cli_main(argv: list[str] | None = None) -> int:
    """Entry for multi-boss natural-entry CLI."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(cli_main())
