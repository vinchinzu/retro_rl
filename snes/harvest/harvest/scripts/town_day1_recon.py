#!/usr/bin/env python3
"""Spring D1 town recon — capture entry, record, replay for automation.

The clean natural entry is power-on → Spring D1 07:00 town gate map ``0x04``
at ``(712,424)``. The D1 handoff needs six conversations (mask ``0x3F`` at
``d1_town_event_mask`` / WRAM ``0x11F74``), then the truck leave response,
return to farm, and sleep to D2.

Workflow (record → automate):

1. ``capture-entry`` — power-on, save a stable town-gate state for iteration.
2. ``record`` — human controller capture with live mask HUD (F5 saves).
3. ``replay`` — headless replay + mask / day assertions for skill extraction.

See ``docs/town_day1_recon.md``.

Examples::

    # One-shot: power-on, save entry state, open recorder
    uv run python -m harvest.scripts.town_day1_recon record --power-on

    # Iterative: reuse captured entry
    HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon capture-entry
    uv run python -m harvest.scripts.town_day1_recon record
    HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon replay \\
      --task town_day1_handoff --out recordings/town_day1_handoff_replay.json

    uv run python -m harvest.scripts.town_day1_recon checklist
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from harvest.paths import PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

from harvest.scripts.town_day1_recon_cmds import (
    cmd_auto,
    cmd_capture_entry,
    cmd_checklist,
    cmd_record,
    cmd_replay,
    cmd_status,
)
from harvest.scripts.town_day1_recon_lib import (
    D1_TOWN_BITS,
    DEFAULT_ENTRY_STATE,
    DEFAULT_RECORD_NAME,
    GATE_PIXEL,
    GATE_TOLERANCE_PX,
    STILL_TO_RECORD,
    TARGET_MASK,
    TOWN_TILEMAP,
    TRUCK_PIXEL,
    VERIFIED_ROUTES,
    TownSnapshot,
    decode_mask_bits,
    is_town_gate_entry,
    read_town_snapshot,
)

# Re-export public API used by tests and callers.
__all__ = [
    "D1_TOWN_BITS",
    "DEFAULT_ENTRY_STATE",
    "DEFAULT_RECORD_NAME",
    "GATE_PIXEL",
    "GATE_TOLERANCE_PX",
    "STILL_TO_RECORD",
    "TARGET_MASK",
    "TOWN_TILEMAP",
    "TRUCK_PIXEL",
    "VERIFIED_ROUTES",
    "TownSnapshot",
    "decode_mask_bits",
    "is_town_gate_entry",
    "main",
    "read_town_snapshot",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("checklist", help="Print recon checklist and suggested record order")

    cap = sub.add_parser("capture-entry", help="Power-on and save Spring D1 town-gate state")
    cap.add_argument("--name", default=DEFAULT_ENTRY_STATE, help="State name to write")
    cap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "town_day1_entry.json",
        help="JSON report path",
    )

    rec = sub.add_parser("record", help="Interactive controller recording with mask HUD")
    rec.add_argument("--name", default=DEFAULT_RECORD_NAME, help="Task name under tasks/")
    rec.add_argument(
        "--state",
        default=DEFAULT_ENTRY_STATE,
        help=f"Start state (default {DEFAULT_ENTRY_STATE}; ignored with --power-on)",
    )
    rec.add_argument(
        "--power-on",
        action="store_true",
        help="Clean boot via PowerOnStartTask before recording",
    )
    rec.add_argument(
        "--save-entry",
        default=DEFAULT_ENTRY_STATE,
        help="With --power-on, also pin this entry state after boot (empty string to skip)",
    )
    rec.add_argument("--scale", type=int, default=2, help="Window scale")

    rep = sub.add_parser("replay", help="Headless replay + mask/day assertions")
    rep.add_argument("--task", default=DEFAULT_RECORD_NAME, help="Task name or JSON path")
    rep.add_argument("--state", default=None, help="Override start state")
    rep.add_argument("--power-on", action="store_true", help="Power-on before replaying frames")
    rep.add_argument(
        "--require-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail unless end mask is 0x3F (default: true)",
    )
    rep.add_argument(
        "--require-day2",
        action="store_true",
        help="Also require calendar day >= 2 after replay",
    )
    rep.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "town_day1_handoff_replay.json",
        help="JSON report path",
    )

    st = sub.add_parser("status", help="Print mask/position for a state or power-on")
    st.add_argument("--state", default=DEFAULT_ENTRY_STATE)
    st.add_argument("--power-on", action="store_true")

    auto = sub.add_parser(
        "auto",
        help="Run precomputed D1 handoff (six talks → truck → farm → sleep)",
    )
    auto.add_argument("--state", default=DEFAULT_ENTRY_STATE, help="Start state")
    auto.add_argument("--power-on", action="store_true", help="Clean power-on entry")
    auto.add_argument(
        "--save-entry",
        default=DEFAULT_ENTRY_STATE,
        help="With --power-on, also pin entry state ('' to skip)",
    )
    auto.add_argument(
        "--sleep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include farmhouse sleep to D2 (default: true)",
    )
    auto.add_argument(
        "--require-full-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require mask 0x3F + truck/sleep (default true). "
        "Use --no-require-full-mask for Ann|Eve baseline progress runs.",
    )
    auto.add_argument("--timeout", type=int, default=90_000, help="Max handoff frames")
    auto.add_argument(
        "--save-end-state",
        default="Y1_Spring_D2_After_Town_Handoff",
        help="Optional end state name (empty string skips)",
    )
    auto.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "recordings" / "town_day1_auto.json",
        help="JSON report path",
    )

    return p


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # Normalize empty optional string flags.
    for key in ("save_entry", "save_end_state"):
        if getattr(args, key, None) == "":
            setattr(args, key, None)
    if args.command == "checklist":
        return cmd_checklist(args)
    if args.command == "capture-entry":
        return cmd_capture_entry(args)
    if args.command == "record":
        return cmd_record(args)
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "auto":
        return cmd_auto(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
