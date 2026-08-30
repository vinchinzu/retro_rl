"""``python -m super_metroid.splice preflight`` — digest snapshot, no emulator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SNES = _ROOT / "snes"
for _p in (_ROOT, _SNES):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from retro_harness.repo import ensure_import_paths  # noqa: E402

ensure_import_paths(root=_ROOT)

from super_metroid.human_tape.product_chain import DEFAULT_BOARD, DEFAULT_TASK  # noqa: E402
from super_metroid.splice.errors import PreflightError  # noqa: E402
from super_metroid.splice.preflight import (  # noqa: E402
    format_preflight_summary,
    run_preflight,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m super_metroid.splice",
        description=(
            "Planning/verification over tips.play_hops. "
            "Phase 0: artifact digest preflight (no emulator)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    pre = sub.add_parser(
        "preflight",
        help="Snapshot artifact digests and report gaps (default: report-only)",
    )
    pre.add_argument("--task", type=Path, default=DEFAULT_TASK)
    pre.add_argument("--out", type=Path, default=DEFAULT_BOARD)
    pre.add_argument("--write", action="store_true", help="Write the product-chain board")
    pre.add_argument("--json", action="store_true", help="Print JSON report only")
    pre.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if a selected product-chain artifact is missing",
    )
    pre.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd != "preflight":
        parser.print_help()
        return 2
    try:
        report = run_preflight(
            args.task,
            include_live=not args.no_live,
            write=bool(args.write),
            out=args.out,
            strict=bool(args.strict),
        )
    except PreflightError as exc:
        if args.json:
            print(json.dumps(exc.to_dict(), indent=2))
        else:
            print(str(exc), file=sys.stderr)
            missing = exc.details.get("missing") or []
            for label in missing[:20]:
                print(f"  - {label}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_preflight_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
