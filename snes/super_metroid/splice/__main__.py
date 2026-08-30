"""``python -m super_metroid.splice`` — preflight, cards, prepare, grade, assemble (no emulator)."""

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
from super_metroid.splice.cards import format_cards, generate_cards  # noqa: E402
from super_metroid.splice.errors import (  # noqa: E402
    AssembleError,
    GradeError,
    PrepareError,
    PreflightError,
    SchemaError,
    SpliceError,
)
from super_metroid.splice.manifest import load_manifest, manifest_from_product_chain  # noqa: E402
from super_metroid.splice.preflight import (  # noqa: E402
    format_preflight_summary,
    run_preflight,
)
from super_metroid.splice.prepare import prepare  # noqa: E402
from super_metroid.splice.schema import INTERVENTION_PROFILES  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m super_metroid.splice",
        description=(
            "Planning/verification over tips.play_hops. "
            "Artifact digest preflight, read-only task cards, fail-closed "
            "prepare, grade, and assemble (refuses to boot without a session factory)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    pre = sub.add_parser(
        "preflight",
        help="Snapshot artifact digests and report gaps (default: report-only)",
    )
    pre.add_argument("--task", type=Path, default=DEFAULT_TASK)
    pre.add_argument("--out", type=Path, default=DEFAULT_BOARD)
    pre.add_argument(
        "--write",
        action="store_true",
        help="Write the hop board with rewritten paths (never host-absolute)",
    )
    pre.add_argument("--json", action="store_true", help="Print JSON report only")
    pre.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if a selected product-chain artifact is missing",
    )
    pre.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    cards = sub.add_parser(
        "cards",
        help="Print immutable task cards from a route manifest (read-only)",
    )
    cards.add_argument("--task", help="Print one TASK_ID")
    cards.add_argument("--json", action="store_true", help="Print JSON cards only")
    cards.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    cards.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    cards.add_argument(
        "--chain",
        type=Path,
        default=DEFAULT_TASK,
        help="Product-chain task used when --manifest is omitted",
    )
    cards.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    prep = sub.add_parser(
        "prepare",
        help="Validate a task card before boot (report-only JSON; no emulator)",
    )
    prep.add_argument("task_id")
    prep.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    prep.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    prep.add_argument(
        "--chain",
        type=Path,
        default=DEFAULT_TASK,
        help="Product-chain task used when --manifest is omitted",
    )
    prep.add_argument("--no-live", action="store_true", help="Skip the live task tape")
    prep.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the task cannot be prepared",
    )
    grd = sub.add_parser(
        "grade",
        help="Replay+Join grade (refuses to boot without a runner hook)",
    )
    grd.add_argument("task_id")
    grd.add_argument("candidate")
    grd.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    grd.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    asm = sub.add_parser(
        "assemble",
        help="Assemble a route (refuses to boot without a session factory)",
    )
    asm.add_argument("route_id")
    asm.add_argument("--manifest", type=Path, help="Route-manifest JSON (skip board adapter)")
    asm.add_argument(
        "--profile",
        default="scaffold",
        choices=INTERVENTION_PROFILES,
        help="Planner selection profile (default: scaffold)",
    )
    return parser


def _print_error(exc: SpliceError, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(exc.to_dict(), indent=2))
        return
    print(str(exc), file=sys.stderr)
    missing = exc.details.get("missing") or []
    for label in missing[:20]:
        print(f"  - {label}", file=sys.stderr)


def _run_preflight(args: argparse.Namespace) -> int:
    try:
        report = run_preflight(
            args.task,
            include_live=not args.no_live,
            write=bool(args.write),
            out=args.out,
            strict=bool(args.strict),
        )
    except PreflightError as exc:
        _print_error(exc, as_json=bool(args.json))
        return 1
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_preflight_summary(report))
    return 0


def _run_cards(args: argparse.Namespace) -> int:
    try:
        if args.manifest is not None:
            manifest = load_manifest(args.manifest)
        else:
            manifest = manifest_from_product_chain(
                args.chain, include_live=not args.no_live
            )
        cards = generate_cards(manifest, profile=args.profile)
    except (PreflightError, SchemaError, OSError) as exc:
        if isinstance(exc, (PreflightError, SchemaError)):
            _print_error(exc, as_json=bool(args.json))
        else:
            print(str(exc), file=sys.stderr)
        return 1
    if args.task:
        cards = tuple(c for c in cards if c.task_id == args.task)
        if not cards:
            msg = f"task {args.task!r} not in manifest"
            if args.json:
                print(json.dumps({"error": "task_not_found", "message": msg}, indent=2))
            else:
                print(msg, file=sys.stderr)
            return 1
    if args.json:
        print(json.dumps([c.to_dict() for c in cards], indent=2))
    else:
        print(format_cards(cards))
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    try:
        prepared = prepare(
            args.task_id,
            manifest=args.manifest,
            profile=args.profile,
            chain=args.chain,
            include_live=not args.no_live,
        )
    except (PrepareError, SchemaError, OSError) as exc:
        if isinstance(exc, (PrepareError, SchemaError)):
            _print_error(exc, as_json=True)
        else:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))
        if isinstance(exc, PrepareError):
            return 1 if args.strict else 0
        return 1
    print(json.dumps(prepared.to_dict(), indent=2))
    return 0


def _run_grade(args: argparse.Namespace) -> int:
    # Dry planner: never boots. A runner hook is required to replay.
    err = GradeError(
        "grade refuses to boot without a runner hook",
        code="grade.runner",
        details={"task_id": args.task_id, "candidate": args.candidate},
    )
    _print_error(err, as_json=True)
    return 1


def _run_assemble(args: argparse.Namespace) -> int:
    # Dry planner: never boots. A session factory is required to play_hops.
    err = AssembleError(
        "assemble refuses to boot without a session factory",
        code="assemble.session",
        details={"route_id": args.route_id, "profile": args.profile},
    )
    _print_error(err, as_json=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "preflight":
        return _run_preflight(args)
    if args.cmd == "cards":
        return _run_cards(args)
    if args.cmd == "prepare":
        return _run_prepare(args)
    if args.cmd == "grade":
        return _run_grade(args)
    if args.cmd == "assemble":
        return _run_assemble(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
