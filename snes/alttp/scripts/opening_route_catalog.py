#!/usr/bin/env python3
"""CLI for ALttP opening-route catalog status / validate / emit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp.opening_route.data import (  # noqa: E402
    DEFAULT_ARTIFACT,
    OpeningCheckpoint,
    opening_checkpoints,
)
from alttp.opening_route.validate import (  # noqa: E402
    build_catalog_artifact,
    correlate_boot_report,
    load_and_validate,
    validate_against_z3,
    write_artifact,
)
from alttp.paths import Z3_JSON_DATA_PIN  # noqa: E402
from alttp.z3_json_data import (  # noqa: E402
    Z3JsonData,
    Z3JsonDataError,
    Z3JsonDataNotFoundError,
    resolve_data_root,
    source_status,
)

def _load_boot_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"boot report must be a JSON object: {path}")
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    status = source_status(args.root)
    print(f"root:      {status.root}")
    print(f"present:   {status.present}")
    print(f"shape_ok:  {status.shape_ok}")
    print(f"revision:  {status.revision or '(unknown)'}")
    print(f"pin:       {status.pin}")
    if not status.present:
        print(
            "\nFetch with:\n"
            "  uv run python alttp/scripts/setup_z3_json_data.py"
        )
        return 1
    if not status.shape_ok:
        for issue in status.issues:
            print(f"  - {issue}")
        return 2
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"catalog_required_ok: {validation.required_ok}")
    print(f"opening_rooms:       {len(data.opening_route_rooms())}")
    print(f"opening_connections: {len(data.opening_route_connections())}")
    if validation.connections_optional_missing:
        print(
            "optional_missing_connections: "
            + ", ".join(validation.connections_optional_missing)
        )
    return 0 if validation.required_ok else 3


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"z3 root:   {data.root}")
    print(f"revision:  {data.revision or '(unknown)'}")
    print(f"pin:       {Z3_JSON_DATA_PIN}")
    print(f"required_ok: {validation.required_ok}")
    print(f"rooms:     present={len(validation.rooms_present)} "
          f"missing={len(validation.rooms_missing)}")
    print(f"nodes:     present={len(validation.nodes_present)} "
          f"missing={len(validation.nodes_missing)}")
    print(
        f"connections: present={len(validation.connections_present)} "
        f"missing_required={len(validation.connections_missing)} "
        f"optional_missing={len(validation.connections_optional_missing)}"
    )
    for r in validation.results:
        if r.ok:
            continue
        tag = "REQUIRED" if r.required else "optional"
        print(f"  [{tag}] {r.kind}: {r.name} — {r.detail}")
    if args.verbose:
        for r in validation.results:
            if r.ok:
                print(f"  [ok] {r.kind}: {r.name}")
    return 0 if validation.required_ok else 3


def _cmd_emit(args: argparse.Namespace) -> int:
    try:
        data, validation = load_and_validate(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        boot_report = _load_boot_report(args.from_boot_report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"failed to load boot report: {exc}", file=sys.stderr)
        return 2

    artifact = build_catalog_artifact(
        data,
        boot_report=boot_report,
        validation=validation,
    )
    out = Path(args.out) if args.out else DEFAULT_ARTIFACT
    write_artifact(artifact, out)
    print(f"wrote {out}")
    print(
        json.dumps(
            {
                "required_ok": validation.required_ok,
                "metrics": artifact["metrics"],
                "observed": artifact.get("observed"),
            },
            indent=2,
        )
    )
    if args.require_ok and not validation.required_ok:
        return 3
    return 0


def _cmd_list_checkpoints(args: argparse.Namespace) -> int:
    for cp in opening_checkpoints():
        print(f"{cp.id:28s}  role={cp.role:18s}  {cp.label}")
        if args.verbose:
            print(f"  gameplay: {json.dumps(cp.gameplay)}")
            if cp.z3_rooms:
                print(f"  z3_rooms: {', '.join(cp.z3_rooms)}")
            for c in cp.z3_connections:
                req = "required" if c.required else "optional"
                print(f"  conn[{req}]: {c.origin} -> {c.destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alttp.opening_route_catalog",
        description=(
            "Validate and emit the Link's House → castle opening-route "
            "catalog against local z3-json-data (no silent download)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="override z3-json-data root",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser(
        "status", help="checkout + catalog required-check summary"
    )
    p_status.set_defaults(func=_cmd_status)

    p_val = sub.add_parser(
        "validate",
        help="validate expected opening rooms/nodes/connections against z3",
    )
    p_val.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also print successful checks",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_emit = sub.add_parser(
        "emit",
        help="write structured catalog/progress JSON under alttp/",
    )
    p_emit.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help=f"output path (default: {DEFAULT_ARTIFACT})",
    )
    p_emit.add_argument(
        "--from-boot-report",
        type=Path,
        default=None,
        help=(
            "optional boot_to_castle.json to attach only real observed "
            "milestone facts (no invented intermediate screens)"
        ),
    )
    p_emit.add_argument(
        "--require-ok",
        action="store_true",
        help="exit non-zero if required z3 checks fail",
    )
    p_emit.set_defaults(func=_cmd_emit)

    p_list = sub.add_parser(
        "list-checkpoints", help="list curated opening checkpoints"
    )
    p_list.add_argument("-v", "--verbose", action="store_true")
    p_list.set_defaults(func=_cmd_list_checkpoints)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
