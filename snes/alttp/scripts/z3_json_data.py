#!/usr/bin/env python3
"""CLI for Z3 JSON data status / validate / list / show."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from alttp.paths import Z3_JSON_DATA_DIR  # noqa: E402
from alttp.z3_json_data import (  # noqa: E402
    Z3JsonData,
    Z3JsonDataError,
    Z3JsonDataNotFoundError,
    Z3JsonDataShapeError,
    Z3SourceStatus,
    resolve_data_root,
    source_status,
    validate_source_shape,
)

def _print_status(status: Z3SourceStatus) -> int:
    print(f"root:      {status.root}")
    print(f"present:   {status.present}")
    print(f"shape_ok:  {status.shape_ok}")
    print(f"revision:  {status.revision or '(unknown)'}")
    print(f"pin:       {status.pin}")
    if status.revision and status.revision != status.pin:
        print("note:      working tree revision differs from pin")
    if status.issues:
        print("issues:")
        for issue in status.issues:
            print(f"  - {issue}")
    if not status.present:
        print(
            "\nFetch with:\n"
            "  uv run python alttp/scripts/setup_z3_json_data.py"
        )
        return 1
    if not status.shape_ok:
        return 2
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    return _print_status(source_status(args.root))


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        path = validate_source_shape(args.root)
    except Z3JsonDataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except Z3JsonDataShapeError as exc:
        print(exc, file=sys.stderr)
        return 2
    rev = read_git_revision(path)
    print(f"OK: shape checks passed for {path}")
    print(f"revision: {rev or '(unknown)'}")
    print(f"pin:      {Z3_JSON_DATA_PIN}")
    return 0


def _load_or_exit(root: Path | None) -> Z3JsonData:
    try:
        return Z3JsonData.load(root)
    except Z3JsonDataError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


def _cmd_list_regions(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    rooms: Sequence[Z3Room]
    if args.opening:
        rooms = data.opening_route_rooms()
    elif args.query:
        rooms = data.find_rooms(args.query)
    else:
        rooms = data.rooms
    for room in rooms:
        print(
            f"{room.id:4d}  {room.room_type:12s}  {room.name}  "
            f"({len(room.nodes)} nodes)  [{room.source_path}]"
        )
    print(f"# {len(rooms)} room(s)", file=sys.stderr)
    return 0


def _cmd_list_connections(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    if args.query:
        conns = data.find_connections(args.query)
    elif args.opening:
        conns = data.opening_route_connections()
    else:
        conns = list(data.connections)
    for conn in conns:
        print(
            f"{conn.connection_type:8s}  {conn.origin}  ->  {conn.destination}"
        )
        if args.verbose and conn.description:
            print(f"         {conn.description}")
    print(f"# {len(conns)} connection(s)", file=sys.stderr)
    return 0


def _cmd_list_items(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    items = data.find_items(args.query) if args.query else list(data.items)
    for item in items:
        data_s = f"  data={item.data}" if item.data else ""
        print(f"{item.category:16s}  {item.name}{data_s}")
    print(f"# {len(items)} item(s)", file=sys.stderr)
    return 0


def _cmd_list_enemies(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    enemies = data.find_enemies(args.query) if args.query else list(data.enemies)
    for enemy in enemies:
        names = ", ".join(enemy.names)
        hp = f"  hp={enemy.hp}" if enemy.hp is not None else ""
        print(f"{enemy.id:4d}  {names}{hp}")
    print(f"# {len(enemies)} enemy/ies", file=sys.stderr)
    return 0


def _cmd_show_room(args: argparse.Namespace) -> int:
    data = _load_or_exit(args.root)
    rooms = data.rooms_by_name(args.name, exact=not args.fuzzy)
    if not rooms and not args.fuzzy:
        rooms = data.find_rooms(args.name)
    if not rooms:
        print(f"no room matching {args.name!r}", file=sys.stderr)
        return 1
    for room in rooms:
        print(f"id:        {room.id}")
        print(f"name:      {room.name}")
        print(f"type:      {room.room_type}")
        print(f"source:    {room.source_path}")
        print(f"nodes ({len(room.nodes)}):")
        for node in room.nodes:
            extra = ""
            if node.node_item:
                extra += f" item={node.node_item}"
            if node.node_address:
                extra += f" addr={node.node_address}"
            print(
                f"  {node.id:3d}  {node.node_type:8s}  {node.name}  "
                f"[{node.area}]{extra}"
            )
        conns = data.connections_for_room(room)
        if conns:
            print(f"connections ({len(conns)}):")
            for conn in conns:
                print(
                    f"  {conn.connection_type:8s}  "
                    f"{conn.origin}  ->  {conn.destination}"
                )
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alttp.z3_json_data",
        description=(
            "Inspect a local vg-json-data/z3-json-data checkout "
            "(no ROM/emulator required)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"override data root (default: {Z3_JSON_DATA_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="report checkout presence and shape")
    p_status.set_defaults(func=_cmd_status)

    p_validate = sub.add_parser(
        "validate", help="exit non-zero if missing or shape-invalid"
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_regions = sub.add_parser("list-regions", help="list rooms/regions")
    p_regions.add_argument("-q", "--query", help="substring filter on room name")
    p_regions.add_argument(
        "--opening",
        action="store_true",
        help="only rooms used by the title→castle opening route",
    )
    p_regions.set_defaults(func=_cmd_list_regions)

    p_conn = sub.add_parser("list-connections", help="list connections")
    p_conn.add_argument("-q", "--query", help="substring filter")
    p_conn.add_argument(
        "--opening",
        action="store_true",
        help="connections touching opening-route rooms",
    )
    p_conn.add_argument("-v", "--verbose", action="store_true")
    p_conn.set_defaults(func=_cmd_list_connections)

    p_items = sub.add_parser("list-items", help="list item catalog entries")
    p_items.add_argument("-q", "--query", help="substring filter on item name")
    p_items.set_defaults(func=_cmd_list_items)

    p_enemies = sub.add_parser("list-enemies", help="list enemies")
    p_enemies.add_argument("-q", "--query", help="substring filter on enemy name")
    p_enemies.set_defaults(func=_cmd_list_enemies)

    p_show = sub.add_parser("show-room", help="show one room and its connections")
    p_show.add_argument("name", help="exact room name (or substring with --fuzzy)")
    p_show.add_argument(
        "--fuzzy",
        action="store_true",
        help="substring match instead of exact name",
    )
    p_show.set_defaults(func=_cmd_show_room)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
