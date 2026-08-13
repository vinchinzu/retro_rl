"""CLI for practice-hack repertoire: route, policy board, recovery."""

from __future__ import annotations

import argparse
import json
import sys

from super_metroid.practice_repertoire.board import (
    gap_report,
    graduation_status,
    mapped_sessions,
    policy_board,
    session_work_card,
)
from super_metroid.practice_repertoire.catalog import (
    PRODUCT_CATEGORY,
    RepertoireSession,
    _parse_hex_int,
    categories,
    route_sessions,
    sessions,
)
from super_metroid.practice_repertoire.spine import (
    hop_key_for_session,
    product_route_edges,
    recover_session,
    route_edge,
)

_DOC = """Practice-hack preset repertoire — shared spine for human + bot work.

Catalog: maps/practice_repertoire.json (regenerate with
scripts/export/practice_repertoire.py).

  uv run python -m super_metroid.practice_repertoire --route
  uv run python -m super_metroid.practice_repertoire --policy-board
  uv run python -m super_metroid.practice_repertoire --stitch kpdr25/crateria/morph
  uv run python -m super_metroid.practice_repertoire --recovery 0x9E9F --items 0x0004
"""


def _print_session(s: RepertoireSession) -> None:
    bits = [s.id, s.name]
    if s.room_hex:
        bits.append(s.room_hex)
    if s.items is not None:
        bits.append(f"items=0x{s.items:04X}")
    if s.beams is not None:
        bits.append(f"beams=0x{s.beams:04X}")
    if s.x is not None and s.y is not None:
        bits.append(f"xy=({s.x},{s.y})")
    g = graduation_status(s)
    if g != "none":
        bits.append(f"[{g}]")
    m = s.product_map()
    if m:
        bits.append(f"→ {m.get('start_preset', m)}")
    print("  ".join(str(b) for b in bits))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=_DOC)
    p.add_argument("--list-categories", action="store_true")
    p.add_argument("--category", default=None, help="filter by category id")
    p.add_argument("--area", default=None, help="filter by area slug")
    p.add_argument("--session", default=None, help="show one session work card")
    p.add_argument("--mapped", action="store_true", help="product map only")
    p.add_argument("--gaps", action="store_true", help="coverage / graduation report")
    p.add_argument(
        "--route",
        action="store_true",
        help="ordered route for category",
    )
    p.add_argument(
        "--stitch",
        metavar="SESSION_ID",
        default=None,
        help="route edge from SESSION_ID → next (alias of route-edge)",
    )
    p.add_argument(
        "--stitch-board",
        action="store_true",
        help="all route edges for category",
    )
    p.add_argument(
        "--recovery",
        metavar="ROOM",
        default=None,
        help="recovery pin for room id (e.g. 0x9E9F)",
    )
    p.add_argument(
        "--items",
        default=None,
        help="inventory mask for --recovery (e.g. 0x1004)",
    )
    p.add_argument(
        "--policy-board",
        action="store_true",
        help="policy tune/graduate board for category",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    if args.list_categories:
        cats = categories()
        if args.json:
            print(json.dumps(cats, indent=2))
        else:
            for c in cats:
                mark = " ★" if c["id"] == PRODUCT_CATEGORY else ""
                print(
                    f"{c['menu_index']:2d}  {c['id']:16s}  "
                    f"{c['session_count']:4d} sessions  {c['name']}{mark}"
                )
        return 0

    if args.session:
        card = session_work_card(args.session)
        if args.json:
            print(json.dumps(card, indent=2))
        else:
            for k, v in card.items():
                if isinstance(v, (dict, list)):
                    print(f"{k}:")
                    print(json.dumps(v, indent=2))
                else:
                    print(f"{k}: {v}")
        return 0

    if args.gaps:
        print(json.dumps(gap_report(args.category or PRODUCT_CATEGORY), indent=2))
        return 0

    if args.stitch is not None:
        edge = route_edge(args.stitch)
        if edge is None:
            print(f"no next session after {args.stitch}", file=sys.stderr)
            return 1
        payload = edge.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for k, v in payload.items():
                print(f"{k}: {v}")
        return 0

    if args.stitch_board:
        board = product_route_edges(args.category or PRODUCT_CATEGORY)
        payload = [e.to_dict() for e in board]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"{len(board)} route edges")
            for e in board:
                print(f"{e.from_session} → {e.to_session}  {e.hop_key}")
        return 0

    if args.recovery is not None:
        room = _parse_hex_int(args.recovery)
        if room is None:
            print("bad --recovery room", file=sys.stderr)
            return 2
        items = _parse_hex_int(args.items) if args.items is not None else None
        hint = recover_session(room, items, category=args.category or PRODUCT_CATEGORY)
        if hint is None:
            print("no repertoire session for room", file=sys.stderr)
            return 1
        payload = hint.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for k, v in payload.items():
                print(f"{k}: {v}")
        return 0

    if args.policy_board:
        cards = policy_board(args.category or PRODUCT_CATEGORY)
        payload = [c.to_dict() for c in cards]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for c in cards:
                room = f"0x{c.room_id:04X}" if c.room_id is not None else "?"
                print(
                    f"[{c.grade:22s}] {c.session_id:42s} {room}  "
                    f"policies={len(c.existing_policies)}"
                )
        return 0

    if args.mapped:
        rows = mapped_sessions()
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "id": s.id,
                            "name": s.name,
                            "room_hex": s.room_hex,
                            "grade": graduation_status(s),
                            **meta,
                        }
                        for s, meta in rows
                    ],
                    indent=2,
                )
            )
        else:
            for s, meta in rows:
                living = s.living_state_path()
                flag = "OK" if living and living.is_file() else "—"
                print(
                    f"[{flag}] [{graduation_status(s):22s}] "
                    f"{s.id:42s} → {meta.get('start_preset', '')}"
                )
        return 0

    if args.route:
        cat = args.category or PRODUCT_CATEGORY
        rows = route_sessions(cat)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "index": s.route_index,
                            "id": s.id,
                            "name": s.name,
                            "room_hex": s.room_hex,
                            "items": s.items,
                            "grade": graduation_status(s),
                            "hop_key": hop_key_for_session(s),
                        }
                        for s in rows
                    ],
                    indent=2,
                )
            )
        else:
            print(f"route {cat}: {len(rows)} sessions")
            for s in rows:
                print(f"{s.route_index:3d}  ", end="")
                _print_session(s)
        return 0

    cat = args.category or PRODUCT_CATEGORY
    rows = sessions(category=cat, area=args.area)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": s.id,
                        "name": s.name,
                        "area": s.area,
                        "room_hex": s.room_hex,
                        "items": s.items,
                        "beams": s.beams,
                        "grade": graduation_status(s),
                    }
                    for s in rows
                ],
                indent=2,
            )
        )
    else:
        print(f"{cat}: {len(rows)} sessions")
        for s in rows:
            _print_session(s)
    return 0
