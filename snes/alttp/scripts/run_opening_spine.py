"""Thin alias of ``run_spine``. Names live here so callers can monkeypatch them.

``run_spine.py`` is the spine CLI. This module keeps the old import path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from alttp.opening_route.full_tip import run_to_verified_tip
from alttp.paths import RECORDINGS_DIR
from alttp.scripts.run_spine import (
    DEFAULT_THROUGH,
    SPINE_THROUGH,
    build_parser as _spine_build_parser,
    dispatch,
    leftover_path,
)
from alttp.spine.ledger import RunLedger
from alttp.scripts.run_spine import write_leftover as _write_leftover
from alttp.spine.hops import ROOM_01_BLOCKER, ROOM_72_BLOCKER, ZELDA_BLOCKER
from alttp.startup import build_boot_env

VERIFIED_TIP_RUN = RECORDINGS_DIR / "verified_tip_run.json"

__all__ = [
    "DEFAULT_THROUGH",
    "ROOM_01_BLOCKER",
    "ROOM_72_BLOCKER",
    "SPINE_THROUGH",
    "VERIFIED_TIP_RUN",
    "ZELDA_BLOCKER",
    "build_boot_env",
    "build_parser",
    "leftover_path",
    "main",
    "run_opening_spine",
    "run_to_verified_tip",
    "write_leftover",
]


def write_leftover(path: Path, payload: dict[str, Any]) -> Path:
    """Refuse ``VERIFIED_TIP_RUN`` on this module (tests patch the name)."""
    return _write_leftover(path, payload, banned=VERIFIED_TIP_RUN)


def build_parser():
    """Spine parser plus ``--ledger``."""
    parser = _spine_build_parser()
    parser.add_argument(
        "--ledger",
        action="store_true",
        help="Print rooms, hits, and inventory writes and add them to the JSON.",
    )
    return parser


def ledger_fields(ledger: RunLedger) -> dict[str, Any]:
    book = ledger.report()
    return {
        "rooms": book["rooms"],
        "hits": book["hits"],
        "writes": book["writes"],
        "write_count": len(book["writes"]),
    }


def add_ledger(payload: dict[str, Any], ledger: RunLedger) -> None:
    fields = ledger_fields(ledger)
    payload.update(fields)
    nested = payload.get("report")
    if isinstance(nested, dict):
        nested.update(fields)


def print_ledger(ledger: RunLedger) -> None:
    fields = ledger_fields(ledger)
    print("rooms:")
    for room, row in fields["rooms"].items():
        print(
            f"  {room} visits={row['visits']} frames={row['frames']} "
            f"max={row['max_visit']} flutters={row['flutters']}"
        )
    print(f"hits={len(fields['hits'])}")
    for hit in fields["hits"]:
        print(
            f"  frame={hit['frame']} room={hit['room']} damage={hit['damage']} "
            f"cause={hit['cause']} distance={hit['distance']}"
        )
    print(f"writes={fields['write_count']}")
    for row in fields["writes"]:
        print(
            f"  {row['field']} {row['from']}->{row['to']} "
            f"frame={row['frame']} room={row['room']}"
        )


def run_opening_spine(
    through: str,
    *,
    env: object | None = None,
    close: bool = True,
    run_tip_fn: Any = None,
) -> dict[str, Any]:
    """Dispatch one spine stop. Fail-closed stubs never boot the ROM."""
    fn = run_to_verified_tip if run_tip_fn is None else run_tip_fn
    return dispatch(through, env=env, close=close, run_tip_fn=fn)


def main(argv: list[str] | None = None) -> int:
    from alttp.scripts.run_spine import main as spine_main

    raw = list(sys.argv[1:] if argv is None else argv)
    use_ledger = "--ledger" in raw
    forwarded = [arg for arg in raw if arg != "--ledger"]
    holder: dict[str, RunLedger | None] = {"ledger": None}

    def tip(env: object, **kwargs: Any) -> Any:
        if holder["ledger"] is None:
            holder["ledger"] = RunLedger()
            holder["ledger"].attach(env)
        return run_to_verified_tip(env, **kwargs)

    def writer(path: Path, payload: dict[str, Any]) -> Path:
        ledger = holder["ledger"]
        if ledger is not None:
            add_ledger(payload, ledger)
            print_ledger(ledger)
        return write_leftover(path, payload)

    return spine_main(
        forwarded,
        banned_report=VERIFIED_TIP_RUN,
        boot_fn=build_boot_env,
        tip_fn=tip if use_ledger else run_to_verified_tip,
        leftover_writer=writer if use_ledger else write_leftover,
    )


if __name__ == "__main__":
    raise SystemExit(main())
