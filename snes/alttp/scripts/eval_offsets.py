"""Score one controller at RNG offsets 0..11. One emulator at a time.

Each offset is one in-process ``run_controller`` call with ``idle=N``.
The emulator is deterministic, so one offset has one outcome. Do not score
a fight on a single tape.

HP lost is the sum of ``$F36D`` drops during the controller (8 = one heart).

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/eval_offsets.py \\
        CastleMain alttp.opening_route.room_engine:run_room_edge room_61 west_to_0x60
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys

from alttp.scripts.stage_replay import run_controller

_OFFSETS = range(12)


def _run_one(
    pin: str,
    target: str,
    call_args: list[str],
    offset: int,
    last_heart: bool,
) -> tuple[str, str, str, str]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            row = run_controller(
                pin,
                target,
                call_args,
                idle=offset,
                last_heart=last_heart,
            )
    except (Exception, SystemExit) as exc:
        tail = " | ".join(buf.getvalue().splitlines()[-20:])
        detail = str(exc) or type(exc).__name__
        if tail:
            detail = f"{detail} | {tail}" if detail else tail
        return ("ERR", "-", "-", detail)
    result = row.result
    passed = "1" if result.ok else "0"
    return (passed, str(int(result.frames)), str(int(row.hp_lost)), row.hits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pin", help="save-state name, or a .state path")
    parser.add_argument("target", help="module:Name controller")
    parser.add_argument("call_args", nargs="*", help="extra positionals forwarded to the controller")
    parser.add_argument(
        "--last-heart",
        action="store_true",
        help="forward last-heart (what-if one heart; not a route result)",
    )
    args = parser.parse_args(argv)
    extra = " ".join(args.call_args)
    print(
        f"pin={args.pin} controller={args.target} args={extra or '-'} "
        f"offsets=0..11"
    )
    print("hp_lost is the sum of $F36D drops (8 = one heart)")
    print(f"{'offset':<8}{'pass':<8}{'frames':<10}{'hp_lost':<10}hits_by_cause")
    failed = False
    for offset in _OFFSETS:
        passed, frames, hp_lost, hits = _run_one(
            args.pin, args.target, list(args.call_args), offset, args.last_heart
        )
        if passed == "ERR":
            failed = True
        print(f"{offset:<8}{passed:<8}{frames:<10}{hp_lost:<10}{hits}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
