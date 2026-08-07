"""Stub: 0x1e bomb-N → Dodongo path graduated into ``level2_boss_path``.

Historical recon (rr-n5i): walk-UP after Goriya clear is solid despite doors
bit UP|DOWN=12; physical open is bomb-N @(120,101) via
``BOMB_WALL_1E_NORTH`` / ``Level2BombNorth1EController``.

Prefer the assisted runner::

    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --trials 1
    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --from-state Level2_0E

Library: ``zelda_i.level2_boss_path`` (bomb_north_1e_wall, run_boss_path).
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from zelda_i.level2_boss_path import (
    BOMB_STAND_1E,
    BOMB_WALL_1E,
    ROOM_0E,
    ROOM_1E,
    bomb_1e_open_predicate,
    make_bomb_north_1e_controller,
)
from zelda_i.scripts.run_level2_dodongo import run_once


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2Boom")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="l2_1e_up")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-state", action="store_true")
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print graduated path facts without running the emulator",
    )
    args = p.parse_args()

    if args.print_only:
        ctrl = make_bomb_north_1e_controller()
        print(
            f"room_1e=0x{ROOM_1E:02x} boss=0x{ROOM_0E:02x} "
            f"stand={BOMB_STAND_1E} face={BOMB_WALL_1E.face} "
            f"open={bomb_1e_open_predicate(from_room=ROOM_1E, to_room=ROOM_0E)} "
            f"ctrl_phase={ctrl.phase.name}"
        )
        print("delegate: run_level2_dodongo / level2_boss_path")
        return

    inf = not args.no_infinite_life
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            tag=tag,
            save_checkpoint=args.save_state and t == 0,
        )
        print(
            f"trial{t}: result={r.get('result')} ok={r.get('ok')} "
            f"reason={r.get('reason')} sc={(r.get('final') or {}).get('sc')}"
        )


if __name__ == "__main__":
    main()
