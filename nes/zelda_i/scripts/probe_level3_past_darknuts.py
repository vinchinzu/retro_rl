"""Stub: Level3 past-Darknuts recon graduated to durable runners.

Raft path (Level3Darknuts → ADDR_RAFT) lives in::

    uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level3_raft.py --from-state Level3Darknuts --infinite-life

Post-Raft boss path (Level3Raft → Manhandla → TF)::

    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --trials 2

Library controllers:

- ``zelda_i.level3_dungeon.Level3RaftPathController``
- ``zelda_i.level3_boss_path.Level3BossPathController``

This stub forwards to the Raft runner (historical probe start state).
"""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from zelda_i.scripts.run_level3_raft import main as _raft_main
from zelda_i.scripts.run_level3_raft import run_once

__all__ = ["main", "run_once"]


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv in (["--help"], ["-h"]):
        print(__doc__)
        if argv in (["--help"], ["-h"]):
            return _raft_main(["--help"])
        print(
            "\n[stub] recon graduated → run_level3_raft.py "
            "(default: --infinite-life)\n"
        )
        return _raft_main(["--infinite-life"])
    print(
        "[stub] probe_level3_past_darknuts → run_level3_raft "
        "(Level3RaftPathController)",
        file=sys.stderr,
    )
    return _raft_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
