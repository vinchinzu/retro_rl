"""Stub: Level3 Manhandla recon graduated to durable runner.

LIVE path + combat live in ``zelda_i.level3_boss_path.Level3BossPathController``.
Use the thin runner instead of this recon dump::

    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --kill --poke-bombs 16
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --phase to5d
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --phase gate5d

Raft segment (Level3Darknuts → ADDR_RAFT)::

    uv run python nes/zelda_i/scripts/run_level3_raft.py --infinite-life --trials 2

This stub re-exports ``run_once`` / ``main`` from the durable runner for any
callers that still invoke the probe module path.
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

from zelda_i.scripts.run_level3_to_boss import main as _runner_main
from zelda_i.scripts.run_level3_to_boss import run_once

__all__ = ["main", "run_once"]


def main(argv: list[str] | None = None) -> int:
    """Forward to durable runner; print graduation note on bare --help."""
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv or argv == ["--help"] or argv == ["-h"]:
        print(__doc__)
        if argv in (["--help"], ["-h"]):
            return _runner_main(["--help"])
        print(
            "\n[stub] recon graduated → run_level3_to_boss.py "
            "(default args: --infinite-life)\n"
        )
        return _runner_main(["--infinite-life"])
    print(
        "[stub] probe_level3_manhandla → run_level3_to_boss "
        "(Level3BossPathController)",
        file=sys.stderr,
    )
    return _runner_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
