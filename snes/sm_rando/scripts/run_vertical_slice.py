"""Run the audited natural Landing → Pit SolverSession proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from sm_rando.paths import RECORDINGS_DIR
from sm_rando.vertical_slice import run_real_vertical_slice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "vertical_slice.run.json",
    )
    args = parser.parse_args(argv)
    result, manifest = run_real_vertical_slice(args.output)
    print(
        f"status={result.status.value} replans={result.replans} "
        f"edges={','.join(result.completed_edges)} frames={manifest.frames} "
        f"manifest={args.output}"
    )
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
