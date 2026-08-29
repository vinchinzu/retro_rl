"""Run the audited natural FirstPlay → uncle SolverSession proof."""

from __future__ import annotations

import argparse
from pathlib import Path

from alttp_rando.house_to_uncle_session import (
    HOUSE_TO_UNCLE_SESSION_MANIFEST,
    run_real_house_to_uncle_session,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=HOUSE_TO_UNCLE_SESSION_MANIFEST,
    )
    args = parser.parse_args(argv)
    result, manifest = run_real_house_to_uncle_session(args.output)
    print(
        f"status={result.status.value} replans={result.replans} "
        f"edges={','.join(result.completed_edges)} frames={manifest.frames} "
        f"manifest={args.output}"
    )
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
