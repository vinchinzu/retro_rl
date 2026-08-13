"""CLI: residual-profile a few short Level1_1 segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smb.paths import RECORDINGS_DIR
from smb.residual_harness import SEGMENTS, measure_segment, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "segments",
        nargs="*",
        default=list(SEGMENTS),
        help=f"subset of {', '.join(SEGMENTS)} (default: all)",
    )
    parser.add_argument(
        "--no-emu",
        action="store_true",
        help="roll the stepper only (profile will be unmeasured)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the report JSON instead of the text table",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RECORDINGS_DIR / "residual" / "level1_1_first_pass.json",
        help="write the report JSON here",
    )
    args = parser.parse_args()

    results = [
        measure_segment(name, run_emulator=not args.no_emu) for name in args.segments
    ]
    write_report(results, args.out)
    if args.json:
        print(json.dumps({"results": [item.to_dict() for item in results]}, indent=2))
    else:
        for item in results:
            print(item.summary())
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
