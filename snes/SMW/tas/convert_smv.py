"""CLI for the deterministic SMW SMV-to-BK2 conversion step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from SMW.tas.smv import parse_smv, write_bizhawk_bk2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smv", type=Path)
    parser.add_argument("rom", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    movie = parse_smv(args.smv)
    output = write_bizhawk_bk2(
        movie,
        args.output,
        rom_path=args.rom,
        max_frames=args.max_frames,
    )
    print(json.dumps({"movie": movie.summary(), "bk2": str(output)}, indent=2))


if __name__ == "__main__":
    main()
