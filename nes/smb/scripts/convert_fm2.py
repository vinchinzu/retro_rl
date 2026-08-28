"""Convert vendored FCEUX FM2 movies to BizHawk NesHawk BK2.

Same mapping as the on-disk any% warps conversion
(``happylee_warps_1715M.fm2.bk2``): headerless iNES hashes, UDLRSsBA log,
Reset ``r`` on FM2 command bit 1. Does not invoke EmuHawk.

```bash
uv run python -m smb.scripts.convert_fm2
uv run python -m smb.scripts.convert_fm2 nes/smb/tas/ref/happylee_mars608_warpless_3728M.fm2
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smb.paths import GAME_DIR
from smb.tas.bk2 import parse_bk2, write_neshawk_bk2
from smb.tas.fm2 import parse_fm2

REF_DIR = GAME_DIR / "tas" / "ref"
_DEFAULT_MOVIES = (
    REF_DIR / "happylee_warps_1715M.fm2",
    REF_DIR / "happylee_mars608_warpless_3728M.fm2",
    REF_DIR / "flamexx_warps_rta_4_54_099.fm2",
)


def _convert_one(fm2: Path, out: Path | None) -> dict[str, object]:
    movie = parse_fm2(fm2)
    dest = write_neshawk_bk2(fm2, out)
    bk = parse_bk2(dest)
    return {
        "fm2": str(fm2),
        "bk2": str(dest),
        "fm2_frames": movie.num_frames,
        "bk2_frames": bk.num_frames,
        "author": movie.author,
        "lr_frames": movie.lr_frames,
        "matched_frames": movie.num_frames == bk.num_frames
        and movie.frames == bk.frames,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "fm2",
        type=Path,
        nargs="?",
        default=None,
        help="FM2 to convert (default: vendored warps + warpless + flamexx if present)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="BK2 path (default: <fm2>.bk2)",
    )
    args = p.parse_args(argv)

    targets = [args.fm2] if args.fm2 is not None else list(_DEFAULT_MOVIES)
    reports: list[dict[str, object]] = []
    for fm2 in targets:
        if fm2 is None or not fm2.exists():
            print(f"skip missing {fm2}", file=sys.stderr)
            continue
        report = _convert_one(fm2, args.out if args.fm2 is not None else None)
        reports.append(report)
        print(json.dumps(report, indent=2))
        print(
            f"wrote {report['bk2']} frames={report['bk2_frames']} "
            f"match={report['matched_frames']}",
            file=sys.stderr,
        )
    if not reports:
        print("no FM2 movies to convert", file=sys.stderr)
        return 1
    if any(not r.get("matched_frames") for r in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
