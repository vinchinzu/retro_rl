"""Download vendored Super Mario Bros. TAS movies into ``tas/ref/``.

Movies are gitignored (``*.fm2`` / ``*.bk2``); re-fetch before import.

```bash
uv run python -m smb.tas.fetch_refs
uv run python -m smb.tas.fetch_refs --force
uv run python -m smb.scripts.convert_fm2
```
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from smb.paths import GAME_DIR

REF_DIR = GAME_DIR / "tas" / "ref"

_UA = "retro_rl-smb-fetch/1.0 (https://github.com; SMB TAS adapt)"

# (filename, url)
_MOVIES: tuple[tuple[str, str], ...] = (
    (
        "happylee_warps_1715M.fm2",
        "https://tasvideos.org/1715M?handler=Download",
    ),
    (
        "happylee_mars608_warpless_3728M.fm2",
        "https://tasvideos.org/3728M?handler=Download",
    ),
)


def _gunzip_if_needed(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _extract_fm2(data: bytes) -> bytes:
    """Unwrap zip/gzip TASVideos downloads to raw ``.fm2`` text bytes."""
    data = _gunzip_if_needed(data)
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".fm2")]
            if not names:
                names = zf.namelist()
            if not names:
                raise ValueError("zip has no members")
            return zf.read(names[0])
    return data


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def fetch_all(*, force: bool = False) -> list[Path]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url in _MOVIES:
        dest = REF_DIR / name
        if dest.exists() and not force and dest.stat().st_size > 1000:
            print(f"keep {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
            written.append(dest)
            continue
        print(f"fetch {url}", file=sys.stderr)
        dest.write_bytes(_extract_fm2(_download(url)))
        print(f"wrote {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    args = p.parse_args(argv)
    fetch_all(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
