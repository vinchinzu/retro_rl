"""Download vendored Legend of Zelda TAS movies into ``tas/ref/``.

Default: **non-glitch** published movies only (no ``Heavy glitch abuse`` tag).
TASVideos has no formal glitchless any% branch for this game; the cleanest
complete FM2s are the **all-items** publications.

Movies are gitignored (``*.fm2``); re-fetch before import.

```bash
uv run python -m zelda_i.tas.fetch_refs
uv run python -m zelda_i.tas.fetch_refs --include-glitched   # optional
uv run python -m zelda_i.tas.import_fm2 --summary-only
```
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from zelda_i.paths import GAME_DIR

REF_DIR = GAME_DIR / "tas" / "ref"

# (filename, url, postprocess)
# Postprocess writes the final .fm2 bytes (unzip / gunzip as needed).

# No "Heavy glitch abuse" on TASVideos. Soft reset / damage boost / item-drop
# manip still appear; major route-breaking glitches and status-bar scroll
# abuse are not the focus (chatterbox explicitly skipped recorder-warp routes).
_NON_GLITCH: list[tuple[str, str, str]] = [
    (
        # Primary: published all-items, console-verified, FCEUX 2.6.1.
        # ROM USA Rev 1 (PRG1) — matches our integration SHA-1.
        # 114_913 frames / 31:52.07.
        "chatterbox_allitems_4767M.fm2",
        "https://tasvideos.org/4767M?handler=Download",
        "extract_fm2",
    ),
    (
        # Prior all-items (obsoleted). Still no heavy-glitch tag. PRG0.
        # Useful as second reference if PRG0 tooling is needed.
        "taseditor_allitems_2508M.fm2",
        "https://tasvideos.org/2508M?handler=Download",
        "extract_fm2",
    ),
]

# Tagged Heavy glitch abuse (or explicit game-end glitch). Opt-in only.
_GLITCHED: list[tuple[str, str, str]] = [
    (
        "lordtom_any_3232M.fm2",
        "https://tasvideos.org/3232M?handler=Download",
        "extract_fm2",
    ),
    (
        "lordtom_swordless_3289M.fm2",
        "https://tasvideos.org/3289M?handler=Download",
        "extract_fm2",
    ),
    (
        "chatterbox_2nd_4715M.fm2",
        "https://tasvideos.org/4715M?handler=Download",
        "extract_fm2",
    ),
    (
        # Explicit game-end glitch (FDS). Not useful for Clean graph nav.
        "taseditor_gameend_2868M.fm2",
        "https://tasvideos.org/2868M?handler=Download",
        "extract_fm2",
    ),
]


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


_POST = {
    "extract_fm2": _extract_fm2,
}


def fetch_all(
    *,
    force: bool = False,
    include_glitched: bool = False,
) -> list[Path]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    sources = list(_NON_GLITCH)
    if include_glitched:
        sources.extend(_GLITCHED)
    written: list[Path] = []
    for name, url, post in sources:
        dest = REF_DIR / name
        if dest.exists() and not force and dest.stat().st_size > 1000:
            print(f"keep {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
            written.append(dest)
            continue
        print(f"fetch {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        data = _POST[post](data)
        dest.write_bytes(data)
        print(f"wrote {dest} ({len(data)} bytes)", file=sys.stderr)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    p.add_argument(
        "--include-glitched",
        action="store_true",
        help="Also fetch Heavy-glitch / game-end-glitch movies (any%%, swordless, …)",
    )
    args = p.parse_args(argv)
    fetch_all(force=args.force, include_glitched=args.include_glitched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
