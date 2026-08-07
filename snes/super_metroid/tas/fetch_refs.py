"""Download vendored Super Metroid TAS movies into ``tas/ref/``.

Movies are gitignored (``.bk2`` / ``.lsmv``); re-fetch before slice export.

```bash
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.export_slices --finish
```
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from super_metroid.tas.slice import REF_DIR

# (filename, url, postprocess)
_SOURCES: list[tuple[str, str, str]] = [
    (
        "sniq_any_3653M.lsmv",
        "https://tasvideos.org/3653M?handler=Download",
        "unwrap_nested_lsmv",
    ),
    (
        "sniq_100p.bk2",
        "https://tasvideos.org/UserFiles/Info/55928342467251616?handler=Download",
        "gunzip_if_needed",
    ),
    (
        "sniq_any_wip.lsmv",
        "https://tasvideos.org/UserFiles/Info/36208532992045040?handler=Download",
        "gunzip_if_needed",
    ),
    (
        "moozooh_smtc4.bk2",
        "https://tasvideos.org/UserFiles/Info/638502075337523909?handler=Download",
        "gunzip_if_needed",
    ),
]


def _gunzip_if_needed(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _unwrap_nested_lsmv(data: bytes) -> bytes:
    data = _gunzip_if_needed(data)
    if data[:2] != b"PK":
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if "input" in names:
            return data
        nested = [n for n in names if n.lower().endswith(".lsmv")]
        if len(nested) == 1:
            inner = zf.read(nested[0])
            return _gunzip_if_needed(inner)
    return data


_POST = {
    "gunzip_if_needed": _gunzip_if_needed,
    "unwrap_nested_lsmv": _unwrap_nested_lsmv,
}


def fetch_all(*, force: bool = False) -> list[Path]:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, url, post in _SOURCES:
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
    args = p.parse_args(argv)
    fetch_all(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
