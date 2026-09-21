"""Download vendored ALttP TAS movies into ``tas/ref/``.

Movies are gitignored (``.bk2`` / ``.lsmv`` / ``.smv``); re-fetch before
slice export.

```bash
uv run python -m alttp.tas.fetch_refs
uv run python -m alttp.tas.fetch_refs --list
uv run python -m alttp.tas.fetch_refs --force fmp_geg_3898M.lsmv
```
"""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from alttp.tas.catalog import MOVIES, REF_DIR, SKIPPED, MovieRef, fetchable

_UA = "retro_rl-alttp-fetch/1.0 (https://github.com; ALttP TAS adapt)"

_POST_SUFFIX = {
    "lsmv": (".lsmv",),
    "bk2": (".bk2",),
    "smv": (".smv",),
}


def _gunzip_if_needed(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _unwrap_zip(data: bytes, suffixes: tuple[str, ...]) -> bytes:
    if data[:2] != b"PK":
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if "input" in names or "Input Log.txt" in names:
            return data
        nested = [n for n in names if n.lower().endswith(suffixes)]
        if len(nested) == 1:
            return _gunzip_if_needed(zf.read(nested[0]))
    return data


def unwrap_movie(data: bytes, ref: MovieRef) -> bytes:
    """Unwrap TASVideos gzip / publication-zip wrappers to the movie bytes."""
    data = _gunzip_if_needed(data)
    data = _unwrap_zip(data, _POST_SUFFIX[ref.kind])
    if ref.kind == "lsmv":
        data = _gunzip_if_needed(data)
        data = _unwrap_zip(data, (".lsmv",))
    return data


def _looks_like_movie(data: bytes, kind: str) -> bool:
    if data[:2] in (b"PK", b"\x1f\x8b"):
        return True
    if kind == "smv":
        return data[:4] == b"SMV\x1a"
    return False


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def fetch_one(ref: MovieRef, *, force: bool = False) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    dest = ref.path
    if dest.exists() and not force and dest.stat().st_size > 200:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if ref.sha256 and digest != ref.sha256:
            raise ValueError(
                f"sha256 mismatch for existing {dest.name}: {digest} != {ref.sha256}"
            )
        print(f"keep {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
        return dest
    print(f"fetch {ref.url}", file=sys.stderr)
    data = unwrap_movie(_download(ref.url), ref)
    if not _looks_like_movie(data, ref.kind):
        preview = data[:120].decode("ascii", errors="replace")
        raise ValueError(f"download for {ref.filename} is not a movie: {preview!r}")
    digest = hashlib.sha256(data).hexdigest()
    if ref.sha256 and digest != ref.sha256:
        raise ValueError(
            f"sha256 mismatch for {ref.filename}: {digest} != {ref.sha256}"
        )
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data)} bytes)", file=sys.stderr)
    return dest


def fetch_all(
    *,
    force: bool = False,
    names: list[str] | None = None,
) -> list[Path]:
    if names:
        wanted = {n.lower() for n in names}
        refs = [
            m
            for m in MOVIES
            if m.fetch
            and (
                m.filename.lower() in wanted
                or m.stem.lower() in wanted
            )
        ]
        have = {m.filename.lower() for m in refs} | {m.stem.lower() for m in refs}
        missing = wanted - have
        if missing:
            raise KeyError(f"unknown movie id(s): {sorted(missing)}")
    else:
        refs = list(fetchable())
    return [fetch_one(ref, force=force) for ref in refs]


def _list_catalog(*, skipped: bool) -> None:
    rows = list(MOVIES) + (list(SKIPPED) if skipped else [])
    for movie in rows:
        flag = "FETCH" if movie.fetch else "SKIP"
        extra = movie.skip_reason or movie.notes
        print(
            f"{flag:5s} {movie.filename:36s} {movie.kind:4s}  "
            f"{movie.category:18s} {extra[:70]}"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    p.add_argument("--list", action="store_true", help="Print catalog and exit")
    p.add_argument(
        "--skipped",
        action="store_true",
        help="With --list, include explicit skips (watches, lua)",
    )
    p.add_argument(
        "names",
        nargs="*",
        help="Optional filename / stem filter",
    )
    args = p.parse_args(argv)
    if args.list:
        _list_catalog(skipped=args.skipped)
        return 0
    fetch_all(force=args.force, names=args.names or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
