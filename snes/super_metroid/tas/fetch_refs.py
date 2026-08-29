"""Download vendored Super Metroid TAS movies into ``tas/ref/``.

Movies are gitignored (``.bk2`` / ``.lsmv`` / ``.smv``); re-fetch before
slice export. SMV files also get a BizHawk BK2 sidecar for later sync.

```bash
uv run python -m super_metroid.tas.fetch_refs
uv run python -m super_metroid.tas.fetch_refs --list
uv run python -m super_metroid.tas.fetch_refs --vanilla
uv run python -m super_metroid.tas.export_slices --catalog
```
"""

from __future__ import annotations

import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from super_metroid.tas.catalog import (
    MOVIES,
    REF_DIR,
    SKIPPED,
    MovieRef,
    fetchable,
    vanilla_fetchable,
)

_UA = "retro_rl-sm-fetch/1.0 (https://github.com; Super Metroid TAS adapt)"

_POST_SUFFIX = {
    "lsmv": (".lsmv",),
    "bk2": (".bk2",),
    "smv": (".smv",),
}


def _gunzip_if_needed(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    return data


def _unwrap_zip(data: bytes, suffixes: tuple[str, ...], *, nested_lsmv: bool) -> bytes:
    if data[:2] != b"PK":
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if nested_lsmv and "input" in names:
            return data
        nested = [n for n in names if n.lower().endswith(suffixes)]
        if nested_lsmv and not nested:
            nested = [n for n in names if n.lower().endswith(".lsmv")]
        if len(nested) == 1:
            return _gunzip_if_needed(zf.read(nested[0]))
        if nested_lsmv and "input" in names:
            return data
    return data


def unwrap_movie(data: bytes, ref: MovieRef) -> bytes:
    data = _gunzip_if_needed(data)
    nested = ref.postprocess == "unwrap_nested_lsmv"
    data = _unwrap_zip(data, _POST_SUFFIX[ref.kind], nested_lsmv=nested)
    if nested:
        data = _gunzip_if_needed(data)
        data = _unwrap_zip(data, (".lsmv",), nested_lsmv=True)
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


def _convert_smv_sidecar(smv_path: Path) -> Path | None:
    try:
        from super_metroid.tas.smv import parse_smv_env, write_bizhawk_bk2
    except ImportError as exc:
        print(f"skip BK2 sidecar: {exc}", file=sys.stderr)
        return None
    dest = smv_path.with_suffix(".bk2")
    movie = parse_smv_env(smv_path)
    write_bizhawk_bk2(movie, dest)
    print(f"bk2 {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
    return dest


def fetch_one(ref: MovieRef, *, force: bool = False) -> Path:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    dest = ref.path
    if dest.exists() and not force and dest.stat().st_size > 200:
        print(f"keep {dest} ({dest.stat().st_size} bytes)", file=sys.stderr)
        if ref.kind == "smv" and not dest.with_suffix(".bk2").exists():
            _convert_smv_sidecar(dest)
        return dest
    print(f"fetch {ref.url}", file=sys.stderr)
    data = unwrap_movie(_download(ref.url), ref)
    if not _looks_like_movie(data, ref.kind):
        preview = data[:120].decode("ascii", errors="replace")
        raise ValueError(f"download for {ref.filename} is not a movie: {preview!r}")
    dest.write_bytes(data)
    print(f"wrote {dest} ({len(data)} bytes)", file=sys.stderr)
    if ref.kind == "smv":
        _convert_smv_sidecar(dest)
    return dest


def fetch_all(
    *,
    force: bool = False,
    vanilla_only: bool = False,
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
                or (m.full_slice_id or "").lower() in wanted
            )
        ]
        missing = wanted - {
            m.filename.lower() for m in refs
        } - {m.stem.lower() for m in refs} - {
            (m.full_slice_id or "").lower() for m in refs
        }
        if missing:
            raise KeyError(f"unknown movie id(s): {sorted(missing)}")
    elif vanilla_only:
        refs = list(vanilla_fetchable())
    else:
        refs = list(fetchable())
    written: list[Path] = []
    for ref in refs:
        written.append(fetch_one(ref, force=force))
    return written


def _list_catalog(*, skipped: bool) -> None:
    rows = list(MOVIES) + (list(SKIPPED) if skipped else [])
    for movie in rows:
        flag = "FETCH" if movie.fetch else "SKIP"
        extra = movie.skip_reason or movie.notes
        print(
            f"{flag:5s} {movie.filename:32s} {movie.kind:4s}  "
            f"{movie.category:16s} {extra[:70]}"
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download even if present")
    p.add_argument(
        "--vanilla",
        action="store_true",
        help="Skip the Map Rando contest smoke BK2",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print catalog and exit",
    )
    p.add_argument(
        "--skipped",
        action="store_true",
        help="With --list, include explicit skips (hacks, watches)",
    )
    p.add_argument(
        "names",
        nargs="*",
        help="Optional filename / stem / slice id filter",
    )
    args = p.parse_args(argv)
    if args.list:
        _list_catalog(skipped=args.skipped)
        return 0
    fetch_all(force=args.force, vanilla_only=args.vanilla, names=args.names or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
