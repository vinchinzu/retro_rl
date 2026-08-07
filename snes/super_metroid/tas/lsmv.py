"""lsnes ``.lsmv`` movie import for Super Metroid TAS seeds.

LSMV is a zip archive. Text member ``input`` holds one line per frame::

    F.|BYsSudlrAXLR

Some cores prefix port fields (observed on Sniq any% #3653M)::

    F. 0 0|BYsSudlrAXLR

Button field order matches env logical order
``[B, Y, Select, Start, Up, Down, Left, Right, A, X, L, R]``.

Spec: https://tasvideos.org/EmulatorResources/Lsnes/LSMV
"""

from __future__ import annotations

import gzip
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import SNES_BUTTON_NAMES

# LSMV gamepad order is identical to stable-retro env order.
_LSMV_LEN = 12
assert _LSMV_LEN == SNES_ACTION_SIZE
assert len(SNES_BUTTON_NAMES) == SNES_ACTION_SIZE


@dataclass
class LsmvMovie:
    """Parsed lsnes movie with SNES-12 frames in env order."""

    path: Path
    meta: dict[str, str] = field(default_factory=dict)
    frames: list[list[int]] = field(default_factory=list)
    raw_p1: list[str] = field(default_factory=list)
    resets: list[bool] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def author(self) -> str | None:
        return self.meta.get("authors") or self.meta.get("author")

    @property
    def gamename(self) -> str | None:
        return self.meta.get("gamename")

    def summary(self) -> dict[str, Any]:
        first_nz = next((i for i, fr in enumerate(self.frames) if any(fr)), None)
        return {
            "path": str(self.path),
            "format": "lsmv",
            "num_frames": self.num_frames,
            "author": self.author,
            "gamename": self.gamename,
            "gametype": self.meta.get("gametype"),
            "coreversion": self.meta.get("coreversion"),
            "rerecords": self.meta.get("rerecords"),
            "first_nonzero_frame": first_nz,
            "reset_frames": sum(1 for r in self.resets if r),
        }


def _read_zip_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def _open_lsmv_zip(path: Path) -> zipfile.ZipFile:
    """Open LSMV; unwrap nested single-member zip downloads from TASVideos."""
    raw = _read_zip_bytes(path)
    if raw[:2] != b"PK":
        raise ValueError(f"not a zip LSMV: {path}")
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    if "input" in names:
        return zf
    nested = [n for n in names if n.lower().endswith(".lsmv")]
    if len(nested) == 1:
        inner = zf.read(nested[0])
        zf.close()
        if inner[:2] == b"\x1f\x8b":
            inner = gzip.decompress(inner)
        return zipfile.ZipFile(io.BytesIO(inner))
    zf.close()
    raise ValueError(f"LSMV has no 'input' member: {path} members={names}")


def _parse_button_field(field: str) -> list[int]:
    field = field[:_LSMV_LEN].ljust(_LSMV_LEN, ".")
    action = [0] * SNES_ACTION_SIZE
    for i, ch in enumerate(field):
        if ch != ".":
            action[i] = 1
    return action


def parse_lsmv(path: Path | str) -> LsmvMovie:
    """Parse an ``.lsmv`` (or gzip/nested-zip wrapper) into SNES-12 frames."""
    path = Path(path)
    meta: dict[str, str] = {}
    frames: list[list[int]] = []
    raw_p1: list[str] = []
    resets: list[bool] = []

    with _open_lsmv_zip(path) as zf:
        names = set(zf.namelist())
        for key in (
            "authors",
            "gamename",
            "gametype",
            "coreversion",
            "systemid",
            "rerecords",
            "projectid",
            "rom.sha256",
            "rom.hint",
        ):
            if key in names:
                meta[key] = zf.read(key).decode("utf-8", errors="replace").strip()
        if "input" not in names:
            raise ValueError(f"LSMV missing input: {path}")
        text = zf.read("input").decode("utf-8", errors="replace")

    for line in text.splitlines():
        if not line or "|" not in line:
            continue
        # Subframe continuation (rare): line starts with whitespace / . / |
        if line[0] in " \t\r\n.|":
            # Append as its own frame if present (keep length honest).
            # Sniq SM movies are frame-aligned with no subframes.
            pass
        head, _, btn = line.partition("|")
        # head is like "F." or "F. 0 0" or "FR 1 2"
        is_reset = len(head) >= 2 and head[1] not in ". \t"
        btn = btn.strip()
        # Drop trailing port groups if any (first 12 chars are P1).
        if "|" in btn:
            btn = btn.split("|", 1)[0]
        btn = btn[:_LSMV_LEN].ljust(_LSMV_LEN, ".")
        frames.append(_parse_button_field(btn))
        raw_p1.append(btn)
        resets.append(bool(is_reset))

    if not frames:
        raise ValueError(f"no input frames in {path}")
    return LsmvMovie(
        path=path,
        meta=meta,
        frames=frames,
        raw_p1=raw_p1,
        resets=resets,
    )


def lsmv_to_snes12_frames(path: Path | str) -> list[list[int]]:
    """Convenience: path → list of 12-int SNES frames."""
    return parse_lsmv(path).frames
