"""BizHawk ``.bk2`` movie import for Super Metroid TAS seeds.

BK2 is a zip with ``Input Log.txt``. Modern SNES logs look like::

    LogKey:#Reset|Power|#P1 Up|P1 Down|...|P1 R|
    |..|.....S......|............|

Sniq 100% (converter) uses LogKey order::

    Up Down Left Right Select Start Y B X A L R

Older platformer code assumed a reversed hardware order; this parser
**reads the LogKey** when present and falls back to that order.

Spec: https://tasvideos.org/Bizhawk/BK2Format
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import (
    SNES_BUTTON_NAME_TO_INDEX,
    SNES_BUTTON_NAMES,
)

# Fallback when LogKey is missing (legacy hardware reverse).
# BK2 hardware: R L X A Right Left Down Up Start Select Y B
_LEGACY_BK2_TO_ENV = [11 - i for i in range(12)]

# BizHawk 1.x BKM / modern mnemonic order (Up Down Left Right Select Start Y B X A L R).
_MNEMONIC_LOGKEY = (
    "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
    "P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|"
)

_P1_TOKEN_RE = re.compile(r"P1\s+(\w+)", re.I)


def _normalize_btn_name(name: str) -> str | None:
    key = name.strip().upper()
    aliases = {
        "SEL": "SELECT",
        "SELECT": "SELECT",
        "START": "START",
        "UP": "UP",
        "DOWN": "DOWN",
        "LEFT": "LEFT",
        "RIGHT": "RIGHT",
        "A": "A",
        "B": "B",
        "X": "X",
        "Y": "Y",
        "L": "L",
        "R": "R",
    }
    return aliases.get(key)


def parse_logkey_p1_to_env(logkey_line: str) -> list[int] | None:
    """Return BK2 P1 char-index → env index map from a LogKey line."""
    # Strip leading "LogKey:"
    body = logkey_line.split(":", 1)[-1]
    # P1 tokens appear between # markers or as P1 Name|
    tokens = _P1_TOKEN_RE.findall(body)
    if len(tokens) < 12:
        return None
    mapping: list[int] = []
    for tok in tokens[:12]:
        name = _normalize_btn_name(tok)
        if name is None or name not in SNES_BUTTON_NAME_TO_INDEX:
            return None
        mapping.append(SNES_BUTTON_NAME_TO_INDEX[name])
    return mapping


@dataclass
class Bk2Movie:
    """Parsed BizHawk movie with SNES-12 frames in env order."""

    path: Path
    header: dict[str, str] = field(default_factory=dict)
    logkey: str | None = None
    p1_to_env: list[int] = field(default_factory=lambda: list(_LEGACY_BK2_TO_ENV))
    frames: list[list[int]] = field(default_factory=list)
    raw_p1: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def summary(self) -> dict[str, Any]:
        first_nz = next((i for i, fr in enumerate(self.frames) if any(fr)), None)
        return {
            "path": str(self.path),
            "format": "bk2",
            "num_frames": self.num_frames,
            "author": self.header.get("Author"),
            "game_name": self.header.get("GameName"),
            "core": self.header.get("Core"),
            "sha1": self.header.get("SHA1"),
            "rerecord_count": self.header.get("rerecordCount"),
            "logkey": self.logkey,
            "first_nonzero_frame": first_nz,
            "p1_order": [
                SNES_BUTTON_NAMES[self.p1_to_env[i]] for i in range(SNES_ACTION_SIZE)
            ],
        }


def _read_zip_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw


def parse_bk2(path: Path | str) -> Bk2Movie:
    """Parse a ``.bk2`` (or gzip-wrapped zip) into SNES-12 env frames."""
    path = Path(path)
    raw = _read_zip_bytes(path)
    if raw[:2] != b"PK":
        raise ValueError(f"not a zip BK2: {path}")

    header: dict[str, str] = {}
    logkey: str | None = None
    frames: list[list[int]] = []
    raw_p1: list[str] = []
    p1_to_env = list(_LEGACY_BK2_TO_ENV)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = set(zf.namelist())
        if "Header.txt" in names:
            for line in zf.read("Header.txt").decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                if " " in line:
                    k, _, v = line.partition(" ")
                    header[k] = v
                else:
                    header[line] = ""
        if "Input Log.txt" in names:
            log_text = zf.read("Input Log.txt").decode("utf-8", errors="replace")
        else:
            bkm = [n for n in names if n.lower().endswith(".bkm")]
            if len(bkm) != 1:
                raise ValueError(f"BK2 missing Input Log.txt: {path}")
            log_text = zf.read(bkm[0]).decode("utf-8", errors="replace")
            logkey = _MNEMONIC_LOGKEY
            mapped = parse_logkey_p1_to_env(_MNEMONIC_LOGKEY)
            if mapped is not None:
                p1_to_env = mapped

    for line in log_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("LogKey:"):
            logkey = line
            mapped = parse_logkey_p1_to_env(line)
            if mapped is not None:
                p1_to_env = mapped
            continue
        if not line.startswith("|"):
            continue
        groups = [g for g in line.split("|") if g != ""]
        if len(groups) < 2:
            continue
        # groups[0] = reset/power (BK2: 2 chars; BKM: often 1 char), groups[1] = P1
        p1 = groups[1]
        if len(p1) < SNES_ACTION_SIZE:
            continue
        p1 = p1[:SNES_ACTION_SIZE]
        action = [0] * SNES_ACTION_SIZE
        for bk2_i, ch in enumerate(p1):
            if ch != ".":
                action[p1_to_env[bk2_i]] = 1
        frames.append(action)
        raw_p1.append(p1)

    if not frames:
        raise ValueError(f"no input frames in {path}")
    return Bk2Movie(
        path=path,
        header=header,
        logkey=logkey,
        p1_to_env=p1_to_env,
        frames=frames,
        raw_p1=raw_p1,
    )
