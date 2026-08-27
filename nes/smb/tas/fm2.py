"""FCEUX ``.fm2`` movie import for Super Mario Bros. seeds.

Maps FM2 player-1 buttons onto the stable-retro NES 9-slot layout
``[B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A]``.

FM2 line format (version 3)::

    |commands|RLDUTSBA|........||

where each of ``R L D U T S B A`` is either the letter or ``.`` (released).
``T`` = Select, ``S`` = Start. Command bit ``1`` on the first frame is a
soft-reset / power-on marker and is ignored for button replay (our runner
starts from ``env.reset()``).

HappyLee warps (#1715M) and community RTA-rules movies are the intended
inputs. Simultaneous Left+Right is preserved — do **not** run
``sanitize_action`` on TAS frames (it zeroes L+R, which the TAS uses).
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from retro_harness.controls import (
    NES_A,
    NES_ACTION_SIZE,
    NES_B,
    NES_DOWN,
    NES_LEFT,
    NES_RIGHT,
    NES_SELECT,
    NES_START,
    NES_UP,
)

# FM2 player-1 button string positions → NES action index.
# FM2 order (FCEUX): R L D U T S B A
#   T = sTart, S = Select  (easy to swap — do not invert)
_FM2_TO_NES: tuple[int, ...] = (
    NES_RIGHT,  # R
    NES_LEFT,  # L
    NES_DOWN,  # D
    NES_UP,  # U
    NES_START,  # T = sTart
    NES_SELECT,  # S = Select
    NES_B,  # B
    NES_A,  # A
)

_FRAME_RE = re.compile(r"^\|(\d+)\|([^|]*)\|")


@dataclass
class Fm2Movie:
    """Parsed FCEUX movie with metadata + NES-9 frames."""

    path: Path
    header: dict[str, str] = field(default_factory=dict)
    frames: list[list[int]] = field(default_factory=list)
    commands: list[int] = field(default_factory=list)
    raw_p1: list[str] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def author(self) -> str | None:
        for key in ("comment author", "author"):
            if key in self.header:
                return self.header[key].strip()
        # FCEUX writes ``comment author Name`` (first space splits to key=comment).
        for k, v in self.header.items():
            if k.startswith("comment") and "author" in k.lower():
                return v.strip()
            if k == "comment" and v.lower().startswith("author"):
                rest = v.split(None, 1)
                return rest[1].strip() if len(rest) > 1 else v.strip()
        return self.header.get("comment")

    @property
    def rom_filename(self) -> str | None:
        return self.header.get("romFilename")

    @property
    def lr_frames(self) -> int:
        """Count of frames with simultaneous Left+Right held."""
        n = 0
        for fr in self.frames:
            if fr[NES_LEFT] and fr[NES_RIGHT]:
                n += 1
        return n

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "num_frames": self.num_frames,
            "author": self.author,
            "rom_filename": self.rom_filename,
            "rerecord_count": self.header.get("rerecordCount"),
            "pal_flag": self.header.get("palFlag"),
            "lr_frames": self.lr_frames,
            "first_nonzero_frame": next(
                (i for i, fr in enumerate(self.frames) if any(fr)), None
            ),
        }


def _open_fm2_text(path: Path) -> TextIO:
    """Open an FM2 path; transparently unzip single-entry zip downloads."""
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".fm2")]
            if not names:
                names = zf.namelist()
            if not names:
                raise ValueError(f"zip has no members: {path}")
            data = zf.read(names[0])
        return _text_from_bytes(data)
    if raw[:2] == b"\x1f\x8b":
        import gzip

        data = gzip.decompress(raw)
        return _text_from_bytes(data)
    return path.open("r", encoding="utf-8", errors="replace")


def _text_from_bytes(data: bytes) -> TextIO:
    import io

    return io.StringIO(data.decode("utf-8", errors="replace"))


def parse_movie(path: Path | str) -> Fm2Movie:
    """Parse ``.fm2`` or NesHawk ``.bk2`` (including ``*.fm2.bk2``)."""
    path = Path(path)
    if path.name.lower().endswith(".bk2"):
        from smb.tas.bk2 import parse_bk2

        return parse_bk2(path)
    return parse_fm2(path)


def parse_fm2(path: Path | str) -> Fm2Movie:
    """Parse an ``.fm2`` (or zip/gzip wrapper) into NES-9 button frames."""
    path = Path(path)
    header: dict[str, str] = {}
    frames: list[list[int]] = []
    commands: list[int] = []
    raw_p1: list[str] = []

    with _open_fm2_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if not line.startswith("|"):
                if " " in line:
                    key, _, val = line.partition(" ")
                    header[key] = val
                else:
                    header[line] = ""
                continue
            m = _FRAME_RE.match(line)
            if not m:
                continue
            cmd = int(m.group(1))
            p1 = m.group(2)
            # pad/truncate to 8 buttons
            if len(p1) < 8:
                p1 = p1.ljust(8, ".")
            p1 = p1[:8]
            action = [0] * NES_ACTION_SIZE
            for i, ch in enumerate(p1):
                if ch != "." and i < len(_FM2_TO_NES):
                    action[_FM2_TO_NES[i]] = 1
            frames.append(action)
            commands.append(cmd)
            raw_p1.append(p1)

    if not frames:
        raise ValueError(f"no input frames in {path}")
    return Fm2Movie(
        path=path,
        header=header,
        frames=frames,
        commands=commands,
        raw_p1=raw_p1,
    )


def fm2_to_nes9_frames(path: Path | str) -> list[list[int]]:
    """Convenience: path → list of 9-int NES frames."""
    return parse_fm2(path).frames


def frames_to_nes9_rle_payload(
    frames: list[list[int]],
    *,
    route_id: str,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``nes9_rle`` seed dict from raw frames (no verify)."""
    from smb.policy import compress_nes9_rle

    payload: dict[str, Any] = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": "SuperMarioBros-Nes",
        "num_frames": len(frames),
        "source": source,
        "segments": compress_nes9_rle(frames),
    }
    if extra:
        payload.update(extra)
    return payload
