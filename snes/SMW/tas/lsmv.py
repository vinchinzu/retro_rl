"""Parse lsnes LSMV movies and convert them to native BizHawk BK2.

LSMV stores one text input row per emulated frame in this button order::

    B Y Select Start Up Down Left Right A X L R

The converter normalizes those rows to the same 12-bit words used by the SMV
lane, then emits a deterministic power-on BK2 for BizHawk's bsnes v115+
compatibility profile.  Synchronization is established separately by the
RAM-backed oracle.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from SMW.tas.smv import BK2_BUTTON_NAMES, word_to_bk2_mnemonic

_LSMV_BUTTON_NAMES = (
    "B",
    "Y",
    "Select",
    "Start",
    "Up",
    "Down",
    "Left",
    "Right",
    "A",
    "X",
    "L",
    "R",
)

# Normalized word bit positions are defined by SMV_BUTTON_NAMES in smv.py.
_NORMALIZED_BITS = {
    "Right": 0,
    "Left": 1,
    "Down": 2,
    "Up": 3,
    "Start": 4,
    "Select": 5,
    "Y": 6,
    "B": 7,
    "R": 8,
    "L": 9,
    "X": 10,
    "A": 11,
}

BizHawkCoreProfile = Literal["v115", "subframe-v115", "legacy"]

_V115_SYNC_SETTINGS = {
    "o": {
        "$type": (
            "BizHawk.Emulation.Cores.Nintendo.BSNES.BsnesCore+SnesSyncSettings, "
            "BizHawk.Emulation.Cores"
        ),
        "Profile": "Compatibility",
    }
}

_LEGACY_SYNC_SETTINGS = {
    "o": {
        "$type": (
            "BizHawk.Emulation.Cores.Nintendo.SNES.LibsnesCore+SnesSyncSettings, "
            "BizHawk.Emulation.Cores"
        ),
        "Profile": "Compatibility",
    }
}

_CORE_NAMES: dict[BizHawkCoreProfile, str] = {
    "v115": "BSNESv115+",
    "subframe-v115": "SubBSNESv115+",
    "legacy": "BSNES",
}


@dataclass(frozen=True, slots=True)
class LSMVMovie:
    """Parsed, frame-aligned SNES LSMV input movie."""

    path: Path
    metadata: dict[str, str]
    p1_words: tuple[int, ...]

    @property
    def num_frames(self) -> int:
        return len(self.p1_words)

    @property
    def first_input_frame(self) -> int | None:
        return next(
            (
                frame
                for frame, word in enumerate(self.p1_words)
                if word not in {0, 0xFFFF}
            ),
            None,
        )

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "format": "lsmv",
            "num_input_samples": self.num_frames,
            "author": self.metadata.get("authors") or None,
            "gametype": self.metadata.get("gametype"),
            "coreversion": self.metadata.get("coreversion"),
            "systemid": self.metadata.get("systemid"),
            "rerecord_count": _optional_int(self.metadata.get("rerecords")),
            "rom_sha256": self.metadata.get("rom.sha256"),
            "rom_hint": self.metadata.get("rom.hint"),
            "first_input_frame": self.first_input_frame,
            "reset_frames": sum(word == 0xFFFF for word in self.p1_words),
        }


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _open_lsmv(path: Path) -> zipfile.ZipFile:
    raw = path.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)
    if not raw.startswith(b"PK"):
        raise ValueError(f"not a zip LSMV: {path}")

    archive = zipfile.ZipFile(io.BytesIO(raw))
    if "input" in archive.namelist():
        return archive

    nested = [name for name in archive.namelist() if name.lower().endswith(".lsmv")]
    if len(nested) != 1:
        archive.close()
        raise ValueError(f"LSMV has no input member: {path}")
    inner = archive.read(nested[0])
    archive.close()
    if inner.startswith(b"\x1f\x8b"):
        inner = gzip.decompress(inner)
    return zipfile.ZipFile(io.BytesIO(inner))


def _button_field_to_word(field: str) -> int:
    field = field[: len(_LSMV_BUTTON_NAMES)].ljust(len(_LSMV_BUTTON_NAMES), ".")
    word = 0
    for name, marker in zip(_LSMV_BUTTON_NAMES, field, strict=True):
        if marker != ".":
            word |= 1 << _NORMALIZED_BITS[name]
    return word


def parse_lsmv(path: Path | str) -> LSMVMovie:
    """Parse a direct, gzip-wrapped, or single-member-wrapped LSMV."""

    path = Path(path)
    metadata: dict[str, str] = {}
    words: list[int] = []
    with _open_lsmv(path) as archive:
        names = set(archive.namelist())
        if "input" not in names:
            raise ValueError(f"LSMV missing input: {path}")
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
                metadata[key] = (
                    archive.read(key).decode("utf-8", errors="replace").strip()
                )
        input_text = archive.read("input").decode("utf-8", errors="strict")

    for line_number, line in enumerate(input_text.splitlines(), start=1):
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"LSMV input line {line_number} has no controller field")
        head, field = line.split("|", 1)
        if not head.startswith("F"):
            raise ValueError(
                f"LSMV subframe/non-frame input is unsupported at line {line_number}"
            )
        reset = len(head) >= 2 and head[1] not in ". \t"
        p1_field = field.split("|", 1)[0].strip()
        words.append(0xFFFF if reset else _button_field_to_word(p1_field))

    if not words:
        raise ValueError(f"LSMV contains no input frames: {path}")
    return LSMVMovie(path=path, metadata=metadata, p1_words=tuple(words))


def _zip_write(archive: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data.encode("utf-8"))


def write_bizhawk_bk2(
    movie: LSMVMovie,
    output_path: Path | str,
    *,
    rom_path: Path | str,
    max_frames: int | None = None,
    core_profile: BizHawkCoreProfile = "v115",
) -> Path:
    """Write a deterministic BizHawk compatibility-profile BK2."""

    output_path = Path(output_path)
    rom_path = Path(rom_path)
    rom_data = rom_path.read_bytes()
    rom_sha256 = hashlib.sha256(rom_data).hexdigest()
    expected_sha256 = movie.metadata.get("rom.sha256", "").lower()
    if expected_sha256 and expected_sha256 != rom_sha256:
        raise ValueError(
            f"LSMV ROM SHA-256 mismatch: expected {expected_sha256}, got {rom_sha256}"
        )
    words = movie.p1_words[:max_frames]
    if not words:
        raise ValueError("cannot write a BK2 without input frames")

    rerecords = _optional_int(movie.metadata.get("rerecords")) or 0
    author = movie.metadata.get("authors") or "unknown"
    game_name = movie.metadata.get("gamename") or movie.metadata.get("rom.hint")
    game_name = game_name or "Super Mario World"
    core_name = _CORE_NAMES[core_profile]
    header_lines = [
        "MovieVersion BizHawk v2.0.0",
        f"rerecordCount {rerecords}",
        f"Author {author}",
        f"emuVersion {movie.metadata.get('coreversion', 'lsnes conversion')}",
        "Platform SNES",
        f"GameName {game_name}",
        f"SHA1 {hashlib.sha1(rom_data).hexdigest().upper()}",
        f"Core {core_name}",
        "StartsFromSavestate False",
        f"PAL {movie.metadata.get('gametype') == 'snes_pal'}",
    ]
    log_key = "LogKey:#Reset|Power|" + "".join(
        f"#P1 {name}|" for name in BK2_BUTTON_NAMES
    )
    log_key += "".join(f"#P2 {name}|" for name in BK2_BUTTON_NAMES)
    input_lines = ["[Input]", log_key]
    for word in words:
        reset = "R." if word == 0xFFFF else ".."
        input_lines.append(f"|{reset}|{word_to_bk2_mnemonic(word)}|............|")

    comments = {
        "source_format": "lsmv",
        "source_path": str(movie.path),
        "source_sha256": hashlib.sha256(movie.path.read_bytes()).hexdigest(),
        "source_emulator": movie.metadata.get("coreversion"),
        "source_rom_sha256": expected_sha256 or None,
        "converted_rom_sha256": rom_sha256,
        "conversion": (
            f"frame-aligned LSMV input to BizHawk {core_name} compatibility"
        ),
        "sync_claim": "unverified; run SMW TAS oracle",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        _zip_write(archive, "Header.txt", "\n".join(header_lines) + "\n")
        _zip_write(archive, "Input Log.txt", "\n".join(input_lines) + "\n")
        _zip_write(archive, "Comments.txt", json.dumps(comments, indent=2) + "\n")
        sync_settings = (
            _LEGACY_SYNC_SETTINGS if core_profile == "legacy" else _V115_SYNC_SETTINGS
        )
        _zip_write(archive, "SyncSettings.json", json.dumps(sync_settings) + "\n")
        _zip_write(archive, "Subtitles.txt", "\n")
    return output_path
