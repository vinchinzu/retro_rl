"""Read native BizHawk BK2 movies into normalized SNES input words."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from SMW.tas.smv import BK2_BUTTON_NAMES, SMV_BUTTON_NAMES

_NORMALIZED_BITS = {name: bit for bit, name in enumerate(SMV_BUTTON_NAMES)}

BizHawkCoreProfile = Literal["v115", "subframe-v115", "legacy"]

_CORE_NAMES: dict[BizHawkCoreProfile, str] = {
    "v115": "BSNESv115+",
    "subframe-v115": "SubBSNESv115+",
    "legacy": "BSNES",
}

_SYNC_TYPES: dict[BizHawkCoreProfile, str] = {
    "v115": (
        "BizHawk.Emulation.Cores.Nintendo.BSNES.BsnesCore+SnesSyncSettings, "
        "BizHawk.Emulation.Cores"
    ),
    "subframe-v115": (
        "BizHawk.Emulation.Cores.Nintendo.BSNES.BsnesCore+SnesSyncSettings, "
        "BizHawk.Emulation.Cores"
    ),
    "legacy": (
        "BizHawk.Emulation.Cores.Nintendo.SNES.LibsnesCore+SnesSyncSettings, "
        "BizHawk.Emulation.Cores"
    ),
}


@dataclass(frozen=True, slots=True)
class BK2Movie:
    """Native BK2 provenance and player-one input frames."""

    path: Path
    header: dict[str, str]
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

    def verify_rom(self, rom_path: Path | str) -> None:
        expected_sha1 = self.header.get("SHA1", "").upper()
        actual_sha1 = hashlib.sha1(Path(rom_path).read_bytes()).hexdigest().upper()
        if expected_sha1 and expected_sha1 != actual_sha1:
            raise ValueError(
                f"BK2 ROM SHA-1 mismatch: expected {expected_sha1}, got {actual_sha1}"
            )

    def summary(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "format": "bk2",
            "num_input_samples": self.num_frames,
            "author": self.header.get("Author"),
            "game_name": self.header.get("GameName"),
            "emulator_version": self.header.get("emuVersion"),
            "movie_version": self.header.get("MovieVersion"),
            "core": self.header.get("Core") or "legacy SyncSettings/default",
            "rerecord_count": _optional_int(self.header.get("rerecordCount")),
            "rom_sha1": self.header.get("SHA1"),
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


def _parse_header(text: str) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            header[key] = value.strip()
    return header


def _mnemonic_to_word(mnemonic: str) -> int:
    mnemonic = mnemonic[: len(BK2_BUTTON_NAMES)].ljust(len(BK2_BUTTON_NAMES), ".")
    word = 0
    for name, marker in zip(BK2_BUTTON_NAMES, mnemonic, strict=True):
        if marker != ".":
            word |= 1 << _NORMALIZED_BITS[name]
    return word


def parse_bk2(path: Path | str) -> BK2Movie:
    """Parse a frame-based SNES BK2 with the standard player-one joypad."""

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        header = _parse_header(archive.read("Header.txt").decode("utf-8-sig"))
        input_lines = archive.read("Input Log.txt").decode("utf-8-sig").splitlines()

    if not input_lines or input_lines[0] != "[Input]":
        raise ValueError(f"BK2 has no frame input section: {path}")
    if len(input_lines) < 2 or "#P1 Up|" not in input_lines[1]:
        raise ValueError(f"BK2 has no standard SNES P1 controls: {path}")

    words: list[int] = []
    for line_number, line in enumerate(input_lines[2:], start=3):
        if line == "[/Input]":
            break
        fields = line.split("|")
        if len(fields) < 4 or fields[0] != "":
            raise ValueError(f"unsupported BK2 input row at line {line_number}")
        reset_power = fields[1]
        p1_mnemonic = fields[2]
        words.append(
            0xFFFF if reset_power.startswith("R") else _mnemonic_to_word(p1_mnemonic)
        )

    if not words:
        raise ValueError(f"BK2 contains no input frames: {path}")
    return BK2Movie(path=path, header=header, p1_words=tuple(words))


def _zip_write_bytes(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def retarget_bk2(
    source_path: Path | str,
    output_path: Path | str,
    *,
    core_profile: BizHawkCoreProfile,
) -> Path:
    """Copy a BK2 while replacing only its explicit core/sync metadata."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    core_name = _CORE_NAMES[core_profile]
    sync_settings = {
        "o": {"$type": _SYNC_TYPES[core_profile], "Profile": "Compatibility"}
    }
    sync_text = json.dumps(sync_settings, separators=(",", ":"))

    with zipfile.ZipFile(source_path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    header_lines = members["Header.txt"].decode("utf-8-sig").splitlines()
    header_lines = [
        line
        for line in header_lines
        if not line.startswith("Core ") and not line.startswith("SyncSettings ")
    ]
    header_lines.extend((f"Core {core_name}", f"SyncSettings {sync_text}"))
    members["Header.txt"] = ("\n".join(header_lines) + "\n").encode()
    members["SyncSettings.json"] = (sync_text + "\n").encode()
    provenance = (
        f"\nretargeted_core={core_name}\n"
        f"source_sha256={hashlib.sha256(source_path.read_bytes()).hexdigest()}\n"
    ).encode()
    members["Comments.txt"] = members.get("Comments.txt", b"") + provenance

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as output:
        for name in sorted(members):
            _zip_write_bytes(output, name, members[name])
    return output_path
