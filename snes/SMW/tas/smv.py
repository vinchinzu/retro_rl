"""Parse Snes9x SMV input movies and convert them to native BizHawk BK2.

The converter follows BizHawk 2.11's own ``SmvImport`` mapping.  It converts
inputs, provenance, and sync metadata; it does not claim that a movie made on
an older Snes9x version remains synchronized on BizHawk's current Snes9x core.
That claim belongs to the RAM-backed oracle verifier.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

SMV_SIGNATURE = b"SMV\x1a"
SUPPORTED_VERSIONS = {1: "1.43", 4: "1.51", 5: "1.52"}

# SMV's normalized 12-bit order, matching BizHawk's built-in importer.
SMV_BUTTON_NAMES = (
    "Right",
    "Left",
    "Down",
    "Up",
    "Start",
    "Select",
    "Y",
    "B",
    "R",
    "L",
    "X",
    "A",
)

# Current BizHawk SNES joypad mnemonic order.
BK2_BUTTON_NAMES = (
    "Up",
    "Down",
    "Left",
    "Right",
    "Select",
    "Start",
    "Y",
    "B",
    "X",
    "A",
    "L",
    "R",
)

_BK2_ON_CHARS = {
    "Up": "U",
    "Down": "D",
    "Left": "L",
    "Right": "R",
    "Select": "s",
    "Start": "S",
    "Y": "Y",
    "B": "B",
    "X": "X",
    "A": "A",
    "L": "l",
    "R": "r",
}


@dataclass(frozen=True, slots=True)
class SMVMovie:
    """Parsed standard-controller SMV movie."""

    path: Path
    version: int
    emulator_version: str
    uid: int
    rerecord_count: int
    header_frame_count: int
    controller_mask: int
    starts_from_reset: bool
    pal: bool
    sync_flags: int
    savestate_offset: int
    controller_data_offset: int
    author: str
    rom_crc32: str | None
    rom_name: str | None
    save_ram: bytes | None
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
            "format": "smv",
            "version": self.version,
            "emulator_version": self.emulator_version,
            "uid": self.uid,
            "rerecord_count": self.rerecord_count,
            "header_frame_count": self.header_frame_count,
            "num_input_samples": self.num_frames,
            "controller_mask": self.controller_mask,
            "starts_from_reset": self.starts_from_reset,
            "pal": self.pal,
            "author": self.author,
            "rom_crc32": self.rom_crc32,
            "rom_name": self.rom_name,
            "save_ram_bytes": len(self.save_ram) if self.save_ram is not None else None,
            "first_input_frame": self.first_input_frame,
        }


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _decode_utf16z(data: bytes) -> str:
    end = 0
    while end + 1 < len(data) and data[end : end + 2] != b"\0\0":
        end += 2
    return data[:end].decode("utf-16le", errors="replace").strip()


def _decompress_first_gzip_stream(data: bytes) -> bytes:
    decoder = zlib.decompressobj(zlib.MAX_WBITS | 16)
    return decoder.decompress(data) + decoder.flush()


def _controller_count(mask: int) -> int:
    return sum((mask >> index) & 1 for index in range(5))


def parse_smv(path: Path | str) -> SMVMovie:
    """Parse an SMV with standard joypad controller data.

    Reset-anchored movies expose their embedded SRAM for provenance, but the
    BK2 converter intentionally starts BizHawk from an isolated blank SaveRAM
    directory.  BizHawk's own SMV importer behaves the same way.
    """

    path = Path(path)
    data = path.read_bytes()
    if len(data) < 32 or data[:4] != SMV_SIGNATURE:
        raise ValueError(f"not a valid SMV: {path}")

    version = _u32(data, 4)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported SMV version {version}: {path}")

    header_size = 32 if version == 1 else 64
    uid = _u32(data, 8)
    rerecord_count = _u32(data, 12)
    header_frame_count = _u32(data, 16)
    controller_mask = data[0x14]
    movie_options = data[0x15]
    sync_flags = data[0x17]
    savestate_offset = _u32(data, 0x18)
    controller_data_offset = _u32(data, 0x1C)

    if controller_mask & 1 == 0:
        raise ValueError("SMV controller 1 must be enabled")
    if not (header_size <= savestate_offset <= controller_data_offset <= len(data)):
        raise ValueError("SMV contains invalid state/controller offsets")

    has_rom_info = bool(sync_flags & 0x01 and sync_flags & 0x40)
    rom_info_size = 30 if has_rom_info else 0
    metadata_end = savestate_offset - rom_info_size
    if metadata_end < header_size:
        raise ValueError("SMV ROM-info offset overlaps its header")
    author = _decode_utf16z(data[header_size:metadata_end])

    rom_crc32: str | None = None
    rom_name: str | None = None
    if has_rom_info:
        rom_info = data[metadata_end:savestate_offset]
        rom_crc32 = f"{struct.unpack_from('<I', rom_info, 3)[0]:08X}"
        rom_name = rom_info[7:30].split(b"\0", 1)[0].decode("ascii", errors="replace")

    starts_from_reset = bool(movie_options & 0x01)
    save_ram: bytes | None = None
    if starts_from_reset:
        save_ram = _decompress_first_gzip_stream(
            data[savestate_offset:controller_data_offset]
        )

    controller_count = _controller_count(controller_mask)
    sample_size = controller_count * 2
    if version != 1:
        port_types = data[0x24:0x26]
        sample_size += sum(5 for kind in port_types if kind == 2)
        sample_size += sum(6 for kind in port_types if kind == 3)
        sample_size += sum(11 for kind in port_types if kind == 4)
    if sample_size <= 0:
        raise ValueError("SMV has no controller samples")

    available = len(data) - controller_data_offset
    if available % sample_size:
        raise ValueError("SMV controller data is not sample-aligned")
    sample_count = available // sample_size
    expected_samples = header_frame_count + 1
    if version != 1:
        expected_samples = _u32(data, 0x20)
    if sample_count < expected_samples:
        raise ValueError(
            f"SMV has {sample_count} samples, expected at least {expected_samples}"
        )

    p1_words: list[int] = []
    for sample in range(expected_samples):
        offset = controller_data_offset + sample * sample_size
        low = data[offset]
        high = data[offset + 1]
        # Exact normalization used by BizHawk 2.11 SmvImport.RunImport.
        p1_words.append(((low << 4) & 0xF00) | high)
        if low == 0xFF and high == 0xFF:
            p1_words[-1] = 0xFFFF

    return SMVMovie(
        path=path,
        version=version,
        emulator_version=SUPPORTED_VERSIONS[version],
        uid=uid,
        rerecord_count=rerecord_count,
        header_frame_count=header_frame_count,
        controller_mask=controller_mask,
        starts_from_reset=starts_from_reset,
        pal=bool(movie_options & 0x02),
        sync_flags=sync_flags,
        savestate_offset=savestate_offset,
        controller_data_offset=controller_data_offset,
        author=author,
        rom_crc32=rom_crc32,
        rom_name=rom_name,
        save_ram=save_ram,
        p1_words=tuple(p1_words),
    )


def word_to_buttons(word: int) -> frozenset[str]:
    """Decode one normalized SMV controller word."""

    if word == 0xFFFF:
        return frozenset()
    return frozenset(
        name for bit, name in enumerate(SMV_BUTTON_NAMES) if word & (1 << bit)
    )


def word_to_bk2_mnemonic(word: int) -> str:
    buttons = word_to_buttons(word)
    return "".join(
        _BK2_ON_CHARS[name] if name in buttons else "." for name in BK2_BUTTON_NAMES
    )


def _zip_write(zf: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data.encode("utf-8"))


def write_bizhawk_bk2(
    movie: SMVMovie,
    output_path: Path | str,
    *,
    rom_path: Path | str,
    max_frames: int | None = None,
) -> Path:
    """Write a deterministic, power-on BizHawk Snes9x BK2 input movie."""

    output_path = Path(output_path)
    rom_path = Path(rom_path)
    rom_sha1 = hashlib.sha1(rom_path.read_bytes()).hexdigest().upper()
    words = movie.p1_words[:max_frames]
    if not words:
        raise ValueError("cannot write a BK2 without input frames")

    header_lines = [
        "MovieVersion BizHawk v2.0.0",
        f"rerecordCount {movie.rerecord_count}",
        f"Author {movie.author or 'unknown'}",
        f"emuVersion Snes9x {movie.emulator_version} input conversion",
        "Platform SNES",
        f"GameName {movie.rom_name or 'Super Mario World'}",
        f"SHA1 {rom_sha1}",
        "Core Snes9x",
        "StartsFromSavestate False",
        f"PAL {movie.pal}",
    ]
    log_key = "LogKey:#Reset|Power|" + "".join(
        f"#P1 {name}|" for name in BK2_BUTTON_NAMES
    )
    input_lines = ["[Input]", log_key]
    for word in words:
        reset = "R." if word == 0xFFFF else ".."
        input_lines.append(f"|{reset}|{word_to_bk2_mnemonic(word)}|")

    comments = {
        "source_format": "smv",
        "source_path": str(movie.path),
        "source_sha256": hashlib.sha256(movie.path.read_bytes()).hexdigest(),
        "source_emulator": f"Snes9x {movie.emulator_version}",
        "conversion": "BizHawk 2.11 SmvImport-compatible input mapping",
        "sync_claim": "unverified; run SMW TAS oracle",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as zf:
        _zip_write(zf, "Header.txt", "\n".join(header_lines) + "\n")
        _zip_write(zf, "Input Log.txt", "\n".join(input_lines) + "\n")
        _zip_write(zf, "Comments.txt", json.dumps(comments, indent=2) + "\n")
        _zip_write(zf, "Subtitles.txt", "\n")
    return output_path
