"""Snes9x ``.smv`` → SNES-12 env frames (BizHawk SmvImport mapping).

Reuses ``SMW.tas.smv.parse_smv`` (second consumer of that parser) and maps
normalized controller words onto harness SNES-12 order. Reset samples
(``0xFFFF``) become idle frames — SNES-12 has no reset bit.

Spec / mapping: BizHawk 2.11 ``SmvImport``; see ``SMW.tas.smv``.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.actions import SNES_ACTION_SIZE
from retro_harness.controls import SNES_BUTTON_NAME_TO_INDEX, SNES_BUTTON_NAMES
from SMW.tas.smv import (
    BK2_BUTTON_NAMES,
    SMVMovie,
    parse_smv,
    word_to_bk2_mnemonic,
    word_to_buttons,
)

assert SNES_ACTION_SIZE == len(SNES_BUTTON_NAMES)


@dataclass
class SmvEnvMovie:
    """Parsed SMV with SNES-12 frames in env order."""

    path: Path
    raw: SMVMovie
    frames: list[list[int]]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def summary(self) -> dict[str, Any]:
        first_nz = next((i for i, fr in enumerate(self.frames) if any(fr)), None)
        base = self.raw.summary()
        base.update(
            {
                "num_frames": self.num_frames,
                "first_nonzero_frame": first_nz,
                "env_order": list(SNES_BUTTON_NAMES),
            }
        )
        return base


def _word_to_env_frame(word: int) -> list[int]:
    action = [0] * SNES_ACTION_SIZE
    for name in word_to_buttons(word):
        idx = SNES_BUTTON_NAME_TO_INDEX.get(name.upper())
        if idx is None:
            continue
        action[idx] = 1
    return action


def parse_smv_env(path: Path | str) -> SmvEnvMovie:
    """Parse an SMV into harness SNES-12 frames."""
    path = Path(path)
    raw = parse_smv(path)
    frames = [_word_to_env_frame(word) for word in raw.p1_words]
    return SmvEnvMovie(path=path, raw=raw, frames=frames)


def _zip_write(zf: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data.encode("utf-8"))


def write_bizhawk_bk2(
    movie: SMVMovie | SmvEnvMovie,
    output_path: Path | str,
    *,
    game_name: str = "Super Metroid",
    rom_sha1: str | None = None,
) -> Path:
    """Write a power-on BizHawk BK2 from an SMV (inputs only; sync unverified)."""
    raw = movie.raw if isinstance(movie, SmvEnvMovie) else movie
    output_path = Path(output_path)
    words = raw.p1_words
    if not words:
        raise ValueError("cannot write a BK2 without input frames")

    header_lines = [
        "MovieVersion BizHawk v2.0.0",
        f"rerecordCount {raw.rerecord_count}",
        f"Author {raw.author or 'unknown'}",
        f"emuVersion Snes9x {raw.emulator_version} input conversion",
        "Platform SNES",
        f"GameName {raw.rom_name or game_name}",
        "Core Snes9x",
        "StartsFromSavestate False",
        f"PAL {raw.pal}",
    ]
    if rom_sha1:
        header_lines.append(f"SHA1 {rom_sha1}")
    log_key = "LogKey:#Reset|Power|" + "".join(
        f"#P1 {name}|" for name in BK2_BUTTON_NAMES
    )
    input_lines = ["[Input]", log_key]
    for word in words:
        reset = "R." if word == 0xFFFF else ".."
        input_lines.append(f"|{reset}|{word_to_bk2_mnemonic(word)}|")

    comments = {
        "source_format": "smv",
        "source_path": str(raw.path),
        "source_sha256": hashlib.sha256(raw.path.read_bytes()).hexdigest(),
        "source_emulator": f"Snes9x {raw.emulator_version}",
        "conversion": "BizHawk 2.11 SmvImport-compatible input mapping",
        "sync_claim": "unverified; play on BizHawk Snes9x/BSNES and re-anchor",
        "game": game_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as zf:
        _zip_write(zf, "Header.txt", "\n".join(header_lines) + "\n")
        _zip_write(zf, "Input Log.txt", "\n".join(input_lines) + "\n")
        _zip_write(zf, "Comments.txt", json.dumps(comments, indent=2) + "\n")
        _zip_write(zf, "Subtitles.txt", "\n")
    return output_path
