"""BizHawk NesHawk ``.bk2`` for SMB TAS movies.

Matches the conversion already on disk for HappyLee warps
(``happylee_warps_1715M.fm2.bk2``): FCEUX FM2 → NesHawk 2.11 BK2, headerless
iNES MD5/SHA1, L+R preserved.

```bash
uv run python -m smb.scripts.convert_fm2
uv run python -m smb.scripts.convert_fm2 nes/smb/tas/ref/happylee_mars608_warpless_3728M.fm2
```
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

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
from smb.paths import GAME_DIR, ROMS_DIR
from smb.tas.fm2 import Fm2Movie, parse_fm2

# LogKey order used by BizHawk 2.11 NesHawk FM2 import.
BK2_LOGKEY = (
    "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
    "P1 Start|P1 Select|P1 B|P1 A|#P2 Up|P2 Down|P2 Left|P2 Right|"
    "P2 Start|P2 Select|P2 B|P2 A|"
)

# FM2 player-1 ``RLDUTSBA`` → BK2 ``UDLRSsBA`` (Start=S, Select=s).
_FM2_P1_TO_BK2: tuple[tuple[int, int, str], ...] = (
    (0, 3, "R"),
    (1, 2, "L"),
    (2, 1, "D"),
    (3, 0, "U"),
    (4, 4, "S"),
    (5, 5, "s"),
    (6, 6, "B"),
    (7, 7, "A"),
)

_BK2_P1_TO_NES: tuple[int, ...] = (
    NES_UP,
    NES_DOWN,
    NES_LEFT,
    NES_RIGHT,
    NES_START,
    NES_SELECT,
    NES_B,
    NES_A,
)

# Same NesHawk sync blob BizHawk 2.11 wrote for the warps conversion.
NESHAWK_SYNC_SETTINGS = (
    '{"o":{"$type":"BizHawk.Emulation.Cores.Nintendo.NES.NES+NESSyncSettings, '
    'BizHawk.Emulation.Cores","BoardProperties":{},"RegionOverride":0,'
    '"Controls":{"Famicom":false,"NesLeftPort":"ControllerNES",'
    '"NesRightPort":"ControllerNES","FamicomExpPort":"UnpluggedFam"},'
    '"VSDipswitches":{"Dip_Switch_1":false,"Dip_Switch_2":false,'
    '"Dip_Switch_3":false,"Dip_Switch_4":false,"Dip_Switch_5":false,'
    '"Dip_Switch_6":false,"Dip_Switch_7":false,"Dip_Switch_8":false},'
    '"InitialWRamStatePattern":[]}}'
)

KNOWN_HEADERLESS_MD5 = "8e3630186e35d477231bf8fd50e54cdd"
KNOWN_HEADERLESS_SHA1 = "FACEE9C577A5262DBE33AC4930BB0B58C8C037F7"

_DEFAULT_ROM_CANDIDATES = (
    ROMS_DIR / "Super Mario Bros..nes",
    GAME_DIR / "roms" / "Super Mario Bros..nes",
    GAME_DIR / "custom_integrations" / "SuperMarioBros-Nes" / "rom.nes",
)


def default_smb_rom() -> Path | None:
    """First existing local SMB ``.nes`` (integration symlink is allowed)."""
    for path in _DEFAULT_ROM_CANDIDATES:
        try:
            if path.exists() and path.stat().st_size > 16:
                return path
        except OSError:
            continue
    return None


def headerless_rom_hashes(rom_path: Path | None = None) -> tuple[str, str]:
    """MD5/SHA1 of the iNES payload (BizHawk NES movie hashes)."""
    path = rom_path if rom_path is not None else default_smb_rom()
    if path is None:
        return KNOWN_HEADERLESS_MD5, KNOWN_HEADERLESS_SHA1
    data = Path(path).read_bytes()
    body = data[16:] if data[:4] == b"NES\x1a" and len(data) > 16 else data
    md5 = hashlib.md5(body).hexdigest()
    sha1 = hashlib.sha1(body).hexdigest().upper()
    return md5, sha1


def bk2_output_path(fm2_path: Path | str) -> Path:
    """``movie.fm2`` → ``movie.fm2.bk2`` (same suffix as the warps conversion)."""
    path = Path(fm2_path)
    return path.with_name(path.name + ".bk2")


def fm2_p1_to_bk2(p1: str) -> str:
    """Map one FM2 player-1 mnemonic onto NesHawk BK2 order."""
    padded = (p1 + "........")[:8]
    out = ["."] * 8
    for src, dst, letter in _FM2_P1_TO_BK2:
        if padded[src] != ".":
            out[dst] = letter
    return "".join(out)


def fm2_p2_to_bk2(p2: str) -> str:
    """Player-2 uses the same UDLRSsBA mnemonic as player-1."""
    return fm2_p1_to_bk2(p2 if p2 else "........")


def fm2_cmd_to_bk2_reset(cmd: int) -> str:
    """FM2 command bits → BK2 ``Reset|Power`` pair (``r`` / ``P``)."""
    reset = "r" if cmd & 1 else "."
    power = "P" if cmd & 2 else "."
    return f"{reset}{power}"


def _fm2_input_lines(path: Path) -> list[tuple[int, str, str]]:
    """Raw ``(cmd, p1, p2)`` triples, including zip-wrapped downloads."""
    movie = parse_fm2(path)
    triples: list[tuple[int, str, str]] = []
    # Re-read source text so P2 is preserved (parse_fm2 keeps P1 only).
    from smb.tas.fm2 import _open_fm2_text

    with _open_fm2_text(path) as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            # '', cmd, p1, p2, ...
            cmd = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            p1 = parts[2] if len(parts) > 2 else "........"
            p2 = parts[3] if len(parts) > 3 else "........"
            triples.append((cmd, p1, p2))
    if len(triples) != movie.num_frames:
        # Fall back to parsed P1 if the text splitter drifted.
        return [(c, p, "........") for c, p in zip(movie.commands, movie.raw_p1)]
    return triples


def build_input_log(path: Path | str) -> str:
    """Return the BK2 ``Input Log.txt`` body for an FM2 movie."""
    lines = ["[Input]", BK2_LOGKEY]
    for cmd, p1, p2 in _fm2_input_lines(Path(path)):
        lines.append(
            f"|{fm2_cmd_to_bk2_reset(cmd)}|{fm2_p1_to_bk2(p1)}|{fm2_p2_to_bk2(p2)}|"
        )
    lines.append("[/Input]")
    return "\n".join(lines) + "\n"


def _header_text(movie: Fm2Movie, *, md5: str, sha1: str) -> str:
    author = (movie.author or "").strip()
    pal = movie.header.get("palFlag", "0")
    game = movie.rom_filename or "Super Mario Bros. (JU) [!]"
    rerec = movie.header.get("rerecordCount", "0")
    return (
        "MovieVersion BizHawk v2.0.0\n"
        "Core NesHawk\n"
        "Platform NES\n"
        f"rerecordCount {rerec}\n"
        f"PAL {pal}\n"
        f"GameName {game}\n"
        f"MD5 {md5}\n"
        f"Author {author}\n"
        f"SHA1 {sha1}\n"
        "OriginalEmuVersion \n"
        "emuVersion Version 2.11\n"
    )


def _comments_text(movie: Fm2Movie) -> str:
    version = movie.header.get("version", "3")
    emu = movie.header.get("emuVersion", "")
    origin = f"emuOrigin FCEUX version {emu}\n" if emu else ""
    return f"MovieOrigin .fm2 version {version}\n{origin}\n"


def _zip_write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def write_neshawk_bk2(
    fm2_path: Path | str,
    output_path: Path | str | None = None,
    *,
    rom_path: Path | str | None = None,
) -> Path:
    """Write a NesHawk BK2 next to the FM2 (``*.fm2.bk2`` by default)."""
    fm2_path = Path(fm2_path)
    movie = parse_fm2(fm2_path)
    out = Path(output_path) if output_path is not None else bk2_output_path(fm2_path)
    rom = Path(rom_path) if rom_path is not None else default_smb_rom()
    md5, sha1 = headerless_rom_hashes(rom)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        _zip_write(zf, "BizState 1.0", b"3\n")
        _zip_write(zf, "BizVersion.txt", b"Version 2.11\n")
        _zip_write(zf, "Header.txt", _header_text(movie, md5=md5, sha1=sha1).encode())
        _zip_write(zf, "Comments.txt", _comments_text(movie).encode())
        _zip_write(zf, "Subtitles.txt", b"\n")
        _zip_write(zf, "SyncSettings.json", NESHAWK_SYNC_SETTINGS.encode())
        _zip_write(zf, "Input Log.txt", build_input_log(fm2_path).encode())
    return out


def _open_bk2_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def parse_bk2(path: Path | str) -> Fm2Movie:
    """Parse a NesHawk BK2 into the same NES-9 layout as :func:`parse_fm2`."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        header_raw = _open_bk2_text(zf, "Header.txt")
        log = _open_bk2_text(zf, "Input Log.txt")
    header: dict[str, str] = {}
    for line in header_raw.splitlines():
        if not line.strip():
            continue
        key, _, val = line.partition(" ")
        header[key] = val
    # Alias NesHawk keys onto the FM2 summary fields.
    if "Author" in header:
        header.setdefault("author", header["Author"])
    if "GameName" in header:
        header.setdefault("romFilename", header["GameName"])
    if "PAL" in header:
        header.setdefault("palFlag", header["PAL"])
    frames: list[list[int]] = []
    commands: list[int] = []
    raw_p1: list[str] = []
    for line in log.splitlines():
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        reset = parts[1] if len(parts) > 1 else ".."
        p1 = parts[2] if len(parts) > 2 else "........"
        cmd = 0
        if reset[:1] in {"r", "R"}:
            cmd |= 1
        if len(reset) > 1 and reset[1] in {"p", "P"}:
            cmd |= 2
        padded = (p1 + "........")[:8]
        action = [0] * NES_ACTION_SIZE
        for i, nes in enumerate(_BK2_P1_TO_NES):
            if padded[i] != ".":
                action[nes] = 1
        frames.append(action)
        commands.append(cmd)
        raw_p1.append(padded)
    if not frames:
        raise ValueError(f"no input frames in {path}")
    return Fm2Movie(
        path=path,
        header=header,
        frames=frames,
        commands=commands,
        raw_p1=raw_p1,
    )


def parse_movie(path: Path | str) -> Fm2Movie:
    """Parse ``.fm2`` or ``.bk2`` (including ``*.fm2.bk2``)."""
    path = Path(path)
    if path.name.lower().endswith(".bk2"):
        return parse_bk2(path)
    return parse_fm2(path)


def input_log_from_bk2(path: Path | str) -> str:
    """Read ``Input Log.txt`` from a BK2 zip."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        return _open_bk2_text(zf, "Input Log.txt")
