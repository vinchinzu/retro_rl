"""Vanilla ALttP TAS movie catalog (TASVideos game 191).

Published movies + small userfiles with machine-readable button logs.
RAM watches and Lua are listed as skips. Movies are gitignored; re-fetch
with ``python -m alttp.tas.fetch_refs``.

Sources: https://tasvideos.org/191G · https://tasvideos.org/UserFiles/Game/191
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alttp.paths import GAME_DIR

MovieKind = Literal["lsmv", "bk2", "smv"]

TAS_DIR = GAME_DIR / "tas"
REF_DIR = TAS_DIR / "ref"
SLICE_DIR = TAS_DIR / "slices"

_TV = "https://tasvideos.org"
_UF = f"{_TV}/UserFiles/Info"

# USA No-Intro (workspace ROM) and JP 1.0 (glitch movies).
USA_SHA1 = "6D4F10A8B10E10DBE624CB23CF03B88BB8252973"
JP_SHA1 = "E7E852F0159CE612E3911164878A9B08B3CB9060"
JP_SHA256 = "794e040b02c7591b59ad8843b51e7c619b88f87cddc6083a8e7a4027b96a2271"


@dataclass(frozen=True)
class MovieRef:
    """One vendored TASVideos movie (or an explicit skip)."""

    filename: str
    url: str
    kind: MovieKind
    source: str
    notes: str
    category: str
    sha256: str
    expected_frames: int
    emulator: str = ""
    region: str = "JP"
    vanilla: bool = True
    fetch: bool = True
    tags: tuple[str, ...] = ()
    skip_reason: str | None = None

    @property
    def path(self) -> Path:
        return REF_DIR / self.filename

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


# Unwrapped-on-disk SHA-256 (after gzip / publication-zip unwrap).
MOVIES: tuple[MovieRef, ...] = (
    MovieRef(
        filename="fmp_geg_3898M.lsmv",
        url=f"{_TV}/3898M?handler=Download",
        kind="lsmv",
        source="fmp, total & Yuzuhara_3 game end glitch #3898M",
        notes="Published 54.5s GEG. lsnes ygamepad16 (16-char P1; first 12 used).",
        category="game end glitch",
        sha256="58a36cf1ad958609b7582a7b6b2234d565e596abea0c194a92ecf15c75358a71",
        expected_frames=3277,
        emulator="lsnes rr2-β23",
        tags=("glitch", "geg", "published", "short"),
    ),
    MovieRef(
        filename="m_riss_100_wip_10k.bk2",
        url=f"{_UF}/639202560055329291?handler=Download",
        kind="bk2",
        source="m_riss 100% WIP userfile (first 10k frames)",
        notes="USA ROM, BizHawk 2.11. Tiny opening smoke; not a finished 100%.",
        category="100% wip",
        sha256="e3ddf701de31861738642fdc04e0043c3ce7afa24f6d32d009dd1bb6eacfe6f0",
        expected_frames=10_000,
        emulator="BizHawk 2.11.1 BSNESv115+",
        region="US",
        tags=("100%", "wip", "usa", "short", "opening"),
    ),
    MovieRef(
        filename="taseditor_glitched.lsmv",
        url=f"{_UF}/8710388382096206?handler=Download",
        kind="lsmv",
        source="TASeditor glitched any% userfile",
        notes="2:15 glitched any% (8127f). Ancestor of published GEG.",
        category="glitched any%",
        sha256="6f73361aef0e0a132f9e65358ae33ef21e9d429715b7af8808163d08eb290714",
        expected_frames=8127,
        emulator="lsnes (bsnes v085)",
        tags=("glitch", "wip", "short"),
    ),
    MovieRef(
        filename="fmp_nmg.bk2",
        url=f"{_UF}/67832544397586850?handler=Download",
        kind="bk2",
        source="fmp NMG TAS userfile (NMGttas.bk2)",
        notes="31:22 JP NMG. Likely covers castle/sewers/sanctuary; windows unverified.",
        category="nmg",
        sha256="524c2a2004334246389aff1f71dd77fe543102a3eefb97823de2594758416b3a",
        expected_frames=113_135,
        emulator="BizHawk 2.3 BSNES",
        tags=("nmg", "opening", "userfile"),
    ),
    MovieRef(
        filename="tompa_1269M.smv",
        url=f"{_TV}/1269M?handler=Download",
        kind="smv",
        source="Tompa no-major-glitch #1269M",
        notes="Published USA full completion 1:16:11 (Snes9x). Best published opening ref.",
        category="no major glitch",
        sha256="ec38484d46021b8b4cad7dff4c719354870ba10fdb183ccfb270440b93658df2",
        expected_frames=274_264,
        emulator="Snes9x",
        region="US",
        tags=("nmg", "published", "usa", "opening"),
    ),
    MovieRef(
        filename="fmp_yuzuhara_fullinv_3874M.bk2",
        url=f"{_TV}/3874M?handler=Download",
        kind="bk2",
        source="fmp & Yuzuhara_3 full inventory #3874M",
        notes="Published JP 100%/full inventory 52:52. Glitch-heavy; not NMG.",
        category="full inventory",
        sha256="2df07e38297f770188f98f8d7fd78df3e4c752f1c5c6ebf2c015c7d3faf13128",
        expected_frames=190_660,
        emulator="BizHawk 2.3 BSNES",
        tags=("100%", "published", "glitch"),
    ),
)


SKIPPED: tuple[MovieRef, ...] = (
    MovieRef(
        filename="TheLegendofZelda-ALinktothePast-114.wch",
        url=f"{_UF}/637936906508015623?handler=Download",
        kind="bk2",
        source="BizHawk RAM watch",
        notes="Not a movie.",
        category="watch",
        sha256="",
        expected_frames=0,
        vanilla=False,
        fetch=False,
        skip_reason="RAM watch, not button presses.",
    ),
    MovieRef(
        filename="minimap.lua",
        url=f"{_UF}/51554590290287988?handler=Download",
        kind="bk2",
        source="fmp minimap.lua",
        notes="Lua overlay, not a movie.",
        category="lua",
        sha256="",
        expected_frames=0,
        vanilla=False,
        fetch=False,
        skip_reason="Lua overlay, not button presses.",
    ),
    MovieRef(
        filename="maptracker.lua",
        url=f"{_UF}/51326125340200575?handler=Download",
        kind="bk2",
        source="fmp maptracker.lua",
        notes="Lua overlay, not a movie.",
        category="lua",
        sha256="",
        expected_frames=0,
        vanilla=False,
        fetch=False,
        skip_reason="Lua overlay, not button presses.",
    ),
)


def fetchable() -> tuple[MovieRef, ...]:
    return tuple(m for m in MOVIES if m.fetch)


def by_filename(name: str) -> MovieRef:
    for movie in MOVIES:
        if movie.filename == name:
            return movie
    raise KeyError(name)
