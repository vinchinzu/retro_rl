"""Named ALttP TAS slices (verified short refs + unverified opening stubs).

Frame windows are **movie-relative** (power-on index). Castle / sewers /
sanctuary entries are stubs — intended cuts, not measured, not claimed to
play under the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alttp.paths import GAME_DIR
from alttp.tas.bk2 import parse_bk2
from alttp.tas.catalog import REF_DIR, SLICE_DIR, MovieKind
from alttp.tas.lsmv import parse_lsmv
from alttp.tas.rle import frames_to_snes12_rle_payload, write_snes12_rle_seed

REF_GEG = REF_DIR / "fmp_geg_3898M.lsmv"
REF_WIP = REF_DIR / "m_riss_100_wip_10k.bk2"
REF_GLITCHED = REF_DIR / "taseditor_glitched.lsmv"
REF_NMG = REF_DIR / "fmp_nmg.bk2"
REF_TOMPA = REF_DIR / "tompa_1269M.smv"
REF_FULLINV = REF_DIR / "fmp_yuzuhara_fullinv_3874M.bk2"

GEG_FRAMES = 3277
WIP_FRAMES = 10_000
GLITCHED_FRAMES = 8127
NMG_FRAMES = 113_135
TOMPA_FRAMES = 274_264
FULLINV_FRAMES = 190_660


@dataclass(frozen=True)
class SliceSpec:
    """One named export from a vendored movie."""

    id: str
    movie: Path
    kind: MovieKind
    start: int | None  # None = unverified stub (do not export)
    end: int | None  # exclusive; None = EOF when start is set
    source: str
    notes: str
    tags: tuple[str, ...] = ()
    verified: bool = True

    def resolve_end(self, num_frames: int) -> int:
        if self.end is None:
            return num_frames
        if self.end < 0:
            return num_frames + self.end
        return min(self.end, num_frames)


SLICE_CATALOG: dict[str, SliceSpec] = {
    "fmp_geg_full": SliceSpec(
        id="fmp_geg_full",
        movie=REF_GEG,
        kind="lsmv",
        start=0,
        end=None,
        source="fmp/total/Yuzuhara_3 GEG #3898M",
        notes="Full 3277-frame game-end-glitch movie. JP 1.0. Not an opening route.",
        tags=("glitch", "geg", "full"),
    ),
    "m_riss_100_wip_full": SliceSpec(
        id="m_riss_100_wip_full",
        movie=REF_WIP,
        kind="bk2",
        start=0,
        end=None,
        source="m_riss 100% WIP 10k",
        notes="First 10k frames of a USA 100% WIP. Opening smoke only.",
        tags=("100%", "wip", "usa", "full", "opening"),
    ),
    "m_riss_100_wip_menu": SliceSpec(
        id="m_riss_100_wip_menu",
        movie=REF_WIP,
        kind="bk2",
        start=0,
        end=1200,
        source="m_riss 100% WIP 10k",
        notes="Boot/menu through first Start (~frame 958). Parser smoke.",
        tags=("100%", "wip", "usa", "menu"),
    ),
    "taseditor_glitched_full": SliceSpec(
        id="taseditor_glitched_full",
        movie=REF_GLITCHED,
        kind="lsmv",
        start=0,
        end=None,
        source="TASeditor glitched any% userfile",
        notes="Full 8127-frame glitched any%. Not sanctuary spine.",
        tags=("glitch", "full"),
    ),
    # --- intended opening cuts; windows not measured, do not claim they play ---
    "castle": SliceSpec(
        id="castle",
        movie=REF_TOMPA,
        kind="smv",
        start=None,
        end=None,
        source="Tompa #1269M (intended)",
        notes=(
            "Intended: house → courtyard → secret entrance → main hall. "
            "Frame window not measured. Do not claim this plays."
        ),
        tags=("stub", "opening", "castle", "unverified"),
        verified=False,
    ),
    "sewers": SliceSpec(
        id="sewers",
        movie=REF_NMG,
        kind="bk2",
        start=None,
        end=None,
        source="fmp NMG userfile (intended)",
        notes=(
            "Intended: B1 lamp / keys / Zelda cell. "
            "Frame window not measured. Do not claim this plays."
        ),
        tags=("stub", "opening", "sewers", "unverified"),
        verified=False,
    ),
    "sanctuary": SliceSpec(
        id="sanctuary",
        movie=REF_NMG,
        kind="bk2",
        start=None,
        end=None,
        source="fmp NMG userfile (intended)",
        notes=(
            "Intended: Zelda escort 0x50 → Sanctuary. "
            "Frame window not measured. Do not claim this plays."
        ),
        tags=("stub", "opening", "sanctuary", "unverified"),
        verified=False,
    ),
}


def load_movie_frames(path: Path | str, kind: MovieKind | None = None) -> list[list[int]]:
    """Load SNES-12 frames from a ref movie path."""
    path = Path(path)
    if kind is None:
        suf = path.suffix.lower()
        if suf == ".lsmv":
            kind = "lsmv"
        elif suf == ".bk2":
            kind = "bk2"
        elif suf == ".smv":
            kind = "smv"
        else:
            raise ValueError(f"unknown movie kind for {path}")
    if kind == "lsmv":
        return parse_lsmv(path).frames
    if kind == "bk2":
        return parse_bk2(path).frames
    if kind == "smv":
        from alttp.tas.smv import parse_smv_env

        return parse_smv_env(path).frames
    raise ValueError(f"bad kind {kind}")


def slice_frames(
    frames: list[list[int]],
    start: int,
    end: int | None,
) -> list[list[int]]:
    """Return frames[start:end] with bounds checks."""
    n = len(frames)
    if start < 0 or start >= n:
        raise ValueError(f"start {start} out of range for {n} frames")
    stop = n if end is None else (n + end if end < 0 else end)
    stop = min(max(stop, start), n)
    return [list(fr) for fr in frames[start:stop]]


def export_slice(
    spec: SliceSpec | str,
    *,
    out_path: Path | None = None,
    frames: list[list[int]] | None = None,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Export one catalog slice to ``tas/slices/<id>.json``."""
    if isinstance(spec, str):
        if spec not in SLICE_CATALOG:
            raise KeyError(f"unknown slice {spec!r}; known={sorted(SLICE_CATALOG)}")
        spec = SLICE_CATALOG[spec]
    if not spec.verified and not allow_unverified:
        raise ValueError(
            f"slice {spec.id!r} is an unverified stub; will not export "
            "(frame window not measured, does not claim to play)"
        )
    if spec.start is None:
        raise ValueError(f"slice {spec.id!r} has no start index")
    if not spec.movie.exists():
        raise FileNotFoundError(f"missing ref movie: {spec.movie}")

    if frames is None:
        frames = load_movie_frames(spec.movie, spec.kind)
    stop = spec.resolve_end(len(frames))
    body = slice_frames(frames, spec.start, stop)
    rel_movie = spec.movie
    try:
        rel_movie = spec.movie.relative_to(GAME_DIR)
    except ValueError:
        pass
    payload = frames_to_snes12_rle_payload(
        body,
        route_id=spec.id,
        source=spec.source,
        extra={
            "movie": str(rel_movie).replace("\\", "/"),
            "movie_kind": spec.kind,
            "movie_start_index": spec.start,
            "movie_end_index": stop,
            "movie_num_frames": len(frames),
            "notes": spec.notes,
            "tags": list(spec.tags),
            "verified": spec.verified,
        },
    )
    out = out_path or (SLICE_DIR / f"{spec.id}.json")
    write_snes12_rle_seed(out, payload)
    return payload


def verified_slice_ids() -> list[str]:
    return [sid for sid, sp in SLICE_CATALOG.items() if sp.verified]


def stub_slice_ids() -> list[str]:
    return [sid for sid, sp in SLICE_CATALOG.items() if not sp.verified]
