"""Named Super Metroid TAS slices (Sniq any% / 100%).

Frame windows are **movie-relative** (power-on index). They are intentional
coarse cuts for finish-route work (late MB/escape, full seeds, menu), not
room-ID-verified control points — re-anchor under the harness before STATUS
claims.

Sources
-------
- any%: Sniq #3653M / submission #5833, lsnes ``.lsmv``, 129_712 frames
- 100%: Sniq 100% converted BK2 (feos userfile), 222_789 frames
- any% WIP: Sniq userfile to Red Brinstar 2nd visit, 55_037 frames
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from super_metroid.paths import GAME_DIR
from super_metroid.tas.bk2 import parse_bk2
from super_metroid.tas.lsmv import parse_lsmv
from super_metroid.tas.rle import frames_to_snes12_rle_payload, write_snes12_rle_seed

TAS_DIR = GAME_DIR / "tas"
REF_DIR = TAS_DIR / "ref"
SLICE_DIR = TAS_DIR / "slices"

REF_ANY = REF_DIR / "sniq_any_3653M.lsmv"
REF_ANY_WIP = REF_DIR / "sniq_any_wip.lsmv"
REF_100 = REF_DIR / "sniq_100p.bk2"
REF_SMTC4 = REF_DIR / "moozooh_smtc4.bk2"

MovieKind = Literal["lsmv", "bk2"]


@dataclass(frozen=True)
class SliceSpec:
    """One named export from a vendored movie."""

    id: str
    movie: Path
    kind: MovieKind
    start: int
    end: int | None  # exclusive; None = EOF
    source: str
    notes: str
    tags: tuple[str, ...] = ()

    def resolve_end(self, num_frames: int) -> int:
        if self.end is None:
            return num_frames
        if self.end < 0:
            return num_frames + self.end
        return min(self.end, num_frames)


# Frame counts verified by parsers (see tests).
ANY_FRAMES = 129_712
HUNDRED_FRAMES = 222_789
ANY_WIP_FRAMES = 55_037

# Coarse finish-oriented windows (any% Sniq). Last ~15k / ~10k cover
# Tourian-ish through ship based on activity tail of the movie.
SLICE_CATALOG: dict[str, SliceSpec] = {
    # --- any% full + chapters ---
    "sniq_any_full": SliceSpec(
        id="sniq_any_full",
        movie=REF_ANY,
        kind="lsmv",
        start=0,
        end=None,
        source="Sniq any% #3653M LSMV (lsnes rr2)",
        notes="Full power-on → credits inputs. Core desync risk on non-lsnes.",
        tags=("any%", "full", "finish"),
    ),
    "sniq_any_menu": SliceSpec(
        id="sniq_any_menu",
        movie=REF_ANY,
        kind="lsmv",
        start=0,
        end=600,
        source="Sniq any% #3653M",
        notes="Title/file menu Start+A mash (~first activity block).",
        tags=("any%", "menu"),
    ),
    "sniq_any_ceres_open": SliceSpec(
        id="sniq_any_ceres_open",
        movie=REF_ANY,
        kind="lsmv",
        start=8_639,
        end=13_235,
        source="Sniq any% #3653M",
        notes="First post-intro movement block (Ceres-ish open).",
        tags=("any%", "ceres", "early"),
    ),
    "sniq_any_midgame": SliceSpec(
        id="sniq_any_midgame",
        movie=REF_ANY,
        kind="lsmv",
        start=40_103,
        end=74_336,
        source="Sniq any% #3653M",
        notes="Long middle activity block (Norfair/Maridia-ish by length).",
        tags=("any%", "mid"),
    ),
    "sniq_any_late": SliceSpec(
        id="sniq_any_late",
        movie=REF_ANY,
        kind="lsmv",
        start=83_933,
        end=110_526,
        source="Sniq any% #3653M",
        notes="Long late block (pre-Tourian / bosses by position).",
        tags=("any%", "late", "finish"),
    ),
    "sniq_any_tourian_escape": SliceSpec(
        id="sniq_any_tourian_escape",
        movie=REF_ANY,
        kind="lsmv",
        start=111_291,
        end=None,
        source="Sniq any% #3653M",
        notes="Tail: last two activity blocks through movie end (MB/escape/ship).",
        tags=("any%", "tourian", "mb", "escape", "finish"),
    ),
    "sniq_any_final_10k": SliceSpec(
        id="sniq_any_final_10k",
        movie=REF_ANY,
        kind="lsmv",
        start=ANY_FRAMES - 10_000,
        end=None,
        source="Sniq any% #3653M",
        notes="Last 10k frames — escape/ship polish reference.",
        tags=("any%", "escape", "finish"),
    ),
    # --- 100% ---
    "sniq_100_full": SliceSpec(
        id="sniq_100_full",
        movie=REF_100,
        kind="bk2",
        start=0,
        end=None,
        source="Sniq 100% BK2 (feos converter userfile #55928342467251616)",
        notes="Full 100% button log. Prefer for item-route reference.",
        tags=("100%", "full", "finish"),
    ),
    "sniq_100_menu": SliceSpec(
        id="sniq_100_menu",
        movie=REF_100,
        kind="bk2",
        start=0,
        end=600,
        source="Sniq 100% BK2",
        notes="Menu Start+A mash (matches any% timing).",
        tags=("100%", "menu"),
    ),
    "sniq_100_ceres_open": SliceSpec(
        id="sniq_100_ceres_open",
        movie=REF_100,
        kind="bk2",
        start=8_640,
        end=13_237,
        source="Sniq 100% BK2",
        notes="First post-intro movement (aligned with any% open).",
        tags=("100%", "ceres", "early"),
    ),
    "sniq_100_late": SliceSpec(
        id="sniq_100_late",
        movie=REF_100,
        kind="bk2",
        start=HUNDRED_FRAMES - 30_000,
        end=None,
        source="Sniq 100% BK2",
        notes="Last 30k frames — late item cleanup + Tourian/MB/escape.",
        tags=("100%", "late", "finish"),
    ),
    "sniq_100_final_15k": SliceSpec(
        id="sniq_100_final_15k",
        movie=REF_100,
        kind="bk2",
        start=HUNDRED_FRAMES - 15_000,
        end=None,
        source="Sniq 100% BK2",
        notes="Last 15k — endgame finish reference.",
        tags=("100%", "escape", "finish"),
    ),
    # --- shorter refs ---
    "sniq_any_wip_full": SliceSpec(
        id="sniq_any_wip_full",
        movie=REF_ANY_WIP,
        kind="lsmv",
        start=0,
        end=None,
        source="Sniq any% WIP userfile (to Red Brinstar 2nd visit)",
        notes="Shorter early/mid route experiment; good parser + early slice tests.",
        tags=("any%", "wip", "early"),
    ),
    "moozooh_smtc4_full": SliceSpec(
        id="moozooh_smtc4_full",
        movie=REF_SMTC4,
        kind="bk2",
        start=0,
        end=None,
        source="moozooh SM TAS Contest Round 4 final BK2",
        notes="Short contest segment (Wrecked Ship E-Tank start). Smoke-test size.",
        tags=("contest", "short"),
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
        else:
            raise ValueError(f"unknown movie kind for {path}")
    if kind == "lsmv":
        return parse_lsmv(path).frames
    if kind == "bk2":
        return parse_bk2(path).frames
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
) -> dict[str, Any]:
    """Export one catalog slice to ``tas/slices/<id>.json``."""
    if isinstance(spec, str):
        if spec not in SLICE_CATALOG:
            raise KeyError(f"unknown slice {spec!r}; known={sorted(SLICE_CATALOG)}")
        spec = SLICE_CATALOG[spec]
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
        },
    )
    out = out_path or (SLICE_DIR / f"{spec.id}.json")
    write_snes12_rle_seed(out, payload)
    return payload


def finish_slice_ids() -> list[str]:
    """Slice ids tagged for game-finish work."""
    return [sid for sid, sp in SLICE_CATALOG.items() if "finish" in sp.tags]
