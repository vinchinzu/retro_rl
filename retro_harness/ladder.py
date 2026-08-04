"""Ladder status tracker — entries loaded from docs/manifests/*.yaml setup blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_DIR = REPO_ROOT / "docs" / "manifests"
DEFAULT_ROM_LIBRARY = "Super Nintendo"
SHARED_ROM_DIR = REPO_ROOT / "roms" / DEFAULT_ROM_LIBRARY


class LadderStatus(Enum):
    """Per-game completion status on the oneshot ladder."""

    NOT_STARTED = auto()
    SCAFFOLDED = auto()
    BOOT_WORKS = auto()
    INPUT_BLOCKED = auto()
    SEGMENTS = auto()
    ENDING_SEGMENTED = auto()
    FULL_GAME = auto()


@dataclass(frozen=True)
class LadderEntry:
    """One game on the recommended oneshot ladder."""

    rank: int
    slug: str
    title: str
    rom_zip: str
    integration: str
    tier: int
    status: LadderStatus = LadderStatus.NOT_STARTED
    video: str | None = None
    rom_library: str = DEFAULT_ROM_LIBRARY


def _parse_ladder_status(
    raw: object,
    *,
    maturity: object = None,
    project_state: object = None,
    best_video: object = None,
) -> LadderStatus:
    """Resolve LadderStatus from setup.ladder_status or maturity heuristics."""
    if raw is not None:
        name = str(raw).strip().upper()
        try:
            return LadderStatus[name]
        except KeyError as exc:
            known = ", ".join(s.name for s in LadderStatus)
            raise ValueError(
                f"unknown ladder_status {raw!r}; expected one of: {known}"
            ) from exc

    maturity_s = str(maturity or "").strip().upper()
    state_s = str(project_state or "").strip().lower()
    has_credits_video = bool(best_video) and "credit" in str(best_video).lower()
    if maturity_s in {"M7", "M8"} or (
        state_s == "verified" and has_credits_video
    ):
        return LadderStatus.FULL_GAME
    return LadderStatus.NOT_STARTED


def _video_from_manifest(setup: dict[str, object], data: dict[str, object]) -> str | None:
    """Prefer setup.video; else best_video stripped of console/ prefix when useful."""
    if setup.get("video"):
        return str(setup["video"])
    best = data.get("best_video")
    if not best:
        return None
    text = str(best)
    # Manifests often store snes/<slug>/...; ladder historically used <slug>/...
    for prefix in ("snes/", "nes/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _entry_from_manifest(data: dict[str, object], path: Path) -> LadderEntry | None:
    """Build a LadderEntry if the manifest has setup.ladder_rank; else None."""
    setup = data.get("setup")
    if not isinstance(setup, dict):
        return None
    rank = setup.get("ladder_rank")
    if rank is None:
        return None

    missing = [k for k in ("rom_zip", "integration") if not setup.get(k)]
    if missing:
        raise ValueError(f"{path}: setup missing required keys {missing}")

    slug = str(data.get("game") or path.stem)
    title = str(data.get("title") or slug)
    tier_raw = setup.get("tier", data.get("capability_phase", 0))
    rom_library = str(setup.get("rom_library") or DEFAULT_ROM_LIBRARY)

    return LadderEntry(
        rank=int(rank),
        slug=slug,
        title=title,
        rom_zip=str(setup["rom_zip"]),
        integration=str(setup["integration"]),
        tier=int(tier_raw),
        status=_parse_ladder_status(
            setup.get("ladder_status"),
            maturity=data.get("maturity"),
            project_state=data.get("project_state"),
            best_video=data.get("best_video"),
        ),
        video=_video_from_manifest(setup, data),
        rom_library=rom_library,
    )


def load_ladder(manifest_dir: Path | None = None) -> tuple[LadderEntry, ...]:
    """Load ladder entries from manifests that declare setup.ladder_rank.

    Entries are ordered by ladder_rank ascending. Duplicate ranks raise.
    """
    root = Path(manifest_dir) if manifest_dir is not None else DEFAULT_MANIFEST_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"manifest directory not found: {root}")

    entries: list[LadderEntry] = []
    for path in sorted(root.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        entry = _entry_from_manifest(raw, path)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e.rank)
    ranks = [e.rank for e in entries]
    if len(ranks) != len(set(ranks)):
        raise ValueError(f"duplicate ladder_rank values in {root}: {ranks}")
    return tuple(entries)


# Built from docs/manifests/*.yaml (setup.ladder_rank) at import time.
LADDER: tuple[LadderEntry, ...] = load_ladder()


def entry_for(slug: str) -> LadderEntry:
    """Look up a ladder entry by game directory slug."""
    for entry in LADDER:
        if entry.slug == slug:
            return entry
    raise KeyError(f"Unknown ladder slug: {slug}")


def shared_rom_zip(entry: LadderEntry, *, repo_root: Path = REPO_ROOT) -> Path:
    """Absolute path to the shared ROM zip for a ladder entry."""
    return repo_root / "roms" / entry.rom_library / entry.rom_zip
