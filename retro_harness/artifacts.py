"""Clean-track artifact naming shared by continuous full-run games.

Assisted and Clean basenames must never collide. Games keep their own stems
and recordings dirs; this module only owns the stem rewrite + path pair shape.
"""

from __future__ import annotations

from pathlib import Path


def clean_artifact_stem(stem: str) -> str:
    """Append ``_clean`` once so Clean runs never share assisted basenames."""
    if stem.endswith("_clean"):
        return stem
    return f"{stem}_clean"


def recording_artifacts(
    recordings_dir: str | Path,
    stem: str,
    *,
    clean: bool = False,
    dry_run: bool = False,
    dry_run_suffix: str = "_dry_run",
) -> tuple[Path, Path]:
    """Return ``(video.mp4, report.json)`` under ``recordings_dir``.

    When ``clean=True``, the stem is rewritten via :func:`clean_artifact_stem`.
    When ``dry_run=True``, the report becomes ``{stem}{dry_run_suffix}.json``
    (video basename unchanged).
    """
    root = Path(recordings_dir)
    root.mkdir(parents=True, exist_ok=True)
    resolved = clean_artifact_stem(stem) if clean else stem
    video = root / f"{resolved}.mp4"
    if dry_run:
        report = root / f"{resolved}{dry_run_suffix}.json"
    else:
        report = root / f"{resolved}.json"
    return video, report
