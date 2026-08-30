"""Tape-scaffold splice planning over ``tips.play_hops``.

Planning/verification layer, not a second runner. Resolve selected artifacts
by digest before boot, rewrite host-absolute paths, and report the first
uncovered route edge. Does not boot an emulator.
"""

from __future__ import annotations

from super_metroid.splice.errors import PreflightError, SpliceError
from super_metroid.splice.preflight import (
    ArtifactRef,
    CorePreflight,
    HopPreflight,
    InventoryRegression,
    PreflightReport,
    RomPreflight,
    SegmentArtifacts,
    file_digest,
    format_preflight_summary,
    repo_relative,
    run_preflight,
)

__all__ = [
    "ArtifactRef",
    "CorePreflight",
    "HopPreflight",
    "InventoryRegression",
    "PreflightError",
    "PreflightReport",
    "RomPreflight",
    "SegmentArtifacts",
    "SpliceError",
    "file_digest",
    "format_preflight_summary",
    "repo_relative",
    "run_preflight",
]
