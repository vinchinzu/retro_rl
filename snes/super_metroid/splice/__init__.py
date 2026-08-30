"""Tape-scaffold splice planning over ``tips.play_hops``.

Eventual public surface (not in this package yet):

- ``prepare(task_id)`` — validate and materialize an immutable room start
- ``grade(prepared, candidate_ref)`` — replay, Join, successor probe
- ``assemble(route_id, selection)`` — one emulator session via ``play_hops``

This module is a planning/verification layer, not a second runner. Phase 0
lands digest preflight only: resolve selected artifacts before boot, rewrite
host-absolute paths, and report the first uncovered route edge.
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
