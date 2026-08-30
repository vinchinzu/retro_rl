"""Tape-scaffold splice planning over ``tips.play_hops``.

Planning/verification layer, not a second runner. Resolve selected artifacts
by digest before boot, rewrite host-absolute paths, and report the first
uncovered route edge. Does not boot an emulator.
"""

from __future__ import annotations

from super_metroid.splice.cards import assembly_table, generate_cards
from super_metroid.splice.errors import PrepareError, PreflightError, SchemaError, SpliceError
from super_metroid.splice.manifest import load_manifest, manifest_from_board
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
from super_metroid.splice.prepare import PreparedTask, prepare
from super_metroid.splice.schema import (
    CANDIDATE_KINDS,
    FORBIDDEN_HOT_FILES,
    INTERVENTION_PROFILES,
    NON_CLAIMS,
    CandidateArtifact,
    EntryFingerprint,
    JoinPredicate,
    LeaveSpecRef,
    RouteEdge,
    RouteManifest,
    TaskCard,
)

__all__ = [
    "ArtifactRef",
    "CANDIDATE_KINDS",
    "CandidateArtifact",
    "CorePreflight",
    "EntryFingerprint",
    "FORBIDDEN_HOT_FILES",
    "HopPreflight",
    "INTERVENTION_PROFILES",
    "InventoryRegression",
    "JoinPredicate",
    "LeaveSpecRef",
    "NON_CLAIMS",
    "PreparedTask",
    "PrepareError",
    "PreflightError",
    "PreflightReport",
    "RomPreflight",
    "RouteEdge",
    "RouteManifest",
    "SchemaError",
    "SegmentArtifacts",
    "SpliceError",
    "TaskCard",
    "assembly_table",
    "file_digest",
    "format_preflight_summary",
    "generate_cards",
    "load_manifest",
    "manifest_from_board",
    "prepare",
    "repo_relative",
    "run_preflight",
]
