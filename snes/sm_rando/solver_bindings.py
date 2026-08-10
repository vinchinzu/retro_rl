"""Evidence-checked bindings from SM randomizer logic edges to vanilla skills."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from retro_harness.adventure.bindings import (
    BindingCatalog,
    EdgeEvidence,
    ExecutionReadiness,
    PromotionPolicy,
    SkillBinding,
)
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillSpec,
)
from retro_harness.identity import canonical_json, sha256_bytes, sha256_file
from sm_rando.logic_graph import N_MORPH, N_SHIP
from sm_rando.paths import RECORDINGS_DIR, REPO_ROOT
from sm_rando.solver_adapter import OBSERVATION_SCHEMA_DIGEST
from super_metroid.routes.kpdr.early_spine import play_ship_to_morph

SHIP_TO_MORPH_EVIDENCE = RECORDINGS_DIR / "ship_to_morph.evidence.json"
SM_RANDO_OBSERVATION_SCHEMA_DIGEST = OBSERVATION_SCHEMA_DIGEST

SHIP_TO_MORPH_SPEC = SkillSpec(
    skill_id="sm.ship_to_morph.vanilla_pure",
    dispatch_key=(
        "super_metroid.routes.kpdr.early_spine:play_ship_to_morph"
    ),
    observation_requirement=ObservationRequirement(
        schema_digest=SM_RANDO_OBSERVATION_SCHEMA_DIGEST,
        allowed_nodes=(N_SHIP,),
        required_values={"game_state": 8, "room_id": 0x91F8},
    ),
    expected_delta=ProgressionDelta(
        target_node=N_MORPH,
        acquired_capabilities=frozenset({"morph_ball"}),
    ),
    timeout_frames=8_000,
    max_retries=1,
)

_SCAFFOLD_BINDING = SkillBinding(
    edge_id="ship_to_morph",
    skill_id=SHIP_TO_MORPH_SPEC.skill_id,
    dispatch_key=SHIP_TO_MORPH_SPEC.dispatch_key,
    entry_requirement_digest=(
        SHIP_TO_MORPH_SPEC.observation_requirement.identity_digest
    ),
    progression_delta_digest=SHIP_TO_MORPH_SPEC.expected_delta.identity_digest,
)


def _observation_digest(
    *,
    frame: int,
    node_id: str,
    room_id: int,
    morph_ball: bool,
    report_sha256: str,
) -> str:
    record = {
        "frame": frame,
        "morph_ball": morph_ball,
        "node_id": node_id,
        "report_sha256": report_sha256,
        "room_id": room_id,
        "schema_digest": SM_RANDO_OBSERVATION_SCHEMA_DIGEST,
    }
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def _resolve_repo_path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source_report must be a non-empty repository path")
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("source_report escapes repository root") from exc
    return path


def load_ship_to_morph_evidence(
    path: Path = SHIP_TO_MORPH_EVIDENCE,
) -> EdgeEvidence:
    """Validate a retained real-run report and return promotion evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported ship-to-morph evidence schema")
    if payload.get("edge_id") != _SCAFFOLD_BINDING.edge_id:
        raise ValueError("evidence edge_id mismatch")

    source_path = _resolve_repo_path(payload.get("source_report"))
    source_sha256 = sha256_file(source_path)
    if payload.get("source_report_sha256") != source_sha256:
        raise ValueError("source report digest mismatch")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if not report.get("success") or report.get("outcome") != "morph_ball_acquired":
        raise ValueError("source report is not a successful Morph Ball clear")
    if report.get("state_loads") != 0 or report.get("progression_writes") != 0:
        raise ValueError("source report is not clean natural-entry evidence")

    split_by_id = {
        split["split_id"]: split for split in report.get("splits", ())
    }
    entry = split_by_id.get("zebes_landing")
    exit_ = split_by_id.get("morph_ball")
    if not entry or not exit_ or entry["frame"] >= exit_["frame"]:
        raise ValueError("source report lacks ordered landing/morph splits")
    final = report.get("final_state", {})
    if not final.get("morph_ball") or final.get("room_id") != 0x9E9F:
        raise ValueError("source report final state is not Morph Ball acquired")

    entry_digest = _observation_digest(
        frame=int(entry["frame"]),
        node_id=N_SHIP,
        room_id=int(entry["room_id"]),
        morph_ball=False,
        report_sha256=source_sha256,
    )
    exit_digest = _observation_digest(
        frame=int(exit_["frame"]),
        node_id=N_MORPH,
        room_id=int(exit_["room_id"]),
        morph_ball=True,
        report_sha256=source_sha256,
    )
    return EdgeEvidence(
        edge_id=_SCAFFOLD_BINDING.edge_id,
        binding_digest=_SCAFFOLD_BINDING.identity_digest,
        readiness=ExecutionReadiness.NATURAL_ENTRY,
        predecessor_edge_id="zebes_landing",
        predecessor_exit_observation_digest=entry_digest,
        target_entry_observation_digest=entry_digest,
        target_exit_observation_digest=exit_digest,
        attempts=int(payload.get("attempts", 1)),
        successes=int(payload.get("successes", 1)),
        artifact_digest=source_sha256,
    )


def ship_to_morph_binding(
    path: Path = SHIP_TO_MORPH_EVIDENCE,
) -> SkillBinding:
    """Return the binding promoted only after retained evidence validates."""
    return PromotionPolicy().promote(
        _SCAFFOLD_BINDING,
        load_ship_to_morph_evidence(path),
    )


def build_early_binding_catalog() -> BindingCatalog:
    return BindingCatalog((ship_to_morph_binding(),))


__all__ = [
    "SHIP_TO_MORPH_EVIDENCE",
    "SHIP_TO_MORPH_SPEC",
    "SM_RANDO_OBSERVATION_SCHEMA_DIGEST",
    "build_early_binding_catalog",
    "load_ship_to_morph_evidence",
    "play_ship_to_morph",
    "ship_to_morph_binding",
]
