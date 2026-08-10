"""Evidence-checked bindings from ALTTP rando logic edges to vanilla skills."""

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
from retro_harness.contracts import ObservationContract, ObservationField
from retro_harness.identity import canonical_json, sha256_bytes, sha256_file
from retro_harness.solver import (
    ObservationRequirement,
    ProgressionDelta,
    SkillSpec,
)
from alttp_rando.house_to_uncle import (
    HOUSE_TO_UNCLE_EVIDENCE,
    OUTCOME_SWORD,
    PREDECESSOR_EDGE_ID,
    play_house_to_uncle,
    run_house_to_uncle_from_first_play,
)
from alttp_rando.logic_graph import N_LINKS_HOUSE, N_UNCLE
from alttp_rando.paths import REPO_ROOT

ALTTP_RANDO_OBSERVATION_CONTRACT = ObservationContract(
    fields=(
        ObservationField("room_base_id", "uint16", semantic="current room base id"),
        ObservationField("game_mode", "uint8", semantic="engine module / game mode"),
        ObservationField("indoors", "bool", semantic="indoor vs overworld"),
        ObservationField("has_control", "bool", semantic="player has control"),
        ObservationField("has_fighter_sword", "bool", semantic="fighter sword equip"),
        ObservationField("has_lamp", "bool", semantic="lamp inventory"),
    ),
    preprocessing={"adapter": "alttp_rando.solver_bindings/v1"},
)
ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST = ALTTP_RANDO_OBSERVATION_CONTRACT.identity_digest

HOUSE_TO_UNCLE_SPEC = SkillSpec(
    skill_id="z3.house_to_uncle.vanilla_opening",
    dispatch_key="alttp_rando.house_to_uncle:play_house_to_uncle",
    observation_requirement=ObservationRequirement(
        schema_digest=ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST,
        allowed_nodes=(N_LINKS_HOUSE,),
        required_values={
            "game_mode": 0x07,
            "room_base_id": 0x04,
            "indoors": True,
            "has_control": True,
            "has_fighter_sword": False,
        },
    ),
    expected_delta=ProgressionDelta(
        target_node=N_UNCLE,
        acquired_capabilities=frozenset({"sword"}),
    ),
    timeout_frames=12_000,
    max_retries=1,
)

_SCAFFOLD_BINDING = SkillBinding(
    edge_id="house_to_uncle",
    skill_id=HOUSE_TO_UNCLE_SPEC.skill_id,
    dispatch_key=HOUSE_TO_UNCLE_SPEC.dispatch_key,
    entry_requirement_digest=(
        HOUSE_TO_UNCLE_SPEC.observation_requirement.identity_digest
    ),
    progression_delta_digest=HOUSE_TO_UNCLE_SPEC.expected_delta.identity_digest,
)


def _observation_digest(
    *,
    frame: int,
    node_id: str,
    room_base_id: int,
    has_fighter_sword: bool,
    report_sha256: str,
) -> str:
    record = {
        "frame": frame,
        "has_fighter_sword": has_fighter_sword,
        "node_id": node_id,
        "report_sha256": report_sha256,
        "room_base_id": room_base_id,
        "schema_digest": ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST,
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


def load_house_to_uncle_evidence(
    path: Path = HOUSE_TO_UNCLE_EVIDENCE,
) -> EdgeEvidence:
    """Validate a retained FirstPlay→uncle report and return promotion evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported house-to-uncle evidence schema")
    if payload.get("edge_id") != _SCAFFOLD_BINDING.edge_id:
        raise ValueError("evidence edge_id mismatch")

    source_path = _resolve_repo_path(payload.get("source_report"))
    source_sha256 = sha256_file(source_path)
    if payload.get("source_report_sha256") != source_sha256:
        raise ValueError("source report digest mismatch")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if not report.get("success") or report.get("outcome") != OUTCOME_SWORD:
        raise ValueError("source report is not a successful fighter-sword clear")
    if int(report.get("progression_writes", -1)) != 0:
        raise ValueError("source report is not clean (progression writes)")
    # Exactly one FirstPlay predecessor load is the natural-entry handoff.
    if int(report.get("state_loads", -1)) != 1:
        raise ValueError("source report must load only the FirstPlay predecessor")
    if report.get("predecessor_edge_id") != PREDECESSOR_EDGE_ID:
        raise ValueError("source report predecessor_edge_id mismatch")
    if not report.get("clean_chain"):
        raise ValueError("source report is not a clean natural-entry chain")

    split_by_id = {split["split_id"]: split for split in report.get("splits", ())}
    entry = split_by_id.get("links_house_control")
    exit_ = split_by_id.get("uncle_sword")
    if not entry or not exit_ or int(entry["frame"]) >= int(exit_["frame"]):
        raise ValueError("source report lacks ordered house/uncle splits")
    final = report.get("final_state", {})
    if not final.get("has_fighter_sword") or int(final.get("room_base_id", -1)) != 0x55:
        raise ValueError("source report final state is not uncle fighter sword")
    if int(entry.get("room_base_id", -1)) != 0x04 or not entry.get("has_control"):
        raise ValueError("source report entry is not Link's House control")

    entry_digest = _observation_digest(
        frame=int(entry["frame"]),
        node_id=N_LINKS_HOUSE,
        room_base_id=int(entry["room_base_id"]),
        has_fighter_sword=bool(entry.get("has_fighter_sword")),
        report_sha256=source_sha256,
    )
    exit_digest = _observation_digest(
        frame=int(exit_["frame"]),
        node_id=N_UNCLE,
        room_base_id=int(exit_["room_base_id"]),
        has_fighter_sword=bool(exit_.get("has_fighter_sword")),
        report_sha256=source_sha256,
    )
    return EdgeEvidence(
        edge_id=_SCAFFOLD_BINDING.edge_id,
        binding_digest=_SCAFFOLD_BINDING.identity_digest,
        readiness=ExecutionReadiness.NATURAL_ENTRY,
        predecessor_edge_id=PREDECESSOR_EDGE_ID,
        predecessor_exit_observation_digest=entry_digest,
        target_entry_observation_digest=entry_digest,
        target_exit_observation_digest=exit_digest,
        attempts=int(payload.get("attempts", 1)),
        successes=int(payload.get("successes", 1)),
        artifact_digest=source_sha256,
    )


def house_to_uncle_binding(
    path: Path = HOUSE_TO_UNCLE_EVIDENCE,
) -> SkillBinding:
    """Return the binding promoted only after retained evidence validates."""
    return PromotionPolicy().promote(
        _SCAFFOLD_BINDING,
        load_house_to_uncle_evidence(path),
    )


def build_early_binding_catalog() -> BindingCatalog:
    return BindingCatalog((house_to_uncle_binding(),))


__all__ = [
    "ALTTP_RANDO_OBSERVATION_SCHEMA_DIGEST",
    "HOUSE_TO_UNCLE_SPEC",
    "build_early_binding_catalog",
    "house_to_uncle_binding",
    "load_house_to_uncle_evidence",
    "play_house_to_uncle",
    "run_house_to_uncle_from_first_play",
]
