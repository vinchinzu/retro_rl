"""Executable graph-edge bindings with evidence-gated readiness."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from math import isfinite
from typing import Any, Iterable

from retro_harness.adventure.graph import GraphEdge
from retro_harness.identity import (
    digest_record as _digest_record,
    require_nonempty as _nonempty,
)


class ExecutionReadiness(IntEnum):
    """Typed evidence ladder for one executable edge binding."""

    SCAFFOLD = 0
    ISOLATED = 1
    NATURAL_ENTRY = 2
    CONTINUOUS = 3
    PUBLICATION_READY = 4


def _require_readiness(value: Any, field_name: str) -> ExecutionReadiness:
    if not isinstance(value, ExecutionReadiness):
        raise TypeError(f"{field_name} must be an ExecutionReadiness")
    return value


@dataclass(frozen=True, slots=True)
class SkillBinding:
    """Versioned dispatch binding for exactly one graph edge ID."""

    edge_id: str
    skill_id: str
    dispatch_key: str
    entry_requirement_digest: str
    progression_delta_digest: str
    version: str = "1"
    readiness: ExecutionReadiness = ExecutionReadiness.SCAFFOLD
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "edge_id",
            "skill_id",
            "dispatch_key",
            "entry_requirement_digest",
            "progression_delta_digest",
            "version",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "readiness",
            _require_readiness(self.readiness, "readiness"),
        )
        if self.readiness is ExecutionReadiness.SCAFFOLD:
            if self.evidence_digest is not None:
                raise ValueError("scaffold bindings cannot carry evidence_digest")
        else:
            object.__setattr__(
                self,
                "evidence_digest",
                _nonempty(self.evidence_digest, "evidence_digest"),
            )

    def _identity_record(self) -> dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "skill_id": self.skill_id,
            "dispatch_key": self.dispatch_key,
            "entry_requirement_digest": self.entry_requirement_digest,
            "progression_delta_digest": self.progression_delta_digest,
            "version": self.version,
        }

    @property
    def identity_digest(self) -> str:
        return _digest_record("skill-binding-v1", self._identity_record())

    def to_record(self) -> dict[str, Any]:
        return {
            **self._identity_record(),
            "identity_digest": self.identity_digest,
            "readiness": self.readiness.name,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class EdgeEvidence:
    """Digest-linked observations and outcomes supporting one promotion."""

    edge_id: str
    binding_digest: str
    readiness: ExecutionReadiness
    target_entry_observation_digest: str
    target_exit_observation_digest: str
    attempts: int
    successes: int
    predecessor_edge_id: str | None = None
    predecessor_exit_observation_digest: str | None = None
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "edge_id",
            "binding_digest",
            "target_entry_observation_digest",
            "target_exit_observation_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty(getattr(self, field_name), field_name),
            )
        readiness = _require_readiness(self.readiness, "readiness")
        if readiness is ExecutionReadiness.SCAFFOLD:
            raise ValueError("evidence cannot promote to SCAFFOLD")
        object.__setattr__(self, "readiness", readiness)
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 1
        ):
            raise ValueError("attempts must be a positive integer")
        if (
            isinstance(self.successes, bool)
            or not isinstance(self.successes, int)
            or not 0 <= self.successes <= self.attempts
        ):
            raise ValueError("successes must be between zero and attempts")
        if readiness >= ExecutionReadiness.NATURAL_ENTRY:
            predecessor_edge_id = _nonempty(
                self.predecessor_edge_id,
                "predecessor_edge_id",
            )
            predecessor_digest = _nonempty(
                self.predecessor_exit_observation_digest,
                "predecessor_exit_observation_digest",
            )
            if predecessor_digest != self.target_entry_observation_digest:
                raise ValueError(
                    "predecessor exit observation digest must match target entry "
                    "observation digest"
                )
            object.__setattr__(self, "predecessor_edge_id", predecessor_edge_id)
            object.__setattr__(
                self,
                "predecessor_exit_observation_digest",
                predecessor_digest,
            )
        elif (
            self.predecessor_edge_id is not None
            or self.predecessor_exit_observation_digest is not None
        ):
            raise ValueError(
                "isolated evidence cannot claim predecessor linkage"
            )
        if self.artifact_digest is not None:
            object.__setattr__(
                self,
                "artifact_digest",
                _nonempty(self.artifact_digest, "artifact_digest"),
            )

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts

    def _identity_record(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "binding_digest": self.binding_digest,
            "readiness": self.readiness.name,
            "target_entry_observation_digest": self.target_entry_observation_digest,
            "target_exit_observation_digest": self.target_exit_observation_digest,
            "predecessor_edge_id": self.predecessor_edge_id,
            "predecessor_exit_observation_digest": (
                self.predecessor_exit_observation_digest
            ),
            "attempts": self.attempts,
            "successes": self.successes,
            "artifact_digest": self.artifact_digest,
        }

    @property
    def identity_digest(self) -> str:
        return _digest_record("edge-evidence-v1", self._identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self._identity_record(), "identity_digest": self.identity_digest}


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Success and publication thresholds for binding promotion."""

    minimum_attempts: int = 1
    minimum_success_rate: float = 1.0
    publication_minimum: ExecutionReadiness = ExecutionReadiness.NATURAL_ENTRY

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_attempts, bool)
            or not isinstance(self.minimum_attempts, int)
            or self.minimum_attempts < 1
        ):
            raise ValueError("minimum_attempts must be a positive integer")
        if (
            isinstance(self.minimum_success_rate, bool)
            or not isinstance(self.minimum_success_rate, (int, float))
            or not isfinite(float(self.minimum_success_rate))
            or not 0.0 <= float(self.minimum_success_rate) <= 1.0
        ):
            raise ValueError("minimum_success_rate must be between zero and one")
        object.__setattr__(
            self,
            "minimum_success_rate",
            float(self.minimum_success_rate),
        )
        object.__setattr__(
            self,
            "publication_minimum",
            _require_readiness(self.publication_minimum, "publication_minimum"),
        )

    def promote(self, binding: SkillBinding, evidence: EdgeEvidence) -> SkillBinding:
        if not isinstance(binding, SkillBinding):
            raise TypeError("binding must be a SkillBinding")
        if not isinstance(evidence, EdgeEvidence):
            raise TypeError("evidence must be an EdgeEvidence")
        if evidence.edge_id != binding.edge_id:
            raise ValueError("edge evidence does not match binding edge_id")
        if evidence.binding_digest != binding.identity_digest:
            raise ValueError("edge evidence does not match binding digest")
        if evidence.readiness <= binding.readiness:
            raise ValueError("edge evidence must advance binding readiness")
        if evidence.attempts < self.minimum_attempts:
            raise ValueError("edge evidence does not meet minimum_attempts")
        if evidence.success_rate < self.minimum_success_rate:
            raise ValueError("edge evidence does not meet minimum_success_rate")
        return replace(
            binding,
            readiness=evidence.readiness,
            evidence_digest=evidence.identity_digest,
        )

    def is_publication_ready(self, binding: SkillBinding) -> bool:
        return binding.readiness >= self.publication_minimum


class BindingCatalog:
    """Edge-ID keyed bindings; parallel graph edges stay independent."""

    def __init__(self, bindings: Iterable[SkillBinding] = ()) -> None:
        by_edge: dict[str, SkillBinding] = {}
        for binding in bindings:
            if not isinstance(binding, SkillBinding):
                raise TypeError("binding catalog values must be SkillBinding")
            if binding.edge_id in by_edge:
                raise ValueError(f"duplicate binding edge ID: {binding.edge_id!r}")
            by_edge[binding.edge_id] = binding
        self._by_edge = by_edge

    @property
    def bindings(self) -> tuple[SkillBinding, ...]:
        return tuple(self._by_edge[key] for key in sorted(self._by_edge))

    def binding_for(self, edge_id: str) -> SkillBinding | None:
        return self._by_edge.get(edge_id)

    def with_binding(self, binding: SkillBinding) -> "BindingCatalog":
        values = dict(self._by_edge)
        values[binding.edge_id] = binding
        return BindingCatalog(values.values())

    def edges_for_planning(
        self,
        edges: Iterable[GraphEdge],
        *,
        minimum_readiness: ExecutionReadiness,
    ) -> tuple[GraphEdge, ...]:
        threshold = _require_readiness(minimum_readiness, "minimum_readiness")
        selected = []
        for edge in edges:
            binding = self.binding_for(edge.edge_id)
            if binding is not None and binding.readiness >= threshold:
                selected.append(edge)
        return tuple(sorted(selected, key=lambda edge: edge.edge_id))

    def publication_edges(
        self,
        edges: Iterable[GraphEdge],
        *,
        policy: PromotionPolicy | None = None,
    ) -> tuple[GraphEdge, ...]:
        promotion = policy or PromotionPolicy()
        return self.edges_for_planning(
            edges,
            minimum_readiness=promotion.publication_minimum,
        )
