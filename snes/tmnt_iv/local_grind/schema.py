"""Typed experiment records for the local grind loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class TrialDecision(str, Enum):
    """Keep/discard outcome for one trial."""

    KEEP = "keep"
    DISCARD = "discard"
    BASELINE = "baseline"
    ERROR = "error"


@dataclass(frozen=True)
class ProbeTarget:
    """One cheap eval checkpoint."""

    state: str
    max_frames: int
    stop_stage_gt: int | None
    label: str


# Default cheap targets ranked by remaining damage ROI.
# Prefer RaphFullHard* (continuous Raphael) over legacy Leo FullHard*.
DEFAULT_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(
        state="RaphFullHardBoss5",
        max_frames=40_000,
        stop_stage_gt=4,
        label="slash",
    ),
    ProbeTarget(
        state="RaphFullHardTank",
        max_frames=20_000,
        stop_stage_gt=3,
        label="technodrome_tank",
    ),
    ProbeTarget(
        state="RaphFullHardDuo",
        max_frames=25_000,
        stop_stage_gt=3,
        label="tokka_rahzar",
    ),
    ProbeTarget(
        state="RaphFullHardStage4",
        max_frames=45_000,
        stop_stage_gt=3,
        label="technodrome_full",
    ),
    # Leo legacy (only for regression / comparison)
    ProbeTarget(
        state="FullHardBoss5",
        max_frames=40_000,
        stop_stage_gt=4,
        label="slash_leo",
    ),
    ProbeTarget(
        state="FullHardTank",
        max_frames=20_000,
        stop_stage_gt=3,
        label="technodrome_tank_leo",
    ),
    ProbeTarget(
        state="FullHardFinale",
        max_frames=45_000,
        stop_stage_gt=None,
        label="super_shredder",
    ),
)


@dataclass
class ExperimentProposal:
    """JSON shape the local model must return."""

    hypothesis: str
    target_label: str
    knobs: dict[str, int] = field(default_factory=dict)
    rationale: str = ""

    @classmethod
    def from_mapping(cls, raw: Any) -> ExperimentProposal:
        if not isinstance(raw, dict):
            raise ValueError("proposal must be a JSON object")
        knobs = normalize_knobs(raw.get("knobs", {}))
        hypothesis = str(raw.get("hypothesis", "")).strip()
        target = str(raw.get("target_label", raw.get("target", ""))).strip()
        if not hypothesis:
            raise ValueError("hypothesis is required")
        if not target:
            raise ValueError("target_label is required")
        if not knobs:
            raise ValueError("knobs must contain at least one named int")
        return cls(
            hypothesis=hypothesis,
            target_label=target,
            knobs=knobs,
            rationale=str(raw.get("rationale", "")).strip(),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialRecord:
    """One propose → eval → decide cycle."""

    trial_id: int
    decision: TrialDecision
    proposal: ExperimentProposal | None
    metrics: dict[str, Any]
    score: float
    baseline_score: float
    delta_score: float
    image_paths: list[str] = field(default_factory=list)
    model_notes: str = ""
    error: str = ""

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        if self.proposal is not None:
            payload["proposal"] = self.proposal.to_jsonable()
        return payload


def _is_intable(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def normalize_knobs(raw: Any) -> dict[str, int]:
    """Accept object maps or list-of-{name,value} / list-of-pairs."""
    if isinstance(raw, dict):
        return {
            str(key): int(value)
            for key, value in raw.items()
            if _is_intable(value)
        }
    if not isinstance(raw, list):
        raise ValueError("knobs must be an object")
    knobs: dict[str, int] = {}
    for item in raw:
        if isinstance(item, Mapping):
            name = (
                item.get("name")
                or item.get("knob")
                or item.get("key")
                or item.get("id")
            )
            value = item.get("value", item.get("val"))
            if name is None or not _is_intable(value):
                continue
            knobs[str(name)] = int(value)
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            name, value = item
            if _is_intable(value):
                knobs[str(name)] = int(value)
    if not knobs:
        raise ValueError(
            "knobs must be an object map of name->int "
            "(got [number,...] without names)"
        )
    return knobs


def target_by_label(label: str) -> ProbeTarget:
    """Resolve a target label or raise ``KeyError``."""
    for target in DEFAULT_TARGETS:
        if target.label == label or target.state == label:
            return target
    raise KeyError(f"unknown probe target: {label}")


__all__ = [
    "DEFAULT_TARGETS",
    "ExperimentProposal",
    "ProbeTarget",
    "TrialDecision",
    "TrialRecord",
    "normalize_knobs",
    "target_by_label",
]
