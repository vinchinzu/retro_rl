"""Canonical time-series experience and retained counterexample storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from retro_harness.contracts import canonical_json, contract_digest
from retro_harness.solver import (
    SkillOutcomeStatus,
    SolverActionEvent,
    SolverSessionResult,
)

TRAJECTORY_SCHEMA_VERSION = "1"
TRAJECTORY_SCHEMA_DIGEST = contract_digest(
    "trajectory-schema-v1",
    {
        "version": TRAJECTORY_SCHEMA_VERSION,
        "required_identity": [
            "observation_schema_digest",
            "action_schema_digest",
            "reward_schema_digest",
            "contract_bundle_digest",
            "policy_identity_digest",
        ],
        "step_fields": [
            "sequence",
            "edge_id",
            "skill_id",
            "frame_start",
            "frame_end",
            "applied_frames",
            "observation_before",
            "observation_after",
            "action",
            "reward_components",
            "milestones",
            "state_digest_before",
            "state_digest_after",
        ],
    },
)


class TrajectoryError(ValueError):
    """Raised when experience data is malformed or fails identity checks."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryError(f"{name} must be a non-empty string")
    return value.strip()


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    record = dict(value)
    try:
        canonical_json(record)
    except (TypeError, ValueError) as exc:
        raise TrajectoryError(f"{name} is not canonical JSON: {exc}") from exc
    return record


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One replayable action and its exact observation boundary."""

    sequence: int
    edge_id: str
    skill_id: str
    frame_start: int
    frame_end: int
    applied_frames: int
    observation_before: Mapping[str, Any]
    observation_after: Mapping[str, Any]
    action: Mapping[str, Any]
    reward_components: Mapping[str, float] = field(default_factory=dict)
    milestones: tuple[str, ...] = ()
    state_digest_before: str | None = None
    state_digest_after: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise TrajectoryError("step sequence must be a non-negative integer")
        object.__setattr__(self, "edge_id", _nonempty(self.edge_id, "edge_id"))
        object.__setattr__(self, "skill_id", _nonempty(self.skill_id, "skill_id"))
        if self.frame_start < 0 or self.frame_end < self.frame_start:
            raise TrajectoryError("step frame range is invalid")
        if self.applied_frames < 1:
            raise TrajectoryError("step applied_frames must be positive")
        object.__setattr__(
            self,
            "observation_before",
            _mapping(self.observation_before, "observation_before"),
        )
        object.__setattr__(
            self,
            "observation_after",
            _mapping(self.observation_after, "observation_after"),
        )
        object.__setattr__(self, "action", _mapping(self.action, "action"))
        rewards: dict[str, float] = {}
        for key, value in self.reward_components.items():
            name = _nonempty(key, "reward component")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrajectoryError("reward components must be numeric")
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                raise TrajectoryError("reward components must be finite")
            rewards[name] = number
        object.__setattr__(self, "reward_components", dict(sorted(rewards.items())))
        object.__setattr__(
            self,
            "milestones",
            tuple(_nonempty(item, "milestone") for item in self.milestones),
        )
        for field_name in ("state_digest_before", "state_digest_after"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _nonempty(value, field_name))

    def to_record(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "edge_id": self.edge_id,
            "skill_id": self.skill_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "applied_frames": self.applied_frames,
            "observation_before": dict(self.observation_before),
            "observation_after": dict(self.observation_after),
            "action": dict(self.action),
            "reward_components": dict(self.reward_components),
            "milestones": list(self.milestones),
            "state_digest_before": self.state_digest_before,
            "state_digest_after": self.state_digest_after,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrajectoryStep":
        return cls(
            sequence=record["sequence"],
            edge_id=record["edge_id"],
            skill_id=record["skill_id"],
            frame_start=record["frame_start"],
            frame_end=record["frame_end"],
            applied_frames=record["applied_frames"],
            observation_before=record["observation_before"],
            observation_after=record["observation_after"],
            action=record["action"],
            reward_components=record.get("reward_components", {}),
            milestones=tuple(record.get("milestones", ())),
            state_digest_before=record.get("state_digest_before"),
            state_digest_after=record.get("state_digest_after"),
        )


RewardFunction = Callable[[SolverActionEvent], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Versioned trajectory aligned with environment and policy contracts."""

    observation_schema_digest: str
    action_schema_digest: str
    reward_schema_digest: str
    contract_bundle_digest: str
    policy_identity_digest: str
    steps: tuple[TrajectoryStep, ...]
    succeeded: bool
    terminal_reason: str
    milestones: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    initial_observation_digest: str | None = None
    final_observation_digest: str | None = None
    version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "observation_schema_digest",
            "action_schema_digest",
            "reward_schema_digest",
            "contract_bundle_digest",
            "policy_identity_digest",
            "terminal_reason",
            "version",
        ):
            object.__setattr__(self, field_name, _nonempty(getattr(self, field_name), field_name))
        if self.version != TRAJECTORY_SCHEMA_VERSION:
            raise TrajectoryError(f"unsupported trajectory version: {self.version}")
        steps = tuple(self.steps)
        if not all(isinstance(step, TrajectoryStep) for step in steps):
            raise TrajectoryError("steps must contain TrajectoryStep values")
        if [step.sequence for step in steps] != list(range(len(steps))):
            raise TrajectoryError("step sequences must be contiguous from zero")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "milestones",
            tuple(_mapping(item, "milestone") for item in self.milestones),
        )
        object.__setattr__(self, "provenance", _mapping(self.provenance, "provenance"))
        for field_name in ("initial_observation_digest", "final_observation_digest"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _nonempty(value, field_name))

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema_digest": TRAJECTORY_SCHEMA_DIGEST,
            "observation_schema_digest": self.observation_schema_digest,
            "action_schema_digest": self.action_schema_digest,
            "reward_schema_digest": self.reward_schema_digest,
            "contract_bundle_digest": self.contract_bundle_digest,
            "policy_identity_digest": self.policy_identity_digest,
            "steps": [step.to_record() for step in self.steps],
            "succeeded": self.succeeded,
            "terminal_reason": self.terminal_reason,
            "milestones": [dict(item) for item in self.milestones],
            "provenance": dict(self.provenance),
            "initial_observation_digest": self.initial_observation_digest,
            "final_observation_digest": self.final_observation_digest,
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("trajectory-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "Trajectory":
        if record.get("schema_digest") != TRAJECTORY_SCHEMA_DIGEST:
            raise TrajectoryError("trajectory schema digest mismatch")
        value = cls(
            observation_schema_digest=record["observation_schema_digest"],
            action_schema_digest=record["action_schema_digest"],
            reward_schema_digest=record["reward_schema_digest"],
            contract_bundle_digest=record["contract_bundle_digest"],
            policy_identity_digest=record["policy_identity_digest"],
            steps=tuple(TrajectoryStep.from_record(item) for item in record["steps"]),
            succeeded=record["succeeded"],
            terminal_reason=record["terminal_reason"],
            milestones=tuple(record.get("milestones", ())),
            provenance=record.get("provenance", {}),
            initial_observation_digest=record.get("initial_observation_digest"),
            final_observation_digest=record.get("final_observation_digest"),
            version=record.get("version", TRAJECTORY_SCHEMA_VERSION),
        )
        if record.get("identity_digest") != value.identity_digest:
            raise TrajectoryError("trajectory identity digest mismatch")
        return value

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))


def _step_from_action(
    event: SolverActionEvent,
    sequence: int,
    reward_fn: RewardFunction | None,
    state_digests: Mapping[str, str],
) -> TrajectoryStep:
    before = event.observation_before
    after = event.observation_after
    return TrajectoryStep(
        sequence=sequence,
        edge_id=event.edge_id,
        skill_id=event.skill_id,
        frame_start=event.frame_start,
        frame_end=event.frame_end,
        applied_frames=event.applied_frames,
        observation_before=before.to_record(),
        observation_after=after.to_record(),
        action=event.action,
        reward_components=reward_fn(event) if reward_fn else {},
        milestones=(event.edge_id,) if before.node_id != after.node_id else (),
        state_digest_before=state_digests.get(before.identity_digest),
        state_digest_after=state_digests.get(after.identity_digest),
    )


def trajectory_from_solver_result(
    result: SolverSessionResult,
    *,
    action_schema_digest: str,
    reward_schema_digest: str,
    contract_bundle_digest: str,
    policy_identity_digest: str,
    provenance: Mapping[str, Any],
    reward_fn: RewardFunction | None = None,
    state_digests: Mapping[str, str] | None = None,
) -> Trajectory:
    """Export one complete solver session, including recovery milestones."""
    digests = dict(state_digests or {})
    steps = tuple(
        _step_from_action(event, index, reward_fn, digests)
        for index, event in enumerate(result.actions)
    )
    reason = result.status.value
    if not result.status.value == "completed" and result.outcomes:
        reason = result.outcomes[-1].reason or result.outcomes[-1].status.value
    initial = (
        steps[0].observation_before.get("identity_digest")
        if steps
        else (result.outcomes[0].start_observation_digest if result.outcomes else result.final_observation.identity_digest)
    )
    return Trajectory(
        observation_schema_digest=result.final_observation.schema_digest,
        action_schema_digest=action_schema_digest,
        reward_schema_digest=reward_schema_digest,
        contract_bundle_digest=contract_bundle_digest,
        policy_identity_digest=policy_identity_digest,
        steps=steps,
        succeeded=result.status.value == "completed",
        terminal_reason=reason,
        milestones=tuple(outcome.to_record() for outcome in result.outcomes),
        provenance={**dict(provenance), "source": "SolverSession"},
        initial_observation_digest=initial,
        final_observation_digest=result.final_observation.identity_digest,
    )


def counterexamples_from_solver_result(
    result: SolverSessionResult,
    *,
    action_schema_digest: str,
    reward_schema_digest: str,
    contract_bundle_digest: str,
    policy_identity_digest: str,
    provenance: Mapping[str, Any],
    reward_fn: RewardFunction | None = None,
) -> tuple[Trajectory, ...]:
    """Split every failed solver outcome into a retained training episode."""
    failures: list[Trajectory] = []
    for outcome_index, outcome in enumerate(result.outcomes):
        if outcome.status is SkillOutcomeStatus.SUCCESS:
            continue
        matching = tuple(
            event
            for event in result.actions
            if event.edge_id == outcome.edge_id and event.skill_id == outcome.skill_id
        )
        steps = tuple(
            _step_from_action(event, index, reward_fn, {})
            for index, event in enumerate(matching)
        )
        failures.append(
            Trajectory(
                observation_schema_digest=result.final_observation.schema_digest,
                action_schema_digest=action_schema_digest,
                reward_schema_digest=reward_schema_digest,
                contract_bundle_digest=contract_bundle_digest,
                policy_identity_digest=policy_identity_digest,
                steps=steps,
                succeeded=False,
                terminal_reason=outcome.reason or outcome.status.value,
                milestones=(outcome.to_record(),),
                provenance={
                    **dict(provenance),
                    "source": "SolverSession.failed-outcome",
                    "outcome_index": outcome_index,
                    "recovery_hint": outcome.recovery_hint,
                },
                initial_observation_digest=outcome.start_observation_digest,
                final_observation_digest=outcome.end_observation_digest,
            )
        )
    return tuple(failures)


class CounterexampleLibrary:
    """Content-addressed failed trajectories grouped by stable signatures."""

    INDEX_NAME = "index.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _index(self) -> dict[str, Any]:
        path = self.root / self.INDEX_NAME
        if not path.exists():
            return {"schema_version": 1, "counterexamples": []}
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise TrajectoryError("unsupported counterexample index version")
        return value

    @staticmethod
    def cluster_key(trajectory: Trajectory) -> str:
        skills = sorted({step.skill_id for step in trajectory.steps})
        payload = {
            "terminal_reason": trajectory.terminal_reason,
            "skills": skills,
            "milestone_statuses": [
                item.get("status") for item in trajectory.milestones
            ],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]

    def add(self, trajectory: Trajectory, *, cluster: str | None = None) -> Path:
        if trajectory.succeeded:
            raise TrajectoryError("counterexample library accepts failed trajectories only")
        cluster_id = cluster or self.cluster_key(trajectory)
        _nonempty(cluster_id, "cluster")
        relative = Path("trajectories") / f"{trajectory.identity_digest}.json"
        output = self.root / relative
        trajectory.write(output)
        index = self._index()
        entry = {
            "trajectory_digest": trajectory.identity_digest,
            "path": relative.as_posix(),
            "cluster": cluster_id,
            "terminal_reason": trajectory.terminal_reason,
            "step_count": len(trajectory.steps),
            "provenance": dict(trajectory.provenance),
        }
        rows = [
            row
            for row in index["counterexamples"]
            if row["trajectory_digest"] != trajectory.identity_digest
        ]
        rows.append(entry)
        index["counterexamples"] = sorted(
            rows, key=lambda row: (row["cluster"], row["trajectory_digest"])
        )
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.INDEX_NAME).write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output

    def trajectories(self, *, cluster: str | None = None) -> tuple[Trajectory, ...]:
        rows = self._index()["counterexamples"]
        if cluster is not None:
            rows = [row for row in rows if row["cluster"] == cluster]
        values = tuple(Trajectory.load(self.root / row["path"]) for row in rows)
        for row, value in zip(rows, values, strict=True):
            if value.identity_digest != row["trajectory_digest"]:
                raise TrajectoryError("counterexample index digest mismatch")
        return values

    def offline_actions(self, *, cluster: str) -> tuple[Mapping[str, Any], ...]:
        """Import one failure cluster as an offline replay/BC action stream."""
        return tuple(
            step.action
            for trajectory in self.trajectories(cluster=cluster)
            for step in trajectory.steps
        )


__all__ = [
    "CounterexampleLibrary",
    "TRAJECTORY_SCHEMA_DIGEST",
    "TRAJECTORY_SCHEMA_VERSION",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryStep",
    "counterexamples_from_solver_result",
    "trajectory_from_solver_result",
]
