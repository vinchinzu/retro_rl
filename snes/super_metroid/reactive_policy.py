"""Reactive, equipment-aware room policies built from reference trajectories.

Unlike a raw button tape, a :class:`ReactivePolicyRunner` chooses the next
reference action from live position, pose, and kinematics every frame.  It can
therefore attach in the middle of a room, recover after a human handoff, and
reuse separate trajectories for different equipment loadouts.

The reference is deliberately transparent JSON.  Genetic optimization changes
matching / rejoin parameters and may select or splice whole reference skills;
it does not mutate an opaque frame array one button at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from super_metroid.ram import HI_JUMP_MASK, SuperMetroidState

ACTION_SIZE = 12
POLICY_KIND = "super_metroid_reactive_room_policy"
POLICY_SCHEMA_VERSION = 1


def _as_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def _action_tuple(value: Sequence[int]) -> tuple[int, ...]:
    action = tuple(int(v) for v in value)
    if len(action) != ACTION_SIZE or any(v not in (0, 1) for v in action):
        raise ValueError(f"expected SNES-12 binary action, got {value!r}")
    return action


@dataclass(frozen=True)
class KinematicWeights:
    """Interpretable feature weights for trajectory projection."""

    position: float = 1.0
    velocity: float = 5.0
    momentum: float = 2.0
    pose: float = 18.0
    facing: float = 3.0
    movement_type: float = 8.0
    vertical_direction: float = 3.0
    # Optional preference for a later sample when two states are otherwise
    # equal.  Zero is the replay-safe default; tuning may increase it to skip
    # already-achieved spans without mutating individual input frames.
    advance_bias: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "position": self.position,
            "velocity": self.velocity,
            "momentum": self.momentum,
            "pose": self.pose,
            "facing": self.facing,
            "movement_type": self.movement_type,
            "vertical_direction": self.vertical_direction,
            "advance_bias": self.advance_bias,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> KinematicWeights:
        raw = dict(data or {})
        return cls(
            position=float(raw.get("position", 1.0)),
            velocity=float(raw.get("velocity", 5.0)),
            momentum=float(raw.get("momentum", 2.0)),
            pose=float(raw.get("pose", 18.0)),
            facing=float(raw.get("facing", 3.0)),
            movement_type=float(raw.get("movement_type", 8.0)),
            vertical_direction=float(raw.get("vertical_direction", 3.0)),
            advance_bias=float(raw.get("advance_bias", 0.0)),
        )


@dataclass(frozen=True)
class ReferenceSample:
    """One live-state → timed action span on a room's critical path."""

    x: int
    y: int
    velocity_x: int
    velocity_y: int
    momentum_x: int
    pose: int
    facing: int
    movement_type: int
    vertical_direction: int
    action: tuple[int, ...]
    frames: int = 1

    @classmethod
    def from_state(
        cls,
        state: SuperMetroidState,
        action: Sequence[int],
        *,
        frames: int = 1,
    ) -> ReferenceSample:
        if int(frames) <= 0:
            raise ValueError("reference action span must contain at least one frame")
        return cls(
            x=int(state.samus_x),
            y=int(state.samus_y),
            velocity_x=int(state.velocity_x),
            velocity_y=int(state.velocity_y),
            momentum_x=int(state.momentum_x),
            pose=int(state.pose),
            facing=int(state.facing),
            movement_type=int(state.movement_type),
            vertical_direction=int(state.vertical_direction),
            action=_action_tuple(action),
            frames=int(frames),
        )

    def distance(self, state: SuperMetroidState, weights: KinematicWeights) -> float:
        score = weights.position * (
            abs(int(state.samus_x) - self.x) + abs(int(state.samus_y) - self.y)
        )
        score += weights.velocity * (
            abs(int(state.velocity_x) - self.velocity_x)
            + abs(int(state.velocity_y) - self.velocity_y)
        )
        score += weights.momentum * abs(int(state.momentum_x) - self.momentum_x)
        if int(state.pose) != self.pose:
            score += weights.pose
        if int(state.facing) != self.facing:
            score += weights.facing
        if int(state.movement_type) != self.movement_type:
            score += weights.movement_type
        if int(state.vertical_direction) != self.vertical_direction:
            score += weights.vertical_direction
        return float(score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.velocity_x,
            "vy": self.velocity_y,
            "mx": self.momentum_x,
            "pose": self.pose,
            "facing": self.facing,
            "movement": self.movement_type,
            "vertical": self.vertical_direction,
            "action": list(self.action),
            "hold": self.frames,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReferenceSample:
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            velocity_x=int(data.get("vx", 0)),
            velocity_y=int(data.get("vy", 0)),
            momentum_x=int(data.get("mx", 0)),
            pose=int(data.get("pose", 0)),
            facing=int(data.get("facing", 0)),
            movement_type=int(data.get("movement", 0)),
            vertical_direction=int(data.get("vertical", 0)),
            action=_action_tuple(data["action"]),
            frames=max(1, int(data.get("hold", 1))),
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    trajectory_id: str
    samples: tuple[ReferenceSample, ...]
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError(f"trajectory {self.trajectory_id!r} has no samples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.trajectory_id,
            "source": dict(self.source),
            "sampleCount": len(self.samples),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReferenceTrajectory:
        return cls(
            trajectory_id=str(data["id"]),
            samples=tuple(ReferenceSample.from_dict(row) for row in data["samples"]),
            source=dict(data.get("source") or {}),
        )


@dataclass(frozen=True)
class PolicyVariant:
    """One physics/inventory variant, optionally learned from several takes."""

    variant_id: str
    trajectories: tuple[ReferenceTrajectory, ...]
    required_items: int = 0
    forbidden_items: int = 0
    weights: KinematicWeights = field(default_factory=KinematicWeights)
    lookahead: int = 12
    recovery_rewind: int = 24
    recovery_lookahead: int = 240
    rejoin_threshold: float = 96.0
    adapter_threshold: float = 180.0

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError(f"variant {self.variant_id!r} has no trajectories")
        if self.required_items & self.forbidden_items:
            raise ValueError("required_items and forbidden_items overlap")

    def matches_items(self, items: int) -> bool:
        return (
            int(items) & self.required_items == self.required_items
            and int(items) & self.forbidden_items == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.variant_id,
            "requiredItems": f"0x{self.required_items:04X}",
            "forbiddenItems": f"0x{self.forbidden_items:04X}",
            "weights": self.weights.to_dict(),
            "lookahead": self.lookahead,
            "recoveryRewind": self.recovery_rewind,
            "recoveryLookahead": self.recovery_lookahead,
            "rejoinThreshold": self.rejoin_threshold,
            "adapterThreshold": self.adapter_threshold,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PolicyVariant:
        return cls(
            variant_id=str(data["id"]),
            trajectories=tuple(
                ReferenceTrajectory.from_dict(row) for row in data["trajectories"]
            ),
            required_items=_as_int(data.get("requiredItems", 0)),
            forbidden_items=_as_int(data.get("forbiddenItems", 0)),
            weights=KinematicWeights.from_dict(data.get("weights")),
            lookahead=max(1, int(data.get("lookahead", 12))),
            recovery_rewind=max(0, int(data.get("recoveryRewind", 24))),
            recovery_lookahead=max(1, int(data.get("recoveryLookahead", 240))),
            rejoin_threshold=float(data.get("rejoinThreshold", 96.0)),
            adapter_threshold=float(data.get("adapterThreshold", 180.0)),
        )


@dataclass(frozen=True)
class ReactiveRoomPolicy:
    policy_id: str
    room_id: int
    exit_room_id: int
    variants: tuple[PolicyVariant, ...]
    from_room_id: int | None = None
    route_id: str = "kpdr"
    status: str = "candidate"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(f"policy {self.policy_id!r} has no variants")
        ids = [variant.variant_id for variant in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate variant ids in {self.policy_id}: {ids}")

    def select_variant(self, items: int) -> PolicyVariant | None:
        matches = [variant for variant in self.variants if variant.matches_items(items)]
        if not matches:
            return None
        # Prefer the most specific equipment contract.
        return max(
            matches,
            key=lambda variant: (
                variant.required_items.bit_count() + variant.forbidden_items.bit_count(),
                variant.variant_id,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": POLICY_SCHEMA_VERSION,
            "kind": POLICY_KIND,
            "policyId": self.policy_id,
            "routeId": self.route_id,
            "status": self.status,
            "roomId": self.room_id,
            "roomIdHex": f"0x{self.room_id:04X}",
            "fromRoomId": self.from_room_id,
            "exitRoomId": self.exit_room_id,
            "exitRoomIdHex": f"0x{self.exit_room_id:04X}",
            "variants": [variant.to_dict() for variant in self.variants],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReactiveRoomPolicy:
        if data.get("kind") != POLICY_KIND:
            raise ValueError(f"not a {POLICY_KIND}: {data.get('kind')!r}")
        version = int(data.get("schemaVersion", 0))
        if version != POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported reactive policy schema {version}")
        from_room = data.get("fromRoomId")
        return cls(
            policy_id=str(data["policyId"]),
            room_id=_as_int(data["roomId"]),
            exit_room_id=_as_int(data["exitRoomId"]),
            variants=tuple(PolicyVariant.from_dict(row) for row in data["variants"]),
            from_room_id=None if from_room is None else _as_int(from_room),
            route_id=str(data.get("routeId") or "kpdr"),
            status=str(data.get("status") or "candidate"),
            meta=dict(data.get("meta") or {}),
        )

    def save(self, path: Path | str) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return dest

    @classmethod
    def load(cls, path: Path | str) -> ReactiveRoomPolicy:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RejoinTarget:
    trajectory_index: int
    sample_index: int
    score: float
    sample: ReferenceSample


class ReactivePolicyRunner:
    """Stateful nearest-trajectory controller with bounded forward search."""

    def __init__(self, variant: PolicyVariant) -> None:
        self.variant = variant
        self.trajectory_index = 0
        self.cursor = 0
        self.last_score = float("inf")
        self.rejoins = 0
        self._held_action: np.ndarray | None = None
        self._hold_remaining = 0

    @property
    def trajectory(self) -> ReferenceTrajectory:
        return self.variant.trajectories[self.trajectory_index]

    def _best(
        self,
        state: SuperMetroidState,
        candidates: Sequence[tuple[int, int]],
    ) -> RejoinTarget:
        weights = self.variant.weights
        best: tuple[tuple[float, int, int], RejoinTarget] | None = None
        for trajectory_i, sample_i in candidates:
            trajectory = self.variant.trajectories[trajectory_i]
            sample = trajectory.samples[sample_i]
            distance = sample.distance(state, weights)
            adjusted = distance - weights.advance_bias * sample_i
            target = RejoinTarget(trajectory_i, sample_i, distance, sample)
            key = (adjusted, trajectory_i, sample_i)
            if best is None or key < best[0]:
                best = (key, target)
        if best is None:
            raise RuntimeError("reactive policy has no projection candidates")
        return best[1]

    def project_global(self, state: SuperMetroidState) -> RejoinTarget:
        candidates = [
            (trajectory_i, sample_i)
            for trajectory_i, trajectory in enumerate(self.variant.trajectories)
            for sample_i in range(len(trajectory.samples))
        ]
        return self._best(state, candidates)

    def resume(self, state: SuperMetroidState) -> RejoinTarget:
        """Attach at the closest critical-path state from any point in-room."""
        target = self.project_global(state)
        self.trajectory_index = target.trajectory_index
        self.cursor = target.sample_index
        self.last_score = target.score
        self.rejoins += 1
        self._held_action = None
        self._hold_remaining = 0
        return target

    @property
    def has_held_action(self) -> bool:
        """True when normal playback can skip state parsing/projection."""
        return self._held_action is not None and self._hold_remaining > 0

    def continue_action(self) -> np.ndarray:
        """Return one cached span frame without doing trajectory search."""
        if not self.has_held_action or self._held_action is None:
            raise RuntimeError("no cached reactive-policy action")
        self._hold_remaining -= 1
        return self._held_action

    def action(self, state: SuperMetroidState) -> np.ndarray:
        if self.has_held_action:
            return self.continue_action()
        trajectory = self.trajectory
        start = min(self.cursor, len(trajectory.samples) - 1)
        end = min(len(trajectory.samples), start + self.variant.lookahead + 1)
        candidates = [(self.trajectory_index, i) for i in range(start, end)]
        target = self._best(state, candidates)

        if target.score > self.variant.rejoin_threshold:
            lo = max(0, start - self.variant.recovery_rewind)
            hi = min(len(trajectory.samples), start + self.variant.recovery_lookahead)
            local = [(self.trajectory_index, i) for i in range(lo, hi)]
            target = self._best(state, local)
            if target.score > self.variant.adapter_threshold:
                # A different take may contain a much closer recovery state.
                global_target = self.project_global(state)
                if global_target.score < target.score:
                    target = global_target
                    self.rejoins += 1

        self.trajectory_index = target.trajectory_index
        self.last_score = target.score
        selected = target.sample
        # Always advance at least one reference frame. This preserves idle runs
        # while still permitting lookahead to skip frames already achieved.
        self.cursor = min(
            target.sample_index + 1,
            len(self.trajectory.samples),
        )
        self._held_action = np.asarray(selected.action, dtype=np.int8)
        self._hold_remaining = max(0, int(selected.frames) - 1)
        return self._held_action

    def status(self) -> dict[str, Any]:
        return {
            "variant": self.variant.variant_id,
            "trajectory": self.trajectory.trajectory_id,
            "cursor": self.cursor,
            "samples": len(self.trajectory.samples),
            "projection_score": self.last_score,
            "rejoins": self.rejoins,
            "hold_remaining": self._hold_remaining,
        }


def default_variant_contract(variant_id: str) -> tuple[int, int]:
    """Conventional item contract used by compile CLI for base/Hi-Jump."""
    key = variant_id.strip().lower().replace("-", "_")
    if key in {"hijump", "hi_jump", "hj"}:
        return HI_JUMP_MASK, 0
    if key in {"base", "no_hijump", "no_hi_jump"}:
        return 0, HI_JUMP_MASK
    return 0, 0


__all__ = [
    "ACTION_SIZE",
    "KinematicWeights",
    "PolicyVariant",
    "ReactivePolicyRunner",
    "ReactiveRoomPolicy",
    "ReferenceSample",
    "ReferenceTrajectory",
    "RejoinTarget",
    "default_variant_contract",
]
