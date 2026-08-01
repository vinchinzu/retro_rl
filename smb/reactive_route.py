"""State-gated route primitives for continuous SMB controllers.

The M8 seed remains a useful baseline, but absolute-frame stitching does not
survive an earlier segment becoming faster.  These primitives make a route
controller report and validate its progress against the route declaration
instead: each leg has an entry gate, a named policy, and an explicit legal
successor.  They intentionally own no emulator state, so they can be used by
both a Clean single-environment runner and a development evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from smb.ram import PLAYER_STATE_DYING, SmbSnapshot
from smb.routes import ExitRoute, ExitSegment

SnapshotPredicate = Callable[[SmbSnapshot], bool]


@dataclass(frozen=True)
class StateGate:
    """Named RAM predicate used to start a control-relative policy."""

    gate_id: str
    predicate: SnapshotPredicate
    description: str = ""

    def matches(self, snap: SmbSnapshot) -> bool:
        return bool(self.predicate(snap))


@dataclass
class GateWaiter:
    """Track a gate wait with timeout and compact telemetry."""

    gate: StateGate
    max_frames: int
    frames_waited: int = 0
    matched: bool = False
    timed_out: bool = False
    match_snapshot: dict[str, int] | None = None

    def reset(self) -> None:
        self.frames_waited = 0
        self.matched = False
        self.timed_out = False
        self.match_snapshot = None

    def observe(self, snap: SmbSnapshot) -> bool:
        """Observe one pre-action state and return whether the gate opened."""
        if self.matched:
            return True
        if self.gate.matches(snap):
            self.matched = True
            self.match_snapshot = snapshot_fingerprint(snap)
            return True
        self.frames_waited += 1
        self.timed_out = self.frames_waited >= self.max_frames
        return False

    def report(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate.gate_id,
            "description": self.gate.description,
            "matched": self.matched,
            "timed_out": self.timed_out,
            "frames_waited": self.frames_waited,
            "match_snapshot": self.match_snapshot,
        }


def snapshot_fingerprint(snap: SmbSnapshot) -> dict[str, int]:
    """Small, stable state signature for entry/exit logs and re-solves."""
    return {
        "world": snap.world,
        "level": snap.level,
        "area_pointer": snap.area_pointer,
        "oper_mode": snap.oper_mode,
        "player_state": snap.player_state,
        "player_x": snap.player_x,
        "player_y": snap.player_y,
        "x_speed": snap.x_speed,
        "y_speed": snap.y_speed,
        "timer": snap.timer,
        "lives": snap.lives,
    }


def level_control_gate(exit_seg: ExitSegment) -> StateGate:
    """Return the default controllable-entry gate for a declared stage."""
    if not exit_seg.world or not exit_seg.level:
        raise ValueError(f"{exit_seg.exit_id} has no stage identity")

    def _matches(snap: SmbSnapshot) -> bool:
        return (
            snap.world == exit_seg.world - 1
            and snap.level == exit_seg.level - 1
            and snap.oper_mode == 1
            and snap.player_state in (7, 8)
            and not snap.dying
        )

    return StateGate(
        gate_id=f"{exit_seg.exit_id}:control",
        predicate=_matches,
        description=f"controllable natural entry for {exit_seg.exit_id}",
    )


@dataclass
class RouteProgressTracker:
    """Validate ordered route exits and record actual split timing.

    ``start_index`` permits development of a suffix without pretending it is
    a full-run benchmark.  A complete route is only reported when every
    declared exit has passed its declared successor contract.
    """

    route: ExitRoute
    start_lives: int
    start_index: int = 0
    completed: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None
    entry_fingerprints: dict[str, dict[str, int]] = field(default_factory=dict)
    entry_frames: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.start_index < len(self.route.exits):
            raise ValueError(
                f"start_index {self.start_index} outside route of {len(self.route.exits)} exits"
            )

    @property
    def next_index(self) -> int:
        return self.start_index + len(self.completed)

    @property
    def next_exit(self) -> ExitSegment | None:
        if self.next_index >= len(self.route.exits):
            return None
        return self.route.exits[self.next_index]

    @property
    def complete(self) -> bool:
        return self.next_index == len(self.route.exits) and self.failure is None

    def observe_entry(self, snap: SmbSnapshot, *, frame: int | None = None) -> None:
        """Capture the first natural control state of the current leg."""
        exit_seg = self.next_exit
        if exit_seg is None or exit_seg.exit_id in self.entry_fingerprints:
            return
        if level_control_gate(exit_seg).matches(snap):
            self.entry_fingerprints[exit_seg.exit_id] = snapshot_fingerprint(snap)
            if frame is not None:
                self.entry_frames[exit_seg.exit_id] = frame

    def observe(self, snap: SmbSnapshot, *, frame: int) -> bool:
        """Observe a post-action state; return True exactly when an exit lands."""
        if self.failure is not None or self.complete:
            return False
        if snap.lives < self.start_lives or snap.player_state == PLAYER_STATE_DYING:
            self.failure = "death"
            return False
        self.observe_entry(snap, frame=frame)
        exit_seg = self.next_exit
        if exit_seg is None or not exit_seg.accepts_successor(snap):
            return False
        self.completed.append(
            {
                "exit_id": exit_seg.exit_id,
                "frame": frame,
                "world": snap.world,
                "level": snap.level,
                "lives": snap.lives,
                "successor": next(
                    destination.label or f"{destination.world}-{destination.level}"
                    for destination in exit_seg.successors
                    if destination.matches(snap)
                ),
                "successor_fingerprint": snapshot_fingerprint(snap),
            }
        )
        return True

    def report(self) -> dict[str, Any]:
        expected = [exit_seg.exit_id for exit_seg in self.route.exits[self.start_index :]]
        actual = [row["exit_id"] for row in self.completed]
        return {
            "route_id": self.route.route_id,
            "start_exit": self.route.exits[self.start_index].exit_id,
            "expected_exits": expected,
            "completed_exits": actual,
            "complete": self.complete,
            "failure": self.failure,
            "entry_fingerprints": self.entry_fingerprints,
            "entry_frames": self.entry_frames,
            "splits": list(self.completed),
        }


def missing_policies(
    route: ExitRoute,
    available_policy_ids: Iterable[str],
) -> list[dict[str, str]]:
    """Return explicit controller gaps; never treat missing legs as skipped."""
    available = set(available_policy_ids)
    return [
        {"exit_id": exit_seg.exit_id, "policy_id": exit_seg.policy_id}
        for exit_seg in route.exits
        if exit_seg.policy_id not in available
    ]
