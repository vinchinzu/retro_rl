"""Validated raw-input policies and natural-entry segment evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

import numpy as np

from super_metroid.paths import EARLY_POLICY_DIR
from super_metroid.ram import GameplayPhase, SuperMetroidState


@dataclass(frozen=True)
class StateRequirement:
    """Small composable state predicate used at policy boundaries.

    Position and kinematics fields support door-entry contracts (run speed,
    speed-booster charge, pose, facing) without requiring a separate check.
    """

    room_id: int | None = None
    phases: frozenset[GameplayPhase] = field(default_factory=frozenset)
    game_states: frozenset[int] = field(default_factory=frozenset)
    collected_items_mask: int = 0
    minimum_ammo_capacities: tuple[int, int, int] = (0, 0, 0)
    x_range: tuple[int, int] | None = None
    y_range: tuple[int, int] | None = None
    velocity_x_range: tuple[int, int] | None = None
    velocity_y_range: tuple[int, int] | None = None
    momentum_x_range: tuple[int, int] | None = None
    speed_counter_min: int | None = None
    speed_counter_max: int | None = None
    require_speed_boost: bool | None = None
    require_shinespark: bool | None = None
    poses: frozenset[int] = field(default_factory=frozenset)
    facings: frozenset[int] = field(default_factory=frozenset)

    def failures(self, state: SuperMetroidState) -> tuple[str, ...]:
        failures: list[str] = []
        if self.room_id is not None and state.room_id != self.room_id:
            failures.append(
                f"room 0x{state.room_id:04X} != required 0x{self.room_id:04X}"
            )
        if self.phases and state.phase not in self.phases:
            failures.append(f"phase {state.phase.value} not in {sorted(self.phases)}")
        if self.game_states and state.game_state not in self.game_states:
            failures.append(
                f"game state {state.game_state} not in {sorted(self.game_states)}"
            )
        if state.collected_items & self.collected_items_mask != self.collected_items_mask:
            failures.append(
                f"items 0x{state.collected_items:04X} missing "
                f"0x{self.collected_items_mask:04X}"
            )
        actual_ammo = (
            state.max_missiles,
            state.max_super_missiles,
            state.max_power_bombs,
        )
        if any(
            actual < minimum
            for actual, minimum in zip(actual_ammo, self.minimum_ammo_capacities)
        ):
            failures.append(
                f"ammo capacities {actual_ammo} below {self.minimum_ammo_capacities}"
            )
        if self.x_range is not None and not self.x_range[0] <= state.samus_x <= self.x_range[1]:
            failures.append(f"x {state.samus_x} outside {self.x_range}")
        if self.y_range is not None and not self.y_range[0] <= state.samus_y <= self.y_range[1]:
            failures.append(f"y {state.samus_y} outside {self.y_range}")
        if self.velocity_x_range is not None and not (
            self.velocity_x_range[0] <= state.velocity_x <= self.velocity_x_range[1]
        ):
            failures.append(
                f"velocity_x {state.velocity_x} outside {self.velocity_x_range}"
            )
        if self.velocity_y_range is not None and not (
            self.velocity_y_range[0] <= state.velocity_y <= self.velocity_y_range[1]
        ):
            failures.append(
                f"velocity_y {state.velocity_y} outside {self.velocity_y_range}"
            )
        if self.momentum_x_range is not None and not (
            self.momentum_x_range[0] <= state.momentum_x <= self.momentum_x_range[1]
        ):
            failures.append(
                f"momentum_x {state.momentum_x} outside {self.momentum_x_range}"
            )
        if (
            self.speed_counter_min is not None
            and state.speed_counter < self.speed_counter_min
        ):
            failures.append(
                f"speed_counter {state.speed_counter} < min {self.speed_counter_min}"
            )
        if (
            self.speed_counter_max is not None
            and state.speed_counter > self.speed_counter_max
        ):
            failures.append(
                f"speed_counter {state.speed_counter} > max {self.speed_counter_max}"
            )
        if self.require_speed_boost is True and not state.speed_boosting:
            failures.append("expected speed_boosting")
        if self.require_speed_boost is False and state.speed_boosting:
            failures.append("unexpected speed_boosting")
        if self.require_shinespark is True and not state.shinesparking:
            failures.append("expected shinespark timer")
        if self.require_shinespark is False and state.shinesparking:
            failures.append("unexpected shinespark timer")
        if self.poses and state.pose not in self.poses:
            failures.append(f"pose {state.pose} not in {sorted(self.poses)}")
        if self.facings and state.facing not in self.facings:
            failures.append(f"facing {state.facing} not in {sorted(self.facings)}")
        return tuple(failures)

    def matches(self, state: SuperMetroidState) -> bool:
        return not self.failures(state)


@dataclass(frozen=True)
class PolicySegment:
    segment_id: str
    filename: str
    entry: StateRequirement
    exit: StateRequirement
    expected_policy_id: str

    @property
    def path(self) -> Path:
        return EARLY_POLICY_DIR / self.filename


@dataclass(frozen=True)
class SegmentEvidence:
    segment_id: str
    policy_path: str
    policy_sha256: str
    source_sha256: str
    source_slice: str
    start_frame: int
    end_frame: int
    action_frames: int
    max_identical_navigation_frames: int
    start_button_frames: int
    opposite_direction_frames: int
    entry_state: dict[str, object]
    exit_state: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PolicySession(Protocol):
    frame: int
    state: SuperMetroidState

    def step(self, action: np.ndarray, reason: str) -> SuperMetroidState: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_state(state: SuperMetroidState) -> dict[str, object]:
    return {
        "frame": state.frame,
        "room_id": state.room_id,
        "room_id_hex": f"0x{state.room_id:04X}",
        "game_state": state.game_state,
        "phase": state.phase.value,
        "samus_x": state.samus_x,
        "samus_x_sub": state.samus_x_sub,
        "samus_y": state.samus_y,
        "samus_y_sub": state.samus_y_sub,
        "velocity_x": state.velocity_x,
        "velocity_y": state.velocity_y,
        "momentum_x": state.momentum_x,
        "speed_counter": state.speed_counter,
        "speed_boosting": state.speed_boosting,
        "facing": state.facing,
        "movement_type": state.movement_type,
        "shinespark_timer": state.shinespark_timer,
        "pose": state.pose,
        "door_def_ptr": state.door_def_ptr,
        "door_def_ptr_hex": f"0x{state.door_def_ptr:04X}",
        "health": state.health,
        "missiles": state.missiles,
        "max_missiles": state.max_missiles,
        "selected_item": state.selected_item,
        "collected_items": state.collected_items,
        "collected_items_hex": f"0x{state.collected_items:04X}",
        "enemy0_hp": state.enemy0_hp,
    }


def load_policy(segment: PolicySegment) -> tuple[list[np.ndarray], dict[str, object]]:
    payload = json.loads(segment.path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("policy_id") != segment.expected_policy_id:
        raise ValueError(
            f"{segment.segment_id}: policy id {metadata.get('policy_id')!r} "
            f"!= {segment.expected_policy_id!r}"
        )
    raw_actions = payload.get("raw_buttons")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"{segment.segment_id}: raw_buttons is empty")
    if payload.get("num_frames") != len(raw_actions):
        raise ValueError(f"{segment.segment_id}: num_frames does not match actions")

    actions: list[np.ndarray] = []
    for index, raw in enumerate(raw_actions):
        action = np.asarray(raw, dtype=np.int8)
        if action.shape != (12,) or not np.isin(action, (0, 1)).all():
            raise ValueError(f"{segment.segment_id}: invalid action at frame {index}")
        actions.append(action)
    return actions, metadata


def play_policy(
    session: PolicySession,
    segment: PolicySegment,
    *,
    stop_when: Callable[[SuperMetroidState], bool] | None = None,
    require_exit: bool = True,
    action_slice: slice | None = None,
) -> SegmentEvidence:
    """Replay a hash-pinned policy segment.

    Optional ``stop_when`` ends the replay early (no exit check unless the
    predicate never matched and ``require_exit`` remains True). ``action_slice``
    selects a sub-range of the policy (e.g. exit tail after a hybrid fight).
    """
    entry_failures = segment.entry.failures(session.state)
    if entry_failures and action_slice is None:
        raise RuntimeError(
            f"{segment.segment_id} entry mismatch: {'; '.join(entry_failures)}; "
            f"state={compact_state(session.state)}"
        )

    actions, metadata = load_policy(segment)
    if action_slice is not None:
        actions = actions[action_slice]
    start_frame = session.frame
    entry_state = compact_state(session.state)
    max_identical = 0
    identical = 0
    previous_navigation = None
    start_button_frames = 0
    opposite_frames = 0
    stopped_early = False

    for action in actions:
        if action[3]:
            start_button_frames += 1
        if (action[4] and action[5]) or (action[6] and action[7]):
            opposite_frames += 1
        state = session.step(action, f"policy_{segment.segment_id}")
        navigation = (
            state.progress_vector(),
            state.samus_x,
            state.samus_y,
            state.velocity_x,
            state.velocity_y,
            state.pose,
        )
        if navigation == previous_navigation:
            identical += 1
            max_identical = max(max_identical, identical)
        else:
            identical = 0
            previous_navigation = navigation
        if stop_when is not None and stop_when(state):
            stopped_early = True
            break

    if require_exit and not stopped_early:
        exit_failures = segment.exit.failures(session.state)
        if exit_failures:
            raise RuntimeError(
                f"{segment.segment_id} exit mismatch: {'; '.join(exit_failures)}; "
                f"entry={entry_state}; state={compact_state(session.state)}"
            )
    return SegmentEvidence(
        segment_id=segment.segment_id,
        policy_path=str(segment.path.resolve()),
        policy_sha256=_sha256(segment.path),
        source_sha256=str(metadata["source_sha256"]),
        source_slice=str(metadata["source_slice"]),
        start_frame=start_frame,
        end_frame=session.frame,
        action_frames=session.frame - start_frame,
        max_identical_navigation_frames=max_identical,
        start_button_frames=start_button_frames,
        opposite_direction_frames=opposite_frames,
        entry_state=entry_state,
        exit_state=compact_state(session.state),
    )
