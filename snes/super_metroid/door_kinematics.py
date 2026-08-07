"""Door entry/leave kinematics for TAS and speedrun door tech.

Super Metroid preserves (or mutates) Samus speed, momentum, pose, and
subpixel position across door transitions. Many route-critical tricks
depend on *how* a door is entered — not merely which door:

* run vs walk vs speed-boost echo level into a door (carry / blue suit)
* horizontal position band on vertical doors / Y band on horizontal doors
* shine-spark charge timer surviving a transition
* mockball / momentum flag state at the lip
* subpixel alignment for tight door clips and early triggers

This module is pure data + predicates over :class:`SuperMetroidState`.
Continuous sessions attach leave/entry snapshots on
:class:`~super_metroid.progression.types.ObservedTransition`; policies and
segment contracts can require kinematics bands without door-warping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from super_metroid.ram import SuperMetroidState


class SpeedBand(str, Enum):
    """Coarse horizontal speed class at a door boundary.

    Thresholds match vanilla speed-booster charge semantics (``speed_counter``
    hi-byte of ``$0B3E``): echoes / blue-suit charge starts at 4.
    """

    STATIONARY = "stationary"
    WALK = "walk"
    RUN = "run"
    SPEED_BOOST = "speed_boost"
    SHINESPARK = "shinespark"


# Pixel-speed thresholds for non-boost classification (absolute |velocity_x|).
_WALK_MAX_PX = 2
_RUN_MAX_PX = 5


@dataclass(frozen=True)
class DoorKinematics:
    """Point-in-time Samus kinematics at leave or entry of a door hop."""

    frame: int
    room_id: int
    samus_x: int
    samus_y: int
    samus_x_sub: int
    samus_y_sub: int
    velocity_x: int
    velocity_y: int
    velocity_x_sub: int
    velocity_y_sub: int
    momentum_x: int
    momentum_x_sub: int
    speed_counter: int
    speed_flag: int
    vertical_direction: int
    facing: int
    movement_type: int
    shinespark_timer: int
    pose: int
    door_transition: int
    transition_direction: int
    door_def_ptr: int
    game_state: int
    phase: str

    @classmethod
    def from_state(cls, state: SuperMetroidState) -> DoorKinematics:
        return cls(
            frame=state.frame,
            room_id=state.room_id,
            samus_x=state.samus_x,
            samus_y=state.samus_y,
            samus_x_sub=state.samus_x_sub,
            samus_y_sub=state.samus_y_sub,
            velocity_x=state.velocity_x,
            velocity_y=state.velocity_y,
            velocity_x_sub=state.velocity_x_sub,
            velocity_y_sub=state.velocity_y_sub,
            momentum_x=state.momentum_x,
            momentum_x_sub=state.momentum_x_sub,
            speed_counter=state.speed_counter,
            speed_flag=state.speed_flag,
            vertical_direction=state.vertical_direction,
            facing=state.facing,
            movement_type=state.movement_type,
            shinespark_timer=state.shinespark_timer,
            pose=state.pose,
            door_transition=state.door_transition,
            transition_direction=state.transition_direction,
            door_def_ptr=state.door_def_ptr,
            game_state=state.game_state,
            phase=state.phase.value,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DoorKinematics:
        return cls(
            frame=int(data.get("frame", 0)),
            room_id=int(data.get("room_id", 0)),
            samus_x=int(data.get("samus_x", 0)),
            samus_y=int(data.get("samus_y", 0)),
            samus_x_sub=int(data.get("samus_x_sub", 0)),
            samus_y_sub=int(data.get("samus_y_sub", 0)),
            velocity_x=int(data.get("velocity_x", 0)),
            velocity_y=int(data.get("velocity_y", 0)),
            velocity_x_sub=int(data.get("velocity_x_sub", 0)),
            velocity_y_sub=int(data.get("velocity_y_sub", 0)),
            momentum_x=int(data.get("momentum_x", 0)),
            momentum_x_sub=int(data.get("momentum_x_sub", 0)),
            speed_counter=int(data.get("speed_counter", 0)),
            speed_flag=int(data.get("speed_flag", 0)),
            vertical_direction=int(data.get("vertical_direction", 0)),
            facing=int(data.get("facing", 0)),
            movement_type=int(data.get("movement_type", 0)),
            shinespark_timer=int(data.get("shinespark_timer", 0)),
            pose=int(data.get("pose", 0)),
            door_transition=int(data.get("door_transition", 0)),
            transition_direction=int(data.get("transition_direction", 0)),
            door_def_ptr=int(data.get("door_def_ptr", 0)),
            game_state=int(data.get("game_state", 0)),
            phase=str(data.get("phase", "")),
        )

    @property
    def speed_boosting(self) -> bool:
        return self.speed_counter >= 4

    @property
    def shinesparking(self) -> bool:
        return self.shinespark_timer > 0

    @property
    def speed_band(self) -> SpeedBand:
        return classify_speed_band(
            velocity_x=self.velocity_x,
            speed_counter=self.speed_counter,
            shinespark_timer=self.shinespark_timer,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "samus_x": self.samus_x,
            "samus_x_sub": self.samus_x_sub,
            "samus_y": self.samus_y,
            "samus_y_sub": self.samus_y_sub,
            "velocity_x": self.velocity_x,
            "velocity_x_sub": self.velocity_x_sub,
            "velocity_y": self.velocity_y,
            "velocity_y_sub": self.velocity_y_sub,
            "momentum_x": self.momentum_x,
            "momentum_x_sub": self.momentum_x_sub,
            "speed_counter": self.speed_counter,
            "speed_flag": self.speed_flag,
            "speed_boosting": self.speed_boosting,
            "speed_band": self.speed_band.value,
            "vertical_direction": self.vertical_direction,
            "facing": self.facing,
            "movement_type": self.movement_type,
            "shinespark_timer": self.shinespark_timer,
            "shinesparking": self.shinesparking,
            "pose": self.pose,
            "door_transition": self.door_transition,
            "transition_direction": self.transition_direction,
            "door_def_ptr": self.door_def_ptr,
            "door_def_ptr_hex": f"0x{self.door_def_ptr:04X}",
            "game_state": self.game_state,
            "phase": self.phase,
        }


def classify_speed_band(
    *,
    velocity_x: int,
    speed_counter: int,
    shinespark_timer: int = 0,
) -> SpeedBand:
    """Map live kinematics into a coarse speed band for route contracts."""
    if shinespark_timer > 0:
        return SpeedBand.SHINESPARK
    if speed_counter >= 4:
        return SpeedBand.SPEED_BOOST
    abs_vx = abs(int(velocity_x))
    if abs_vx == 0:
        return SpeedBand.STATIONARY
    if abs_vx <= _WALK_MAX_PX:
        return SpeedBand.WALK
    if abs_vx <= _RUN_MAX_PX or speed_counter > 0:
        return SpeedBand.RUN
    return SpeedBand.RUN


@dataclass(frozen=True)
class DoorKinematicsRequirement:
    """Optional bands for leave (into door) or entry (spawn) kinematics.

    All fields are optional; unset means "do not check". Use at policy
    boundaries, pure-hop asserts, and continuous integrity hooks.
    """

    x_range: tuple[int, int] | None = None
    y_range: tuple[int, int] | None = None
    # Subpixel windows are rare but needed for TAS-tight clips.
    x_sub_range: tuple[int, int] | None = None
    y_sub_range: tuple[int, int] | None = None
    velocity_x_range: tuple[int, int] | None = None
    velocity_y_range: tuple[int, int] | None = None
    momentum_x_range: tuple[int, int] | None = None
    speed_counter_min: int | None = None
    speed_counter_max: int | None = None
    speed_bands: frozenset[SpeedBand] = frozenset()
    require_speed_boost: bool | None = None
    require_shinespark: bool | None = None
    poses: frozenset[int] = frozenset()
    facings: frozenset[int] = frozenset()
    movement_types: frozenset[int] = frozenset()
    door_def_ptrs: frozenset[int] = frozenset()
    transition_directions: frozenset[int] = frozenset()

    def failures(self, kin: DoorKinematics | SuperMetroidState) -> tuple[str, ...]:
        if isinstance(kin, SuperMetroidState):
            snap = DoorKinematics.from_state(kin)
        else:
            snap = kin
        failures: list[str] = []
        if self.x_range is not None and not (
            self.x_range[0] <= snap.samus_x <= self.x_range[1]
        ):
            failures.append(f"x {snap.samus_x} outside {self.x_range}")
        if self.y_range is not None and not (
            self.y_range[0] <= snap.samus_y <= self.y_range[1]
        ):
            failures.append(f"y {snap.samus_y} outside {self.y_range}")
        if self.x_sub_range is not None and not (
            self.x_sub_range[0] <= snap.samus_x_sub <= self.x_sub_range[1]
        ):
            failures.append(f"x_sub {snap.samus_x_sub} outside {self.x_sub_range}")
        if self.y_sub_range is not None and not (
            self.y_sub_range[0] <= snap.samus_y_sub <= self.y_sub_range[1]
        ):
            failures.append(f"y_sub {snap.samus_y_sub} outside {self.y_sub_range}")
        if self.velocity_x_range is not None and not (
            self.velocity_x_range[0] <= snap.velocity_x <= self.velocity_x_range[1]
        ):
            failures.append(
                f"velocity_x {snap.velocity_x} outside {self.velocity_x_range}"
            )
        if self.velocity_y_range is not None and not (
            self.velocity_y_range[0] <= snap.velocity_y <= self.velocity_y_range[1]
        ):
            failures.append(
                f"velocity_y {snap.velocity_y} outside {self.velocity_y_range}"
            )
        if self.momentum_x_range is not None and not (
            self.momentum_x_range[0] <= snap.momentum_x <= self.momentum_x_range[1]
        ):
            failures.append(
                f"momentum_x {snap.momentum_x} outside {self.momentum_x_range}"
            )
        if (
            self.speed_counter_min is not None
            and snap.speed_counter < self.speed_counter_min
        ):
            failures.append(
                f"speed_counter {snap.speed_counter} < min {self.speed_counter_min}"
            )
        if (
            self.speed_counter_max is not None
            and snap.speed_counter > self.speed_counter_max
        ):
            failures.append(
                f"speed_counter {snap.speed_counter} > max {self.speed_counter_max}"
            )
        if self.speed_bands and snap.speed_band not in self.speed_bands:
            failures.append(
                f"speed_band {snap.speed_band.value} not in "
                f"{sorted(b.value for b in self.speed_bands)}"
            )
        if self.require_speed_boost is True and not snap.speed_boosting:
            failures.append("expected speed_boosting")
        if self.require_speed_boost is False and snap.speed_boosting:
            failures.append("unexpected speed_boosting")
        if self.require_shinespark is True and not snap.shinesparking:
            failures.append("expected shinespark charge/timer")
        if self.require_shinespark is False and snap.shinesparking:
            failures.append("unexpected shinespark timer")
        if self.poses and snap.pose not in self.poses:
            failures.append(f"pose {snap.pose} not in {sorted(self.poses)}")
        if self.facings and snap.facing not in self.facings:
            failures.append(f"facing {snap.facing} not in {sorted(self.facings)}")
        if self.movement_types and snap.movement_type not in self.movement_types:
            failures.append(
                f"movement_type {snap.movement_type} not in "
                f"{sorted(self.movement_types)}"
            )
        if self.door_def_ptrs and snap.door_def_ptr not in self.door_def_ptrs:
            failures.append(
                f"door_def_ptr 0x{snap.door_def_ptr:04X} not in "
                f"{[f'0x{p:04X}' for p in sorted(self.door_def_ptrs)]}"
            )
        if (
            self.transition_directions
            and snap.transition_direction not in self.transition_directions
        ):
            failures.append(
                f"transition_direction {snap.transition_direction} not in "
                f"{sorted(self.transition_directions)}"
            )
        return tuple(failures)

    def matches(self, kin: DoorKinematics | SuperMetroidState) -> bool:
        return not self.failures(kin)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.x_range is not None:
            payload["xRange"] = list(self.x_range)
        if self.y_range is not None:
            payload["yRange"] = list(self.y_range)
        if self.x_sub_range is not None:
            payload["xSubRange"] = list(self.x_sub_range)
        if self.y_sub_range is not None:
            payload["ySubRange"] = list(self.y_sub_range)
        if self.velocity_x_range is not None:
            payload["velocityXRange"] = list(self.velocity_x_range)
        if self.velocity_y_range is not None:
            payload["velocityYRange"] = list(self.velocity_y_range)
        if self.momentum_x_range is not None:
            payload["momentumXRange"] = list(self.momentum_x_range)
        if self.speed_counter_min is not None:
            payload["speedCounterMin"] = self.speed_counter_min
        if self.speed_counter_max is not None:
            payload["speedCounterMax"] = self.speed_counter_max
        if self.speed_bands:
            payload["speedBands"] = sorted(b.value for b in self.speed_bands)
        if self.require_speed_boost is not None:
            payload["requireSpeedBoost"] = self.require_speed_boost
        if self.require_shinespark is not None:
            payload["requireShinespark"] = self.require_shinespark
        if self.poses:
            payload["poses"] = sorted(self.poses)
        if self.facings:
            payload["facings"] = sorted(self.facings)
        if self.movement_types:
            payload["movementTypes"] = sorted(self.movement_types)
        if self.door_def_ptrs:
            payload["doorDefPtrs"] = [
                f"0x{p:04X}" for p in sorted(self.door_def_ptrs)
            ]
        if self.transition_directions:
            payload["transitionDirections"] = sorted(self.transition_directions)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> DoorKinematicsRequirement | None:
        if not isinstance(raw, dict) or not raw:
            return None

        def _range(key: str) -> tuple[int, int] | None:
            val = raw.get(key)
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                return None
            return int(val[0]), int(val[1])

        bands_raw = raw.get("speedBands") or raw.get("speed_bands") or ()
        bands = frozenset(SpeedBand(str(b)) for b in bands_raw)

        ptrs_raw = raw.get("doorDefPtrs") or raw.get("door_def_ptrs") or ()
        ptrs: set[int] = set()
        for item in ptrs_raw:
            if isinstance(item, str):
                ptrs.add(int(item, 0))
            else:
                ptrs.add(int(item))

        return cls(
            x_range=_range("xRange") or _range("x_range"),
            y_range=_range("yRange") or _range("y_range"),
            x_sub_range=_range("xSubRange") or _range("x_sub_range"),
            y_sub_range=_range("ySubRange") or _range("y_sub_range"),
            velocity_x_range=_range("velocityXRange") or _range("velocity_x_range"),
            velocity_y_range=_range("velocityYRange") or _range("velocity_y_range"),
            momentum_x_range=_range("momentumXRange") or _range("momentum_x_range"),
            speed_counter_min=(
                int(raw["speedCounterMin"])
                if raw.get("speedCounterMin") is not None
                else (
                    int(raw["speed_counter_min"])
                    if raw.get("speed_counter_min") is not None
                    else None
                )
            ),
            speed_counter_max=(
                int(raw["speedCounterMax"])
                if raw.get("speedCounterMax") is not None
                else (
                    int(raw["speed_counter_max"])
                    if raw.get("speed_counter_max") is not None
                    else None
                )
            ),
            speed_bands=bands,
            require_speed_boost=(
                bool(raw["requireSpeedBoost"])
                if raw.get("requireSpeedBoost") is not None
                else (
                    bool(raw["require_speed_boost"])
                    if raw.get("require_speed_boost") is not None
                    else None
                )
            ),
            require_shinespark=(
                bool(raw["requireShinespark"])
                if raw.get("requireShinespark") is not None
                else (
                    bool(raw["require_shinespark"])
                    if raw.get("require_shinespark") is not None
                    else None
                )
            ),
            poses=frozenset(int(p) for p in (raw.get("poses") or ())),
            facings=frozenset(int(f) for f in (raw.get("facings") or ())),
            movement_types=frozenset(
                int(m) for m in (raw.get("movementTypes") or raw.get("movement_types") or ())
            ),
            door_def_ptrs=frozenset(ptrs),
            transition_directions=frozenset(
                int(d)
                for d in (
                    raw.get("transitionDirections")
                    or raw.get("transition_directions")
                    or ()
                )
            ),
        )


def require_door_kinematics(
    state: SuperMetroidState,
    requirement: DoorKinematicsRequirement,
    *,
    label: str,
) -> None:
    """Raise ``RuntimeError`` if live state fails a kinematics contract."""
    failures = requirement.failures(state)
    if failures:
        raise RuntimeError(
            f"{label}: door kinematics mismatch: {'; '.join(failures)}; "
            f"kin={DoorKinematics.from_state(state).to_dict()}"
        )


__all__ = [
    "DoorKinematics",
    "DoorKinematicsRequirement",
    "SpeedBand",
    "classify_speed_band",
    "require_door_kinematics",
]
