"""In-room takeoff windows — the same kinematics contract as doors.

Jump when x / x_sub / |momentum| / facing match, not after N runup frames.
Room geometry stays in the room package; this module is the shared matcher
and platform descriptor so every hop uses one type.

``TakeoffWindow.ready`` is ``DoorKinematicsRequirement.matches``. Do not
invent a second band checker in a probe or room controller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping

from retro_harness.controls import (
    SNES_DPAD_LEFT,
    SNES_DPAD_RIGHT,
    SNES_SHOULDER_L,
    SNES_SHOULDER_R,
)
from super_metroid.door_kinematics import DoorKinematicsRequirement
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState

# Hop/walk side is D-pad. Shoulders are a different pair — never "L"/"R" here.
_SIDES = frozenset({SNES_DPAD_LEFT, SNES_DPAD_RIGHT})
# Classic dir+B shoulder-pump period (same as runway_dash). Rooms do not own this.
DEFAULT_PUMP_PERIOD = 2


def _facing_for_side(side: str) -> frozenset[int]:
    if side == SNES_DPAD_RIGHT:
        return frozenset({FACING_RIGHT})
    if side == SNES_DPAD_LEFT:
        return frozenset({FACING_LEFT})
    raise ValueError(f"takeoff side must be LEFT or RIGHT, got {side!r}")


def _range_pair(raw: object, *keys: str) -> tuple[int, int] | None:
    if isinstance(raw, Mapping):
        for key in keys:
            val = raw.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return int(val[0]), int(val[1])
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    return None


@dataclass(frozen=True)
class TakeoffWindow:
    """When a grounded hop may jump. Room-agnostic.

    ``side`` is the dash/jump direction. Facing is the WRAM nibble
    (``FACING_LEFT`` / ``FACING_RIGHT``), not a pose-family guess.
    """

    x_range: tuple[int, int]
    side: str
    x_sub_range: tuple[int, int] = (0, 65535)
    min_momentum: int = 1
    pump: bool = True
    release_vy: int = 0
    facings: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.side in {SNES_SHOULDER_L, SNES_SHOULDER_R}:
            raise ValueError(
                f"takeoff side is D-pad LEFT/RIGHT, not shoulder L/R; got {self.side!r}"
            )
        if self.side not in _SIDES:
            raise ValueError(f"takeoff side must be LEFT or RIGHT, got {self.side!r}")
        if self.x_range[0] > self.x_range[1]:
            raise ValueError(f"takeoff x_range inverted: {self.x_range}")

    def facing_set(self) -> frozenset[int]:
        if self.facings is not None:
            return self.facings
        return _facing_for_side(self.side)

    def requirement(self) -> DoorKinematicsRequirement:
        """Canonical band check — same type doors and rooms already use."""
        return DoorKinematicsRequirement(
            x_range=self.x_range,
            x_sub_range=self.x_sub_range,
            facings=self.facing_set(),
            min_abs_momentum=self.min_momentum,
        )

    def ready(self, state: SuperMetroidState) -> bool:
        return self.requirement().matches(state)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "xRange": list(self.x_range),
            "side": self.side,
            "xSubRange": list(self.x_sub_range),
            "minMomentum": self.min_momentum,
            "pump": self.pump,
            "releaseVy": self.release_vy,
        }
        if self.facings is not None:
            payload["facings"] = sorted(self.facings)
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TakeoffWindow:
        x_range = _range_pair(raw, "xRange", "x_range")
        if x_range is None and raw.get("x_jump_lo") is not None:
            x_range = (int(raw["x_jump_lo"]), int(raw["x_jump_hi"]))
        if x_range is None:
            raise ValueError(f"TakeoffWindow missing x range: {raw!r}")
        x_sub = _range_pair(raw, "xSubRange", "x_sub_range")
        if x_sub is None and "x_sub_lo" in raw:
            x_sub = (int(raw.get("x_sub_lo", 0)), int(raw.get("x_sub_hi", 65535)))
        facings_raw = raw.get("facings")
        facings = (
            frozenset(int(f) for f in facings_raw)
            if facings_raw is not None
            else None
        )
        min_mom = raw.get("minMomentum", raw.get("min_momentum", 1))
        release = raw.get("releaseVy", raw.get("release_vy", 0))
        return cls(
            x_range=x_range,
            side=str(raw.get("side", "RIGHT")),
            x_sub_range=x_sub if x_sub is not None else (0, 65535),
            min_momentum=int(min_mom),
            pump=bool(raw.get("pump", True)),
            release_vy=int(release),
            facings=facings,
        )


@dataclass(frozen=True)
class PlatformHop:
    """A platform and the takeoff that leaves it.

    ``y`` / ``x_lo`` / ``x_hi`` are the seat. The jump window lives on
    ``takeoff`` so rooms do not grow a second hop type.
    """

    y: int
    x_lo: int
    x_hi: int
    takeoff: TakeoffWindow

    @property
    def side(self) -> str:
        return self.takeoff.side

    def ready(self, state: SuperMetroidState) -> bool:
        return self.takeoff.ready(state)

    def covers_y(self, y: int, slack: int = 16) -> bool:
        return abs(int(y) - self.y) <= slack

    def at_ledge_end(self, x: int, slack: int = 12) -> bool:
        if self.takeoff.side == SNES_DPAD_RIGHT:
            return int(x) >= self.x_hi - slack
        return int(x) <= self.x_lo + slack

    def with_takeoff(self, **kwargs: Any) -> PlatformHop:
        return replace(self, takeoff=replace(self.takeoff, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "y": self.y,
            "xLo": self.x_lo,
            "xHi": self.x_hi,
            "takeoff": self.takeoff.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PlatformHop:
        nested = raw.get("takeoff")
        if isinstance(nested, Mapping):
            takeoff = TakeoffWindow.from_dict(nested)
        else:
            takeoff = TakeoffWindow.from_dict(raw)
        x_lo = raw.get("xLo", raw.get("x_lo"))
        x_hi = raw.get("xHi", raw.get("x_hi"))
        if x_lo is None or x_hi is None:
            raise ValueError(f"PlatformHop missing x_lo/x_hi: {raw!r}")
        return cls(
            y=int(raw["y"]),
            x_lo=int(x_lo),
            x_hi=int(x_hi),
            takeoff=takeoff,
        )


def hop_for_y(
    y: int,
    hops: Iterable[PlatformHop],
    *,
    slack: int = 16,
) -> PlatformHop | None:
    """Nearest hop whose seat covers ``y``."""
    for hop in hops:
        if hop.covers_y(y, slack):
            return hop
    return None


def next_hop_above(
    y: int,
    hops: Iterable[PlatformHop],
    *,
    slack: int = 16,
) -> PlatformHop | None:
    """Highest seat still below ``y`` (smaller y is higher on screen)."""
    above = [hop for hop in hops if hop.y < int(y) - slack]
    return max(above, key=lambda hop: hop.y) if above else None


def walk_toward_x(x: int, target: int, *, slack: int = 6) -> tuple[str, ...]:
    """One-frame D-pad walk onto ``target``. Empty when already inside slack."""
    if int(x) > int(target) + slack:
        return (SNES_DPAD_LEFT,)
    if int(x) < int(target) - slack:
        return (SNES_DPAD_RIGHT,)
    return ()


def spin_jump(side: str) -> tuple[str, ...]:
    """dir+B+A — a gun-jump (dir+A, no B) never latches."""
    if side in {SNES_SHOULDER_L, SNES_SHOULDER_R} or side not in _SIDES:
        raise ValueError(f"spin side must be D-pad LEFT/RIGHT, got {side!r}")
    return (side, "B", "A")


def shoulder_pump_button(
    i: int, period: int = DEFAULT_PUMP_PERIOD
) -> Literal["L", "R"]:
    """Alternate shoulder L/R for arm-pump. Not D-pad LEFT/RIGHT."""
    period = max(1, int(period))
    if (int(i) // period) % 2 == 0:
        return SNES_SHOULDER_L
    return SNES_SHOULDER_R


def approach_window(
    state: SuperMetroidState,
    hop: PlatformHop,
    *,
    pump_i: int,
    period: int = DEFAULT_PUMP_PERIOD,
) -> tuple[tuple[str, ...], int]:
    """Dash into ``hop``; shoulder L/R pump only after momentum is up.

    Returns ``(buttons, next_pump_i)``. Jump is the caller's job when
    ``hop.ready(state)``. ``hop.side`` is D-pad LEFT/RIGHT.
    """
    period = max(1, int(period))
    side = hop.side
    if int(getattr(state, "facing", 0)) not in hop.takeoff.facing_set():
        return (side, "B"), pump_i + 1
    running = (
        int(getattr(state, "speed_flag", 0)) != 0
        or abs(int(getattr(state, "momentum_x", 0))) >= 1
    )
    if hop.takeoff.pump and running:
        return (side, "B", shoulder_pump_button(pump_i, period)), pump_i + 1
    return (side, "B"), pump_i + 1


def should_release_over(
    state: SuperMetroidState,
    nxt: PlatformHop | None,
    *,
    release_vy: int = 0,
    slack: int = 20,
    x_pad: int = 8,
) -> bool:
    """True when the next seat is underfoot on descent — drop A to land."""
    if nxt is None:
        return False
    y = int(state.samus_y)
    x = int(state.samus_x)
    vy = int(state.velocity_y)
    return (
        abs(y - nxt.y) <= slack
        and nxt.x_lo - x_pad <= x <= nxt.x_hi + x_pad
        and vy >= release_vy
    )


__all__ = [
    "DEFAULT_PUMP_PERIOD",
    "PlatformHop",
    "TakeoffWindow",
    "approach_window",
    "hop_for_y",
    "next_hop_above",
    "should_release_over",
    "shoulder_pump_button",
    "spin_jump",
    "walk_toward_x",
]
