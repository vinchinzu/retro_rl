"""Natural L2 bomb budget / farm — no inventory poke.

Spine claim cannot write bombs, keys, doors, or progression. Isolated
``Level2Boom`` → Dodongo used ``--poke-bombs`` (recon only).

Live RAM (2026-08-15, ``make_env`` + 1 idle frame):

- ``Level2Entrance`` room 0x7d: **bombs=0** (``ADDR_BOMBS`` / max=8)
- ``Level1Complete`` (mode 18): bombs=4
- ``survival_spine.json`` final snapshot has **no** ``bombs`` field
  (keys=0, room 0x7d) — missing is unknown, not zero
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Literal

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.overworld.common import swing_action, track_stuck, unstick_wiggle, wake_or_wait_mode
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

# Successful placements through TF (LEVEL2_ROUTE):
#   bomb-N 0x6f→0x5f, 0x5f→0x4f, 0x4f→0x3f, 0x1e→0x0e + two Dodongo mouths.
L2_BOMB_WALLS: tuple[tuple[int, int], ...] = (
    (0x6F, 0x5F),
    (0x5F, 0x4F),
    (0x4F, 0x3F),
    (0x1E, 0x0E),
)
L2_DODONGO_MOUTHS: int = 2
L2_BOMB_BUDGET: int = len(L2_BOMB_WALLS) + L2_DODONGO_MOUTHS  # 6
L2_BOMB_BUDGET_BOOM: int = 3  # walls through Magical Boomerang 0x4f
L2_BOMB_CARRY: int = 8  # skip farm; L2 entry max_bombs is typically 8
L2_BOMB_BUDGET_BY_THROUGH: dict[str, int] = {
    "boom": L2_BOMB_BUDGET_BOOM,
    "tf": L2_BOMB_BUDGET,
}

# Documented natural farm after boom (LEVEL2_ROUTE / walkthrough):
# 0x4f blue Goriya, 0x3e Moldorm, 0x1e 5× Red Goriya bomb drop.
L2_BOMB_FARM_ROOMS: tuple[int, ...] = (0x4F, 0x3E, 0x1E)
L2_BOMB_FARM_SCREEN: int = 0x1E

# Live checkpoint reads (not a spine tape).
L2_ENTRY_BOMBS_MEASURED: int = 0
L2_ENTRY_BOMBS_SOURCE: str = (
    "Level2Entrance.state RAM ADDR_BOMBS=0x0658 (room 0x7d, 2026-08-15)"
)
L1_COMPLETE_BOMBS_MEASURED: int = 4
L1_COMPLETE_BOMBS_SOURCE: str = (
    "Level1Complete.state RAM ADDR_BOMBS=0x0658 (mode 18, 2026-08-15)"
)

_FARM_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (64, 141),
    (120, 109),
    (176, 141),
    (120, 173),
    (120, 141),
)
_FARM_ENEMY_TYPES: frozenset[int] = frozenset({0x05, 0x06, 0x41})  # blue/red Goriya, Moldorm
_PICKUP_TYPES: frozenset[int] = frozenset({0x60, 0x61, 0x62, 0x63})  # rupee / bomb-class drops
_IGNORE_TYPES: frozenset[int] = frozenset({0x49, 0x55, 0x4E, 0x5C, 0x4A})
DEFAULT_MAX_FRAMES = 3600
DEFAULT_STUCK_THRESHOLD = 40
FARM_SWING_PERIOD = 8
FARM_SWING_HOLD = 3
WAYPOINT_TOL = 6

Through = Literal["boom", "tf"]
PlanAction = Literal["carry", "farm"]


def bomb_budget(*, through: Through = "tf") -> int:
    """Successful-bomb count needed for ``through`` (no miss slack)."""
    key = str(through)
    if key not in L2_BOMB_BUDGET_BY_THROUGH:
        raise ValueError(f"unknown through={through!r}; expected boom|tf")
    return L2_BOMB_BUDGET_BY_THROUGH[key]


def bombs_from_snapshot(snap: object) -> int | None:
    """Inventory bombs, or None when the field is absent (not zero)."""
    if snap is None:
        return None
    if isinstance(snap, Mapping):
        if "bombs" not in snap:
            return None
        val = snap["bombs"]
    elif hasattr(snap, "bombs"):
        val = getattr(snap, "bombs")
    else:
        return None
    if val is None:
        return None
    return int(val)


def enough_bombs(n: int | None, *, through: Through = "tf") -> bool:
    """True when ``n`` covers the successful-placement budget for ``through``."""
    if n is None:
        return False
    return int(n) >= bomb_budget(through=through)


@dataclass(frozen=True)
class BombPlan:
    """Carry existing bombs or farm after 0x4f / 0x3e. Never pokes."""

    action: PlanAction
    bombs_in: int | None
    budget: int
    through: str
    farm_required: bool
    poke_bombs: bool = False
    notes: tuple[str, ...] = ()

    def report(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "bombs_in": self.bombs_in,
            "budget": self.budget,
            "through": self.through,
            "farm_required": self.farm_required,
            "farm_stage": self.action == "farm",
            "poke_bombs": False,
            "notes": list(self.notes),
        }


def natural_bomb_plan(
    bombs_in: int | None,
    *,
    through: Through = "tf",
    carry_at: int = L2_BOMB_CARRY,
) -> BombPlan:
    """``carry`` if inventory already covers slack; else ``farm`` (no poke)."""
    budget = bomb_budget(through=through)
    notes: list[str] = [
        f"budget_{through}={budget}",
        f"carry_at={carry_at}",
        f"farm_rooms={[f'0x{r:02x}' for r in L2_BOMB_FARM_ROOMS]}",
    ]
    if bombs_in is None:
        notes.append("unknown_bombs_in")
        notes.append("missing_field_is_not_zero")
        return BombPlan(
            action="farm",
            bombs_in=None,
            budget=budget,
            through=through,
            farm_required=True,
            notes=tuple(notes),
        )
    n = int(bombs_in)
    notes.append(f"bombs_in={n}")
    if n < L2_BOMB_BUDGET_BOOM:
        notes.append(
            f"pre_farm_short walls_to_boom={L2_BOMB_BUDGET_BOOM} have={n}"
        )
    if n >= carry_at:
        notes.append("carry_enough")
        return BombPlan(
            action="carry",
            bombs_in=n,
            budget=budget,
            through=through,
            farm_required=False,
            notes=tuple(notes),
        )
    notes.append("farm_after_0x4f_0x3e")
    return BombPlan(
        action="farm",
        bombs_in=n,
        budget=budget,
        through=through,
        farm_required=True,
        notes=tuple(notes),
    )


def poke_bombs_used(report: Mapping[str, Any] | None) -> bool:
    """True if a fight/path/spine report admits a bomb-count poke."""
    if not isinstance(report, Mapping):
        return False
    if report.get("poke") is True:
        return True
    pb = report.get("poke_bombs")
    if pb is True:
        return True
    if isinstance(pb, int) and pb > 0:
        return True
    if report.get("bomb_count_poke") is True:
        return True
    notes = report.get("poke_notes")
    if isinstance(notes, str):
        notes = [notes]
    if isinstance(notes, (list, tuple)):
        for note in notes:
            text = str(note)
            if text.startswith("bombs=") or "RECON poke" in text:
                return True
    for key in ("fight", "path", "dodongo", "boss"):
        inner = report.get(key)
        if isinstance(inner, Mapping) and poke_bombs_used(inner):
            return True
    return False


detect_poke_bombs = poke_bombs_used


def poke_kwarg_default(fn: Any) -> bool | None:
    """``poke`` default on a library callable, or None if the kwarg is absent."""
    param = inspect.signature(fn).parameters.get("poke")
    if param is None or param.default is inspect.Parameter.empty:
        return None
    return bool(param.default)


def spine_bomb_flags(*, poke: bool = False, bombs: int | None = None) -> dict[str, Any]:
    """Spine inventory-count flags. Poke is a documented Survival shortcut."""
    if poke:
        return {
            "poke_bombs": bombs if bombs is not None else True,
            "inventory_assist": True,
            "capacity_writes": 0,
        }
    return {"poke_bombs": False}


def spine_bomb_report(
    bombs_in: int | None,
    *,
    through: Through = "tf",
    bombs_out: int | None = None,
    farm: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Budget + farm telemetry for a continuous Survival tape."""
    plan = natural_bomb_plan(bombs_in, through=through)
    return {
        **plan.report(),
        "bombs_out": bombs_out,
        "farm": dict(farm) if farm is not None else None,
        "poke_bombs": False,
        "measured": {
            "level2_entrance": L2_ENTRY_BOMBS_MEASURED,
            "source": L2_ENTRY_BOMBS_SOURCE,
            "level1_complete": L1_COMPLETE_BOMBS_MEASURED,
        },
    }


class BombFarmPhase(Enum):
    FARM = auto()
    DONE = auto()
    FAILED = auto()


@dataclass
class Level2BombFarmController:
    """Patrol 0x4f / 0x3e / 0x1e until ``bombs >= min_bombs``. No RAM writes."""

    min_bombs: int = L2_BOMB_CARRY
    max_frames: int = DEFAULT_MAX_FRAMES
    farm_screen: int = L2_BOMB_FARM_SCREEN
    allowed_screens: tuple[int, ...] = L2_BOMB_FARM_ROOMS
    waypoints: tuple[tuple[int, int], ...] = _FARM_WAYPOINTS
    phase: BombFarmPhase = BombFarmPhase.FARM
    frames: int = 0
    waypoint_index: int = 0
    stuck: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_bombs: int = -1
    peak_bombs: int = 0

    def reset(self) -> None:
        self.phase = BombFarmPhase.FARM
        self.frames = 0
        self.waypoint_index = 0
        self.stuck = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_bombs = -1
        self.peak_bombs = 0

    def already_satisfied(self, snap: ZeldaSnapshot) -> bool:
        return snap.bombs >= self.min_bombs and self.min_bombs > 0

    def _set_done(self, note: str) -> FrameAction:
        self.success = True
        self.phase = BombFarmPhase.DONE
        if note and (not self.notes or self.notes[-1] != note):
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "farm_done")

    def _set_failed(self, note: str) -> FrameAction:
        self.success = False
        self.phase = BombFarmPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.start_bombs < 0:
            self.start_bombs = int(snap.bombs)
            self.peak_bombs = int(snap.bombs)
        self.peak_bombs = max(self.peak_bombs, int(snap.bombs))

        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if self.min_bombs <= 0:
            return self._set_done("farm_skipped")

        if snap.mode == 17:
            return self._set_failed("link_death")

        if self.frames >= self.max_frames:
            if snap.bombs >= self.min_bombs:
                return self._set_done("farm_timeout_ok")
            self.notes.append(f"farm_timeout bombs={snap.bombs}/{self.min_bombs}")
            self.phase = BombFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "farm_timeout")

        if snap.bombs >= self.min_bombs:
            return self._set_done(f"farm_ok_{self.start_bombs}_to_{snap.bombs}")

        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            if snap.mode not in (PLAY_MODE, 8, 6, 7, 16):
                return wake_or_wait_mode(self.frames, snap.mode)
            return FrameAction(nes_idle_action(), "farm_wait_mode")

        if snap.level != 2:
            return self._set_failed("left_dungeon")
        if snap.screen not in self.allowed_screens:
            self.notes.append(f"left_screen_{snap.screen:02x}")
            return self._set_failed("left_farm_screen")

        if self.stuck > DEFAULT_STUCK_THRESHOLD:
            action, self.stuck = unstick_wiggle(self.stuck, reason="farm_unstick")
            return action

        pickups = [
            o
            for o in snap.objects
            if o.slot >= 1
            and o.type_id in _PICKUP_TYPES
            and 40 < o.y < 220
            and 8 < o.x < 248
        ]
        if pickups:
            nearest = min(
                pickups,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return _walk_toward(snap, nearest.x, nearest.y, "farm_pickup")

        enemies = [
            o
            for o in snap.objects
            if o.slot >= 1
            and o.type_id in _FARM_ENEMY_TYPES
            and o.type_id not in _IGNORE_TYPES
            and 40 < o.y < 220
            and 8 < o.x < 248
        ]
        if enemies:
            nearest = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dx = nearest.x - snap.link_x
            dy = nearest.y - snap.link_y
            if abs(dx) >= abs(dy) and abs(dx) > 4:
                d = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 4:
                d = "DOWN" if dy > 0 else "UP"
            else:
                d = "RIGHT" if dx >= 0 else "LEFT"
            return swing_action(
                self.frames,
                d,
                "farm_chase",
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )

        if not self.waypoints:
            return swing_action(
                self.frames,
                "RIGHT",
                "farm_patrol",
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )

        tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if abs(snap.link_x - tx) <= WAYPOINT_TOL and abs(snap.link_y - ty) <= WAYPOINT_TOL:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.stuck = 0
            tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if abs(snap.link_x - tx) > WAYPOINT_TOL:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        else:
            d = "DOWN" if snap.link_y < ty else "UP"
        return swing_action(
            self.frames,
            d,
            "farm",
            period=FARM_SWING_PERIOD,
            hold=FARM_SWING_HOLD,
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "min_bombs": self.min_bombs,
            "farm_screen": self.farm_screen,
            "allowed_screens": [f"0x{s:02x}" for s in self.allowed_screens],
            "start_bombs": self.start_bombs,
            "peak_bombs": self.peak_bombs,
            "waypoint_index": self.waypoint_index,
            "poke_bombs": False,
            "notes": list(self.notes),
        }


def _walk_toward(snap: ZeldaSnapshot, tx: int, ty: int, reason: str) -> FrameAction:
    if abs(snap.link_x - tx) > WAYPOINT_TOL:
        d = "RIGHT" if snap.link_x < tx else "LEFT"
    elif abs(snap.link_y - ty) > WAYPOINT_TOL:
        d = "DOWN" if snap.link_y < ty else "UP"
    else:
        return FrameAction(nes_idle_action(), f"{reason}_at")
    return FrameAction(nes_action(d), reason)


__all__ = [
    "L2_BOMB_WALLS",
    "L2_DODONGO_MOUTHS",
    "L2_BOMB_BUDGET",
    "L2_BOMB_BUDGET_BOOM",
    "L2_BOMB_CARRY",
    "L2_BOMB_BUDGET_BY_THROUGH",
    "L2_BOMB_FARM_ROOMS",
    "L2_BOMB_FARM_SCREEN",
    "L2_ENTRY_BOMBS_MEASURED",
    "L2_ENTRY_BOMBS_SOURCE",
    "L1_COMPLETE_BOMBS_MEASURED",
    "BombPlan",
    "BombFarmPhase",
    "Level2BombFarmController",
    "bomb_budget",
    "bombs_from_snapshot",
    "enough_bombs",
    "natural_bomb_plan",
    "poke_bombs_used",
    "detect_poke_bombs",
    "poke_kwarg_default",
    "spine_bomb_flags",
    "spine_bomb_report",
]
