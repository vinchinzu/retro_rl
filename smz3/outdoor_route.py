"""Fortune Teller overworld → Link's House (no sword).

After the SM→Z3 portal settles on light-world screen ``$35`` (Fortune Teller
exterior), Link has **no sword** on the test seed (uncle sword not yet
collected).  Navigation avoids combat: follow a verified corridor policy and
**side-step** when hostiles close in (never reverse the phase goal).

Verified path (PortalSettled, test seed 1337):

1. Spawn ``$35`` ~(2648, 3275) is south of the Fortune Teller door.
2. Pure UP at spawn enters the house — **forbidden**.
3. Walk **DOWN** to ``y ≥ 3440``, then **RIGHT** to corridor ``x ≈ 2704``.
4. Hold **UP** on that X band → screen ``$2D``.
5. **UP+LEFT** (not pure LEFT; wall at entry Y) → screen ``$2C`` (Link's House).

Screen path (8×8 light-world grid)::

    $35 → $2D → $2C

This controller is an honest **fixed corridor policy** for the verified early
leg (not a general multi-target OW navigator). Screen BFS helpers remain for
tooling / tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from alttp.overworld import direction_to_screen, next_screen_in_path, shortest_screen_path
from alttp.primitives import SpriteSnapshot, active_sprites
from alttp.ram import LINKS_HOUSE_SCREEN
from smz3.control import hold, is_z3_dead, wait_z3_control
from smz3.ram import ComboSnapshot, snapshot_env
from smz3.segment import SegmentResult

# Re-export shared wait for scripts that imported it from outdoor_route.
__all__ = [
    "CORRIDOR_X",
    "FORTUNE_TELLER_SCREEN",
    "LINKS_HOUSE_OW_SCREEN",
    "OUTDOOR_SCREEN_PATH",
    "OutdoorSegmentResult",
    "choose_outdoor_buttons",
    "on_links_house_screen",
    "outdoor_path_screens",
    "preferred_direction",
    "run_fortune_teller_to_links_house",
    "wait_z3_control",
]

# Gameplay targets (stable-retro screen ids).
FORTUNE_TELLER_SCREEN = 0x35
LINKS_HOUSE_OW_SCREEN = LINKS_HOUSE_SCREEN  # 0x2C
MID_SCREEN = 0x2D
OUTDOOR_SCREEN_PATH: tuple[int, ...] = (0x35, 0x2D, 0x2C)

# Northbound corridor past Fortune Teller house (world X).
CORRIDOR_X = 2704
CORRIDOR_X_TOL = 16
HOUSE_CLEAR_Y = 3440
HOUSE_SOUTH_MAX_Y = 3480

FLEE_RADIUS = 40
SOFT_RADIUS = 64

OW_HOSTILE_TYPES = frozenset(
    {
        0x08,
        0x1B,
        0x41,
        0x46,
        0x55,
        0x58,
    }
)

MAX_OUTDOOR_FRAMES = 12_000
_OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


@dataclass
class OutdoorSegmentResult(SegmentResult):
    """Outcome of Fortune Teller OW → Link's House screen."""

    goal: str = "fortune_teller_to_links_house"
    start_screen: int | None = None
    final_screen: int | None = None
    screens_visited: list[int] = field(default_factory=list)
    fled_frames: int = 0
    died: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        snap = self.final_snapshot
        d.update(
            {
                "start_screen": (
                    f"0x{self.start_screen:02X}" if self.start_screen is not None else None
                ),
                "final_screen": (
                    f"0x{self.final_screen:02X}" if self.final_screen is not None else None
                ),
                "screens_visited": [f"0x{s:02X}" for s in self.screens_visited],
                "screen_path": [f"0x{s:02X}" for s in OUTDOOR_SCREEN_PATH],
                "fled_frames": self.fled_frames,
                "died": self.died,
                "snapshot": snap.to_dict() if snap is not None else None,
            }
        )
        # Prefer "snapshot" key used by probes; drop duplicate final_snapshot noise.
        d.pop("final_snapshot", None)
        return d


def preferred_direction(
    snap: ComboSnapshot, target_screen: int = LINKS_HOUSE_OW_SCREEN
) -> str | None:
    """Cardinal toward the next screen on the shortest path to *target_screen*."""
    if not snap.z3_controllable or snap.z3_indoors:
        return None
    if snap.z3_screen_id == target_screen:
        return None
    try:
        nxt = next_screen_in_path(snap.z3_screen_id, target_screen)
    except ValueError:
        return None
    return direction_to_screen(snap.z3_screen_id, nxt)


def _nearest_hostile(
    snap: ComboSnapshot,
    sprites: tuple[SpriteSnapshot, ...],
    *,
    radius: int,
) -> SpriteSnapshot | None:
    best: SpriteSnapshot | None = None
    best_d = radius + 1
    for sp in sprites:
        if sp.sprite_type not in OW_HOSTILE_TYPES:
            continue
        d = abs(sp.x - snap.z3_link_x) + abs(sp.y - snap.z3_link_y)
        if d < best_d:
            best_d = d
            best = sp
    return best


def _dedupe(buttons: list[str], *, max_n: int = 2) -> list[str]:
    seen: list[str] = []
    for b in buttons:
        if b not in seen:
            seen.append(b)
    return seen[:max_n]


def _phase_buttons(snap: ComboSnapshot, *, house_cleared: bool) -> list[str]:
    """Primary buttons from the verified outdoor corridor policy."""
    scr = snap.z3_screen_id
    x, y = snap.z3_link_x, snap.z3_link_y

    if scr == FORTUNE_TELLER_SCREEN:
        if not house_cleared:
            if y < HOUSE_CLEAR_Y:
                if x < CORRIDOR_X - CORRIDOR_X_TOL:
                    return ["DOWN", "RIGHT"]
                if x > CORRIDOR_X + CORRIDOR_X_TOL:
                    return ["DOWN", "LEFT"]
                return ["DOWN"]
            if y > HOUSE_SOUTH_MAX_Y:
                return ["UP", "RIGHT"] if x < CORRIDOR_X else ["UP"]
        if x < CORRIDOR_X - CORRIDOR_X_TOL:
            return ["RIGHT", "UP"] if house_cleared else ["RIGHT"]
        if x > CORRIDOR_X + CORRIDOR_X_TOL:
            return ["LEFT", "UP"] if house_cleared else ["LEFT"]
        return ["UP"]

    if scr == MID_SCREEN:
        return ["UP", "LEFT"]

    if scr == LINKS_HOUSE_OW_SCREEN:
        return []

    goal = preferred_direction(snap)
    return [goal] if goal else []


def _side_step(primary: str, snap: ComboSnapshot, enemy: SpriteSnapshot) -> str | None:
    """Perpendicular away from enemy; never reverse *primary*."""
    if primary in ("UP", "DOWN"):
        dx = snap.z3_link_x - enemy.x
        side = "RIGHT" if dx >= 0 else "LEFT"
        if abs(dx) < 4:
            side = "RIGHT" if snap.z3_link_x < CORRIDOR_X else "LEFT"
        return side if side != _OPP.get(primary) else None
    if primary in ("LEFT", "RIGHT"):
        # On $2D the path is NW — always prefer UP (never reverse LEFT/RIGHT).
        if snap.z3_screen_id == MID_SCREEN:
            side = "UP"
        else:
            dy = snap.z3_link_y - enemy.y
            side = "DOWN" if dy >= 0 else "UP"
        return side if side != _OPP.get(primary) else None
    return None


def choose_outdoor_buttons(
    snap: ComboSnapshot,
    sprites: tuple[SpriteSnapshot, ...],
    *,
    house_cleared: bool = False,
) -> tuple[tuple[str, ...], bool]:
    """Pick D-pad buttons for the fixed Fortune→House corridor policy.

    Second value is True when side-stepping a hostile. Flee only adds a
    perpendicular button so Link slips past soldiers without reversing progress
    or walking into the Fortune Teller door.
    """
    base = _phase_buttons(snap, house_cleared=house_cleared)
    # Strip UP only before the sticky south-clear completes (door band).
    if (
        snap.z3_screen_id == FORTUNE_TELLER_SCREEN
        and not house_cleared
        and snap.z3_link_y < HOUSE_CLEAR_Y
    ):
        base = [b for b in base if b != "UP"] or ["DOWN"]

    if not base:
        return (), False

    primary = base[0]
    sprinting_north = (
        snap.z3_screen_id == FORTUNE_TELLER_SCREEN
        and house_cleared
        and abs(snap.z3_link_x - CORRIDOR_X) <= CORRIDOR_X_TOL
        and primary == "UP"
    )

    hostile = _nearest_hostile(
        snap, sprites, radius=FLEE_RADIUS if sprinting_north else SOFT_RADIUS
    )
    if hostile is None:
        return tuple(_dedupe(base)), False

    d = abs(hostile.x - snap.z3_link_x) + abs(hostile.y - snap.z3_link_y)
    if sprinting_north and d > FLEE_RADIUS:
        return tuple(_dedupe(base)), False
    if d > FLEE_RADIUS:
        return tuple(_dedupe(base)), False

    side = _side_step(primary, snap, hostile)
    if side is None:
        return tuple(_dedupe(base)), True
    return tuple(_dedupe([primary, side] + base[1:])), True


def on_links_house_screen(snap: ComboSnapshot) -> bool:
    return (
        snap.z3_controllable
        and not snap.z3_indoors
        and snap.z3_screen_id == LINKS_HOUSE_OW_SCREEN
    )


def run_fortune_teller_to_links_house(
    env: Any,
    *,
    start_frame: int = 0,
    max_frames: int = MAX_OUTDOOR_FRAMES,
    on_frame: Callable[[int, ComboSnapshot], None] | None = None,
    step_size: int = 4,
) -> OutdoorSegmentResult:
    """Drive Link from Fortune Teller OW to Link's House screen (no sword).

    Expects env already at settled Z3 overworld (e.g. ``PortalSettled``).
    Does not grant items or poke progression RAM.
    """
    frame = start_frame
    frame, snap = wait_z3_control(env, start_frame=frame)
    start_screen = snap.z3_screen_id if snap.z3_controllable else None
    visited: list[int] = []
    fled_frames = 0
    stuck = 0
    prev_xy: tuple[int, int] | None = None
    house_cleared = bool(
        snap.z3_controllable
        and snap.z3_screen_id == FORTUNE_TELLER_SCREEN
        and snap.z3_link_y >= HOUSE_CLEAR_Y
    )

    if not snap.z3_controllable:
        return OutdoorSegmentResult(
            ok=False,
            frames=frame - start_frame,
            detail=f"not controllable at start module=${snap.z3_module:02X}",
            start_screen=start_screen,
            final_screen=snap.z3_screen_id,
            final_snapshot=snap,
        )

    if on_frame is not None:
        on_frame(frame, snap)

    while frame - start_frame < max_frames:
        snap = snapshot_env(env, frame=frame)
        if on_frame is not None:
            on_frame(frame, snap)

        if is_z3_dead(snap):
            return OutdoorSegmentResult(
                ok=False,
                frames=frame - start_frame,
                detail="Link died",
                start_screen=start_screen,
                final_screen=snap.z3_screen_id,
                screens_visited=visited,
                final_snapshot=snap,
                fled_frames=fled_frames,
                died=True,
            )

        if not snap.z3_controllable:
            pre_mod = snap.z3_module
            frame, snap = wait_z3_control(env, start_frame=frame, max_frames=300)
            if is_z3_dead(snap):
                return OutdoorSegmentResult(
                    ok=False,
                    frames=frame - start_frame,
                    detail="Link died during transition",
                    start_screen=start_screen,
                    final_screen=snap.z3_screen_id,
                    screens_visited=visited,
                    final_snapshot=snap,
                    fled_frames=fled_frames,
                    died=True,
                )
            if not snap.z3_controllable:
                return OutdoorSegmentResult(
                    ok=False,
                    frames=frame - start_frame,
                    detail=(
                        f"lost control module=${snap.z3_module:02X} "
                        f"(was ${pre_mod:02X}) — likely Fortune Teller door"
                    ),
                    start_screen=start_screen,
                    final_screen=snap.z3_screen_id,
                    screens_visited=visited,
                    final_snapshot=snap,
                    fled_frames=fled_frames,
                )
            continue

        if snap.z3_screen_id not in visited:
            visited.append(snap.z3_screen_id)

        if (
            snap.z3_screen_id == FORTUNE_TELLER_SCREEN
            and snap.z3_link_y >= HOUSE_CLEAR_Y
        ):
            house_cleared = True
        if snap.z3_screen_id != FORTUNE_TELLER_SCREEN:
            house_cleared = True

        if on_links_house_screen(snap):
            frame = hold(env, None, 30, frame=frame)
            snap = snapshot_env(env, frame=frame)
            return OutdoorSegmentResult(
                ok=True,
                frames=frame - start_frame,
                detail=(
                    f"reached Link's House screen=${snap.z3_screen_id:02X} "
                    f"xy=({snap.z3_link_x},{snap.z3_link_y}) "
                    f"fled_frames={fled_frames}"
                ),
                start_screen=start_screen,
                final_screen=snap.z3_screen_id,
                screens_visited=visited,
                final_snapshot=snap,
                fled_frames=fled_frames,
            )

        sprites = active_sprites(env)
        buttons, fleeing = choose_outdoor_buttons(
            snap, sprites, house_cleared=house_cleared
        )
        if fleeing:
            fled_frames += step_size

        if not buttons:
            frame = hold(env, None, step_size, frame=frame)
        else:
            frame = hold(env, tuple(buttons), step_size, frame=frame)

        snap2 = snapshot_env(env, frame=frame)
        xy = (snap2.z3_link_x, snap2.z3_link_y)
        if xy == prev_xy and snap2.z3_controllable:
            stuck += 1
        else:
            stuck = 0
            prev_xy = xy

        if stuck >= 12:
            stuck = 0
            scr = snap2.z3_screen_id
            if scr == FORTUNE_TELLER_SCREEN and not house_cleared:
                frame = hold(env, ("DOWN", "RIGHT"), 20, frame=frame)
            elif scr == FORTUNE_TELLER_SCREEN:
                if snap2.z3_link_x < CORRIDOR_X:
                    frame = hold(env, ("RIGHT",), 16, frame=frame)
                elif snap2.z3_link_x > CORRIDOR_X:
                    frame = hold(env, ("LEFT",), 16, frame=frame)
                frame = hold(env, ("UP",), 20, frame=frame)
            elif scr == MID_SCREEN:
                frame = hold(env, ("UP",), 24, frame=frame)
                frame = hold(env, ("LEFT",), 24, frame=frame)
            else:
                frame = hold(env, ("UP", "LEFT"), 16, frame=frame)

    snap = snapshot_env(env, frame=frame)
    return OutdoorSegmentResult(
        ok=False,
        frames=frame - start_frame,
        detail=(
            f"timeout screen=${snap.z3_screen_id:02X} "
            f"xy=({snap.z3_link_x},{snap.z3_link_y}) visited={visited}"
        ),
        start_screen=start_screen,
        final_screen=snap.z3_screen_id,
        screens_visited=visited,
        final_snapshot=snap,
        fled_frames=fled_frames,
    )


def outdoor_path_screens(
    start: int = FORTUNE_TELLER_SCREEN,
    target: int = LINKS_HOUSE_OW_SCREEN,
) -> list[int]:
    """BFS screen path (same as :func:`alttp.overworld.shortest_screen_path`)."""
    return shortest_screen_path(start, target)
