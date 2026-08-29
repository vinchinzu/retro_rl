"""Level 6 Gohma 0x1C: poke wooden arrows, shoot the open eye.

Leftover is north2c play 0x1C ``(120,205)``. Bow must already be earned
(L1 Survival splice). Operator exception: ``ADDR_ARROWS=1`` + B-slot 2.
Do not write ``ADDR_BOW``. Do not poke doors/keys. Isolated BFS banned.
Enter-stop was unarmed; this hop is the kill. TF ``0x20`` is next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.assist import poke_wooden_arrows
from zelda_i.dungeon_ids import (
    FIREBALL_OBJECT_TYPE,
    GOHMA_BLUE_OBJECT_TYPE,
    GOHMA_OBJECT_TYPE,
    MANHANDLA_PROJECTILE_TYPE,
)
from zelda_i.hop_controller import CELLAR_MODE, HopController, WAIT_SCROLL_B
from zelda_i.level6_door_hop import NORTH2C_SPEC, SOUTH1D_SPEC, WEST2D_SPEC, door_hop_stages
from zelda_i.level6_occupancy import occupancy_new_miss, record_l6_walk
from zelda_i.level6_overworld import LEVEL6, LEVEL6_GOHMA_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "ALIGN_X_TOL",
    "GOHMA_MAX_FRAMES",
    "GOHMA_STAND_X",
    "GOHMA_STAND_Y",
    "Level6GohmaController",
    "gohma_live",
    "level6_gohma_stages",
    "level6_gohma_success",
    "make_gohma_controller",
]

GOHMA_MAX_FRAMES = 20000
GOHMA_STAND_X = 120
GOHMA_STAND_Y = 165
ALIGN_X_TOL = 8
SHOT_COOLDOWN = 20
GOHMA_TYPES = frozenset({GOHMA_OBJECT_TYPE, GOHMA_BLUE_OBJECT_TYPE})
_PROJECTILES = frozenset({FIREBALL_OBJECT_TYPE, MANHANDLA_PROJECTILE_TYPE})
_SKIP_TYPES = frozenset({0, 0xFF})
SAMPLE_PERIOD = 16
GOHMA_WAIT = tuple(sorted(set(WAIT_SCROLL_B) | {CELLAR_MODE}))


def gohma_live(snap: ZeldaSnapshot) -> list:
    """Red 0x33 (or blue 0x34) slots 1–12. TYPE presence; HP may be 0."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in GOHMA_TYPES
    ]


def _projectiles(snap: ZeldaSnapshot) -> list:
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in _PROJECTILES
    ]


@dataclass
class Level6GohmaController(HopController):
    """Poke wooden arrows, occupancy inland, x-align, UP+B until Gohma gone."""

    spec_id: str = "level6_gohma_0x1c"
    room: int = LEVEL6_GOHMA_ROOM
    max_frames: int = GOHMA_MAX_FRAMES
    wait_modes: tuple[int, ...] = GOHMA_WAIT
    done_reason: str = "body_gone"
    stand_x: int = GOHMA_STAND_X
    stand_y: int = GOHMA_STAND_Y
    cooldown: int = 0
    saw_gohma: bool = False
    poked: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    inventory_assist: dict[str, Any] | None = None
    env: Any | None = None
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)
    arrow_pulses: int = 0

    def bind_env(self, env: Any) -> None:
        self.env = env

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_bow={int(snap.bow)}_arrows={int(snap.arrows)}"
        )

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"body_gone_{snap.link_x}_{snap.link_y}_pulses={self.arrow_pulses}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        bodies = gohma_live(snap)
        body = bodies[0] if bodies else None
        self.leftover = record_l6_walk(
            self.samples,
            snap,
            reason=action.reason,
            frames=self.frames,
            period=SAMPLE_PERIOD,
            misses=self.walker.misses,
            force=force,
        )
        if force or self.frames <= 2 or self.frames % 250 == 0:
            types = [
                int(obj.type_id)
                for obj in snap.objects
                if 1 <= obj.slot <= 12 and int(obj.type_id) not in _SKIP_TYPES
            ]
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "reason": action.reason,
                    "gx": None if body is None else int(body.x),
                    "gy": None if body is None else int(body.y),
                    "ghp": None if body is None else int(body.hp),
                    "gst": None if body is None else int(body.state),
                    "n": len(bodies),
                    "types": types,
                    "bow": int(snap.bow),
                    "arrows": int(snap.arrows),
                    "misses": self.walker.misses,
                }
            )
        return action

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        bodies = gohma_live(snap)
        if bodies:
            self.saw_gohma = True
            return False
        return self.saw_gohma

    def _poke(self, snap: ZeldaSnapshot) -> FrameAction | None:
        if self.poked:
            return None
        if int(snap.bow) < 1:
            return self.mark_fail("unarmed_no_bow")
        if self.env is None:
            if int(snap.arrows) >= 1:
                self.poked = True
                self.notes.append("arrows_already_set")
                return None
            return self.mark_fail("no_env_for_arrow_write")
        self.inventory_assist = poke_wooden_arrows(
            self.env, from_arrows=int(snap.arrows), select=True
        )
        self.poked = True
        n = int(self.inventory_assist.get("inventory_writes") or 0)
        self.notes.append(
            f"arrow_poke_writes={n}_from={int(snap.arrows)}"
        )
        if int(self.inventory_assist.get("progression_writes") or 0) != 0:
            return self.mark_fail("arrow_poke_progression")
        return FrameAction(nes_idle_action(), "arrow_poke")

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")

        poked = self._poke(snap)
        if poked is not None:
            return poked

        bodies = gohma_live(snap)
        if not bodies:
            return FrameAction(nes_idle_action(), "wait_body")

        if snap.link_y > self.stand_y + 4:
            xy = (int(snap.link_x), int(snap.link_y))
            occupancy_new_miss(self.walker, xy, allow_first=True)
            dest = (self.stand_x, self.stand_y)
            direction = self.walker.next_dir(xy, dest)
            if direction is None:
                return FrameAction(nes_idle_action(), "occupancy_stand")
            return FrameAction(nes_action(direction), "inland_path")

        body = bodies[0]
        balls = _projectiles(snap)
        if balls:
            nearest = min(
                balls,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            if abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y) <= 24:
                dodge = "RIGHT" if nearest.x <= snap.link_x else "LEFT"
                return FrameAction(nes_action(dodge), "fb_dodge")

        dx = int(body.x) - int(snap.link_x)
        if abs(dx) > ALIGN_X_TOL:
            face = "RIGHT" if dx > 0 else "LEFT"
            return FrameAction(nes_action(face), "align_x")
        if self.cooldown > 0:
            return FrameAction(nes_action("UP"), "face_up")
        self.cooldown = SHOT_COOLDOWN
        self.arrow_pulses += 1
        return FrameAction(nes_action("UP", "B"), "arrow_shot")

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.cooldown > 0:
            self.cooldown -= 1
        return super().step(snap)

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "leftover": dict(self.leftover),
            "inventory_assist": self.inventory_assist,
            "policy": "poke ADDR_ARROWS=1 B=2; occupancy to (120,165); UP+B",
            "saw_gohma": self.saw_gohma,
            "arrow_pulses": self.arrow_pulses,
            "spec_id": self.spec_id,
            "room": self.room,
            "body_type": GOHMA_OBJECT_TYPE,
        }


def make_gohma_controller() -> Level6GohmaController:
    """Kill Gohma 0x1C with poked wooden arrows. Bow already earned."""
    return Level6GohmaController()


def level6_gohma_stages():
    """West 0x2C KEY-UP leftover → poke arrows → Gohma gone."""
    return (
        *door_hop_stages(SOUTH1D_SPEC),
        *door_hop_stages(WEST2D_SPEC),
        *door_hop_stages(NORTH2C_SPEC),
        ("level6_gohma_0x1c", make_gohma_controller(), GOHMA_MAX_FRAMES),
    )


def level6_gohma_success(snap: ZeldaSnapshot) -> bool:
    """Play 0x1C, Gohma gone, bow+arrows set. TF 0x20 is the next hop."""
    if snap.level != LEVEL6 or snap.triforce != 0x1F:
        return False
    if snap.mode != PLAY_MODE or snap.transitioning or snap.screen != LEVEL6_GOHMA_ROOM:
        return False
    if int(snap.bow) < 1 or int(snap.arrows) < 1:
        return False
    return not gohma_live(snap)
