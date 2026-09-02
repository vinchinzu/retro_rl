"""Level 6 Gohma 0x1C: poke wooden arrows, shoot the spawn-open eye.

Leftover is north2c play 0x1C ``(120,205)``. Bow is earned on the L1
Survival splice. Operator exception: ``ADDR_ARROWS=1`` + B-slot 2 until
the 80R shop splice. One UP+B while ``|gx-x|<=8`` (spawn gx~128). Do
not walk to y=165 first (v3 f29 gx=141 dx=21, ghp stayed 32). Do not
align_x (v2 f144 spray). Do not hold UP on cooldown (v1 leftover
``(115,93)``). Do not occupancy. Do not write ``ADDR_BOW``. Do not poke
doors/keys. Isolated BFS banned. Heart / north 0x0C / TF ``0x20`` are
later SpineHops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.assist import poke_wooden_arrows
from zelda_i.dungeon.ids import GOHMA_BLUE_OBJECT_TYPE, GOHMA_OBJECT_TYPE
from zelda_i.dungeon.hop_controller import (
    CELLAR_MODE,
    HopController,
    WAIT_SCROLL_B,
)
from zelda_i.level6.door_hop import NORTH2C_SPEC, SOUTH1D_SPEC, WEST2D_SPEC, door_hop_stages
from zelda_i.level6.occupancy import record_l6_walk
from zelda_i.level6.overworld import LEVEL6, LEVEL6_GOHMA_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

__all__ = [
    "COLUMN_DX",
    "GOHMA_MAX_FRAMES",
    "SPAWN_WINDOW",
    "Level6GohmaController",
    "gohma_live",
    "level6_gohma_stages",
    "level6_gohma_success",
    "make_gohma_controller",
]

# Spawn-window hop. Occupancy poked-warp kill was 54f / 1 pulse; do not spray.
GOHMA_MAX_FRAMES = 120
COLUMN_DX = 8
SHOT_COOLDOWN = 20
SPAWN_WINDOW = 54
GOHMA_TYPES = frozenset({GOHMA_OBJECT_TYPE, GOHMA_BLUE_OBJECT_TYPE})
_SKIP_TYPES = frozenset({0, 0xFF})
GOHMA_WAIT = tuple(sorted(set(WAIT_SCROLL_B) | {CELLAR_MODE}))


def gohma_live(snap: ZeldaSnapshot) -> list:
    """Red 0x33 (or blue 0x34) slots 1–12. TYPE presence; HP may be 0."""
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in GOHMA_TYPES
    ]


@dataclass
class Level6GohmaController(HopController):
    """Poke wooden arrows, one UP+B while gx is still in column, idle."""

    spec_id: str = "level6_gohma_0x1c"
    room: int = LEVEL6_GOHMA_ROOM
    max_frames: int = GOHMA_MAX_FRAMES
    wait_modes: tuple[int, ...] = GOHMA_WAIT
    done_reason: str = "body_gone"
    cooldown: int = 0
    saw_gohma: bool = False
    poked: bool = False
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    inventory_assist: dict[str, Any] | None = None
    env: Any | None = None
    arrow_pulses: int = 0

    def bind_env(self, env: Any) -> None:
        self.env = env

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_bow={int(snap.bow)}_arrows={int(snap.arrows)}"
            f"_pulses={self.arrow_pulses}"
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
            period=16,
            misses=0,
            force=force,
        )
        if force or self.frames <= SPAWN_WINDOW or self.frames % 16 == 0:
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
                    "rupees": int(snap.rupees),
                    "misses": 0,
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
        return None

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

        # One pulse then idle. v1 hold-UP walked through the body; v2 spray
        # after align_x spent 43R at the closed eye; v3 waited for y=165
        # and shot at gx=141.
        if self.arrow_pulses >= 1:
            if self.cooldown > 0:
                return FrameAction(nes_idle_action(), "shot_wait")
            return FrameAction(nes_idle_action(), "spawn_wait")

        body = bodies[0]
        dx = abs(int(body.x) - int(snap.link_x))
        if dx > COLUMN_DX:
            return FrameAction(nes_idle_action(), "column_wait")

        if self.cooldown > 0:
            return FrameAction(nes_idle_action(), "shot_wait")
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
            "policy": "poke ADDR_ARROWS=1 B=2; one UP+B while |gx-x|<=8; idle",
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
