"""Level 3 room 0x5c Darknut combat controller (Clean).

Harden wooden-sword combat against 3 Darknuts without health refill:
1. Retain waist / south positioning: never chase north onto diamond blocks (y <= 109).
2. Spend up to 3 carried bombs from the waist when Darknuts approach, keeping >= 4
   bombs for Manhandla.
3. Retreat safely after bomb placement to avoid self-inflicted blast damage.
4. Strategic sword slashes: only strike from the rear or flank (avoid Darknut shield).
   Flank or backstep when an enemy faces Link directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.engine import DungeonPhase
from zelda_i.level3.dungeon import DARKNUT_OBJECT_TYPE, ROOM_5C_SPEC
from zelda_i.ram import ZeldaSnapshot

FACING_RIGHT = 1
FACING_LEFT = 2
FACING_DOWN = 4
FACING_UP = 8

MIN_MANHANDLA_BOMBS = 4
BOMB_COOLDOWN_FRAMES = 70
RETREAT_FRAMES = 28
BOMB_ENGAGE_MIN_DIST = 20
BOMB_ENGAGE_MAX_DIST = 48
CLOSE_CONTACT_DIST = 20
SWORD_HIT_DIST = 28
NORTH_DIAMOND_TRAP_Y = 109
WAIST_CLIMB_LIMIT_Y = 125
WAIST_REST_Y = 141


@dataclass
class Level3Clear5cController:
    """Harden 0x5c Darknut combat: waist restraint, strategic slashes, bomb spend."""

    spec: Any = field(default=ROOM_5C_SPEC)
    phase: DungeonPhase = DungeonPhase.FIGHT
    frames: int = 0
    combat_frames: int = 0
    success: bool = False
    failed: bool = False
    bomb_cd: int = 0
    retreat_frames: int = 0
    retreat_dir: str = "LEFT"
    notes: list[str] = field(default_factory=list)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        self.combat_frames += 1
        if self.bomb_cd > 0:
            self.bomb_cd -= 1

        if self.retreat_frames > 0:
            self.retreat_frames -= 1
            return FrameAction(nes_action(self.retreat_dir), "retreat_bomb")

        live = [
            e
            for e in snap.objects
            if 1 <= e.slot <= 10
            and e.type_id == DARKNUT_OBJECT_TYPE
            and e.hp > 0
        ]
        if not live:
            self.phase = DungeonPhase.DONE
            self.success = True
            return FrameAction(nes_idle_action(), "done")

        target = min(
            live,
            key=lambda e: abs(e.x - snap.link_x) + abs(e.y - snap.link_y),
        )
        dx = target.x - snap.link_x
        dy = target.y - snap.link_y
        dist = abs(dx) + abs(dy)

        if abs(dx) >= abs(dy):
            towards = "RIGHT" if dx > 0 else "LEFT"
            away = "LEFT" if dx > 0 else "RIGHT"
        else:
            towards = "DOWN" if dy > 0 else "UP"
            away = "UP" if dy > 0 else "DOWN"

        # Bomb spend from waist if bombs > 4 (saving >= 4 for Manhandla)
        if (
            snap.bombs > MIN_MANHANDLA_BOMBS
            and self.bomb_cd == 0
            and BOMB_ENGAGE_MIN_DIST <= dist <= BOMB_ENGAGE_MAX_DIST
        ):
            self.bomb_cd = BOMB_COOLDOWN_FRAMES
            self.retreat_frames = RETREAT_FRAMES
            self.retreat_dir = away
            self.notes.append(f"bomb_drop_{self.frames}")
            return FrameAction(nes_action(towards, "B"), "place_bomb")

        # Avoid contact damage when enemy is too close
        if dist < CLOSE_CONTACT_DIST:
            return FrameAction(nes_action(away), "combat_backstep")

        # Do not chase north of waist onto diamonds (y <= 109)
        if target.y <= NORTH_DIAMOND_TRAP_Y or (
            towards == "UP" and snap.link_y <= WAIST_CLIMB_LIMIT_Y
        ):
            if snap.link_y < WAIST_REST_Y:
                return FrameAction(nes_action("DOWN"), "stay_south")
            if snap.link_x < 64:
                return FrameAction(nes_action("RIGHT"), "patrol_waist")
            if snap.link_x > 96:
                return FrameAction(nes_action("LEFT"), "patrol_waist")
            return FrameAction(nes_idle_action(), "wait_for_darknut")

        # Strategic sword slashes (side / rear hits only; avoid shield)
        shield_facing = (
            (target.facing == FACING_RIGHT and towards == "LEFT")
            or (target.facing == FACING_LEFT and towards == "RIGHT")
            or (target.facing == FACING_DOWN and towards == "UP")
            or (target.facing == FACING_UP and towards == "DOWN")
        )
        if shield_facing:
            if towards in ("LEFT", "RIGHT"):
                perp = "DOWN" if snap.link_y <= 173 else "UP"
            else:
                perp = "RIGHT" if snap.link_x <= 96 else "LEFT"
            return FrameAction(nes_action(perp), "flank")

        if dist <= SWORD_HIT_DIST:
            return FrameAction(nes_action(towards, "A"), "sword_slash")

        return FrameAction(nes_action(towards), "approach")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "combat_frames": self.combat_frames,
            "notes": list(self.notes),
            "phase": self.phase.name,
            "route_eligible": False,
        }


__all__ = ["Level3Clear5cController"]
