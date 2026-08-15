"""Live Level 9 Ganon and ending anchors.

The room/object/RAM values in this module were verified in fceumm on
2026-08-14.  They support an explicitly composed endgame recon fixture; they
are not evidence that the natural Level 9 route has earned these items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ops import B_ITEM_ARROWS, B_ITEM_BOMBS
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
    read_snapshot,
)

LEVEL9 = 9

ROOM_BEFORE_GANON = 0x52
ROOM_GANON = 0x42
ROOM_ZELDA = 0x32

OBJ_GANON = 0x3E
OBJ_ZELDA = 0x37
OBJ_GUARD_FIRE = 0x3F

ADDR_GANON_OBJ_PHASE_BASE = 0x042C
ADDR_GANON_SCENE_PHASE = 0x0445
ADDR_LAST_BOSS_DEFEATED = 0x0672

GANON_SCENE_FIGHT = 2
GANON_HP_START = 0xF0
GANON_BROWN_STATE = 0xFF
GANON_DEFEATED_PHASE = 0xFF

MODE_ENDING = 0x13
ENDING_SUBMODE_CREDITS = 3
ENDING_SUBMODE_FINAL_SCREEN = 4


def ganon_object(snap: ZeldaSnapshot) -> ZeldaObject | None:
    """Return the live Ganon slot, including his invisible phases."""
    return next((obj for obj in snap.objects if obj.type_id == OBJ_GANON), None)


def zelda_object(snap: ZeldaSnapshot) -> ZeldaObject | None:
    return next((obj for obj in snap.objects if obj.type_id == OBJ_ZELDA), None)


def in_room_before_ganon(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_BEFORE_GANON
    )


def in_ganon_fight(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_GANON
        and ganon_object(snap) is not None
    )


def in_zelda_room(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == PLAY_MODE
        and snap.level == LEVEL9
        and snap.screen == ROOM_ZELDA
        and zelda_object(snap) is not None
    )


def ganon_is_brown(snap: ZeldaSnapshot) -> bool:
    boss = ganon_object(snap)
    # The engine seeds 0xFF, then decrements every other frame.  The first
    # externally observable post-step value is commonly 0xFE; any nonzero
    # ObjState is the brown / Silver-Arrow-vulnerable phase.
    return boss is not None and boss.state != 0


def ganon_defeated(ram) -> bool:
    return int(ram[ADDR_LAST_BOSS_DEFEATED]) != 0


def credits_rolling(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == MODE_ENDING
        and snap.is_updating_mode != 0
        and snap.submode == ENDING_SUBMODE_CREDITS
    )


def final_ending_screen(snap: ZeldaSnapshot) -> bool:
    return (
        snap.mode == MODE_ENDING
        and snap.is_updating_mode != 0
        and snap.submode == ENDING_SUBMODE_FINAL_SCREEN
    )


def _toward(link_x: int, link_y: int, target_x: int, target_y: int) -> str:
    dx = int(target_x) - int(link_x)
    dy = int(target_y) - int(link_y)
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def _sword_direction(snap: ZeldaSnapshot, boss: ZeldaObject) -> str | None:
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if in_sword_hitbox(
            snap.link_x,
            snap.link_y,
            direction,
            boss.x,
            boss.y,
            reach=24,
            half_width=16,
        ):
            return direction
    return None


def _arrow_direction(snap: ZeldaSnapshot, boss: ZeldaObject) -> str | None:
    dx = int(boss.x) - int(snap.link_x)
    dy = int(boss.y) - int(snap.link_y)
    if abs(dx) <= 8 and dy:
        return "DOWN" if dy > 0 else "UP"
    if abs(dy) <= 8 and dx:
        return "RIGHT" if dx > 0 else "LEFT"
    return None


def ganon_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
) -> tuple[list[int], str, int]:
    """Choose one Ganon combat frame from live boss coordinates.

    A is pulsed because holding it does not start a second sword swing.  Once
    state ``0xFF`` exposes brown Ganon, B fires only on an aligned axis.
    """
    boss = ganon_object(snap)
    if boss is None:
        return nes_idle_action(), "wait_ganon", max(0, cooldown - 1)
    if cooldown > 0:
        return nes_idle_action(), "attack_cooldown", cooldown - 1

    if boss.state != 0:
        arrow_dir = _arrow_direction(snap, boss)
        if arrow_dir is not None:
            return nes_action(arrow_dir, "B"), "silver_arrow", 16
        # Align on the nearer axis before firing across the other one.
        if abs(int(boss.x) - int(snap.link_x)) <= abs(
            int(boss.y) - int(snap.link_y)
        ):
            direction = "RIGHT" if snap.link_x < boss.x else "LEFT"
        else:
            direction = "DOWN" if snap.link_y < boss.y else "UP"
        return nes_action(direction), "align_arrow", 0

    sword_dir = _sword_direction(snap, boss)
    if sword_dir is not None:
        return nes_action(sword_dir, "A"), "sword_pulse", 12
    direction = _toward(snap.link_x, snap.link_y, boss.x, boss.y)
    return nes_action(direction), "chase_ganon", 0


@dataclass
class GanonFightController:
    """Coordinate-chase controller for the four-hit + Silver Arrow finish."""

    max_frames: int = 7000
    frames: int = 0
    sword_pulses: int = 0
    arrow_pulses: int = 0
    selected_item_writes: int = 0
    brown_seen: bool = False
    hp_changes: list[int] = field(default_factory=list)
    reasons: dict[str, int] = field(default_factory=dict)

    def run(
        self,
        env: Any,
        *,
        assist: Any | None = None,
        total: list[int] | None = None,
    ) -> dict[str, Any]:
        total_frames = total if total is not None else [0]
        cooldown = 0
        last_hp: int | None = None

        for _ in range(self.max_frames):
            ram = env.get_ram()
            snap = read_snapshot(ram)
            if ganon_defeated(ram):
                return self.report(ok=True, snap=snap, ram=ram)

            boss = ganon_object(snap)
            if boss is not None:
                if boss.hp != last_hp:
                    self.hp_changes.append(int(boss.hp))
                    last_hp = boss.hp
                self.brown_seen = self.brown_seen or boss.state != 0

            action, reason, cooldown = ganon_action(snap, cooldown=cooldown)
            if reason == "sword_pulse":
                self.sword_pulses += 1
            elif reason == "silver_arrow":
                # The recon fixture preselects arrows.  Keep a disclosed
                # fallback for callers that load an older fixture; it makes
                # that run fixture-only, never route-eligible.
                if int(env.get_ram()[ADDR_SELECTED_ITEM]) != B_ITEM_ARROWS:
                    env.unwrapped.data.memory.assign(
                        ADDR_SELECTED_ITEM, "|u1", B_ITEM_ARROWS
                    )
                    self.selected_item_writes += 1
                self.arrow_pulses += 1
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            env.step(action)
            self.frames += 1
            total_frames[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total_frames[0])

        ram = env.get_ram()
        return self.report(ok=False, snap=read_snapshot(ram), ram=ram)

    def report(
        self,
        *,
        ok: bool,
        snap: ZeldaSnapshot,
        ram,
    ) -> dict[str, Any]:
        boss = ganon_object(snap)
        return {
            "ok": ok,
            "frames": self.frames,
            "sword_pulses": self.sword_pulses,
            "arrow_pulses": self.arrow_pulses,
            "selected_item_writes": self.selected_item_writes,
            "brown_seen": self.brown_seen,
            "hp_changes": list(self.hp_changes),
            "reasons": dict(self.reasons),
            "last_boss_defeated": int(ram[ADDR_LAST_BOSS_DEFEATED]),
            "ganon_scene_phase": int(ram[ADDR_GANON_SCENE_PHASE]),
            "boss": (
                {
                    "slot": boss.slot,
                    "hp": boss.hp,
                    "state": boss.state,
                    "object_phase": int(
                        ram[ADDR_GANON_OBJ_PHASE_BASE + boss.slot]
                    ),
                }
                if boss is not None
                else None
            ),
        }


__all__ = [
    "ADDR_GANON_OBJ_PHASE_BASE",
    "ADDR_GANON_SCENE_PHASE",
    "ADDR_LAST_BOSS_DEFEATED",
    "B_ITEM_ARROWS",
    "B_ITEM_BOMBS",
    "ENDING_SUBMODE_CREDITS",
    "ENDING_SUBMODE_FINAL_SCREEN",
    "GANON_BROWN_STATE",
    "GANON_DEFEATED_PHASE",
    "GANON_HP_START",
    "GANON_SCENE_FIGHT",
    "GanonFightController",
    "LEVEL9",
    "MODE_ENDING",
    "OBJ_GANON",
    "OBJ_GUARD_FIRE",
    "OBJ_ZELDA",
    "ROOM_BEFORE_GANON",
    "ROOM_GANON",
    "ROOM_ZELDA",
    "credits_rolling",
    "final_ending_screen",
    "ganon_action",
    "ganon_defeated",
    "ganon_is_brown",
    "ganon_object",
    "in_ganon_fight",
    "in_room_before_ganon",
    "in_zelda_room",
    "zelda_object",
]
