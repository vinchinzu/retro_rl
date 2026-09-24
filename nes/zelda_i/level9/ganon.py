"""Live Level 9 Ganon and ending anchors.

The room/object/RAM values in this module were verified in fceumm on
2026-08-14.  They support an explicitly composed endgame recon fixture; they
are not evidence that the natural Level 9 route has earned these items.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.combat import (
    CONTACT_CHEBYSHEV,
    CONTACT_MANHATTAN,
    direction_to_facing,
)
from zelda_i.dungeon.ids import FIREBALL_OBJECT_TYPE, MANHANDLA_PROJECTILE_TYPE
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    ZeldaObject,
    ZeldaSnapshot,
    read_snapshot,
)

LEVEL9 = 9

# $0656 SelectedItemSlot. Local so door_graph L9 exits can import this module.
B_ITEM_BOMBS = 1
B_ITEM_ARROWS = 2

ROOM_BEFORE_GANON = 0x52
ROOM_GANON = 0x42
ROOM_ZELDA = 0x32
NORTH_DOOR = 0x08

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

GANON_FIREBALL_TYPES = frozenset({FIREBALL_OBJECT_TYPE, MANHANDLA_PROJECTILE_TYPE})
DODGE_DIST = CONTACT_MANHATTAN
DODGE_X_LO = 56
DODGE_X_HI = 200


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


def hazard_dodge_dir(
    snap: ZeldaSnapshot,
    hazards: tuple[ZeldaObject, ...],
    *,
    thr: int = DODGE_DIST,
    x_lo: int = DODGE_X_LO,
    x_hi: int = DODGE_X_HI,
) -> str | None:
    """Step away from the nearest hazard (manhattan). Horizontal; flip at edges."""
    if not hazards:
        return None
    nearest = min(
        hazards,
        key=lambda o: abs(int(o.x) - int(snap.link_x))
        + abs(int(o.y) - int(snap.link_y)),
    )
    dist = abs(int(nearest.x) - int(snap.link_x)) + abs(
        int(nearest.y) - int(snap.link_y)
    )
    if dist > thr:
        return None
    if int(nearest.x) >= int(snap.link_x):
        return "LEFT" if int(snap.link_x) > x_lo else "RIGHT"
    return "RIGHT" if int(snap.link_x) < x_hi else "LEFT"


def ganon_fireballs(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    return tuple(
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and int(obj.type_id) in GANON_FIREBALL_TYPES
    )


# Ganon's blade and contact windows, as (Link - Ganon ObjX/ObjY) ranges,
# measured by pinning both and swinging from a 4 px grid (what-if writes,
# ``scratch/probe_ganon_hitbox.py``, Blue Ring power-on 14 pin). His RAM x/y
# is the corner of a 32 px sprite: the old chase treated it as a 16 px body,
# stood inside the contact window, and took all 7 hits of that fight (15.5
# hearts). Sword beams pass through him, so the blade is the only weapon.
GANON_CONTACT = ((-2, 22), (-14, 18))
GANON_BLADE_WINDOWS = {
    "UP": ((0, 20), (20, 32)),  # below him
    "DOWN": ((0, 20), (-20, -14)),  # above him
    "LEFT": ((24, 36), (-4, 16)),  # right of him
    "RIGHT": ((-16, -4), (-4, 16)),  # left of him
}
# The Silver Arrow, loosed from outside the sprite (one that spawns inside
# him never lands) on a lane narrower than the blade's: arrows at dy -3 flew
# over him 60 times (Blue Ring power-on 10 pin); dy 0, 6 and 16 all landed.
GANON_ARROW_LANES = {
    "UP": ((0, 16), (48, 255)),
    "DOWN": ((0, 16), (-255, -32)),
    "LEFT": ((48, 255), (0, 16)),
    "RIGHT": ((-255, -32), (0, 16)),
}
# A turn press walks Link ~2 px at Ganon: stands keep that off the near edge,
# or the press leaves the window and the walk comes back (a 96<->98 flutter).
GANON_FACE_MARGIN = 4
# Link's turn nodes in room 0x42 (x%8==0, y%8==5); the four corners are wall.
GANON_NODES = tuple(
    (x, y)
    for x in range(32, 209, 8)
    for y in range(85, 190, 8)
    if 48 <= x <= 192 or 101 <= y <= 173
)


def _ganon_rel(xy: tuple[int, int], boss: ZeldaObject) -> tuple[int, int]:
    return int(xy[0]) - int(boss.x), int(xy[1]) - int(boss.y)


def _inside(rel: tuple[int, int], window) -> bool:
    (xlo, xhi), (ylo, yhi) = window
    return xlo <= rel[0] <= xhi and ylo <= rel[1] <= yhi


def _link_xy(snap: ZeldaSnapshot) -> tuple[int, int]:
    return int(snap.link_x), int(snap.link_y)


def ganon_contact(snap: ZeldaSnapshot, boss: ZeldaObject) -> bool:
    return _inside(_ganon_rel(_link_xy(snap), boss), GANON_CONTACT)


def _can_face(snap: ZeldaSnapshot, direction: str) -> bool:
    """Facing ``direction`` costs no slide: already faced, or on its turn line.

    A press across the lattice slides Link onto it first, so an arrow lane
    at y=147 flipped 147<->149 for 638 frames until brown Ganon healed
    (Blue Ring power-on 10 pin).
    """
    if int(snap.facing) == direction_to_facing(direction):
        return True
    if direction in ("UP", "DOWN"):
        return int(snap.link_x) % 8 == 0
    return int(snap.link_y) % 8 == 5


def _window_direction(
    snap: ZeldaSnapshot, boss: ZeldaObject, windows
) -> tuple[str | None, bool]:
    """``(direction, faced)``: fire when already faced inside a window; turn
    only from inside its stand (the window less the turn's walk). One edge
    for both flipped Link across it every frame (113<->114 on a lane)."""
    rel = _ganon_rel(_link_xy(snap), boss)
    for direction, window in windows.items():
        if _inside(rel, window) and int(snap.facing) == direction_to_facing(direction):
            return direction, True
    for direction, window in windows.items():
        if _inside(rel, _stand_window(direction, window)) and _can_face(snap, direction):
            return direction, False
    return None, False


def _stand_window(direction: str, window):
    """``window`` less ``GANON_FACE_MARGIN`` on the edge the turn walks toward."""
    (xlo, xhi), (ylo, yhi) = window
    m = GANON_FACE_MARGIN
    return {
        "UP": ((xlo, xhi), (ylo + m, yhi)),
        "DOWN": ((xlo, xhi), (ylo, yhi - m)),
        "LEFT": ((xlo + m, xhi), (ylo, yhi)),
        "RIGHT": ((xlo, xhi - m), (ylo, yhi)),
    }[direction]


def _nearest_node(snap: ZeldaSnapshot, boss: ZeldaObject, windows) -> tuple[int, int] | None:
    """Nearest turn node inside one of ``windows`` and out of contact."""
    lx, ly = _link_xy(snap)
    stands = [_stand_window(d, w) for d, w in windows.items()]
    best: tuple[int, int] | None = None
    for node in GANON_NODES:
        rel = _ganon_rel(node, boss)
        if _inside(rel, GANON_CONTACT) or not any(_inside(rel, w) for w in stands):
            continue
        if best is None or abs(node[0] - lx) + abs(node[1] - ly) < abs(best[0] - lx) + abs(best[1] - ly):
            best = node
    return best


def _away_from(snap: ZeldaSnapshot, boss: ZeldaObject) -> str:
    """Step out of the contact window the short way, inward at a wall."""
    (cxlo, cxhi), (cylo, cyhi) = GANON_CONTACT
    rx, ry = _ganon_rel(_link_xy(snap), boss)
    lx, ly = _link_xy(snap)
    exits = sorted(
        (
            (rx - cxlo + 1, "LEFT", lx > 32),
            (cxhi - rx + 1, "RIGHT", lx < 208),
            (ry - cylo + 1, "UP", ly > 85),
            (cyhi - ry + 1, "DOWN", ly < 189),
        ),
        key=lambda e: e[0],
    )
    return next((d for _, d, ok in exits if ok), exits[0][1])


def _walk_to(snap: ZeldaSnapshot, goal: tuple[int, int] | None, reason: str, cd: int):
    from zelda_i.dungeon.hop_controller import room_step

    if goal is None:
        return nes_idle_action(), "no_stand", cd
    step = room_step(snap, goal, tol=0)
    if step is None:
        return nes_idle_action(), "hold_stand", cd
    return nes_action(step), reason, cd


def ganon_action(
    snap: ZeldaSnapshot,
    *,
    cooldown: int,
) -> tuple[list[int], str, int]:
    """One Ganon frame from live boss coordinates and measured windows.

    Blue: the blade from a window, else out of contact, else to the nearest
    window node. Brown: the Silver Arrow from a lane node (his heal timer is
    the clock). No fireball dodge: re-picking a sidestep each frame took 39
    hits where standing took 12, and stretched the fight into more fireballs.
    """
    boss = ganon_object(snap)
    next_cd = max(0, cooldown - 1)
    if boss is None:
        return nes_idle_action(), "wait_ganon", next_cd

    if boss.state != 0:
        if int(boss.hp) < GANON_HP_START:
            # Brown resets HP to 240 and the Silver Arrow's hit is the kill
            # (240 -> 176, the brown timer freezes). The defeat flag waits
            # for Link on the remains: arrows into them, or a stand off
            # them, held it unset for 7000 frames.
            return _walk_to(snap, (int(boss.x), int(boss.y)), "to_ganon_remains", next_cd)
        lane, faced = _window_direction(snap, boss, GANON_ARROW_LANES)
        if lane is not None and not faced:
            return nes_action(lane), "face_arrow", next_cd
        if lane is not None:
            if cooldown > 0:
                return nes_idle_action(), "arrow_cooldown", next_cd
            return nes_action("B"), "silver_arrow", 16
        return _walk_to(snap, _nearest_node(snap, boss, GANON_ARROW_LANES), "align_arrow", next_cd)

    blade, faced = _window_direction(snap, boss, GANON_BLADE_WINDOWS)
    if blade is not None and not faced:
        return nes_action(blade), "face_sword", next_cd
    if blade is not None:
        if cooldown > 0:
            return nes_idle_action(), "cooldown_stand", next_cd
        return nes_action("A"), "sword_pulse", 12
    if ganon_contact(snap, boss):
        return nes_action(_away_from(snap, boss)), "leave_ganon_body", next_cd
    return _walk_to(snap, _nearest_node(snap, boss, GANON_BLADE_WINDOWS), "walk_stand", next_cd)


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


def _env_step(env: Any, action: list[int], *, assist: Any, total: list[int]):
    obs, *_ = env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    return obs


def _enter_ganon(env: Any, *, assist: Any, total: list[int]):
    from zelda_i.level9.path import final_patra_to_ganon_step

    obs = None
    leftover: tuple[int, int] | None = None
    for _ in range(900):
        snap = read_snapshot(env.get_ram())
        ram = env.get_ram()
        if leftover is None:
            leftover = (int(snap.link_x), int(snap.link_y))
        if (
            in_ganon_fight(snap)
            and int(ram[ADDR_GANON_SCENE_PHASE]) == GANON_SCENE_FIGHT
        ):
            return obs, True
        frame_action = final_patra_to_ganon_step(snap, leftover=leftover)
        obs = _env_step(env, frame_action.action, assist=assist, total=total)
    return obs, False


def _collect_power_triforce(env: Any, *, assist: Any, total: list[int]):
    obs = None
    for _ in range(1400):
        snap = read_snapshot(env.get_ram())
        if snap.cur_opened_doors & NORTH_DOOR:
            return obs, True
        boss = ganon_object(snap)
        if boss is None:
            action = nes_idle_action()
        elif abs(snap.link_x - boss.x) > 4:
            action = nes_action("RIGHT" if snap.link_x < boss.x else "LEFT")
        elif abs(snap.link_y - boss.y) > 4:
            action = nes_action("DOWN" if snap.link_y < boss.y else "UP")
        else:
            action = nes_idle_action()
        obs = _env_step(env, action, assist=assist, total=total)
    return obs, False


def _enter_zelda(env: Any, *, assist: Any, total: list[int]):
    from zelda_i.level9.path import ZELDA_DOOR_GOAL, leftover_door_step

    obs = None
    leftover: tuple[int, int] | None = None
    for _ in range(1200):
        snap = read_snapshot(env.get_ram())
        if leftover is None:
            leftover = (int(snap.link_x), int(snap.link_y))
        if in_zelda_room(snap):
            return obs, True
        if snap.screen == ROOM_GANON:
            action = leftover_door_step(
                snap, leftover, "UP", ZELDA_DOOR_GOAL, reason="zelda"
            ).action
        else:
            action = nes_action("UP")
        obs = _env_step(env, action, assist=assist, total=total)
    return obs, False


def _rescue_zelda(env: Any, *, assist: Any, total: list[int]):
    obs = None
    for frame in range(3500):
        snap = read_snapshot(env.get_ram())
        if snap.mode == MODE_ENDING:
            return obs, True
        if snap.link_x < 0x70:
            direction = "RIGHT"
        elif snap.link_x > 0x80:
            direction = "LEFT"
        elif snap.link_y > 0x95:
            direction = "UP"
        elif snap.link_y < 0x95:
            direction = "DOWN"
        else:
            direction = "UP"
        action = (
            nes_action(direction, "A") if frame % 12 == 0 else nes_action(direction)
        )
        obs = _env_step(env, action, assist=assist, total=total)
    return obs, False


__all__ = [
    "ADDR_GANON_OBJ_PHASE_BASE",
    "ADDR_GANON_SCENE_PHASE",
    "ADDR_LAST_BOSS_DEFEATED",
    "B_ITEM_ARROWS",
    "B_ITEM_BOMBS",
    "CONTACT_CHEBYSHEV",
    "CONTACT_MANHATTAN",
    "DODGE_DIST",
    "DODGE_X_HI",
    "DODGE_X_LO",
    "ENDING_SUBMODE_CREDITS",
    "ENDING_SUBMODE_FINAL_SCREEN",
    "GANON_BROWN_STATE",
    "GANON_DEFEATED_PHASE",
    "GANON_FIREBALL_TYPES",
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
    "ganon_fireballs",
    "ganon_is_brown",
    "ganon_object",
    "hazard_dodge_dir",
    "in_ganon_fight",
    "in_room_before_ganon",
    "in_zelda_room",
    "zelda_object",
]
