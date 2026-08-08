"""Shared dungeon frame/env helpers for Zelda I runners.

Pure-ish ops used by thin scripts and library path controllers (goto, door
exit, bomb stand, patrol clear). Prefer ``DoorDir`` bits from
``zelda_i.door_graph`` over redefining door masks.

``poke_bombs`` is RECON-only inventory mutation — durable runners must leave it
opt-in (default off) and document any poke in the trial report.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.anchors import TF_BIT_L3 as TRIFORCE_BIT_L3
from zelda_i.dungeon_ids import (
    DARKNUT_OBJECT_TYPE,
    GEL_OBJECT_TYPE as GEL_ALT_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    INVULN_MOVER_OBJECT_TYPE as INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    MANHANDLA_PROJECTILE_TYPE,
    object_name,
    room_item_name,
)
from zelda_i.ram import (
    ADDR_RAFT,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUSH_FRAMES = 110
SETTLE_FRAMES = 70
ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

# Types that must never be combat-clear targets.
NON_COMBAT_TYPES: frozenset[int] = frozenset(
    {
        0x5A,
        0x4F,
        0x4E,
        0x60,
        0x61,
        0x62,
        0x5B,
        0x5C,
        0x49,
        0x55,
        0x40,  # bubble
        INVULN_MOVER_0X2B,  # 0x2b invuln — never "clear"
        MANHANDLA_PROJECTILE_TYPE,  # manhandla projectile
    }
)

_DEFAULT_PATROL: tuple[tuple[int, int], ...] = (
    (64, 109),
    (120, 109),
    (176, 109),
    (176, 141),
    (176, 173),
    (120, 173),
    (64, 173),
    (64, 141),
    (120, 141),
    (100, 125),
    (140, 157),
    (80, 157),
    (160, 125),
)


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def objs(snap: ZeldaSnapshot) -> list[dict[str, Any]]:
    """Serialize live object slots 1–12 for reports."""
    out: list[dict[str, Any]] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12) or o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "x": o.x,
                "y": o.y,
                "hp": o.hp,
                "facing": o.facing,
            }
        )
    return out


def room_fields(snap: ZeldaSnapshot, ram: Any | None = None) -> dict[str, Any]:
    """Room summary for path logs / trial reports."""
    types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    )
    tf = int(read_u8(ram, ADDR_TRIFORCE)) if ram is not None else None
    raft = int(read_u8(ram, ADDR_RAFT)) if ram is not None else None
    return {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "health": snap.health,
        "heart_containers": snap.heart_containers,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "doors": {
            "R": bool(snap.cur_opened_doors & DoorDir.RIGHT),
            "L": bool(snap.cur_opened_doors & DoorDir.LEFT),
            "D": bool(snap.cur_opened_doors & DoorDir.DOWN),
            "U": bool(snap.cur_opened_doors & DoorDir.UP),
            "raw": snap.cur_opened_doors,
        },
        "type_counts": {f"0x{k:02x}": v for k, v in sorted(types.items())},
        "type_names": {f"0x{k:02x}": object_name(k) for k in sorted(types)},
        "objects": objs(snap),
        "raft": raft,
        "triforce": tf,
        "tf04": bool(tf & TRIFORCE_BIT_L3) if tf is not None else None,
    }


def live_killables(snap: ZeldaSnapshot, types: Sequence[int]) -> list:
    """Liveness for room clear.

    - Keese (0x1b): type presence (HP often 0 while alive)
    - Gel (0x14/0x15): type presence until slot frees
    - Zol / Darknut / others: type + HP>0

    Slots **1–12** (not just 1–10): LIVE 0x5d left a gel in slot 11 sealing UP.
    Never includes ``INVULN_MOVER_0X2B`` unless explicitly listed in *types*.
    """
    type_set = set(types)
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12) or o.type_id not in type_set:
            continue
        if o.type_id in (KEESE_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_ALT_OBJECT_TYPE):
            out.append(o)
        elif o.hp > 0:
            out.append(o)
    return out


# ---------------------------------------------------------------------------
# Frame / env helpers
# ---------------------------------------------------------------------------


def idle(env: Any, assist: Any | None, total: list[int], frames: int = 30) -> None:
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def goto(
    env: Any,
    assist: Any | None,
    total: list[int],
    tx: int,
    ty: int,
    *,
    tol: int = 4,
    max_f: int = 500,
) -> bool:
    """Walk Link toward (tx, ty). Returns True if within *tol*."""
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - tx) <= tol and abs(snap.link_y - ty) <= tol:
            return True
        if abs(snap.link_x - tx) > tol:
            act = nes_action("RIGHT" if snap.link_x < tx else "LEFT")
        else:
            act = nes_action("DOWN" if snap.link_y < ty else "UP")
        env.step(act)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    return False


def push_dir(
    env: Any,
    assist: Any | None,
    total: list[int],
    direction: str,
    frames: int = PUSH_FRAMES,
) -> None:
    """Hold a cardinal direction then settle through room scroll."""
    room0 = read_snapshot(env.get_ram()).screen
    mode0 = read_snapshot(env.get_ram()).mode
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode in (PLAY_MODE, 9, 10):
            break
        env.step(nes_action(direction))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(SETTLE_FRAMES):
        snap = read_snapshot(env.get_ram())
        # Any dungeon level (L1–L9); mode-9 is underworld passage settle.
        if snap.mode == PLAY_MODE and snap.level > 0:
            if snap.screen != room0 or mode0 == 9:
                idle(env, assist, total, 30)
                break
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def ensure_bomb(env: Any) -> str:
    """Select bomb on B button via RAM poke."""
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "set_byte"):
            mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            return "selected_item=bomb"
    except Exception:
        pass
    try:
        env.unwrapped.data.set_value("selected_item", B_ITEM_BOMB)
        return "selected_item=bomb"
    except Exception as exc:
        return f"select_fail={exc!r}"


def poke_bombs(env: Any, n: int = 16) -> str:
    """RECON-ONLY inventory poke. Document in report — not Clean.

    Durable runners must keep this opt-in (default off).
    """
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        return f"poke_fail={exc!r}"


def exit_door(
    env: Any,
    assist: Any | None,
    total: list[int],
    direction: str,
    *,
    y_force: int | None = None,
    x_force: int | None = None,
    push: int = PUSH_FRAMES,
) -> dict[str, Any]:
    """Align to door target and push; return before/after room fields."""
    snap0 = read_snapshot(env.get_ram())
    before = room_fields(snap0, env.get_ram())
    tx, ty = DOOR_TARGETS[direction]
    if y_force is not None:
        ty = y_force
    if x_force is not None:
        tx = x_force
    if direction in ("LEFT", "RIGHT"):
        # Align y, then hard-drive to door plane (mid-room sticks cost a trial).
        goto(env, assist, total, snap0.link_x, ty, tol=3, max_f=400)
        ok = goto(env, assist, total, tx, ty, tol=6, max_f=700)
        if not ok:
            snap = read_snapshot(env.get_ram())
            goto(env, assist, total, snap.link_x, ty + 16, tol=4, max_f=200)
            goto(env, assist, total, tx, ty, tol=6, max_f=500)
    else:
        goto(env, assist, total, tx, snap0.link_y, tol=3, max_f=400)
        ok = goto(env, assist, total, tx, ty, tol=6, max_f=700)
        if not ok:
            snap = read_snapshot(env.get_ram())
            goto(env, assist, total, tx + 16, snap.link_y, tol=4, max_f=200)
            goto(env, assist, total, tx, ty, tol=6, max_f=500)
    at = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    push_dir(env, assist, total, direction, frames=push)
    after = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"] or after["mode"] != before["mode"]
    return {
        "direction": direction,
        "before": before,
        "at_door": at,
        "after": after,
        "changed_room": changed,
        "result": "room_change" if changed else "blocked",
    }


def bomb_stand(
    env: Any,
    assist: Any | None,
    total: list[int],
    face: str,
    x: int,
    y: int,
    *,
    push: int = PUSH_FRAMES + 40,
) -> dict[str, Any]:
    """Walk to bomb stand, place B bomb, push through opened wall."""
    snap0 = read_snapshot(env.get_ram())
    before = room_fields(snap0, env.get_ram())
    bombs0 = before["bombs"]
    ensure_bomb(env)
    goto(env, assist, total, x, y, tol=3, max_f=500)
    for _ in range(8):
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    env.step(nes_action(face, "B"))
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    for _ in range(100):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    room0 = before["screen"]
    for _ in range(push):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    idle(env, assist, total, SETTLE_FRAMES)
    after = room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"]
    return {
        "face": face,
        "stand": [x, y],
        "before": before,
        "after": after,
        "bombs_spent": bombs0 - after["bombs"],
        "changed_room": changed,
        "result": "room_change" if changed else "no_open",
    }


def fight_clear(
    env: Any,
    assist: Any | None,
    total: list[int],
    *,
    enemy_types: tuple[int, ...],
    max_frames: int = 8000,
    use_bombs: bool = False,
    require_door_pair: bool = False,
    level: int | None = None,
    patrol: tuple[tuple[int, int], ...] | None = None,
) -> dict[str, Any]:
    """Center-patrol sword/bomb clear. Never treats 0x2b as killable.

    ``require_door_pair`` (0x5c): success only when live=0 AND doors raw R|L
    (value 3). Premature live=0 with raw=1 leaves RIGHT sealed.

    *level* defaults to the current snapshot dungeon level (or 1 if overworld).
    """
    enemy_types = tuple(t for t in enemy_types if t not in NON_COMBAT_TYPES)
    if not enemy_types:
        return {"ok": True, "skipped": True, "frames": 0}

    snap = read_snapshot(env.get_ram())
    room = snap.screen
    if level is None:
        level = int(snap.level) if snap.level > 0 else 1
    patrol_pts = patrol if patrol is not None else _DEFAULT_PATROL
    type_only = all(
        t in (KEESE_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_ALT_OBJECT_TYPE)
        for t in enemy_types
    )
    engage = 40 if DARKNUT_OBJECT_TYPE in enemy_types else 48
    spec = DungeonRoomSpec(
        spec_id=f"dungeon_ops_clear_0x{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=enemy_types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE if type_only else AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=patrol_pts,
            engage_distance=engage,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=level,
    )
    ctl = GenericDungeonRoomController(spec)
    last_progress = 0
    prev_live = -1
    bomb_cd = 0
    zero_streak = 0
    zero_need = 100 if DARKNUT_OBJECT_TYPE in enemy_types else 45
    min_fight = 120 if DARKNUT_OBJECT_TYPE in enemy_types else 40
    if use_bombs:
        ensure_bomb(env)

    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame}
        if snap.screen != room and snap.mode == PLAY_MODE:
            return {
                "ok": True,
                "frames": frame,
                "left_room": True,
                "final": room_fields(snap, env.get_ram()),
            }
        live_objs = live_killables(snap, enemy_types) if enemy_types else []
        if not live_objs and not any(
            t in (KEESE_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_ALT_OBJECT_TYPE)
            for t in enemy_types
        ):
            live_objs = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 12
                and o.type_id in enemy_types
                and (o.hp > 0 or type_only)
            ]
        live = live_objs
        n = len(live)
        if prev_live < 0:
            prev_live = n
            last_progress = frame
        elif n < prev_live:
            prev_live = n
            last_progress = frame

        raw = snap.cur_opened_doors
        door_pair = (raw & (DoorDir.RIGHT | DoorDir.LEFT)) == (
            DoorDir.RIGHT | DoorDir.LEFT
        )
        doors_ok = (not require_door_pair) or (
            door_pair and snap.room_all_dead >= 8
        )

        if n == 0 and frame > min_fight:
            zero_streak += 1
            if zero_streak >= zero_need and doors_ok:
                idle(env, assist, total, 40)
                s2 = read_snapshot(env.get_ram())
                live2 = live_killables(s2, enemy_types)
                raw2 = s2.cur_opened_doors
                pair2 = (raw2 & (DoorDir.RIGHT | DoorDir.LEFT)) == (
                    DoorDir.RIGHT | DoorDir.LEFT
                )
                doors2 = (not require_door_pair) or (
                    pair2 and s2.room_all_dead >= 8
                )
                if live2:
                    zero_streak = 0
                    prev_live = len(live2)
                    last_progress = frame
                    continue
                if not doors2:
                    continue
                return {
                    "ok": True,
                    "frames": frame,
                    "final": room_fields(s2, env.get_ram()),
                }
            if (
                zero_streak >= zero_need
                and require_door_pair
                and not doors_ok
            ):
                env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                continue
        else:
            zero_streak = 0

        if use_bombs and live and bomb_cd <= 0 and snap.bombs > 0:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            if dist < 40:
                dx = nearest.x - snap.link_x
                dy = nearest.y - snap.link_y
                if abs(dx) >= abs(dy):
                    face = "RIGHT" if dx > 0 else "LEFT"
                else:
                    face = "DOWN" if dy > 0 else "UP"
                ensure_bomb(env)
                if dist > 18:
                    env.step(nes_action(face))
                else:
                    env.step(nes_action(face, "B"))
                    bomb_cd = 80
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                continue
        if bomb_cd > 0:
            bomb_cd -= 1

        act = ctl.step(snap)
        env.step(act.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        if frame - last_progress > 900 and n > 0:
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[frame % 4]))
            total[0] += 1

    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "final": room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


__all__ = [
    "ADDR_SELECTED_ITEM",
    "B_ITEM_BOMB",
    "DARKNUT_OBJECT_TYPE",
    "DOOR_TARGETS",
    "GEL_ALT_OBJECT_TYPE",
    "GEL_SPLIT_OBJECT_TYPE",
    "INVULN_MOVER_0X2B",
    "KEESE_OBJECT_TYPE",
    "MANHANDLA_PROJECTILE_TYPE",
    "NON_COMBAT_TYPES",
    "PUSH_FRAMES",
    "SETTLE_FRAMES",
    "TRIFORCE_BIT_L3",
    "bomb_stand",
    "ensure_bomb",
    "exit_door",
    "fight_clear",
    "goto",
    "idle",
    "live_killables",
    "objs",
    "poke_bombs",
    "push_dir",
    "room_fields",
]
