"""Assisted: Level3Raft → Manhandla 0x4d → TF bit 0x04 (rr-vpl residual).

Directed path (LIVE 2026-08-07)::

    0x0f mode9 reverse channel + NW stairs UP → 0x69
    UP → 0x59
    BOMB_RIGHT@(192,141) → 0x5a   *** walk-RIGHT sealed post-Raft ***
    RIGHT → 0x5b
    BOMB_RIGHT@(192,141) → 0x5c (3× Darknut)
    full clear (doors raw=3) → RIGHT @ y≈141 → 0x5d
    clear Zol+Keese only (ignore invuln 0x2b) → UP → 0x4d Manhandla 0x3c
    bombs → HC → TF room (bit 0x04)

Intervention: Survival (``--infinite-life``). Not Clean STATUS.
Does **not** rewrite ``run_level3_raft.py``.

Examples::

    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --to-boss --trials 2
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --kill --poke-bombs 16
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --phase gate5d --tag l3_gate
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import (
    BOMB_STAND_59_RIGHT,
    BOMB_STAND_5B_RIGHT,
    DARKNUT_OBJECT_TYPE,
    DOOR_5C_RIGHT_Y,
    INVULN_MOVER_0X2B,
    KEESE_OBJECT_TYPE,
    LEVEL3_TRIFORCE_BIT,
    MANHANDLA_OBJECT_TYPE,
    PASSAGE_EXIT_WAYPOINTS,
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_WEST_DARKNUTS,
    ZOL_OBJECT_TYPE,
    level3_has_raft,
    level3_reached_boss,
    level3_reached_boss_prep,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# --- Door / combat anchors ---
DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08
DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}
PUSH_FRAMES = 110
SETTLE_FRAMES = 70
ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

# 0x5d killables only — 0x2b invuln must NOT be in clear set.
# Wooden sword splits Zol (0x13) → Gel (0x14); include gels. Keese often HP=0.
GEL_OBJECT_TYPE = 0x14
PREP_CLEAR_TYPES = (ZOL_OBJECT_TYPE, GEL_OBJECT_TYPE, 0x15, KEESE_OBJECT_TYPE)
# LIVE: after only 0x2b remain, doors raw=10 (U|L) and walk-UP → 0x4d.

# North-door UP approaches on 0x5d (order matters; x≈120 is primary).
UP_APPROACHES: tuple[tuple[int, int], ...] = (
    (120, 93),
    (120, 101),
    (112, 93),
    (128, 93),
    (100, 93),
    (140, 93),
    (120, 109),
    (96, 101),
    (144, 101),
)

# Bomb north stands if walk-UP fails after clear.
BOMB_NORTH_STANDS: tuple[tuple[int, int], ...] = (
    (120, 101),
    (96, 101),
    (144, 101),
    (120, 109),
    (112, 101),
    (128, 101),
)

_NON_COMBAT = {
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
    0x56,  # manhandla projectile
}


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out: list[dict] = []
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


def _room_fields(snap: ZeldaSnapshot, ram=None) -> dict:
    types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    )
    tf = int(read_u8(ram, ADDR_TRIFORCE)) if ram is not None else None
    raft = None
    if ram is not None:
        from zelda_i.ram import ADDR_RAFT

        raft = int(read_u8(ram, ADDR_RAFT))
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
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "type_counts": {f"0x{k:02x}": v for k, v in sorted(types.items())},
        "type_names": {f"0x{k:02x}": object_name(k) for k in sorted(types)},
        "objects": _objs(snap),
        "raft": raft,
        "triforce": tf,
        "tf04": bool(tf & LEVEL3_TRIFORCE_BIT) if tf is not None else None,
    }


def _idle(env, assist, total: list[int], frames: int = 30) -> None:
    for _ in range(frames):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def _goto(
    env,
    assist,
    total: list[int],
    tx: int,
    ty: int,
    *,
    tol: int = 4,
    max_f: int = 500,
) -> bool:
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


def _push_dir(
    env, assist, total: list[int], direction: str, frames: int = PUSH_FRAMES
) -> None:
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
        if snap.mode == PLAY_MODE and snap.level == LEVEL3:
            if snap.screen != room0 or mode0 == 9:
                _idle(env, assist, total, 30)
                break
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])


def _ensure_bomb(env) -> str:
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


def _poke_bombs(env, n: int = 16) -> str:
    """RECON-ONLY inventory poke. Document in report — not Clean."""
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        return f"poke_fail={exc!r}"


def _exit_door(
    env,
    assist,
    total: list[int],
    direction: str,
    *,
    y_force: int | None = None,
    x_force: int | None = None,
    push: int = PUSH_FRAMES,
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    tx, ty = DOOR_TARGETS[direction]
    if y_force is not None:
        ty = y_force
    if x_force is not None:
        tx = x_force
    if direction in ("LEFT", "RIGHT"):
        # Align y, then hard-drive to door plane (mid-room sticks cost a trial).
        _goto(env, assist, total, snap0.link_x, ty, tol=3, max_f=400)
        ok = _goto(env, assist, total, tx, ty, tol=6, max_f=700)
        if not ok:
            # Unstick: step vertically then re-approach
            snap = read_snapshot(env.get_ram())
            _goto(env, assist, total, snap.link_x, ty + 16, tol=4, max_f=200)
            _goto(env, assist, total, tx, ty, tol=6, max_f=500)
    else:
        _goto(env, assist, total, tx, snap0.link_y, tol=3, max_f=400)
        ok = _goto(env, assist, total, tx, ty, tol=6, max_f=700)
        if not ok:
            snap = read_snapshot(env.get_ram())
            _goto(env, assist, total, tx + 16, snap.link_y, tol=4, max_f=200)
            _goto(env, assist, total, tx, ty, tol=6, max_f=500)
    at = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    _push_dir(env, assist, total, direction, frames=push)
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"] or after["mode"] != before["mode"]
    return {
        "direction": direction,
        "before": before,
        "at_door": at,
        "after": after,
        "changed_room": changed,
        "result": "room_change" if changed else "blocked",
    }


def _bomb_stand(
    env,
    assist,
    total: list[int],
    face: str,
    x: int,
    y: int,
    *,
    push: int = PUSH_FRAMES + 40,
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    bombs0 = before["bombs"]
    _ensure_bomb(env)
    _goto(env, assist, total, x, y, tol=3, max_f=500)
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
    _idle(env, assist, total, SETTLE_FRAMES)
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
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


def exit_raft_passage(env, assist, total: list[int]) -> dict:
    """Leave mode-9 0x0f via reverse channel + NW stairs UP → 0x69 play."""
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    if not (
        snap0.mode == 9
        and snap0.screen == ROOM_L3_RAFT_PASSAGE
        and snap0.level == LEVEL3
    ):
        # Already out?
        if (
            snap0.mode == PLAY_MODE
            and snap0.level == LEVEL3
            and snap0.screen == ROOM_L3_SOUTH_DARKNUTS
        ):
            return {"ok": True, "skipped": True, "after": before}
        return {
            "ok": False,
            "error": (
                f"expected mode9 0x0f; got mode={snap0.mode} "
                f"sc=0x{snap0.screen:02x}"
            ),
            "before": before,
        }

    wp_log: list[dict] = []
    for tx, ty in PASSAGE_EXIT_WAYPOINTS:
        ok = _goto(env, assist, total, tx, ty, tol=3, max_f=600)
        s = read_snapshot(env.get_ram())
        wp_log.append(
            {
                "target": [tx, ty],
                "ok": ok,
                "mode": s.mode,
                "sc": f"0x{s.screen:02x}",
                "xy": [s.link_x, s.link_y],
            }
        )
        if s.mode != 9 or s.screen != ROOM_L3_RAFT_PASSAGE:
            break

    for i in range(220):
        s = read_snapshot(env.get_ram())
        if s.mode != 9 or s.screen != ROOM_L3_RAFT_PASSAGE:
            break
        env.step(nes_action("UP"))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    else:
        i = 220

    _idle(env, assist, total, 120)
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    ok = (
        after["mode"] == PLAY_MODE
        and after["level"] == LEVEL3
        and after["screen"] == ROOM_L3_SOUTH_DARKNUTS
    )
    return {
        "ok": ok,
        "before": before,
        "after": after,
        "waypoints": wp_log,
        "stairs_push_frames": i + 1,
    }


def _fight_clear(
    env,
    assist,
    total: list[int],
    *,
    enemy_types: tuple[int, ...],
    max_frames: int = 8000,
    use_bombs: bool = False,
    require_door_pair: bool = False,
) -> dict:
    """Center-patrol sword/bomb clear. Never includes 0x2b.

    ``require_door_pair`` (0x5c): success only when live=0 AND doors raw R|L
    (value 3). Premature live=0 with raw=1 leaves RIGHT sealed.
    """
    enemy_types = tuple(t for t in enemy_types if t not in _NON_COMBAT)
    if not enemy_types:
        return {"ok": True, "skipped": True, "frames": 0}

    snap = read_snapshot(env.get_ram())
    room = snap.screen
    patrol = (
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
    type_only = all(t in (KEESE_OBJECT_TYPE, 0x15) for t in enemy_types)
    engage = 40 if DARKNUT_OBJECT_TYPE in enemy_types else 48
    spec = DungeonRoomSpec(
        spec_id=f"l3_boss_clear_0x{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=enemy_types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE if type_only else AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=engage,
            attack_phase=2,
            patrol_attack_period=6,
            patrol_attack_hold=3,
            engage_attack_period=5,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=LEVEL3,
    )
    ctl = GenericDungeonRoomController(spec)
    last_progress = 0
    prev_live = -1
    bomb_cd = 0
    zero_streak = 0
    zero_need = 100 if DARKNUT_OBJECT_TYPE in enemy_types else 45
    min_fight = 120 if DARKNUT_OBJECT_TYPE in enemy_types else 40
    if use_bombs:
        _ensure_bomb(env)

    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame}
        if snap.screen != room and snap.mode == PLAY_MODE:
            return {
                "ok": True,
                "frames": frame,
                "left_room": True,
                "final": _room_fields(snap, env.get_ram()),
            }
        # Prefer shared liveness (keese/gel type; slots 1–12)
        live_objs = _live_killables(snap, enemy_types) if enemy_types else []
        if not live_objs and not any(
            t in (KEESE_OBJECT_TYPE, GEL_OBJECT_TYPE, 0x15) for t in enemy_types
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
        door_pair = (raw & (DOOR_RIGHT | DOOR_LEFT)) == (DOOR_RIGHT | DOOR_LEFT)
        doors_ok = (not require_door_pair) or (
            door_pair and snap.room_all_dead >= 8
        )

        if n == 0 and frame > min_fight:
            zero_streak += 1
            if zero_streak >= zero_need and doors_ok:
                _idle(env, assist, total, 40)
                s2 = read_snapshot(env.get_ram())
                live2 = _live_killables(s2, enemy_types)
                raw2 = s2.cur_opened_doors
                pair2 = (raw2 & (DOOR_RIGHT | DOOR_LEFT)) == (
                    DOOR_RIGHT | DOOR_LEFT
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
                    # Keep waiting for shutter pair
                    continue
                return {
                    "ok": True,
                    "frames": frame,
                    "final": _room_fields(s2, env.get_ram()),
                }
            if (
                zero_streak >= zero_need
                and require_door_pair
                and not doors_ok
            ):
                # live=0 but doors not open yet — idle (don't thrash)
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
                _ensure_bomb(env)
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
        "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


def _live_killables(snap: ZeldaSnapshot, types: tuple[int, ...]) -> list:
    """Liveness for clear.

    - Keese (0x1b): type presence (HP often 0 while alive)
    - Gel (0x14/0x15): type presence until slot frees (HP0 residual blocks 0x5d UP)
    - Zol / Darknut / others: type + HP>0

    Slots **1–12** (not just 1–10): LIVE 0x5d left a gel in slot 11 sealing UP.
    """
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12) or o.type_id not in types:
            continue
        if o.type_id in (KEESE_OBJECT_TYPE, GEL_OBJECT_TYPE, 0x15):
            out.append(o)
        elif o.hp > 0:
            out.append(o)
    return out


def _prep_5d_still_killable(snap: ZeldaSnapshot) -> list:
    """Enemies that must die before 0x5d UP shutter (ignore 0x2b)."""
    return _live_killables(snap, PREP_CLEAR_TYPES)


def _fight_manhandla(
    env,
    assist,
    total: list[int],
    *,
    max_frames: int = 16000,
    poke_bombs: int | None = 16,
) -> dict:
    """Circle + bomb near Manhandla heads (type 0x3c)."""
    log: list[dict] = []
    notes: list[str] = []
    if poke_bombs is not None:
        notes.append(f"RECON poke {_poke_bombs(env, poke_bombs)}")
    _ensure_bomb(env)
    bomb_cd = 0
    last_hps: list[int] | None = None
    dmg_events = 0
    enemy_type = MANHANDLA_OBJECT_TYPE

    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        ram = env.get_ram()
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        if tf & LEVEL3_TRIFORCE_BIT:
            log.append({"event": "tf04", "frame": frame, **_room_fields(snap, ram)})
            return {
                "ok": True,
                "tf04": True,
                "frames": frame,
                "dmg_events": dmg_events,
                "log": log[-30:],
                "notes": notes,
                "final": _room_fields(snap, ram),
            }
        if snap.mode == 17:
            return {
                "ok": False,
                "error": "death",
                "frames": frame,
                "notes": notes,
                "dmg_events": dmg_events,
            }
        if snap.mode != PLAY_MODE:
            env.step(nes_idle_action())
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            continue

        heads = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 10 and o.type_id == enemy_type and o.hp > 0
        ]
        heads_any = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 10 and o.type_id == enemy_type
        ]
        hps = [o.hp for o in heads]
        if last_hps is not None and hps and last_hps and sum(hps) < sum(last_hps):
            dmg_events += 1
            log.append(
                {
                    "event": "hp_drop",
                    "frame": frame,
                    "hps": hps,
                    "prev": last_hps,
                    "bombs": snap.bombs,
                }
            )
        last_hps = hps

        if not heads_any and snap.room_all_dead >= 12 and frame > 80:
            log.append(
                {"event": "boss_dead", "frame": frame, **_room_fields(snap, ram)}
            )
            hc0 = snap.heart_containers
            # 1) HC LIVE at ~(128,133) mid-room (item 0x1A)
            for tx, ty in (
                (128, 133),
                (120, 141),
                (112, 133),
                (136, 141),
                (120, 125),
                (104, 141),
                (144, 133),
            ):
                _goto(env, assist, total, tx, ty, tol=4, max_f=300)
                s2 = read_snapshot(env.get_ram())
                if s2.heart_containers > hc0:
                    notes.append(f"HC at ({tx},{ty}) → hc={s2.heart_containers}")
                    break
            else:
                # dense fallback
                for y in range(109, 173, 12):
                    for x in range(80, 161, 12):
                        _goto(env, assist, total, x, y, tol=3, max_f=120)
                        if read_snapshot(env.get_ram()).heart_containers > hc0:
                            notes.append(f"HC dense ({x},{y})")
                            break
                    else:
                        continue
                    break
            log.append(
                {
                    "event": "post_hc",
                    **_room_fields(
                        read_snapshot(env.get_ram()), env.get_ram()
                    ),
                }
            )
            # 2) UP → TF room 0x3d (item 0x1B). LIVE touch ~(124,93) → mode18 + bit 0x04.
            # Do not thrash other exits first — restore would drop HC room progress.
            pr = _exit_door(
                env, assist, total, "UP", x_force=120, y_force=93, push=PUSH_FRAMES + 80
            )
            log.append(
                {
                    "event": "post_exit",
                    "dir": "UP",
                    "result": pr["result"],
                    "to": pr["after"]["sc"] if pr["changed_room"] else None,
                    "item": pr["after"].get("room_item_id"),
                }
            )
            if pr["changed_room"] and pr["after"]["screen"] == 0x3D:
                notes.append("entered TF room 0x3d")
                # Waypoints: south spawn → mid → north TF stand
                for tx, ty in (
                    (120, 173),
                    (120, 141),
                    (124, 109),
                    (124, 93),
                    (120, 93),
                    (112, 93),
                    (136, 93),
                    (120, 125),
                    (128, 149),
                    (120, 149),
                ):
                    _goto(env, assist, total, tx, ty, tol=3, max_f=400)
                    if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT:
                        notes.append(f"TF 0x04 at ({tx},{ty})")
                        break
                # Dense north band if still missing
                if not (
                    int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT
                ):
                    for y in range(93, 165, 8):
                        for x in range(96, 145, 8):
                            _goto(env, assist, total, x, y, tol=2, max_f=80)
                            if (
                                int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                                & LEVEL3_TRIFORCE_BIT
                            ):
                                notes.append(f"TF dense ({x},{y})")
                                break
                        else:
                            continue
                        break
                # Fanfare settle (mode 18)
                for _ in range(200):
                    if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT:
                        s3 = read_snapshot(env.get_ram())
                        if s3.mode == PLAY_MODE or s3.mode == 18:
                            env.step(nes_idle_action())
                            total[0] += 1
                            if assist is not None:
                                assist.apply_env(env, frame=total[0])
                            if s3.mode == PLAY_MODE and _ > 30:
                                break
                            continue
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
            elif not pr["changed_room"]:
                # Fallback other exits
                st_post = env.em.get_state()
                for direction in ("RIGHT", "LEFT", "DOWN"):
                    env.em.set_state(st_post)
                    _idle(env, assist, total, 2)
                    pr2 = _exit_door(
                        env, assist, total, direction, push=PUSH_FRAMES + 40
                    )
                    log.append(
                        {
                            "event": "post_exit",
                            "dir": direction,
                            "result": pr2["result"],
                            "to": (
                                pr2["after"]["sc"] if pr2["changed_room"] else None
                            ),
                        }
                    )
            final = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
            return {
                "ok": True,
                "tf04": bool(final.get("tf04")),
                "frames": frame,
                "dmg_events": dmg_events,
                "log": log[-40:],
                "notes": notes,
                "final": final,
            }

        if assist is not None and snap.bombs < 2 and poke_bombs:
            notes.append(f"topup {_poke_bombs(env, poke_bombs)}")
            _ensure_bomb(env)

        if not heads:
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[frame // 15 % 4], "A"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            continue

        nearest = min(
            heads, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y)
        )
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) >= abs(dy):
            face = "RIGHT" if dx > 0 else "LEFT"
            approach = face
            circle = "DOWN" if (frame // 30) % 2 == 0 else "UP"
        else:
            face = "DOWN" if dy > 0 else "UP"
            approach = face
            circle = "RIGHT" if (frame // 30) % 2 == 0 else "LEFT"

        if bomb_cd > 0:
            bomb_cd -= 1

        if dist < 42 and bomb_cd <= 0 and snap.bombs > 0:
            _ensure_bomb(env)
            if dist > 16:
                env.step(nes_action(approach))
            else:
                env.step(nes_action(face, "B"))
                bomb_cd = 65
                log.append(
                    {
                        "event": "bomb_place",
                        "frame": frame,
                        "at": [snap.link_x, snap.link_y],
                        "target": [nearest.x, nearest.y, nearest.hp],
                        "bombs": snap.bombs,
                    }
                )
        elif dist > 48:
            if frame % 4 == 0:
                env.step(nes_action(approach, "A"))
            else:
                env.step(nes_action(approach))
        else:
            d = circle if frame % 3 else approach
            if frame % 3 == 0:
                env.step(nes_action(d, "A"))
            else:
                env.step(nes_action(d))

        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

        if frame % 200 == 0:
            log.append(
                {
                    "event": "tick",
                    "frame": frame,
                    "heads": len(heads),
                    "hps": hps,
                    "xy": [snap.link_x, snap.link_y],
                    "bombs": snap.bombs,
                    "sc": f"0x{snap.screen:02x}",
                }
            )

    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "dmg_events": dmg_events,
        "log": log[-30:],
        "notes": notes,
        "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


def _clear_5d_prep(
    env,
    assist,
    total: list[int],
    *,
    max_frames: int = 14000,
    poke_bombs: int | None = None,
) -> dict:
    """Clear Zol/Gel/Keese on 0x5d until only invuln 0x2b remain.

    LIVE: full killable clear → doors raw=10 (U|L); residual gel HP0 blocks UP.
    """
    from zelda_i.dungeon import (
        AliveRule,
        CombatTuning,
        DoorRoute,
        DungeonRoomSpec,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )

    if poke_bombs is not None:
        _poke_bombs(env, poke_bombs)
        _ensure_bomb(env)

    patrol = (
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
    spec = DungeonRoomSpec(
        spec_id="l3_5d_prep",
        source_room=ROOM_L3_BOSS_PREP,
        room_id=ROOM_L3_BOSS_PREP,
        entry=DoorRoute("LEFT", ((32, 141),)),
        enemy_types=PREP_CLEAR_TYPES,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE,  # keese/gel type liveness
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=48,
            attack_phase=2,
            patrol_attack_period=5,
            patrol_attack_hold=3,
            engage_attack_period=4,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        max_frames=max_frames,
        level=LEVEL3,
    )
    ctl = GenericDungeonRoomController(spec)
    zero_streak = 0
    bomb_cd = 0
    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame}
        if snap.screen != ROOM_L3_BOSS_PREP:
            return {
                "ok": True,
                "left_room": True,
                "frames": frame,
                "final": _room_fields(snap, env.get_ram()),
            }
        live = _prep_5d_still_killable(snap)
        if not live:
            zero_streak += 1
            # LIVE: after only 0x2b remain, wait for all_dead ramp + UP bit
            # (raw=10). Returning at zero_streak=200 with doors=0 fails UP.
            doors_up = bool(snap.cur_opened_doors & DOOR_UP)
            all_dead_ok = snap.room_all_dead >= 20
            if zero_streak >= 80 and doors_up and all_dead_ok:
                _idle(env, assist, total, 50)
                s2 = read_snapshot(env.get_ram())
                if _prep_5d_still_killable(s2):
                    zero_streak = 0
                    continue
                return {
                    "ok": True,
                    "frames": frame,
                    "final": _room_fields(s2, env.get_ram()),
                }
            # Keep idling (and light roam) so kill-shutter can open
            if zero_streak > 60 and zero_streak % 40 < 8:
                env.step(
                    nes_action(("LEFT", "UP", "RIGHT", "DOWN")[zero_streak // 40 % 4])
                )
            else:
                env.step(nes_idle_action())
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            # Soft success after long wait if only 0x2b (doors lag residual)
            if zero_streak >= 600:
                s2 = read_snapshot(env.get_ram())
                return {
                    "ok": True,
                    "frames": frame,
                    "soft": True,
                    "final": _room_fields(s2, env.get_ram()),
                }
            continue
        zero_streak = 0

        if bomb_cd > 0:
            bomb_cd -= 1
        if live and bomb_cd <= 0 and snap.bombs > 0 and frame % 70 == 0:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            if dist < 36:
                _ensure_bomb(env)
                face = (
                    "RIGHT"
                    if nearest.x >= snap.link_x
                    else "LEFT"
                )
                env.step(nes_action(face, "B"))
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                bomb_cd = 85
                continue

        act = ctl.step(snap)
        env.step(act.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


def _open_5d_up(
    env,
    assist,
    total: list[int],
    *,
    tag: str,
    poke_bombs: int | None,
) -> dict:
    """Stabilize 0x5d → 0x4d: full killable clear → doors U|L → walk UP."""
    report: dict = {
        "ok": False,
        "attempts": [],
        "clear": None,
        "pre": None,
        "post": None,
    }
    snap = read_snapshot(env.get_ram())
    if snap.screen != ROOM_L3_BOSS_PREP:
        report["error"] = f"expected 0x5d; got 0x{snap.screen:02x}"
        return report

    # Spawn settle
    _idle(env, assist, total, 80)
    report["pre"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())

    # Clear until only 0x2b (LIVE: residual gel blocks UP shutter)
    clr = _clear_5d_prep(
        env, assist, total, max_frames=14000, poke_bombs=poke_bombs
    )
    report["clear"] = {
        "ok": clr.get("ok"),
        "frames": clr.get("frames"),
        "error": clr.get("error"),
        "final_doors": (clr.get("final") or {}).get("doors"),
        "final_types": (clr.get("final") or {}).get("type_counts"),
        "room_all_dead": (clr.get("final") or {}).get("room_all_dead"),
    }

    # Wait for UP bit (raw often 10 = U|L after true clear)
    for wait_i in range(20):
        s = read_snapshot(env.get_ram())
        fields = _room_fields(s, env.get_ram())
        killable = _prep_5d_still_killable(s)
        report["attempts"].append(
            {
                "kind": "settle_wait",
                "i": wait_i,
                "doors": fields["doors"],
                "mask": fields["open_doorway_mask"],
                "all_dead": fields["room_all_dead"],
                "types": fields["type_counts"],
                "killable": len(killable),
            }
        )
        if killable:
            # Resume clear briefly
            _clear_5d_prep(
                env, assist, total, max_frames=4000, poke_bombs=poke_bombs
            )
            continue
        if s.cur_opened_doors & DOOR_UP:
            break
        _idle(env, assist, total, 30)

    st_base = env.em.get_state()

    # LIVE path: side approach then UP (direct mid-room y-align often sticks)
    side_paths: tuple[tuple[tuple[int, int], ...], ...] = (
        ((160, 141), (160, 109), (120, 109), (120, 93)),
        ((80, 141), (80, 109), (120, 109), (120, 93)),
        ((120, 141), (120, 109), (120, 93)),
        ((120, 141),),  # center UP once doors open (diag: works)
        ((120, 125),),
        ((120, 157),),
    )
    for path in side_paths:
        env.em.set_state(st_base)
        _idle(env, assist, total, 2)
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        ok_path = True
        for tx, ty in path:
            if not _goto(env, assist, total, tx, ty, tol=4, max_f=500):
                ok_path = False
                break
        s = read_snapshot(env.get_ram())
        report["attempts"].append(
            {
                "kind": "side_path",
                "path": [list(p) for p in path],
                "ok_path": ok_path,
                "at": [s.link_x, s.link_y],
                "doors": _room_fields(s, env.get_ram())["doors"],
            }
        )
        _push_dir(env, assist, total, "UP", frames=PUSH_FRAMES + 80)
        after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["attempts"][-1]["result"] = (
            "room_change" if after["screen"] == ROOM_L3_BOSS else "blocked"
        )
        report["attempts"][-1]["to"] = after["sc"]
        if after["screen"] == ROOM_L3_BOSS:
            report["ok"] = True
            report["method"] = f"side_path_up@{path[0]}"
            report["post"] = after
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_0x4d.png")
            return report

    # Walk-UP multi-align fallback
    for ax, ay in UP_APPROACHES:
        env.em.set_state(st_base)
        _idle(env, assist, total, 2)
        pr = _exit_door(
            env,
            assist,
            total,
            "UP",
            x_force=ax,
            y_force=ay,
            push=PUSH_FRAMES + 80,
        )
        report["attempts"].append(
            {
                "kind": "walk_up",
                "xy": [ax, ay],
                "result": pr["result"],
                "to": pr["after"]["sc"] if pr["changed_room"] else None,
                "at_doors": pr["at_door"]["doors"],
            }
        )
        if pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_BOSS:
            report["ok"] = True
            report["method"] = f"walk_up@({ax},{ay})"
            report["post"] = pr["after"]
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_0x4d.png")
            return report

    # Bomb-north residual (usually sealed — document)
    if poke_bombs is not None:
        _poke_bombs(env, poke_bombs)
        _ensure_bomb(env)
    for bx, by in BOMB_NORTH_STANDS[:3]:
        env.em.set_state(st_base)
        _idle(env, assist, total, 2)
        br = _bomb_stand(env, assist, total, "UP", bx, by)
        report["attempts"].append(
            {
                "kind": "bomb_north",
                "stand": [bx, by],
                "result": br["result"],
                "to": br["after"]["sc"] if br["changed_room"] else None,
            }
        )
        if br["changed_room"] and br["after"]["screen"] == ROOM_L3_BOSS:
            report["ok"] = True
            report["method"] = f"bomb_north@({bx},{by})"
            report["post"] = br["after"]
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_0x4d.png")
            return report

    report["post"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_finalroom_0x5d.png")
    return report


def path_to_5d(
    env,
    assist,
    total: list[int],
    *,
    tag: str,
    poke_bombs: int | None,
) -> dict:
    """Directed: Level3Raft / 0x0f → 0x5d boss prep."""
    path_log: list[dict] = []
    traps: list[str] = []
    notes: list[str] = []

    # --- passage exit ---
    ex = exit_raft_passage(env, assist, total)
    path_log.append(
        {
            "step": "passage_exit",
            "ok": ex.get("ok"),
            "to": (ex.get("after") or {}).get("sc"),
            "error": ex.get("error"),
        }
    )
    if not ex.get("ok"):
        return {"ok": False, "error": "passage_exit_failed", "path_log": path_log, "exit": ex}
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_exit_0x69.png")

    if poke_bombs is not None:
        notes.append(f"RECON poke {_poke_bombs(env, poke_bombs)}")
        _ensure_bomb(env)

    # --- 0x69 UP → 0x59 (no full darknut clear; door often open post-raft path) ---
    _idle(env, assist, total, 60)
    # If darknuts still live, light clear optional — walk UP often open after visit
    snap = read_snapshot(env.get_ram())
    if snap.screen == ROOM_L3_SOUTH_DARKNUTS:
        live_dn = _live_killables(snap, (DARKNUT_OBJECT_TYPE,))
        if live_dn and len(live_dn) > 0:
            # Try UP first without clear (doors may already open from prior visit)
            st = env.em.get_state()
            pr = _exit_door(env, assist, total, "UP")
            if not (pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS):
                env.em.set_state(st)
                _idle(env, assist, total, 2)
                clr = _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=(DARKNUT_OBJECT_TYPE,),
                    max_frames=6000,
                )
                path_log.append({"step": "clear_69", "ok": clr.get("ok"), "frames": clr.get("frames")})
                pr = _exit_door(env, assist, total, "UP")
        else:
            pr = _exit_door(env, assist, total, "UP")
        path_log.append(
            {
                "step": "69_up",
                "ok": pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS,
                "to": pr["after"]["sc"] if pr["changed_room"] else None,
            }
        )
        if not (pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_WEST_DARKNUTS):
            return {
                "ok": False,
                "error": "failed_69_up",
                "path_log": path_log,
                "final": pr["after"],
            }

    # --- 0x59 BOMB_RIGHT → 0x5a (walk sealed) ---
    _idle(env, assist, total, 40)
    if poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
        _poke_bombs(env, poke_bombs)
    bx, by = BOMB_STAND_59_RIGHT
    # Try walk first to document sealed, then bomb
    st = env.em.get_state()
    walk = _exit_door(env, assist, total, "RIGHT", y_force=141)
    if walk["changed_room"] and walk["after"]["screen"] == ROOM_L3_COMPASS:
        path_log.append({"step": "59_right_walk", "ok": True, "to": "0x5a"})
    else:
        env.em.set_state(st)
        _idle(env, assist, total, 2)
        if not walk["changed_room"]:
            traps.append("0x59 walk-RIGHT sealed post-Raft (expected)")
        br = _bomb_stand(env, assist, total, "RIGHT", bx, by)
        path_log.append(
            {
                "step": "59_bomb_right",
                "ok": br["changed_room"] and br["after"]["screen"] == ROOM_L3_COMPASS,
                "to": br["after"]["sc"] if br["changed_room"] else None,
                "stand": [bx, by],
            }
        )
        if not (br["changed_room"] and br["after"]["screen"] == ROOM_L3_COMPASS):
            return {
                "ok": False,
                "error": "failed_59_bomb_right",
                "path_log": path_log,
                "final": br["after"],
                "traps": traps,
            }

    # --- 0x5a RIGHT → 0x5b ---
    _idle(env, assist, total, 20)
    pr = _exit_door(env, assist, total, "RIGHT", y_force=141)
    path_log.append(
        {
            "step": "5a_right",
            "ok": pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_DARKNUTS,
            "to": pr["after"]["sc"] if pr["changed_room"] else None,
        }
    )
    if not (pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_DARKNUTS):
        return {
            "ok": False,
            "error": "failed_5a_right",
            "path_log": path_log,
            "final": pr["after"],
            "traps": traps,
        }

    # --- 0x5b BOMB_RIGHT → 0x5c ---
    _idle(env, assist, total, 30)
    if poke_bombs is not None and read_snapshot(env.get_ram()).bombs < 2:
        _poke_bombs(env, poke_bombs)
    bx, by = BOMB_STAND_5B_RIGHT
    st = env.em.get_state()
    walk = _exit_door(env, assist, total, "RIGHT", y_force=141)
    if walk["changed_room"] and walk["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT:
        path_log.append({"step": "5b_right_walk", "ok": True, "to": "0x5c"})
    else:
        env.em.set_state(st)
        _idle(env, assist, total, 2)
        br = _bomb_stand(env, assist, total, "RIGHT", bx, by)
        path_log.append(
            {
                "step": "5b_bomb_right",
                "ok": br["changed_room"]
                and br["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT,
                "to": br["after"]["sc"] if br["changed_room"] else None,
            }
        )
        if not (
            br["changed_room"] and br["after"]["screen"] == ROOM_L3_BOMB_SHORTCUT
        ):
            return {
                "ok": False,
                "error": "failed_5b_bomb_right",
                "path_log": path_log,
                "final": br["after"],
                "traps": traps,
            }

    # --- 0x5c clear Darknuts → RIGHT @ y≈141 → 0x5d ---
    _idle(env, assist, total, 110)  # spawn settle (Darknuts lag ~75–100f)
    snap = read_snapshot(env.get_ram())
    if snap.screen == ROOM_L3_BOMB_SHORTCUT:
        # Wait for darknut spawn before deciding skip
        for _ in range(6):
            live = _live_killables(
                read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
            )
            if live:
                break
            _idle(env, assist, total, 25)
        live = _live_killables(
            read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
        )
        if live:
            if poke_bombs is not None:
                _poke_bombs(env, poke_bombs)
                _ensure_bomb(env)
            clr = _fight_clear(
                env,
                assist,
                total,
                enemy_types=(DARKNUT_OBJECT_TYPE,),
                max_frames=16000,
                use_bombs=True,  # diag: bomb clear → raw=3; sword also raw=3 but slower path
                require_door_pair=True,  # raw==3 before success (raw=1 seals RIGHT)
            )
            path_log.append(
                {
                    "step": "clear_5c",
                    "ok": clr.get("ok"),
                    "frames": clr.get("frames"),
                    "doors": (clr.get("final") or {}).get("doors"),
                    "live_after": len(
                        _live_killables(
                            read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
                        )
                    ),
                }
            )
            # Hard verify — do not trust door bit alone
            still = _live_killables(
                read_snapshot(env.get_ram()), (DARKNUT_OBJECT_TYPE,)
            )
            if still:
                traps.append(
                    f"0x5c clear residual: {len(still)} darknuts still live"
                )
                return {
                    "ok": False,
                    "error": "failed_5c_clear",
                    "path_log": path_log,
                    "final": _room_fields(
                        read_snapshot(env.get_ram()), env.get_ram()
                    ),
                    "traps": traps,
                }
        else:
            path_log.append({"step": "clear_5c", "ok": True, "skipped": True})

        # LIVE: full clear → doors raw=3 (R|L) + room_all_dead ramp.
        # raw=1 alone is a false-clear trap — RIGHT stays sealed.
        doors_ok = False
        for wait_i in range(40):
            s = read_snapshot(env.get_ram())
            live_n = len(_live_killables(s, (DARKNUT_OBJECT_TYPE,)))
            raw = s.cur_opened_doors
            pair = (raw & (DOOR_RIGHT | DOOR_LEFT)) == (DOOR_RIGHT | DOOR_LEFT)
            if live_n == 0 and pair and s.room_all_dead >= 10:
                doors_ok = True
                path_log.append(
                    {
                        "step": "5c_doors_ready",
                        "wait_i": wait_i,
                        "doors": {
                            "R": bool(raw & DOOR_RIGHT),
                            "L": bool(raw & DOOR_LEFT),
                            "raw": raw,
                        },
                        "all_dead": s.room_all_dead,
                    }
                )
                break
            if live_n > 0:
                # Resume bomb clear
                if poke_bombs is not None:
                    _poke_bombs(env, poke_bombs)
                    _ensure_bomb(env)
                _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=(DARKNUT_OBJECT_TYPE,),
                    max_frames=5000,
                    use_bombs=True,
                    require_door_pair=True,
                )
            else:
                _idle(env, assist, total, 30)

        if not doors_ok:
            f = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
            traps.append(
                f"0x5c doors not raw=3 after clear (got raw={f['doors']['raw']} "
                f"all_dead={f['room_all_dead']})"
            )
            # Still attempt RIGHT / bomb-R as last resort below

        # y≈141 only when raw=3 (diag confirmed y=149/133 blocked)
        st_5c = env.em.get_state()
        pr = None
        for ytry in (DOOR_5C_RIGHT_Y, 141):
            env.em.set_state(st_5c)
            _idle(env, assist, total, 2)
            pr = _exit_door(
                env,
                assist,
                total,
                "RIGHT",
                y_force=ytry,
                push=PUSH_FRAMES + 100,
            )
            path_log.append(
                {
                    "step": "5c_right",
                    "y": ytry,
                    "ok": pr["changed_room"]
                    and pr["after"]["screen"] == ROOM_L3_BOSS_PREP,
                    "to": pr["after"]["sc"] if pr["changed_room"] else None,
                    "at_xy": [pr["at_door"]["x"], pr["at_door"]["y"]],
                    "doors": pr["at_door"]["doors"],
                    "mask": pr["at_door"]["open_doorway_mask"],
                    "all_dead": pr["at_door"]["room_all_dead"],
                }
            )
            if pr["changed_room"] and pr["after"]["screen"] == ROOM_L3_BOSS_PREP:
                break
        # Bomb-RIGHT fallback if walk sealed
        if not (
            pr
            and pr["changed_room"]
            and pr["after"]["screen"] == ROOM_L3_BOSS_PREP
        ):
            env.em.set_state(st_5c)
            _idle(env, assist, total, 2)
            if poke_bombs is not None:
                _poke_bombs(env, poke_bombs)
                _ensure_bomb(env)
            br = _bomb_stand(env, assist, total, "RIGHT", 192, 141)
            path_log.append(
                {
                    "step": "5c_bomb_right",
                    "ok": br["changed_room"]
                    and br["after"]["screen"] == ROOM_L3_BOSS_PREP,
                    "to": br["after"]["sc"] if br["changed_room"] else None,
                }
            )
            if br["changed_room"] and br["after"]["screen"] == ROOM_L3_BOSS_PREP:
                pr = {
                    "changed_room": True,
                    "after": br["after"],
                }
        if not (
            pr
            and pr["changed_room"]
            and pr["after"]["screen"] == ROOM_L3_BOSS_PREP
        ):
            # Walk RIGHT can fail even at y=141 with raw=3 if Link sticks mid-band;
            # bomb-RIGHT is LIVE-proven when walk fails.
            traps.append(
                "0x5c walk-RIGHT y≈141 failed after raw=3 clear "
                "(bomb-RIGHT fallback also failed)"
            )
            return {
                "ok": False,
                "error": "failed_5c_right",
                "path_log": path_log,
                "final": (pr or {}).get("after")
                or _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
                "traps": traps,
            }

    # Settle scroll → play in 0x5d
    for _ in range(90):
        snap = read_snapshot(env.get_ram())
        if (
            snap.screen == ROOM_L3_BOSS_PREP
            and snap.mode == PLAY_MODE
            and not snap.transitioning
        ):
            break
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    snap = read_snapshot(env.get_ram())
    ok = snap.screen == ROOM_L3_BOSS_PREP and snap.level == LEVEL3
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_prep_0x5d.png")
    return {
        "ok": ok,
        "path_log": path_log,
        "traps": traps,
        "notes": notes,
        "final": _room_fields(snap, env.get_ram()),
        "mode_at_5d": snap.mode,
    }


def run_once(
    *,
    start_state: str = "Level3Raft",
    infinite_life: bool = True,
    to_boss: bool = True,
    kill: bool = False,
    poke_bombs: int | None = 16,
    save_checkpoint: bool = False,
    tag: str = "l3_to_boss",
    phase: str = "all",
) -> dict:
    """One assisted trial from Level3Raft toward boss / TF."""
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    report: dict = {
        "ok": False,
        "track": track,
        "intervention_class": intervention,
        "start_state": start_state,
        "phase": phase,
        "to_boss": to_boss,
        "kill": kill,
        "tag": tag,
        "reached_5d": False,
        "reached_4d": False,
        "boss_beaten": False,
        "tf04": False,
        "manhandla_confirmed": False,
        "dmg_events": 0,
        "path_log": [],
        "traps": [],
        "notes": [],
    }

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total[0] = 1
        _idle(env, assist, total, 20)
        entry = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["entry"] = entry
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_start.png")

        if not (entry["level"] == LEVEL3 and entry.get("raft")):
            report["error"] = (
                f"expected Level3Raft (raft set); got level={entry['level']} "
                f"raft={entry.get('raft')} sc={entry['sc']}"
            )
            return report

        # Path to 0x5d
        if phase in ("all", "to5d", "gate5d", "boss", "kill"):
            if entry["screen"] == ROOM_L3_BOSS_PREP and entry["mode"] == PLAY_MODE:
                report["path_to_5d"] = {"ok": True, "skipped": True}
                report["reached_5d"] = True
            elif entry["screen"] == ROOM_L3_BOSS and entry["mode"] == PLAY_MODE:
                report["path_to_5d"] = {"ok": True, "skipped": True, "already_boss": True}
                report["reached_5d"] = True
                report["reached_4d"] = True
            else:
                p5 = path_to_5d(
                    env, assist, total, tag=tag, poke_bombs=poke_bombs
                )
                report["path_to_5d"] = {
                    "ok": p5.get("ok"),
                    "path_log": p5.get("path_log"),
                    "error": p5.get("error"),
                    "final": p5.get("final"),
                }
                report["path_log"].extend(p5.get("path_log") or [])
                report["traps"].extend(p5.get("traps") or [])
                report["notes"].extend(p5.get("notes") or [])
                report["reached_5d"] = bool(p5.get("ok"))
                if not p5.get("ok"):
                    report["final"] = p5.get("final")
                    report["total_frames"] = total[0]
                    report["error"] = p5.get("error")
                    out = RECORDINGS_DIR / f"{tag}_report.json"
                    write_json_report(out, report)
                    report["report_path"] = str(out)
                    return report

        if phase == "to5d":
            report["ok"] = report["reached_5d"]
            report["final"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
            report["total_frames"] = total[0]
            out = RECORDINGS_DIR / f"{tag}_report.json"
            write_json_report(out, report)
            report["report_path"] = str(out)
            return report

        # Gate 0x5d → 0x4d
        if (
            to_boss or kill or phase in ("all", "gate5d", "boss", "kill")
        ) and not report["reached_4d"]:
            if read_snapshot(env.get_ram()).screen == ROOM_L3_BOSS:
                report["reached_4d"] = True
            else:
                gate = _open_5d_up(
                    env, assist, total, tag=tag, poke_bombs=poke_bombs
                )
                report["gate_5d"] = {
                    "ok": gate.get("ok"),
                    "method": gate.get("method"),
                    "clear": gate.get("clear"),
                    "attempts_n": len(gate.get("attempts") or []),
                    "attempts_tail": (gate.get("attempts") or [])[-12:],
                    "pre": gate.get("pre"),
                    "post": gate.get("post"),
                    "error": gate.get("error"),
                }
                report["reached_4d"] = bool(gate.get("ok"))
                if gate.get("ok"):
                    report["notes"].append(f"0x4d via {gate.get('method')}")
                else:
                    report["traps"].append(
                        "0x5d UP gate residual — walk/bomb approaches exhausted"
                    )

        if phase == "gate5d":
            report["ok"] = report["reached_4d"]
            report["final"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
            report["total_frames"] = total[0]
            out = RECORDINGS_DIR / f"{tag}_report.json"
            write_json_report(out, report)
            report["report_path"] = str(out)
            return report

        # Confirm Manhandla + optional kill
        if report["reached_4d"] or (
            read_snapshot(env.get_ram()).screen == ROOM_L3_BOSS
        ):
            report["reached_4d"] = True
            _idle(env, assist, total, 40)
            snap = read_snapshot(env.get_ram())
            heads = [
                o
                for o in snap.objects
                if 1 <= o.slot <= 10
                and o.type_id == MANHANDLA_OBJECT_TYPE
                and o.hp > 0
            ]
            report["boss_room"] = _room_fields(snap, env.get_ram())
            report["manhandla_confirmed"] = len(heads) > 0
            if heads:
                report["notes"].append(
                    f"Manhandla type 0x3c: {len(heads)} live heads "
                    f"hps={[o.hp for o in heads]}"
                )
            if save_checkpoint and report["reached_4d"]:
                path = save_state(env, GAME_DIR, GAME, "Level3Boss")
                report["saved_boss"] = str(path)
                write_state_provenance(
                    path,
                    source_state_path=(
                        GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / f"{start_state}.state"
                    ),
                    request={
                        "segment": "level3_to_boss",
                        "natural_entry": False,
                        "start_state": start_state,
                        "intervention_class": intervention,
                    },
                    selected_trial={"reached_4d": True},
                    natural_entry=False,
                )

            if kill or phase in ("all", "boss", "kill"):
                fight = _fight_manhandla(
                    env,
                    assist,
                    total,
                    max_frames=16000,
                    poke_bombs=poke_bombs,
                )
                report["fight"] = {
                    "ok": fight.get("ok"),
                    "tf04": fight.get("tf04"),
                    "frames": fight.get("frames"),
                    "dmg_events": fight.get("dmg_events"),
                    "error": fight.get("error"),
                    "notes": fight.get("notes"),
                    "log_tail": (fight.get("log") or [])[-15:],
                    "final": fight.get("final"),
                }
                report["dmg_events"] = int(fight.get("dmg_events") or 0)
                report["boss_beaten"] = bool(fight.get("ok") and not fight.get("error"))
                report["tf04"] = bool(fight.get("tf04"))
                if report["tf04"] and save_checkpoint:
                    path = save_state(env, GAME_DIR, GAME, "Level3Complete")
                    report["saved_complete"] = str(path)
                    write_state_provenance(
                        path,
                        source_state_path=(
                            GAME_DIR
                            / "custom_integrations"
                            / GAME
                            / f"{start_state}.state"
                        ),
                        request={
                            "segment": "level3_complete",
                            "natural_entry": False,
                            "start_state": start_state,
                            "intervention_class": intervention,
                        },
                        selected_trial={"tf04": True},
                        natural_entry=False,
                    )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_after.png")

        report["final"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["total_frames"] = total[0]
        report["ok"] = bool(
            report["reached_4d"]
            or report["tf04"]
            or (report["boss_beaten"] and report["dmg_events"] > 0)
        )
        # Success tiers for multi-trial rollup
        report["success_tier"] = (
            "tf04"
            if report["tf04"]
            else (
                "boss_kill"
                if report["boss_beaten"]
                else ("enter_4d" if report["reached_4d"] else "partial")
            )
        )
        out = RECORDINGS_DIR / f"{tag}_report.json"
        write_json_report(out, report)
        report["report_path"] = str(out)
        return report
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level3Raft")
    p.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--trials", type=int, default=1)
    p.add_argument(
        "--to-boss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after entering 0x4d (default True); with --kill continues",
    )
    p.add_argument(
        "--kill",
        action="store_true",
        help="Fight Manhandla after enter 0x4d",
    )
    p.add_argument(
        "--phase",
        choices=("all", "to5d", "gate5d", "boss", "kill"),
        default="all",
        help="all=path+gate+fight; to5d/gate5d for short probes",
    )
    p.add_argument(
        "--poke-bombs",
        type=int,
        default=16,
        help="RECON bomb poke (0=disable). Default 16.",
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l3_to_boss")
    args = p.parse_args(argv)

    poke = None if args.poke_bombs == 0 else args.poke_bombs
    kill = args.kill or args.phase in ("kill", "boss", "all")
    # phase=all includes kill attempt; --to-boss alone without --kill stops at enter
    if args.phase == "all" and not args.kill and args.to_boss:
        # Default all tries kill; use --no-to-boss? Actually goal wants kill if possible
        kill = True
    if args.phase in ("to5d", "gate5d"):
        kill = False

    trials: list[dict] = []
    for i in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{i}"
        rep = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            to_boss=args.to_boss,
            kill=kill,
            poke_bombs=poke,
            save_checkpoint=args.save_state and i == 0,
            tag=tag,
            phase=args.phase,
        )
        trials.append(rep)
        print(
            f"trial{i}: ok={rep.get('ok')} tier={rep.get('success_tier')} "
            f"5d={rep.get('reached_5d')} 4d={rep.get('reached_4d')} "
            f"man={rep.get('manhandla_confirmed')} kill={rep.get('boss_beaten')} "
            f"tf04={rep.get('tf04')} dmg={rep.get('dmg_events')} "
            f"frames={rep.get('total_frames')} err={rep.get('error')}"
        )
        if rep.get("gate_5d"):
            g = rep["gate_5d"]
            print(f"  gate: ok={g.get('ok')} method={g.get('method')} clear={g.get('clear')}")
        if rep.get("notes"):
            for n in rep["notes"][:6]:
                print(f"  note: {n}")
        if rep.get("traps"):
            for t in rep["traps"][:6]:
                print(f"  trap: {t}")

    n4d = sum(1 for t in trials if t.get("reached_4d"))
    nkill = sum(1 for t in trials if t.get("boss_beaten"))
    ntf = sum(1 for t in trials if t.get("tf04"))
    rollup = {
        "trials": len(trials),
        "enter_4d": f"{n4d}/{len(trials)}",
        "boss_kill": f"{nkill}/{len(trials)}",
        "tf04": f"{ntf}/{len(trials)}",
        "intervention_class": "survival" if args.infinite_life else "clean",
        "trial_summaries": [
            {
                "ok": t.get("ok"),
                "tier": t.get("success_tier"),
                "reached_5d": t.get("reached_5d"),
                "reached_4d": t.get("reached_4d"),
                "manhandla": t.get("manhandla_confirmed"),
                "boss_beaten": t.get("boss_beaten"),
                "tf04": t.get("tf04"),
                "dmg_events": t.get("dmg_events"),
                "error": t.get("error"),
                "method": (t.get("gate_5d") or {}).get("method"),
                "frames": t.get("total_frames"),
            }
            for t in trials
        ],
        "reports": [t.get("report_path") for t in trials],
    }
    out = RECORDINGS_DIR / f"{args.tag}_rollup.json"
    write_json_report(out, rollup)
    print(f"rollup: 4d={n4d}/{len(trials)} kill={nkill}/{len(trials)} tf={ntf}/{len(trials)}")
    print(f"rollup_path={out}")
    return 0 if n4d > 0 or nkill > 0 or ntf > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
