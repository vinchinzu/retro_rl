"""Assisted recon: Level3Raft → Manhandla → Triforce bit 0x04.

Starts in mode-9 passage 0x0f with ``ADDR_RAFT`` already set. Maps the stairs
exit back into the dungeon graph, free-explores toward boss (bomb-RIGHT 0x5c
shortcut preferred when keys=0), identifies Manhandla object type / HP, and
attempts bomb combat. Survival (``--infinite-life``) default — not Clean STATUS.

LIVE path (2026-08-07, assisted Survival)::

    0x0f mode9 (Raft mid ~(128,141)):
      RIGHT → (176,141) → DOWN → (176,189) → LEFT → (48,189)
      → UP → (48,77) → hold UP → mode10 → settle play 0x69 ~(96,141)
    0x69 UP → 0x59
    0x59 BOMB_RIGHT@(192,141) → 0x5a   *** walk-RIGHT sealed post-Raft ***
    0x5a RIGHT → 0x5b → BOMB_RIGHT@(192,141) → 0x5c (3× Darknut)
    0x5c full clear → doors raw=3 → RIGHT @ y≈141 → 0x5d
    0x5d UP → 0x4d Manhandla candidate type 0x3c (residual / flaky gate)

False boss: type 0x2b HP240 on 0x49/0x5d is invulnerable (not Manhandla).
TF bit 0x04 not yet collected assisted.

Examples::

    uv run python nes/zelda_i/scripts/probe_level3_manhandla.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level3_manhandla.py \\
        --infinite-life --tag l3_manhandla --max-hops 12 --poke-bombs 12
    uv run python nes/zelda_i/scripts/probe_level3_manhandla.py \\
        --infinite-life --phase exit-only
    uv run python nes/zelda_i/scripts/probe_level3_manhandla.py \\
        --infinite-life --phase explore --max-hops 8
    uv run python nes/zelda_i/scripts/probe_level3_manhandla.py \\
        --infinite-life --phase boss --poke-bombs 16 --save-state
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
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.level3_dungeon import (
    DARKNUT_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_DARKNUTS,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_RAFT,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

# --- Anchors ---
LEVEL3_TRIFORCE_BIT = 0x04
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

# Passage reverse (LIVE): mid → channel → south → west → stairs NW → UP.
PASSAGE_EXIT_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (176, 141),
    (176, 189),
    (48, 189),
    (48, 77),
)
PASSAGE_STAIRS_PUSH = "UP"
PASSAGE_EXIT_ROOM = ROOM_L3_SOUTH_DARKNUTS  # 0x69

# Bomb stands (wall recon). Include 0x5b boss-shortcut stand.
BOMB_STANDS: list[tuple[str, int, int]] = [
    ("RIGHT", 192, 141),
    ("UP", 120, 101),
    ("LEFT", 48, 141),
    ("DOWN", 120, 189),
    ("UP", 96, 101),
    ("UP", 144, 101),
    ("RIGHT", 192, 117),
    ("RIGHT", 192, 165),
    ("LEFT", 48, 117),
    ("LEFT", 48, 165),
]

PUSH_FRAMES = 100
SETTLE_FRAMES = 80
ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

# Non-combat / projectile types for explore clear.
_NON_COMBAT_TYPES = {
    0x5A,
    0x4F,
    0x4E,
    0x60,
    0x61,
    0x62,
    0x5B,
    0x5C,
    0x49,  # blade trap
    0x55,  # fireball
    0x40,  # bubble (disarms; do not "clear")
}

# Boss-like type candidates.
# LIVE: Manhandla candidate 0x3c on 0x4d; Dodongo 0x32; Aquamentus 0x3d.
# 0x2b is invuln mover on 0x49/0x5d — exclude from auto-fight.
_KNOWN_BOSS_TYPES = {0x32, 0x3C, 0x3D}
_FALSE_BOSS_TYPES = {0x2B}  # HP240 invuln residual
_BOSS_HP_HINT = 48


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10) or o.type_id in (0, 0xFF):
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
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    )
    raft = int(read_u8(ram, ADDR_RAFT)) if ram is not None else None
    tf = int(read_u8(ram, ADDR_TRIFORCE)) if ram is not None else None
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
        if snap.mode in (6, 7) or snap.transitioning:
            env.step(nes_action(direction))
        else:
            env.step(nes_action(direction))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(SETTLE_FRAMES):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.level == LEVEL3:
            if snap.screen != room0 or mode0 == 9:
                for _ in range(40):
                    env.step(nes_idle_action())
                    total[0] += 1
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
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


def _poke_bombs(env, n: int = 12) -> str:
    """RECON-ONLY inventory poke. Document in report — not Clean."""
    try:
        env.unwrapped.data.set_value("bombs", int(n) & 0xFF)
        return f"bombs={n}"
    except Exception as exc:
        return f"poke_fail={exc!r}"


def _poke_keys(env, n: int = 4) -> str:
    """RECON-ONLY keys poke for locked-door mapping. Document — not Clean."""
    try:
        env.unwrapped.data.set_value("keys", int(n) & 0xFF)
        return f"keys={n}"
    except Exception as exc:
        return f"poke_keys_fail={exc!r}"


def exit_raft_passage(env, assist, total: list[int]) -> dict:
    """Leave mode-9 0x0f via reverse channel + NW stairs UP → 0x69 play."""
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    if not (
        snap0.mode == 9
        and snap0.screen == ROOM_L3_RAFT_PASSAGE
        and snap0.level == LEVEL3
    ):
        return {
            "ok": False,
            "error": (
                f"expected mode9 room 0x0f; got mode={snap0.mode} "
                f"sc=0x{snap0.screen:02x} level={snap0.level}"
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

    # Hold UP into stairs.
    for i in range(220):
        s = read_snapshot(env.get_ram())
        if s.mode != 9 or s.screen != ROOM_L3_RAFT_PASSAGE:
            break
        env.step(nes_action(PASSAGE_STAIRS_PUSH))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    else:
        i = 220

    # Settle mode10 → play.
    for _ in range(120):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])

    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    ok = (
        after["mode"] == PLAY_MODE
        and after["level"] == LEVEL3
        and after["screen"] == PASSAGE_EXIT_ROOM
    )
    return {
        "ok": ok,
        "before": before,
        "after": after,
        "waypoints": wp_log,
        "stairs_push_frames": i + 1 if "i" in dir() else 0,
        "recipe": (
            "RIGHT(176,141)→DOWN(176,189)→LEFT(48,189)→UP(48,77)→hold UP "
            "→ mode10 → play 0x69"
        ),
    }


def _try_exit(
    env,
    assist,
    total: list[int],
    direction: str,
    *,
    tag: str,
    stem: str,
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    before = _room_fields(snap0, env.get_ram())
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        _goto(env, assist, total, snap0.link_x, ty, tol=3)
        _goto(env, assist, total, tx, ty, tol=4)
    else:
        _goto(env, assist, total, tx, snap0.link_y, tol=3)
        _goto(env, assist, total, tx, ty, tol=4)

    mid = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    _push_dir(env, assist, total, direction, frames=PUSH_FRAMES + 50)
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"] or after["mode"] != before["mode"]
    keys_spent = before["keys"] - after["keys"]
    png = RECORDINGS_DIR / f"{tag}_{stem}_{direction.lower()}.png"
    return {
        "direction": direction,
        "before": before,
        "at_door": mid,
        "after": after,
        "changed_room": changed,
        "keys_spent": keys_spent,
        "result": (
            "room_change"
            if changed
            else ("key_spent_no_room" if keys_spent else "blocked")
        ),
        "screenshot": str(png),
    }


def _try_bomb_stand(
    env,
    assist,
    total: list[int],
    face: str,
    x: int,
    y: int,
    *,
    tag: str,
    stem: str,
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
    for _ in range(PUSH_FRAMES + 60):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        env.step(nes_action(face))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    for _ in range(SETTLE_FRAMES):
        env.step(nes_idle_action())
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    changed = after["screen"] != before["screen"]
    png = RECORDINGS_DIR / f"{tag}_{stem}_bomb_{face.lower()}_{x}_{y}.png"
    return {
        "face": face,
        "stand": [x, y],
        "before": before,
        "after": after,
        "bombs_before": bombs0,
        "bombs_after": after["bombs"],
        "bombs_spent": bombs0 - after["bombs"],
        "changed_room": changed,
        "result": "room_change" if changed else "no_open",
        "screenshot": str(png),
    }


def _combat_types(snap: ZeldaSnapshot) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                o.type_id
                for o in snap.objects
                if 1 <= o.slot <= 10
                and o.type_id not in (0, 0xFF)
                and o.type_id not in _NON_COMBAT_TYPES
                and (o.hp > 0 or o.type_id == KEESE_OBJECT_TYPE)
            }
        )
    )


def _looks_like_boss(snap: ZeldaSnapshot) -> dict | None:
    """Heuristic: known boss type OR multi-slot high-HP identical types."""
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 10
        and o.type_id not in (0, 0xFF)
        and o.type_id not in _NON_COMBAT_TYPES
        and o.type_id not in _FALSE_BOSS_TYPES
    ]
    for o in live:
        if o.type_id in _KNOWN_BOSS_TYPES and o.hp > 0:
            return {
                "kind": "known_boss",
                "type": o.type_id,
                "type_name": object_name(o.type_id),
                "slots": 1,
                "objects": _objs(snap),
            }
    by_type: dict[int, list] = {}
    for o in live:
        if o.hp >= _BOSS_HP_HINT or o.type_id not in (
            ZOL_OBJECT_TYPE,
            DARKNUT_OBJECT_TYPE,
            KEESE_OBJECT_TYPE,
            0x14,
            0x15,
            0x28,
            0x06,
            0x0B,
        ):
            by_type.setdefault(o.type_id, []).append(o)
    for tid, group in by_type.items():
        # Manhandla: 4 heads — multiple slots same exotic type
        if tid in (ZOL_OBJECT_TYPE, DARKNUT_OBJECT_TYPE, KEESE_OBJECT_TYPE, 0x14, 0x15):
            continue
        if len(group) >= 2 or (len(group) == 1 and group[0].hp >= _BOSS_HP_HINT):
            return {
                "kind": "candidate",
                "type": tid,
                "type_name": object_name(tid),
                "slots": len(group),
                "hps": [o.hp for o in group],
                "objects": [
                    {
                        "slot": o.slot,
                        "type": o.type_id,
                        "x": o.x,
                        "y": o.y,
                        "hp": o.hp,
                        "facing": o.facing,
                    }
                    for o in group
                ],
            }
    return None


def _fight_clear(
    env,
    assist,
    total: list[int],
    *,
    enemy_types: tuple[int, ...],
    max_frames: int = 10000,
    use_bombs: bool = False,
) -> dict:
    """Center-patrol sword clear; optional bomb drops near nearest enemy."""
    from zelda_i.dungeon import (
        AliveRule,
        CombatTuning,
        DoorRoute,
        DungeonRoomSpec,
        GenericDungeonRoomController,
        RewardKind,
        RewardSpec,
    )

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
    )
    type_only = all(t in (KEESE_OBJECT_TYPE, 0x15) for t in enemy_types)
    spec = DungeonRoomSpec(
        spec_id=f"l3_man_probe_0x{room:02x}",
        source_room=room,
        room_id=room,
        entry=DoorRoute("DOWN", ((120, 205),)),
        enemy_types=enemy_types,
        expected_enemy_count=1,
        alive_rule=AliveRule.TYPE if type_only else AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=patrol,
            engage_distance=48,
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
    if use_bombs:
        _ensure_bomb(env)

    for frame in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame}
        live = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 10
            and o.type_id in enemy_types
            and (o.hp > 0 or type_only)
        ]
        n = len(live)
        if prev_live < 0:
            prev_live = n
            last_progress = frame
        elif n < prev_live:
            prev_live = n
            last_progress = frame
        if n == 0 and frame > 40:
            for _ in range(40):
                env.step(nes_idle_action())
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
            return {
                "ok": True,
                "frames": frame,
                "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
            }

        if use_bombs and live and bomb_cd <= 0 and snap.bombs > 0:
            nearest = min(
                live,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
            if dist < 36:
                # Face toward enemy and drop bomb.
                dx = nearest.x - snap.link_x
                dy = nearest.y - snap.link_y
                if abs(dx) >= abs(dy):
                    face = "RIGHT" if dx > 0 else "LEFT"
                else:
                    face = "DOWN" if dy > 0 else "UP"
                _ensure_bomb(env)
                env.step(nes_action(face, "B"))
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
                bomb_cd = 90
                continue
        if bomb_cd > 0:
            bomb_cd -= 1

        act = ctl.step(snap)
        env.step(act.action)
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        if frame - last_progress > 800 and n > 0:
            # unstick roam
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[frame % 4]))
            total[0] += 1

    return {
        "ok": False,
        "error": "timeout",
        "frames": max_frames,
        "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
    }


def _fight_boss_bombs(
    env,
    assist,
    total: list[int],
    *,
    enemy_type: int,
    max_frames: int = 18000,
    poke_bombs: int | None = 16,
) -> dict:
    """Circle + bomb near multi-head / boss type; sword secondary."""
    log: list[dict] = []
    notes: list[str] = []
    if poke_bombs is not None:
        notes.append(f"RECON poke {_poke_bombs(env, poke_bombs)}")
    _ensure_bomb(env)
    bomb_cd = 0
    last_hps: list[int] | None = None
    dmg_events = 0

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
                "log": log[-20:],
                "notes": notes,
                "final": _room_fields(snap, ram),
            }
        if snap.mode == 17:
            return {"ok": False, "error": "death", "frames": frame, "notes": notes}
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
        # Also accept type present with HP residual 0 briefly after hit
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

        if not heads_any and snap.room_all_dead >= 15 and frame > 80:
            log.append(
                {
                    "event": "boss_dead",
                    "frame": frame,
                    **_room_fields(snap, ram),
                }
            )
            # Roam for heart container / door open.
            for _ in range(400):
                s2 = read_snapshot(env.get_ram())
                if s2.room_item_id in (0x1A, 0x1B) or (
                    int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT
                ):
                    break
                tx = 72 + (_ % 6) * 16
                ty = 109 + ((_ // 6) % 5) * 16
                if abs(s2.link_x - tx) > 4:
                    d = "RIGHT" if s2.link_x < tx else "LEFT"
                elif abs(s2.link_y - ty) > 4:
                    d = "DOWN" if s2.link_y < ty else "UP"
                else:
                    d = ("LEFT", "UP", "RIGHT", "DOWN")[_ % 4]
                env.step(nes_action(d))
                total[0] += 1
                if assist is not None:
                    assist.apply_env(env, frame=total[0])
            return {
                "ok": True,
                "tf04": bool(
                    int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT
                ),
                "frames": frame,
                "dmg_events": dmg_events,
                "log": log[-30:],
                "notes": notes,
                "final": _room_fields(read_snapshot(env.get_ram()), env.get_ram()),
            }

        if assist is not None and snap.bombs < 2 and poke_bombs:
            notes.append(f"topup {_poke_bombs(env, poke_bombs)}")
            _ensure_bomb(env)

        if not heads:
            # wander + sword while waiting respawn/despawn
            env.step(nes_action(("UP", "RIGHT", "DOWN", "LEFT")[frame // 15 % 4], "A"))
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            continue

        nearest = min(
            heads,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        dist = abs(nearest.x - snap.link_x) + abs(nearest.y - snap.link_y)
        dx = nearest.x - snap.link_x
        dy = nearest.y - snap.link_y
        if abs(dx) >= abs(dy):
            face = "RIGHT" if dx > 0 else "LEFT"
            approach = face
            # circle: perpendicular
            circle = "DOWN" if (frame // 30) % 2 == 0 else "UP"
        else:
            face = "DOWN" if dy > 0 else "UP"
            approach = face
            circle = "RIGHT" if (frame // 30) % 2 == 0 else "LEFT"

        if bomb_cd > 0:
            bomb_cd -= 1

        # Place bomb when close; else approach or circle
        if dist < 40 and bomb_cd <= 0 and snap.bombs > 0:
            _ensure_bomb(env)
            # step toward then B
            if dist > 18:
                env.step(nes_action(approach))
            else:
                env.step(nes_action(face, "B"))
                bomb_cd = 70
                log.append(
                    {
                        "event": "bomb_place",
                        "frame": frame,
                        "at": [snap.link_x, snap.link_y],
                        "target": [nearest.x, nearest.y, nearest.hp],
                        "bombs": snap.bombs,
                    }
                )
        elif dist > 50:
            if frame % 4 == 0:
                env.step(nes_action(approach, "A"))
            else:
                env.step(nes_action(approach))
        else:
            # circle and slash
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
                    "item": snap.room_item_id,
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


def _edge(from_room: int | str, direction: str, probe: dict) -> dict:
    fr = from_room if isinstance(from_room, str) else f"0x{from_room:02x}"
    return {
        "from": fr,
        "dir": direction,
        "to": (
            f"0x{probe['after']['screen']:02x}" if probe["changed_room"] else None
        ),
        "mode_after": probe["after"].get("mode"),
        "keys_spent": probe["keys_spent"],
        "result": probe["result"],
        "after_types": probe["after"].get("type_counts"),
        "after_item": probe["after"].get("room_item_id"),
        "after_keys": probe["after"].get("keys"),
        "after_bombs": probe["after"].get("bombs"),
        "after_doors": probe["after"].get("doors"),
        "after_tf": probe["after"].get("triforce"),
    }


def _reload(env, start_state: str, assist, total: list[int]):
    env.close()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    result = env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    obs, *_ = env.step(nes_idle_action())
    if assist is not None:
        assist.apply_env(env, frame=0)
    total[0] = 1
    return env, obs


def _roam_pickup(env, assist, total: list[int], frames: int = 300) -> dict:
    k0 = read_snapshot(env.get_ram()).keys
    b0 = read_snapshot(env.get_ram()).bombs
    hc0 = read_snapshot(env.get_ram()).heart_containers
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    for i in range(frames):
        snap = read_snapshot(env.get_ram())
        if (
            snap.keys > k0
            or snap.bombs > b0
            or snap.heart_containers > hc0
            or (int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT)
            or snap.room_item_id in (0, 0xFF)
        ):
            break
        tx = 72 + (i % 6) * 16
        ty = 109 + ((i // 6) % 5) * 16
        if abs(snap.link_x - tx) > 4:
            d = "RIGHT" if snap.link_x < tx else "LEFT"
        elif abs(snap.link_y - ty) > 4:
            d = "DOWN" if snap.link_y < ty else "UP"
        else:
            d = ("LEFT", "UP", "RIGHT", "DOWN")[i % 4]
        env.step(nes_action(d))
        total[0] += 1
        if assist is not None:
            assist.apply_env(env, frame=total[0])
    after = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
    return {
        "keys": f"{k0}->{after['keys']}",
        "bombs": f"{b0}->{after['bombs']}",
        "hc": f"{hc0}->{after['heart_containers']}",
        "tf": f"{tf0:02x}->{after['triforce']:02x}" if after["triforce"] is not None else None,
        "final": after,
    }


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    phase: str,
    max_hops: int,
    poke_bombs: int | None,
    poke_keys: int | None,
    try_bombs: bool,
    save_checkpoints: bool,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    track = "assisted" if infinite_life else "clean"
    recon_notes: list[str] = []
    trap_notes: list[str] = []

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total[0] = 1
        _idle(env, assist, total, 25)

        entry = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        reports: dict = {
            "ok": False,
            "track": track,
            "start_state": start_state,
            "phase": phase,
            "entry": entry,
            "graph_edges": [],
            "room_notes": {},
            "path_log": [],
            "trap_notes": trap_notes,
            "recon_notes": recon_notes,
            "boss": None,
            "tf04": False,
            "intervention_class": "survival" if infinite_life else "clean",
        }
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_start.png")
        reports["room_notes"][entry["sc"]] = entry

        if not (
            entry["level"] == LEVEL3
            and entry.get("raft")
            and (
                (entry["mode"] == 9 and entry["screen"] == ROOM_L3_RAFT_PASSAGE)
                or entry["screen"] == PASSAGE_EXIT_ROOM
            )
        ):
            # Soft allow if already left passage but has raft
            if not (entry["level"] == LEVEL3 and entry.get("raft")):
                reports["error"] = (
                    f"expected Level3Raft (raft set, mode9 0x0f or 0x69); "
                    f"got level={entry['level']} mode={entry['mode']} "
                    f"sc=0x{entry['screen']:02x} raft={entry.get('raft')}"
                )
                out = RECORDINGS_DIR / f"{tag}_recon.json"
                write_json_report(out, reports)
                reports["report_path"] = str(out)
                return reports
            recon_notes.append("start already outside passage with raft")

        # --- Phase A: exit passage ---
        if entry["mode"] == 9 and entry["screen"] == ROOM_L3_RAFT_PASSAGE:
            exit_rep = exit_raft_passage(env, assist, total)
            reports["passage_exit"] = {
                "ok": exit_rep.get("ok"),
                "recipe": exit_rep.get("recipe"),
                "waypoints": exit_rep.get("waypoints"),
                "after": exit_rep.get("after"),
                "error": exit_rep.get("error"),
            }
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_exit_0x69.png")
            reports["path_log"].append(
                {
                    "phase": "passage_exit",
                    "ok": exit_rep.get("ok"),
                    "to": (
                        exit_rep.get("after", {}).get("sc")
                        if exit_rep.get("after")
                        else None
                    ),
                }
            )
            if exit_rep.get("ok"):
                reports["graph_edges"].append(
                    {
                        "from": "0x0f",
                        "dir": "STAIRS_UP_NW",
                        "to": "0x69",
                        "result": "room_change",
                        "notes": exit_rep.get("recipe"),
                        "phase": "passage_exit",
                    }
                )
                reports["room_notes"]["0x69"] = exit_rep["after"]
                recon_notes.append(
                    "LIVE exit 0x0f→0x69 via reverse channel + NW stairs UP"
                )
            else:
                reports["error"] = exit_rep.get("error") or "passage_exit_failed"
                out = RECORDINGS_DIR / f"{tag}_recon.json"
                write_json_report(out, reports)
                reports["report_path"] = str(out)
                return reports
        else:
            recon_notes.append("skipped passage exit (already in dungeon)")

        if phase == "exit-only":
            reports["ok"] = True
            out = RECORDINGS_DIR / f"{tag}_recon.json"
            write_json_report(out, reports)
            reports["report_path"] = str(out)
            return reports

        # Optional recon inventory pokes after exit.
        if poke_bombs is not None:
            note = _poke_bombs(env, poke_bombs)
            recon_notes.append(f"RECON inventory poke: {note} (not Clean)")
            _ensure_bomb(env)
            _idle(env, assist, total, 3)
        if poke_keys is not None:
            note = _poke_keys(env, poke_keys)
            recon_notes.append(f"RECON inventory poke: {note} (not Clean)")
            _idle(env, assist, total, 3)

        # Snapshot at 0x69 for restore during door probes.
        state_base = env.em.get_state()
        base_room = read_snapshot(env.get_ram()).screen

        # --- Phase B: door probe from 0x69 ---
        door_probes = []
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            env.em.set_state(state_base)
            obs, *_ = env.step(nes_idle_action())
            if assist is not None:
                assist.apply_env(env, frame=total[0])
            pr = _try_exit(
                env, assist, total, direction, tag=tag, stem=f"{base_room:02x}_raw"
            )
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, Path(pr["screenshot"]))
            door_probes.append(
                {
                    "direction": pr["direction"],
                    "changed_room": pr["changed_room"],
                    "keys_spent": pr["keys_spent"],
                    "to": (
                        f"0x{pr['after']['screen']:02x}"
                        if pr["changed_room"]
                        else None
                    ),
                    "types": pr["after"].get("type_counts"),
                    "item": pr["after"].get("room_item_id"),
                    "result": pr["result"],
                }
            )
            edge = _edge(base_room, direction, pr)
            edge["phase"] = f"0x{base_room:02x}_raw"
            reports["graph_edges"].append(edge)
            if pr["changed_room"]:
                reports["room_notes"][pr["after"]["sc"]] = pr["after"]
        reports[f"door_probes_0x{base_room:02x}"] = door_probes

        # Restore for explore.
        env.em.set_state(state_base)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=total[0])

        # --- Phase C: BFS-lite explore toward boss ---
        # Prefer: UP (backtrack) → RIGHT when on 0x5b bomb → north doors.
        prefer_by_room = {
            0x69: ("UP", "RIGHT", "LEFT", "DOWN"),
            # walk-RIGHT sealed post-Raft — explore will bomb; prefer UP only as fallback
            0x59: ("RIGHT", "UP", "DOWN", "LEFT"),
            0x5A: ("RIGHT", "UP", "LEFT", "DOWN"),
            0x5B: ("RIGHT", "UP", "LEFT", "DOWN"),  # bomb R → 0x5c shortcut
            0x5C: ("RIGHT", "UP", "LEFT", "DOWN"),  # RIGHT@y141 → 0x5d after clear
            0x5D: ("UP", "RIGHT", "LEFT", "DOWN"),  # UP → 0x4d Manhandla residual
            0x4D: ("UP", "RIGHT", "LEFT", "DOWN"),
            0x4B: ("RIGHT", "UP", "LEFT", "DOWN"),
            0x4C: ("UP", "RIGHT", "LEFT", "DOWN"),
            0x4A: ("RIGHT", "UP", "LEFT", "DOWN"),
            0x49: ("RIGHT", "DOWN", "LEFT", "UP"),
        }
        visited: set[int] = {base_room}
        hops: list[dict] = []
        boss_hit: dict | None = None

        for hop_i in range(max_hops):
            snap = read_snapshot(env.get_ram())
            room_before = snap.screen
            if snap.mode != PLAY_MODE or snap.level != LEVEL3:
                trap_notes.append(
                    f"hop{hop_i}: unexpected mode={snap.mode} sc=0x{snap.screen:02x}"
                )
                break

            # Spawn settle for Darknuts etc.
            _idle(env, assist, total, 50)

            # Detect boss.
            boss_info = _looks_like_boss(read_snapshot(env.get_ram()))
            if boss_info is not None:
                boss_hit = {
                    "room": f"0x{room_before:02x}",
                    **boss_info,
                    "room_fields": _room_fields(
                        read_snapshot(env.get_ram()), env.get_ram()
                    ),
                }
                reports["boss"] = boss_hit
                recon_notes.append(
                    f"BOSS candidate in 0x{room_before:02x}: "
                    f"type 0x{boss_info['type']:02x} "
                    f"({boss_info['type_name']}) slots={boss_info['slots']}"
                )
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_boss_0x{room_before:02x}.png"
                )
                if save_checkpoints:
                    path = save_state(
                        env, GAME_DIR, GAME, f"L3Boss_{room_before:02X}"
                    )
                    reports["saved_boss"] = str(path)
                if phase in ("explore", "all", "boss"):
                    # Fight immediately.
                    fight = _fight_boss_bombs(
                        env,
                        assist,
                        total,
                        enemy_type=boss_info["type"],
                        max_frames=16000,
                        poke_bombs=poke_bombs if poke_bombs is not None else 16,
                    )
                    reports["boss_fight"] = {
                        "ok": fight.get("ok"),
                        "tf04": fight.get("tf04"),
                        "frames": fight.get("frames"),
                        "dmg_events": fight.get("dmg_events"),
                        "error": fight.get("error"),
                        "notes": fight.get("notes"),
                        "final": fight.get("final"),
                        "log_tail": fight.get("log", [])[-12:],
                    }
                    reports["tf04"] = bool(fight.get("tf04"))
                    obs, *_ = env.step(nes_idle_action())
                    save_rgb_png(
                        obs,
                        RECORDINGS_DIR / f"{tag}_boss_fight_0x{room_before:02x}.png",
                    )
                    if fight.get("tf04") or fight.get("ok"):
                        # Try exits to TF room after kill.
                        st_post = env.em.get_state()
                        for direction in ("UP", "RIGHT", "LEFT", "DOWN"):
                            env.em.set_state(st_post)
                            obs, *_ = env.step(nes_idle_action())
                            if assist is not None:
                                assist.apply_env(env, frame=total[0])
                            pr = _try_exit(
                                env,
                                assist,
                                total,
                                direction,
                                tag=tag,
                                stem=f"postboss_{room_before:02x}",
                            )
                            edge = _edge(room_before, direction, pr)
                            edge["phase"] = "post_boss"
                            reports["graph_edges"].append(edge)
                            if pr["changed_room"]:
                                reports["room_notes"][pr["after"]["sc"]] = pr[
                                    "after"
                                ]
                                # Collect TF if room item / roam
                                if pr["after"].get("room_item_id") in (
                                    0x1A,
                                    0x1B,
                                ) or not pr["after"].get("tf04"):
                                    pick = _roam_pickup(
                                        env, assist, total, frames=600
                                    )
                                    reports["post_boss_pickup"] = pick
                                    if pick["final"].get("tf04"):
                                        reports["tf04"] = True
                                        recon_notes.append(
                                            f"TF 0x04 collected in "
                                            f"{pick['final']['sc']}"
                                        )
                                        if save_checkpoints:
                                            path = save_state(
                                                env,
                                                GAME_DIR,
                                                GAME,
                                                "Level3Complete",
                                            )
                                            reports["saved_complete"] = str(path)
                                        obs, *_ = env.step(nes_idle_action())
                                        save_rgb_png(
                                            obs,
                                            RECORDINGS_DIR / f"{tag}_tf04.png",
                                        )
                        break
                    if phase == "boss":
                        break
                if phase == "boss":
                    break

            # Clear non-boss combat (skip bubbles).
            snap = read_snapshot(env.get_ram())
            ctypes = _combat_types(snap)
            # Don't full-clear if boss candidate mixed in
            if boss_info is None and ctypes:
                # Cap darknut clears to keep hop budget
                max_f = 8000 if DARKNUT_OBJECT_TYPE in ctypes else 6000
                if len(ctypes) == 1 and ctypes[0] == DARKNUT_OBJECT_TYPE:
                    # 0x69 may already be clear; short attempt
                    max_f = 5000
                clr = _fight_clear(
                    env,
                    assist,
                    total,
                    enemy_types=ctypes,
                    max_frames=max_f,
                    use_bombs=False,
                )
            else:
                clr = {"ok": True, "skipped": True}

            pick = _roam_pickup(env, assist, total, frames=200)

            # Bomb wall tries on key rooms (0x5b shortcut).
            bomb_edges = []
            if try_bombs and room_before in (
                ROOM_L3_DARKNUTS,
                ROOM_L3_BOMB_SHORTCUT,
                0x4C,
                0x4B,
                0x5C,
                0x59,  # post-Raft reopen east
                0x5D,
                0x49,
            ):
                if read_snapshot(env.get_ram()).bombs <= 0 and poke_bombs:
                    _poke_bombs(env, poke_bombs)
                    _ensure_bomb(env)
                st_bomb = env.em.get_state()
                for face, bx, by in BOMB_STANDS[:4]:
                    if read_snapshot(env.get_ram()).bombs <= 0:
                        if poke_bombs:
                            _poke_bombs(env, poke_bombs)
                            _ensure_bomb(env)
                        else:
                            break
                    env.em.set_state(st_bomb)
                    obs, *_ = env.step(nes_idle_action())
                    if assist is not None:
                        assist.apply_env(env, frame=total[0])
                    if read_snapshot(env.get_ram()).bombs <= 0 and poke_bombs:
                        _poke_bombs(env, poke_bombs)
                    br = _try_bomb_stand(
                        env,
                        assist,
                        total,
                        face,
                        bx,
                        by,
                        tag=tag,
                        stem=f"h{hop_i}_{room_before:02x}",
                    )
                    obs, *_ = env.step(nes_idle_action())
                    save_rgb_png(obs, Path(br["screenshot"]))
                    bomb_edges.append(
                        {
                            "face": face,
                            "stand": br["stand"],
                            "result": br["result"],
                            "to": (
                                f"0x{br['after']['screen']:02x}"
                                if br["changed_room"]
                                else None
                            ),
                            "types": br["after"].get("type_counts"),
                        }
                    )
                    if br["changed_room"]:
                        dest = br["after"]["screen"]
                        reports["graph_edges"].append(
                            {
                                "from": f"0x{room_before:02x}",
                                "dir": f"BOMB_{face}",
                                "to": f"0x{dest:02x}",
                                "result": "bomb_open",
                                "stand": br["stand"],
                                "after_types": br["after"].get("type_counts"),
                                "phase": f"hop{hop_i}_bomb",
                            }
                        )
                        reports["room_notes"][f"0x{dest:02x}"] = br["after"]
                        # Prefer taking bomb shortcut if new room.
                        if dest not in visited:
                            visited.add(dest)
                            hops.append(
                                {
                                    "hop": hop_i,
                                    "from": f"0x{room_before:02x}",
                                    "dir": f"BOMB_{face}",
                                    "to": f"0x{dest:02x}",
                                    "types": br["after"].get("type_counts"),
                                    "item": br["after"].get("room_item_id"),
                                    "clear": clr.get("ok"),
                                }
                            )
                            # stay in new room
                            st_bomb = None  # don't restore
                            room_before = dest
                            break
                if st_bomb is not None:
                    env.em.set_state(st_bomb)
                    obs, *_ = env.step(nes_idle_action())
                reports.setdefault("bomb_probes", []).extend(bomb_edges)

            # Door hop preference.
            order = prefer_by_room.get(
                room_before, ("UP", "RIGHT", "LEFT", "DOWN")
            )
            moved = False
            for require_new in (True, False):
                if moved:
                    break
                for direction in order:
                    st = env.em.get_state()
                    pr = _try_exit(
                        env,
                        assist,
                        total,
                        direction,
                        tag=tag,
                        stem=f"hop{hop_i}",
                    )
                    if pr["changed_room"] and pr["after"]["screen"] != room_before:
                        dest_r = pr["after"]["screen"]
                        if require_new and dest_r in visited:
                            env.em.set_state(st)
                            obs, *_ = env.step(nes_idle_action())
                            continue
                        hops.append(
                            {
                                "hop": hop_i,
                                "from": f"0x{room_before:02x}",
                                "dir": direction,
                                "to": f"0x{dest_r:02x}",
                                "keys_spent": pr["keys_spent"],
                                "types": pr["after"].get("type_counts"),
                                "item": pr["after"].get("room_item_id"),
                                "item_name": pr["after"].get("room_item_name"),
                                "bombs": pr["after"].get("bombs"),
                                "keys": pr["after"].get("keys"),
                                "clear": clr.get("ok"),
                                "pickup": {
                                    "keys": pick.get("keys"),
                                    "bombs": pick.get("bombs"),
                                },
                            }
                        )
                        reports["graph_edges"].append(
                            {
                                **_edge(room_before, direction, pr),
                                "phase": f"explore_hop{hop_i}",
                            }
                        )
                        reports["room_notes"][f"0x{dest_r:02x}"] = pr["after"]
                        visited.add(dest_r)
                        obs, *_ = env.step(nes_idle_action())
                        save_rgb_png(
                            obs,
                            RECORDINGS_DIR
                            / f"{tag}_hop{hop_i}_0x{dest_r:02x}.png",
                        )
                        if save_checkpoints and dest_r not in (
                            ROOM_L3_SOUTH_DARKNUTS,
                            ROOM_L3_DARKNUTS,
                        ):
                            try:
                                path = save_state(
                                    env, GAME_DIR, GAME, f"L3Room_{dest_r:02X}"
                                )
                                reports[f"saved_0x{dest_r:02x}"] = str(path)
                            except Exception as exc:
                                recon_notes.append(f"save_fail 0x{dest_r:02x}: {exc!r}")
                        moved = True
                        # Boss check on arrival
                        bi = _looks_like_boss(read_snapshot(env.get_ram()))
                        if bi is not None:
                            recon_notes.append(
                                f"boss on arrive 0x{dest_r:02x} type "
                                f"0x{bi['type']:02x}"
                            )
                        break
                    env.em.set_state(st)
                    obs, *_ = env.step(nes_idle_action())

            if not moved:
                trap_notes.append(
                    f"hop{hop_i}: no exit from 0x{room_before:02x} "
                    f"(keys={read_snapshot(env.get_ram()).keys})"
                )
                # If stuck with 0 keys, poke keys for recon mapping.
                if poke_keys is None and read_snapshot(env.get_ram()).keys == 0:
                    recon_notes.append(
                        "stuck keys=0 — RECON poke keys=4 for door map"
                    )
                    _poke_keys(env, 4)
                    poke_keys = 4  # only once auto
                    continue
                break

            # Stop if TF
            if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & LEVEL3_TRIFORCE_BIT:
                reports["tf04"] = True
                recon_notes.append("TF 0x04 set during explore")
                if save_checkpoints:
                    path = save_state(env, GAME_DIR, GAME, "Level3Complete")
                    reports["saved_complete"] = str(path)
                break

        reports["explore_hops"] = hops
        reports["visited"] = [f"0x{r:02x}" for r in sorted(visited)]
        reports["final"] = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        reports["tf04"] = bool(
            reports.get("tf04")
            or (
                reports["final"].get("triforce") is not None
                and reports["final"]["triforce"] & LEVEL3_TRIFORCE_BIT
            )
        )
        reports["ok"] = bool(
            reports.get("passage_exit", {}).get("ok", True)
            or reports["tf04"]
            or reports.get("boss")
            or len(hops) > 0
        )
        reports["total_frames"] = total[0]

        # Summary graph ascii
        edge_lines = []
        for e in reports["graph_edges"]:
            if e.get("to"):
                edge_lines.append(
                    f"{e.get('from')} --{e.get('dir')}--> {e.get('to')} "
                    f"[{e.get('phase','')}]"
                )
        reports["graph_summary"] = edge_lines

        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")
        out = RECORDINGS_DIR / f"{tag}_recon.json"
        write_json_report(out, reports)
        reports["report_path"] = str(out)
        return reports
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level3Raft")
    p.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Survival assist (default on for recon)",
    )
    p.add_argument(
        "--phase",
        choices=("all", "exit-only", "explore", "boss"),
        default="all",
    )
    p.add_argument("--max-hops", type=int, default=14)
    p.add_argument(
        "--poke-bombs",
        type=int,
        default=12,
        help="RECON bomb inventory poke (0=disable). Default 12.",
    )
    p.add_argument(
        "--poke-keys",
        type=int,
        default=None,
        help="RECON keys poke (optional). Auto-pokes 4 if stuck at 0.",
    )
    p.add_argument(
        "--try-bombs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l3_manhandla")
    args = p.parse_args(argv)

    poke_bombs = None if args.poke_bombs == 0 else args.poke_bombs
    rep = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        phase=args.phase,
        max_hops=args.max_hops,
        poke_bombs=poke_bombs,
        poke_keys=args.poke_keys,
        try_bombs=args.try_bombs,
        save_checkpoints=args.save_state,
        tag=args.tag,
    )
    print(
        f"ok={rep.get('ok')} tf04={rep.get('tf04')} boss={rep.get('boss') is not None} "
        f"visited={rep.get('visited')} frames={rep.get('total_frames')} "
        f"report={rep.get('report_path')}"
    )
    if rep.get("graph_summary"):
        print("graph:")
        for line in rep["graph_summary"][:40]:
            print(" ", line)
    if rep.get("recon_notes"):
        print("notes:")
        for n in rep["recon_notes"]:
            print(" ", n)
    if rep.get("trap_notes"):
        print("traps:")
        for n in rep["trap_notes"]:
            print(" ", n)
    if rep.get("error"):
        print("error:", rep["error"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
