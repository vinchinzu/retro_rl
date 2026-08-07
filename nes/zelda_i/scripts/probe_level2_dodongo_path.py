"""Recon: post-boom path 0x4f / 0x3e → Moldorm → Dodongo → TF 0x02 (rr-n5i).

Walkthrough after Magical Boomerang:
  bomb N / Moldorm key → traps+Keese bombs → ropes unlock → Goriya bombs →
  Dodongo (2 mouths) → E → triforce & 0x02.

Strategy: finish-easy geometry under ``--infinite-life``; inventory poke for
bombs/keys. Does **not** claim Clean STATUS.

Default start: ``Level2Boom`` (0x4f, magic boom collected, bombs≈10).

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_dodongo_path.py \\
        --infinite-life --from-state Level2Boom --tag l2_n5i_boom
    uv run python nes/zelda_i/scripts/probe_level2_dodongo_path.py \\
        --infinite-life --from-state Level2_5E --via-3e --tag l2_n5i_3e
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

from retro_harness.env import make_env
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
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    GORIYA_OBJECT_TYPE,
    RewardKind,
    RewardSpec,
    ROPE_OBJECT_TYPE,
)
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.nav_common import diamond_east_phase
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOOMERANG,
    ADDR_COMPASS,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAP,
    ADDR_TRIFORCE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

LEVEL_2 = 2
ROOM_4E = 0x4E
ROOM_4F = 0x4F
ROOM_3E = 0x3E
ROOM_3F = 0x3F
ROOM_5E = 0x5E
ROOM_5F = 0x5F
ROOM_6F = 0x6F

DOOR_RIGHT = 0x01
DOOR_LEFT = 0x02
DOOR_DOWN = 0x04
DOOR_UP = 0x08

ADDR_SELECTED_ITEM = 0x0656
B_ITEM_BOMB = 0x02

TYPE_ONLY = frozenset({0x15, 0x1B})
DROP_TYPES = frozenset({0x60, 0x61, 0x62, 0x63})
# Fireballs / traps / projectiles — not room-clear primary.
IGNORE_TYPES = frozenset({0x55, 0x49, 0x4E})

DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

# Dense bomb stands; prioritize verified (120,101) first.
BOMB_STANDS: list[tuple[str, int, int]] = [
    ("UP", 120, 101),
    ("UP", 120, 97),
    ("UP", 120, 105),
    ("UP", 112, 101),
    ("UP", 128, 101),
    ("UP", 120, 109),
    ("RIGHT", 176, 141),
    ("RIGHT", 184, 141),
    ("RIGHT", 168, 141),
    ("RIGHT", 176, 133),
    ("RIGHT", 176, 149),
    ("LEFT", 64, 141),
    ("LEFT", 56, 141),
    ("LEFT", 72, 141),
    ("DOWN", 120, 173),
    ("DOWN", 120, 181),
    ("DOWN", 120, 165),
]

PUSH_CENTERS: list[tuple[int, int]] = [
    (120, 141),
    (136, 141),
    (104, 141),
    (120, 125),
    (120, 157),
    (152, 141),
    (88, 141),
    (120, 113),
    (120, 169),
]

# Known rooms before this residual (not "new" for acceptance).
KNOWN_ROOMS = frozenset(
    {
        0x7D,
        0x6D,
        0x6C,
        0x7E,
        0x6E,
        0x6F,
        0x5F,
        0x5E,
        0x4E,
        0x4F,
        0x3E,
    }
)


def _door_bits(raw: int) -> dict:
    return {
        "R": bool(raw & DOOR_RIGHT),
        "L": bool(raw & DOOR_LEFT),
        "D": bool(raw & DOOR_DOWN),
        "U": bool(raw & DOOR_UP),
        "raw": int(raw),
    }


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out = []
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
            }
        )
    return out


def _live_enemies(snap: ZeldaSnapshot) -> list:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in DROP_TYPES:
            continue
        if o.type_id in IGNORE_TYPES:
            continue
        if o.type_id in TYPE_ONLY or o.hp > 0:
            out.append(o)
    return out


def _inventory(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "boomerang": read_u8(ram, ADDR_BOOMERANG),
        "magical_boomerang": read_u8(ram, ADDR_MAGIC_BOOMERANG),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "selected": read_u8(ram, ADDR_SELECTED_ITEM),
        "compass": read_u8(ram, ADDR_COMPASS),
        "map": read_u8(ram, ADDR_MAP),
        "sword": int(snap.sword),
        "triforce": int(read_u8(ram, ADDR_TRIFORCE)),
    }


def _sample(snap: ZeldaSnapshot, ram, *, f: int = 0, event: str = "sample") -> dict:
    live = _live_enemies(snap)
    types = Counter(o.type_id for o in live)
    return {
        "f": f,
        "event": event,
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "facing": snap.facing,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "cur_opened_doors": snap.cur_opened_doors,
        "doors": _door_bits(snap.cur_opened_doors),
        "open_doorway_mask": snap.open_doorway_mask,
        "live_enemy_count": len(live),
        "live_type_counts": {f"0x{k:02x}": v for k, v in types.items()},
        "live_type_names": {f"0x{k:02x}": object_name(k) for k in types},
        "objects": _objs(snap),
        "inventory": _inventory(ram),
        "triforce_bit_0x02": bool(read_u8(ram, ADDR_TRIFORCE) & 0x02),
    }


def _poke_inventory(env, *, bombs: int | None, keys: int | None, select_bomb: bool):
    data = env.unwrapped.data
    notes = []
    if bombs is not None:
        data.set_value("bombs", int(bombs) & 0xFF)
        notes.append(f"bombs={bombs}")
    if keys is not None:
        data.set_value("keys", int(keys) & 0xFF)
        notes.append(f"keys={keys}")
    if select_bomb:
        try:
            mem = env.unwrapped.data.memory
            if hasattr(mem, "set_byte"):
                mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            else:
                data.set_value("selected_item", B_ITEM_BOMB)
            notes.append(f"selected=0x{B_ITEM_BOMB:02x}")
        except Exception as exc:
            notes.append(f"selected_fail={exc!r}")
    return notes


def _ensure_bomb_selected(env) -> None:
    ram = env.get_ram()
    if read_u8(ram, ADDR_SELECTED_ITEM) == B_ITEM_BOMB:
        return
    try:
        mem = env.unwrapped.data.memory
        if hasattr(mem, "set_byte"):
            mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            return
    except Exception:
        pass
    try:
        env.unwrapped.data.set_value("selected_item", B_ITEM_BOMB)
    except Exception:
        pass


def _goto_xy(snap: ZeldaSnapshot, tx: int, ty: int, tol: int = 6):
    if abs(snap.link_x - tx) > tol:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT"), False
    if abs(snap.link_y - ty) > tol:
        return nes_action("DOWN" if snap.link_y < ty else "UP"), False
    return nes_idle_action(), True


def _push_door(snap: ZeldaSnapshot, direction: str):
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - ty) > 4:
            return nes_action("DOWN" if snap.link_y < ty else "UP")
        return nes_action(direction)
    if abs(snap.link_x - tx) > 6:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT")
    return nes_action(direction)


def _idle(env, n: int = 1) -> None:
    for _ in range(n):
        env.step(nes_idle_action())


def _drain_scroll(env, hold_dir: str | None = None, budget: int = 120) -> None:
    for _ in range(budget):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning:
            return
        if hold_dir and (snap.transitioning or snap.mode in (6, 7)):
            env.step(nes_action(hold_dir))
        else:
            env.step(nes_idle_action())


def _try_door(
    env,
    direction: str,
    *,
    budget: int = 220,
    diamond_band: int | None = None,
    label: str = "",
    return_home: bool = True,
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    start_keys = snap0.keys
    start_doors = snap0.cur_opened_doors
    start_mask = snap0.open_doorway_mask
    start_xy = [snap0.link_x, snap0.link_y]
    diamond_state = {"phase": "free", "cycle": 0}
    reached = False
    used = 0

    for bf in range(budget):
        used = bf + 1
        snap = read_snapshot(env.get_ram())
        if snap.mode == 17:
            break
        if snap.screen != start_sc and snap.mode in (PLAY_MODE, 6, 7):
            reached = True
            break
        if snap.transitioning or snap.mode in (6, 7):
            act = nes_action(direction)
        elif diamond_band is not None and direction == "RIGHT":
            phase = diamond_state.get("phase", "free")
            cycle = int(diamond_state.get("cycle", 0))
            fa, next_phase = diamond_east_phase(
                snap, phase=phase, band_y=diamond_band, cycle=cycle
            )
            diamond_state["phase"] = next_phase
            diamond_state["cycle"] = cycle + 1
            act = fa.action
        else:
            act = _push_door(snap, direction)
        env.step(act)

    if reached:
        _drain_scroll(env, hold_dir=direction, budget=90)

    snap = read_snapshot(env.get_ram())
    end_sc = snap.screen
    result = {
        "label": label or direction,
        "dir": direction,
        "diamond_band": diamond_band,
        "ok": reached and end_sc != start_sc,
        "frames": used,
        "keys_before": start_keys,
        "keys_after": snap.keys,
        "keys_consumed": int(start_keys) - int(snap.keys),
        "doors_before": start_doors,
        "doors_after": snap.cur_opened_doors,
        "doors_before_bits": _door_bits(start_doors),
        "doors_after_bits": _door_bits(snap.cur_opened_doors),
        "mask_before": start_mask,
        "mask_after": snap.open_doorway_mask,
        "start_sc": f"0x{start_sc:02x}",
        "end_sc": f"0x{end_sc:02x}",
        "start_xy": start_xy,
        "end_xy": [snap.link_x, snap.link_y],
        "end_mode": snap.mode,
        "end_sample": _sample(snap, env.get_ram(), event="door_end"),
    }

    if return_home and result["ok"] and end_sc != start_sc:
        opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[direction]
        for _ in range(budget + 100):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == start_sc:
                break
            if snap.transitioning or snap.mode in (6, 7):
                env.step(nes_action(opp))
            else:
                env.step(_push_door(snap, opp))
        _idle(env, 20)
    return result


def _try_bomb(
    env,
    face: str,
    sx: int,
    sy: int,
    *,
    wait_blast: int = 100,
    push_budget: int = 280,
    return_home: bool = True,
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    bombs_before = snap0.bombs
    doors_before = snap0.cur_opened_doors
    mask_before = snap0.open_doorway_mask
    keys_before = snap0.keys

    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        act, at = _goto_xy(snap, sx, sy, tol=4)
        env.step(act)
        if at:
            break

    _ensure_bomb_selected(env)
    for _ in range(6):
        env.step(nes_action(face))
    env.step(nes_action(face, "B"))
    _idle(env, 2)
    snap = read_snapshot(env.get_ram())
    bombs_after_place = snap.bombs
    bomb_consumed = bombs_after_place < bombs_before

    for _ in range(wait_blast):
        env.step(nes_action(face) if face in ("UP", "DOWN") else nes_idle_action())

    reached = False
    for bf in range(push_budget):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start_sc and snap.mode in (PLAY_MODE, 6, 7):
            reached = True
            break
        if snap.transitioning or snap.mode in (6, 7):
            env.step(nes_action(face))
        else:
            env.step(_push_door(snap, face))

    if reached:
        _drain_scroll(env, hold_dir=face, budget=90)

    snap = read_snapshot(env.get_ram())
    end_sc = snap.screen
    ok = reached and end_sc != start_sc
    result = {
        "face": face,
        "stand": [sx, sy],
        "ok": ok,
        "bomb_consumed": bomb_consumed,
        "bombs_before": bombs_before,
        "bombs_after_place": bombs_after_place,
        "bombs_after": snap.bombs,
        "keys_before": keys_before,
        "keys_after": snap.keys,
        "doors_before": doors_before,
        "doors_after": snap.cur_opened_doors,
        "doors_before_bits": _door_bits(doors_before),
        "doors_after_bits": _door_bits(snap.cur_opened_doors),
        "mask_before": mask_before,
        "mask_after": snap.open_doorway_mask,
        "start_sc": f"0x{start_sc:02x}",
        "end_sc": f"0x{end_sc:02x}",
        "end_xy": [snap.link_x, snap.link_y],
        "end_mode": snap.mode,
        "end_sample": (
            _sample(snap, env.get_ram(), event="bomb_end") if ok else None
        ),
    }

    if return_home and ok and end_sc != start_sc:
        opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[face]
        for _ in range(push_budget + 100):
            snap = read_snapshot(env.get_ram())
            if snap.mode == PLAY_MODE and snap.screen == start_sc:
                break
            if snap.transitioning or snap.mode in (6, 7):
                env.step(nes_action(opp))
            else:
                env.step(_push_door(snap, opp))
        _idle(env, 20)
    return result


def _push_blocks(env, centers: list[tuple[int, int]], *, per_center: int = 80) -> list[dict]:
    results = []
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    doors0 = snap0.cur_opened_doors
    for cx, cy in centers:
        for _ in range(60):
            snap = read_snapshot(env.get_ram())
            act, at = _goto_xy(snap, cx, cy, tol=5)
            env.step(act)
            if at:
                break
        for d in ("UP", "RIGHT", "DOWN", "LEFT"):
            for _ in range(per_center // 4):
                env.step(nes_action(d))
        snap = read_snapshot(env.get_ram())
        results.append(
            {
                "center": [cx, cy],
                "sc": f"0x{snap.screen:02x}",
                "doors_before": doors0,
                "doors_after": snap.cur_opened_doors,
                "doors_changed": snap.cur_opened_doors != doors0,
                "left_room": snap.screen != start_sc,
                "xy": [snap.link_x, snap.link_y],
            }
        )
        if snap.screen != start_sc:
            break
    return results


def _generic_clear(env, enemy_types: tuple[int, ...], *, max_frames: int = 12000) -> dict:
    """Fight whatever is live using a probe-local room spec."""
    snap = read_snapshot(env.get_ram())
    if not enemy_types:
        return {"success": True, "already_clear": True, "frames": 0}
    spec = DungeonRoomSpec(
        spec_id=f"probe_clear_0x{snap.screen:02x}",
        source_room=snap.screen,
        room_id=snap.screen,
        entry=DoorRoute("UP", ((120, 141),)),
        enemy_types=enemy_types,
        expected_enemy_count=max(1, len(_live_enemies(snap))),
        alive_rule=AliveRule.TYPE_AND_HP,
        combat=CombatTuning(
            patrol=(
                (64, 109),
                (120, 109),
                (176, 109),
                (176, 141),
                (176, 173),
                (120, 173),
                (64, 173),
                (64, 141),
                (120, 141),
            ),
            engage_distance=72,
            attack_phase=2,
            patrol_attack_period=8,
            patrol_attack_hold=3,
            engage_attack_period=6,
            engage_attack_hold=3,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        max_frames=max_frames,
        level=LEVEL_2,
    )
    controller = GenericDungeonRoomController(spec)
    controller.phase = DungeonPhase.FIGHT
    frames = 0
    for frames in range(max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        if (
            controller.success
            or controller.phase is DungeonPhase.FAILED
            or controller.phase is DungeonPhase.DONE
        ):
            break
    return {**controller.report(), "frames": frames + 1}


def _expand_room(
    env,
    *,
    room: int,
    tag: str,
    phase: str,
    door_budget: int,
    bomb_stands: list[tuple[str, int, int]],
    do_push: bool,
    do_diamond: bool,
    bomb_limit: int | None,
    clear_first: bool,
) -> dict:
    snap = read_snapshot(env.get_ram())
    if snap.screen != room or snap.mode != PLAY_MODE:
        return {
            "phase": phase,
            "ok": False,
            "error": f"not_on_room sc=0x{snap.screen:02x} mode={snap.mode}",
            "sample": _sample(snap, env.get_ram(), event=f"{phase}_skip"),
        }

    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{phase}_entry.png")
    entry = _sample(read_snapshot(env.get_ram()), env.get_ram(), event=f"{phase}_entry")

    clear_report = None
    if clear_first:
        live = _live_enemies(read_snapshot(env.get_ram()))
        types = tuple(sorted({o.type_id for o in live}))
        if types:
            clear_report = _generic_clear(env, types)
            _idle(env, 30)

    door_tests: list[dict] = []
    for direction in ("RIGHT", "UP", "LEFT", "DOWN"):
        if read_snapshot(env.get_ram()).screen != room:
            break
        door_tests.append(
            _try_door(
                env,
                direction,
                budget=door_budget,
                label=f"{phase}_{direction}",
            )
        )
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{phase}_after_{direction}.png")

    diamond_tests: list[dict] = []
    if do_diamond and read_snapshot(env.get_ram()).screen == room:
        for band in (113, 125, 141, 157, 101):
            diamond_tests.append(
                _try_door(
                    env,
                    "RIGHT",
                    budget=door_budget + 120,
                    diamond_band=band,
                    label=f"{phase}_diamond_R_band{band}",
                )
            )
            if diamond_tests[-1]["ok"]:
                break

    bomb_tests: list[dict] = []
    stands = bomb_stands if bomb_limit is None else bomb_stands[:bomb_limit]
    for face, sx, sy in stands:
        snap = read_snapshot(env.get_ram())
        if snap.screen != room or snap.bombs <= 0:
            bomb_tests.append(
                {
                    "face": face,
                    "stand": [sx, sy],
                    "ok": False,
                    "skipped": True,
                    "reason": f"sc=0x{snap.screen:02x} bombs={snap.bombs}",
                }
            )
            continue
        bt = _try_bomb(env, face, sx, sy)
        bomb_tests.append(bt)
        if bt["ok"]:
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(
                obs,
                RECORDINGS_DIR
                / f"{tag}_{phase}_bomb_{face}_{sx}_{sy}_0x{int(bt['end_sc'], 16):02x}.png",
            )

    push_tests: list[dict] = []
    if do_push and read_snapshot(env.get_ram()).screen == room:
        push_tests = _push_blocks(env, PUSH_CENTERS)

    final = _sample(read_snapshot(env.get_ram()), env.get_ram(), event=f"{phase}_final")
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{phase}_final.png")

    live_edges = []
    for t in door_tests + diamond_tests:
        if t.get("ok"):
            live_edges.append(
                {
                    "kind": "door",
                    "dir": t["dir"],
                    "from": t["start_sc"],
                    "to": t["end_sc"],
                    "keys_consumed": t.get("keys_consumed"),
                    "label": t.get("label"),
                    "diamond_band": t.get("diamond_band"),
                }
            )
    for t in bomb_tests:
        if t.get("ok"):
            live_edges.append(
                {
                    "kind": "bomb",
                    "dir": t["face"],
                    "from": t["start_sc"],
                    "to": t["end_sc"],
                    "stand": t.get("stand"),
                }
            )

    return {
        "phase": phase,
        "ok": True,
        "room": f"0x{room:02x}",
        "entry": entry,
        "clear": clear_report,
        "door_tests": door_tests,
        "diamond_tests": diamond_tests,
        "bomb_tests": bomb_tests,
        "push_tests": push_tests,
        "live_edges": live_edges,
        "final": final,
        "screenshots": {
            "entry": str(RECORDINGS_DIR / f"{tag}_{phase}_entry.png"),
            "final": str(RECORDINGS_DIR / f"{tag}_{phase}_final.png"),
        },
    }


def _enter_dir(env, direction: str, dest: int, *, budget: int = 700) -> bool:
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == dest and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(
                nes_action(direction)
                if s.transitioning or s.mode in (6, 7)
                else nes_idle_action()
            )
            continue
        env.step(_push_door(s, direction))
    s = read_snapshot(env.get_ram())
    return s.screen == dest and s.mode == PLAY_MODE


def _commit_bomb_enter(
    env, face: str, sx: int, sy: int, *, dest: int | None = None
) -> dict:
    """Place bomb and stay in the opened room (no return)."""
    bt = _try_bomb(env, face, sx, sy, return_home=False)
    snap = read_snapshot(env.get_ram())
    bt["committed_sc"] = f"0x{snap.screen:02x}"
    bt["committed_ok"] = (
        bt.get("ok")
        and (dest is None or snap.screen == dest)
        and snap.mode == PLAY_MODE
    )
    return bt


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    poke_bombs: int | None,
    poke_keys: int | None,
    via_3e: bool,
    do_push: bool,
    do_diamond: bool,
    door_budget: int,
    bomb_limit: int | None,
    max_depth: int,
    tag: str,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"

    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        poke_notes = _poke_inventory(
            env,
            bombs=poke_bombs,
            keys=poke_keys,
            select_bomb=True,
        )
        if assist is not None:
            assist.apply_env(env, frame=0)
        _idle(env, 20)
        if assist is not None:
            assist.apply_env(env, frame=20)

        snap = read_snapshot(env.get_ram())
        entry = _sample(snap, env.get_ram(), f=0, event="boot")
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t0.png")

        phases: list[dict] = []
        timeline: list[dict] = [entry]
        all_live_edges: list[dict] = []
        visited_rooms: list[str] = [f"0x{snap.screen:02x}"]
        triforce_02 = bool(read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02)

        # Optional: Level2_5E → UP 0x4e → UP 0x3e expand first.
        if via_3e and snap.screen == ROOM_5E:
            if _enter_dir(env, "UP", ROOM_4E):
                timeline.append(
                    _sample(
                        read_snapshot(env.get_ram()),
                        env.get_ram(),
                        event="entered_4e",
                    )
                )
                # Clear ropes if present then expand, then go UP to 0x3e.
                live = _live_enemies(read_snapshot(env.get_ram()))
                if any(o.type_id == ROPE_OBJECT_TYPE for o in live):
                    cr = _generic_clear(env, (ROPE_OBJECT_TYPE,))
                    phases.append({"phase": "clear_4e", "ok": True, "clear": cr})
                if _enter_dir(env, "UP", ROOM_3E):
                    timeline.append(
                        _sample(
                            read_snapshot(env.get_ram()),
                            env.get_ram(),
                            event="entered_3e",
                        )
                    )
                    visited_rooms.append("0x3e")
                    exp = _expand_room(
                        env,
                        room=ROOM_3E,
                        tag=tag,
                        phase="on_3e",
                        door_budget=door_budget,
                        bomb_stands=BOMB_STANDS,
                        do_push=do_push,
                        do_diamond=do_diamond,
                        bomb_limit=bomb_limit,
                        clear_first=True,
                    )
                    phases.append(exp)
                    all_live_edges.extend(exp.get("live_edges") or [])
                    # Return to 0x4e then try RIGHT to 0x4f for boom path.
                    _enter_dir(env, "DOWN", ROOM_4E)
                    if read_snapshot(env.get_ram()).keys >= 1:
                        _enter_dir(env, "RIGHT", ROOM_4F)

        # Main: expand current room if 0x4f, or navigate to 0x4f.
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_4F and snap.screen == ROOM_4E:
            if snap.keys >= 1:
                _enter_dir(env, "RIGHT", ROOM_4F)
            else:
                # Try free RIGHT in case already open.
                _enter_dir(env, "RIGHT", ROOM_4F)

        def _settle_play(budget: int = 180) -> None:
            for _ in range(budget):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and not s.transitioning:
                    return
                env.step(nes_idle_action())

        def _expand_here(
            room: int,
            phase: str,
            *,
            bomb_stands: list[tuple[str, int, int]] | None = None,
            push: bool = False,
            diamond: bool = False,
            blimit: int | None = 6,
        ) -> dict:
            _settle_play()
            if assist is not None:
                assist.apply_env(env, frame=len(phases) * 500)
            exp = _expand_room(
                env,
                room=room,
                tag=tag,
                phase=phase,
                door_budget=door_budget,
                bomb_stands=bomb_stands if bomb_stands is not None else BOMB_STANDS,
                do_push=push,
                do_diamond=diamond,
                bomb_limit=blimit if blimit is not None else bomb_limit,
                clear_first=True,
            )
            phases.append(exp)
            all_live_edges.extend(exp.get("live_edges") or [])
            visited_rooms.append(f"0x{room:02x}")
            return exp

        def _prefer_residual_edge(cur: int) -> dict | None:
            """Pick north/east residual edge; skip known south graph."""
            skip_dest = {ROOM_5F, ROOM_4E, ROOM_5E, ROOM_6F, ROOM_4F}
            candidates = []
            for e in all_live_edges:
                if e.get("from") != f"0x{cur:02x}":
                    continue
                dest = int(e["to"], 16)
                if dest == cur or dest in skip_dest:
                    continue
                score = 0
                if e.get("dir") == "UP":
                    score += 30
                if e.get("dir") == "RIGHT":
                    score += 20
                if e.get("kind") == "bomb":
                    score += 10
                if dest not in KNOWN_ROOMS:
                    score += 50
                candidates.append((score, e))
            if not candidates:
                return None
            candidates.sort(key=lambda x: -x[0])
            return candidates[0][1]

        def _commit_edge(edge: dict, cur: int) -> int | None:
            """Enter edge destination; return room id or None."""
            if edge.get("kind") == "bomb":
                stand = edge.get("stand") or [120, 101]
                # If wall already open, walk may suffice; try walk first for UP/RIGHT.
                d = edge["dir"]
                walk = _try_door(
                    env, d, budget=door_budget, return_home=False, label=f"walk_{d}"
                )
                phases.append({"phase": f"walk_after_bomb_{d}", "door": walk})
                if walk.get("ok"):
                    return int(walk["end_sc"], 16)
                bt = _commit_bomb_enter(env, d, int(stand[0]), int(stand[1]))
                phases.append({"phase": f"commit_bomb_{d}", "edge": edge, "bomb": bt})
                if bt.get("ok"):
                    return int(bt["end_sc"], 16)
                return None
            d = edge["dir"]
            tr = _try_door(
                env, d, budget=door_budget, return_home=False, label=f"commit_{d}"
            )
            phases.append({"phase": f"commit_door_{d}", "edge": edge, "door": tr})
            if tr.get("ok"):
                return int(tr["end_sc"], 16)
            return None

        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_4F and snap.mode == PLAY_MODE:
            # Expand boom room: doors + bomb stands; skip push (open bomb wall
            # makes UP nudge leave the room mid-test).
            exp_4f = _expand_here(
                ROOM_4F,
                "on_4f",
                bomb_stands=BOMB_STANDS,
                push=False,
                diamond=do_diamond,
                blimit=bomb_limit if bomb_limit is not None else 6,
            )
        else:
            phases.append(
                {
                    "phase": "on_4f",
                    "ok": False,
                    "error": f"not_on_0x4f sc=0x{snap.screen:02x}",
                    "sample": _sample(snap, env.get_ram(), event="4f_missing"),
                }
            )

        # Explicit first hop: bomb/walk UP 0x4f → residual (expect 0x3f Moldorm).
        _settle_play()
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_4F and snap.mode == PLAY_MODE:
            # Prefer recorded bomb-UP edge; else force stand (120,101).
            edge = _prefer_residual_edge(ROOM_4F)
            if edge is None or edge.get("dir") != "UP":
                edge = {
                    "kind": "bomb",
                    "dir": "UP",
                    "from": "0x4f",
                    "to": "0x3f",
                    "stand": [120, 101],
                }
            dest = _commit_edge(edge, ROOM_4F)
            _settle_play()
            if dest is not None:
                all_live_edges.append(
                    {
                        "kind": edge.get("kind", "door"),
                        "dir": edge.get("dir"),
                        "from": "0x4f",
                        "to": f"0x{dest:02x}",
                        "stand": edge.get("stand"),
                    }
                )
                # Wait for enemy spawn (Moldorms often lag).
                for _ in range(180):
                    if assist is not None:
                        assist.apply_env(env, frame=2000)
                    env.step(nes_idle_action())
                exp_n = _expand_here(
                    dest,
                    f"on_0x{dest:02x}",
                    bomb_stands=BOMB_STANDS,
                    push=False,
                    diamond=do_diamond,
                    blimit=bomb_limit if bomb_limit is not None else 6,
                )
            else:
                phases.append(
                    {
                        "phase": "enter_north_of_4f",
                        "ok": False,
                        "error": "bomb_or_walk_UP_failed",
                    }
                )

        # Depth-limited residual walk: prefer UP/RIGHT (and bomb variants).
        depth = 0
        while depth < max_depth:
            depth += 1
            _settle_play()
            snap = read_snapshot(env.get_ram())
            if snap.mode != PLAY_MODE or snap.level != LEVEL_2:
                phases.append(
                    {
                        "phase": f"depth_{depth}_bail",
                        "mode": snap.mode,
                        "sc": f"0x{snap.screen:02x}",
                    }
                )
                break
            cur = snap.screen
            # Already expanded this residual room?
            already = any(
                p.get("room") == f"0x{cur:02x}" and p.get("ok") for p in phases
            )
            if not already and cur not in (ROOM_5F, ROOM_4E, ROOM_5E, ROOM_4F):
                for _ in range(120):
                    env.step(nes_idle_action())
                _expand_here(
                    cur,
                    f"on_0x{cur:02x}_d{depth}",
                    push=False,
                    diamond=do_diamond,
                    blimit=4,
                )

            edge = _prefer_residual_edge(cur)
            if edge is None:
                # Force bomb-UP then walk UP/RIGHT.
                forced = None
                if snap.bombs > 0:
                    bt = _commit_bomb_enter(env, "UP", 120, 101)
                    phases.append({"phase": f"force_bomb_up_d{depth}", "bomb": bt})
                    if bt.get("ok"):
                        forced = int(bt["end_sc"], 16)
                        all_live_edges.append(
                            {
                                "kind": "bomb",
                                "dir": "UP",
                                "from": f"0x{cur:02x}",
                                "to": f"0x{forced:02x}",
                                "stand": [120, 101],
                            }
                        )
                if forced is None:
                    for d in ("UP", "RIGHT"):
                        tr = _try_door(
                            env,
                            d,
                            budget=door_budget,
                            return_home=False,
                            label=f"force_{d}_d{depth}",
                        )
                        phases.append({"phase": f"force_door_{d}_d{depth}", "door": tr})
                        if tr.get("ok"):
                            forced = int(tr["end_sc"], 16)
                            if forced in (ROOM_5F, ROOM_4E, ROOM_5E) and d != "RIGHT":
                                # Back out south into known graph.
                                opp = "DOWN" if d == "UP" else "LEFT"
                                _enter_dir(env, opp, cur)
                                forced = None
                                continue
                            all_live_edges.append(
                                {
                                    "kind": "door",
                                    "dir": d,
                                    "from": f"0x{cur:02x}",
                                    "to": f"0x{forced:02x}",
                                }
                            )
                            break
                if forced is None:
                    break
                dest = forced
            else:
                dest = _commit_edge(edge, cur)
                if dest is None:
                    break

            _settle_play()
            if dest in (ROOM_5F, ROOM_4E, ROOM_5E):
                break
            # Spawn settle + expand new room if needed.
            for _ in range(150):
                if assist is not None:
                    assist.apply_env(env, frame=3000 + depth * 200)
                env.step(nes_idle_action())
            if not any(p.get("room") == f"0x{dest:02x}" and p.get("ok") for p in phases):
                _expand_here(
                    dest,
                    f"on_0x{dest:02x}_d{depth}",
                    push=False,
                    diamond=do_diamond,
                    blimit=4,
                )
            triforce_02 = bool(read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02)
            if triforce_02:
                break

        final_snap = read_snapshot(env.get_ram())
        final = _sample(final_snap, env.get_ram(), event="final")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")

        new_rooms = sorted(
            {
                e["to"]
                for e in all_live_edges
                if int(e["to"], 16) not in KNOWN_ROOMS
            }
        )
        residual_edges = [
            e
            for e in all_live_edges
            if int(e.get("to", "0x0"), 16) not in KNOWN_ROOMS
            or (
                e.get("from") in ("0x4f", "0x3e")
                and e.get("dir") in ("UP", "RIGHT")
            )
        ]

        report = {
            "ok": True,
            "bead": "rr-n5i",
            "track": track,
            "start_state": start_state,
            "poke_notes": poke_notes,
            "entry": entry,
            "phases": phases,
            "timeline": timeline,
            "live_edges": all_live_edges,
            "residual_edges": residual_edges,
            "visited_rooms": visited_rooms,
            "new_rooms": new_rooms,
            "found_new_room": bool(new_rooms),
            "triforce_bit_0x02": triforce_02
            or bool(read_u8(env.get_ram(), ADDR_TRIFORCE) & 0x02),
            "final": final,
            "screenshots": {
                "t0": str(RECORDINGS_DIR / f"{tag}_t0.png"),
                "final": str(RECORDINGS_DIR / f"{tag}_final.png"),
            },
        }
        return report
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level2Boom")
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Survival assist (default on for first-pass geometry)",
    )
    parser.add_argument("--poke-bombs", type=int, default=None)
    parser.add_argument("--poke-keys", type=int, default=None)
    parser.add_argument(
        "--via-3e",
        action="store_true",
        help="From Level2_5E go UP→0x4e UP→0x3e expand first",
    )
    parser.add_argument(
        "--push-blocks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diamond",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--door-budget", type=int, default=240)
    parser.add_argument("--bomb-limit", type=int, default=None)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="How many residual rooms to commit into after boom",
    )
    parser.add_argument("--tag", default="l2_n5i_dodongo")
    args = parser.parse_args()

    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        poke_bombs=args.poke_bombs,
        poke_keys=args.poke_keys,
        via_3e=args.via_3e,
        do_push=args.push_blocks,
        do_diamond=args.diamond,
        door_budget=args.door_budget,
        bomb_limit=args.bomb_limit,
        max_depth=args.max_depth,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print(f"wrote {out}")
    print(
        f"ok={report.get('ok')} found_new={report.get('found_new_room')} "
        f"new={report.get('new_rooms')} tf02={report.get('triforce_bit_0x02')} "
        f"visited={report.get('visited_rooms')}"
    )
    print(f"live_edges={report.get('live_edges')}")
    print(f"residual_edges={report.get('residual_edges')}")
    for p in report.get("phases") or []:
        print(f"  phase={p.get('phase')} ok={p.get('ok')} edges={p.get('live_edges')}")
        for t in p.get("door_tests") or []:
            print(
                f"    door {t.get('label') or t.get('dir')}: ok={t.get('ok')} "
                f"{t.get('start_sc')}→{t.get('end_sc')} "
                f"doors {t.get('doors_before')}→{t.get('doors_after')} "
                f"keys {t.get('keys_before')}→{t.get('keys_after')}"
            )
        ok_b = [t for t in (p.get("bomb_tests") or []) if t.get("ok")]
        print(f"    bombs ok={len(ok_b)}/{len(p.get('bomb_tests') or [])}")
        for t in ok_b:
            print(
                f"      BOMB LIVE {t.get('face')} stand={t.get('stand')} "
                f"{t.get('start_sc')}→{t.get('end_sc')}"
            )
        if p.get("clear"):
            c = p["clear"]
            print(
                f"    clear success={c.get('success')} frames={c.get('frames')} "
                f"notes={c.get('notes')}"
            )
        if p.get("entry"):
            e = p["entry"]
            print(
                f"    entry sc={e.get('sc')} live={e.get('live_type_counts')} "
                f"item={e.get('room_item_id')} doors={e.get('cur_opened_doors')} "
                f"xy={e.get('xy')}"
            )
    if not report.get("ok"):
        raise SystemExit(1)
    if report.get("triforce_bit_0x02"):
        print("SUCCESS: triforce & 0x02 set")
    elif report.get("found_new_room"):
        print("PROGRESS: new room(s) past known graph — see JSON")
    else:
        print("RECON DONE: no new room; see negatives / residual in JSON")


if __name__ == "__main__":
    main()
