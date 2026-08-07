"""Recon: open further exits from L2 room 0x5f past Goriya clear (rr-cjf).

Tests RIGHT / UP / bomb / push-block / diamond-east from:

1. ``Level2_5E`` (post Goriya clear) → RIGHT back to 0x5f → expand
2. ``Level2_5F`` gel clear + map, optional visit 0x5e clear path

Survival track by default (``--infinite-life``). Inventory poke optional for
bombs/keys. Documents LIVE room IDs + sealed negatives for door_graph.

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_5f_exits.py \\
        --infinite-life --from-state Level2_5E --tag l2_5f_exits_5e
    uv run python nes/zelda_i/scripts/probe_level2_5f_exits.py \\
        --infinite-life --from-state Level2_5F --clear-gels --tag l2_5f_exits_5f
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
    GEL_OBJECT_TYPE,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
    ROOM_5E_SPEC,
)
from zelda_i.dungeon_ids import object_name, room_item_name
from zelda_i.nav_common import diamond_east_phase
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOOMERANG,
    ADDR_COMPASS,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAP,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)

LEVEL_2 = 2
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

DOOR_TARGETS: dict[str, tuple[int, int]] = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}

# Dense bomb stands — include verified 0x6f north stand + neighbors.
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

# Probe-local gel clear (not ROOM_5F_SPEC).
_PROBE_5F_CLEAR = DungeonRoomSpec(
    spec_id="level2_room5f_probe_clear_exits",
    source_room=ROOM_6F,
    room_id=ROOM_5F,
    entry=DoorRoute("UP", ((120, 189),)),
    enemy_types=(GEL_OBJECT_TYPE,),
    expected_enemy_count=5,
    alive_rule=AliveRule.TYPE,
    combat=CombatTuning(
        patrol=(
            (120, 141),
            (168, 141),
            (168, 109),
            (120, 109),
            (72, 109),
            (72, 141),
            (72, 173),
            (120, 173),
            (168, 173),
            (120, 141),
        ),
        engage_distance=56,
        patrol_attack_period=8,
        patrol_attack_hold=3,
        engage_attack_period=6,
        engage_attack_hold=3,
    ),
    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
    room_item_id=0x17,
    max_frames=10000,
    level=LEVEL_2,
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
        "triforce": int(snap.triforce),
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
    expect_leave_room: int | None = None,
    diamond_band: int | None = None,
    label: str = "",
) -> dict:
    """Push one door; return to start room if left."""
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    start_keys = snap0.keys
    start_doors = snap0.cur_opened_doors
    start_mask = snap0.open_doorway_mask
    start_xy = [snap0.link_x, snap0.link_y]
    diamond_state = {"phase": "free", "cycle": 0}
    reached = False
    used = 0
    obs = None

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
        obs, *_ = env.step(act)

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

    # Return home if we left.
    if result["ok"] and end_sc != start_sc:
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
) -> dict:
    snap0 = read_snapshot(env.get_ram())
    start_sc = snap0.screen
    bombs_before = snap0.bombs
    doors_before = snap0.cur_opened_doors
    mask_before = snap0.open_doorway_mask
    keys_before = snap0.keys

    # Goto stand.
    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if snap.mode != PLAY_MODE:
            env.step(nes_idle_action())
            continue
        act, at = _goto_xy(snap, sx, sy, tol=4)
        env.step(act)
        if at:
            break

    # Face + place.
    _ensure_bomb_selected(env)
    for _ in range(6):
        env.step(nes_action(face))
    env.step(nes_action(face, "B"))
    _idle(env, 2)
    snap = read_snapshot(env.get_ram())
    bombs_after_place = snap.bombs
    bomb_consumed = bombs_after_place < bombs_before

    # Wait blast.
    for _ in range(wait_blast):
        env.step(nes_action(face) if face in ("UP", "DOWN") else nes_idle_action())

    # Push into wall.
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

    if ok and end_sc != start_sc:
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
        # Nudge in 4 dirs around center.
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


def _clear_gels_5f(env) -> dict:
    controller = GenericDungeonRoomController(_PROBE_5F_CLEAR)
    controller.phase = DungeonPhase.FIGHT
    map_before = read_u8(env.get_ram(), ADDR_MAP)
    doors_before = read_snapshot(env.get_ram()).cur_opened_doors
    frames = 0
    for frames in range(_PROBE_5F_CLEAR.max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        if (
            controller.success
            or controller.phase is DungeonPhase.FAILED
            or controller.phase is DungeonPhase.DONE
        ):
            break
    # Map wander.
    map_got = False
    for wf in range(500):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if read_u8(ram, ADDR_MAP) != map_before and read_u8(ram, ADDR_MAP) != 0:
            map_got = True
            break
        targets = (
            (120, 141),
            (160, 141),
            (80, 141),
            (120, 109),
            (120, 173),
            (168, 109),
            (72, 173),
        )
        tx, ty = targets[wf // 50 % len(targets)]
        act, _ = _goto_xy(snap, tx, ty, tol=6)
        env.step(act)
    snap = read_snapshot(env.get_ram())
    map_after = read_u8(env.get_ram(), ADDR_MAP)
    return {
        **controller.report(),
        "frames": frames + 1,
        "doors_before": doors_before,
        "doors_after": snap.cur_opened_doors,
        "clear_opened_new_doors": bool(
            (snap.cur_opened_doors & ~doors_before) != 0
        ),
        "map_before": map_before,
        "map_after": map_after,
        "map_gained": map_got or (map_after != map_before and map_after != 0),
    }


def _clear_goriya_5e(env) -> dict:
    controller = GenericDungeonRoomController(ROOM_5E_SPEC)
    controller.phase = DungeonPhase.FIGHT
    frames = 0
    for frames in range(ROOM_5E_SPEC.max_frames):
        action = controller.step(read_snapshot(env.get_ram()))
        env.step(action.action)
        if (
            controller.success
            or controller.phase is DungeonPhase.FAILED
            or controller.phase is DungeonPhase.DONE
        ):
            break
    return {**controller.report(), "frames": frames + 1}


def _enter_key_left_5f_to_5e(env) -> bool:
    for _ in range(400):
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_5E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(nes_action("LEFT") if s.transitioning else nes_idle_action())
            continue
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            break
    for _ in range(600):
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_5E and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(nes_action("LEFT"))
            continue
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("LEFT"))
    s = read_snapshot(env.get_ram())
    return s.screen == ROOM_5E and s.mode == PLAY_MODE


def _return_right_5e_to_5f(env) -> bool:
    for _ in range(600):
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_5F and s.mode == PLAY_MODE:
            return True
        if s.mode != PLAY_MODE:
            env.step(nes_action("RIGHT") if s.transitioning or s.mode in (6, 7) else nes_idle_action())
            continue
        if abs(s.link_y - 141) > 4:
            env.step(nes_action("DOWN" if s.link_y < 141 else "UP"))
        else:
            env.step(nes_action("RIGHT"))
    s = read_snapshot(env.get_ram())
    return s.screen == ROOM_5F and s.mode == PLAY_MODE


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
) -> dict:
    """On ``room``: door cycle, bomb stands, push blocks. Collect LIVE edges."""
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

    door_tests: list[dict] = []
    # Door order: residual first, then known.
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
            # Keep probing other stands after return; live edge already recorded.

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


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    poke_bombs: int | None,
    poke_keys: int | None,
    clear_gels: bool,
    visit_goriya: bool,
    do_push: bool,
    do_diamond: bool,
    door_budget: int,
    bomb_limit: int | None,
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
        clear_5f_report = None
        clear_5e_report = None
        all_live_edges: list[dict] = []

        # --- Phase A: if start on 0x5e, expand 0x5e then return to 0x5f ---
        if snap.screen == ROOM_5E and snap.mode == PLAY_MODE:
            # Goriya room residual exits (UP/LEFT/DOWN/bomb).
            exp_5e = _expand_room(
                env,
                room=ROOM_5E,
                tag=tag,
                phase="on_5e_postclear",
                door_budget=door_budget,
                bomb_stands=BOMB_STANDS,
                do_push=do_push,
                do_diamond=do_diamond,
                bomb_limit=bomb_limit,
            )
            phases.append(exp_5e)
            all_live_edges.extend(exp_5e.get("live_edges") or [])
            timeline.append(
                _sample(
                    read_snapshot(env.get_ram()),
                    env.get_ram(),
                    event="after_5e_expand",
                )
            )
            if not _return_right_5e_to_5f(env):
                return {
                    "ok": False,
                    "bead": "rr-cjf",
                    "error": "failed_return_5e_to_5f",
                    "entry": entry,
                    "phases": phases,
                    "start_state": start_state,
                }
            timeline.append(
                _sample(
                    read_snapshot(env.get_ram()),
                    env.get_ram(),
                    event="returned_5f_after_goriya",
                )
            )
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_returned_5f.png")

        # --- Phase B: optional gel clear + map on 0x5f ---
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_5F and clear_gels and _live_enemies(snap):
            clear_5f_report = _clear_gels_5f(env)
            timeline.append(
                _sample(
                    read_snapshot(env.get_ram()),
                    env.get_ram(),
                    event="after_gel_clear",
                )
            )
            if assist is not None:
                assist.apply_env(env, frame=1000)

        # --- Phase C: expand 0x5f (main residual target) ---
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_5F and snap.mode == PLAY_MODE:
            exp_5f = _expand_room(
                env,
                room=ROOM_5F,
                tag=tag,
                phase="on_5f",
                door_budget=door_budget,
                bomb_stands=BOMB_STANDS,
                do_push=do_push,
                do_diamond=do_diamond,
                bomb_limit=bomb_limit,
            )
            phases.append(exp_5f)
            all_live_edges.extend(exp_5f.get("live_edges") or [])
        else:
            phases.append(
                {
                    "phase": "on_5f",
                    "ok": False,
                    "error": f"not_on_0x5f sc=0x{snap.screen:02x}",
                    "sample": _sample(snap, env.get_ram(), event="5f_missing"),
                }
            )

        # --- Phase D: optional Goriya visit from 0x5f (kill-all quirk recheck) ---
        snap = read_snapshot(env.get_ram())
        if (
            visit_goriya
            and snap.screen == ROOM_5F
            and snap.keys >= 1
            and snap.mode == PLAY_MODE
        ):
            # Snapshot doors before leaving.
            doors_before_goriya_trip = snap.cur_opened_doors
            if _enter_key_left_5f_to_5e(env):
                timeline.append(
                    _sample(
                        read_snapshot(env.get_ram()),
                        env.get_ram(),
                        event="entered_5e_for_quirk",
                    )
                )
                live = _live_enemies(read_snapshot(env.get_ram()))
                if any(o.type_id == 0x06 for o in live):
                    clear_5e_report = _clear_goriya_5e(env)
                else:
                    clear_5e_report = {
                        "success": True,
                        "already_clear": True,
                        "live": len(live),
                    }
                # Expand 0x5e post-clear if we haven't.
                if not any(p.get("phase") == "on_5e_postclear" for p in phases):
                    exp_5e2 = _expand_room(
                        env,
                        room=ROOM_5E,
                        tag=tag,
                        phase="on_5e_after_clear",
                        door_budget=door_budget,
                        bomb_stands=BOMB_STANDS,
                        do_push=do_push,
                        do_diamond=False,
                        bomb_limit=bomb_limit,
                    )
                    phases.append(exp_5e2)
                    all_live_edges.extend(exp_5e2.get("live_edges") or [])
                if _return_right_5e_to_5f(env):
                    snap = read_snapshot(env.get_ram())
                    doors_after = snap.cur_opened_doors
                    timeline.append(
                        _sample(snap, env.get_ram(), event="reentered_5f_post_goriya")
                    )
                    # Re-test residual doors only (R/U).
                    recheck = {
                        "doors_before_trip": doors_before_goriya_trip,
                        "doors_after_return": doors_after,
                        "doors_new_bits": int(doors_after & ~doors_before_goriya_trip),
                        "door_tests": [],
                    }
                    for direction in ("RIGHT", "UP"):
                        recheck["door_tests"].append(
                            _try_door(
                                env,
                                direction,
                                budget=door_budget,
                                label=f"recheck_{direction}",
                            )
                        )
                    # Bomb UP dense if still sealed.
                    if not any(t.get("ok") for t in recheck["door_tests"]):
                        for face, sx, sy in BOMB_STANDS[:6]:
                            if read_snapshot(env.get_ram()).bombs <= 0:
                                break
                            bt = _try_bomb(env, face, sx, sy)
                            recheck.setdefault("bomb_tests", []).append(bt)
                            if bt["ok"]:
                                all_live_edges.append(
                                    {
                                        "kind": "bomb",
                                        "dir": bt["face"],
                                        "from": bt["start_sc"],
                                        "to": bt["end_sc"],
                                        "stand": bt.get("stand"),
                                        "label": "recheck_post_goriya",
                                    }
                                )
                                break
                    for t in recheck["door_tests"]:
                        if t.get("ok"):
                            all_live_edges.append(
                                {
                                    "kind": "door",
                                    "dir": t["dir"],
                                    "from": t["start_sc"],
                                    "to": t["end_sc"],
                                    "keys_consumed": t.get("keys_consumed"),
                                    "label": t.get("label"),
                                }
                            )
                    phases.append(
                        {
                            "phase": "recheck_5f_post_goriya",
                            "ok": True,
                            **recheck,
                        }
                    )
            else:
                phases.append(
                    {
                        "phase": "visit_goriya",
                        "ok": False,
                        "error": "key_left_failed",
                    }
                )

        final = _sample(read_snapshot(env.get_ram()), env.get_ram(), event="final")
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")

        # Dedup live edges by (kind, dir, from, to).
        seen = set()
        uniq_edges = []
        for e in all_live_edges:
            key = (e.get("kind"), e.get("dir"), e.get("from"), e.get("to"))
            if key in seen:
                continue
            seen.add(key)
            uniq_edges.append(e)

        new_beyond = [
            e
            for e in uniq_edges
            if e.get("to")
            not in ("0x5e", "0x5f", "0x6f", None)
            or (
                e.get("from") == "0x5f"
                and e.get("to") not in ("0x5e", "0x6f")
            )
            or (
                e.get("from") == "0x5e"
                and e.get("to") not in ("0x5f",)
            )
        ]

        report = {
            "ok": True,
            "bead": "rr-cjf",
            "segment": "level2_0x5f_further_exits",
            "start_state": start_state,
            "intervention_class": track,
            "poke_notes": poke_notes,
            "clear_gels": clear_gels,
            "visit_goriya": visit_goriya,
            "entry": entry,
            "clear_5f": clear_5f_report,
            "clear_5e": clear_5e_report,
            "phases": phases,
            "live_edges": uniq_edges,
            "new_rooms_beyond_5e_5f_6f": new_beyond,
            "found_new_room": len(new_beyond) > 0,
            "final": final,
            "timeline": timeline,
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
    parser.add_argument("--from-state", default="Level2_5E")
    parser.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Survival infinite-life (default on for recon)",
    )
    parser.add_argument("--poke-bombs", type=int, default=None)
    parser.add_argument("--poke-keys", type=int, default=None)
    parser.add_argument(
        "--clear-gels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear 0x5f gels + try map before exit cycle (default on)",
    )
    parser.add_argument(
        "--visit-goriya",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="From 0x5f, key-LEFT clear 0x5e and recheck 0x5f doors (default off; "
        "Level2_5E already post-clear)",
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
        help="Try diamond-east RIGHT bands on residual rooms",
    )
    parser.add_argument("--door-budget", type=int, default=240)
    parser.add_argument(
        "--bomb-limit",
        type=int,
        default=None,
        help="Cap bomb stands (default all dense stands)",
    )
    parser.add_argument("--tag", default="l2_5f_exits")
    args = parser.parse_args()

    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        poke_bombs=args.poke_bombs,
        poke_keys=args.poke_keys,
        clear_gels=args.clear_gels,
        visit_goriya=args.visit_goriya,
        do_push=args.push_blocks,
        do_diamond=args.diamond,
        door_budget=args.door_budget,
        bomb_limit=args.bomb_limit,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print(f"wrote {out}")
    print(
        f"ok={report.get('ok')} found_new={report.get('found_new_room')} "
        f"live_edges={report.get('live_edges')} "
        f"new={report.get('new_rooms_beyond_5e_5f_6f')}"
    )
    for p in report.get("phases") or []:
        print(f"  phase={p.get('phase')} ok={p.get('ok')} edges={p.get('live_edges')}")
        for t in p.get("door_tests") or []:
            print(
                f"    door {t.get('label') or t.get('dir')}: ok={t.get('ok')} "
                f"{t.get('start_sc')}→{t.get('end_sc')} "
                f"doors {t.get('doors_before')}→{t.get('doors_after')} "
                f"keys {t.get('keys_before')}→{t.get('keys_after')}"
            )
        for t in p.get("diamond_tests") or []:
            print(
                f"    diamond {t.get('label')}: ok={t.get('ok')} "
                f"{t.get('start_sc')}→{t.get('end_sc')}"
            )
        bombs = p.get("bomb_tests") or []
        ok_b = [t for t in bombs if t.get("ok")]
        print(f"    bombs tried={len(bombs)} ok={len(ok_b)}")
        for t in ok_b:
            print(
                f"      BOMB LIVE {t.get('face')} stand={t.get('stand')} "
                f"{t.get('start_sc')}→{t.get('end_sc')}"
            )
        # Show first few failed bomb that consumed for debug.
        for t in bombs[:4]:
            if not t.get("ok") and not t.get("skipped"):
                print(
                    f"      bomb fail {t.get('face')} stand={t.get('stand')} "
                    f"consumed={t.get('bomb_consumed')} doors "
                    f"{t.get('doors_before')}→{t.get('doors_after')}"
                )
    if report.get("clear_5f"):
        c = report["clear_5f"]
        print(
            f"  clear_5f success={c.get('success')} doors "
            f"{c.get('doors_before')}→{c.get('doors_after')} "
            f"map {c.get('map_before')}→{c.get('map_after')}"
        )
    if not report.get("ok"):
        raise SystemExit(1)
    # Exit 0 even if no new room — recon completed; acceptance is docs/evidence.
    if report.get("found_new_room"):
        print("SUCCESS: new room ID beyond 0x5e/0x5f")
    else:
        print("RECON DONE: no new room ID; see negatives in JSON")


if __name__ == "__main__":
    main()
