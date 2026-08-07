"""Assisted recon: open exits past Level 2 compass room 0x6f.

Maps bomb walls, key doors, diamond-block pushes, and new room IDs after
the live key-RIGHT 0x6e→0x6f edge. Survival health + optional inventory poke
for bombs/keys (documented; not Clean STATUS).

Walkthrough after compass: bomb N shortcut → Red Goriya → Map → boom room
path → Dodongo. Goal: measurable graph expansion beyond 0x6f.

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_past_6f.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level2_past_6f.py \\
        --infinite-life --poke-bombs 8 --poke-keys 3 --tag l2_past6f
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
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
ADDR_SELECTED_ITEM = 0x0656
# LoZ NES B-slot: 0x02 = bomb (Data Crystal + live poke trials).
B_ITEM_BOMB = 0x02

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

# Bomb placement standing spots (just inside wall centers).
BOMB_STAND: dict[str, tuple[int, int, str]] = {
    "UP": (120, 109, "UP"),
    "DOWN": (120, 173, "DOWN"),
    "LEFT": (64, 141, "LEFT"),
    "RIGHT": (176, 141, "RIGHT"),
}

DIAMOND_EAST_ROOMS: dict[int, int] = {
    0x7D: 157,
    0x6E: 113,
    0x6F: 113,
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


def _room_fields(snap: ZeldaSnapshot, ram=None) -> dict:
    types = Counter(
        o.type_id
        for o in snap.objects
        if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
    )
    fields = {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "sc": f"0x{snap.screen:02x}",
        "x": snap.link_x,
        "y": snap.link_y,
        "xy": [snap.link_x, snap.link_y],
        "facing": snap.facing,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "cur_opened_doors": snap.cur_opened_doors,
        "open_doorway_mask": snap.open_doorway_mask,
        "doors": {
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "type_counts": {f"0x{k:02x}": v for k, v in types.items()},
        "type_names": {f"0x{k:02x}": object_name(k) for k in types},
        "objects": _objs(snap),
    }
    if ram is not None:
        fields["inventory"] = _inventory(ram)
    return fields


def _live_enemies(snap: ZeldaSnapshot, *, type_only: frozenset[int] | None = None):
    """Combat targets. Gels/Keese are TYPE-only (hp=0 while alive)."""
    type_only = type_only or frozenset({0x15, 0x1B})
    drop = {0x60, 0x61, 0x62, 0x63}
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in drop:
            continue
        if o.type_id in type_only:
            out.append(o)
        elif o.hp > 0:
            out.append(o)
    return out


def _swing(frames: int, direction: str, *, period: int = 8, hold: int = 3):
    if frames % period < hold:
        return nes_action(direction, "A")
    return nes_action(direction)


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


def _diamond_or_door(snap, direction: str, room: int | None, phase_state: dict):
    if direction == "RIGHT" and room is not None and room in DIAMOND_EAST_ROOMS:
        band = DIAMOND_EAST_ROOMS[room]
        phase = phase_state.get("phase", "free")
        cycle = int(phase_state.get("cycle", 0))
        act, next_phase = diamond_east_phase(
            snap, phase=phase, band_y=band, cycle=cycle
        )
        phase_state["phase"] = next_phase
        phase_state["cycle"] = cycle + 1
        return act.action
    phase_state["phase"] = "free"
    phase_state["cycle"] = 0
    # Center slightly before vertical doors.
    if direction in ("UP", "DOWN"):
        cx = 120
        if abs(snap.link_x - cx) > 10:
            return nes_action("RIGHT" if snap.link_x < cx else "LEFT")
    return _push_door(snap, direction)


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
        # Not in data.json — direct WRAM via emulator memory when possible.
        try:
            # stable-retro / fceumm: set_value only for mapped keys; raw poke:
            mem = env.unwrapped.data.memory
            if hasattr(mem, "set_byte"):
                mem.set_byte(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            elif hasattr(env.unwrapped, "set_ram"):
                env.unwrapped.set_ram(ADDR_SELECTED_ITEM, B_ITEM_BOMB)
            else:
                # Fallback: try data key if present.
                try:
                    data.set_value("selected_item", B_ITEM_BOMB)
                except Exception:
                    pass
            notes.append(f"selected=0x{B_ITEM_BOMB:02x}")
        except Exception as exc:
            notes.append(f"selected_fail={exc!r}")
    return notes


def _ensure_bomb_selected(env, ram) -> None:
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


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    poke_bombs: int | None,
    poke_keys: int | None,
    max_frames: int,
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
            select_bomb=poke_bombs is not None and poke_bombs > 0,
        )
        if assist is not None:
            assist.apply_env(env, frame=0)

        snap = read_snapshot(env.get_ram())
        entry = _room_fields(snap, env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t0.png")

        # Phase machine: navigate to 0x6f then expand.
        # Stages: to_7e_key | to_6e_west | open_6f | clear_6f | expand
        stage = "bootstrap"
        stage_frames = 0
        diamond_state: dict = {"phase": "free", "cycle": 0}
        fight_frames = 0
        door_push = 0
        current_door = "RIGHT"
        expand_phase = "loot"  # loot → bomb_cycle → door_cycle → push_blocks → wander
        bomb_dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
        bomb_idx = 0
        bomb_sub = "goto"  # goto → face → place → wait → push
        bomb_wait = 0
        door_dirs = ["RIGHT", "UP", "DOWN", "LEFT"]
        door_idx = 0
        door_budget = 0
        push_cycle = 0
        wander_step = 0

        transitions: list[dict] = []
        rooms: dict[str, dict] = {}
        bomb_tests: list[dict] = []
        door_tests: list[dict] = []
        timeline: list[dict] = []
        graph: dict[str, dict[str, str | None]] = {}
        visited: list[int] = []
        play_room: int | None = snap.screen if snap.mode == PLAY_MODE else None
        if play_room is not None:
            visited.append(play_room)
            rooms[f"0x{play_room:02x}"] = _room_fields(snap, env.get_ram())

        keys_enter_6e: int | None = None
        reached_6f = False
        compass_inv = False
        boom_got = False
        frames_run = 0

        def mark_room(label: str) -> None:
            nonlocal obs
            s = read_snapshot(env.get_ram())
            key = f"0x{s.screen:02x}:{label}"
            rooms[key] = {**_room_fields(s, env.get_ram()), "label": label}
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{label}_0x{s.screen:02x}.png")

        def note_transition(prev: int | None, new: int, direction: str | None) -> None:
            s = read_snapshot(env.get_ram())
            transitions.append(
                {
                    "f": frames_run,
                    "from": None if prev is None else f"0x{prev:02x}",
                    "to": f"0x{new:02x}",
                    "dir": direction,
                    "keys": s.keys,
                    "bombs": s.bombs,
                    "doors": s.cur_opened_doors,
                    "xy": [s.link_x, s.link_y],
                    "item": s.room_item_id,
                    "types": {
                        f"0x{t:02x}": object_name(t)
                        for t in {
                            o.type_id
                            for o in s.objects
                            if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
                        }
                    },
                }
            )
            if prev is not None and direction:
                g = graph.setdefault(f"0x{prev:02x}", {})
                g[direction] = f"0x{new:02x}"
            if new not in visited:
                visited.append(new)
            rooms[f"0x{new:02x}"] = _room_fields(s, env.get_ram())
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_room_0x{new:02x}.png")

        for f in range(max_frames):
            frames_run = f + 1
            ram = env.get_ram()
            snap = read_snapshot(ram)
            inv = _inventory(ram)
            stage_frames += 1

            if snap.mode == 17:
                timeline.append({"f": f, "event": "death", **_room_fields(snap, ram)})
                break
            if snap.level != LEVEL_2:
                timeline.append(
                    {"f": f, "event": "left_level2", **_room_fields(snap, ram)}
                )
                break
            if inv["magical_boomerang"]:
                boom_got = True
                timeline.append(
                    {"f": f, "event": "magical_boomerang", **_room_fields(snap, ram)}
                )
                mark_room("mboom")
                break
            if inv["compass"]:
                compass_inv = True
            if (inv["triforce"] & 0x02) != 0:
                timeline.append(
                    {"f": f, "event": "triforce_2", **_room_fields(snap, ram)}
                )
                mark_room("tf2")
                break

            # Room change bookkeeping.
            if snap.mode == PLAY_MODE and not snap.transitioning:
                if play_room is None or snap.screen != play_room:
                    prev = play_room
                    play_room = snap.screen
                    note_transition(prev, play_room, current_door if prev else None)
                    diamond_state = {"phase": "free", "cycle": 0}
                    fight_frames = 0
                    door_push = 0
                    stage_frames = 0
                    if play_room == 0x6F:
                        reached_6f = True
                    if play_room == 0x6E and keys_enter_6e is None:
                        keys_enter_6e = snap.keys

            # --- stage selection ---
            if not reached_6f:
                # Bootstrap nav toward 0x6f with both keys.
                # Preferred: west key → entry → east key → west into 6e → RIGHT.
                if play_room == 0x6C:
                    stage = "leave_6c"
                    current_door = "RIGHT"
                elif play_room == 0x6D:
                    # Prefer DOWN to entry for east key if keys < 2, else RIGHT to 6e.
                    if snap.keys < 2 and 0x7E not in visited:
                        stage = "6d_to_entry"
                        current_door = "DOWN"
                    else:
                        stage = "6d_to_6e"
                        current_door = "RIGHT"
                elif play_room == 0x7D:
                    if 0x7E not in visited or snap.keys < 2:
                        stage = "7d_to_7e"
                        current_door = "RIGHT"
                    else:
                        stage = "7d_to_6d"
                        current_door = "UP"
                elif play_room == 0x7E:
                    live = _live_enemies(snap)
                    if live or (snap.keys < 2 and snap.room_item_id == 0x19):
                        stage = "clear_7e"
                    else:
                        stage = "7e_to_6e"
                        # Prefer LEFT→7d→6d→6e west entry for diamond band.
                        current_door = "LEFT"
                elif play_room == 0x6E:
                    live = _live_enemies(snap)
                    if live:
                        stage = "clear_6e"
                    else:
                        stage = "6e_to_6f"
                        current_door = "RIGHT"
                else:
                    stage = "unknown"
                    current_door = "RIGHT"
            else:
                stage = "expand"
                if expand_phase == "loot":
                    # After enter: clear gels + walk compass.
                    pass

            act = nes_idle_action()
            live = (
                _live_enemies(snap)
                if snap.mode == PLAY_MODE and not snap.transitioning
                else []
            )

            if snap.transitioning or snap.mode in (4, 6, 7, 16):
                act = nes_action(current_door)
            elif stage in ("clear_7e", "clear_6e") or (
                stage == "expand" and expand_phase == "loot" and live
            ):
                fight_frames += 1
                target = min(
                    live,
                    key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                )
                dx, dy = target.x - snap.link_x, target.y - snap.link_y
                if abs(dx) > 10:
                    d = "RIGHT" if dx > 0 else "LEFT"
                elif abs(dy) > 10:
                    d = "DOWN" if dy > 0 else "UP"
                else:
                    d = "RIGHT" if dx >= 0 else "LEFT"
                # Key hunt mid-fight in 7e.
                if (
                    play_room == 0x7E
                    and snap.room_item_id == 0x19
                    and fight_frames % 40 < 12
                ):
                    if abs(snap.link_x - 136) > 8 or abs(snap.link_y - 141) > 8:
                        if abs(snap.link_x - 136) >= abs(snap.link_y - 141):
                            d = "RIGHT" if snap.link_x < 136 else "LEFT"
                        else:
                            d = "DOWN" if snap.link_y < 141 else "UP"
                        act = nes_action(d)
                    else:
                        act = _swing(f, d)
                else:
                    act = _swing(f, d)
                # Failsafe leave after long fight.
                if fight_frames > 6000 and not live:
                    pass
            elif stage == "expand" and expand_phase == "loot" and not live:
                # Walk compass corner then proceed.
                inv_c = inv["compass"]
                if not inv_c and snap.room_item_id in (0x16, 22):
                    # Compass often NE; sweep mid-room.
                    tx, ty = 160, 125
                    act, at = _goto_xy(snap, tx, ty, tol=8)
                    if at:
                        # sweep a bit
                        act = nes_action("RIGHT" if (f // 20) % 2 == 0 else "LEFT")
                    if stage_frames > 400:
                        expand_phase = "bomb_cycle"
                        bomb_idx = 0
                        bomb_sub = "goto"
                        mark_room("post_loot")
                        timeline.append(
                            {
                                "f": f,
                                "event": "loot_done",
                                "compass": inv["compass"],
                                "doors": snap.cur_opened_doors,
                                "mask": snap.open_doorway_mask,
                            }
                        )
                else:
                    expand_phase = "bomb_cycle"
                    bomb_idx = 0
                    bomb_sub = "goto"
                    mark_room("post_loot")
            elif stage == "expand" and expand_phase == "bomb_cycle":
                if bomb_idx >= len(bomb_dirs):
                    expand_phase = "door_cycle"
                    door_idx = 0
                    door_budget = 0
                    timeline.append({"f": f, "event": "bombs_done", "n": len(bomb_tests)})
                else:
                    face = bomb_dirs[bomb_idx]
                    sx, sy, face_dir = BOMB_STAND[face]
                    bombs_before = snap.bombs
                    doors_before = snap.cur_opened_doors
                    mask_before = snap.open_doorway_mask
                    room_before = play_room
                    if bomb_sub == "goto":
                        act, at = _goto_xy(snap, sx, sy, tol=6)
                        if at:
                            bomb_sub = "face"
                    elif bomb_sub == "face":
                        # Face the wall.
                        act = nes_action(face_dir)
                        bomb_sub = "place"
                    elif bomb_sub == "place":
                        _ensure_bomb_selected(env, ram)
                        act = nes_action(face_dir, "B")
                        bomb_sub = "wait"
                        bomb_wait = 0
                    elif bomb_sub == "wait":
                        bomb_wait += 1
                        # Step back slightly so blast doesn't kill pathing.
                        if bomb_wait < 8:
                            back = {
                                "UP": "DOWN",
                                "DOWN": "UP",
                                "LEFT": "RIGHT",
                                "RIGHT": "LEFT",
                            }[face_dir]
                            act = nes_action(back)
                        else:
                            act = nes_idle_action()
                        if bomb_wait >= 90:
                            # Bomb should have exploded (~60-90f).
                            bomb_sub = "push"
                            bomb_wait = 0
                    elif bomb_sub == "push":
                        bomb_wait += 1
                        act = _push_door(snap, face_dir)
                        if play_room != room_before:
                            bomb_tests.append(
                                {
                                    "face": face,
                                    "bombs": f"{bombs_before}->{snap.bombs}",
                                    "doors": f"{doors_before}->{snap.cur_opened_doors}",
                                    "mask": f"{mask_before}->{snap.open_doorway_mask}",
                                    "from": f"0x{room_before:02x}",
                                    "to": f"0x{play_room:02x}",
                                    "bomb_ok": snap.bombs < bombs_before,
                                    "opened": True,
                                }
                            )
                            mark_room(f"bomb_{face.lower()}")
                            # New room: switch to clear/loot there, keep expanding.
                            expand_phase = "new_room"
                            current_door = face
                        elif bomb_wait > 200:
                            bomb_tests.append(
                                {
                                    "face": face,
                                    "bombs": f"{bombs_before}->{snap.bombs}",
                                    "doors": f"{doors_before}->{snap.cur_opened_doors}",
                                    "mask": f"{mask_before}->{snap.open_doorway_mask}",
                                    "from": f"0x{room_before:02x}",
                                    "to": None,
                                    "bomb_ok": snap.bombs < bombs_before,
                                    "opened": False,
                                    "final_xy": [snap.link_x, snap.link_y],
                                }
                            )
                            mark_room(f"bombfail_{face.lower()}")
                            bomb_idx += 1
                            bomb_sub = "goto"
                            bomb_wait = 0
            elif stage == "expand" and expand_phase == "new_room":
                # Clear + snapshot new room then try continue forward.
                if live:
                    fight_frames += 1
                    target = min(
                        live,
                        key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                    )
                    dx, dy = target.x - snap.link_x, target.y - snap.link_y
                    if abs(dx) > 10:
                        d = "RIGHT" if dx > 0 else "LEFT"
                    elif abs(dy) > 10:
                        d = "DOWN" if dy > 0 else "UP"
                    else:
                        d = "RIGHT" if dx >= 0 else "LEFT"
                    act = _swing(f, d)
                else:
                    mark_room("new_clear")
                    # Prefer walkthrough forward: RIGHT/UP.
                    current_door = "RIGHT"
                    expand_phase = "door_cycle"
                    door_idx = 0
                    door_budget = 0
                    timeline.append(
                        {
                            "f": f,
                            "event": "new_room_ready",
                            "room": f"0x{play_room:02x}",
                            "doors": snap.cur_opened_doors,
                            "item": snap.room_item_id,
                            "types": rooms.get(f"0x{play_room:02x}", {}).get(
                                "type_names"
                            ),
                        }
                    )
            elif stage == "expand" and expand_phase == "door_cycle":
                if door_idx >= len(door_dirs):
                    expand_phase = "push_blocks"
                    push_cycle = 0
                    timeline.append({"f": f, "event": "doors_done"})
                else:
                    ddir = door_dirs[door_idx]
                    current_door = ddir
                    room_before = play_room
                    keys_before = snap.keys
                    doors_before = snap.cur_opened_doors
                    if door_budget == 0:
                        door_budget = 350
                    act = _diamond_or_door(
                        snap, ddir, room=play_room, phase_state=diamond_state
                    )
                    door_budget -= 1
                    if play_room != room_before:
                        door_tests.append(
                            {
                                "door": ddir,
                                "from": f"0x{room_before:02x}",
                                "to": f"0x{play_room:02x}",
                                "keys": f"{keys_before}->{snap.keys}",
                                "doors": f"{doors_before}->{snap.cur_opened_doors}",
                                "ok": True,
                            }
                        )
                        expand_phase = "new_room"
                        mark_room(f"door_{ddir.lower()}")
                    elif door_budget <= 0:
                        door_tests.append(
                            {
                                "door": ddir,
                                "from": f"0x{room_before:02x}",
                                "to": None,
                                "keys": f"{keys_before}->{snap.keys}",
                                "doors": f"{doors_before}->{snap.cur_opened_doors}",
                                "ok": False,
                                "xy": [snap.link_x, snap.link_y],
                            }
                        )
                        door_idx += 1
                        door_budget = 0
                        diamond_state = {"phase": "free", "cycle": 0}
            elif stage == "expand" and expand_phase == "push_blocks":
                # Push center diamond cluster in cardinal directions.
                centers = [(120, 141), (136, 141), (104, 141), (120, 125), (120, 157)]
                push_cycle += 1
                cidx = (push_cycle // 80) % len(centers)
                pdir = ["UP", "RIGHT", "DOWN", "LEFT"][(push_cycle // 20) % 4]
                tx, ty = centers[cidx]
                act, at = _goto_xy(snap, tx, ty, tol=4)
                if at:
                    act = nes_action(pdir)
                if push_cycle > 800:
                    expand_phase = "wander"
                    wander_step = 0
                    mark_room("post_push")
            elif stage == "expand" and expand_phase == "wander":
                # BFS-ish: keep trying any new door from wherever we are.
                wander_step += 1
                # If new room appeared with unopened exits, cycle doors.
                if live:
                    target = min(
                        live,
                        key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
                    )
                    dx, dy = target.x - snap.link_x, target.y - snap.link_y
                    if abs(dx) > 10:
                        d = "RIGHT" if dx > 0 else "LEFT"
                    elif abs(dy) > 10:
                        d = "DOWN" if dy > 0 else "UP"
                    else:
                        d = "RIGHT" if dx >= 0 else "LEFT"
                    act = _swing(f, d)
                else:
                    ddir = door_dirs[(wander_step // 400) % 4]
                    current_door = ddir
                    act = _diamond_or_door(
                        snap, ddir, room=play_room, phase_state=diamond_state
                    )
                if wander_step > 8000:
                    timeline.append({"f": f, "event": "wander_done"})
                    break
            else:
                # Door nav stages toward 0x6f.
                if live and stage.startswith("clear"):
                    pass  # handled above
                else:
                    door_push += 1
                    act = _diamond_or_door(
                        snap,
                        current_door,
                        room=play_room,
                        phase_state=diamond_state,
                    )
                    if door_push > 900 and stage not in ("6e_to_6f", "7d_to_7e"):
                        # rotate
                        door_push = 0
                        for alt in ("RIGHT", "UP", "LEFT", "DOWN"):
                            if alt != current_door:
                                current_door = alt
                                break

            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=f + 1)

            # Re-select bomb after inventory poke if game cleared it.
            if poke_bombs and f % 120 == 0:
                _ensure_bomb_selected(env, env.get_ram())

            if f % 200 == 0:
                timeline.append(
                    {
                        "f": f,
                        "stage": stage,
                        "expand": expand_phase if stage == "expand" else None,
                        "room": None if play_room is None else f"0x{play_room:02x}",
                        "xy": [snap.link_x, snap.link_y],
                        "keys": snap.keys,
                        "bombs": snap.bombs,
                        "doors": snap.cur_opened_doors,
                        "door_goal": current_door,
                        "mboom": inv["magical_boomerang"],
                        "compass": inv["compass"],
                        "sel": inv["selected"],
                    }
                )

        final = _room_fields(read_snapshot(env.get_ram()), env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")

        new_rooms = [
            f"0x{r:02x}"
            for r in visited
            if r not in {0x6C, 0x6D, 0x6E, 0x6F, 0x7D, 0x7E}
        ]
        return {
            "ok": bool(boom_got or (final.get("inventory") or {}).get("triforce", 0) & 2)
            or bool(new_rooms),
            "track": track,
            "intervention_class": (
                "survival_plus_inventory_poke"
                if (poke_bombs is not None or poke_keys is not None)
                else ("survival" if infinite_life else "clean")
            ),
            "poke_notes": poke_notes,
            "start_state": start_state,
            "entry": entry,
            "reached_6f": reached_6f,
            "compass_inventory": compass_inv,
            "magical_boomerang": boom_got,
            "triforce_bit2": bool((final.get("inventory") or {}).get("triforce", 0) & 2),
            "visited_order": [f"0x{r:02x}" for r in visited],
            "new_rooms_beyond_known": new_rooms,
            "graph": graph,
            "bomb_tests": bomb_tests,
            "door_tests": door_tests,
            "rooms": rooms,
            "transitions": transitions,
            "timeline_tail": timeline[-60:],
            "timeline_events": [
                e
                for e in timeline
                if e.get("event")
            ],
            "final": final,
            "frames": frames_run,
            "assist": assist.report() if assist else None,
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level2WestKey")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument(
        "--poke-bombs",
        type=int,
        default=None,
        help="Recon-only inventory poke (not Clean); also selects B=bomb",
    )
    parser.add_argument(
        "--poke-keys",
        type=int,
        default=None,
        help="Recon-only keys poke (not Clean)",
    )
    parser.add_argument("--max-frames", type=int, default=45000)
    parser.add_argument("--tag", default="l2_past6f")
    args = parser.parse_args(argv)

    report = run_probe(
        start_state=args.from_state,
        infinite_life=args.infinite_life,
        poke_bombs=args.poke_bombs,
        poke_keys=args.poke_keys,
        max_frames=args.max_frames,
        tag=args.tag,
    )
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(
        out,
        {
            "segment": "level2_past_6f_recon",
            "bead": ["rr-ebe", "rr-n5i"],
            **report,
        },
    )
    print(
        f"ok={report.get('ok')} reached_6f={report.get('reached_6f')} "
        f"visited={report.get('visited_order')} new={report.get('new_rooms_beyond_known')} "
        f"mboom={report.get('magical_boomerang')} tf2={report.get('triforce_bit2')} "
        f"bombs_tests={len(report.get('bomb_tests') or [])} "
        f"door_tests={len(report.get('door_tests') or [])} frames={report.get('frames')}"
    )
    for bt in report.get("bomb_tests") or []:
        print(f"  bomb {bt}")
    for dt in report.get("door_tests") or []:
        print(f"  door {dt}")
    print(f"graph={report.get('graph')}")
    print(f"wrote {out}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
