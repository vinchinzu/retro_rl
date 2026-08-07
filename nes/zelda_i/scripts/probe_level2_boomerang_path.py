"""Assisted recon: Level 2 west key → Magical Boomerang path (walkthrough).

Maps room IDs, enemy types, door bits, RoomItemId, and boomerang inventory
addrs (0x0674 wooden / 0x0675 magical). Survival assist optional.

Default start: ``Level2WestKey`` (keys≥1 in 0x6c). Walkthrough sequence after
west key: return entry → east key room → north/east branches → Blue Goriya
boomerang room. **Planning path only until rooms are live-verified.**

Room-goal map drives doors (not a scroll-fragile step counter).

Examples::

    uv run python nes/zelda_i/scripts/probe_level2_boomerang_path.py --infinite-life
    uv run python nes/zelda_i/scripts/probe_level2_boomerang_path.py \\
        --from-state Level2WestKey --infinite-life --tag l2_boom_path
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
from zelda_i.ram import PLAY_MODE, ZeldaObject, ZeldaSnapshot, read_snapshot, read_u8

LEVEL_2 = 2
ADDR_BOOMERANG = 0x0674
ADDR_MAGIC_BOOMERANG = 0x0675
ADDR_COMPASS = 0x0667
ADDR_MAP = 0x0668

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

# Preferred exit door when current room is clear (or empty). Walkthrough-seeded;
# unknown rooms fall back to rotate-try.
ROOM_EXIT_GOAL: dict[int, str] = {
    0x6C: "RIGHT",  # west key → ropes
    0x6D: "DOWN",  # ropes → entry (return path)
    # 0x7d/0x6e RIGHT: diamond-nav (nav_common.diamond_east_phase)
    0x7D: "RIGHT",  # entry → 0x7e east key (band≈157)
    0x7E: "UP",  # east key → 0x6e (or LEFT return → 6d RIGHT for west entry)
    0x6E: "RIGHT",  # key door → 0x6f compass gels (band≈113; WEST entry)
    0x6F: "RIGHT",  # residual past compass room
    0x5E: "RIGHT",
    0x5D: "RIGHT",
    0x5C: "RIGHT",
    0x4E: "RIGHT",
    0x4D: "RIGHT",
    0x4C: "RIGHT",
}

# Rooms whose RIGHT exit needs diamond-east (not naive y=141 align).
DIAMOND_EAST_ROOMS: dict[int, int] = {
    0x7D: 157,
    0x6E: 113,
    0x6F: 113,
}

# When stuck, try doors in this order (prefer forward path).
DOOR_ROTATE: tuple[str, ...] = ("RIGHT", "UP", "LEFT", "DOWN")


def _objs(snap: ZeldaSnapshot) -> list[dict]:
    out: list[dict] = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF):
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


def _inventory(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "boomerang": read_u8(ram, ADDR_BOOMERANG),
        "magical_boomerang": read_u8(ram, ADDR_MAGIC_BOOMERANG),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "compass": read_u8(ram, ADDR_COMPASS),
        "map": read_u8(ram, ADDR_MAP),
        "sword": int(snap.sword),
        "triforce": int(snap.triforce),
    }


def _room_fields(snap: ZeldaSnapshot, ram=None) -> dict:
    fields = {
        "mode": snap.mode,
        "level": snap.level,
        "screen": snap.screen,
        "x": snap.link_x,
        "y": snap.link_y,
        "facing": snap.facing,
        "health": snap.health,
        "keys": snap.keys,
        "bombs": snap.bombs,
        "room_item_id": snap.room_item_id,
        "room_item_name": room_item_name(snap.room_item_id),
        "room_all_dead": snap.room_all_dead,
        "room_obj_count": snap.room_obj_count,
        "cur_opened_doors": snap.cur_opened_doors,
        "doors": {
            "R": bool(snap.cur_opened_doors & DOOR_RIGHT),
            "L": bool(snap.cur_opened_doors & DOOR_LEFT),
            "D": bool(snap.cur_opened_doors & DOOR_DOWN),
            "U": bool(snap.cur_opened_doors & DOOR_UP),
            "raw": snap.cur_opened_doors,
        },
        "objects": _objs(snap),
        "type_counts": dict(Counter(o["type"] for o in _objs(snap))),
        "type_names": {
            f"0x{t:02x}": object_name(t)
            for t in Counter(o["type"] for o in _objs(snap))
        },
    }
    if ram is not None:
        fields["inventory"] = _inventory(ram)
    return fields


def _live_combat(snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
    drop_types = {0x60, 0x61, 0x62, 0x63}
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 10):
            continue
        if o.type_id in (0, 0xFF) or o.type_id in drop_types:
            continue
        if o.hp <= 0:
            continue
        out.append(o)
    return tuple(out)


def _swing(frames: int, direction: str, *, period: int = 8, hold: int = 3):
    if frames % period < hold:
        return nes_action(direction, "A")
    return nes_action(direction)


def _push_door(snap: ZeldaSnapshot, direction: str) -> object:
    tx, ty = DOOR_TARGETS[direction]
    if direction in ("LEFT", "RIGHT"):
        if abs(snap.link_y - ty) > 4:
            return nes_action("DOWN" if snap.link_y < ty else "UP")
        return nes_action(direction)
    if abs(snap.link_x - tx) > 6:
        return nes_action("RIGHT" if snap.link_x < tx else "LEFT")
    return nes_action(direction)


def _center_then_door(snap: ZeldaSnapshot, direction: str) -> object:
    """Nudge toward room center before door if jammed on wrong y/x."""
    cx, cy = 120, 141
    if direction == "DOWN":
        # From mid-room go south; avoid hugging side walls.
        if abs(snap.link_x - cx) > 12:
            return nes_action("RIGHT" if snap.link_x < cx else "LEFT")
        if snap.link_y < 180:
            return nes_action("DOWN")
        return nes_action("DOWN")
    if direction == "UP":
        if abs(snap.link_x - cx) > 12:
            return nes_action("RIGHT" if snap.link_x < cx else "LEFT")
        return _push_door(snap, "UP")
    return _push_door(snap, direction)


def _diamond_or_door(
    snap: ZeldaSnapshot,
    direction: str,
    *,
    room: int | None,
    phase_state: dict,
) -> object:
    """Use diamond-east for known rooms; otherwise naive door push."""
    if (
        direction == "RIGHT"
        and room is not None
        and room in DIAMOND_EAST_ROOMS
    ):
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
    return _center_then_door(snap, direction)


def run_probe(
    *,
    start_state: str,
    infinite_life: bool,
    max_frames: int,
    fight_budget: int,
    door_push_budget: int,
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
        if assist is not None:
            assist.apply_env(env, frame=0)

        snap = read_snapshot(env.get_ram())
        entry = _room_fields(snap, env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t0.png")

        rooms_cleared: dict[str, dict] = {}
        transitions: list[dict] = []
        timeline: list[dict] = []
        visited_order: list[int] = []
        room_enter_frame: dict[int, int] = {}
        room_max_live: dict[int, int] = {}
        room_types: dict[int, Counter] = {}
        room_cleared_frame: dict[int, int | None] = {}
        room_item_seen: dict[int, int] = {}
        room_doors_peak: dict[int, int] = {}
        keys_on_enter: dict[int, int] = {}
        inv_on_enter: dict[int, dict] = {}
        door_try_index: dict[int, int] = {}

        play_room: int | None = (
            snap.screen if snap.mode == PLAY_MODE else None
        )
        if play_room is not None:
            room_enter_frame[play_room] = 0
            visited_order.append(play_room)
            keys_on_enter[play_room] = snap.keys
            inv_on_enter[play_room] = _inventory(env.get_ram())
            room_types[play_room] = Counter()
            room_max_live[play_room] = 0
            room_cleared_frame[play_room] = None
            room_item_seen[play_room] = snap.room_item_id
            room_doors_peak[play_room] = snap.cur_opened_doors
            door_try_index[play_room] = 0

        door_push_frames = 0
        fight_frames_room = 0
        last_new_room_frame = 0
        frames_run = 0
        diamond_phase_state: dict = {"phase": "free", "cycle": 0}
        current_door = ROOM_EXIT_GOAL.get(
            play_room if play_room is not None else snap.screen, "RIGHT"
        )

        for f in range(max_frames):
            frames_run = f + 1
            ram = env.get_ram()
            snap = read_snapshot(ram)
            inv = _inventory(ram)

            if snap.mode == 17:
                timeline.append({"f": f, "event": "death", **_room_fields(snap, ram)})
                break
            if snap.level != LEVEL_2:
                timeline.append(
                    {"f": f, "event": "left_level2", **_room_fields(snap, ram)}
                )
                break
            if inv["magical_boomerang"]:
                timeline.append(
                    {
                        "f": f,
                        "event": "magical_boomerang",
                        **_room_fields(snap, ram),
                    }
                )
                break

            # Commit room arrival only on play mode.
            if snap.mode == PLAY_MODE and not snap.transitioning:
                if play_room is None or snap.screen != play_room:
                    prev = play_room
                    play_room = snap.screen
                    transitions.append(
                        {
                            "f": f,
                            "from": prev,
                            "to": play_room,
                            "xy": (snap.link_x, snap.link_y),
                            "keys": snap.keys,
                            "doors": snap.cur_opened_doors,
                        }
                    )
                    if play_room not in room_enter_frame:
                        room_enter_frame[play_room] = f
                        visited_order.append(play_room)
                        keys_on_enter[play_room] = snap.keys
                        inv_on_enter[play_room] = inv
                        room_types[play_room] = Counter()
                        room_max_live[play_room] = 0
                        room_cleared_frame[play_room] = None
                        room_item_seen[play_room] = snap.room_item_id
                        room_doors_peak[play_room] = snap.cur_opened_doors
                        door_try_index[play_room] = 0
                        last_new_room_frame = f
                        save_rgb_png(
                            obs,
                            RECORDINGS_DIR / f"{tag}_room_{play_room:02x}.png",
                        )
                    door_push_frames = 0
                    fight_frames_room = 0
                    diamond_phase_state = {"phase": "free", "cycle": 0}
                    current_door = ROOM_EXIT_GOAL.get(play_room, "RIGHT")

                live = _live_combat(snap)
                types = Counter(
                    o.type_id
                    for o in snap.objects
                    if 1 <= o.slot <= 10 and o.type_id not in (0, 0xFF)
                )
                room_types[play_room] |= types
                room_max_live[play_room] = max(room_max_live[play_room], len(live))
                room_item_seen[play_room] = snap.room_item_id
                room_doors_peak[play_room] = max(
                    room_doors_peak.get(play_room, 0), snap.cur_opened_doors
                )

                keys_in = keys_on_enter.get(play_room, 0)
                key_gained = snap.keys > keys_in
                cleared = (
                    not live
                    and room_max_live[play_room] >= 1
                    and (
                        snap.room_all_dead >= 20
                        or key_gained
                        or room_max_live[play_room] == 0
                    )
                )
                # Empty rooms (entry-like): treat as ready after short settle.
                empty_ready = (
                    not live
                    and room_max_live[play_room] == 0
                    and (f - room_enter_frame[play_room]) >= 40
                )
                if (
                    cleared or empty_ready
                ) and room_cleared_frame[play_room] is None:
                    if cleared or empty_ready:
                        room_cleared_frame[play_room] = f
                        rooms_cleared[f"0x{play_room:02x}"] = {
                            **_room_fields(snap, ram),
                            "cleared_frame": f,
                            "max_live": room_max_live[play_room],
                            "type_peak": {
                                f"0x{k:02x}": v
                                for k, v in room_types[play_room].items()
                            },
                            "type_peak_names": {
                                f"0x{k:02x}": object_name(k)
                                for k in room_types[play_room]
                            },
                            "keys_on_enter": keys_in,
                            "keys_now": snap.keys,
                            "empty_ready": empty_ready and not cleared,
                        }
                        timeline.append(
                            {
                                "f": f,
                                "event": "room_ready",
                                "room": play_room,
                                "keys": snap.keys,
                                "doors": snap.cur_opened_doors,
                                "item": snap.room_item_id,
                                "max_live": room_max_live[play_room],
                            }
                        )

                ready_to_leave = room_cleared_frame.get(play_room) is not None
                if live and fight_frames_room < fight_budget and not (
                    ready_to_leave and key_gained and not live
                ):
                    # Still fighting.
                    fight_frames_room += 1
                    target = min(
                        live,
                        key=lambda o: abs(o.x - snap.link_x)
                        + abs(o.y - snap.link_y),
                    )
                    dx = target.x - snap.link_x
                    dy = target.y - snap.link_y
                    if abs(dx) > 10:
                        d = "RIGHT" if dx > 0 else "LEFT"
                    elif abs(dy) > 10:
                        d = "DOWN" if dy > 0 else "UP"
                    else:
                        d = "RIGHT" if dx >= 0 else "LEFT"
                    # If key drop visible mid-fight, prioritise walking over it.
                    if snap.room_item_id == 0x19 and snap.keys == keys_in:
                        # Hunt key near mid-room then resume combat.
                        d_key = None
                        if abs(snap.link_x - 136) > 8 or abs(snap.link_y - 141) > 8:
                            if abs(snap.link_x - 136) >= abs(snap.link_y - 141):
                                d_key = "RIGHT" if snap.link_x < 136 else "LEFT"
                            else:
                                d_key = "DOWN" if snap.link_y < 141 else "UP"
                        if d_key and fight_frames_room % 40 < 12:
                            act = nes_action(d_key)
                        else:
                            act = _swing(f, d)
                    else:
                        act = _swing(f, d)
                elif ready_to_leave or empty_ready or (
                    not live and fight_frames_room >= fight_budget
                ):
                    door_push_frames += 1
                    if door_push_frames > door_push_budget:
                        door_try_index[play_room] = (
                            door_try_index.get(play_room, 0) + 1
                        )
                        idx = door_try_index[play_room] % len(DOOR_ROTATE)
                        current_door = DOOR_ROTATE[idx]
                        door_push_frames = 0
                        timeline.append(
                            {
                                "f": f,
                                "event": "door_stuck_rotate",
                                "room": play_room,
                                "try_door": current_door,
                                "doors": snap.cur_opened_doors,
                                "keys": snap.keys,
                            }
                        )
                    act = _diamond_or_door(
                        snap,
                        current_door,
                        room=play_room,
                        phase_state=diamond_phase_state,
                    )
                else:
                    # Waiting for spawn.
                    act = _swing(f, "UP", period=12, hold=2)
            elif snap.transitioning or snap.mode in (4, 6, 7, 16):
                act = nes_action(current_door)
            else:
                act = nes_idle_action()

            obs, *_ = env.step(act)
            if assist is not None:
                assist.apply_env(env, frame=f + 1)

            if f % 150 == 0:
                timeline.append(
                    {
                        "f": f,
                        "room": play_room,
                        "screen": snap.screen,
                        "mode": snap.mode,
                        "xy": (snap.link_x, snap.link_y),
                        "keys": snap.keys,
                        "doors": snap.cur_opened_doors,
                        "door_goal": current_door,
                        "mboom": inv["magical_boomerang"],
                    }
                )

            if (
                play_room is not None
                and f - last_new_room_frame > 12000
                and f - room_enter_frame.get(play_room, 0) > 12000
            ):
                timeline.append(
                    {"f": f, "event": "stalled", **_room_fields(snap, ram)}
                )
                break

        final_snap = read_snapshot(env.get_ram())
        final = _room_fields(final_snap, env.get_ram())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_final.png")

        room_summary = []
        for rid in visited_order:
            room_summary.append(
                {
                    "room": rid,
                    "room_hex": f"0x{rid:02x}",
                    "enter_frame": room_enter_frame.get(rid),
                    "keys_on_enter": keys_on_enter.get(rid),
                    "inventory_on_enter": inv_on_enter.get(rid),
                    "max_live": room_max_live.get(rid, 0),
                    "cleared_frame": room_cleared_frame.get(rid),
                    "doors_peak": room_doors_peak.get(rid),
                    "room_item_id": room_item_seen.get(rid),
                    "room_item_name": room_item_name(room_item_seen.get(rid, 0)),
                    "type_peak": {
                        f"0x{k:02x}": v
                        for k, v in room_types.get(rid, Counter()).items()
                    },
                    "type_peak_names": {
                        f"0x{k:02x}": object_name(k)
                        for k in room_types.get(rid, Counter())
                    },
                }
            )

        ok_boom = bool(final.get("inventory", {}).get("magical_boomerang"))
        return {
            "ok": ok_boom,
            "track": track,
            "start_state": start_state,
            "infinite_life": infinite_life,
            "assist": assist.report() if assist else None,
            "entry": entry,
            "visited_order": [f"0x{r:02x}" for r in visited_order],
            "transitions": transitions,
            "room_summary": room_summary,
            "rooms_cleared_detail": rooms_cleared,
            "timeline_tail": timeline[-50:],
            "timeline_events": [
                e
                for e in timeline
                if e.get("event")
                in {
                    "room_ready",
                    "magical_boomerang",
                    "death",
                    "stalled",
                    "door_stuck_rotate",
                    "left_level2",
                }
            ],
            "final": final,
            "magical_boomerang": ok_boom,
            "frames": frames_run,
            "screenshots": {
                "t0": str(RECORDINGS_DIR / f"{tag}_t0.png"),
                "final": str(RECORDINGS_DIR / f"{tag}_final.png"),
            },
        }
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default="Level2WestKey")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--max-frames", type=int, default=30000)
    parser.add_argument("--fight-budget", type=int, default=5000)
    parser.add_argument("--door-push-budget", type=int, default=450)
    parser.add_argument("--tag", default="l2_boom_path")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args(argv)

    reports = []
    for trial in range(args.trials):
        tag = f"{args.tag}_t{trial}" if args.trials > 1 else args.tag
        report = run_probe(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            max_frames=args.max_frames,
            fight_budget=args.fight_budget,
            door_push_budget=args.door_push_budget,
            tag=tag,
        )
        reports.append(report)
        print(
            f"trial={trial} ok={report.get('ok')} "
            f"visited={report.get('visited_order')} "
            f"keys_final={report.get('final', {}).get('keys')} "
            f"mboom={report.get('final', {}).get('inventory', {}).get('magical_boomerang')} "
            f"room=0x{report.get('final', {}).get('screen', 0):02x} "
            f"frames={report.get('frames')}"
        )
        for rs in report.get("room_summary") or []:
            print(
                f"  room={rs['room_hex']} max_live={rs['max_live']} "
                f"cleared={rs['cleared_frame']} doors_peak={rs['doors_peak']} "
                f"item=0x{(rs['room_item_id'] or 0):02x} "
                f"types={rs['type_peak_names']} keys_in={rs['keys_on_enter']}"
            )

    out = RECORDINGS_DIR / f"{args.tag}_probe.json"
    write_json_report(
        out,
        {
            "segment": "level2_boomerang_path_recon",
            "track": "assisted" if args.infinite_life else "clean",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "start_state": args.from_state,
            "boomerang_addrs": {
                "wooden": "0x0674",
                "magical": "0x0675",
            },
            "room_item_boomerang": "0x1D",
            "trials": args.trials,
            "successes": sum(1 for r in reports if r.get("ok")),
            "reports": reports,
        },
    )
    print(f"wrote {out}")
    return 0 if any(r.get("ok") for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
