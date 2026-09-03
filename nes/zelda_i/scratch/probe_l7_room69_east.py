"""0x79 north (green) then 0x69 kill-clear → east door. Do not poke.

Chains ``EntryNorthDoorController`` then ``Room69EastController``.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l7_room69_east.py --tag room69_east_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.door_graph.core import DoorDir
from zelda_i.dungeon.ids import object_name
from zelda_i.level7.path import (
    EAST_DOOR,
    ENTRY_SCREEN,
    LEVEL7,
    ROOM_69,
    EntryNorthDoorController,
    Room69EastController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_FOOD,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import add_common_args, make_assist, write_report
from zelda_i.screen_glance import leftover_from_snapshot

STUCK_SHOT = 250


def _glance(env) -> dict[str, Any]:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    leftover = leftover_from_snapshot(snap)
    types = sorted(
        {
            int(o.type_id)
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
        }
    )
    leftover.update(
        {
            "level": int(snap.level),
            "food": int(read_u8(ram, ADDR_FOOD)),
            "whistle": int(read_u8(ram, ADDR_WHISTLE)),
            "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "triforce": int(snap.triforce),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "doors": {
                "R": bool(snap.cur_opened_doors & DoorDir.RIGHT),
                "L": bool(snap.cur_opened_doors & DoorDir.LEFT),
                "D": bool(snap.cur_opened_doors & DoorDir.DOWN),
                "U": bool(snap.cur_opened_doors & DoorDir.UP),
            },
            "open_mask": {
                "R": bool(snap.open_doorway_mask & DoorDir.RIGHT),
                "L": bool(snap.open_doorway_mask & DoorDir.LEFT),
                "D": bool(snap.open_doorway_mask & DoorDir.DOWN),
                "U": bool(snap.open_doorway_mask & DoorDir.UP),
            },
            "room_item_id": int(snap.room_item_id),
            "room_all_dead": int(snap.room_all_dead),
            "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        }
    )
    return leftover


def _shot(obs, tag: str, frame: int, snap, *, suffix: str = "") -> Path:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    extra = f"_{suffix}" if suffix else ""
    png = (
        RECORDINGS_DIR
        / f"{tag}{extra}_f{frame}_L{snap.level}_s{snap.screen:02x}_m{snap.mode}.png"
    )
    save_rgb_png(obs, png)
    return png


def _sample(
    samples: list[dict[str, Any]], ctl, snap: ZeldaSnapshot, reason: str
) -> None:
    samples.append(
        {
            "frame": ctl.frames,
            "reason": reason,
            "level": int(snap.level),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile),
            "doors": int(snap.cur_opened_doors),
            "mask": int(snap.open_doorway_mask),
            "dead": int(snap.room_all_dead),
            "misses": getattr(getattr(ctl, "walker", None), "misses", None),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="room69_east_v1")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    north = EntryNorthDoorController()
    east = Room69EastController()
    ctl: EntryNorthDoorController | Room69EastController = north
    shots: list[str] = []
    samples: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "from_state": args.from_state,
        "infinite_life": args.infinite_life,
        "tag": args.tag,
        "route_eligible": False,
        "door": "RIGHT",
        "goal": list(EAST_DOOR),
    }
    try:
        obs, _ = reset_obs(env)
        for _ in range(2):
            obs, *_ = env.step(nes_idle_action())
        start_snap = read_snapshot(env.get_ram())
        start = _glance(env)
        payload["start"] = start
        shots.append(str(_shot(obs, args.tag, 0, start_snap, suffix="start")))
        print("START", start)
        food0 = int(start.get("food", -1))
        if (
            int(start.get("level", -1)) != LEVEL7
            or int(start.get("screen", -1)) != ENTRY_SCREEN
            or int(start.get("mode", -1)) != PLAY_MODE
            or food0 != 0
        ):
            payload["success"] = False
            payload["failed"] = "pin_mismatch"
            payload["leftover"] = start
            payload["screenshots"] = shots
            out = write_report("l7_room69_east", payload, tag=args.tag)
            print(out)
            return

        last_key = (start_snap.level, start_snap.mode, start_snap.screen)
        last_xy = (start_snap.link_x, start_snap.link_y)
        stuck = 0
        total = 0
        budget = north.max_frames + east.max_frames
        for frame in range(budget):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            obs, *_ = env.step(action.action)
            if assist is not None:
                assist.apply_env(env, frame=frame)
            snap = read_snapshot(env.get_ram())
            total = frame + 1
            key = (snap.level, snap.mode, snap.screen)
            xy = (snap.link_x, snap.link_y)
            if key != last_key:
                shots.append(str(_shot(obs, args.tag, total, snap, suffix="room")))
                _sample(samples, ctl, snap, f"change_{action.reason}")
                last_key = key
                stuck = 0
            if xy == last_xy:
                stuck += 1
                if stuck % STUCK_SHOT == 0:
                    shots.append(
                        str(_shot(obs, args.tag, total, snap, suffix=f"stuck{stuck}"))
                    )
                    _sample(samples, ctl, snap, f"stuck_{stuck}_{action.reason}")
            else:
                stuck = 0
                last_xy = xy
            if frame % 60 == 0:
                _sample(samples, ctl, snap, action.reason)
            if ctl is north and north.success:
                payload["north"] = north.report()
                ctl = east
                stuck = 0
                continue
            if ctl.success or ctl.failed:
                break

        leftover = _glance(env)
        leftover["deaths"] = int(assist.telemetry.deaths) if assist is not None else 0
        payload["leftover"] = leftover
        report = east.report()
        report["samples"] = samples[-48:]
        payload["controller"] = report
        payload["north"] = north.report()
        payload["frames"] = total
        payload["assist"] = assist.report() if assist is not None else None
        deaths = leftover["deaths"]
        writes_ok = True
        if assist is not None:
            writes_ok = (
                assist.telemetry.progression_writes == 0
                and assist.telemetry.capacity_writes == 0
            )
        dest_ok = (
            leftover.get("level") == LEVEL7
            and leftover.get("mode") == PLAY_MODE
            and leftover.get("screen") not in {ENTRY_SCREEN, ROOM_69}
            and leftover.get("screen") is not None
            and deaths == 0
            and writes_ok
            and bool(east.success)
            and not east.failed
        )
        payload["success"] = dest_ok
        if not dest_ok:
            payload["failed"] = (
                east.notes[-1]
                if east.notes
                else (north.notes[-1] if north.failed else "not_left_0x69")
            )
        shots.append(str(_shot(obs, f"{args.tag}_final", total, read_snapshot(env.get_ram()))))
        png = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(obs, png)
        payload["screenshot"] = str(png)
        payload["screenshots"] = shots
        out = write_report("l7_room69_east", payload, tag=args.tag)
        print(out)
        print(
            f"success={payload['success']} frames={total} "
            f"leftover={leftover} dest={east.dest} "
            f"failed={payload.get('failed')} "
            f"east_opened={east.east_opened_frame} objs={east.obj_types}"
        )
        for line in north.notes[-8:] + east.notes[-16:]:
            print(line)
        for rec in samples[-8:]:
            print(rec)
    finally:
        env.close()


if __name__ == "__main__":
    main()
