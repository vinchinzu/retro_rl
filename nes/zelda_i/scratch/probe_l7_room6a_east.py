"""0x79 north (green) → 0x69 kill-clear east (green) → 0x6A blind east push.

Chains ``EntryNorthDoorController`` → ``Room69EastController`` →
``Room6AEastController``.  0x6A is the KEESE dark room; the entry pin carries
Candle 0 so it stays unlit and the traverse is a blind y=141 push.  Do not
poke.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l7_room6a_east.py --tag room6a_east_v1
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
    ENTRY_SCREEN,
    LEVEL7,
    ROOM_69,
    ROOM_6A,
    EntryNorthDoorController,
    Room6AEastController,
    Room69EastController,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
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

STUCK_SHOT = 200


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
            "candle": int(read_u8(ram, ADDR_CANDLE)),
            "selected_item": int(read_u8(ram, ADDR_SELECTED_ITEM)),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "triforce": int(snap.triforce),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
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


def _sample(samples: list[dict[str, Any]], ctl, snap: ZeldaSnapshot, reason: str) -> None:
    samples.append(
        {
            "stage": type(ctl).__name__,
            "frame": ctl.frames,
            "reason": reason,
            "screen": f"0x{snap.screen:02x}",
            "mode": int(snap.mode),
            "xy": [int(snap.link_x), int(snap.link_y)],
            "tile": int(snap.colliding_tile),
            "doors": int(snap.cur_opened_doors),
            "mask": int(snap.open_doorway_mask),
            "dead": int(snap.room_all_dead),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_state="Level7Entrance", default_tag="room6a_east_v1")
    args = parser.parse_args()
    configure_headless()
    assist = make_assist(args.infinite_life)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    north = EntryNorthDoorController()
    east69 = Room69EastController()
    east6a = Room6AEastController()
    chain = [north, east69, east6a]
    idx = 0
    ctl = chain[idx]
    shots: list[str] = []
    samples: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "from_state": args.from_state,
        "infinite_life": args.infinite_life,
        "tag": args.tag,
        "route_eligible": False,
        "door": "RIGHT",
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
        if (
            int(start.get("level", -1)) != LEVEL7
            or int(start.get("screen", -1)) != int(ENTRY_SCREEN)
            or int(start.get("mode", -1)) != PLAY_MODE
            or int(start.get("food", -1)) != 0
        ):
            payload["success"] = False
            payload["failed"] = "pin_mismatch"
            payload["leftover"] = start
            payload["screenshots"] = shots
            print(write_report("l7_room6a_east", payload, tag=args.tag))
            return

        last_key = (start_snap.level, start_snap.mode, start_snap.screen)
        last_xy = (start_snap.link_x, start_snap.link_y)
        stuck = 0
        total = 0
        budget = sum(c.max_frames for c in chain)
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
            if ctl.success and idx < len(chain) - 1:
                payload[type(ctl).__name__] = ctl.report()
                _sample(samples, ctl, snap, "stage_done")
                idx += 1
                ctl = chain[idx]
                stuck = 0
                continue
            if ctl.success or ctl.failed:
                break

        leftover = _glance(env)
        leftover["deaths"] = int(assist.telemetry.deaths) if assist is not None else 0
        payload["leftover"] = leftover
        for c in chain:
            payload[type(c).__name__] = c.report()
        payload["samples"] = samples[-80:]
        payload["frames"] = total
        payload["assist"] = assist.report() if assist is not None else None
        writes_ok = True
        if assist is not None:
            writes_ok = (
                assist.telemetry.progression_writes == 0
                and assist.telemetry.capacity_writes == 0
            )
        dest_ok = (
            leftover.get("level") == LEVEL7
            and leftover.get("mode") == PLAY_MODE
            and leftover.get("screen") not in {int(ENTRY_SCREEN), ROOM_69, ROOM_6A}
            and leftover.get("screen") is not None
            and leftover["deaths"] == 0
            and writes_ok
            and bool(east6a.success)
            and not east6a.failed
        )
        payload["success"] = dest_ok
        if not dest_ok:
            payload["failed"] = next(
                (c.notes[-1] for c in reversed(chain) if c.notes), "no_progress"
            )
        shots.append(
            str(_shot(obs, f"{args.tag}_final", total, read_snapshot(env.get_ram())))
        )
        png = RECORDINGS_DIR / f"{args.tag}_final.png"
        save_rgb_png(obs, png)
        payload["screenshot"] = str(png)
        payload["screenshots"] = shots
        print(write_report("l7_room6a_east", payload, tag=args.tag))
        print(
            f"success={payload['success']} frames={total} "
            f"leftover_screen={leftover.get('screen')} "
            f"xy={leftover.get('link_x')},{leftover.get('link_y')} "
            f"failed={payload.get('failed')}"
        )
        for c in chain:
            print(type(c).__name__, c.notes[-6:])
        for rec in samples[-16:]:
            print(rec)
    finally:
        env.close()


if __name__ == "__main__":
    main()
