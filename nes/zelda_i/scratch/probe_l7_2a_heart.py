"""Where does the 0x2A heart container actually spawn? (rr-8t4.3)

The L1 engine's fixed pickup cell ``(192,141)`` collected on the C1 lineage
and missed twice on the walk-on lineage (W1/W2), so this dumps the room after
the kill: all 13 object slots (type/x/y/hp/state), the ``$6530`` tile map, and
a screenshot, then walks a stuck-aware serpentine until ``heart_containers``
rises. Read-only; no writes.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \
        nes/zelda_i/scratch/probe_l7_2a_heart.py --tag 20260904_H1
"""
from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.tilemap import ascii_room
from zelda_i.level7.aquamentus import make_level7_aquamentus_heart_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ITEM_ID,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

FROM = "Level7Interior2AAquamentusReconFixture"


def slots(env) -> list[dict]:
    ram = env.get_ram()
    out = []
    for slot in range(13):
        t = int(read_u8(ram, ADDR_OBJ_TYPE + slot))
        out.append(
            {
                "slot": slot,
                "type": f"0x{t:02X}",
                "name": object_name(t),
                "xy": [int(read_u8(ram, ADDR_LINK_X + slot)),
                       int(read_u8(ram, ADDR_LINK_Y + slot))],
                "hp": int(read_u8(ram, ADDR_OBJ_HP + slot)),
                "state": int(read_u8(ram, 0x00AC + slot)),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260904_H1")
    ap.add_argument("--from-state", default=FROM)
    args = ap.parse_args()
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    out: dict = {"from_state": args.from_state, "route_eligible": False}
    frame = 0
    for _ in range(4):
        env.step(nes_idle_action())
        assist.apply_env(env, frame=frame)
        frame += 1

    ctl = make_level7_aquamentus_heart_controller()
    for _ in range(ctl.max_frames):
        snap = read_snapshot(env.get_ram())
        act = ctl.step(snap)
        env.step(act.action)
        assist.apply_env(env, frame=frame)
        frame += 1
        if ctl.boss_defeated or ctl.success or ctl.failed:
            break
    snap = read_snapshot(env.get_ram())
    out["after_kill"] = {
        "frame": frame,
        "xy": [int(snap.link_x), int(snap.link_y)],
        "hc": int(snap.heart_containers),
        "room_item": f"0x{int(read_u8(env.get_ram(), ADDR_ROOM_ITEM_ID)):02X}",
        "room_all_dead": int(snap.room_all_dead),
        "slots": slots(env),
    }
    print("AFTER KILL", json.dumps(out["after_kill"], indent=1))
    print(ascii_room(env.get_ram()))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_after_kill.png")

    # Settle a moment, then re-dump: the item may spawn a few frames later.
    for _ in range(90):
        env.step(nes_idle_action())
        assist.apply_env(env, frame=frame)
        frame += 1
    out["settled"] = {"slots": slots(env), "hc": int(read_snapshot(env.get_ram()).heart_containers)}
    print("SETTLED", json.dumps(out["settled"], indent=1))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_settled.png")

    # Stuck-aware serpentine over the boss floor band.
    rows = (141, 133, 125, 117, 149, 157, 165, 173, 109)
    xs = (56, 200)
    hc0 = int(read_snapshot(env.get_ram()).heart_containers)
    trail: list = []
    got = False
    for row in rows:
        for target in ((xs[0], row), (xs[1], row)):
            tx, ty = target
            stuck = 0
            last_xy = None
            for _ in range(600):
                s = read_snapshot(env.get_ram())
                if int(s.heart_containers) > hc0:
                    got = True
                    break
                xy = (int(s.link_x), int(s.link_y))
                if xy == last_xy:
                    stuck += 1
                else:
                    stuck = 0
                last_xy = xy
                if stuck > 40:
                    trail.append({"target": target, "stuck_at": xy})
                    break
                dx, dy = tx - xy[0], ty - xy[1]
                if abs(dx) <= 3 and abs(dy) <= 3:
                    break
                btn = ("DOWN" if dy > 0 else "UP") if abs(dy) > 3 else ("RIGHT" if dx > 0 else "LEFT")
                env.step(nes_action(btn))
                assist.apply_env(env, frame=frame)
                frame += 1
            if got:
                break
        if got:
            break
    s = read_snapshot(env.get_ram())
    out["sweep"] = {
        "collected": got,
        "hc": int(s.heart_containers),
        "xy": [int(s.link_x), int(s.link_y)],
        "frames": frame,
        "stuck_trail": trail,
        "slots": slots(env),
    }
    print("SWEEP", json.dumps(out["sweep"], indent=1))
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_sweep.png")
    (RECORDINGS_DIR / f"l7_2a_heart_{args.tag}.json").write_text(json.dumps(out, indent=1))
    env.close()
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
