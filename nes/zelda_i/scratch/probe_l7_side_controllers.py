"""Drive Room68DownController / Room58NorthController from the recon fixtures.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_side_controllers.py --which 68down --tag 68_ctl_v1
    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_side_controllers.py --which 58north --tag 58_ctl_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.level7.hops import (
    make_room58_north_controller,
    make_room68_down_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_MAX_BOMBS,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

WHICH = {
    "68down": ("Level7Interior68ReconFixture", make_room68_down_controller),
    "58north": ("Level7Interior58ReconFixture", make_room58_north_controller),
}


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=list(WHICH))
    ap.add_argument("--tag", default="side_ctl_v1")
    args = ap.parse_args()
    state, factory = WHICH[args.which]
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    ctl = factory()
    out: dict = {"which": args.which, "route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        last_screen = int(read_snapshot(env.get_ram()).screen)
        stuck = 0
        last_xy = None
        rep = None
        while f < ctl.max_frames + 10:
            snap = read_snapshot(env.get_ram())
            if int(snap.screen) != last_screen:
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_f{f}_0x{int(snap.screen):02x}.png",
                )
                last_screen = int(snap.screen)
            xy = (int(snap.link_x), int(snap.link_y))
            stuck = stuck + 1 if xy == last_xy else 0
            last_xy = xy
            if stuck and stuck % 250 == 0:
                save_rgb_png(
                    env.render(),
                    RECORDINGS_DIR / f"{args.tag}_stuck_f{f}_{xy[0]}_{xy[1]}.png",
                )
            action = ctl.step(snap)
            env.step(action.action)
            a.apply_env(env, frame=f)
            f += 1
            if ctl.success or ctl.failed:
                rep = ctl.report()
                break
        out["frames"] = f
        out["report"] = rep
        for _ in range(150):
            env.step(nes_idle_action())
        out["end"] = _glance(env)
        out["end"]["deaths"] = int(a.telemetry.deaths)
        out["end"]["progression_writes"] = int(a.telemetry.progression_writes)
        out["end"]["capacity_writes"] = int(a.telemetry.capacity_writes)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
