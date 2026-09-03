"""Drive Room49UpController from Level7Interior49LadderReconFixture and
confirm it walks to live $EB=0x39 (DIGDOGGER_2).  Run twice for 2/2.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_room49_controller.py --tag 49_ctl_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.level7.hops import make_room49_up_controller
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_WHISTLE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist


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
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="49_ctl_v1")
    ap.add_argument("--from-state", default="Level7Interior49LadderReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    ctl = make_room49_up_controller()
    out: dict = {}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        last_screen = out["start"]["screen"]
        rep = None
        while f < ctl.max_frames + 10:
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            env.step(action.action)
            a.apply_env(env, frame=f)
            f += 1
            screen = f"0x{int(snap.screen):02x}"
            if screen != last_screen:
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png")
                last_screen = screen
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
