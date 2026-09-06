"""Drive the real Room1ACandleController from Level7Interior1AReconFixture and
trace every position change, phase, hunt index, reason, and enemy census.
Dumps the $6530 ascii room map at the end.  Read-only diagnostic.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_controller_trace.py --tag 1a_ctl_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.tilemap import ascii_room, has_room_tile_map
from zelda_i.level7.cellar import Room1ACandleController
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, read_snapshot, read_u8
from zelda_i.runner import make_assist


def _objs(s):
    return [
        {
            "slot": int(o.slot),
            "t": f"0x{int(o.type_id) & 0xFF:02x}:{object_name(int(o.type_id) & 0xFF)}",
            "hp": int(o.hp),
            "st": int(o.state),
            "xy": [int(o.x), int(o.y)],
        }
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) & 0xFF not in (0, 0xFF)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_ctl_v1")
    ap.add_argument("--from-state", default="Level7Interior1AReconFixture")
    ap.add_argument("--max-frames", type=int, default=42000)
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ctl = Room1ACandleController()
    bind = getattr(ctl, "bind_env", None)
    if bind is not None:
        bind(env)
    out: dict = {}
    trace: list[dict] = []
    reason_counts: dict[str, int] = {}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        s = read_snapshot(env.get_ram())
        out["start"] = {
            "xy": [int(s.link_x), int(s.link_y)],
            "candle": int(read_u8(env.get_ram(), ADDR_CANDLE)),
            "objects": _objs(s),
        }
        last_xy = None
        last_phase = None
        last_hi = None
        f = 0
        while f < args.max_frames:
            s = read_snapshot(env.get_ram())
            action = ctl.step(s)
            env.step(action.action)
            a.apply_env(env, frame=f)
            f += 1
            reason_counts[action.reason] = reason_counts.get(action.reason, 0) + 1
            xy = (int(s.link_x), int(s.link_y))
            phase = getattr(ctl, "_phase", None)
            hi = getattr(ctl, "_hunt_i", None)
            if f % 500 == 0 or xy != last_xy or phase != last_phase or hi != last_hi:
                lg = live_goriyas(s)
                trace.append(
                    {
                        "f": f,
                        "xy": list(xy),
                        "phase": phase,
                        "hi": hi,
                        "reason": action.reason,
                        "mode": int(s.mode),
                        "scr": f"0x{int(s.screen):02x}",
                        "rad": int(s.room_all_dead),
                        "ng": len(lg),
                        "g": [[int(o.x), int(o.y), int(o.hp), int(o.state)] for o in lg],
                    }
                )
                last_xy, last_phase, last_hi = xy, phase, hi
            if ctl.success or getattr(ctl, "failed", False):
                break
        s = read_snapshot(env.get_ram())
        out["end"] = {
            "f": f,
            "success": bool(ctl.success),
            "failed": bool(getattr(ctl, "failed", False)),
            "xy": [int(s.link_x), int(s.link_y)],
            "mode": int(s.mode),
            "scr": f"0x{int(s.screen):02x}",
            "candle": int(read_u8(env.get_ram(), ADDR_CANDLE)),
            "phase": getattr(ctl, "_phase", None),
            "hunt_i": getattr(ctl, "_hunt_i", None),
            "objects": _objs(s),
            "notes": list(getattr(ctl, "notes", [])),
        }
        ram = env.get_ram()
        if has_room_tile_map(ram):
            out["room_map"] = ascii_room(ram)
        out["reason_counts"] = dict(
            sorted(reason_counts.items(), key=lambda kv: -kv[1])
        )
        out["trace_len"] = len(trace)
        out["trace_head"] = trace[:60]
        out["trace_tail"] = trace[-80:]
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("trace_head", "trace_tail")}, indent=1))
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
