"""Chain the red-candle chapter stages from OW_L7PondNatural as far as they go.

pond drain (0x42 -> 0x79) then every stage in
``level7_red_candle_chapter_stages()`` in order, stopping at the first
failure and dumping a deep diagnostic for that stage: position trace,
enemy census, occupancy misses (if the controller exposes a ``walker``),
and the ``$6530`` tile map of the room it failed in.  Read-only diagnostic.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_red_candle_chain.py --tag candle_chain_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.bomb_wall import BombWallPhase
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.tilemap import ascii_room, has_room_tile_map
from zelda_i.level7.hops import (
    level7_complete_chapter_stages,
    level7_red_candle_chapter_stages,
    make_pond_entry_controller,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_WHISTLE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist


def _objects(snap) -> list[dict]:
    out = []
    for o in snap.objects:
        t = int(o.type_id) & 0xFF
        if t in (0, 0xFF) or not (1 <= int(o.slot) <= 12):
            continue
        out.append(
            {
                "slot": int(o.slot),
                "type": f"0x{t:02x}:{object_name(t)}",
                "hp": int(o.hp),
                "state": int(o.state),
                "x": int(o.x),
                "y": int(o.y),
            }
        )
    return out


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "room_all_dead": int(s.room_all_dead),
        "objects": _objects(s),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="candle_chain_v1")
    ap.add_argument("--from-state", default="OW_L7PondNatural")
    ap.add_argument("--food", type=int, default=None)
    ap.add_argument("--complete", action="store_true", default=False)
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    out: dict = {"stages": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        if args.food is not None:
            env.unwrapped.data.memory.assign(int(ADDR_FOOD), "|u1", int(args.food) & 0xFF)
        out["start"] = _glance(env)

        drain_ctl = make_pond_entry_controller()
        bind = getattr(drain_ctl, "bind_env", None)
        if bind is not None:
            bind(env)
        f = 0
        while f < drain_ctl.max_frames + 10:
            snap = read_snapshot(env.get_ram())
            action = drain_ctl.step(snap)
            env.step(action.action)
            a.apply_env(env, frame=f)
            f += 1
            if drain_ctl.success or getattr(drain_ctl, "failed", False):
                break
        out["pond_drain_frames"] = f
        out["pond_drain_success"] = bool(drain_ctl.success)

        stages = list(level7_red_candle_chapter_stages())
        if args.complete:
            stages.extend(level7_complete_chapter_stages())
        for name, ctl, max_frames in stages:
            bind = getattr(ctl, "bind_env", None)
            if bind is not None:
                bind(env)
            f = 0
            trace = []
            last_xy = None
            while f < max_frames + 10:
                snap = read_snapshot(env.get_ram())
                action = ctl.step(snap)
                env.step(action.action)
                a.apply_env(env, frame=f)
                f += 1
                xy = (int(snap.link_x), int(snap.link_y))
                if xy != last_xy:
                    trace.append(
                        {
                            "f": f,
                            "xy": list(xy),
                            "reason": action.reason,
                            "screen": f"0x{int(snap.screen):02x}",
                        }
                    )
                last_xy = xy
                failed = getattr(ctl, "failed", None)
                if failed is None:
                    failed = getattr(ctl, "phase", None) is BombWallPhase.FAILED
                if ctl.success or failed:
                    break
            success = bool(ctl.success)
            rep = ctl.report() if hasattr(ctl, "report") else None
            stage_out = {
                "name": name,
                "frames": f,
                "success": success,
                "report": rep,
                "trace_len": len(trace),
                "trace_tail": trace[-40:],
            }
            out["stages"].append(stage_out)
            print(f"STAGE {name}: success={success} frames={f}")
            if not success:
                out["failed_stage"] = name
                out["failed_glance"] = _glance(env)
                misses = getattr(ctl, "walker", None)
                if misses is not None:
                    out["failed_misses"] = getattr(misses, "misses", None)
                ram = env.get_ram()
                if has_room_tile_map(ram):
                    out["failed_tilemap"] = ascii_room(ram)
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_{name}_fail.png"
                )
                break
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "stages"}, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
