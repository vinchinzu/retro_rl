"""Diagnose the level7_room69_west_bomb blocker from OW_L7PondNatural.

Chains: pond drain (0x42 -> 0x79) -> entry north door (0x79 -> 0x69) ->
make_room69_west_bomb_controller (0x69 west BOMB wall -> 0x68), dumping
Link's position trace, enemy census, occupancy misses, and the room 0x69
$6530 tile map.  Read-only diagnostic; drives no new writes.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_room69_west_diag.py --tag room69_west_diag_v1
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
    make_entry_first_door_controller,
    make_pond_entry_controller,
    make_room68_north_controller,
    make_room69_west_bomb_controller,
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
        if t in (0, 0xFF):
            continue
        if not (1 <= int(o.slot) <= 12):
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


def _run_stage(env, ctl, assist, out: dict, name: str, trace: list, sample_every: int = 25) -> dict:
    bind = getattr(ctl, "bind_env", None)
    if bind is not None:
        bind(env)
    f = 0
    rep = None
    last_xy = None
    while f < ctl.max_frames + 10:
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        env.step(action.action)
        assist.apply_env(env, frame=f)
        f += 1
        xy = (int(snap.link_x), int(snap.link_y))
        if name in ("room69_west_bomb", "room68_north") and (
            xy != last_xy and f % sample_every == 0 or xy != last_xy
        ):
            if f % 4 == 0 or xy != last_xy:
                trace.append(
                    {
                        "f": f,
                        "xy": list(xy),
                        "phase": getattr(ctl, "phase", None).name
                        if hasattr(ctl, "phase")
                        else None,
                        "reason": action.reason,
                        "screen": f"0x{int(snap.screen):02x}",
                    }
                )
        last_xy = xy
        failed = getattr(ctl, "failed", None)
        if failed is None:
            failed = getattr(ctl, "phase", None) is BombWallPhase.FAILED
        if ctl.success or failed:
            rep = ctl.report()
            break
    out[f"{name}_frames"] = f
    out[f"{name}_report"] = rep
    return rep or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="room69_west_diag_v1")
    ap.add_argument("--from-state", default="OW_L7PondNatural")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    out: dict = {}
    trace: list = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)

        drain_ctl = make_pond_entry_controller()
        _run_stage(env, drain_ctl, a, out, "pond_drain", trace)

        door_ctl = make_entry_first_door_controller()
        _run_stage(env, door_ctl, a, out, "entry_first_door", trace)

        # Census + tile map right at 0x69 arrival, before the bomb-wall
        # controller takes over.
        arrival_snap = read_snapshot(env.get_ram())
        out["room69_arrival"] = _glance(env)
        ram = env.get_ram()
        if has_room_tile_map(ram):
            out["room69_tilemap"] = ascii_room(ram)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_arrival.png")

        wall_ctl = make_room69_west_bomb_controller()
        _run_stage(env, wall_ctl, a, out, "room69_west_bomb", trace)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_after_bomb.png")

        if wall_ctl.success:
            north_ctl = make_room68_north_controller()
            _run_stage(env, north_ctl, a, out, "room68_north", trace)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_after_68north.png")

        out["trace"] = trace
        out["end"] = _glance(env)
        out["end"]["deaths"] = int(a.telemetry.deaths)
        out["end"]["progression_writes"] = int(a.telemetry.progression_writes)
        out["end"]["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"]["position_writes"] = int(
            getattr(a.telemetry, "position_writes", 0)
        )
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
