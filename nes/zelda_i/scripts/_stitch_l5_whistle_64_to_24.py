"""Level5Whistle64 → 0x65 → 0x66 → 0x56 → west corridor → Digdogger 0x24.

One fceumm session. Survival assist only. No key/door/item pokes.
Kill Digdogger only in 0x24 play mode with whistle=1. No Level5Complete
or STATUS unless TF bit 0x10 is RAM-true.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from dataclasses import replace

from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.level5_dungeon import LEVEL_5, ROOM_66_SPEC
from zelda_i.level5_path import (
    bomb_east_from_65,
    select_b_item_menu,
    walk_axis,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

TAG = "l5_whistle_64_to_24_stitch"
START = "Level5Whistle64"
STITCH_MAP = {
    0x07: "whistle cellar passage",
    0x14: "Triforce room",
    0x24: "Digdogger",
    0x25: "Pols Voice west key",
    0x26: "Gibdo moat",
    0x27: "mixed west key",
    0x37: "Darknut compass",
    0x47: "north Gibdos",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x64: "Blue Darknut stairs (Digdogger side)",
    0x65: "Gibdo bomb room",
    0x66: "Gibdo first key",
}
DIGDOGGER = 0x38


def pin(env):
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "room_name": STITCH_MAP.get(s.screen, f"L5 room 0x{s.screen:02x}"),
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce_0x0671": tf,
        "tf_l5_bit": bool(tf & 0x10),
        "doors": int(s.cur_opened_doors),
    }


def slim(rec):
    if not rec:
        return rec
    return {k: rec[k] for k in rec if k not in ("log", "steps", "progress", "menu")}


def door(env, assist, n, direction, ax, ay, expect):
    walk_axis(env, assist, n, "y", ay, max_f=400)
    walk_axis(env, assist, n, "x", ax, max_f=400)
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, n, direction, frames=240)
    idle(env, assist, n, 12)
    for _ in range(220):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen != room0:
            break
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    idle(env, assist, n, 8)
    s = read_snapshot(env.get_ram())
    return {
        "dir": direction,
        "expect": expect,
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "mode": s.mode,
        "success": s.level == LEVEL_5 and s.screen == expect and s.mode == PLAY_MODE,
    }


def kill_digdogger(env, assist, n):
    s = read_snapshot(env.get_ram())
    if not (s.screen == 0x24 and s.mode == PLAY_MODE and int(read_u8(env.get_ram(), ADDR_WHISTLE)) == 1):
        return {"ok": False, "reason": "not_24_play_whistle1", "room": s.screen, "mode": s.mode}
    menu = select_b_item_menu(env, assist, n, 5)
    for _ in range(6):
        env.step(nes_action("B"))
        n[0] += 1
        assist.apply_env(env, frame=n[0])
        idle(env, assist, n, 36)
    idle(env, assist, n, 40)
    for _ in range(2400):
        s = read_snapshot(env.get_ram())
        live = [
            o
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id == DIGDOGGER and o.hp > 0
        ]
        if not live:
            break
        env.step(nes_action("A"))
        n[0] += 1
        assist.apply_env(env, frame=n[0])
        if _ % 8 == 0:
            env.step(nes_action("UP", "A") if s.link_y > 141 else nes_action("DOWN", "A"))
            n[0] += 1
            assist.apply_env(env, frame=n[0])
    idle(env, assist, n, 20)
    s = read_snapshot(env.get_ram())
    live = [
        {"t": o.type_id, "hp": o.hp, "xy": [o.x, o.y]}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id == DIGDOGGER and o.hp > 0
    ]
    return {
        "ok": not live,
        "menu": menu,
        "live": live,
        "room": s.screen,
        "mode": s.mode,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def hunt_tf(env, assist, n):
    w0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 16)
    for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141)):
        if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) > w0:
            break
        walk_axis(env, assist, n, "y", ty, max_f=200)
        walk_axis(env, assist, n, "x", tx, max_f=200)
        idle(env, assist, n, 10)
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    s = read_snapshot(env.get_ram())
    return {
        "tf_in": w0,
        "tf_out": tf,
        "tf_l5": bool(tf & 0x10),
        "room": s.screen,
        "xy": [s.link_x, s.link_y],
    }


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle_64_to_24"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    hops = []
    start = None
    after = None
    blocker = None
    last_real = "0x64 Blue Darknut stairs"
    fight = None
    tf = None
    png = None
    dest_24 = False

    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 10)
        start = pin(env)
        after = start
        print("START", start, flush=True)
        if not (
            start["level"] == LEVEL_5
            and start["screen"] == 0x64
            and start["mode"] == PLAY_MODE
            and start["whistle"] == 1
        ):
            blocker = "start_not_whistle64"
        else:
            east = walk_east_from_64(env, assist, n)
            hops.append({"hop": "0x64_east", **slim(east)})
            after = pin(env)
            print("EAST64", hops[-1], after, flush=True)
            if after["screen"] == 0x07:
                last_real = "0x64 Blue Darknut stairs"
                blocker = "east64_dropped_cellar_0x07"
            elif after["screen"] != 0x65:
                last_real = f"0x{after['screen']:02x} {after['room_name']}"
                blocker = "east64_not_0x65"
            else:
                last_real = "0x65 Gibdo bomb room"
                be = bomb_east_from_65(env, assist, n)
                hops.append({"hop": "0x65_bomb_east", **slim(be)})
                after = pin(env)
                print("BOMB65", hops[-1], after, flush=True)
                if after["screen"] != 0x66:
                    last_real = "0x65 Gibdo bomb room"
                    blocker = "bomb_east_not_0x66"
                else:
                    last_real = "0x66 Gibdo first key"
                    spec = replace(
                        ROOM_66_SPEC,
                        source_room=0x65,
                        required_open_doors=0,
                        reward=replace(ROOM_66_SPEC.reward, kind=ROOM_66_SPEC.reward.kind),
                        max_frames=16000,
                    )
                    ctl = GenericDungeonRoomController(spec)
                    for _ in range(spec.max_frames):
                        snap = read_snapshot(env.get_ram())
                        action = ctl.step(snap)
                        env.step(action.action)
                        n[0] += 1
                        assist.apply_env(env, frame=n[0])
                        if ctl.success or ctl.phase is DungeonPhase.FAILED:
                            break
                    after = pin(env)
                    hops.append({
                        "hop": "0x66_clear",
                        "ok": bool(ctl.success),
                        "frames": ctl.frames,
                        "doors": after["doors"],
                    })
                    print("CLEAR66", hops[-1], after, flush=True)
                    if not ctl.success:
                        last_real = "0x66 Gibdo first key"
                        blocker = "66_gibdos_not_cleared"
                    else:
                        rec = door(env, assist, n, "UP", 120, 93, 0x56)
                        hops.append({"hop": "0x66_north", **rec})
                        after = pin(env)
                        print("NORTH66", rec, after, flush=True)
                        if not rec["success"]:
                            last_real = "0x66 Gibdo first key"
                            blocker = "north66_not_0x56"
                        else:
                            last_real = "0x56 north Dodongos"
                        chain = (
                            ("0x56_east", "RIGHT", 224, 141, 0x57, "0x57 east Zols"),
                            ("0x57_north", "UP", 120, 93, 0x47, "0x47 north Gibdos"),
                            ("0x47_north", "UP", 120, 93, 0x37, "0x37 Darknut compass"),
                            ("0x37_north", "UP", 120, 93, 0x27, "0x27 mixed west key"),
                        )
                        for hop, direction, ax, ay, expect, name in chain:
                            if blocker:
                                break
                            rec = door(env, assist, n, direction, ax, ay, expect)
                            hops.append({"hop": hop, **rec})
                            after = pin(env)
                            print(hop.upper(), rec, after["screen_hex"], flush=True)
                            if not rec["success"]:
                                last_real = f"0x{after['screen']:02x} {after['room_name']}"
                                blocker = f"{hop}_missed_0x{expect:02x}"
                            else:
                                last_real = name

                        if not blocker:
                            for fn, hop, expect, name in (
                                (walk_west_from_27, "0x27_west", 0x26, "0x26 Gibdo moat"),
                                (walk_west_from_26, "0x26_west", 0x25, "0x25 Pols Voice west key"),
                                (walk_west_from_25, "0x25_west", 0x24, "0x24 Digdogger"),
                            ):
                                rec = fn(env, assist, n)
                                for _ in range(240):
                                    s = read_snapshot(env.get_ram())
                                    if s.mode == PLAY_MODE and s.screen == expect and not s.transitioning:
                                        break
                                    env.step(nes_idle_action())
                                    n[0] += 1
                                    assist.apply_env(env, frame=n[0])
                                idle(env, assist, n, 8)
                                hops.append({"hop": hop, **slim(rec)})
                                after = pin(env)
                                print(hop.upper(), hops[-1], after, flush=True)
                                if after["screen"] != expect or after["mode"] != PLAY_MODE:
                                    last_real = f"0x{after['screen']:02x} {after['room_name']}"
                                    blocker = f"{hop}_missed_0x{expect:02x}"
                                    break
                                last_real = name
                            else:
                                dest_24 = after["screen"] == 0x24 and after["whistle"] == 1
                                if dest_24:
                                    fight = kill_digdogger(env, assist, n)
                                    hops.append({"hop": "digdogger", **slim(fight)})
                                    print("DIGDOGGER", fight, flush=True)
                                    if fight.get("ok"):
                                        tf = hunt_tf(env, assist, n)
                                        after = pin(env)
                                        print("TF", tf, after, flush=True)
                                    else:
                                        blocker = "digdogger_still_alive"

        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        png = out / f"{TAG}_final.png"
        save_rgb_png(obs, png)
        if dest_24 and after and after["screen"] == 0x24:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle24"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5Whistle24"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{START}.state",
                request={
                    "segment": "Level5Whistle24",
                    "predecessor_entry": True,
                    "start_state": START,
                    "via": "0x64 east 0x65 bomb-east 0x66 north 0x56 east corridor west to 0x24",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, "room": 0x24, "whistle_0x065C": after["whistle"]},
                natural_entry=False,
            )
        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    tf_bit = bool(after and after.get("tf_l5_bit"))
    report = {
        "ok": dest_24,
        "segment": TAG,
        "walkers": ["walk_east_from_64", "bomb_east_from_65", "walk_west_from_27/26/25"],
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": bool(tf_bit),
        "pokes": False,
        "start_state": START,
        "start": start,
        "hops": hops,
        "after": after,
        "dest_screen": None if after is None else after["screen_hex"],
        "dest_room_name": None if after is None else after["room_name"],
        "whistle_still_1": None if after is None else after["whistle"] == 1,
        "tf_l5_bit": None if after is None else after["tf_l5_bit"],
        "fought_digdogger": bool(fight),
        "digdogger": fight,
        "triforce": tf,
        "level5_complete_claimed": bool(tf_bit),
        "last_real_room": last_real,
        "total_frames": n[0],
        "png": str(png) if png else None,
        "bk2": str(bk2s[-1]) if bk2s else None,
        "blocker": blocker,
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} dest={after and after.get('screen_hex')} "
        f"ok={dest_24} last={last_real} blocker={blocker} tf={after and after.get('tf_l5_bit')}",
        flush=True,
    )
    return 0 if dest_24 else 2


if __name__ == "__main__":
    raise SystemExit(main())
