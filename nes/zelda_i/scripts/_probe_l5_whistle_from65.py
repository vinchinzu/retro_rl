"""Reliable 0x65 west-wall bomb, west-mouth 0x64 stairs, then rest of whistle path.

No pokes. Stop on a specific room failure.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level9_stairs import (
    CELLAR_CORRIDOR_Y,
    CELLAR_EXIT_Y,
    CELLAR_LEFT_X,
    CELLAR_RIGHT_X,
    CELLAR_SPLIT_X,
    on_stair_tile,
    stair_transition_modes,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_04,
    ROOM_05,
    ROOM_06,
    ROOM_07,
    ROOM_64,
    ROOM_65,
    digdogger_and_tf,
    dump_and_save_room,
    dump_live,
    exit_cellar_other_mouth,
    fight_darknuts,
    hunt_item,
    key_west,
    live_darknuts,
    push_blocks,
    rom_room,
    select_bombs_menu,
    shot,
    step,
    take_stairs,
    walk_axis,
    wait_play,
    write_dump,
)

STATE = "Level5Cleared65"


def open_env(state=STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def bomb_west_wall(env, assist, total) -> dict:
    """West-wall column first (avoid 0x65 center diamond), stand (32,141), one bomb."""
    walk_axis(env, assist, total, "y", 109, max_f=300)
    walk_axis(env, assist, total, "x", 32, max_f=400)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 32, max_f=200)
    for _ in range(8):
        step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 6)
    before = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    before_png = shot(env, assist, total, "l5_65_west_bomb_before")
    menu = select_bombs_menu(env, assist, total)
    bombs0 = int(read_snapshot(env.get_ram()).bombs)
    room0 = int(read_snapshot(env.get_ram()).screen)
    step(env, assist, total, nes_action("LEFT", "B"))
    for _ in range(16):
        step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 100)
    for _ in range(220):
        snap = read_snapshot(env.get_ram())
        if snap.screen != room0 and snap.mode in (PLAY_MODE, 6, 7, 4):
            break
        step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 12)
    wait_play(env, assist, total, ROOM_64, max_f=280)
    idle(env, assist, total, 24)
    after = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    after_png = shot(env, assist, total, "l5_65_west_bomb")
    dest_changed = after.get("room") != before.get("room")
    rec = {
        "stand": [32, 141],
        "before_xy": [before.get("x"), before.get("y")],
        "menu": menu,
        "before": before,
        "after": after,
        "bombs_in": bombs0,
        "bombs_out": after.get("bombs"),
        "dest_changed": dest_changed,
        "dest": after.get("room_hex") if dest_changed else None,
        "before_screenshot": before_png,
        "screenshot": after_png,
        "pokes": False,
    }
    write_dump("l5_65_west_bomb", rec)
    print(
        "BOMB65W",
        dest_changed,
        before.get("room_hex"),
        "->",
        after.get("room_hex"),
        "xy",
        [after.get("x"), after.get("y")],
        "mode",
        after.get("mode"),
        "bombs",
        bombs0,
        "->",
        after.get("bombs"),
        flush=True,
    )
    return rec


def west_center_stairs(env, assist, total) -> dict:
    """From west mouth of 0x64, approach center stairs. Never go x>168 (east door)."""
    log = []
    paths = (
        (("y", 141), ("x", 80), ("x", 112), ("y", 141)),
        (("y", 109), ("x", 120), ("y", 125), ("x", 120)),
        (("y", 117), ("x", 120), ("y", 141)),
        (("y", 125), ("x", 120)),
        (("y", 157), ("x", 120), ("y", 141)),
        (("y", 173), ("x", 120), ("y", 157)),
        (("y", 109), ("x", 96), ("y", 141)),
        (("y", 109), ("x", 112), ("y", 141)),
        (("y", 109), ("x", 128), ("y", 141)),
        (("y", 141), ("x", 96), ("y", 125), ("x", 120)),
    )
    for steps in paths:
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_64:
            break
        rec_steps = []
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=350)
            snap = read_snapshot(env.get_ram())
            rec_steps.append(
                {
                    "axis": axis,
                    "tgt": tgt,
                    "ok": ok,
                    "xy": [snap.link_x, snap.link_y],
                    "tile": int(snap.colliding_tile),
                    "stair": bool(on_stair_tile(snap)),
                    "mode": snap.mode,
                    "room": f"0x{snap.screen:02x}",
                }
            )
            if snap.link_x > 168 and snap.mode == PLAY_MODE:
                # pull back west
                walk_axis(env, assist, total, "x", 120, max_f=200)
            if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                break
        for direction in ("UP", "DOWN"):
            for _ in range(50):
                snap = read_snapshot(env.get_ram())
                if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                    break
                if snap.link_x > 168:
                    step(env, assist, total, nes_action("LEFT"))
                    continue
                if snap.link_x < 40:
                    step(env, assist, total, nes_action("RIGHT"))
                    continue
                step(env, assist, total, nes_action(direction))
            snap = read_snapshot(env.get_ram())
            rec_steps.append(
                {
                    "nudge": direction,
                    "xy": [snap.link_x, snap.link_y],
                    "tile": int(snap.colliding_tile),
                    "stair": bool(on_stair_tile(snap)),
                    "mode": snap.mode,
                    "room": f"0x{snap.screen:02x}",
                }
            )
            if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.mode == PLAY_MODE):
                break
        log.append(rec_steps)
        snap = read_snapshot(env.get_ram())
        print(
            "PATH",
            steps,
            "end",
            [snap.link_x, snap.link_y],
            "room",
            f"0x{snap.screen:02x}",
            "mode",
            snap.mode,
            "tile",
            snap.colliding_tile,
            "stair",
            on_stair_tile(snap),
            flush=True,
        )
        if stair_transition_modes(snap.mode) or (snap.screen != ROOM_64 and snap.screen != ROOM_65):
            break
        if snap.screen == ROOM_65:
            break
    wait_play(env, assist, total, max_f=240)
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    dump = dump_live(snap, env.get_ram())
    png = shot(env, assist, total, "l5_64_stairs")
    ok = stair_transition_modes(snap.mode) or (snap.screen not in (ROOM_64, ROOM_65))
    write_dump(
        "l5_64_stairs",
        {
            "via": "0x64 west-mouth center stairs",
            "pokes": False,
            "status_claim": None,
            "ok": ok,
            "log": log,
            "dump": dump,
            "screenshot": png,
            "whistle_0x065C": dump.get("whistle_0x065C"),
            "rom": rom_room(int(snap.screen)),
        },
    )
    print("DEST l5_64_stairs", dump.get("room_hex"), "mode", snap.mode, "ok", ok, flush=True)
    return {"took": ok, "dump": dump, "log": log, "screenshot": png}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_whistle_from65.py  "
        "# west-wall bomb 0x65->0x64, west-mouth stairs, cellar 0x07->0x06, key 0x05, whistle 0x04"
    ]
    hops = []
    checkpoints = []
    roms = {r: rom_room(r) for r in (0x65, 0x64, 0x07, 0x06, 0x05, 0x04, 0x24, 0x14)}
    env, assist, obs = open_env()
    total = [1]
    whistle = 0
    boss = None
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", start.get("room_hex"), "bombs", start.get("bombs"), "keys", start.get("keys"), flush=True)
        bomb = bomb_west_wall(env, assist, total)
        hops.append({"hop": "0x65_west_bomb", "dest": bomb.get("dest"), "ok": bomb.get("dest_changed")})
        snap = read_snapshot(env.get_ram())
        if not bomb.get("dest_changed") or snap.screen != ROOM_64:
            report = {
                "ok": False,
                "failed_room": "0x65",
                "reason": "west_bomb_did_not_enter_0x64",
                "bomb": {k: bomb[k] for k in bomb if k not in ("before", "after")},
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d64 = dump_and_save_room(
            env, assist, total, "l5_64_arrive", "Level5Entered64", STATE, "0x65 WEST bomb west mouth"
        )
        checkpoints.append(d64["checkpoint"])
        stairs64 = west_center_stairs(env, assist, total)
        hops.append({"hop": "0x64_stairs", "dest": (stairs64.get("dump") or {}).get("room_hex"), "ok": stairs64.get("took")})
        if not stairs64.get("took"):
            report = {
                "ok": False,
                "failed_room": "0x64",
                "reason": "stairs_in_0x64_not_taken_from_west",
                "arrive64": d64["dump"],
                "stairs64": {k: v for k, v in stairs64.items() if k != "log"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
                "roms": roms,
            }
            write_dump("l5_whistle_path", report)
            return report

        d07 = dump_and_save_room(env, assist, total, "l5_07_arrive", "Level5Entered07", STATE, "0x64 stairs")
        checkpoints.append(d07["checkpoint"])
        cellar = exit_cellar_other_mouth(env, assist, total)
        hops.append({"hop": "0x07_other_mouth", "dest": (cellar.get("end") or {}).get("room_hex"), "ok": cellar.get("changed")})
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_06:
            report = {
                "ok": False,
                "failed_room": "0x07",
                "reason": "other_cellar_mouth_did_not_enter_0x06",
                "cellar": cellar,
                "d07": d07["dump"],
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d06 = dump_and_save_room(env, assist, total, "l5_06_arrive", "Level5Entered06", STATE, "cellar 0x07 other mouth")
        checkpoints.append(d06["checkpoint"])
        west = key_west(env, assist, total, ROOM_05)
        hops.append({"hop": "0x06_key_west", "dest": west.get("dest"), "key_spent": west.get("key_spent"), "ok": west.get("ok")})
        print("KEYWEST", west, flush=True)
        if not west.get("ok"):
            report = {
                "ok": False,
                "failed_room": "0x06",
                "reason": "key_west_did_not_enter_0x05",
                "west": west,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d05 = dump_and_save_room(env, assist, total, "l5_05_arrive", "Level5Entered05", STATE, "0x06 WEST key")
        n_dn = len(live_darknuts(read_snapshot(env.get_ram())))
        fight05 = fight_darknuts(env, assist, total, ROOM_05, expected=max(6, n_dn or 6), source=ROOM_06)
        idle(env, assist, total, 20)
        print("FIGHT05", fight05.get("ok"), "end_n", fight05.get("end_n"), flush=True)
        if not fight05.get("ok"):
            report = {
                "ok": False,
                "failed_room": "0x05",
                "reason": "darknuts_in_0x05_not_cleared",
                "fight05": {k: fight05[k] for k in fight05 if k != "controller"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        cleared05 = dump_and_save_room(env, assist, total, "l5_05_cleared", "Level5Cleared05", STATE, "0x05 all-dead")
        checkpoints.append(cleared05["checkpoint"])
        pushed = push_blocks(env, assist, total, ROOM_05)
        print("PUSH05", pushed.get("took"), pushed.get("blocks_seen"), flush=True)
        stairs05 = None
        snap = read_snapshot(env.get_ram())
        if pushed.get("took") or on_stair_tile(snap) or stair_transition_modes(snap.mode):
            stairs05 = take_stairs(env, assist, total, "l5_05_stairs", ROOM_05, PLAY_MODE)
        if stairs05 is None or not stairs05.get("took"):
            report = {
                "ok": False,
                "failed_room": "0x05",
                "reason": "block_stairs_from_0x05_not_taken",
                "push": {k: v for k, v in pushed.items() if k != "log"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        hops.append({"hop": "0x05_block_stairs", "dest": (stairs05.get("dump") or {}).get("room_hex"), "ok": True})
        d04 = dump_and_save_room(env, assist, total, "l5_04_whistle", "Level5Entered04", STATE, "0x05 block stairs")
        hunt_item(env, assist, total, ADDR_WHISTLE)
        idle(env, assist, total, 12)
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        final04 = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        png04 = shot(env, assist, total, "l5_04_whistle")
        write_dump(
            "l5_04_whistle",
            {
                "via": "0x05 block stairs",
                "pokes": False,
                "status_claim": None,
                "arrive": d04["dump"],
                "final": final04,
                "screenshot": png04,
                "whistle_0x065C": whistle,
                "rom": roms[0x04],
            },
        )
        print("WHISTLE", whistle, flush=True)
        if whistle < 1:
            report = {
                "ok": False,
                "failed_room": "0x04",
                "reason": "whistle_0x065C_still_0",
                "final04": final04,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        from zelda_i.scripts._probe_l5_whistle_path import save_ckpt
        ckpt_w = save_ckpt(
            env,
            "Level5Whistle",
            STATE,
            {
                "segment": "Level5Whistle",
                "predecessor_entry": True,
                "start_state": STATE,
                "via": "0x65 bomb -> 0x64 stairs -> 0x07 -> 0x06 key -> 0x05 block -> 0x04",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            {
                "success": True,
                "room": int(read_snapshot(env.get_ram()).screen),
                "whistle_0x065C": whistle,
                "bombs": int(read_snapshot(env.get_ram()).bombs),
                "keys": int(read_snapshot(env.get_ram()).keys),
            },
        )
        checkpoints.append(ckpt_w)
        hops.append({"hop": "0x04_whistle", "whistle_0x065C": whistle, "ok": True})
    finally:
        env.close()

    if whistle >= 1:
        boss = digdogger_and_tf()
    report = {
        "ok": whistle >= 1,
        "failed_room": None if whistle >= 1 else "see hops",
        "status_claim": None,
        "pokes": False,
        "commands": commands,
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_0x065C": whistle,
        "roms": roms,
        "digdogger": None
        if boss is None
        else {
            "ok": boss.get("ok"),
            "tf_room": boss.get("tf_room"),
            "tf_l5": boss.get("tf_l5"),
            "triforce_0x0671": boss.get("triforce_0x0671"),
        },
    }
    write_dump("l5_whistle_path", report)
    return report


if __name__ == "__main__":
    r = main()
    print("CMD", r.get("commands"))
    print("HOPS", r.get("hops"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("FAILED_ROOM", r.get("failed_room"))
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("status_claim", None)
