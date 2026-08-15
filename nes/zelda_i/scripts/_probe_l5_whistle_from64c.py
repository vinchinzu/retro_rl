"""From Level5Cleared64: proven stairs -> 0x07 -> 0x06 key -> 0x05 -> 0x04 whistle.

Proven take: y=117, x=120 (pinches at 96), y=141, DOWN, RIGHT onto (128,141) -> 0x07.
No pokes. Do not invent detours.
"""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_04,
    ROOM_05,
    ROOM_06,
    ROOM_07,
    ROOM_64,
    dest_report,
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
    save_ckpt,
    shot,
    step,
    take_stairs,
    walk_axis,
    wait_play,
    write_dump,
)

STATE = "Level5Cleared64"


def open_env(state=STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def take_64_stairs(env, assist, total) -> dict:
    """Exact hit path d117_r120_d141: pinch at (96,141), RIGHT onto stairs."""
    log = []
    for axis, tgt in (("y", 117), ("x", 120), ("y", 141)):
        ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
        snap = read_snapshot(env.get_ram())
        rec = {
            "axis": axis,
            "tgt": tgt,
            "ok": ok,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
        }
        log.append(rec)
        print("STAIRNAV", rec, flush=True)
        if stair_transition_modes(snap.mode) or (
            snap.screen != ROOM_64 and snap.mode in (PLAY_MODE, 16, 9, 10, 11)
        ):
            break
    for direction in ("DOWN", "RIGHT", "UP", "LEFT"):
        for _ in range(40):
            snap = read_snapshot(env.get_ram())
            if stair_transition_modes(snap.mode) or (
                snap.screen != ROOM_64 and snap.mode != 7
            ):
                break
            if snap.link_x > 190:
                step(env, assist, total, nes_action("LEFT"))
                continue
            step(env, assist, total, nes_action(direction))
        snap = read_snapshot(env.get_ram())
        rec = {
            "nudge": direction,
            "xy": [snap.link_x, snap.link_y],
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
        }
        log.append(rec)
        print("STAIRNUDGE", rec, flush=True)
        if stair_transition_modes(snap.mode) or snap.screen != ROOM_64:
            break
    # Let stairs-enter (16) settle into cellar play (9) if it will.
    for _ in range(240):
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_07 and snap.mode in (9, 11, PLAY_MODE):
            break
        if snap.screen != ROOM_64 and snap.mode in (9, 11, PLAY_MODE):
            break
        if snap.screen == ROOM_64 and snap.mode == PLAY_MODE and not stair_transition_modes(snap.mode):
            # still in 0x64 after nudges — stop waiting
            if not stair_transition_modes(snap.mode):
                pass
        step(env, assist, total, nes_idle_action())
        if snap.screen != ROOM_64 and snap.mode in (9, 11, PLAY_MODE, 16):
            if snap.mode in (9, 11, PLAY_MODE):
                break
    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    dump = dump_live(snap, env.get_ram())
    png = shot(env, assist, total, "l5_64_stairs")
    dest = dump.get("room_hex")
    ok = dest == "0x07" or (stair_transition_modes(snap.mode) and dest != "0x64")
    write_dump(
        "l5_64_stairs",
        {
            "via": "cleared64 d117_r120_d141 + RIGHT",
            "pokes": False,
            "status_claim": None,
            "ok": ok,
            "log": log,
            "dump": dump,
            "dest": dest_report(snap),
            "screenshot": png,
            "whistle_0x065C": dump.get("whistle_0x065C"),
            "rom": rom_room(int(snap.screen)),
        },
    )
    print("DEST64", dest, "mode", snap.mode, "ok", ok, "xy", [snap.link_x, snap.link_y], flush=True)
    return {"took": ok, "dump": dump, "screenshot": png, "log": log}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_65_west_bomb.py ; "
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_64_clear_stairs.py ; "
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_whistle_from64c.py  "
        "# 0x65 W bomb, 0x64 5/5 0x0C, d117+RIGHT stairs 0x07, other mouth 0x06, key 0x05, whistle 0x04"
    ]
    hops = [
        {"hop": "0x65_west_bomb", "dest": "0x64", "ok": True, "bombs": "4->3"},
        {"hop": "0x64_clear", "dest": "0x64", "ok": True, "darknuts": "5/5 type 0x0c"},
    ]
    checkpoints = ["Level5Entered64", "Level5Cleared64"]
    roms = {r: rom_room(r) for r in (0x65, 0x64, 0x07, 0x06, 0x05, 0x04, 0x24, 0x14)}
    env, assist, obs = open_env()
    total = [1]
    whistle = 0
    boss = None
    try:
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print(
            "START",
            start.get("room_hex"),
            [start.get("x"), start.get("y")],
            "dn",
            len(live_darknuts(read_snapshot(env.get_ram()))),
            "bombs",
            start.get("bombs"),
            "keys",
            start.get("keys"),
            flush=True,
        )
        stairs64 = take_64_stairs(env, assist, total)
        hops.append(
            {
                "hop": "0x64_stairs",
                "dest": (stairs64.get("dump") or {}).get("room_hex"),
                "ok": stairs64.get("took"),
            }
        )
        if not stairs64.get("took"):
            report = {
                "ok": False,
                "failed_room": "0x64",
                "reason": "stairs_not_taken_from_cleared64",
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d07 = dump_and_save_room(
            env, assist, total, "l5_07_arrive", "Level5Entered07", STATE, "0x64 d117+RIGHT stairs"
        )
        checkpoints.append(d07["checkpoint"])
        cellar = exit_cellar_other_mouth(env, assist, total)
        hops.append(
            {
                "hop": "0x07_other_mouth",
                "dest": (cellar.get("end") or {}).get("room_hex"),
                "ok": cellar.get("changed"),
                "side": cellar.get("chose_side"),
            }
        )
        snap = read_snapshot(env.get_ram())
        print(
            "CELLAR",
            f"0x{snap.screen:02x}",
            "mode",
            snap.mode,
            [snap.link_x, snap.link_y],
            "side",
            cellar.get("chose_side"),
            flush=True,
        )
        if snap.screen != ROOM_06:
            report = {
                "ok": False,
                "failed_room": "0x07",
                "reason": "other_mouth_not_0x06",
                "cellar": {k: v for k, v in cellar.items() if k != "end"} | {"end_room": (cellar.get("end") or {}).get("room_hex")},
                "d07": d07["dump"],
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d06 = dump_and_save_room(
            env, assist, total, "l5_06_arrive", "Level5Entered06", STATE, "cellar 0x07 other mouth"
        )
        checkpoints.append(d06["checkpoint"])
        west = key_west(env, assist, total, ROOM_05)
        hops.append(
            {
                "hop": "0x06_key_west",
                "dest": west.get("dest"),
                "key_spent": west.get("key_spent"),
                "ok": west.get("ok"),
            }
        )
        print("KEYWEST", west, flush=True)
        if not west.get("ok"):
            report = {
                "ok": False,
                "failed_room": "0x06",
                "reason": "key_west_not_0x05",
                "west": west,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d05 = dump_and_save_room(
            env, assist, total, "l5_05_arrive", "Level5Entered05", STATE, "0x06 WEST key"
        )
        n_dn = len(live_darknuts(read_snapshot(env.get_ram())))
        fight05 = fight_darknuts(env, assist, total, ROOM_05, expected=max(6, n_dn or 6), source=ROOM_06)
        idle(env, assist, total, 20)
        print("FIGHT05", fight05.get("ok"), "end", fight05.get("end_n"), "f", fight05.get("frames"), flush=True)
        if not fight05.get("ok"):
            report = {
                "ok": False,
                "failed_room": "0x05",
                "reason": "darknuts_not_cleared",
                "fight05": {k: fight05[k] for k in fight05 if k != "controller"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        cleared05 = dump_and_save_room(
            env, assist, total, "l5_05_cleared", "Level5Cleared05", STATE, "0x05 all-dead"
        )
        checkpoints.append(cleared05["checkpoint"])
        pushed = push_blocks(env, assist, total, ROOM_05)
        print("PUSH05", pushed.get("took"), pushed.get("blocks_seen"), flush=True)
        stairs05 = None
        snap = read_snapshot(env.get_ram())
        dest05 = (pushed.get("dest") or {}).get("room_hex") if pushed.get("dest") else None
        if dest05 == "0x04" or (pushed.get("took") and snap.screen == ROOM_04):
            stairs05 = {"took": True, "dump": pushed.get("dest") or dump_live(snap, env.get_ram())}
        elif pushed.get("took") or on_stair_tile(snap) or stair_transition_modes(snap.mode):
            stairs05 = take_stairs(env, assist, total, "l5_05_stairs", ROOM_05, PLAY_MODE)
        if stairs05 is None or not stairs05.get("took"):
            report = {
                "ok": False,
                "failed_room": "0x05",
                "reason": "block_stairs_not_taken",
                "push": {k: v for k, v in pushed.items() if k != "log"},
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        dest_s = (stairs05.get("dump") or {}).get("room_hex")
        hops.append({"hop": "0x05_block_stairs", "dest": dest_s, "ok": dest_s == "0x04"})
        if dest_s != "0x04":
            report = {
                "ok": False,
                "failed_room": "0x05",
                "reason": "stairs_dest_not_0x04",
                "dest": dest_s,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        d04 = dump_and_save_room(
            env, assist, total, "l5_04_whistle", "Level5Entered04", STATE, "0x05 block stairs"
        )
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
                "reason": "whistle_still_0",
                "final04": final04,
                "hops": hops,
                "commands": commands,
                "pokes": False,
                "status_claim": None,
            }
            write_dump("l5_whistle_path", report)
            return report
        ckpt_w = save_ckpt(
            env,
            "Level5Whistle",
            STATE,
            {
                "segment": "Level5Whistle",
                "predecessor_entry": True,
                "start_state": STATE,
                "via": "0x64 stairs 0x07 0x06 key 0x05 block 0x04",
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
            "failed_room": boss.get("failed_room"),
            "reason": boss.get("reason"),
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
