"""Dump 0x06 then apply 0x64 diamond method: push left 0x68 NORTH, idle (120,141).

Start: Level5WhistleFloor (0x05 play, whistle=1). EAST to 0x06.
No door/key pokes. No Clean STATUS. Do not re-get whistle.
"""
from __future__ import annotations

import zipfile

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_path import walk_axis, walk_east_from_05
from zelda_i.level9_stairs import on_stair_tile, on_warp_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, SHARED_ROM_ZIP
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5WhistleFloor"
CELLAR = (9, 10, 11, 16)
DOOR_CODES = {0: "open", 1: "wall", 2: "false", 3: "false2", 4: "bomb", 5: "key", 6: "key2", 7: "shutter"}
SECRETS = {0: "none", 1: "all_dead", 2: "ringleader", 3: "last_boss", 4: "block_door", 5: "block_stairs", 6: "money_or_life", 7: "foes_item"}


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def rom_room(room: int) -> dict:
    with zipfile.ZipFile(SHARED_ROM_ZIP) as zf:
        data = zf.read(zf.namelist()[0])

    def b(dc: int) -> int:
        return data[dc + 0x10]

    ns, ew, flags = b(0x18700 + room), b(0x18780 + room), b(0x18980 + room)
    n, s = (ns >> 5) & 7, (ns >> 2) & 7
    w, e = (ew >> 5) & 7, (ew >> 2) & 7
    secret = flags & 7
    return {
        "room": f"0x{room:02x}",
        "N": DOOR_CODES.get(n, str(n)),
        "S": DOOR_CODES.get(s, str(s)),
        "W": DOOR_CODES.get(w, str(w)),
        "E": DOOR_CODES.get(e, str(e)),
        "secret": SECRETS.get(secret, str(secret)),
        "secret_n": secret,
        "ns": f"0x{ns:02x}",
        "ew": f"0x{ew:02x}",
        "flags": f"0x{flags:02x}",
        "item": f"0x{b(0x18900 + room):02x}",
        "mon": f"0x{b(0x18800 + room):02x}",
    }


def dump(env) -> dict:
    s = read_snapshot(env.get_ram())
    objs = []
    blocks = []
    for o in s.objects:
        if not (1 <= o.slot <= 12) or o.type_id in (0, 0xFF):
            continue
        rec = {"slot": o.slot, "t": o.type_id, "th": f"0x{o.type_id:02x}", "hp": o.hp, "x": o.x, "y": o.y}
        objs.append(rec)
        if o.type_id == 0x68:
            blocks.append({"x": o.x, "y": o.y, "slot": o.slot})
    return {
        "sc": f"0x{s.screen:02x}",
        "next": f"0x{s.next_screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "tile_h": f"0x{int(s.colliding_tile):02x}",
        "stair": bool(on_stair_tile(s)),
        "warp": bool(on_warp_tile(s)),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "item": int(s.room_item_id),
        "all_dead": int(s.room_all_dead),
        "keys": int(s.keys),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "blocks": blocks,
        "objs": objs,
    }


def cellar_raw(env, assist, total, axis: str, target: int, max_f: int = 500) -> bool:
    """Axis walk that does NOT bail on mode 9."""
    last = None
    stall = 0
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x64, 0x06, 0x05):
            return True
        cur = s.link_x if axis == "x" else s.link_y
        if abs(cur - target) <= 1:
            return True
        if axis == "x":
            step(env, assist, total, nes_action("RIGHT" if s.link_x < target else "LEFT"))
        else:
            step(env, assist, total, nes_action("DOWN" if s.link_y < target else "UP"))
        s2 = read_snapshot(env.get_ram())
        pos = (s2.link_x, s2.link_y, s2.mode)
        if pos == last:
            stall += 1
            if stall >= 80:
                return False
        else:
            stall = 0
        last = pos
    return False


def warped(env) -> bool:
    s = read_snapshot(env.get_ram())
    return s.mode in CELLAR or s.screen == 0x07


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = None
    total = [1]
    log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump(env)
        print("START", start, flush=True)
        if start["whistle"] != 1:
            write_json_report(RECORDINGS_DIR / "l5_06_dump_push.json", {"ok": False, "reason": "whistle_not_1", "start": start})
            print("ABORT whistle", start["whistle"], flush=True)
            return
        if start["sc"] != "0x05":
            print("WARN start room", start["sc"], "expected 0x05", flush=True)

        east = walk_east_from_05(env, assist, total)
        idle(env, assist, total, 12)
        arrive = dump(env)
        rom06 = rom_room(0x06)
        rom64 = rom_room(0x64)
        rom65 = rom_room(0x65)
        rom16 = rom_room(0x16)
        print("EAST", east, flush=True)
        print("ARRIVE", arrive, flush=True)
        print("ROM06", rom06, flush=True)
        print("ROM64", rom64, flush=True)
        print("ROM65", rom65, flush=True)
        print("ROM16", rom16, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_arrive.png")
        log.append({"tag": "arrive", **arrive, "rom": rom06})

        if arrive["sc"] != "0x06":
            body = {"ok": False, "reason": "east_missed_06", "start": start, "arrive": arrive, "rom06": rom06, "pokes": False}
            write_json_report(RECORDINGS_DIR / "l5_06_dump_push.json", body)
            print("ABORT not in 0x06", flush=True)
            return

        # --- dump collision samples from north-around (do not take south key) ---
        samples = []
        walk_axis(env, assist, total, "x", 48, max_f=200)
        walk_axis(env, assist, total, "y", 93, max_f=400)
        rec = {"tag": "north_y93", **dump(env)}
        samples.append(rec)
        print("DUMP", rec["tag"], rec, flush=True)
        walk_axis(env, assist, total, "x", 64, max_f=300)
        rec = {"tag": "nw_x64", **dump(env)}
        samples.append(rec)
        print("DUMP", rec["tag"], rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_nw.png")

        # Sample a few north-band tiles without going south-door x-band
        for tx, ty in ((80, 93), (64, 109), (64, 125)):
            if warped(env):
                break
            walk_axis(env, assist, total, "y", ty, max_f=240)
            walk_axis(env, assist, total, "x", tx, max_f=240)
            rec = {"tag": f"sample_{tx}_{ty}", **dump(env)}
            samples.append(rec)
            print("DUMP", rec["tag"], "xy", rec["xy"], "tile", rec["tile_h"], "blocks", rec["blocks"], flush=True)

        # --- kill if needed (ignore 0x68 / bubbles) ---
        s = read_snapshot(env.get_ram())
        foes = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x68, 0x40, 0x2B, 0x55) and o.hp > 0]
        print("FOES", [{"t": o.type_id, "hp": o.hp, "x": o.x, "y": o.y} for o in foes], flush=True)
        if foes:
            for _ in range(4000):
                s = read_snapshot(env.get_ram())
                live = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x68, 0x40, 0x2B, 0x55) and o.hp > 0]
                if not live or warped(env):
                    break
                tgt = live[0]
                if abs(s.link_x - tgt.x) > 8:
                    step(env, assist, total, nes_action("RIGHT" if s.link_x < tgt.x else "LEFT", "A"))
                elif abs(s.link_y - tgt.y) > 8:
                    step(env, assist, total, nes_action("DOWN" if s.link_y < tgt.y else "UP", "A"))
                else:
                    step(env, assist, total, nes_action("A"))
            idle(env, assist, total, 12)
            rec = {"tag": "after_kill", **dump(env)}
            samples.append(rec)
            print("AFTER_KILL", rec, flush=True)

        if warped(env):
            print("WARPED during dump/kill", dump(env), flush=True)
        else:
            # Re-assert north-around then under left 0x68
            walk_axis(env, assist, total, "y", 93, max_f=400)
            walk_axis(env, assist, total, "x", 64, max_f=300)
            rec = {"tag": "re_nw", **dump(env)}
            samples.append(rec)
            print("RE_NW", rec, flush=True)

            # South along LEFT corridor, align under live 0x68, push NORTH
            walk_axis(env, assist, total, "y", 160, max_f=400)
            rec = {"tag": "left_south", **dump(env)}
            samples.append(rec)
            print("LEFT_SOUTH", rec, flush=True)

            s = read_snapshot(env.get_ram())
            blocks = [o for o in s.objects if 1 <= o.slot <= 12 and o.type_id == 0x68]
            bx = blocks[0].x if blocks else 112
            by = blocks[0].y if blocks else 144
            print("BLOCK_PRE", [{"x": o.x, "y": o.y} for o in blocks], "align_x", bx, flush=True)

            # Stand SOUTH of the block (not on stairs). Push NORTH.
            walk_axis(env, assist, total, "x", bx, max_f=300)
            walk_axis(env, assist, total, "y", min(by + 16, 170), max_f=300)
            rec = {"tag": "under_block", **dump(env)}
            samples.append(rec)
            print("UNDER", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_under68.png")

            push_dir(env, assist, total, "UP", frames=160)
            idle(env, assist, total, 16)
            rec = {"tag": "pushed_north", **dump(env)}
            samples.append(rec)
            print("PUSHED", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_pushed.png")

            # THEN stand (120,141) on 0x71 until room changes. Do not walk off.
            if not warped(env):
                walk_axis(env, assist, total, "y", 141, max_f=300)
                rec = {"tag": "y141", **dump(env)}
                samples.append(rec)
                print("Y141", rec, flush=True)
            if not warped(env):
                walk_axis(env, assist, total, "x", 120, max_f=300)
                rec = {"tag": "stand_120_141", **dump(env)}
                samples.append(rec)
                print("STAND", rec, flush=True)
                save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_stand.png")

            for i in range(360):
                if warped(env):
                    rec = {"tag": "warped", "f": i, **dump(env)}
                    samples.append(rec)
                    print("WARPED", rec, flush=True)
                    break
                step(env, assist, total, nes_idle_action())
            else:
                rec = {"tag": "idle_no_warp", **dump(env)}
                samples.append(rec)
                print("NO_WARP_AFTER_IDLE", rec, flush=True)

        after_method = dump(env)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_after_method.png")
        print("AFTER_METHOD", after_method, flush=True)

        cellar = None
        dest64 = None
        if warped(env) or after_method["sc"] == "0x07":
            # Climb LEFT mouth (0x64), not right 0x06 mouth x=192.
            # walk_axis bails on mode 9 — use cellar_raw.
            print("CELLAR_START", dump(env), flush=True)
            cellar_raw(env, assist, total, "y", 189, max_f=500)
            rec = {"tag": "cellar_floor", **dump(env)}
            samples.append(rec)
            print("CELLAR_FLOOR", rec, flush=True)
            cellar_raw(env, assist, total, "x", 48, max_f=600)
            rec = {"tag": "cellar_left", **dump(env)}
            samples.append(rec)
            print("CELLAR_LEFT", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_cellar_left.png")
            push_dir(env, assist, total, "UP", frames=280)
            idle(env, assist, total, 16)
            for _ in range(300):
                s = read_snapshot(env.get_ram())
                if s.mode == PLAY_MODE and s.screen == 0x64:
                    break
                if s.mode == PLAY_MODE and s.screen not in (0x07, 0x06):
                    break
                if s.mode in CELLAR:
                    step(env, assist, total, nes_action("UP"))
                else:
                    step(env, assist, total, nes_idle_action())
            idle(env, assist, total, 16)
            dest64 = dump(env)
            cellar = {"dest": dest64, "ok": dest64["sc"] == "0x64" and dest64["mode"] == 5}
            print("CELLAR_DEST", dest64, "OK", cellar["ok"], flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_06_dump_64.png")
            if cellar["ok"] and dest64["whistle"] == 1:
                write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle64"), env.em.get_state())
                write_state_provenance(
                    state_path(GAME_DIR, GAME, "Level5Whistle64"),
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                    request={
                        "segment": "Level5Whistle64",
                        "predecessor_entry": True,
                        "start_state": STATE,
                        "via": "0x06 dump, north-around y93 x64, push left 0x68 NORTH, idle 120,141, cellar left 0x64",
                        "key_poke": False,
                        "door_poke": False,
                        "bomb_count_poke": False,
                        "selected_item_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x64, "whistle_0x065C": 1, "xy": dest64["xy"]},
                    natural_entry=False,
                )
                print("SAVED Level5Whistle64", dest64, flush=True)

        warp_yes = bool(warped(env) or (after_method["sc"] == "0x07") or (dest64 and dest64["sc"] in ("0x07", "0x64")))
        reached64 = bool(dest64 and dest64["sc"] == "0x64" and dest64["mode"] == 5)
        body = {
            "ok": reached64,
            "warp_06": warp_yes,
            "reached_64": reached64,
            "pokes": False,
            "status_claim": None,
            "start_state": STATE,
            "start": start,
            "east": {k: east[k] for k in east},
            "arrive": arrive,
            "rom": {"0x06": rom06, "0x64": rom64, "0x65": rom65, "0x16": rom16},
            "samples": samples,
            "after_method": after_method,
            "cellar": cellar,
            "dest64": dest64,
            "final": dump(env),
            "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        }
        write_json_report(RECORDINGS_DIR / "l5_06_dump_push.json", body)
        print("FINAL", body["final"], "WARP", warp_yes, "R64", reached64, flush=True)
        if not warp_yes:
            print("STOP tiles/block/pose", after_method, flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
