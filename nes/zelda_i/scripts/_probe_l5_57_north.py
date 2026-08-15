"""Solve L5 0x57 north -> 0x47 from Level5Whistle57, then locked TF route.

Start: Level5Whistle57 (whistle=1, west mouth, 5 Zols). ROM N=open / S=wall /
W=open / E=wall / secret=foes_item. Prior Level5Whistle47 was via 57_up from
this pin; post-clear combat can leave Link on y\approx125 with object 0x5f at
(128,128). Try north first (no clear). If that fails: clear Zols, dump 0x5f,
push it, bomb-north, key-north. Do not restart 0x04/0x06. No pokes. No L6-L8.
No Complete until TF bit 0x10. One env, stop_record.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_in_room_26,
    level5_room_25_cleared,
    level5_room_26_cleared,
    level5_room_27_cleared,
)
from zelda_i.level5_path import (
    select_b_item_menu,
    walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_06_to_tf import digdogger_here, door_hop, fight_if_needed
from zelda_i.scripts._probe_l5_whistle_path import dump_and_save_room, dump_live, fight_type, shot
from zelda_i.scripts._stitch_l4_to_24_door import walk_north_from_47
from zelda_i.scripts._stitch_whistle_to_tf import spec_47

STATE = "Level5Whistle57"
TF_BIT = 0x10
OBJ_5F = 0x5F
ZOL = (0x13, 0x14, 0x15)


def step(env, assist, total, action):
    env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])


def wait_play(env, assist, total, room=None, max_f=360):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            if room is None or s.screen == room:
                idle(env, assist, total, 12)
                return True
        step(env, assist, total, nes_idle_action())
    return False


def pin(env):
    ram = env.get_ram()
    s = read_snapshot(ram)
    tf = int(read_u8(ram, ADDR_TRIFORCE))
    return {
        "room": f"0x{s.screen:02x}",
        "mode": s.mode,
        "level": s.level,
        "xy": [s.link_x, s.link_y],
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "tile": int(s.colliding_tile),
        "all_dead": int(s.room_all_dead),
        "item": int(s.room_item_id),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "tf": tf,
        "tf_l5": bool(tf & TF_BIT),
    }


def objs(env):
    out = []
    for o in read_snapshot(env.get_ram()).objects:
        if not (1 <= o.slot <= 12) or o.type_id in (0, 0xFF):
            continue
        out.append(
            {
                "slot": o.slot,
                "type": o.type_id,
                "type_hex": f"0x{o.type_id:02x}",
                "name": object_name(o.type_id),
                "hp": o.hp,
                "x": o.x,
                "y": o.y,
            }
        )
    return out


def live_zols(env):
    return [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id in ZOL and o.hp > 0
    ]


def obj_5f(env):
    return [o for o in objs(env) if o["type"] == OBJ_5F]


def dump_block(env, tag):
    s = read_snapshot(env.get_ram())
    body = {
        "tag": tag,
        "pin": pin(env),
        "objects": objs(env),
        "obj_5f": obj_5f(env),
        "compact": compact_snapshot(s),
        "pokes": False,
    }
    png = RECORDINGS_DIR / f"{tag}.png"
    save_rgb_png(env.step(nes_idle_action())[0], png)
    body["screenshot"] = str(png.resolve())
    write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
    print("DUMP", tag, body["pin"], "5f", body["obj_5f"], flush=True)
    return body


def hop_north_57(env, assist, total, label):
    """ROM-open north. Avoid the y=125 ladder lock; bank then x=120 UP."""
    notes = []
    s = read_snapshot(env.get_ram())
    start = {"xy": [s.link_x, s.link_y], "5f": obj_5f(env), **pin(env)}
    if abs(s.link_y - 125) <= 6:
        for tx in (32, 48, 208, 192, 64, 176):
            walk_axis(env, assist, total, "x", tx, max_f=360)
            s = read_snapshot(env.get_ram())
            notes.append(f"bank_x{tx}:xy={s.link_x},{s.link_y}")
            if s.screen == 0x47:
                return {"ok": True, "via": f"{label}:bank_x{tx}", "notes": notes, "start": start, **pin(env)}
            if walk_axis(env, assist, total, "y", 141, max_f=80) or walk_axis(
                env, assist, total, "y", 109, max_f=80
            ):
                notes.append(f"off_ladder_{s.link_x}")
                break
    for name, steps in (
        ("y141_x120_up", (("y", 141), ("x", 120), ("y", 93))),
        ("y109_x120_up", (("y", 109), ("x", 120), ("y", 93))),
        ("y173_x120_up", (("y", 173), ("x", 120), ("y", 93))),
        ("x96_y109_x120", (("x", 96), ("y", 109), ("x", 120), ("y", 93))),
        ("x160_y109_x120", (("x", 160), ("y", 109), ("x", 120), ("y", 93))),
        ("x128_y93", (("x", 128), ("y", 93))),
    ):
        for axis, tgt in steps:
            ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
            s = read_snapshot(env.get_ram())
            notes.append(f"{name}:{axis}:{tgt}:ok={ok}:xy={s.link_x},{s.link_y}")
            if s.screen == 0x47:
                wait_play(env, assist, total, 0x47)
                return {"ok": True, "via": f"{label}:{name}", "notes": notes, "start": start, **pin(env)}
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - 120) <= 8 and s.link_y <= 109:
            break
    goto(env, assist, total, 120, 93, tol=3, max_f=240)
    push_dir(env, assist, total, "UP", frames=280)
    idle(env, assist, total, 16)
    wait_play(env, assist, total, 0x47, max_f=240)
    s = read_snapshot(env.get_ram())
    ok = s.level == LEVEL_5 and s.screen == 0x47 and s.mode == PLAY_MODE
    return {
        "ok": ok,
        "via": f"{label}:push_up",
        "notes": notes,
        "start": start,
        "end_5f": obj_5f(env),
        **pin(env),
    }


def push_5f(env, assist, total):
    """Walk into 0x5f from all four sides. Success = object moved or room 0x47."""
    found = obj_5f(env)
    log = []
    if not found:
        return {"moved": False, "found": [], "log": log, **pin(env)}
    bx, by = found[0]["x"], found[0]["y"]
    stands = (
        ("south_up", bx, by + 16, "UP"),
        ("north_down", bx, by - 16, "DOWN"),
        ("west_right", bx - 16, by, "RIGHT"),
        ("east_left", bx + 16, by, "LEFT"),
        ("s128_up", 128, 144, "UP"),
        ("n128_down", 128, 112, "DOWN"),
        ("w112_right", 112, 128, "RIGHT"),
        ("e144_left", 144, 128, "LEFT"),
        ("s120_up", 120, 144, "UP"),
        ("w96_right", 96, 128, "RIGHT"),
    )
    before = [(o["slot"], o["x"], o["y"]) for o in found]
    for name, sx, sy, direction in stands:
        walk_axis(env, assist, total, "y", sy, max_f=280)
        walk_axis(env, assist, total, "x", sx, max_f=280)
        goto(env, assist, total, sx, sy, tol=4, max_f=160)
        push_dir(env, assist, total, direction, frames=120)
        idle(env, assist, total, 8)
        s = read_snapshot(env.get_ram())
        now = obj_5f(env)
        rec = {
            "stand": name,
            "at": [s.link_x, s.link_y],
            "dir": direction,
            "room": f"0x{s.screen:02x}",
            "5f": now,
            "doors": int(s.cur_opened_doors),
            "mask": int(s.open_doorway_mask),
        }
        log.append(rec)
        print("PUSH5F", rec, flush=True)
        if s.screen == 0x47:
            wait_play(env, assist, total, 0x47)
            return {"moved": True, "via": name, "found": found, "log": log, "ok47": True, **pin(env)}
        after = [(o["slot"], o["x"], o["y"]) for o in now]
        if after and after != before:
            return {"moved": True, "via": name, "found": found, "after": now, "log": log, **pin(env)}
    return {"moved": False, "found": found, "after": obj_5f(env), "log": log, **pin(env)}


def bomb_north_57(env, assist, total):
    """One bomb at the north mouth. Dest must become 0x47. Menu select, no poke."""
    s = read_snapshot(env.get_ram())
    if int(s.bombs) <= 0:
        return {"ok": False, "reason": "no_bombs", **pin(env)}
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    walk_axis(env, assist, total, "y", 109, max_f=300)
    goto(env, assist, total, 120, 109, tol=3, max_f=200)
    for _ in range(8):
        step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 6)
    menu = select_b_item_menu(env, assist, total, 1)
    bombs0 = int(read_snapshot(env.get_ram()).bombs)
    room0 = int(read_snapshot(env.get_ram()).screen)
    step(env, assist, total, nes_action("UP", "B"))
    for _ in range(12):
        step(env, assist, total, nes_action("DOWN"))
    idle(env, assist, total, 100)
    push_dir(env, assist, total, "UP", frames=280)
    idle(env, assist, total, 16)
    wait_play(env, assist, total, 0x47, max_f=240)
    s = read_snapshot(env.get_ram())
    ok = s.level == LEVEL_5 and s.screen == 0x47 and s.mode == PLAY_MODE
    return {
        "ok": ok,
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(s.bombs),
        "bombs_spent": bombs0 - int(s.bombs),
        "from": room0,
        **pin(env),
    }


def key_north_57(env, assist, total):
    """Push the north mouth; a key door spends a key. Dest must become 0x47."""
    keys0 = int(read_snapshot(env.get_ram()).keys)
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    goto(env, assist, total, 120, 93, tol=3, max_f=200)
    push_dir(env, assist, total, "UP", frames=280)
    idle(env, assist, total, 16)
    wait_play(env, assist, total, 0x47, max_f=240)
    s = read_snapshot(env.get_ram())
    ok = s.level == LEVEL_5 and s.screen == 0x47 and s.mode == PLAY_MODE
    return {
        "ok": ok,
        "keys_in": keys0,
        "keys_out": int(s.keys),
        "key_spent": int(s.keys) < keys0,
        **pin(env),
    }


def fight_spec(env, spec, assist, total, controller=None):
    ctl = controller or GenericDungeonRoomController(spec)
    for _ in range(spec.max_frames):
        if assist is not None:
            assist.apply_env(env, frame=total[0])
        action = ctl.step(read_snapshot(env.get_ram()))
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    return ctl


def continue_from_47(env, assist, total, hops, checkpoints):
    """0x47 -> 0x37 -> 0x27 -> 0x26 -> 0x25 -> 0x24 whistle-shrink -> TF 0x10."""
    live = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id == 0x30 and o.hp > 0
    ]
    if live:
        ctl = fight_spec(env, spec_47(), assist, total)
        hops.append({"hop": "clear47", "ok": bool(ctl.success), "frames": ctl.frames})
        print("CLEAR47", ctl.success, flush=True)
        idle(env, assist, total, 16)
    dump_and_save_room(env, assist, total, "l5_w47c", "Level5Whistle47Cleared", STATE, "0x57 north then gibdo")
    checkpoints.append("Level5Whistle47Cleared")

    rec = walk_north_from_47(env, assist, total)
    wait_play(env, assist, total, 0x37)
    ok = read_snapshot(env.get_ram()).screen == 0x37
    hops.append({"hop": "47_up", "ok": ok, "dest": rec.get("dest"), "used": rec.get("used")})
    print("HOP47", ok, rec.get("result_room"), rec.get("used"), flush=True)
    if not ok:
        rec = door_hop(env, assist, total, "UP", 0x37)
        hops.append({"hop": "47_up_center", **rec})
        ok = rec.get("ok")
    if not ok:
        return "fail_hop_47_to_37"
    dump_and_save_room(env, assist, total, "l5_w37", "Level5Whistle37", STATE, "0x47 UP")
    checkpoints.append("Level5Whistle37")
    fight_if_needed(env, assist, total, 0x37)
    idle(env, assist, total, 12)

    rec = door_hop(env, assist, total, "UP", 0x27)
    hops.append({"hop": "37_up", **rec})
    if not rec.get("ok"):
        rec2 = walk_north_from_47(env, assist, total)
        wait_play(env, assist, total, 0x27)
        ok = read_snapshot(env.get_ram()).screen == 0x27
        hops.append({"hop": "37_up_pinch", "ok": ok, "dest": rec2.get("dest")})
        if not ok:
            return "fail_hop_37_to_27"
    dump_and_save_room(env, assist, total, "l5_w27", "Level5Whistle27", STATE, "0x37 UP")
    checkpoints.append("Level5Whistle27")
    ctl = fight_spec(env, ROOM_27_SPEC, assist, total)
    ok27 = level5_room_27_cleared(env.get_ram()) or ctl.success
    hops.append({"hop": "clear27", "ok": ok27, "frames": ctl.frames})
    print("CLEAR27", ok27, flush=True)
    if not ok27:
        return "fail_clear_27"

    w = walk_west_from_27(env, assist, total)
    wait_play(env, assist, total, 0x26)
    ok = level5_in_room_26(env.get_ram()) or read_snapshot(env.get_ram()).screen == 0x26
    hops.append({"hop": "27_west", "ok": ok, "dest": w.get("dest"), "success": w.get("success")})
    print("W27", ok, w.get("dest"), flush=True)
    if not ok:
        return "fail_hop_27_to_26"
    dump_and_save_room(env, assist, total, "l5_w26", "Level5Whistle26", STATE, "0x27 WEST")
    checkpoints.append("Level5Whistle26")
    ctl = fight_spec(env, ROOM_26_SPEC, assist, total)
    ok26 = level5_room_26_cleared(env.get_ram()) or ctl.success
    hops.append({"hop": "clear26", "ok": ok26, "frames": ctl.frames})
    print("CLEAR26", ok26, flush=True)
    if not ok26:
        return "fail_clear_26"

    w = walk_west_from_26(env, assist, total)
    wait_play(env, assist, total, 0x25)
    ok = level5_in_room_25(env.get_ram()) or read_snapshot(env.get_ram()).screen == 0x25
    hops.append({"hop": "26_west", "ok": ok, "dest": w.get("dest")})
    print("W26", ok, w.get("dest"), flush=True)
    if not ok:
        return "fail_hop_26_to_25"
    dump_and_save_room(env, assist, total, "l5_w25", "Level5Whistle25", STATE, "0x26 WEST")
    checkpoints.append("Level5Whistle25")
    ctl = fight_spec(
        env, ROOM_25_SPEC, assist, total, controller=Level5PolsVoiceController(spec=ROOM_25_SPEC)
    )
    ok25 = level5_room_25_cleared(env.get_ram()) or ctl.success
    hops.append({"hop": "clear25", "ok": ok25, "frames": ctl.frames})
    print("CLEAR25", ok25, flush=True)
    if not ok25:
        return "fail_clear_25"

    w = walk_west_from_25(env, assist, total)
    wait_play(env, assist, total, 0x24)
    ok = level5_in_room_24(env.get_ram()) or read_snapshot(env.get_ram()).screen == 0x24
    hops.append({"hop": "25_west", "ok": ok, "dest": w.get("dest")})
    print("W25", ok, w.get("dest"), flush=True)
    if not ok:
        return "fail_hop_25_to_24"
    dump_and_save_room(env, assist, total, "l5_w24", "Level5Whistle24", STATE, "0x25 WEST Digdogger")
    checkpoints.append("Level5Whistle24")
    boss = digdogger_here(env, assist, total, STATE)
    hops.append(
        {"hop": "digdogger", "ok": boss.get("ok"), "tf_l5": boss.get("tf_l5"), "tf": boss.get("tf_out")}
    )
    if boss.get("ok"):
        checkpoints.append("Level5Triforce")
    return None if boss.get("ok") else "tf_bit_0x10_not_set"


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    movie = RECORDINGS_DIR / "stitches" / "bk2_57_north_to_tf"
    movie.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array", record=str(movie))
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    attempts = []
    checkpoints = [STATE]
    blocker = None
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        start_pin = pin(env)
        print("START57", start_pin, "5f", obj_5f(env), "zols", len(live_zols(env)), flush=True)
        if start_pin["whistle"] != 1 or start_pin["room"] != "0x57":
            blocker = "bad_start_pin"
            dump_block(env, "l5_57_bad_start")
        else:
            rec = hop_north_57(env, assist, total, "preclear")
            attempts.append({"try": "north_before_clear", **rec})
            print("PRECLEAR_NORTH", rec.get("ok"), rec.get("via"), rec.get("xy"), flush=True)
            if rec.get("ok"):
                dump_and_save_room(
                    env, assist, total, "l5_w47", "Level5Whistle47", STATE, "0x57 UP before Zol clear"
                )
                checkpoints.append("Level5Whistle47")
                hops.append({"hop": "57_up_preclear", **rec})
            else:
                dump_block(env, "l5_57_preclear_fail")
                live = live_zols(env)
                if live:
                    fight = fight_type(env, assist, total, 0x57, 0x13, expected=len(live))
                    hops.append(
                        {"hop": "clear57", **{k: fight[k] for k in ("ok", "frames", "end_n") if k in fight}}
                    )
                    print("CLEAR57", fight.get("ok"), "end_n", fight.get("end_n"), pin(env), flush=True)
                    idle(env, assist, total, 20)
                after = dump_block(env, "l5_57_after_zol")
                attempts.append({"try": "after_zol_dump", "pin": after["pin"], "obj_5f": after["obj_5f"]})

                rec = hop_north_57(env, assist, total, "postclear")
                attempts.append({"try": "north_after_clear", **rec})
                print("POSTCLEAR_NORTH", rec.get("ok"), rec.get("via"), rec.get("xy"), flush=True)
                if rec.get("ok"):
                    dump_and_save_room(
                        env, assist, total, "l5_w47", "Level5Whistle47", STATE, "0x57 UP after Zol clear"
                    )
                    checkpoints.append("Level5Whistle47")
                    hops.append({"hop": "57_up_postclear", **rec})
                else:
                    pushed = push_5f(env, assist, total)
                    slim = {k: pushed[k] for k in pushed if k != "log"}
                    slim["log"] = pushed.get("log")
                    attempts.append({"try": "push_5f", **slim})
                    print("PUSH", pushed.get("moved"), pushed.get("via"), pushed.get("after"), flush=True)
                    if pushed.get("ok47") or read_snapshot(env.get_ram()).screen == 0x47:
                        wait_play(env, assist, total, 0x47)
                        dump_and_save_room(
                            env, assist, total, "l5_w47", "Level5Whistle47", STATE, "0x57 push 0x5f"
                        )
                        checkpoints.append("Level5Whistle47")
                        hops.append({"hop": "57_push_5f", "ok": True, "via": pushed.get("via")})
                    else:
                        if pushed.get("moved"):
                            rec = hop_north_57(env, assist, total, "after_push")
                            attempts.append({"try": "north_after_push", **rec})
                            print("AFTER_PUSH_NORTH", rec.get("ok"), rec.get("xy"), flush=True)
                            if rec.get("ok"):
                                dump_and_save_room(
                                    env,
                                    assist,
                                    total,
                                    "l5_w47",
                                    "Level5Whistle47",
                                    STATE,
                                    "0x57 UP after 0x5f push",
                                )
                                checkpoints.append("Level5Whistle47")
                                hops.append({"hop": "57_up_after_push", **rec})
                        if read_snapshot(env.get_ram()).screen != 0x47:
                            bomb = bomb_north_57(env, assist, total)
                            attempts.append({"try": "bomb_north", **bomb})
                            print("BOMB_N", bomb.get("ok"), bomb.get("bombs_spent"), bomb.get("xy"), flush=True)
                            if bomb.get("ok"):
                                dump_and_save_room(
                                    env, assist, total, "l5_w47", "Level5Whistle47", STATE, "0x57 bomb north"
                                )
                                checkpoints.append("Level5Whistle47")
                                hops.append({"hop": "57_bomb_north", **bomb})
                        if read_snapshot(env.get_ram()).screen != 0x47:
                            key = key_north_57(env, assist, total)
                            attempts.append({"try": "key_north", **key})
                            print("KEY_N", key.get("ok"), key.get("key_spent"), key.get("xy"), flush=True)
                            if key.get("ok"):
                                dump_and_save_room(
                                    env, assist, total, "l5_w47", "Level5Whistle47", STATE, "0x57 key north"
                                )
                                checkpoints.append("Level5Whistle47")
                                hops.append({"hop": "57_key_north", **key})
                        if read_snapshot(env.get_ram()).screen != 0x47:
                            blocker = "fail_hop_57_to_47"
                            dump_block(env, "l5_57_north_fail")

            if read_snapshot(env.get_ram()).screen == 0x47 and blocker is None:
                blocker = continue_from_47(env, assist, total, hops, checkpoints)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        shot(env, assist, total, "l5_57_north_final")
        if hasattr(env, "stop_record"):
            env.stop_record()
        body = {
            "ok": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT) and blocker is None,
            "start_state": STATE,
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "ladder_0x0663": int(read_u8(ram, ADDR_LADDER)),
            "blocker": blocker,
            "tf_bit_0x10": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT),
            "triforce_0x0671": int(final.get("triforce_0x0671") or 0),
            "final_room": final.get("room_hex"),
            "final_xy": [snap.link_x, snap.link_y],
            "final_doors": int(snap.cur_opened_doors),
            "final_mask": int(snap.open_doorway_mask),
            "final_5f": obj_5f(env),
            "attempts": attempts,
            "hops": hops,
            "checkpoints": checkpoints,
            "start": start,
            "final": final,
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
            "stairs_04_06_restarted": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_57_north.json", body)
        print(
            "FINAL",
            body["ok"],
            "blocker",
            blocker,
            "room",
            body["final_room"],
            "tf",
            hex(body["triforce_0x0671"]),
            "ck",
            checkpoints,
            flush=True,
        )
        return body
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "blocker", r.get("blocker"), "tf", r.get("tf_bit_0x10"), flush=True)
