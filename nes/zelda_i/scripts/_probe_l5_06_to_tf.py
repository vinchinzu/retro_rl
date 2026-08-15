"""From latest whistle-on L5 floor save: 0x06 stairs -> 0x07 -> 0x64 -> Digdogger TF.

No door pokes. No Clean STATUS. Survival assist OK. No L6-L8.
If 0x06 stairs fail, STOP with tiles/mode/pose dump.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.level5_path import (
    cellar_other_mouth,
    fight_blue_darknuts,
    walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import on_stair_tile, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import (
    ROOM_06,
    ROOM_07,
    ROOM_14,
    ROOM_24,
    TF_BIT_L5,
    dest_report,
    dump_and_save_room,
    dump_live,
    fight_type,
    hunt_item,
    left_room,
    live_boss,
    push_blocks,
    rom_room,
    save_ckpt,
    select_whistle_menu,
    shot,
    step,
    wait_play,
    write_dump,
)

CANDIDATES = (
    "Level5Whistle06",
    "Level5Whistle05",
    "Level5WhistleFloor",
    "Level5Entered06",
    "Level5Entered05",
)

# Live 0x07 spawn into 0x06 was (96, 133) with 0x68 at (96, 128).
STAIR_STANDS_06 = (
    (96, 133),
    (96, 128),
    (96, 141),
    (96, 125),
    (104, 133),
    (88, 133),
    (112, 133),
    (80, 133),
    (96, 144),
    (96, 157),
    (80, 141),
    (112, 141),
    (120, 141),
    (120, 125),
    (64, 141),
    (96, 109),
    (128, 141),
    (96, 117),
)

DOOR_XY = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}


def pick_start() -> tuple[str, dict]:
    last = {}
    for name in CANDIDATES:
        path = GAME_DIR / "custom_integrations" / GAME / f"{name}.state"
        if not path.exists():
            last[name] = {"exists": False}
            continue
        env = make_env(GAME, name, GAME_DIR, render_mode="rgb_array")
        try:
            reset_obs(env)
            env.step(nes_idle_action())
            snap = read_snapshot(env.get_ram())
            w = int(read_u8(env.get_ram(), ADDR_WHISTLE))
            rec = {
                "exists": True,
                "room": f"0x{snap.screen:02x}",
                "mode": snap.mode,
                "xy": [snap.link_x, snap.link_y],
                "whistle": w,
                "level": snap.level,
            }
            last[name] = rec
            print("CANDIDATE", name, rec, flush=True)
            if (
                snap.level == 5
                and snap.mode == PLAY_MODE
                and w >= 1
                and snap.screen in (0x05, 0x06)
            ):
                return name, rec
        finally:
            env.close()
    return "", {"tried": last}


def in_cellar(snap) -> bool:
    if stair_transition_modes(snap.mode):
        return True
    return snap.level == 5 and snap.screen == ROOM_07


def door_hop(env, assist, total, direction: str, expect: int | None = None) -> dict:
    room0 = read_snapshot(env.get_ram()).screen
    ax, ay = DOOR_XY[direction]
    if direction in ("RIGHT", "LEFT"):
        walk_axis(env, assist, total, "y", ay, max_f=400)
        walk_axis(env, assist, total, "x", ax, max_f=500)
    else:
        walk_axis(env, assist, total, "x", ax, max_f=400)
        walk_axis(env, assist, total, "y", ay, max_f=500)
    goto(env, assist, total, ax, ay, tol=4, max_f=240)
    push_dir(env, assist, total, direction, frames=240)
    idle(env, assist, total, 12)
    wait_play(env, assist, total, max_f=240)
    idle(env, assist, total, 10)
    snap = read_snapshot(env.get_ram())
    rec = {
        "dir": direction,
        "from": f"0x{room0:02x}",
        "dest": f"0x{snap.screen:02x}",
        "changed": snap.screen != room0,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "ok": (expect is None and snap.screen != room0)
        or (expect is not None and snap.screen == expect),
    }
    print("HOP", rec, flush=True)
    return rec


def hunt_06_stairs(env, assist, total) -> dict:
    """Proven: off west door x=48, y=117, pinch (96,141), RIGHT onto 0x07.

    West doorway locks vertical motion. South-door spawn first uses the
    south band to the west side. Never treat 0x05/0x16 doors as stairs.
    """
    start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    log = [{"tag": "start", **{k: start.get(k) for k in ("x", "y", "mode", "room_hex", "colliding_tile")}}]
    snap = read_snapshot(env.get_ram())
    if snap.link_y >= 180:
        walk_axis(env, assist, total, "y", 189, max_f=300)
        walk_axis(env, assist, total, "x", 48, max_f=400)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        snap = read_snapshot(env.get_ram())
        log.append({"tag": "from_south", "xy": [snap.link_x, snap.link_y]})
    # Off the west door, then north of the diamond, pinch, RIGHT.
    for axis, tgt in (("x", 48), ("y", 117), ("x", 120), ("y", 141)):
        ok = walk_axis(env, assist, total, axis, tgt, max_f=400)
        snap = read_snapshot(env.get_ram())
        rec = {
            "step": f"{axis}:{tgt}",
            "ok": ok,
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
        }
        log.append(rec)
        print("ST06", rec, flush=True)
        if in_cellar(snap) or (snap.screen != ROOM_06 and snap.screen not in (0x05, 0x16)):
            return {"took": True, "via": f"{axis}:{tgt}", "end": dump_live(snap, env.get_ram()), "log": log}
    for direction in ("RIGHT", "DOWN", "UP"):
        for _ in range(40):
            snap = read_snapshot(env.get_ram())
            if in_cellar(snap) or snap.screen != ROOM_06:
                break
            if snap.link_y >= 200:
                step(env, assist, total, nes_action("UP"))
                continue
            if snap.link_x <= 24:
                step(env, assist, total, nes_action("RIGHT"))
                continue
            step(env, assist, total, nes_action(direction))
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        rec = {
            "nudge": direction,
            "xy": [snap.link_x, snap.link_y],
            "tile": int(snap.colliding_tile),
            "stair": bool(on_stair_tile(snap)),
            "mode": snap.mode,
            "room": f"0x{snap.screen:02x}",
        }
        log.append(rec)
        print("NUDGE06", rec, flush=True)
        if in_cellar(snap) or (snap.screen != ROOM_06 and snap.screen not in (0x05, 0x16)):
            wait_play(env, assist, total, max_f=240)
            snap = read_snapshot(env.get_ram())
            return {"took": True, "via": f"nudge {direction}", "end": dump_live(snap, env.get_ram()), "log": log}
    for _ in range(200):
        snap = read_snapshot(env.get_ram())
        if in_cellar(snap) or (snap.mode == PLAY_MODE and snap.screen != ROOM_06):
            break
        step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 16)
    snap = read_snapshot(env.get_ram())
    if in_cellar(snap) or (snap.screen != ROOM_06 and snap.screen not in (0x05, 0x16)):
        return {"took": True, "via": "settle", "end": dump_live(snap, env.get_ram()), "log": log}
    end = dump_live(snap, env.get_ram())
    png = shot(env, assist, total, "l5_06_stairs_fail")
    return {
        "took": False,
        "via": "fail",
        "start": start,
        "end": end,
        "log": log,
        "tiles": log,
        "screenshot": png,
        "pose": {
            "x": snap.link_x,
            "y": snap.link_y,
            "facing": snap.facing,
            "mode": snap.mode,
            "tile": int(snap.colliding_tile),
        },
    }


def off_64_stairs_then_east(env, assist, total) -> dict:
    """Arrive on 0x64 center stairs; walk south gap then east bomb hole -> 0x65."""
    # Step off center stairs so we do not drop back into 0x07.
    walk_axis(env, assist, total, "y", 173, max_f=360)
    if in_cellar(read_snapshot(env.get_ram())):
        return {"ok": False, "reason": "reentered_cellar"}
    walk_axis(env, assist, total, "x", 176, max_f=400)
    walk_axis(env, assist, total, "y", 141, max_f=360)
    walk_axis(env, assist, total, "x", 224, max_f=400)
    rec = door_hop(env, assist, total, "RIGHT", 0x65)
    if not rec["ok"]:
        walk_axis(env, assist, total, "y", 189, max_f=300)
        walk_axis(env, assist, total, "x", 224, max_f=400)
        walk_axis(env, assist, total, "y", 141, max_f=300)
        rec = door_hop(env, assist, total, "RIGHT", 0x65)
    return rec


def fight_if_needed(env, assist, total, room: int) -> dict | None:
    snap = read_snapshot(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x40, 0x4E, 0x55, 0x68, 0x5A) and o.hp > 0
    ]
    if not live:
        return None
    t0 = live[0].type_id
    if t0 in (0x0B, 0x0C):
        return fight_blue_darknuts(env, assist, total, room, expected=len(live), source=room)
    return fight_type(env, assist, total, room, t0, expected=len(live))


def digdogger_here(env, assist, total, source: str) -> dict:
    snap = read_snapshot(env.get_ram())
    at24 = dump_live(snap, env.get_ram())
    print("AT24", at24.get("room_hex"), "objs", [(o["type_hex"], o["hp"]) for o in at24.get("objects") or []], flush=True)
    menu = select_whistle_menu(env, assist, total)
    for _ in range(5):
        step(env, assist, total, nes_action("B"))
        idle(env, assist, total, 50)
    idle(env, assist, total, 80)
    after_b = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    print("WHISTLE_B", menu, [(o["type_hex"], o["hp"]) for o in after_b.get("objects") or []], flush=True)
    fight = None
    bosses = live_boss(read_snapshot(env.get_ram()))
    if bosses:
        fight = fight_type(env, assist, total, ROOM_24, 0x38, expected=len(bosses))
        idle(env, assist, total, 16)
        print("BOSS", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)
    leftovers = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x40, 0x4E, 0x55, 0x68) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = fight_type(env, assist, total, ROOM_24, leftovers[0].type_id, expected=len(leftovers))
        idle(env, assist, total, 12)
    # Heart 0x1A then north 0x14.
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    idle(env, assist, total, 12)
    after_heart = dump_live(read_snapshot(env.get_ram()), env.get_ram())
    room0 = read_snapshot(env.get_ram()).screen
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    push_dir(env, assist, total, "UP", frames=240)
    idle(env, assist, total, 16)
    wait_play(env, assist, total, max_f=240)
    snap = read_snapshot(env.get_ram())
    tf_dump = None
    if snap.screen != room0:
        tf_dump = dump_and_save_room(
            env, assist, total, f"l5_{snap.screen:02x}_triforce", "Level5Triforce", source, "0x24 north after Digdogger"
        )
    tf_walk = None
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    snap = read_snapshot(env.get_ram())
    if snap.room_item_id == 0x1B or snap.screen == ROOM_14:
        tf_walk = hunt_item(env, assist, total, ADDR_TRIFORCE)
        idle(env, assist, total, 20)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    final = dump_live(snap, ram)
    png = shot(env, assist, total, "l5_24_whistle_boss")
    rec = {
        "ok": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT_L5),
        "at24": at24,
        "menu": menu,
        "after_whistle": after_b,
        "fight": fight,
        "extra": extra,
        "after_heart": after_heart,
        "tf": tf_dump,
        "tf_walk": None if tf_walk is None else {k: v for k, v in tf_walk.items() if k != "hits"},
        "tf_in": tf0,
        "tf_out": int(final.get("triforce_0x0671") or 0),
        "tf_l5": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT_L5),
        "final": final,
        "screenshot": png,
        "pokes": False,
        "status_claim": None,
    }
    write_dump("l5_24_whistle_boss", rec)
    print("DIGDOGGER", rec["ok"], "tf", hex(rec["tf_out"]), "room", final.get("room_hex"), flush=True)
    return rec


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    hops = []
    checkpoints = []
    source, confirm = pick_start()
    print("START_STATE", source, confirm, flush=True)
    if not source:
        body = {
            "ok": False,
            "reason": "no_whistle_on_floor_save_in_0x05_or_0x06",
            "confirm": confirm,
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_06_to_tf.json", body)
        return body

    env = make_env(GAME, source, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START", start.get("room_hex"), "whistle", start.get("whistle_0x065C"), "xy", [start.get("x"), start.get("y")], flush=True)
        shot(env, assist, total, "l5_06_to_tf_start")
        hops.append({"hop": "start", "state": source, **{k: start[k] for k in ("room_hex", "mode", "whistle_0x065C") if k in start}, "xy": [start.get("x"), start.get("y")]})

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x05:
            rec = door_hop(env, assist, total, "RIGHT", ROOM_06)
            hops.append({"hop": "05_east", **rec})
            if not rec["ok"]:
                body = {
                    "ok": False,
                    "reason": "0x05_east_missed_0x06",
                    "start_state": source,
                    "start": start,
                    "hops": hops,
                    "pokes": False,
                    "status_claim": None,
                    "l6_l8": False,
                }
                write_json_report(RECORDINGS_DIR / "l5_06_to_tf.json", body)
                return body
            dump_and_save_room(env, assist, total, "l5_w06_from05", "Level5Whistle06b", source, "0x05 EAST")
            checkpoints.append("Level5Whistle06b")

        stairs = hunt_06_stairs(env, assist, total)
        hops.append({"hop": "06_stairs", "took": stairs.get("took"), "via": stairs.get("via"), "end": stairs.get("end")})
        print("STAIRS06", stairs.get("took"), stairs.get("via"), stairs.get("end", {}).get("room_hex"), stairs.get("end", {}).get("mode"), flush=True)
        shot(env, assist, total, "l5_06_stairs")
        snap = read_snapshot(env.get_ram())
        if not stairs.get("took") or (snap.screen == ROOM_06 and not in_cellar(snap)):
            fail = {
                "ok": False,
                "reason": "0x06_stairs_failed",
                "start_state": source,
                "start_room": start.get("room_hex"),
                "whistle_0x065C": start.get("whistle_0x065C"),
                "stairs": {
                    "took": stairs.get("took"),
                    "via": stairs.get("via"),
                    "pose": stairs.get("pose"),
                    "tiles": stairs.get("tiles"),
                    "end": stairs.get("end"),
                    "screenshot": stairs.get("screenshot"),
                },
                "dest": dest_report(snap),
                "hops": hops,
                "pokes": False,
                "status_claim": None,
                "l6_l8": False,
                "checkpoints": checkpoints,
            }
            write_dump("l5_06_stairs_fail", fail)
            write_json_report(RECORDINGS_DIR / "l5_06_to_tf.json", fail)
            print("STOP stairs fail pose", stairs.get("pose"), "tiles", (stairs.get("tiles") or [])[:8], flush=True)
            return fail

        # Cellar 0x07
        wait_play(env, assist, total, max_f=240)
        snap = read_snapshot(env.get_ram())
        d07 = dump_and_save_room(env, assist, total, "l5_07_from06", "Level5Whistle07", source, "0x06 stairs")
        checkpoints.append("Level5Whistle07")
        hops.append({"hop": "06_to_07", "dest": d07["dump"].get("room_hex"), "mode": d07["dump"].get("mode")})
        print("AT07", d07["dump"].get("room_hex"), "mode", d07["dump"].get("mode"), "xy", [d07["dump"].get("x"), d07["dump"].get("y")], flush=True)

        cellar = cellar_other_mouth(env, assist, total)
        hops.append({"hop": "07_other", "dest": f"0x{cellar.get('dest'):02x}" if isinstance(cellar.get("dest"), int) else cellar.get("dest"), "xy": cellar.get("xy"), "mode": cellar.get("mode"), "chose": cellar.get("chose_side")})
        print("CELLAR", cellar.get("dest"), cellar.get("xy"), cellar.get("mode"), cellar.get("chose_side"), flush=True)
        wait_play(env, assist, total, max_f=200)
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x64:
            # cellar_other_mouth success flag expects 0x06; try explicit opposite if still in 0x07.
            if in_cellar(snap) or snap.screen == ROOM_07:
                cellar2 = cellar_other_mouth(env, assist, total)
                hops.append({"hop": "07_other_retry", "dest": cellar2.get("dest"), "xy": cellar2.get("xy")})
                wait_play(env, assist, total, max_f=200)
                snap = read_snapshot(env.get_ram())
        d64 = dump_and_save_room(env, assist, total, "l5_64_from07", "Level5Whistle64", source, "0x07 other mouth")
        checkpoints.append("Level5Whistle64")
        hops.append({"hop": "07_to_64", "dest": d64["dump"].get("room_hex"), "mode": d64["dump"].get("mode")})
        print("AT64", d64["dump"].get("room_hex"), d64["dump"].get("mode"), [d64["dump"].get("x"), d64["dump"].get("y")], flush=True)
        if read_snapshot(env.get_ram()).screen != 0x64:
            body = {
                "ok": False,
                "reason": "other_mouth_missed_0x64",
                "start_state": source,
                "start_room": start.get("room_hex"),
                "hops": hops,
                "checkpoints": checkpoints,
                "final": dump_live(read_snapshot(env.get_ram()), env.get_ram()),
                "pokes": False,
                "status_claim": None,
                "l6_l8": False,
            }
            write_json_report(RECORDINGS_DIR / "l5_06_to_tf.json", body)
            return body

        # 0x64 EAST -> 0x65
        e65 = off_64_stairs_then_east(env, assist, total)
        hops.append({"hop": "64_east", **e65})
        if e65.get("ok"):
            dump_and_save_room(env, assist, total, "l5_w65", "Level5Whistle65", source, "0x64 EAST bomb hole")
            checkpoints.append("Level5Whistle65")
            fight_if_needed(env, assist, total, 0x65)

        route = (
            ("65_up", "UP", 0x55),
            ("55_right", "RIGHT", 0x56),
            ("56_right", "RIGHT", 0x57),
            ("57_up", "UP", 0x47),
            ("47_up", "UP", 0x37),
            ("37_up", "UP", 0x27),
        )
        for name, direction, expect in route:
            snap = read_snapshot(env.get_ram())
            if snap.screen == expect:
                hops.append({"hop": name, "already": True, "dest": f"0x{expect:02x}"})
                continue
            # Special: 0x65 has a center diamond — y=109 then x=120 then UP.
            if name == "65_up" and snap.screen == 0x65:
                walk_axis(env, assist, total, "y", 109, max_f=360)
                walk_axis(env, assist, total, "x", 120, max_f=360)
            fight_if_needed(env, assist, total, snap.screen)
            rec = door_hop(env, assist, total, direction, expect)
            hops.append({"hop": name, **rec})
            if rec["ok"]:
                ck = f"Level5Whistle{expect:02X}"
                dump_and_save_room(env, assist, total, f"l5_w{expect:02x}", ck, source, f"{name}")
                checkpoints.append(ck)
            else:
                print("HOP_FAIL", name, rec, flush=True)
                break

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x27:
            w = walk_west_from_27(env, assist, total)
            hops.append({"hop": "27_west", "dest": f"0x{w.get('dest'):02x}" if isinstance(w.get("dest"), int) else w.get("dest"), "ok": w.get("success")})
            print("W27", w.get("success"), w.get("dest"), flush=True)
        if read_snapshot(env.get_ram()).screen == 0x26:
            dump_and_save_room(env, assist, total, "l5_w26", "Level5Whistle26", source, "0x27 WEST")
            checkpoints.append("Level5Whistle26")
            w = walk_west_from_26(env, assist, total)
            hops.append({"hop": "26_west", "dest": f"0x{w.get('dest'):02x}" if isinstance(w.get("dest"), int) else w.get("dest"), "ok": w.get("success")})
            print("W26", w.get("success"), w.get("dest"), flush=True)
        if read_snapshot(env.get_ram()).screen == 0x25:
            dump_and_save_room(env, assist, total, "l5_w25", "Level5Whistle25", source, "0x26 WEST")
            checkpoints.append("Level5Whistle25")
            w = walk_west_from_25(env, assist, total)
            hops.append({"hop": "25_west", "dest": f"0x{w.get('dest'):02x}" if isinstance(w.get("dest"), int) else w.get("dest"), "ok": w.get("success")})
            print("W25", w.get("success"), w.get("dest"), flush=True)

        boss = None
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_24:
            dump_and_save_room(env, assist, total, "l5_w24", "Level5Whistle24", source, "0x25 WEST Digdogger door")
            checkpoints.append("Level5Whistle24")
            boss = digdogger_here(env, assist, total, source)
            hops.append({"hop": "digdogger", "ok": boss.get("ok"), "tf_l5": boss.get("tf_l5"), "tf": boss.get("tf_out")})
            if boss.get("ok"):
                checkpoints.append("Level5Triforce")

        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        body = {
            "ok": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT_L5),
            "start_state": source,
            "start_room": start.get("room_hex"),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "stairs_06_07_64": any(h.get("hop") == "07_to_64" and h.get("dest") == "0x64" for h in hops),
            "digdogger": None if boss is None else {k: boss[k] for k in boss if k not in ("at24", "after_whistle", "after_heart", "final")},
            "tf_bit_0x10": bool(int(final.get("triforce_0x0671") or 0) & TF_BIT_L5),
            "triforce_0x0671": int(final.get("triforce_0x0671") or 0),
            "final_room": final.get("room_hex"),
            "hops": hops,
            "checkpoints": checkpoints,
            "final": final,
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
            "rom06": rom_room(0x06),
            "rom07": rom_room(0x07),
            "rom64": rom_room(0x64),
        }
        write_json_report(RECORDINGS_DIR / "l5_06_to_tf.json", body)
        print("FINAL", body["ok"], "room", body["final_room"], "tf", hex(body["triforce_0x0671"]), "ck", checkpoints, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "stairs_path", r.get("stairs_06_07_64"), "tf", r.get("tf_bit_0x10"), flush=True)
