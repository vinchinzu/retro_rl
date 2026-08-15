"""From Level5Whistle65: 0x65 N shutter is sealed (never opened).

ROM 0x65 N=shutter S=wall W=bomb E=bomb. Honest one-bomb EAST -> 0x66
(N shutter was opened on the original 0x66 UP -> 0x56). Then 0x56 RIGHT
-> 0x57 UP -> 0x47 UP -> 0x37 UP -> 0x27 WEST -> 0x26 WEST -> 0x25 WEST
-> 0x24 Digdogger / TF 0x10. No door pokes. No L6-L8.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import goto, idle, push_dir
from zelda_i.level5_path import (
    select_b_item_menu,
    walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_06_to_tf import (
    digdogger_here,
    door_hop,
    fight_if_needed,
)
from zelda_i.scripts._probe_l5_whistle_path import (
    dump_and_save_room,
    dump_live,
    shot,
    step,
    wait_play,
)

STATE = "Level5Whistle65"


def bomb_east_65(env, assist, total) -> dict:
    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 208, max_f=500)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 208, max_f=200)
    goto(env, assist, total, 208, 141, tol=3, max_f=300)
    for _ in range(8):
        step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 1)
    snap = read_snapshot(env.get_ram())
    bombs0 = int(snap.bombs)
    room0 = int(snap.screen)
    shot(env, assist, total, "l5_65_east_bomb_before")
    step(env, assist, total, nes_action("RIGHT", "B"))
    for _ in range(16):
        step(env, assist, total, nes_action("LEFT"))
    idle(env, assist, total, 100)
    for _ in range(360):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == 0x66:
            break
        step(env, assist, total, nes_action("RIGHT"))
    idle(env, assist, total, 20)
    wait_play(env, assist, total, max_f=240)
    idle(env, assist, total, 12)
    snap = read_snapshot(env.get_ram())
    rec = {
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "from": f"0x{room0:02x}",
        "dest": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "ok": snap.screen == 0x66 and snap.mode == PLAY_MODE,
    }
    print("BOMB65E", rec, flush=True)
    shot(env, assist, total, "l5_65_east_bomb")
    return rec


def leave_66_north(env, assist, total) -> dict:
    """West bomb-hole spawn: off river/ladder, x=120, UP through opened shutter."""
    snap = read_snapshot(env.get_ram())
    log = [{"start": [snap.link_x, snap.link_y]}]
    # If on the west ladder (y<141, x<80), finish down first.
    if snap.link_y < 141 and snap.link_x < 80:
        walk_axis(env, assist, total, "y", 141, max_f=300)
        log.append({"ladder": [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]})
    if read_snapshot(env.get_ram()).link_y > 185:
        walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    rec = door_hop(env, assist, total, "UP", 0x56)
    rec["log"] = log
    if not rec["ok"]:
        walk_axis(env, assist, total, "y", 109, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        rec = door_hop(env, assist, total, "UP", 0x56)
    return rec


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    checkpoints = ["Level5Whistle07", "Level5Whistle64", "Level5Whistle65"]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START65", start.get("room_hex"), "whistle", start.get("whistle_0x065C"), [start.get("x"), start.get("y")], "doors", start.get("doors"), flush=True)
        hops.append({"hop": "start", "room": start.get("room_hex"), "whistle": start.get("whistle_0x065C")})

        # One more honest shutter try (already failed once).
        walk_axis(env, assist, total, "y", 109, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        up = door_hop(env, assist, total, "UP", 0x55)
        hops.append({"hop": "65_up_shutter", **up})
        if not up["ok"]:
            bomb = bomb_east_65(env, assist, total)
            hops.append({"hop": "65_east_bomb", **bomb})
            if bomb["ok"]:
                dump_and_save_room(env, assist, total, "l5_w66", "Level5Whistle66", STATE, "0x65 E bomb (N shutter sealed)")
                checkpoints.append("Level5Whistle66")
                n66 = leave_66_north(env, assist, total)
                hops.append({"hop": "66_up", **{k: n66[k] for k in n66 if k != "log"}})
                if n66.get("ok"):
                    dump_and_save_room(env, assist, total, "l5_w56", "Level5Whistle56", STATE, "0x66 UP")
                    checkpoints.append("Level5Whistle56")

        route = (
            ("56_right", "RIGHT", 0x57),
            ("57_up", "UP", 0x47),
            ("47_up", "UP", 0x37),
            ("37_up", "UP", 0x27),
        )
        for name, direction, expect in route:
            snap = read_snapshot(env.get_ram())
            if snap.screen == expect:
                hops.append({"hop": name, "already": True})
                continue
            if snap.screen not in (0x56, 0x57, 0x47, 0x37, 0x27):
                print("OFF_ROUTE", f"0x{snap.screen:02x}", "at", name, flush=True)
                break
            fight_if_needed(env, assist, total, snap.screen)
            rec = door_hop(env, assist, total, direction, expect)
            hops.append({"hop": name, **rec})
            if rec["ok"]:
                ck = f"Level5Whistle{expect:02X}"
                dump_and_save_room(env, assist, total, f"l5_w{expect:02x}", ck, STATE, name)
                checkpoints.append(ck)
            else:
                print("HOP_FAIL", name, rec, flush=True)
                break

        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x27:
            w = walk_west_from_27(env, assist, total)
            hops.append({"hop": "27_west", "dest": w.get("dest"), "ok": w.get("success")})
            print("W27", w.get("success"), w.get("dest"), flush=True)
        if read_snapshot(env.get_ram()).screen == 0x26:
            dump_and_save_room(env, assist, total, "l5_w26", "Level5Whistle26", STATE, "0x27 WEST")
            checkpoints.append("Level5Whistle26")
            w = walk_west_from_26(env, assist, total)
            hops.append({"hop": "26_west", "dest": w.get("dest"), "ok": w.get("success")})
            print("W26", w.get("success"), w.get("dest"), flush=True)
        if read_snapshot(env.get_ram()).screen == 0x25:
            dump_and_save_room(env, assist, total, "l5_w25", "Level5Whistle25", STATE, "0x26 WEST")
            checkpoints.append("Level5Whistle25")
            w = walk_west_from_25(env, assist, total)
            hops.append({"hop": "25_west", "dest": w.get("dest"), "ok": w.get("success")})
            print("W25", w.get("success"), w.get("dest"), flush=True)

        boss = None
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x24:
            dump_and_save_room(env, assist, total, "l5_w24", "Level5Whistle24", STATE, "0x25 WEST Digdogger")
            checkpoints.append("Level5Whistle24")
            boss = digdogger_here(env, assist, total, STATE)
            hops.append({"hop": "digdogger", "ok": boss.get("ok"), "tf_l5": boss.get("tf_l5"), "tf": boss.get("tf_out")})
            if boss.get("ok"):
                checkpoints.append("Level5Triforce")

        ram = env.get_ram()
        snap = read_snapshot(ram)
        final = dump_live(snap, ram)
        body = {
            "ok": bool(int(final.get("triforce_0x0671") or 0) & 0x10),
            "start_state": STATE,
            "start_room": start.get("room_hex"),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "stairs_06_07_64": True,
            "65_north_shutter": "sealed_rom_shutter_never_opened",
            "detour": "0x65 E bomb -> 0x66 UP -> 0x56 (honest ROM bomb wall)",
            "digdogger": None if boss is None else {k: boss[k] for k in boss if k not in ("at24", "after_whistle", "after_heart", "final")},
            "tf_bit_0x10": bool(int(final.get("triforce_0x0671") or 0) & 0x10),
            "triforce_0x0671": int(final.get("triforce_0x0671") or 0),
            "final_room": final.get("room_hex"),
            "hops": hops,
            "checkpoints": checkpoints,
            "final": final,
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_65_to_tf.json", body)
        print("FINAL", body["ok"], "room", body["final_room"], "tf", hex(body["triforce_0x0671"]), "ck", checkpoints, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "tf", r.get("tf_bit_0x10"), flush=True)
