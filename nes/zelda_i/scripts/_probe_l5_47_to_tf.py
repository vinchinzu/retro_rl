"""From Level5Whistle47: clear 5 Gibdo, center-channel UP 0x37 -> 0x27 west -> TF."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import walk_axis, walk_west_from_25, walk_west_from_26, walk_west_from_27
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_06_to_tf import digdogger_here, door_hop, fight_if_needed
from zelda_i.scripts._probe_l5_whistle_path import dump_and_save_room, dump_live, fight_type, shot

STATE = "Level5Whistle47"


def center_up(env, assist, total, expect: int) -> dict:
    """0x47/0x37 C-pits: y=141 channel, x=120, then north door."""
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    return door_hop(env, assist, total, "UP", expect)


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    checkpoints = [
        "Level5Whistle07", "Level5Whistle64", "Level5Whistle65",
        "Level5Whistle66", "Level5Whistle66Cleared", "Level5Whistle56",
        "Level5Whistle57", "Level5Whistle47",
    ]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START47", start.get("room_hex"), [start.get("x"), start.get("y")], flush=True)
        live = [o for o in read_snapshot(env.get_ram()).objects if 1 <= o.slot <= 12 and o.type_id == 0x30 and o.hp > 0]
        if live:
            fight = fight_type(env, assist, total, 0x47, 0x30, expected=len(live))
            hops.append({"hop": "clear47", "ok": fight.get("ok"), "end_n": fight.get("end_n"), "frames": fight.get("frames")})
            print("CLEAR47", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)
            idle(env, assist, total, 16)
        dump_and_save_room(env, assist, total, "l5_w47c", "Level5Whistle47Cleared", STATE, "5 gibdo then center UP")
        checkpoints.append("Level5Whistle47Cleared")

        rec = center_up(env, assist, total, 0x37)
        hops.append({"hop": "47_up", **rec})
        if rec["ok"]:
            dump_and_save_room(env, assist, total, "l5_w37", "Level5Whistle37", STATE, "0x47 UP")
            checkpoints.append("Level5Whistle37")
            fight_if_needed(env, assist, total, 0x37)
            rec = center_up(env, assist, total, 0x27)
            hops.append({"hop": "37_up", **rec})
            if rec["ok"]:
                dump_and_save_room(env, assist, total, "l5_w27", "Level5Whistle27", STATE, "0x37 UP")
                checkpoints.append("Level5Whistle27")

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
        if read_snapshot(env.get_ram()).screen == 0x24:
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
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "stairs_06_07_64": True,
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
        write_json_report(RECORDINGS_DIR / "l5_47_to_tf.json", body)
        print("FINAL", body["ok"], "room", body["final_room"], "tf", hex(body["triforce_0x0671"]), "ck", checkpoints, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "tf", r.get("tf_bit_0x10"), flush=True)
