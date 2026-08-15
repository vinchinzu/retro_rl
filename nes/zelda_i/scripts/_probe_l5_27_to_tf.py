"""From Level5Whistle27: west 0x26/0x25/0x24, settle play mode, Digdogger TF."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.level5_path import walk_west_from_25, walk_west_from_26, walk_west_from_27
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_06_to_tf import digdogger_here
from zelda_i.scripts._probe_l5_whistle_path import dump_and_save_room, dump_live, wait_play

STATE = "Level5Whistle27"


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    checkpoints = [
        "Level5Whistle07", "Level5Whistle64", "Level5Whistle65",
        "Level5Whistle66Cleared", "Level5Whistle56", "Level5Whistle57",
        "Level5Whistle47Cleared", "Level5Whistle37", "Level5Whistle27",
    ]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START27", start.get("room_hex"), [start.get("x"), start.get("y")], "whistle", start.get("whistle_0x065C"), flush=True)

        w = walk_west_from_27(env, assist, total)
        wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 16)
        snap = read_snapshot(env.get_ram())
        hops.append({"hop": "27_west", "dest": f"0x{snap.screen:02x}", "mode": snap.mode, "keys": snap.keys, "xy": [snap.link_x, snap.link_y]})
        print("W27", hops[-1], flush=True)
        if snap.screen == 0x26 and snap.mode == PLAY_MODE:
            dump_and_save_room(env, assist, total, "l5_w26p", "Level5Whistle26", STATE, "0x27 WEST settled")
            checkpoints.append("Level5Whistle26")
            w = walk_west_from_26(env, assist, total)
            wait_play(env, assist, total, max_f=280)
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            hops.append({"hop": "26_west", "dest": f"0x{snap.screen:02x}", "mode": snap.mode, "xy": [snap.link_x, snap.link_y]})
            print("W26", hops[-1], flush=True)
        if snap.screen == 0x25 and snap.mode == PLAY_MODE:
            dump_and_save_room(env, assist, total, "l5_w25p", "Level5Whistle25", STATE, "0x26 WEST settled")
            checkpoints.append("Level5Whistle25")
            w = walk_west_from_25(env, assist, total)
            wait_play(env, assist, total, max_f=280)
            idle(env, assist, total, 16)
            snap = read_snapshot(env.get_ram())
            hops.append({"hop": "25_west", "dest": f"0x{snap.screen:02x}", "mode": snap.mode, "xy": [snap.link_x, snap.link_y]})
            print("W25", hops[-1], flush=True)

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
        write_json_report(RECORDINGS_DIR / "l5_27_to_tf.json", body)
        print("FINAL", body["ok"], "room", body["final_room"], "tf", hex(body["triforce_0x0671"]), "ck", checkpoints, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "tf", r.get("tf_bit_0x10"), flush=True)
