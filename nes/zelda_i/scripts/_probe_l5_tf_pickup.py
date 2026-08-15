"""From Level5Triforce (0x14, item 0x1B, bit still 0): walk onto TF, save after 0x10."""
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_whistle_path import dump_and_save_room, dump_live, hunt_item, shot

STATE = "Level5Triforce"


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        print("START", start.get("room_hex"), [start.get("x"), start.get("y")], "tf", hex(tf0), "item", start.get("room_item_id"), flush=True)
        walk = hunt_item(env, assist, total, ADDR_TRIFORCE)
        idle(env, assist, total, 30)
        tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        print("WALK", walk.get("in"), "->", walk.get("out"), "now", hex(tf1), "got", walk.get("got"), flush=True)
        shot(env, assist, total, "l5_14_tf_got")
        dump = dump_and_save_room(
            env, assist, total, "l5_14_tf_got", "Level5Triforce", STATE,
            "0x14 walk onto triforce 0x1B, bit 0x10",
        )
        # Also keep a distinct name if we want both entry and got.
        from zelda_i.scripts._probe_l5_whistle_path import save_ckpt
        save_ckpt(
            env,
            "Level5TriforceGot",
            STATE,
            {
                "segment": "Level5TriforceGot",
                "predecessor_entry": True,
                "start_state": STATE,
                "via": "0x14 center triforce walk",
                "key_poke": False,
                "door_poke": False,
                "bomb_count_poke": False,
                "selected_item_poke": False,
            },
            {
                "success": bool(tf1 & 0x10),
                "room": int(read_snapshot(env.get_ram()).screen),
                "triforce_0x0671": tf1,
                "tf_l5_bit": bool(tf1 & 0x10),
                "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
            },
        )
        body = {
            "ok": bool(tf1 & 0x10),
            "tf_in": tf0,
            "tf_out": tf1,
            "tf_l5": bool(tf1 & 0x10),
            "walk": {k: v for k, v in walk.items() if k != "hits"},
            "final": dump_live(read_snapshot(env.get_ram()), env.get_ram()),
            "pokes": False,
            "status_claim": None,
            "l6_l8": False,
        }
        write_json_report(RECORDINGS_DIR / "l5_14_tf_got.json", body)
        print("OK", body["ok"], "tf", hex(tf1), flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    main()
