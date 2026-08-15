"""From Level5Whistle24: recorder twice, then ROOM_59 combat on 0x18/0x38."""
from dataclasses import replace

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController, RewardKind, RewardSpec
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import ROOM_59_SPEC
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level5_path import select_b_item_menu, walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8

def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [
        {"t": f"0x{o.type_id:02x}", "hp": o.hp, "xy": [o.x, o.y]}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF)
    ]
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "sc": f"0x{s.screen:02x}", "mode": s.mode, "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "tf": tf, "tf_l5": bool(tf & 0x10), "doors": int(s.cur_opened_doors),
        "item": int(s.room_item_id), "objs": objs,
    }

def main():
    configure_headless()
    env = make_env(GAME, "Level5Whistle24", GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    n = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        print("START", dump(env), flush=True)
        walk_axis(env, assist, n, "x", 176, max_f=300)
        walk_axis(env, assist, n, "y", 189, max_f=300)
        print("MENU", select_b_item_menu(env, assist, n, 5), flush=True)
        for tag in ("b1", "b2"):
            env.step(nes_action("B"))
            n[0] += 1
            assist.apply_env(env, frame=n[0])
            idle(env, assist, n, 180)
            rec = dump(env)
            log.append({"tag": tag, **rec})
            print(tag.upper(), rec, flush=True)
        bosses = [o for o in read_snapshot(env.get_ram()).objects if 1 <= o.slot <= 12 and o.type_id in (0x18, 0x38) and o.hp > 0]
        types = tuple(sorted({o.type_id for o in bosses})) or (0x18, 0x38)
        spec = replace(
            ROOM_59_SPEC,
            spec_id="level5_digdogger_small",
            source_room=0x25,
            room_id=0x24,
            enemy_types=types,
            expected_enemy_count=max(1, len(bosses)),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            max_frames=20000,
            level=LEVEL_5,
        )
        ctl = GenericDungeonRoomController(spec)
        last_hp = None
        for i in range(spec.max_frames):
            snap = read_snapshot(env.get_ram())
            live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in types and o.hp > 0]
            hp = live[0].hp if live else 0
            if hp != last_hp:
                print(f"HP f{i} {hp} n={len(live)}", flush=True)
                last_hp = hp
            action = ctl.step(snap)
            env.step(action.action)
            n[0] += 1
            assist.apply_env(env, frame=n[0])
            if ctl.success or ctl.phase is DungeonPhase.FAILED or not live:
                break
        rec = dump(env)
        log.append({"tag": "fight", "ok": bool(ctl.success), "frames": ctl.frames, **rec})
        print("FIGHT", ctl.success, ctl.frames, rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24_after_sword.png")
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        walk_axis(env, assist, n, "y", 141, max_f=300)
        walk_axis(env, assist, n, "x", 120, max_f=300)
        walk_axis(env, assist, n, "y", 93, max_f=400)
        push_dir(env, assist, n, "UP", frames=260)
        idle(env, assist, n, 20)
        rec = dump(env)
        print("AT14", rec, flush=True)
        if rec["sc"] == "0x14":
            walk_axis(env, assist, n, "x", 120, max_f=300)
            walk_axis(env, assist, n, "y", 141, max_f=400)
            idle(env, assist, n, 40)
            for _ in range(900):
                tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
                if tf & 0x10:
                    break
                env.step(nes_idle_action())
                n[0] += 1
                assist.apply_env(env, frame=n[0])
            rec = dump(env)
            print("PICKUP", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_14_tf.png")
            if rec["tf_l5"]:
                path = write_state_bytes(state_path(GAME_DIR, GAME, "Level5WhistleTF"), env.em.get_state())
                write_state_provenance(
                    path,
                    source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle24.state",
                    request={
                        "segment": "Level5WhistleTF",
                        "predecessor_entry": True,
                        "start_state": "Level5Whistle24",
                        "via": "0x24 whistle x2 shrink 0x38->0x18, ROOM_59 sword, north 0x14 TF pickup",
                        "key_poke": False,
                        "door_poke": False,
                    },
                    selected_trial={"success": True, "room": 0x14, "tf_0x0671": rec["tf"], "tf_l5_bit": True, "whistle_0x065C": rec["whistle"]},
                    natural_entry=False,
                )
                print("PINNED Level5WhistleTF", rec, flush=True)
        rec = dump(env)
        log.append({"tag": "final", **rec})
        print("FINAL", rec, flush=True)
        write_json_report(RECORDINGS_DIR / "l5_24_whistle_kill.json", {"log": log, "final": rec, "pokes": False, "status_claim": bool(rec["tf_l5"])})
    finally:
        env.close()

if __name__ == "__main__":
    main()
