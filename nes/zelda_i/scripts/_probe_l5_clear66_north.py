"""From Level5Whistle76: enter 0x66, kill 3 Gibdos, UP to 0x56."""
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import ROOM_66_SPEC
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle76"


def dump(env):
    s = read_snapshot(env.get_ram())
    gib = [
        {"hp": o.hp, "x": o.x, "y": o.y}
        for o in s.objects
        if 1 <= o.slot <= 12 and o.type_id == 0x30 and o.hp > 0
    ]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": s.room_all_dead,
        "gibdos": len(gib),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "keys": int(s.keys),
    }


def main():
    configure_headless()
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    log = []
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        walk_axis(env, assist, total, "x", 120, max_f=200)
        push_dir(env, assist, total, "UP", frames=220)
        idle(env, assist, total, 16)
        log.append({"tag": "enter66", **dump(env)})
        print("ENTER", log[-1], flush=True)

        ctl = GenericDungeonRoomController(ROOM_66_SPEC)
        for _ in range(ROOM_66_SPEC.max_frames):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            env.step(action.action)
            total[0] += 1
            assist.apply_env(env, frame=total[0])
            if ctl.success or ctl.phase is DungeonPhase.FAILED:
                break
        idle(env, assist, total, 20)
        rec = {"tag": "fight", "ok": bool(ctl.success), "frames": ctl.frames, **dump(env)}
        log.append(rec)
        print("FIGHT", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_cleared.png")

        walk_axis(env, assist, total, "y", 141, max_f=300)
        walk_axis(env, assist, total, "x", 120, max_f=300)
        walk_axis(env, assist, total, "y", 93, max_f=300)
        push_dir(env, assist, total, "UP", frames=280)
        idle(env, assist, total, 16)
        for _ in range(240):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x56:
                break
            env.step(nes_action("UP"))
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 16)
        rec = {"tag": "north", **dump(env)}
        log.append(rec)
        print("NORTH", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_cleared_north.png")

        s = read_snapshot(env.get_ram())
        if s.screen == 0x56:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle56"), env.em.get_state())
            write_state_provenance(
                GAME_DIR / "custom_integrations" / GAME / "Level5Whistle56.state",
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle76.state",
                request={"segment": "Level5Whistle56", "via": "0x66 reclear Gibdos then UP", "key_poke": False, "door_poke": False},
                selected_trial={"success": True, "room": 0x56, "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED Level5Whistle56", flush=True)
        body = {"ok": s.screen == 0x56, "log": log, "final": dump(env), "pokes": False}
        write_json_report(RECORDINGS_DIR / "l5_w66_reclear_north.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
