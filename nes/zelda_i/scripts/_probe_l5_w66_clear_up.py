"""Level5Whistle66: clear 3 Gibdos (all_dead north shutter) → UP 0x56 → RIGHT 0x57."""
from dataclasses import replace

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DoorRoute, DungeonPhase, GenericDungeonRoomController, RewardKind, RewardSpec
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level5_dungeon import GIBDO_OBJECT_TYPE, LEVEL_5, ROOM_66_SPEC
from zelda_i.level5_path import walk_axis
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5Whistle66"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    objs = [{"t": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
            for o in s.objects if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF) and o.type_id < 0x40]
    return {
        "sc": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "objs": objs,
    }


def fight(env, assist, total, spec):
    ctl = GenericDungeonRoomController(spec)
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = spec.live_enemies(snap) if snap.mode == PLAY_MODE else []
    return {"ok": bool(ctl.success) and not live, "frames": ctl.frames, "end_n": len(live), "xy": [snap.link_x, snap.link_y]}


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
        print("START", dump(env), flush=True)
        snap = read_snapshot(env.get_ram())
        n = len([o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == GIBDO_OBJECT_TYPE and o.hp > 0])
        print("N", n, flush=True)
        if n:
            spec = replace(
                ROOM_66_SPEC,
                spec_id="level5_whistle_66_gibdos",
                source_room=0x65,
                room_id=0x66,
                entry=DoorRoute("RIGHT", ((32, 141),)),
                expected_enemy_count=n,
                required_open_doors=0,
                reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                max_frames=28000,
                level=LEVEL_5,
            )
            rec = fight(env, assist, total, spec)
            log.append({"tag": "fight", **rec, **dump(env)})
            print("FIGHT", rec, dump(env), flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_cleared.png")

        walk_axis(env, assist, total, "y", 109, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 93, max_f=300)
        push_dir(env, assist, total, "UP", frames=300)
        idle(env, assist, total, 20)
        for _ in range(240):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen == 0x56:
                break
            env.step(nes_idle_action())
            total[0] += 1
            assist.apply_env(env, frame=total[0])
        rec = {"tag": "up56", **dump(env)}
        log.append(rec)
        print("UP56", rec, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_up56.png")

        if read_snapshot(env.get_ram()).screen == 0x56:
            walk_axis(env, assist, total, "y", 141, max_f=400)
            walk_axis(env, assist, total, "x", 224, max_f=500)
            push_dir(env, assist, total, "RIGHT", frames=260)
            idle(env, assist, total, 20)
            rec = {"tag": "east57", **dump(env)}
            log.append(rec)
            print("EAST57", rec, flush=True)
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w66_east57.png")

        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen in (0x56, 0x57) and int(read_u8(env.get_ram(), ADDR_WHISTLE)) == 1:
            name = f"Level5Whistle{s.screen:02X}"
            write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, name),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={
                    "segment": name,
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x66 clear gibdos, north shutter, 0x56 east 0x57",
                    "key_poke": False,
                    "door_poke": False,
                },
                selected_trial={"success": True, "room": int(s.screen), "whistle_0x065C": 1},
                natural_entry=False,
            )
            print("SAVED", name, dump(env), flush=True)

        body = {"ok": read_snapshot(env.get_ram()).screen in (0x56, 0x57), "pokes": False, "log": log, "final": dump(env)}
        write_json_report(RECORDINGS_DIR / "l5_w66_clear_up.json", body)
        print("FINAL", body["final"], "OK", body["ok"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
