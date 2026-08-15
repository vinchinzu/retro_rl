"""From Level5Whistle66: kill 3 Gibdo (opens N shutter 0x08), UP 0x56, on to TF."""
from __future__ import annotations

from dataclasses import replace

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DoorRoute, DungeonPhase, GenericDungeonRoomController, RewardKind, RewardSpec
from zelda_i.dungeon_ops import idle
from zelda_i.level5_dungeon import LEVEL_5, ROOM_66_SPEC
from zelda_i.level5_path import walk_axis, walk_west_from_25, walk_west_from_26, walk_west_from_27
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.scripts._probe_l5_06_to_tf import digdogger_here, door_hop, fight_if_needed
from zelda_i.scripts._probe_l5_whistle_path import dump_and_save_room, dump_live, shot, step, wait_play

STATE = "Level5Whistle66"


def clear_66(env, assist, total) -> dict:
    spec = replace(
        ROOM_66_SPEC,
        spec_id="level5_room66_gibdos_from_west",
        source_room=0x65,
        entry=DoorRoute("LEFT", ((32, 141),)),
        required_open_doors=0x08,  # north shutter
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        exit_routes=(DoorRoute("UP", ((120, 93),)),),
        max_frames=20000,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == 0x66:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n})
                print("GIBDO", last_n, "f", ctl.frames, flush=True)
        action = ctl.step(snap)
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x30 and o.hp > 0]
    rec = {
        "ok": bool(ctl.success) or (not live and bool(snap.cur_opened_doors & 0x08)),
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "doors": int(snap.cur_opened_doors),
        "north": bool(snap.cur_opened_doors & 0x08),
        "progress": progress,
        "xy": [snap.link_x, snap.link_y],
    }
    print("CLEAR66", rec, flush=True)
    return rec


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    checkpoints = ["Level5Whistle07", "Level5Whistle64", "Level5Whistle65", "Level5Whistle66"]
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 12)
        start = dump_live(read_snapshot(env.get_ram()), env.get_ram())
        print("START66", start.get("room_hex"), [start.get("x"), start.get("y")], "objs", [(o["type_hex"], o["hp"]) for o in start.get("objects") or []], flush=True)
        fight = clear_66(env, assist, total)
        hops.append({"hop": "clear66", **{k: fight[k] for k in fight if k != "progress"}})
        idle(env, assist, total, 20)
        shot(env, assist, total, "l5_66_cleared")
        dump_and_save_room(env, assist, total, "l5_w66c", "Level5Whistle66Cleared", STATE, "3 gibdo, N shutter")
        checkpoints.append("Level5Whistle66Cleared")

        rec = door_hop(env, assist, total, "UP", 0x56)
        hops.append({"hop": "66_up", **rec})
        if not rec["ok"]:
            walk_axis(env, assist, total, "y", 141, max_f=300)
            walk_axis(env, assist, total, "x", 120, max_f=400)
            rec = door_hop(env, assist, total, "UP", 0x56)
            hops.append({"hop": "66_up_retry", **rec})
        if rec["ok"]:
            dump_and_save_room(env, assist, total, "l5_w56", "Level5Whistle56", STATE, "0x66 UP after gibdos")
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
                print("OFF_ROUTE", f"0x{snap.screen:02x}", name, flush=True)
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
            "clear66": fight,
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
        write_json_report(RECORDINGS_DIR / "l5_66_to_tf.json", body)
        print("FINAL", body["ok"], "room", body["final_room"], "tf", hex(body["triforce_0x0671"]), "ck", checkpoints, flush=True)
        return body
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "tf", r.get("tf_bit_0x10"), flush=True)
