"""Level5Whistle65 → 0x57 north → known floor → whistle Digdogger → TF 0x10.

One fceumm env, assisted, record= BK2, stop_record. No pokes.
Do not fight 0x57 Zols (foes_item 0x5f seals north). Do not start from
Level5Cleared25. No Level5Complete / STATUS claim. No L6–L8.

Locked floor:
  0x65 east bomb → 0x66 clear → UP 0x56 → RIGHT 0x57 → north 0x47
  → UP 0x37 → UP 0x27 → W 0x26 → W 0x25 → W key 0x24
  → whistle-shrink → north 0x14 TF.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    Level5PolsVoiceController,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
)
from zelda_i.level5_path import (
    BLUE_DARKNUT_TYPE,
    bomb_east_from_65,
    walk_axis,
    walk_north_from_57,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("w65tf", HERE.parent / "_probe_l5_w65_east_to_tf.py")
w65 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w65)

TAG = "l5_w65_57north_tf"
START = "Level5Whistle65"
TF_BIT = 0x10
STITCH_MAP = {
    0x14: "L5 triforce",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x65: "west Gibdo pocket",
    0x66: "3x Gibdo first key",
}


def pin_state(name: str) -> dict:
    env = make_env(GAME, name, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        ram = env.get_ram()
        s = read_snapshot(ram)
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        live_boss = [
            {"type": o.type_id, "hp": o.hp, "x": o.x, "y": o.y}
            for o in s.objects
            if 1 <= o.slot <= 12 and o.type_id in (0x38, 0x18) and o.hp > 0
        ]
        rec = {
            "state": name,
            "room": f"0x{s.screen:02x}",
            "name": STITCH_MAP.get(s.screen),
            "mode": s.mode,
            "xy": [s.link_x, s.link_y],
            "keys": int(s.keys),
            "bombs": int(s.bombs),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "tf_0x0671": tf,
            "tf_hex": hex(tf),
            "tf_l5_bit_0x10": bool(tf & TF_BIT),
            "digdogger_live": bool(live_boss),
            "live_boss": live_boss,
        }
        print("PIN", rec, flush=True)
        return rec
    finally:
        env.close()


def room_name(screen: int) -> str:
    return STITCH_MAP.get(int(screen), f"room 0x{int(screen):02x}")


def pin(env) -> dict:
    s = read_snapshot(env.get_ram())
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "name": room_name(s.screen) if s.level == 5 else None,
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce": tf,
        "tf_hex": hex(tf),
        "tf_l5_bit": bool(tf & TF_BIT),
    }


def hop_ok(env, dest: int) -> bool:
    s = read_snapshot(env.get_ram())
    return s.level == LEVEL_5 and s.screen == dest and s.mode == PLAY_MODE


def wait_play(env, assist, n, room=None, max_f=280):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            if room is None or s.screen == room:
                idle(env, assist, n, 8)
                return True
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    return hop_ok(env, room) if room is not None else True


def walk_east_from_56(env, assist, n) -> dict:
    """0x56 east is open on first visit. Stay on y=141; retry bands if Dodongos shove."""
    log = []
    for y in (141, 149, 133, 157):
        walk_axis(env, assist, n, "y", y, max_f=400)
        walk_axis(env, assist, n, "x", 208, max_f=500)
        walk_axis(env, assist, n, "y", 141, max_f=200)
        push_dir(env, assist, n, "RIGHT", frames=280)
        idle(env, assist, n, 12)
        wait_play(env, assist, n, 0x57, max_f=200)
        s = read_snapshot(env.get_ram())
        log.append({"y": y, "xy": [s.link_x, s.link_y], "room": s.screen})
        if hop_ok(env, 0x57):
            return {"path": f"y{y}_east", "dest": s.screen, "xy": [s.link_x, s.link_y], "log": log, "success": True}
    s = read_snapshot(env.get_ram())
    return {"path": "east_blocked", "dest": s.screen, "xy": [s.link_x, s.link_y], "log": log, "success": False}


def main() -> int:
    configure_headless()
    tf_pin = pin_state("Level5TF")
    w24_pin = None
    try:
        w24_pin = pin_state("Level5Whistle24")
    except Exception as exc:
        w24_pin = {"state": "Level5Whistle24", "error": str(exc)}

    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_w65_57north_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    seams = []
    hops = []
    blocker = None
    boss = None
    start = None
    final = None
    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        start = pin(env)
        print("START", start, flush=True)
        seams.append({"name": "0x65 west Gibdo pocket", "ok": start["screen"] == 0x65 and start["whistle"] == 1, **start})
        if start["whistle"] != 1 or start["screen"] != 0x65:
            blocker = f"start_not_65_whistle1_got_0x{start['screen']:02x}"
        else:
            hop66 = bomb_east_from_65(env, assist, n)
            wait_play(env, assist, n, 0x66)
            ok66 = hop66.get("success") or hop_ok(env, 0x66)
            hops.append({"hop": "65_east", **{k: hop66[k] for k in hop66 if k != "menu"}})
            seams.append({"name": "0x66 3x Gibdo first key", "ok": ok66, **pin(env)})
            print("66", ok66, pin(env), flush=True)
            if not ok66:
                blocker = "fail_hop_65_to_66"
            else:
                snap = read_snapshot(env.get_ram())
                live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == GIBDO_OBJECT_TYPE and o.hp > 0]
                if live:
                    from dataclasses import replace
                    from zelda_i.dungeon import DoorRoute, RewardKind, RewardSpec

                    spec = replace(
                        ROOM_66_SPEC,
                        spec_id="l5_w65_66_gibdos",
                        source_room=0x65,
                        room_id=0x66,
                        entry=DoorRoute("RIGHT", ((32, 141),)),
                        expected_enemy_count=len(live),
                        required_open_doors=0,
                        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                        max_frames=28000,
                        level=LEVEL_5,
                    )
                    fight = w65.fight_ctl(env, assist, n, spec)
                    hops.append({"hop": "fight_66", **fight})
                    if not fight.get("ok"):
                        blocker = "fail_clear_66"
                if blocker is None:
                    rec = w65.door(env, assist, n, "UP", 0x56)
                    hops.append({"hop": "66_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
                    seams.append({"name": "0x56 north Dodongos", "ok": rec.get("ok"), **pin(env)})
                    print("56", rec.get("ok"), pin(env), flush=True)
                    if not rec.get("ok"):
                        blocker = "fail_hop_66_to_56"
                    else:
                        rec56 = walk_east_from_56(env, assist, n)
                        hops.append({"hop": "56_east", **rec56})
                        seams.append({"name": "0x57 east Zols", "ok": rec56.get("success"), **pin(env)})
                        print("57", rec56.get("success"), pin(env), flush=True)
                        if not rec56.get("success"):
                            blocker = "fail_hop_56_to_57"
                        else:
                            rec57 = walk_north_from_57(env, assist, n)
                            wait_play(env, assist, n, 0x47)
                            ok47 = rec57.get("success") or hop_ok(env, 0x47)
                            hops.append({"hop": "57_north", **{k: rec57[k] for k in rec57 if k != "log"}})
                            seams.append({"name": "0x47 north Gibdos", "ok": ok47, "path": rec57.get("path"), **pin(env)})
                            print("47", ok47, rec57.get("path"), pin(env), flush=True)
                            if not ok47:
                                blocker = "fail_hop_57_to_47"
                            else:
                                rec = w65.north_pinch(env, assist, n, 0x37)
                                hops.append({"hop": "47_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
                                seams.append({"name": "0x37 Darknuts + compass", "ok": rec.get("ok"), **pin(env)})
                                print("37", rec.get("ok"), pin(env), flush=True)
                                if not rec.get("ok"):
                                    blocker = "fail_hop_47_to_37"
                                else:
                                    snap = read_snapshot(env.get_ram())
                                    live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (0x0B, BLUE_DARKNUT_TYPE) and o.hp > 0]
                                    if live:
                                        from dataclasses import replace
                                        from zelda_i.dungeon import DoorRoute, RewardKind, RewardSpec
                                        from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC

                                        spec = replace(
                                            ROOM_5B_SPEC,
                                            spec_id="l5_w65_37_darknuts",
                                            source_room=0x47,
                                            room_id=0x37,
                                            entry=DoorRoute("UP", ((120, 205),)),
                                            enemy_types=(0x0B, BLUE_DARKNUT_TYPE),
                                            expected_enemy_count=len(live),
                                            required_open_doors=0,
                                            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                                            combat=ROOM_59_SPEC.combat,
                                            max_frames=20000,
                                            level=LEVEL_5,
                                        )
                                        fight = w65.fight_ctl(env, assist, n, spec)
                                        hops.append({"hop": "fight_37", **fight})
                                        if not fight.get("ok"):
                                            blocker = "fail_clear_37"
                                    if blocker is None:
                                        rec = w65.north_pinch(env, assist, n, 0x27)
                                        hops.append({"hop": "37_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
                                        seams.append({"name": "0x27 mixed Pols/Gibdo/Keese", "ok": rec.get("ok"), **pin(env)})
                                        print("27", rec.get("ok"), pin(env), flush=True)
                                        if not rec.get("ok"):
                                            blocker = "fail_hop_37_to_27"
                                        else:
                                            snap = read_snapshot(env.get_ram())
                                            live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (POLS_VOICE_OBJECT_TYPE, GIBDO_OBJECT_TYPE, 0x1B) and (o.hp > 0 or o.type_id == 0x1B)]
                                            if live:
                                                from dataclasses import replace

                                                spec = replace(ROOM_27_SPEC, spec_id="l5_w65_27_mixed", expected_enemy_count=len(live), required_open_doors=0)
                                                fight = w65.fight_ctl(env, assist, n, spec, Level5PolsVoiceController)
                                                hops.append({"hop": "fight_27", **fight})
                                                if not fight.get("ok"):
                                                    blocker = "fail_clear_27"
                                            if blocker is None:
                                                west = walk_west_from_27(env, assist, n)
                                                wait_play(env, assist, n, 0x26)
                                                ok26 = hop_ok(env, 0x26) or west.get("success")
                                                hops.append({"hop": "27_west", **{k: west[k] for k in west if k != "log"}})
                                                seams.append({"name": "0x26 west Gibdos", "ok": ok26, **pin(env)})
                                                print("26", ok26, pin(env), flush=True)
                                                if not ok26:
                                                    blocker = "fail_hop_27_to_26"
                                                else:
                                                    snap = read_snapshot(env.get_ram())
                                                    live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == GIBDO_OBJECT_TYPE and o.hp > 0]
                                                    if live:
                                                        from dataclasses import replace

                                                        spec = replace(ROOM_26_SPEC, spec_id="l5_w65_26_gibdos", expected_enemy_count=len(live), required_open_doors=0)
                                                        fight = w65.fight_ctl(env, assist, n, spec)
                                                        hops.append({"hop": "fight_26", **fight})
                                                    west = walk_west_from_26(env, assist, n)
                                                    wait_play(env, assist, n, 0x25)
                                                    ok25 = hop_ok(env, 0x25) or west.get("success")
                                                    hops.append({"hop": "26_west", **{k: west[k] for k in west if k != "log"}})
                                                    seams.append({"name": "0x25 west Pols Voice", "ok": ok25, **pin(env)})
                                                    print("25", ok25, pin(env), flush=True)
                                                    if not ok25:
                                                        blocker = "fail_hop_26_to_25"
                                                    else:
                                                        snap = read_snapshot(env.get_ram())
                                                        live = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == POLS_VOICE_OBJECT_TYPE and o.hp > 0]
                                                        if live:
                                                            from dataclasses import replace

                                                            spec = replace(ROOM_25_SPEC, spec_id="l5_w65_25_pols", expected_enemy_count=len(live), required_open_doors=0)
                                                            fight = w65.fight_ctl(env, assist, n, spec, Level5PolsVoiceController)
                                                            hops.append({"hop": "fight_25", **fight})
                                                        west = walk_west_from_25(env, assist, n)
                                                        wait_play(env, assist, n, 0x24)
                                                        ok24 = hop_ok(env, 0x24) or west.get("success")
                                                        hops.append({"hop": "25_west", **{k: west[k] for k in west if k != "log"}})
                                                        seams.append({"name": "0x24 Digdogger", "ok": ok24, **pin(env)})
                                                        print("24", ok24, pin(env), flush=True)
                                                        if not ok24:
                                                            blocker = "fail_hop_25_to_24"
                                                        else:
                                                            boss = w65.digdogger(env, assist, n)
                                                            hops.append({"hop": "digdogger", **{k: boss[k] for k in boss if k != "log"}})
                                                            seams.append({"name": "0x14 L5 triforce" if hop_ok(env, 0x14) else "0x24 Digdogger", "ok": bool(boss.get("tf_l5")), **pin(env)})
                                                            print("TF", boss.get("tf_l5"), pin(env), flush=True)
                                                            if not boss.get("tf_l5"):
                                                                blocker = "tf_bit_0x10_not_set"

        final = pin(env)
        shot = out / f"{TAG}_final.png"
        save_rgb_png(env.step(nes_idle_action())[0], shot)
        if hasattr(env, "stop_record"):
            env.stop_record()
    finally:
        try:
            if hasattr(env, "stop_record"):
                env.stop_record()
        except Exception:
            pass
        env.close()

    bk2s = sorted(movie.glob("*.bk2"), key=lambda p: p.stat().st_mtime)
    bk2 = str(bk2s[-1]) if bk2s else None
    tf = None if final is None else final.get("triforce")
    tf_l5 = bool(final and final.get("tf_l5_bit"))
    north = next((h for h in hops if h.get("hop") == "57_north"), None)
    report = {
        "ok": tf_l5 and blocker is None,
        "segment": TAG,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "level5_complete_claim": False,
        "start_state": START,
        "level5_tf_pin": tf_pin,
        "level5_whistle24_pin": w24_pin,
        "end_claim": "l5_triforce_bit_0x10" if tf_l5 else None,
        "whistle_0x065C": None if final is None else final.get("whistle"),
        "triforce_0x0671": tf,
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": None if boss is None else boss.get("killed"),
        "hop_57_north": None if north is None else {k: north[k] for k in north if k != "log"},
        "total_frames": n[0],
        "start": start,
        "final": final,
        "seams": seams,
        "hops": hops,
        "room_sequence": [f"0x{s.get('screen'):02x} {s.get('name')}" for s in seams if s.get("screen") is not None],
        "blocker": blocker,
        "bk2": bk2,
        "png": str(out / f"{TAG}_final.png"),
        "pokes": False,
        "path_note": (
            "Level5Whistle65 → 0x66 → 0x56 → 0x57 north (no Zol clear; key-north / "
            "push 0x5f / diamond) → 0x47 → 0x37 → 0x27 → 0x26 → 0x25 → 0x24 "
            "whistle-shrink → 0x14 TF. Did not claim Level5Complete / STATUS."
        ),
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} tf={tf} tf_l5={tf_l5} "
        f"blocker={blocker} 57north={None if north is None else north.get('path')}",
        flush=True,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
