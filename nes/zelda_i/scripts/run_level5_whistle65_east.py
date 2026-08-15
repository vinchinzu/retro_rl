"""From Level5Whistle65: walk-first east hole → known floor → Digdogger → TF.

Do not try UP from 0x65 (north shutter is one-way from 0x55).
0x06 stairs is fallback only if the east hole fails.
Survival / infinite-life. No key/door pokes. Not a Clean STATUS claim.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import DoorRoute, RewardKind, RewardSpec
from zelda_i.dungeon_ops import idle
from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
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
    walk_east_from_65,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5_whistle_tf", HERE.parent / "run_level5_whistle_tf.py")
tf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(tf)

START = "Level5Whistle65"
DODONGO_L5 = 0x31
TF_BIT = 0x10


def run_once(start_state: str, tag: str) -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    total = [1]
    hops = []
    checkpoints = []
    boss = None
    failed = None
    reason = None
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 16)
        start = tf.inv(env)
        print("START", start, flush=True)
        if start["whistle"] != 1:
            failed, reason = start["room"], "whistle_not_1"
            return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
        if start["room"] != "0x65":
            failed, reason = start["room"], "not_in_65"
            return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)

        # --- 0x65 EAST: walk existing 0x66-west hole; bomb only if sealed ---
        rec = walk_east_from_65(env, assist, total)
        hops.append({"hop": "65_east", **{k: rec[k] for k in rec if k not in ("log", "menu")}})
        print(
            "EAST65",
            rec.get("success"),
            "path",
            rec.get("path"),
            "bombed",
            rec.get("bombed"),
            "dest",
            f"0x{rec.get('dest'):02x}",
            rec.get("xy"),
            flush=True,
        )
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / f"{tag}_after_east65.png")
        if not rec.get("success"):
            failed, reason = "0x65", "east_did_not_enter_66"
            return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
        checkpoints.append(
            tf.save_ckpt(
                env,
                "Level5Whistle66",
                start_state,
                f"0x65 east {rec.get('path')} bombed={rec.get('bombed')}",
                {**tf.inv(env), "east_path": rec.get("path"), "bombed": rec.get("bombed")},
            )
        )
        room = 0x66

        if room == 0x66:
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE66 n", n, tf.objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
            if n:
                spec = tf.replace(
                    ROOM_66_SPEC,
                    spec_id="level5_w65_66_gibdos",
                    source_room=0x65,
                    room_id=0x66,
                    entry=DoorRoute("RIGHT", ((32, 141),)),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                    max_frames=28000,
                    level=LEVEL_5,
                )
                fight = tf.fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_66", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x66", "gibdos_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            rec = tf.door(env, assist, total, "UP", 0x56)
            hops.append({"hop": "66_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x66", "up_did_not_enter_56"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle56", start_state, "0x66 clear + UP 0x56", {**tf.inv(env)}))
            room = 0x56

        if room == 0x56:
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (DODONGO_L5,)))
            print("ARRIVE56 n", n, tf.objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
            rec = tf.door(env, assist, total, "RIGHT", 0x57)
            hops.append({"hop": "56_east", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok") and n:
                spec = tf.replace(
                    ROOM_5B_SPEC,
                    spec_id="level5_w65_56_dodongos",
                    source_room=0x66,
                    room_id=0x56,
                    entry=DoorRoute("UP", ((120, 205),)),
                    enemy_types=(DODONGO_L5,),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                    combat=ROOM_59_SPEC.combat,
                    max_frames=28000,
                    level=LEVEL_5,
                )
                fight = tf.fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_56", **{k: fight[k] for k in fight if k != "progress"}})
                rec = tf.door(env, assist, total, "RIGHT", 0x57)
                hops.append({"hop": "56_east_after_clear", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x56", "east_did_not_enter_57"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = 0x57

        if room == 0x57:
            rec = tf.door(env, assist, total, "UP", 0x47)
            hops.append({"hop": "57_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x57", "up_did_not_enter_47"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE47 n", n, tf.objs(snap), flush=True)
            if n:
                spec = tf.replace(
                    ROOM_66_SPEC,
                    spec_id="level5_w65_47_gibdos",
                    source_room=0x57,
                    room_id=0x47,
                    entry=DoorRoute("UP", ((120, 205),)),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                    max_frames=28000,
                    level=LEVEL_5,
                )
                fight = tf.fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_47", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x47", "gibdos_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle47", start_state, "0x57 up, gibdo", {**tf.inv(env)}))
            room = 0x47

        if room == 0x47:
            rec = tf.north_pinch(env, assist, total, 0x37)
            hops.append({"hop": "47_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x47", "up_did_not_enter_37"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (0x0B, BLUE_DARKNUT_TYPE)))
            print("ARRIVE37 n", n, tf.objs(snap), flush=True)
            if n:
                spec = tf.replace(
                    ROOM_5B_SPEC,
                    spec_id="level5_w65_37_darknuts",
                    source_room=0x47,
                    room_id=0x37,
                    entry=DoorRoute("UP", ((120, 205),)),
                    enemy_types=(0x0B, BLUE_DARKNUT_TYPE),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
                    combat=ROOM_59_SPEC.combat,
                    max_frames=20000,
                    level=LEVEL_5,
                )
                fight = tf.fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_37", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x37", "darknuts_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle37", start_state, "0x47 up, compass darknut", {**tf.inv(env)}))
            room = 0x37

        if room == 0x37:
            rec = tf.north_pinch(env, assist, total, 0x27)
            hops.append({"hop": "37_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x37", "up_did_not_enter_27"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (POLS_VOICE_OBJECT_TYPE, GIBDO_OBJECT_TYPE, 0x1B)))
            print("ARRIVE27 n", n, tf.objs(snap), flush=True)
            if n:
                spec = tf.replace(ROOM_27_SPEC, spec_id="level5_w65_27_mixed", expected_enemy_count=n, required_open_doors=0)
                fight = tf.fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_27", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x27", "mixed_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            keys0 = read_snapshot(env.get_ram()).keys
            tf.grab_item(
                env,
                assist,
                total,
                lambda e: read_snapshot(e.get_ram()).keys > keys0,
                ((120, 141), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141)),
            )
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle27", start_state, "0x37 up, mixed clear", {**tf.inv(env)}))
            room = 0x27

        if room == 0x27:
            west = walk_west_from_27(env, assist, total)
            tf.wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x26 and snap.mode == PLAY_MODE
            hops.append({"hop": "27_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST27", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
            if not west.get("success"):
                failed, reason = "0x27", "west_did_not_enter_26"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE26 n", n, tf.objs(snap), flush=True)
            if n:
                spec = tf.replace(ROOM_26_SPEC, spec_id="level5_w65_26_gibdos", expected_enemy_count=n, required_open_doors=0)
                fight = tf.fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_26", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x26", "gibdos_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            keys0 = read_snapshot(env.get_ram()).keys
            tf.grab_item(
                env,
                assist,
                total,
                lambda e: read_snapshot(e.get_ram()).keys > keys0,
                ((224, 141), (120, 141), (96, 141), (144, 141)),
            )
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle26", start_state, "0x27 west, 5 gibdo", {**tf.inv(env)}))
            room = 0x26

        if room == 0x26:
            west = walk_west_from_26(env, assist, total)
            tf.wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x25 and snap.mode == PLAY_MODE
            hops.append({"hop": "26_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST26", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
            if not west.get("success"):
                failed, reason = "0x26", "west_did_not_enter_25"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(tf.live_types(snap, (POLS_VOICE_OBJECT_TYPE,)))
            print("ARRIVE25 n", n, tf.objs(snap), flush=True)
            if n:
                spec = tf.replace(ROOM_25_SPEC, spec_id="level5_w65_25_pols", expected_enemy_count=n, required_open_doors=0)
                fight = tf.fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_25", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x25", "pols_not_cleared"
                    return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(tf.save_ckpt(env, "Level5Whistle25", start_state, "0x26 west, 5 pols", {**tf.inv(env)}))
            room = 0x25

        if room == 0x25:
            west = walk_west_from_25(env, assist, total)
            tf.wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x24 and snap.mode == PLAY_MODE
            hops.append({"hop": "25_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST25", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
            if not west.get("success"):
                failed, reason = "0x25", "west_did_not_enter_24"
                return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(
                tf.save_ckpt(env, "Level5Whistle24", start_state, "0x25 west key door, Digdogger room", {**tf.inv(env)})
            )
            room = 0x24

        if room != 0x24:
            failed, reason = f"0x{room:02x}", "not_in_24"
            return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)

        boss = tf._digdogger_here(env, assist, total)
        hops.append({"hop": "digdogger", **{k: boss[k] for k in boss if k not in ("after_whistle_objs",)}})
        if boss.get("tf_l5"):
            checkpoints.append(
                tf.save_ckpt(
                    env,
                    "Level5TF",
                    start_state,
                    "whistle shrink Digdogger, heart, north TF 0x10",
                    {**tf.inv(env), "killed": boss.get("killed"), "heart": boss.get("heart")},
                )
            )
        elif boss.get("killed"):
            checkpoints.append(tf.save_ckpt(env, "Level5Digdogger", start_state, "Digdogger killed", {**tf.inv(env)}))
        return tf._finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
    finally:
        env.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default=START)
    ap.add_argument("--tag", default="l5_w65_east_tf")
    args = ap.parse_args()
    r = run_once(args.from_state, args.tag)
    print("RESULT_OK", r.get("ok"))
    print("HOPS", [(h.get("hop"), h.get("ok") or h.get("success"), h.get("dest"), h.get("path")) for h in r.get("hops", [])])
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("status_claim", None)
    print("EAST_HOLE", next((h for h in r.get("hops", []) if h.get("hop") == "65_east"), None))


if __name__ == "__main__":
    main()
