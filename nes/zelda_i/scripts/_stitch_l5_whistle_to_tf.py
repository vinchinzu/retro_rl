"""Level5Whistle → 0x05 → 0x06 block-stairs → 0x07 → 0x64 → toward Digdogger 0x24.

One fceumm session. UnlimitedHealthAssist only. No pokes. No STATUS claim.
South key 0x06→0x16 is not the return.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level5_dungeon import LEVEL_5
from zelda_i.level5_path import (
    bomb_east_from_65,
    cellar_07_to_64,
    exit_whistle_04,
    select_b_item_menu,
    take_block_stairs_06,
    walk_axis,
    walk_east_from_05,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import on_stair_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

TAG = "l5_whistle_to_tf_stitch"
ROOM_NAMES = {
    0x04: "Whistle basement",
    0x05: "six-Darknut",
    0x06: "empty passage",
    0x07: "cellar stairs (to Digdogger side)",
    0x14: "Triforce room",
    0x16: "south key drop (NOT return)",
    0x24: "Digdogger",
    0x25: "Pols Voice west",
    0x27: "mixed north",
    0x37: "Darknut compass",
    0x47: "Gibdo west",
    0x55: "Zol north of 0x65",
    0x56: "north of 0x66",
    0x57: "east of 0x56",
    0x64: "Digdogger-side return",
    0x65: "west Gibdo pocket",
    0x66: "Gibdo key",
    0x76: "L5 entrance",
}


def pin(env):
    s = read_snapshot(env.get_ram())
    ram = env.get_ram()
    tf = int(read_u8(ram, ADDR_TRIFORCE))
    objs = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            objs.append(
                {
                    "slot": o.slot,
                    "type": o.type_id,
                    "type_hex": f"0x{o.type_id:02x}",
                    "name": object_name(o.type_id),
                    "hp": o.hp,
                    "x": o.x,
                    "y": o.y,
                }
            )
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "room_name": ROOM_NAMES.get(s.screen, f"L5 room 0x{s.screen:02x}"),
        "next": f"0x{s.next_screen:02x}",
        "x": s.link_x,
        "y": s.link_y,
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "health": int(s.health),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce_0x0671": tf,
        "tf_l5_bit": bool(tf & 0x10),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "tile": int(s.colliding_tile),
        "stair": bool(on_stair_tile(s)),
        "item": int(s.room_item_id),
        "objects": objs,
    }


def hop_rec(name, dest_name, before, after, extra=None):
    rec = {
        "hop": name,
        "from": before.get("screen_hex"),
        "from_name": before.get("room_name"),
        "dest": after.get("screen_hex"),
        "dest_name": after.get("room_name"),
        "xy": [after.get("x"), after.get("y")],
        "keys": after.get("keys"),
        "whistle": after.get("whistle"),
        "mode": after.get("mode"),
        "tf_l5_bit": after.get("tf_l5_bit"),
    }
    if extra:
        rec.update(extra)
    return rec


def wait_play(env, assist, n, max_f=240):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            idle(env, assist, n, 8)
            return True
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=n[0])
    return False


def leave_64_east_65(env, assist, n) -> dict:
    """Off 0x64 center stairs (do not re-enter cellar), east bomb hole → 0x65."""
    snap = read_snapshot(env.get_ram())
    # Step off stairs immediately — south gap, never linger on (120,141).
    walk_axis(env, assist, n, "y", 189, max_f=400)
    walk_axis(env, assist, n, "x", 80, max_f=300)
    walk_axis(env, assist, n, "y", 189, max_f=200)
    walk_axis(env, assist, n, "x", 208, max_f=400)
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 224, max_f=240)
    push_dir(env, assist, n, "RIGHT", frames=240)
    idle(env, assist, n, 16)
    wait_play(env, assist, n)
    snap = read_snapshot(env.get_ram())
    return {
        "path": "south_gap_east_bomb_hole",
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == 0x65 and snap.mode == PLAY_MODE,
    }


def north_65_to_55(env, assist, n) -> dict:
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=300)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 16)
    wait_play(env, assist, n)
    snap = read_snapshot(env.get_ram())
    return {
        "path": "65_north_shutter",
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == 0x55 and snap.mode == PLAY_MODE,
    }


def try_door(env, assist, n, direction, ax, ay):
    room0 = read_snapshot(env.get_ram()).screen
    walk_axis(env, assist, n, "y", ay, max_f=360)
    walk_axis(env, assist, n, "x", ax, max_f=360)
    push_dir(env, assist, n, direction, frames=220)
    idle(env, assist, n, 12)
    wait_play(env, assist, n)
    s = read_snapshot(env.get_ram())
    return {
        "dir": direction,
        "changed": s.screen != room0,
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "mode": s.mode,
        "doors": int(s.cur_opened_doors),
    }


def fight_digdogger(env, assist, n) -> dict:
    from dataclasses import replace

    from zelda_i.dungeon import DungeonPhase, GenericDungeonRoomController, RewardKind, RewardSpec, DoorRoute
    from zelda_i.level3_dungeon import ROOM_5B_SPEC, ROOM_59_SPEC
    from zelda_i.level5_path import fight_blue_darknuts

    menu = select_b_item_menu(env, assist, n, 5)
    for _ in range(6):
        env.step(nes_action("B"))
        n[0] += 1
        assist.apply_env(env, frame=n[0])
        idle(env, assist, n, 50)
    idle(env, assist, n, 40)
    snap = read_snapshot(env.get_ram())
    bosses = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == 0x38 and o.hp > 0]
    after_b = pin(env)
    fight = None
    if bosses:
        spec = replace(
            ROOM_5B_SPEC,
            spec_id="level5_digdogger",
            source_room=0x64,
            room_id=0x24,
            entry=DoorRoute("LEFT", ((224, 141),)),
            enemy_types=(0x38,),
            expected_enemy_count=len(bosses),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("UP", ((120, 93),)),),
            max_frames=20000,
            level=LEVEL_5,
        )
        ctl = GenericDungeonRoomController(spec)
        for _ in range(spec.max_frames):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            env.step(action.action)
            n[0] += 1
            assist.apply_env(env, frame=n[0])
            if ctl.success or ctl.phase is DungeonPhase.FAILED:
                break
        fight = {"ok": bool(ctl.success), "frames": ctl.frames}
    leftovers = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = fight_blue_darknuts(env, assist, n, 0x24, expected=len(leftovers), source=0x64)
    # Heart container then north TF.
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 224, max_f=400)
    idle(env, assist, n, 12)
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 20)
    wait_play(env, assist, n)
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109)):
        walk_axis(env, assist, n, "y", ty, max_f=200)
        walk_axis(env, assist, n, "x", tx, max_f=200)
        idle(env, assist, n, 10)
        if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) != tf0:
            break
    snap = read_snapshot(env.get_ram())
    tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "menu": menu,
        "after_whistle": after_b,
        "bosses_after_b": len(bosses),
        "fight": fight,
        "extra": extra,
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & 0x10),
        "room": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "died": False,
    }


def main() -> int:
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle_to_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    hops = []
    start = None
    blocker = None
    last = None
    mapped_07_64 = False
    digdogger_died = False
    fight = None
    png = None

    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n[0] += 1
        assist.apply_env(env, frame=0)
        idle(env, assist, n, 12)
        start = pin(env)
        last = start
        print("START", {k: start[k] for k in start if k != "objects"}, "objs", start["objects"], flush=True)

        if start["whistle"] != 1 or start["screen"] != 0x04:
            blocker = "start_not_level5whistle_cellar"
        else:
            before = last
            leave = exit_whistle_04(env, assist, n)
            idle(env, assist, n, 10)
            last = pin(env)
            hops.append(hop_rec("leave_0x04", "six-Darknut", before, last, {"walker": leave.get("path"), "ok": leave.get("success")}))
            print("HOP04", hops[-1], flush=True)
            if not (last["screen"] == 0x05 and last["mode"] == PLAY_MODE and last["whistle"] == 1):
                blocker = "fail_leave_0x04"

        if blocker is None:
            before = last
            east = walk_east_from_05(env, assist, n)
            idle(env, assist, n, 8)
            last = pin(env)
            hops.append(hop_rec("0x05_east", "empty passage", before, last, {"ok": east.get("success")}))
            print("HOP05", hops[-1], flush=True)
            if not (last["screen"] == 0x06 and last["mode"] == PLAY_MODE and last["whistle"] == 1):
                blocker = "fail_east_0x05_to_0x06"

        if blocker is None:
            before = last
            stairs = take_block_stairs_06(env, assist, n)
            idle(env, assist, n, 10)
            last = pin(env)
            hops.append(
                hop_rec(
                    "0x06_block_stairs",
                    "cellar stairs",
                    before,
                    last,
                    {
                        "ok": stairs.get("success"),
                        "cellar": stairs.get("cellar"),
                        "south_key_drop": stairs.get("south_key_drop"),
                        "path": stairs.get("path"),
                        "log_tail": stairs.get("log", [])[-8:],
                    },
                )
            )
            print("HOP06", {k: hops[-1][k] for k in hops[-1] if k != "log_tail"}, "tail", hops[-1].get("log_tail"), flush=True)
            if last["screen"] == 0x16:
                blocker = "took_south_key_drop_0x16"
            elif not (stairs.get("success") or last["mode"] in (9, 10, 11, 16) or last["screen"] == 0x07):
                blocker = "fail_0x06_stairs_unmapped"

        if blocker is None:
            before = last
            to64 = cellar_07_to_64(env, assist, n)
            idle(env, assist, n, 10)
            last = pin(env)
            mapped_07_64 = bool(to64.get("success") and last["screen"] == 0x64)
            hops.append(hop_rec("0x07_left_mouth", "Digdogger-side return", before, last, {"ok": to64.get("success")}))
            print("HOP07", hops[-1], flush=True)
            if not mapped_07_64:
                blocker = f"fail_0x07_dest_0x{last['screen']:02x}_not_0x64"

        if blocker is None:
            before = last
            east65 = leave_64_east_65(env, assist, n)
            last = pin(env)
            hops.append(hop_rec("0x64_east", "west Gibdo pocket", before, last, {"ok": east65.get("success"), "path": east65.get("path")}))
            print("HOP64", hops[-1], flush=True)
            if last["screen"] != 0x65:
                blocker = f"fail_0x64_east_landed_0x{last['screen']:02x}"

        if blocker is None:
            before = last
            # 0x65 N shutter is one-way from 0x55. Bomb east → 0x66.
            be = bomb_east_from_65(env, assist, n)
            last = pin(env)
            hops.append(hop_rec("0x65_bomb_east", "Gibdo key", before, last, {"ok": be.get("success"), "bombs_spent": be.get("bombs_spent")}))
            print("HOP65E", hops[-1], flush=True)
            if last["screen"] != 0x66:
                blocker = f"fail_0x65_bomb_east_landed_0x{last['screen']:02x}"

        # Known cleared chain: 66 UP 56 RIGHT 57 UP 47 UP 37 UP 27 WEST 26 WEST 25 WEST 24
        chain = [
            ("0x66_north", "UP", 120, 93, 0x56, "north of 0x66"),
            ("0x56_east", "RIGHT", 224, 141, 0x57, "east of 0x56"),
            ("0x57_north", "UP", 120, 93, 0x47, "Gibdo west"),
            ("0x47_north", "UP", 120, 93, 0x37, "Darknut compass"),
            ("0x37_north", "UP", 120, 93, 0x27, "mixed north"),
        ]
        if blocker is None:
            for name, direction, ax, ay, expect, ename in chain:
                if last["screen"] == expect:
                    continue
                before = last
                d = try_door(env, assist, n, direction, ax, ay)
                last = pin(env)
                hops.append(hop_rec(name, ename, before, last, d))
                print("CHAIN", hops[-1], flush=True)
                if last["screen"] != expect:
                    blocker = f"fail_{name}_landed_0x{last['screen']:02x}"
                    break

        if blocker is None and last["screen"] == 0x27:
            before = last
            w = walk_west_from_27(env, assist, n)
            last = pin(env)
            hops.append(hop_rec("0x27_west", "Gibdo west-26", before, last, {"ok": w.get("success")}))
            print("HOP27", hops[-1], flush=True)
            if last["screen"] != 0x26:
                blocker = f"fail_0x27_west_0x{last['screen']:02x}"

        if blocker is None and last["screen"] == 0x26:
            before = last
            w = walk_west_from_26(env, assist, n)
            last = pin(env)
            hops.append(hop_rec("0x26_west", "Pols Voice west", before, last, {"ok": w.get("success")}))
            print("HOP26", hops[-1], flush=True)
            if last["screen"] != 0x25:
                blocker = f"fail_0x26_west_0x{last['screen']:02x}"

        if blocker is None and last["screen"] == 0x25:
            before = last
            w = walk_west_from_25(env, assist, n)
            last = pin(env)
            hops.append(hop_rec("0x25_west", "Digdogger", before, last, {"ok": w.get("success")}))
            print("HOP25", hops[-1], flush=True)
            if last["screen"] != 0x24:
                blocker = f"fail_0x25_west_0x{last['screen']:02x}"

        if blocker is None and last["screen"] == 0x24 and last["mode"] == PLAY_MODE and last["whistle"] == 1:
            fight = fight_digdogger(env, assist, n)
            last = pin(env)
            digdogger_died = bool(fight.get("tf_l5")) or (
                fight.get("fight") or {}
            ).get("ok", False)
            hops.append(
                {
                    "hop": "digdogger",
                    "dest": last["screen_hex"],
                    "dest_name": last["room_name"],
                    "xy": [last["x"], last["y"]],
                    "keys": last["keys"],
                    "whistle": last["whistle"],
                    "tf_l5_bit": last["tf_l5_bit"],
                    "fight": fight,
                }
            )
            print("DIGDOGGER", hops[-1], flush=True)
            if not last["tf_l5_bit"]:
                blocker = "digdogger_no_tf_bit_0x10"

        obs, *_ = env.step(nes_idle_action())
        n[0] += 1
        png = out / f"{TAG}_final.png"
        save_rgb_png(obs, png)
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
    report = {
        "ok": last is not None and last.get("whistle") == 1,
        "segment": TAG,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "pokes": False,
        "start_state": "Level5Whistle",
        "start": start,
        "hops": hops,
        "final": last,
        "mapped_0x07_to_0x64": mapped_07_64,
        "digdogger_died": digdogger_died,
        "tf_l5_bit": None if last is None else last.get("tf_l5_bit"),
        "triforce_0x0671": None if last is None else last.get("triforce_0x0671"),
        "level5_complete_claimed": False,
        "total_frames": n[0],
        "png": str(png) if png else None,
        "bk2": bk2,
        "blocker": blocker,
        "fight": fight,
    }
    path = out / f"{TAG}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} blocker={blocker} "
        f"mapped07_64={mapped_07_64} tf={None if last is None else last.get('tf_l5_bit')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
