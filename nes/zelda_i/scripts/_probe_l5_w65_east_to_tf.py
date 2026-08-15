"""Level5Whistle65 ONLY: east 0x66, walk to 0x24, whistle Digdogger, TF 0x10.

No north from 0x65. No Level5Cleared25. No pokes. No Clean STATUS. No L6-L8.
If 0x65 east fails: dump doors/tiles and STOP (do not invent 0x67).
0x06 stairs is fallback only and is not used here.
"""
from __future__ import annotations

from dataclasses import replace

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    DoorRoute,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import exit_door, idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
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
    bomb_east_from_65,
    select_b_item_menu,
    walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

START = "Level5Whistle65"
DIG = 0x38
SMALL = 0x18
TF_BIT = 0x10
HC = 0x1A


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])


def inv(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    return {
        "room": f"0x{s.screen:02x}",
        "mode": s.mode,
        "xy": [s.link_x, s.link_y],
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5": bool(int(read_u8(ram, ADDR_TRIFORCE)) & TF_BIT),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "health": int(s.health),
        "item": int(s.room_item_id),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "tile": int(s.colliding_tile),
    }


def objs(snap) -> list[dict]:
    out = []
    for o in snap.objects:
        if not (1 <= o.slot <= 12) or o.type_id in (0, 0xFF):
            continue
        out.append({"slot": o.slot, "type": o.type_id, "hp": o.hp, "x": o.x, "y": o.y})
    return out


def live_types(snap, types) -> list:
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in types and o.hp > 0
    ]


def wait_play(env, assist, total, max_f=240):
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and not s.transitioning:
            return
        step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 8)


def dump_fail(env, tag: str) -> dict:
    s = read_snapshot(env.get_ram())
    body = {
        "tag": tag,
        "inv": inv(env),
        "compact": compact_snapshot(s),
        "objects": objs(s),
        "pokes": False,
        "status_claim": None,
    }
    png = RECORDINGS_DIR / f"{tag}.png"
    save_rgb_png(env.step(nes_idle_action())[0], png)
    body["screenshot"] = str(png.resolve())
    write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
    return body


def save_ckpt(env, name: str, via: str, extra: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{START}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": START,
            "via": via,
            "key_poke": False,
            "door_poke": False,
            "bomb_count_poke": False,
            "selected_item_poke": False,
        },
        selected_trial={"success": True, **extra},
        natural_entry=False,
    )
    print("CKPT", name, extra, flush=True)
    return name


def door(env, assist, total, direction: str, expect: int, **kw) -> dict:
    rec = exit_door(env, assist, total, direction, **kw)
    wait_play(env, assist, total)
    s = read_snapshot(env.get_ram())
    ok = s.level == LEVEL_5 and s.screen == expect and s.mode == PLAY_MODE
    print("DOOR", direction, "->", f"0x{s.screen:02x}", "expect", f"0x{expect:02x}", "ok", ok, "xy", [s.link_x, s.link_y], flush=True)
    rec["ok"] = ok
    rec["dest"] = s.screen
    rec["xy"] = [s.link_x, s.link_y]
    rec["mode"] = s.mode
    return rec


def fight_ctl(env, assist, total, spec, controller_cls=GenericDungeonRoomController) -> dict:
    ctl = controller_cls(spec)
    start_n = None
    last_n = None
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
            elif len(live) != last_n:
                last_n = len(live)
        action = ctl.step(snap)
        step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = spec.live_enemies(snap) if snap.mode == PLAY_MODE and snap.screen == spec.room_id else []
    rec = {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "spec": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "phase": str(ctl.phase),
    }
    print("FIGHT", rec["spec"], rec["ok"], "n", rec["start_n"], "->", rec["end_n"], "f", rec["frames"], flush=True)
    return rec


def walk_east_hole(env, assist, total) -> dict:
    """Diamond then RIGHT through existing 0x65 east bomb hole, no new bomb."""
    walk_axis(env, assist, total, "y", 109, max_f=400)
    walk_axis(env, assist, total, "x", 208, max_f=500)
    walk_axis(env, assist, total, "y", 141, max_f=400)
    walk_axis(env, assist, total, "x", 224, max_f=200)
    room0 = read_snapshot(env.get_ram()).screen
    for _ in range(280):
        s = read_snapshot(env.get_ram())
        if s.screen != room0 or s.mode in (6, 7, 4):
            break
        step(env, assist, total, nes_action("RIGHT"))
    wait_play(env, assist, total)
    s = read_snapshot(env.get_ram())
    return {
        "path": "walk_east_existing_hole",
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "mode": s.mode,
        "success": s.level == LEVEL_5 and s.screen == 0x66 and s.mode == PLAY_MODE,
    }


def north_pinch(env, assist, total, expect: int) -> dict:
    walk_axis(env, assist, total, "y", 109, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=300)
    rec = door(env, assist, total, "UP", expect, x_force=120, y_force=93, push=240)
    if rec.get("ok"):
        return rec
    walk_axis(env, assist, total, "y", 109, max_f=300)
    walk_axis(env, assist, total, "x", 112, max_f=200)
    walk_axis(env, assist, total, "x", 120, max_f=200)
    return door(env, assist, total, "UP", expect, x_force=120, y_force=93, push=280)


def grab_item(env, assist, total, pred, stands, max_each=220) -> bool:
    if pred(env):
        return True
    for tx, ty in stands:
        walk_axis(env, assist, total, "y", ty, max_f=max_each)
        walk_axis(env, assist, total, "x", tx, max_f=max_each)
        idle(env, assist, total, 8)
        if pred(env):
            return True
    return pred(env)


def east_from_65(env, assist, total) -> dict:
    walked = walk_east_hole(env, assist, total)
    print("WALK_EAST", walked, flush=True)
    if walked.get("success"):
        walked["via"] = "existing_hole"
        return walked
    bombed = bomb_east_from_65(env, assist, total)
    print("BOMB_EAST", {k: bombed[k] for k in bombed if k != "menu"}, flush=True)
    bombed["via"] = "bomb_east"
    bombed["walk_first"] = walked
    return bombed


def digdogger(env, assist, total) -> dict:
    start = inv(env)
    print("AT24", start, "foes", objs(read_snapshot(env.get_ram())), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_24_arrive.png")
    menu = select_b_item_menu(env, assist, total, 5)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    walk_axis(env, assist, total, "y", 189, max_f=400)
    idle(env, assist, total, 8)
    print("STAND", inv(env), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_24_stand.png")
    shrunk = False
    log = []
    for i in range(8):
        step(env, assist, total, nes_action("B"))
        idle(env, assist, total, 70)
        snap = read_snapshot(env.get_ram())
        types = [o.type_id for o in live_types(snap, tuple(range(1, 0x80)))]
        rec = {"i": i, "types": [f"0x{t:02x}" for t in types], **inv(env)}
        log.append(rec)
        print("B", rec, flush=True)
        if SMALL in types or (types and DIG not in types):
            shrunk = True
            break
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_24_after_b.png")
    snap = read_snapshot(env.get_ram())
    small = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (SMALL, DIG) and o.hp > 0]
    fight = None
    if small:
        types = tuple({o.type_id for o in small})
        spec = replace(
            ROOM_5B_SPEC,
            spec_id="level5_w65_small_digdogger",
            source_room=0x25,
            room_id=0x24,
            entry=DoorRoute("LEFT", ((224, 141),)),
            enemy_types=types,
            expected_enemy_count=len(small),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("UP", ((120, 93),)),),
            max_frames=16000,
            level=LEVEL_5,
        )
        fight = fight_ctl(env, assist, total, spec)
    killed = not any(
        o.type_id in (SMALL, DIG) and o.hp > 0
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12
    )
    hc0 = ((read_snapshot(env.get_ram()).health >> 4) & 0x0F) + 1
    heart = grab_item(
        env,
        assist,
        total,
        lambda e: (((read_snapshot(e.get_ram()).health >> 4) & 0x0F) + 1) > hc0,
        ((224, 141), (120, 141), (120, 157), (96, 141), (144, 141), (120, 125), (80, 141), (160, 141), (120, 173)),
    )
    print("HEART", heart, "hc", hc0, "->", ((read_snapshot(env.get_ram()).health >> 4) & 0x0F) + 1, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_24_heart.png")
    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    push_dir(env, assist, total, "UP", frames=260)
    wait_play(env, assist, total)
    idle(env, assist, total, 16)
    print("NORTH", inv(env), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_14_arrive.png")
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    grab_item(
        env,
        assist,
        total,
        lambda e: int(read_u8(e.get_ram(), ADDR_TRIFORCE)) > tf0,
        ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109), (80, 141), (160, 141)),
    )
    for _ in range(500):
        if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT:
            break
        step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 30)
    ram = env.get_ram()
    tf1 = int(read_u8(ram, ADDR_TRIFORCE))
    rec = {
        "ok": bool(tf1 & TF_BIT),
        "menu": menu,
        "shrunk": shrunk,
        "fight": fight,
        "killed": killed,
        "heart": heart,
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & TF_BIT),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "final": inv(env),
        "log": log,
    }
    print("DIGDOGGER", rec["ok"], "shrunk", shrunk, "killed", killed, "tf", hex(tf1), flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_tf_done.png")
    return rec


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, START, GAME_DIR, render_mode="rgb_array")
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
        start = inv(env)
        print("START", start, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_east_start.png")
        if start["whistle"] != 1 or start["room"] != "0x65":
            failed, reason = start["room"], "not_65_whistle1"
            return finish(env, hops, checkpoints, start, boss, failed, reason)
        room = read_snapshot(env.get_ram()).screen

        if room == 0x65:
            rec = east_from_65(env, assist, total)
            hops.append({"hop": "65_east", **{k: rec[k] for k in rec if k != "menu"}})
            save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_w65_east_dest.png")
            if not rec.get("success"):
                dump_fail(env, "l5_w65_east_fail")
                failed, reason = "0x65", "east_did_not_enter_66"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle66", f"0x65 east via {rec.get('via')}", {**inv(env)}))
            room = 0x66

        if room == 0x66:
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE66 n", n, objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
            if n:
                spec = replace(
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
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_66", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x66", "gibdos_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            rec = door(env, assist, total, "UP", 0x56)
            hops.append({"hop": "66_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x66", "up_did_not_enter_56"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle56", "0x66 clear + UP 0x56", {**inv(env)}))
            room = 0x56

        if room == 0x56:
            rec = door(env, assist, total, "RIGHT", 0x57)
            hops.append({"hop": "56_east", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x56", "east_did_not_enter_57"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            room = 0x57

        if room == 0x57:
            rec = door(env, assist, total, "UP", 0x47)
            hops.append({"hop": "57_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x57", "up_did_not_enter_47"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE47 n", n, objs(snap), flush=True)
            if n:
                spec = replace(
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
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_47", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x47", "gibdos_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle47", "0x57 up, gibdo", {**inv(env)}))
            room = 0x47

        if room == 0x47:
            rec = north_pinch(env, assist, total, 0x37)
            hops.append({"hop": "47_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x47", "up_did_not_enter_37"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (0x0B, BLUE_DARKNUT_TYPE)))
            print("ARRIVE37 n", n, objs(snap), flush=True)
            if n:
                spec = replace(
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
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_37", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x37", "darknuts_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle37", "0x47 up, darknut", {**inv(env)}))
            room = 0x37

        if room == 0x37:
            rec = north_pinch(env, assist, total, 0x27)
            hops.append({"hop": "37_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x37", "up_did_not_enter_27"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (POLS_VOICE_OBJECT_TYPE, GIBDO_OBJECT_TYPE, 0x1B)))
            print("ARRIVE27 n", n, objs(snap), flush=True)
            if n:
                spec = replace(ROOM_27_SPEC, spec_id="level5_w65_27_mixed", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_27", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x27", "mixed_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            keys0 = read_snapshot(env.get_ram()).keys
            grab_item(env, assist, total, lambda e: read_snapshot(e.get_ram()).keys > keys0, ((120, 141), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141)))
            checkpoints.append(save_ckpt(env, "Level5Whistle27", "0x37 up, mixed", {**inv(env)}))
            room = 0x27

        if room == 0x27:
            west = walk_west_from_27(env, assist, total)
            wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x26 and snap.mode == PLAY_MODE
            hops.append({"hop": "27_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST27", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
            if not west.get("success"):
                failed, reason = "0x27", "west_did_not_enter_26"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE26 n", n, objs(snap), flush=True)
            if n:
                spec = replace(ROOM_26_SPEC, spec_id="level5_w65_26_gibdos", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_26", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x26", "gibdos_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            keys0 = read_snapshot(env.get_ram()).keys
            grab_item(env, assist, total, lambda e: read_snapshot(e.get_ram()).keys > keys0, ((224, 141), (120, 141), (96, 141), (144, 141)))
            checkpoints.append(save_ckpt(env, "Level5Whistle26", "0x27 west, gibdo", {**inv(env)}))
            room = 0x26

        if room == 0x26:
            west = walk_west_from_26(env, assist, total)
            wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x25 and snap.mode == PLAY_MODE
            hops.append({"hop": "26_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST26", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
            if not west.get("success"):
                failed, reason = "0x26", "west_did_not_enter_25"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (POLS_VOICE_OBJECT_TYPE,)))
            print("ARRIVE25 n", n, objs(snap), flush=True)
            if n:
                spec = replace(ROOM_25_SPEC, spec_id="level5_w65_25_pols", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_25", **fight})
                if not fight.get("ok"):
                    failed, reason = "0x25", "pols_not_cleared"
                    return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle25", "0x26 west, pols", {**inv(env)}))
            room = 0x25

        if room == 0x25:
            west = walk_west_from_25(env, assist, total)
            wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x24 and snap.mode == PLAY_MODE
            hops.append({"hop": "25_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST25", west.get("success"), "dest", f"0x{west.get('dest'):02x}", flush=True)
            if not west.get("success"):
                failed, reason = "0x25", "west_did_not_enter_24"
                return finish(env, hops, checkpoints, start, boss, failed, reason)
            checkpoints.append(save_ckpt(env, "Level5Whistle24", "0x25 west key, Digdogger", {**inv(env)}))
            room = 0x24

        if room != 0x24:
            failed, reason = f"0x{room:02x}", "not_in_24"
            return finish(env, hops, checkpoints, start, boss, failed, reason)

        boss = digdogger(env, assist, total)
        hops.append({"hop": "digdogger", **{k: boss[k] for k in boss if k != "log"}})
        if boss.get("tf_l5"):
            checkpoints.append(save_ckpt(env, "Level5Complete", "Whistle65 east-compose, Digdogger, TF 0x10", {**inv(env), "killed": boss.get("killed"), "shrunk": boss.get("shrunk")}))
        return finish(env, hops, checkpoints, start, boss, failed, reason)
    finally:
        env.close()


def finish(env, hops, checkpoints, start, boss, failed, reason):
    snap = read_snapshot(env.get_ram())
    png = RECORDINGS_DIR / "l5_w65_east_to_tf_final.png"
    save_rgb_png(env.step(nes_idle_action())[0], png)
    east = next((h for h in hops if h.get("hop") == "65_east"), None)
    body = {
        "ok": bool(boss and boss.get("tf_l5")),
        "status_claim": None,
        "pokes": False,
        "track": "assisted",
        "start_state": START,
        "start": start,
        "east_65_dest": None if east is None else f"0x{int(east.get('dest') or 0):02x}",
        "east_65_ok": bool(east and east.get("success")),
        "east_65_via": None if east is None else east.get("via"),
        "walk_to_24_ok": any(h.get("hop") == "25_west" and h.get("success") for h in hops) or (boss is not None),
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "tf_0x0671": int(read_u8(env.get_ram(), ADDR_TRIFORCE)),
        "tf_l5_bit_0x10": bool(int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT),
        "failed_room": failed,
        "reason": reason,
        "digdogger": None if boss is None else {k: boss[k] for k in boss if k != "log"},
        "final": {**inv(env), "objects": objs(snap)},
        "screenshot": str(png.resolve()),
    }
    write_json_report(RECORDINGS_DIR / "l5_w65_east_to_tf.json", body)
    print("OK", body["ok"], "EAST", body["east_65_dest"], body["east_65_via"], "WALK24", body["walk_to_24_ok"], "TF", hex(body["tf_0x0671"]), "FAILED", failed, reason, flush=True)
    return body


if __name__ == "__main__":
    r = main()
    print("RESULT_OK", r.get("ok"))
    print("EAST_DEST", r.get("east_65_dest"), "via", r.get("east_65_via"))
    print("WALK24", r.get("walk_to_24_ok"))
    print("HOPS", [(h.get("hop"), h.get("ok") or h.get("success"), h.get("dest")) for h in r.get("hops", [])])
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("TF_BIT", r.get("tf_l5_bit_0x10"), hex(r.get("tf_0x0671") or 0))
    print("status_claim", None)
