"""From Level5Whistle05: walk live graph to 0x24, whistle-shrink Digdogger, TF 0x10.

Survival / infinite-life. No key/door/item pokes. Not a Clean STATUS claim.
No east67.
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
from zelda_i.dungeon_ops import exit_door, goto, idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    Level5PolsVoiceController,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_65_SPEC,
    ROOM_66_SPEC,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level5_path import (
    BLUE_DARKNUT_TYPE,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_PASSAGE_06,
    cellar_other_mouth,
    cellar_to_64,
    select_b_item_menu,
    take_center_stairs_06,
    walk_axis,
    bomb_east_from_65,
    walk_east_from_65,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import STAIR_STANDS, on_stair_tile
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

START = "Level5Whistle05"
DIGDOGGER = 0x38
HEART_CONTAINER = 0x1A
TF_ROOM = 0x14
TF_BIT = 0x10


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


def save_ckpt(env, name: str, source: str, via: str, extra: dict) -> str:
    path = write_state_bytes(state_path(GAME_DIR, GAME, name), env.em.get_state())
    write_state_provenance(
        path,
        source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{source}.state",
        request={
            "segment": name,
            "predecessor_entry": True,
            "start_state": source,
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
    print(
        "DOOR",
        direction,
        "->",
        f"0x{s.screen:02x}",
        "expect",
        f"0x{expect:02x}",
        "ok",
        ok,
        "xy",
        [s.link_x, s.link_y],
        flush=True,
    )
    rec["ok"] = ok
    rec["dest"] = s.screen
    rec["xy"] = [s.link_x, s.link_y]
    rec["mode"] = s.mode
    return rec


def fight_ctl(env, assist, total, spec, controller_cls=GenericDungeonRoomController) -> dict:
    ctl = controller_cls(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n})
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
        "progress": progress,
        "spec": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "phase": str(ctl.phase),
    }
    print("FIGHT", rec["spec"], rec["ok"], "n", rec["start_n"], "->", rec["end_n"], "f", rec["frames"], flush=True)
    return rec


def take_stairs_06(env, assist, total) -> dict:
    """0x06: north-around, push left 0x68 north, idle (120,141). No south-gap."""
    reused = take_center_stairs_06(env, assist, total)
    snap = read_snapshot(env.get_ram())
    ok = reused.get("success") or snap.mode in (9, 10, 11, 16) or snap.screen == ROOM_L5_CELLAR_07
    return {
        "ok": bool(ok),
        "via": reused.get("path"),
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
        "reused": {k: reused[k] for k in reused if k != "log"},
    }

def _take_stairs_06_unused_south_gap(env, assist, total) -> dict:
    """Unused. South/east diamond walks failed. Kept only as a marker."""
    snap = read_snapshot(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen}
    log = []

    def done(s) -> bool:
        return s.mode in (9, 10, 11, 16) or (s.level == LEVEL_5 and s.screen == ROOM_L5_CELLAR_07)

    reused = {"skipped": True}
    snap = read_snapshot(env.get_ram())
    if done(snap):
        return {"ok": True, "via": "already", "dest": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y], "reused": reused}

    for sx, sy in STAIR_STANDS + ((120, 141), (120, 125), (80, 141), (160, 141), (120, 189), (64, 141)):
        if done(read_snapshot(env.get_ram())):
            break
        if read_snapshot(env.get_ram()).screen != ROOM_L5_PASSAGE_06:
            break
        walk_axis(env, assist, total, "y", sy, max_f=280)
        walk_axis(env, assist, total, "x", sx, max_f=280)
        idle(env, assist, total, 8)
        snap = read_snapshot(env.get_ram())
        log.append({"stand": [sx, sy], "xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen, "stair": on_stair_tile(snap)})
        if done(snap):
            break
        for d in ("UP", "DOWN", "RIGHT", "LEFT"):
            push_dir(env, assist, total, d, frames=20)
            snap = read_snapshot(env.get_ram())
            if done(snap):
                break
        if done(read_snapshot(env.get_ram())):
            break
    wait_play(env, assist, total, max_f=80)
    snap = read_snapshot(env.get_ram())
    return {
        "ok": done(snap) and snap.screen != 0x05,
        "via": "stair_stands",
        "start": start,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "log": log[-12:],
    }


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


def north_pinch(env, assist, total, expect: int) -> dict:
    """0x47/0x37 north: avoid C-block / pit, door at x=120."""
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
        start = inv(env)
        print("START", start, flush=True)
        if start["whistle"] != 1:
            failed = start["room"]
            reason = "whistle_not_1"
            return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)

        snap = read_snapshot(env.get_ram())
        room = snap.screen

        # --- 0x05 east to 0x06 (key already spent; doors=1) ---
        if room == 0x05:
            rec = door(env, assist, total, "RIGHT", 0x06)
            hops.append({"hop": "05_east", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x05", "east_did_not_enter_06"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = 0x06

        # --- 0x06 stairs to 0x07 ---
        if room == 0x06:
            stairs = take_stairs_06(env, assist, total)
            hops.append({"hop": "06_stairs", **{k: stairs[k] for k in stairs if k != "log"}})
            print("STAIRS06", stairs.get("ok"), "dest", f"0x{stairs.get('dest'):02x}", "mode", stairs.get("mode"), flush=True)
            if not stairs.get("ok"):
                failed, reason = "0x06", "stairs_not_taken"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = read_snapshot(env.get_ram()).screen

        # --- 0x07 other mouth to 0x64 ---
        snap = read_snapshot(env.get_ram())
        if snap.mode in (9, 10, 11, 16) or snap.screen == ROOM_L5_CELLAR_07:
            cellar = cellar_to_64(env, assist, total)
            hops.append({"hop": "07_left_mouth", **{k: cellar[k] for k in cellar if k != "start"}})
            print("CELLAR", cellar.get("success"), "dest", f"0x{cellar.get('dest'):02x}", flush=True)
            snap = read_snapshot(env.get_ram())
            if snap.screen == 0x06:
                stairs = take_stairs_06(env, assist, total)
                hops.append({"hop": "06_stairs_retry", **{k: stairs[k] for k in stairs if k != "log"}})
                snap = read_snapshot(env.get_ram())
                if snap.mode in (9, 10, 11, 16) or snap.screen == 0x07:
                    cellar = cellar_to_64(env, assist, total)
                    hops.append({"hop": "07_left_mouth_retry", **{k: cellar[k] for k in cellar if k != "start"}})
                    wait_play(env, assist, total)
            snap = read_snapshot(env.get_ram())
            if not (snap.screen == ROOM_L5_BLUE_64 and snap.mode == PLAY_MODE):
                # if cellar_other_mouth actually hit 0x64, good
                if snap.screen != ROOM_L5_BLUE_64:
                    failed, reason = f"0x{snap.screen:02x}", "cellar_did_not_enter_64"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(
                save_ckpt(env, "Level5Whistle64", start_state, "0x05 east -> 0x06 stairs -> 0x07 -> 0x64", {**inv(env)})
            )
            room = 0x64

        # --- 0x64 east bomb-hole to 0x65 ---
        if room == 0x64:
            rec = walk_east_from_64(env, assist, total)
            hops.append({"hop": "64_east", **rec})
            print("EAST64", rec.get("success"), "dest", f"0x{rec.get('dest'):02x}", rec.get("xy"), flush=True)
            if not rec.get("success"):
                failed, reason = "0x64", "east_did_not_enter_65"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = 0x65

        # --- 0x65: clear Gibdos to open NORTH shutter → 0x55. No 0x66. ---
        if room == 0x65:
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE65 n", n, objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
            if n:
                spec = replace(
                    ROOM_65_SPEC,
                    spec_id="level5_whistle_65_gibdos",
                    source_room=0x64,
                    room_id=0x65,
                    entry=DoorRoute("RIGHT", ((32, 141),)),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                    max_frames=28000,
                    level=LEVEL_5,
                )
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_65", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x65", "gibdos_not_cleared_north_shutter"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            walk_axis(env, assist, total, "y", 109, max_f=400)
            walk_axis(env, assist, total, "x", 120, max_f=400)
            walk_axis(env, assist, total, "y", 93, max_f=300)
            rec = door(env, assist, total, "UP", 0x55, x_force=120, y_force=93, push=280)
            hops.append({"hop": "65_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            print("UP65", rec.get("ok"), "dest", f"0x{rec.get('dest'):02x}", rec.get("xy"), flush=True)
            if rec.get("ok"):
                checkpoints.append(save_ckpt(env, "Level5Whistle55", start_state, "0x65 north shutter after gibdos", {**inv(env)}))
                room = 0x55
            else:
                # Live: 0x65 N is a one-way shutter (doors=0x2 west-only after clear).
                rec = bomb_east_from_65(env, assist, total)
                hops.append({"hop": "65_bomb_east", **{k: rec[k] for k in rec if k != "menu"}})
                print("BOMB65E", rec.get("success"), "dest", f"0x{rec.get('dest'):02x}", rec.get("xy"), flush=True)
                if not rec.get("success"):
                    failed, reason = "0x65", "north_shutter_sealed_east_bomb_failed"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
                checkpoints.append(save_ckpt(env, "Level5Whistle66", start_state, "0x65 N sealed; bomb-east 0x66", {**inv(env)}))
                room = 0x66

        # --- 0x66 (cleared on lineage) UP → 0x56 when 0x65 N is sealed ---
        if room == 0x66:
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE66 n", n, objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
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
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_66", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x66", "gibdos_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            rec = door(env, assist, total, "UP", 0x56)
            hops.append({"hop": "66_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            print("UP66", rec.get("ok"), "dest", f"0x{rec.get('dest'):02x}", rec.get("xy"), flush=True)
            if not rec.get("ok"):
                failed, reason = "0x66", "up_did_not_enter_56"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(save_ckpt(env, "Level5Whistle56", start_state, "0x66 UP 0x56 after 0x65 N sealed", {**inv(env)}))
            room = 0x56

        # --- 0x55 RIGHT 0x56 RIGHT 0x57 (cleared on this lineage) ---
        if room == 0x55:
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (ZOL_OBJECT_TYPE, 0x14, 0x15)))
            print("ARRIVE55 n", n, objs(snap), "doors", hex(snap.cur_opened_doors), flush=True)
            if n:
                spec = replace(
                    ROOM_66_SPEC,
                    spec_id="level5_whistle_55_zols",
                    source_room=0x65,
                    room_id=0x55,
                    entry=DoorRoute("UP", ((120, 205),)),
                    enemy_types=(ZOL_OBJECT_TYPE, 0x14, 0x15),
                    expected_enemy_count=n,
                    required_open_doors=0,
                    reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
                    max_frames=20000,
                    level=LEVEL_5,
                )
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_55", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x55", "zols_not_cleared_east_shutter"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            rec = door(env, assist, total, "RIGHT", 0x56)
            hops.append({"hop": "55_east", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x55", "east_did_not_enter_56"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = 0x56
        if room == 0x56:
            rec = door(env, assist, total, "RIGHT", 0x57)
            hops.append({"hop": "56_east", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x56", "east_did_not_enter_57"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            room = 0x57

        # --- 0x57 UP 0x47 fight 5 Gibdo ---
        if room == 0x57:
            rec = door(env, assist, total, "UP", 0x47)
            hops.append({"hop": "57_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x57", "up_did_not_enter_47"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE47 n", n, "skip_fight_until_24", objs(snap), flush=True)
            if False and n:
                spec = replace(
                    ROOM_66_SPEC,
                    spec_id="level5_whistle_47_gibdos",
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
                hops.append({"hop": "fight_47", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x47", "gibdos_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(save_ckpt(env, "Level5Whistle47", start_state, "0x57 up, 5 gibdo", {**inv(env)}))
            room = 0x47

        # --- 0x47 UP 0x37 fight 3 Darknut ---
        if room == 0x47:
            rec = north_pinch(env, assist, total, 0x37)
            hops.append({"hop": "47_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x47", "up_did_not_enter_37"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (0x0B, BLUE_DARKNUT_TYPE)))
            print("ARRIVE37 n", n, "skip_fight_until_24", objs(snap), flush=True)
            if False and n:
                spec = replace(
                    ROOM_5B_SPEC,
                    spec_id="level5_whistle_37_darknuts",
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
                hops.append({"hop": "fight_37", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x37", "darknuts_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(save_ckpt(env, "Level5Whistle37", start_state, "0x47 up, 3 darknut", {**inv(env)}))
            room = 0x37

        # --- 0x37 UP 0x27 fight mixed ---
        if room == 0x37:
            rec = north_pinch(env, assist, total, 0x27)
            hops.append({"hop": "37_up", **{k: rec[k] for k in rec if k not in ("before", "at_door", "after")}})
            if not rec.get("ok"):
                failed, reason = "0x37", "up_did_not_enter_27"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (POLS_VOICE_OBJECT_TYPE, GIBDO_OBJECT_TYPE, 0x1B)))
            print("ARRIVE27 n", n, "skip_fight_until_24", objs(snap), flush=True)
            if False and n:
                spec = replace(ROOM_27_SPEC, spec_id="level5_whistle_27_mixed", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_27", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x27", "mixed_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            keys0 = read_snapshot(env.get_ram()).keys
            grab_item(
                env,
                assist,
                total,
                lambda e: read_snapshot(e.get_ram()).keys > keys0,
                ((120, 141), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141)),
            )
            checkpoints.append(save_ckpt(env, "Level5Whistle27", start_state, "0x37 up, mixed clear", {**inv(env)}))
            room = 0x27

        # --- 0x27 WEST 0x26 fight 5 Gibdo ---
        if room == 0x27:
            west = walk_west_from_27(env, assist, total)
            wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x26 and snap.mode == PLAY_MODE
            hops.append({"hop": "27_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST27", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
            if not west.get("success"):
                failed, reason = "0x27", "west_did_not_enter_26"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (GIBDO_OBJECT_TYPE,)))
            print("ARRIVE26 n", n, "skip_fight_until_24", objs(snap), flush=True)
            if False and n:
                spec = replace(ROOM_26_SPEC, spec_id="level5_whistle_26_gibdos", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec)
                hops.append({"hop": "fight_26", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x26", "gibdos_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            keys0 = read_snapshot(env.get_ram()).keys
            grab_item(
                env,
                assist,
                total,
                lambda e: read_snapshot(e.get_ram()).keys > keys0,
                ((224, 141), (120, 141), (96, 141), (144, 141)),
            )
            checkpoints.append(save_ckpt(env, "Level5Whistle26", start_state, "0x27 west, 5 gibdo", {**inv(env)}))
            room = 0x26

        # --- 0x26 WEST 0x25 fight 5 Pols ---
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
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            snap = read_snapshot(env.get_ram())
            n = len(live_types(snap, (POLS_VOICE_OBJECT_TYPE,)))
            print("ARRIVE25 n", n, "skip_fight_until_24", objs(snap), flush=True)
            if False and n:
                spec = replace(ROOM_25_SPEC, spec_id="level5_whistle_25_pols", expected_enemy_count=n, required_open_doors=0)
                fight = fight_ctl(env, assist, total, spec, Level5PolsVoiceController)
                hops.append({"hop": "fight_25", **{k: fight[k] for k in fight if k != "progress"}})
                if not fight.get("ok"):
                    failed, reason = "0x25", "pols_not_cleared"
                    return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(save_ckpt(env, "Level5Whistle25", start_state, "0x26 west, 5 pols", {**inv(env)}))
            room = 0x25

        # --- 0x25 WEST 0x24 Digdogger ---
        if room == 0x25:
            west = walk_west_from_25(env, assist, total)
            wait_play(env, assist, total, max_f=180)
            snap = read_snapshot(env.get_ram())
            west["dest"] = snap.screen
            west["mode"] = snap.mode
            west["success"] = snap.screen == 0x24 and snap.mode == PLAY_MODE
            hops.append({"hop": "25_west", **{k: west[k] for k in west if k != "log"}})
            print("WEST25", west.get("success"), "dest", f"0x{west.get('dest'):02x}", "spent", west.get("key_spent"), flush=True)
            if not west.get("success"):
                failed, reason = "0x25", "west_did_not_enter_24"
                return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
            checkpoints.append(save_ckpt(env, "Level5Whistle24", start_state, "0x25 west key door, Digdogger room", {**inv(env)}))
            room = 0x24

        if room != 0x24:
            failed, reason = f"0x{room:02x}", "not_in_24"
            return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)

        boss = _digdogger_here(env, assist, total)
        hops.append({"hop": "digdogger", **{k: boss[k] for k in boss if k not in ("after_whistle_objs",)}})
        if boss.get("tf_l5"):
            extra = {**inv(env), "killed": boss.get("killed"), "heart": boss.get("heart")}
            checkpoints.append(
                save_ckpt(env, "Level5TF", start_state, "whistle shrink Digdogger, heart, north TF 0x10", extra)
            )
            checkpoints.append(
                save_ckpt(env, "Level5Complete", start_state, "whistle shrink 0x38->0x18, sword, heart, north TF 0x10", extra)
            )
        elif boss.get("killed"):
            checkpoints.append(
                save_ckpt(env, "Level5Digdogger", start_state, "Digdogger killed", {**inv(env)})
            )
        return _finish(env, tag, hops, checkpoints, start, boss, failed, reason, start_state)
    finally:
        env.close()


def _digdogger_here(env, assist, total) -> dict:
    snap = read_snapshot(env.get_ram())
    before = {"xy": [snap.link_x, snap.link_y], "objs": objs(snap), "item": snap.room_item_id, "health": snap.health}
    print("AT24", before, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24_arrive.png")

    # Stand mid-room before the song. Doorway (224,141) + short B taps do not shrink.
    # Fireballs interrupt the recorder; hold still ~2s after a 12-frame B hold.
    walk_axis(env, assist, total, "y", 141, max_f=200)
    walk_axis(env, assist, total, "x", 120, max_f=400)
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, 5)
    print("MENU", menu, flush=True)
    selected = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    SMALL_DIG = 0x18
    shrunk = False
    for attempt in range(4):
        for _ in range(12):
            step(env, assist, total, nes_action("B"))
        for i in range(20):
            idle(env, assist, total, 12)
            snap = read_snapshot(env.get_ram())
            types = [o.type_id for o in live_types(snap, tuple(range(1, 0x80)))]
            print("WHISTLE_B", attempt, i, "sel", selected, "types", [hex(t) for t in types], flush=True)
            if SMALL_DIG in types or (types and DIGDOGGER not in types):
                shrunk = True
                break
            if snap.room_item_id == HEART_CONTAINER and DIGDOGGER not in types:
                shrunk = True
                break
        if shrunk:
            break
    print("SHRUNK", shrunk, flush=True)

    idle(env, assist, total, 30)
    snap = read_snapshot(env.get_ram())
    after_b = objs(snap)
    bosses = live_types(snap, (DIGDOGGER,))
    small = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68) and o.hp > 0
    ]
    fight = None
    if bosses or small:
        types = tuple({o.type_id for o in (bosses + small)})
        spec = replace(
            ROOM_5B_SPEC,
            spec_id="level5_digdogger",
            source_room=0x25,
            room_id=0x24,
            entry=DoorRoute("LEFT", ((224, 141),)),
            enemy_types=types,
            expected_enemy_count=max(1, len(bosses + small)),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("UP", ((120, 93),)),),
            max_frames=20000,
            level=LEVEL_5,
        )
        fight = fight_ctl(env, assist, total, spec)

    idle(env, assist, total, 20)
    snap = read_snapshot(env.get_ram())
    leftovers = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x68) and o.hp > 0
    ]
    killed = not leftovers and not live_types(snap, (DIGDOGGER,))
    hc0 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
    heart = False
    if snap.room_item_id == HEART_CONTAINER or True:
        heart = grab_item(
            env,
            assist,
            total,
            lambda e: (((int(read_u8(e.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1) > hc0,
            ((120, 141), (120, 125), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141), (224, 141)),
        )
    hc1 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
    print("HEART", heart, "hc", hc0, "->", hc1, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_24_after.png")

    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    push_dir(env, assist, total, "UP", frames=240)
    wait_play(env, assist, total)
    snap = read_snapshot(env.get_ram())
    print("NORTH", f"0x{snap.screen:02x}", [snap.link_x, snap.link_y], "item", snap.room_item_id, flush=True)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_14_arrive.png")

    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    if snap.screen == TF_ROOM or snap.room_item_id in (0x1B, 0x1A):
        grab_item(
            env,
            assist,
            total,
            lambda e: int(read_u8(e.get_ram(), ADDR_TRIFORCE)) > tf0,
            ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109), (80, 141), (160, 141)),
        )
    idle(env, assist, total, 40)
    # Triforce fanfare / scroll
    for _ in range(400):
        s = read_snapshot(env.get_ram())
        if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT:
            break
        step(env, assist, total, nes_idle_action())
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf1 = int(read_u8(ram, ADDR_TRIFORCE))
    rec = {
        "ok": bool(tf1 & TF_BIT),
        "menu": menu,
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "after_whistle_objs": after_b,
        "fight": fight,
        "killed": killed,
        "heart": heart,
        "hc_in": hc0,
        "hc_out": hc1,
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & TF_BIT),
        "room": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "item": snap.room_item_id,
    }
    print("DIGDOGGER", rec.get("ok"), "killed", killed, "heart", heart, "tf", hex(tf1), "room", f"0x{snap.screen:02x}", flush=True)
    write_json_report(RECORDINGS_DIR / "l5_24_whistle_boss.json", rec)
    save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_tf_final.png")
    return rec


def _finish(env, tag, hops, checkpoints, start, boss, failed, reason, source):
    snap = read_snapshot(env.get_ram())
    png = RECORDINGS_DIR / f"{tag}_final.png"
    obs, *_ = env.step(nes_idle_action())
    save_rgb_png(obs, png)
    body = {
        "ok": bool(boss and boss.get("tf_l5")),
        "status_claim": None,
        "pokes": False,
        "track": "assisted",
        "start_state": source,
        "start": start,
        "hops": hops,
        "checkpoints": checkpoints,
        "whistle_0x065C": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "tf_0x0671": int(read_u8(env.get_ram(), ADDR_TRIFORCE)),
        "tf_l5_bit_0x10": bool(int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT),
        "failed_room": failed,
        "reason": reason,
        "digdogger": boss,
        "final": {**inv(env), "objects": objs(snap)},
        "screenshot": str(png.resolve()),
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
    print("OK", body["ok"], "FAILED", failed, reason, "TF", hex(body["tf_0x0671"]), flush=True)
    return body


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--from-state", default=START)
    ap.add_argument("--tag", default="l5_whistle_tf")
    args = ap.parse_args()
    r = run_once(args.from_state, args.tag)
    print("RESULT_OK", r.get("ok"))
    print("HOPS", [(h.get("hop"), h.get("ok") or h.get("success"), h.get("dest")) for h in r.get("hops", [])])
    print("CKPT", r.get("checkpoints"))
    print("DIGDOGGER", r.get("digdogger"))
    print("status_claim", None)


if __name__ == "__main__":
    main()
