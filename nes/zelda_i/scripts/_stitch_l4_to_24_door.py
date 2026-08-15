"""Longest honest one-session tape ending at L5 0x24 Digdogger door.

Preferred start: Level4Complete. Chain existing controllers hop-by-hop.
Do not fight Digdogger (type 0x38). Do not write Level5Cleared24.
Whistle stays RAM-true (still 0). Not STATUS.
"""
from __future__ import annotations

from dataclasses import replace

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon import (
    AliveRule,
    CombatTuning,
    DoorRoute,
    DungeonPhase,
    DungeonRoomSpec,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ids import (
    DARKNUT_OBJECT_TYPE,
    GEL_OBJECT_TYPE,
    GEL_SPLIT_OBJECT_TYPE,
    ZOL_OBJECT_TYPE,
    object_name,
)
from zelda_i.dungeon_ops import exit_door, goto, idle, push_dir
from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC, ROOM_ITEM_COMPASS
from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
    ROOM_77_SPEC,
    ROOM_ITEM_SMALL_KEY,
    ROOM_L5_POLS_77,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_in_room_26,
    level5_room_25_cleared,
    level5_room_26_cleared,
    level5_room_27_cleared,
    level5_room_56_arrived,
    level5_room_66_cleared,
    level5_room_77_key_success,
    make_pols_voice_controller,
)
from zelda_i.level5_overworld import (
    POST_L4_TO_LEVEL5_HOPS,
    OverworldToLevel5Controller,
    level5_entrance_success,
)
from zelda_i.level5_path import (
    level5_east_key_step,
    make_west65_controller,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LADDER, ADDR_RAFT, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

POST_L4_RETURN, SETTLE_MAX, PATH_MAX, ENTER77_MAX = 0x45, 1800, 40000, 2500
DIGDOGGER = 0x38
ZOL_TYPES = (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)
ROOM_NAMES = {
    0x76: "L5 entrance",
    0x66: "3x Gibdo first key",
    0x77: "East Key Pols Voice",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x55: "west Zols",
    0x47: "north Gibdos",
    0x37: "Darknuts + compass",
    0x27: "mixed Pols/Gibdo/Keese",
    0x26: "west Gibdos",
    0x25: "west Pols Voice",
    0x24: "Digdogger door",
}
_ZOL_PATROL = (
    (64, 109), (120, 109), (176, 109), (176, 141), (176, 173),
    (120, 173), (64, 173), (64, 141), (120, 141), (100, 125),
    (140, 157), (80, 157), (160, 125),
)

def room_name(screen):
    return ROOM_NAMES.get(int(screen), f"room 0x{int(screen):02x}")

def pin(env):
    s = read_snapshot(env.get_ram())
    return {
        "mode": s.mode, "level": s.level, "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "name": room_name(s.screen) if s.level == 5 else None,
        "x": s.link_x, "y": s.link_y, "keys": s.keys, "bombs": s.bombs,
        "doors": s.cur_opened_doors, "mask": s.open_doorway_mask,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "raft": int(read_u8(env.get_ram(), ADDR_RAFT)),
        "ladder": int(read_u8(env.get_ram(), ADDR_LADDER)),
        "triforce": s.triforce,
    }

def live_objects(env):
    s = read_snapshot(env.get_ram())
    out = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            out.append({"slot": o.slot, "type": o.type_id, "type_hex": f"0x{o.type_id:02x}",
                        "name": object_name(o.type_id), "hp": o.hp, "x": o.x, "y": o.y})
    return out

def step(env, action, assist, n):
    obs, *_ = env.step(action)
    n[0] += 1
    if assist:
        assist.apply_env(env, frame=n[0])
    return obs

def wait_play(env, assist, n, room, max_f=360):
    saw = False
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.level == LEVEL_5 and s.screen == room:
            saw = True
            if s.mode == PLAY_MODE and not s.transitioning:
                idle(env, assist, n, 16)
                return True
        step(env, nes_idle_action(), assist, n)
    s = read_snapshot(env.get_ram())
    return saw and s.level == LEVEL_5 and s.screen == room

def hop_ok(env, dest):
    s = read_snapshot(env.get_ram())
    return s.level == LEVEL_5 and s.screen == dest

def fight_spec(env, spec, assist, n, controller=None):
    ctl = controller or GenericDungeonRoomController(spec)
    obs = None
    for _ in range(spec.max_frames):
        if assist:
            assist.apply_env(env, frame=n[0])
        obs = step(env, ctl.step(read_snapshot(env.get_ram())).action, assist, n)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    return obs, ctl

def spec_57():
    return DungeonRoomSpec(
        spec_id="level5_room57_zols_type", source_room=0x56, room_id=0x57,
        entry=DoorRoute("RIGHT", ((208, 141),)), enemy_types=ZOL_TYPES,
        expected_enemy_count=5, alive_rule=AliveRule.TYPE,
        combat=CombatTuning(patrol=_ZOL_PATROL, engage_distance=48,
                            engage_attack_period=5, engage_attack_hold=3,
                            patrol_attack_period=8, patrol_attack_hold=2),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=8),
        max_frames=16000, level=LEVEL_5,
    )

def spec_47():
    return replace(
        ROOM_66_SPEC, spec_id="level5_room47_gibdos_reuse66",
        source_room=0x57, room_id=0x47, entry=DoorRoute("UP", ((120, 205),)),
        expected_enemy_count=5, required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_SMALL_KEY,
        exit_routes=(DoorRoute("UP", ((120, 93),)), DoorRoute("DOWN", ((120, 205),)),
                     DoorRoute("LEFT", ((32, 141),))),
        max_frames=28000,
    )

def spec_37():
    return replace(
        ROOM_5B_SPEC, spec_id="level5_room37_darknuts_reuse5b59",
        source_room=0x47, room_id=0x37, entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(DARKNUT_OBJECT_TYPE,), expected_enemy_count=3,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        room_item_id=ROOM_ITEM_COMPASS, combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("UP", ((120, 93),)), DoorRoute("DOWN", ((120, 205),))),
        max_frames=20000, level=LEVEL_5,
    )

def walk_north_from_47(env, assist, n):
    from zelda_i.level5_path import walk_axis
    snap = read_snapshot(env.get_ram())
    start_xy = [snap.link_x, snap.link_y]
    notes = []
    paths = (
        ("y173_x120_north", (("y", 173), ("x", 120), ("y", 93))),
        ("y189_x120_north", (("y", 189), ("x", 120), ("y", 93))),
        ("y109_x120_north", (("y", 109), ("x", 120), ("y", 93))),
        ("x64_y173_x120", (("x", 64), ("y", 173), ("x", 120), ("y", 93))),
        ("pinch_x128_north", (("x", 128), ("y", 93), ("x", 120))),
        ("y141_x120_north", (("y", 141), ("x", 120), ("y", 93))),
    )
    used = None
    for name, steps in paths:
        ok_all = True
        for axis, tgt in steps:
            ok = walk_axis(env, assist, n, axis, tgt, max_f=500)
            snap = read_snapshot(env.get_ram())
            notes.append(f"{name}:{axis}:{tgt}:ok={ok}:xy={snap.link_x},{snap.link_y}")
            if not ok:
                ok_all = False
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 4 and abs(snap.link_y - 93) <= 6:
            used = name
            notes.append(f"aligned_{name}")
            break
        if ok_all and abs(snap.link_x - 120) <= 8:
            used = name
            notes.append(f"near_{name}")
            break
    if used is None:
        notes.append("fallthrough_center")
        goto(env, assist, n, 120, 173, tol=4, max_f=400)
        goto(env, assist, n, 120, 93, tol=2, max_f=500)
    for _ in range(24):
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 1 and abs(snap.link_y - 93) <= 2:
            break
        if abs(snap.link_x - 120) > 1:
            step(env, nes_action("RIGHT" if snap.link_x < 120 else "LEFT"), assist, n)
        else:
            step(env, nes_action("DOWN" if snap.link_y < 93 else "UP"), assist, n)
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, n, "UP", frames=150)
    idle(env, assist, n, 16)
    snap = read_snapshot(env.get_ram())
    changed = snap.screen != room0
    if changed:
        wait_play(env, assist, n, snap.screen, max_f=240)
    snap = read_snapshot(env.get_ram())
    return {"changed_room": changed, "start_xy": start_xy, "at_mouth": at,
            "result_room": f"0x{snap.screen:02x}", "result_xy": [snap.link_x, snap.link_y],
            "notes": notes, "dest": snap.screen}

def movie_dir_for(start_state):
    if start_state == "Level4Complete":
        return "bk2_l4_to_24_door"
    if start_state in ("Level5EastKey", "Level5EastKey77"):
        return "bk2_eastkey_to_24_door"
    slug = start_state.replace("Level5", "").replace("Level", "l").lower()
    return f"bk2_{slug}_to_24_door"

def prefix_l4_to_56(env, assist, n, seams):
    obs = None
    for _ in range(SETTLE_MAX):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.level == 0 and s.screen == POST_L4_RETURN and not s.transitioning:
            break
        obs = step(env, nes_idle_action(), assist, n)
    ow = OverworldToLevel5Controller(hops=POST_L4_TO_LEVEL5_HOPS, max_frames=PATH_MAX)
    while not ow.success:
        s = read_snapshot(env.get_ram())
        if s.mode == 17 or ow.phase.name == "FAILED":
            break
        obs = step(env, ow.step(s).action, assist, n)
    ok76 = level5_entrance_success(env.get_ram()) and ow.success
    seams.append({"name": "0x76 L5 entrance", "ok": ok76, **pin(env)})
    print("76", ok76, pin(env), flush=True)
    if not ok76:
        return obs, "fail_76"
    obs, c66 = fight_spec(env, ROOM_66_SPEC, assist, n)
    ok66 = level5_room_66_cleared(env.get_ram())
    seams.append({"name": "0x66 3x Gibdo first key", "ok": ok66, "ctl": c66.report(), **pin(env)})
    print("66", ok66, pin(env), flush=True)
    if not ok66:
        return obs, "fail_66"
    for _ in range(ENTER77_MAX):
        s = read_snapshot(env.get_ram())
        if s.level == 5 and s.screen == ROOM_L5_POLS_77 and s.mode == PLAY_MODE:
            for _ in range(40):
                obs = step(env, nes_idle_action(), assist, n)
            break
        obs = step(env, level5_east_key_step(s).action, assist, n)
    obs, pols = fight_spec(env, ROOM_77_SPEC, assist, n, controller=make_pols_voice_controller())
    ok77 = level5_room_77_key_success(env.get_ram())
    seams.append({"name": "0x77 East Key Pols Voice", "ok": ok77, "ctl": pols.report(), **pin(env)})
    print("77", ok77, pin(env), flush=True)
    if not ok77:
        return obs, "fail_77"
    nav = make_west65_controller()
    for _ in range(nav.max_frames):
        obs = step(env, nav.step(read_snapshot(env.get_ram())).action, assist, n)
        if nav.success or nav.failed:
            break
    ok56 = level5_room_56_arrived(env.get_ram()) or hop_ok(env, 0x56)
    if hop_ok(env, 0x56) and read_snapshot(env.get_ram()).mode != PLAY_MODE:
        wait_play(env, assist, n, 0x56)
        ok56 = hop_ok(env, 0x56)
    seams.append({"name": "0x56 north Dodongos", "ok": ok56, "ctl": nav.report(), **pin(env)})
    print("56", ok56, pin(env), flush=True)
    if not ok56:
        return obs, "fail_56"
    return obs, None

def from_56_to_27(env, assist, n, seams):
    obs = None
    hop57 = exit_door(env, assist, n, "RIGHT")
    wait_play(env, assist, n, 0x57)
    ok57e = hop_ok(env, 0x57)
    print("hop56_57", ok57e, hop57.get("result"), pin(env), flush=True)
    if not ok57e:
        seams.append({"name": "0x56 east to 0x57", "ok": False, "hop": hop57, **pin(env)})
        return obs, "fail_hop_56_to_57"
    obs, c57 = fight_spec(env, spec_57(), assist, n)
    live57 = [o for o in read_snapshot(env.get_ram()).objects
              if 1 <= o.slot <= 12 and o.type_id in ZOL_TYPES]
    ok57 = hop_ok(env, 0x57) and not live57 and (
        c57.success or read_snapshot(env.get_ram()).room_all_dead >= 8)
    seams.append({"name": "0x57 east Zols", "ok": ok57, "ctl": c57.report(), **pin(env)})
    print("57", ok57, c57.report().get("phase"), pin(env), flush=True)
    if not ok57:
        return obs, "fail_clear_57"
    hop47 = exit_door(env, assist, n, "UP")
    wait_play(env, assist, n, 0x47)
    ok47e = hop_ok(env, 0x47)
    print("hop57_47", ok47e, hop47.get("result"), pin(env), flush=True)
    if not ok47e:
        seams.append({"name": "0x57 north to 0x47", "ok": False, "hop": hop47, **pin(env)})
        return obs, "fail_hop_57_to_47"
    obs, c47 = fight_spec(env, spec_47(), assist, n)
    ok47 = hop_ok(env, 0x47) and c47.success
    seams.append({"name": "0x47 north Gibdos", "ok": ok47, "ctl": c47.report(), **pin(env)})
    print("47", ok47, c47.report().get("phase"), pin(env), flush=True)
    if not ok47:
        return obs, "fail_clear_47"
    hop37 = walk_north_from_47(env, assist, n)
    wait_play(env, assist, n, 0x37)
    ok37e = hop_ok(env, 0x37)
    print("hop47_37", ok37e, hop37, pin(env), flush=True)
    if not ok37e:
        seams.append({"name": "0x47 north to 0x37", "ok": False, "hop": hop37, **pin(env)})
        return obs, "fail_hop_47_to_37"
    obs, c37 = fight_spec(env, spec_37(), assist, n)
    ok37 = hop_ok(env, 0x37) and c37.success
    seams.append({"name": "0x37 Darknuts + compass", "ok": ok37, "ctl": c37.report(), **pin(env)})
    print("37", ok37, c37.report().get("phase"), pin(env), flush=True)
    if not ok37:
        return obs, "fail_clear_37"
    return obs, None

def from_37_to_24(env, assist, n, seams):
    obs, c27 = fight_spec(env, ROOM_27_SPEC, assist, n)
    ok27 = level5_room_27_cleared(env.get_ram())
    seams.append({"name": "0x27 mixed Pols/Gibdo/Keese", "ok": ok27, "ctl": c27.report(), **pin(env)})
    print("27", ok27, pin(env), flush=True)
    if not ok27:
        return obs, "fail_27"
    hop26 = walk_west_from_27(env, assist, n)
    wait_play(env, assist, n, 0x26)
    ok26e = level5_in_room_26(env.get_ram()) or hop_ok(env, 0x26)
    print("hop27_26", ok26e, hop26.get("dest"), pin(env), flush=True)
    if not ok26e:
        seams.append({"name": "0x27 west key to 0x26", "ok": False, "hop": hop26, **pin(env)})
        return obs, "fail_hop_27_to_26"
    obs, c26 = fight_spec(env, ROOM_26_SPEC, assist, n)
    ok26 = level5_room_26_cleared(env.get_ram())
    seams.append({"name": "0x26 west Gibdos", "ok": ok26, "hop": hop26, "ctl": c26.report(), **pin(env)})
    print("26", ok26, pin(env), flush=True)
    if not ok26:
        return obs, "fail_26"
    hop25 = walk_west_from_26(env, assist, n)
    wait_play(env, assist, n, 0x25)
    ok25e = level5_in_room_25(env.get_ram()) or hop_ok(env, 0x25)
    print("hop26_25", ok25e, hop25.get("dest"), pin(env), flush=True)
    if not ok25e:
        seams.append({"name": "0x26 west to 0x25", "ok": False, "hop": hop25, **pin(env)})
        return obs, "fail_hop_26_to_25"
    obs, c25 = fight_spec(env, ROOM_25_SPEC, assist, n,
                         controller=Level5PolsVoiceController(spec=ROOM_25_SPEC))
    ok25 = level5_room_25_cleared(env.get_ram())
    seams.append({"name": "0x25 west Pols Voice", "ok": ok25, "hop": hop25, "ctl": c25.report(), **pin(env)})
    print("25", ok25, pin(env), flush=True)
    if not ok25:
        return obs, "fail_25"
    hop24 = walk_west_from_25(env, assist, n)
    wait_play(env, assist, n, 0x24)
    door24 = level5_in_room_24(env.get_ram()) or hop_ok(env, 0x24)
    objs24 = live_objects(env)
    fought = any(o["type"] == DIGDOGGER and o["hp"] < 240 for o in objs24)
    seams.append({"name": "0x24 Digdogger door", "ok": bool(door24) and not fought,
                  "hop": hop24, "objects": objs24, "fought_digdogger": fought, **pin(env)})
    print("24", door24, "fought", fought, pin(env), objs24, flush=True)
    if not door24:
        return obs, "fail_hop_25_to_24"
    if fought:
        return obs, "fought_digdogger"
    return obs, None

def main(start_state="Level4Complete"):
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / movie_dir_for(start_state)
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    seams = []
    blocker = None
    obs = None
    start = None
    final = None
    tag = f"{start_state.lower()}_to_24_door_stitch".replace("level", "l")
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs = step(env, nes_idle_action(), assist, n)
        start = pin(env)
        print("start", start_state, start, flush=True)
        if start_state == "Level4Complete":
            obs, blocker = prefix_l4_to_56(env, assist, n, seams)
            if blocker is None:
                obs, blocker = from_56_to_27(env, assist, n, seams)
            if blocker is None:
                obs, blocker = from_37_to_24(env, assist, n, seams)
        elif start_state in ("Level5EastKey", "Level5EastKey77"):
            nav = make_west65_controller()
            for _ in range(nav.max_frames):
                obs = step(env, nav.step(read_snapshot(env.get_ram())).action, assist, n)
                if nav.success or nav.failed:
                    break
            ok56 = level5_room_56_arrived(env.get_ram()) or hop_ok(env, 0x56)
            if hop_ok(env, 0x56) and read_snapshot(env.get_ram()).mode != PLAY_MODE:
                wait_play(env, assist, n, 0x56)
                ok56 = hop_ok(env, 0x56)
            seams.append({"name": "0x56 north Dodongos", "ok": ok56, **pin(env)})
            print("56", ok56, pin(env), flush=True)
            if not ok56:
                blocker = "fail_56"
            else:
                obs, blocker = from_56_to_27(env, assist, n, seams)
                if blocker is None:
                    obs, blocker = from_37_to_24(env, assist, n, seams)
        elif start_state in ("Level5North56", "Level5Cleared56", "Level5Entered56"):
            obs, blocker = from_56_to_27(env, assist, n, seams)
            if blocker is None:
                obs, blocker = from_37_to_24(env, assist, n, seams)
        elif start_state == "Level5Cleared47":
            hop37 = walk_north_from_47(env, assist, n)
            wait_play(env, assist, n, 0x37)
            if not hop_ok(env, 0x37):
                blocker = "fail_hop_47_to_37"
            else:
                obs, c37 = fight_spec(env, spec_37(), assist, n)
                seams.append({"name": "0x37 Darknuts + compass", "ok": c37.success,
                              "ctl": c37.report(), **pin(env)})
                if not c37.success:
                    blocker = "fail_clear_37"
                else:
                    obs, blocker = from_37_to_24(env, assist, n, seams)
        elif start_state == "Level5Cleared37":
            obs, blocker = from_37_to_24(env, assist, n, seams)
        elif start_state == "Level5Cleared27":
            seams.append({"name": "0x27 mixed Pols/Gibdo/Keese", "ok": True, "skipped_fight": True, **pin(env)})
            hop26 = walk_west_from_27(env, assist, n)
            wait_play(env, assist, n, 0x26)
            if not (level5_in_room_26(env.get_ram()) or hop_ok(env, 0x26)):
                blocker = "fail_hop_27_to_26"
            else:
                obs, c26 = fight_spec(env, ROOM_26_SPEC, assist, n)
                if not level5_room_26_cleared(env.get_ram()):
                    blocker = "fail_26"
                else:
                    hop25 = walk_west_from_26(env, assist, n)
                    wait_play(env, assist, n, 0x25)
                    if not (level5_in_room_25(env.get_ram()) or hop_ok(env, 0x25)):
                        blocker = "fail_hop_26_to_25"
                    else:
                        obs, c25 = fight_spec(env, ROOM_25_SPEC, assist, n, controller=Level5PolsVoiceController(spec=ROOM_25_SPEC))
                        if not level5_room_25_cleared(env.get_ram()):
                            blocker = "fail_25"
                        else:
                            hop24 = walk_west_from_25(env, assist, n)
                            wait_play(env, assist, n, 0x24)
                            objs24 = live_objects(env)
                            fought = any(o["type"] == DIGDOGGER and o["hp"] < 240 for o in objs24)
                            door24 = level5_in_room_24(env.get_ram()) or hop_ok(env, 0x24)
                            seams.append({"name": "0x24 Digdogger door", "ok": bool(door24) and not fought, "hop": hop24, "objects": objs24, "fought_digdogger": fought, **pin(env)})
                            if not door24:
                                blocker = "fail_hop_25_to_24"
                            elif fought:
                                blocker = "fought_digdogger"
        else:
            blocker = f"unknown_start_{start_state}"
        final = pin(env)
        shot = out / f"{tag}_final.png"
        if obs is not None:
            save_rgb_png(obs, shot)
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
    door24 = bool(final and final.get("screen") == 0x24)
    whistle = final.get("whistle") if final else None
    report = {
        "ok": door24 and blocker is None,
        "segment": tag,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "start_state": start_state,
        "end_claim": "entered_0x24_door_only" if door24 and blocker is None else None,
        "did_not_fight_digdogger": blocker != "fought_digdogger",
        "did_not_write_cleared24": True,
        "whistle_0x065C": whistle,
        "total_frames": n[0],
        "start": start,
        "final": final,
        "seams": seams,
        "room_sequence": [f"0x{s.get('screen'):02x} {s.get('name')}" for s in seams if s.get("screen") is not None],
        "blocker": blocker,
        "bk2": bk2,
        "png": str(shot) if shot else None,
        "path_note": (
            "56->27 is 56 east 57 Zols, north 47 Gibdos, north 37 Darknuts, north 27. "
            "0x55 west Zols is a dead-end (DOWN 0x65 only), not the 47 path."
        ),
    }
    path = out / f"{tag}.json"
    write_json_report(path, report)
    print(f"wrote {path} frames={n[0]} bk2={bk2} door24={door24} whistle={whistle} blocker={blocker}", flush=True)
    return 0 if report["ok"] else 2

if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "Level4Complete"
    raise SystemExit(main(start))
