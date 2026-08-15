"""One-session Level5Whistle -> 0x24 Digdogger -> triforce.

Start: Level5Whistle (0x04 cellar, whistle=1). Leave via x=176 ladder down,
floor left to x=48, UP to 0x05. Back through 0x06 stairs / 0x07 cellar to
0x64, then 65 -> 66 -> 56 -> 57 -> 47 -> 37 -> 27 -> 26 -> 25 -> 24.
Whistle shrink, sword Digdogger, north 0x14 for TF bit 0x10.

Assisted. One env. No pin overwrite. No STATUS claim unless 0x10 is real.
"""
from __future__ import annotations

import dataclasses
from dataclasses import replace as dc_replace

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
    level5_room_66_cleared,
    ROOM_ITEM_SMALL_KEY,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_in_room_26,
    level5_room_25_cleared,
    level5_room_26_cleared,
    level5_room_27_cleared,
)
from zelda_i.level5_path import (
    bomb_east_from_65,
    cellar_to_64,
    exit_whistle_04,
    fight_blue_darknuts,
    make_west65_controller,
    select_b_item_menu,
    take_block_stairs_06,
    take_center_stairs_06,
    walk_axis,
    walk_east_from_05,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

DIGDOGGER = 0x38
DIGDOGGER_SMALL = 0x18
ZOL_TYPES = (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)
ROOM_NAMES = {
    0x04: "Recorder / Whistle cellar",
    0x05: "six blue Darknuts + block stairs",
    0x06: "passage east of 0x05",
    0x07: "cellar from 0x64 stairs",
    0x14: "L5 triforce",
    0x15: "old man (bomb-south to 0x25)",
    0x16: "south key of 0x06",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x64: "blue Darknuts / stairs",
    0x65: "west Gibdos",
    0x66: "3x Gibdo first key",
    0x76: "L5 entrance",
    0x77: "East Key Pols Voice",
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
    tf = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    return {
        "mode": s.mode,
        "level": s.level,
        "screen": s.screen,
        "screen_hex": f"0x{s.screen:02x}",
        "name": room_name(s.screen) if s.level == 5 else None,
        "x": s.link_x,
        "y": s.link_y,
        "keys": s.keys,
        "bombs": s.bombs,
        "doors": s.cur_opened_doors,
        "mask": s.open_doorway_mask,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
        "triforce": tf,
        "tf_hex": hex(tf),
        "tf_l5_bit": bool(tf & 0x10),
    }


def live_objects(env):
    s = read_snapshot(env.get_ram())
    out = []
    for o in s.objects:
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
            out.append(
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
    return out


def step(env, action, assist, n):
    obs, *_ = env.step(action)
    n[0] += 1
    if assist:
        assist.apply_env(env, frame=n[0])
    return obs


def hold(env, assist, n, d, frames):
    obs = None
    for _ in range(frames):
        obs = step(env, nes_action(d), assist, n)
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
        spec_id="level5_room57_zols_type",
        source_room=0x56,
        room_id=0x57,
        entry=DoorRoute("RIGHT", ((208, 141),)),
        enemy_types=ZOL_TYPES,
        expected_enemy_count=5,
        alive_rule=AliveRule.TYPE,
        combat=CombatTuning(
            patrol=_ZOL_PATROL,
            engage_distance=48,
            engage_attack_period=5,
            engage_attack_hold=3,
            patrol_attack_period=8,
            patrol_attack_hold=2,
        ),
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=8),
        max_frames=16000,
        level=LEVEL_5,
    )


def spec_47():
    return dc_replace(
        ROOM_66_SPEC,
        spec_id="level5_room47_gibdos_reuse66",
        source_room=0x57,
        room_id=0x47,
        entry=DoorRoute("UP", ((120, 205),)),
        expected_enemy_count=5,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
        room_item_id=ROOM_ITEM_SMALL_KEY,
        exit_routes=(
            DoorRoute("UP", ((120, 93),)),
            DoorRoute("DOWN", ((120, 205),)),
            DoorRoute("LEFT", ((32, 141),)),
        ),
        max_frames=28000,
    )


def spec_37():
    return dc_replace(
        ROOM_5B_SPEC,
        spec_id="level5_room37_darknuts_reuse5b59",
        source_room=0x47,
        room_id=0x37,
        entry=DoorRoute("UP", ((120, 205),)),
        enemy_types=(DARKNUT_OBJECT_TYPE,),
        expected_enemy_count=3,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        room_item_id=ROOM_ITEM_COMPASS,
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("UP", ((120, 93),)), DoorRoute("DOWN", ((120, 205),))),
        max_frames=20000,
        level=LEVEL_5,
    )


def walk_north_from_47(env, assist, n):
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
        for axis, tgt in steps:
            ok = walk_axis(env, assist, n, axis, tgt, max_f=500)
            snap = read_snapshot(env.get_ram())
            notes.append(f"{name}:{axis}:{tgt}:ok={ok}:xy={snap.link_x},{snap.link_y}")
        snap = read_snapshot(env.get_ram())
        if abs(snap.link_x - 120) <= 4 and abs(snap.link_y - 93) <= 6:
            used = name
            notes.append(f"aligned_{name}")
            break
        if abs(snap.link_x - 120) <= 8:
            used = name
            notes.append(f"near_{name}")
            break
    if used is None:
        notes.append("fallthrough_center")
        goto(env, assist, n, 120, 173, tol=4, max_f=400)
        goto(env, assist, n, 120, 93, tol=2, max_f=500)
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, n, "UP", frames=180)
    idle(env, assist, n, 16)
    snap = read_snapshot(env.get_ram())
    changed = snap.screen != room0
    if changed:
        wait_play(env, assist, n, snap.screen, max_f=240)
    snap = read_snapshot(env.get_ram())
    return {
        "changed_room": changed,
        "start_xy": start_xy,
        "used": used,
        "result_room": f"0x{snap.screen:02x}",
        "result_xy": [snap.link_x, snap.link_y],
        "notes": notes,
        "dest": snap.screen,
    }


def fight_digdogger_and_tf(env, assist, n):
    # Off the east door, into the room, then recorder.
    idle(env, assist, n, 20)
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 160, max_f=400)
    idle(env, assist, n, 16)
    menu = select_b_item_menu(env, assist, n, 5)
    after_b = None
    for burst in range(6):
        for _ in range(16):
            step(env, nes_action("B"), assist, n)
        idle(env, assist, n, 160)
        after_b = live_objects(env)
        bosses = [o for o in after_b if o["type"] in (DIGDOGGER, DIGDOGGER_SMALL)]
        print("WHISTLE_BURST", burst, "bosses", bosses, flush=True)
        if bosses and any(o["hp"] < 240 for o in bosses):
            break
        if not bosses:
            break
    snap = read_snapshot(env.get_ram())
    bosses = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in (DIGDOGGER, DIGDOGGER_SMALL) and o.hp > 0]
    fight = None
    if bosses:
        spec = dc_replace(
            ROOM_5B_SPEC,
            spec_id="level5_digdogger",
            source_room=0x25,
            room_id=0x24,
            entry=DoorRoute("LEFT", ((224, 141),)),
            enemy_types=(DIGDOGGER, DIGDOGGER_SMALL),
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
            step(env, action.action, assist, n)
            if ctl.success or ctl.phase is DungeonPhase.FAILED:
                break
        fight = {"ok": bool(ctl.success), "frames": ctl.frames, "phase": str(ctl.phase)}
    leftovers = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A, DIGDOGGER, DIGDOGGER_SMALL) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = fight_blue_darknuts(env, assist, n, 0x24, expected=len(leftovers), source=0x25)
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 224, max_f=400)
    idle(env, assist, n, 12)
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    push_dir(env, assist, n, "UP", frames=220)
    idle(env, assist, n, 20)
    snap = read_snapshot(env.get_ram())
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    if snap.screen == 0x14 or snap.room_item_id == 0x1B:
        w0 = tf0
        for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109)):
            walk_axis(env, assist, n, "y", ty, max_f=200)
            walk_axis(env, assist, n, "x", tx, max_f=200)
            idle(env, assist, n, 10)
            if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) > w0:
                break
    idle(env, assist, n, 30)
    tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    snap = read_snapshot(env.get_ram())
    dead = not any(
        o.type_id in (DIGDOGGER, DIGDOGGER_SMALL) and o.hp > 0
        for o in snap.objects
        if 1 <= o.slot <= 12
    )
    return {
        "menu": menu,
        "after_whistle_objs": after_b,
        "fight": fight,
        "extra": extra,
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & 0x10),
        "digdogger_dead": dead,
        "room": snap.screen,
        "xy": [snap.link_x, snap.link_y],
    }



def walk_north_from_66(env, assist, n):
    """0x66 west mouth -> wait shutter -> south band y=173 -> x=120 -> 0x56.

    0x66 N=shutter. On re-entry it starts closed (mask has L+D only) and
    reopens after all-dead settle. River pinches west-door x=32 at y=93.
    Fallback: 0x65 north shutter -> 0x55 west Zols -> east shutter 0x56.
    """
    idle(env, assist, n, 80)
    s = read_snapshot(env.get_ram())
    doors0 = {"doors": int(s.cur_opened_doors), "mask": int(s.open_doorway_mask), "all_dead": int(s.room_all_dead)}
    print("66_doors_after_idle", doors0, pin(env), flush=True)
    # Trigger all-dead if anything is still live, then south-band to the mouth.
    walk_axis(env, assist, n, "y", 173, max_f=400)
    walk_axis(env, assist, n, "x", 120, max_f=500)
    idle(env, assist, n, 40)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    goto(env, assist, n, 120, 93, tol=3, max_f=240)
    push_dir(env, assist, n, "UP", frames=280)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x56)
    if hop_ok(env, 0x56):
        s = read_snapshot(env.get_ram())
        return {"path": "idle_south173_x120_up", "dest": s.screen, "xy": [s.link_x, s.link_y], "doors0": doors0, "success": True}
    # 65 north -> 55 east -> 56
    walk_axis(env, assist, n, "y", 141, max_f=400)
    walk_axis(env, assist, n, "x", 32, max_f=400)
    push_dir(env, assist, n, "LEFT", frames=220)
    idle(env, assist, n, 12)
    wait_play(env, assist, n, 0x65)
    if hop_ok(env, 0x65):
        idle(env, assist, n, 40)
        walk_axis(env, assist, n, "y", 109, max_f=300)
        walk_axis(env, assist, n, "x", 120, max_f=400)
        walk_axis(env, assist, n, "y", 93, max_f=300)
        push_dir(env, assist, n, "UP", frames=240)
        idle(env, assist, n, 16)
        wait_play(env, assist, n, 0x55)
        print("55_via_65", pin(env), flush=True)
        if hop_ok(env, 0x55):
            # Clear Zols if present so the east shutter opens.
            live = [o for o in read_snapshot(env.get_ram()).objects if 1 <= o.slot <= 12 and o.type_id in ZOL_TYPES]
            if live:
                spec55 = spec_57()
                spec55 = dc_replace(spec55, spec_id="level5_room55_zols", source_room=0x65, room_id=0x55, entry=DoorRoute("UP", ((120, 205),)))
                fight_spec(env, spec55, assist, n)
            hop56 = exit_door(env, assist, n, "RIGHT")
            wait_play(env, assist, n, 0x56)
            if hop_ok(env, 0x56):
                s = read_snapshot(env.get_ram())
                return {"path": "65_north_55_east", "dest": s.screen, "xy": [s.link_x, s.link_y], "success": True, "hop": hop56}
    s = read_snapshot(env.get_ram())
    return {
        "path": "failed",
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "doors0": doors0,
        "success": False,
    }


def walk_north_from_57(env, assist, n):
    """0x57 north is ROM-open. Combat often ends on the y=125 ladder.

    Horizontal ladder locks Y until Link reaches a bank (x≈32 or x≈208).
    """
    notes = []
    s = read_snapshot(env.get_ram())
    if abs(s.link_y - 125) <= 4:
        for tx in (32, 48, 208, 192, 64, 176):
            walk_axis(env, assist, n, "x", tx, max_f=400)
            s = read_snapshot(env.get_ram())
            notes.append(f"bank_x{tx}:xy={s.link_x},{s.link_y}")
            # Off the ladder if Y can change.
            if walk_axis(env, assist, n, "y", 141, max_f=80) or walk_axis(env, assist, n, "y", 109, max_f=80):
                notes.append(f"off_ladder_at_{s.link_x}")
                break
            s2 = read_snapshot(env.get_ram())
            if abs(s2.link_y - 125) > 4:
                notes.append(f"y_changed_{s2.link_y}")
                break
    for name, steps in (
        ("bank_y141_x120", (("y", 141), ("x", 120), ("y", 93))),
        ("bank_y109_x120", (("y", 109), ("x", 120), ("y", 93))),
        ("bank_y173_x120", (("y", 173), ("x", 120), ("y", 93))),
        ("x32_y141_x120", (("x", 32), ("y", 141), ("x", 120), ("y", 93))),
        ("x208_y141_x120", (("x", 208), ("y", 141), ("x", 120), ("y", 93))),
    ):
        for axis, tgt in steps:
            ok = walk_axis(env, assist, n, axis, tgt, max_f=400)
            s = read_snapshot(env.get_ram())
            notes.append(f"{name}:{axis}:{tgt}:ok={ok}:xy={s.link_x},{s.link_y}")
            if s.screen == 0x47:
                return {"path": name, "dest": 0x47, "success": True, "notes": notes}
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - 120) <= 6 and abs(s.link_y - 93) <= 8:
            break
    goto(env, assist, n, 120, 93, tol=3, max_f=240)
    push_dir(env, assist, n, "UP", frames=280)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x47)
    s = read_snapshot(env.get_ram())
    return {"path": "push_up", "dest": s.screen, "xy": [s.link_x, s.link_y], "success": hop_ok(env, 0x47), "notes": notes}

def from_56_to_24(env, assist, n, seams):
    hop57 = exit_door(env, assist, n, "RIGHT")
    wait_play(env, assist, n, 0x57)
    if not hop_ok(env, 0x57):
        seams.append({"name": "0x56 east to 0x57 east Zols", "ok": False, "hop": hop57, **pin(env)})
        return "fail_hop_56_to_57"
    obs, c57 = fight_spec(env, spec_57(), assist, n)
    live57 = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id in ZOL_TYPES
    ]
    ok57 = hop_ok(env, 0x57) and not live57
    seams.append({"name": "0x57 east Zols", "ok": ok57, "ctl": c57.report(), **pin(env)})
    print("57", ok57, pin(env), flush=True)
    if not ok57:
        return "fail_clear_57"
    hop47 = walk_north_from_57(env, assist, n)
    wait_play(env, assist, n, 0x47)
    if not hop_ok(env, 0x47):
        seams.append({"name": "0x57 north to 0x47 north Gibdos", "ok": False, "hop": hop47, **pin(env)})
        return "fail_hop_57_to_47"
    obs, c47 = fight_spec(env, spec_47(), assist, n)
    ok47 = hop_ok(env, 0x47) and c47.success
    seams.append({"name": "0x47 north Gibdos", "ok": ok47, "ctl": c47.report(), **pin(env)})
    print("47", ok47, pin(env), flush=True)
    if not ok47:
        return "fail_clear_47"
    hop37 = walk_north_from_47(env, assist, n)
    wait_play(env, assist, n, 0x37)
    if not hop_ok(env, 0x37):
        seams.append({"name": "0x47 north to 0x37 Darknuts + compass", "ok": False, "hop": hop37, **pin(env)})
        return "fail_hop_47_to_37"
    obs, c37 = fight_spec(env, spec_37(), assist, n)
    ok37 = hop_ok(env, 0x37) and c37.success
    seams.append({"name": "0x37 Darknuts + compass", "ok": ok37, "ctl": c37.report(), **pin(env)})
    print("37", ok37, pin(env), flush=True)
    if not ok37:
        return "fail_clear_37"
    hop27 = exit_door(env, assist, n, "UP")
    wait_play(env, assist, n, 0x27)
    if not hop_ok(env, 0x27):
        seams.append({"name": "0x37 north to 0x27 mixed Pols/Gibdo/Keese", "ok": False, **pin(env)})
        return "fail_hop_37_to_27"
    obs, c27 = fight_spec(env, ROOM_27_SPEC, assist, n)
    ok27 = level5_room_27_cleared(env.get_ram())
    seams.append({"name": "0x27 mixed Pols/Gibdo/Keese", "ok": ok27, "ctl": c27.report(), **pin(env)})
    print("27", ok27, pin(env), flush=True)
    if not ok27:
        return "fail_27"
    hop26 = walk_west_from_27(env, assist, n)
    wait_play(env, assist, n, 0x26)
    if not (level5_in_room_26(env.get_ram()) or hop_ok(env, 0x26)):
        seams.append({"name": "0x27 west key to 0x26 west Gibdos", "ok": False, "hop": hop26, **pin(env)})
        return "fail_hop_27_to_26"
    obs, c26 = fight_spec(env, ROOM_26_SPEC, assist, n)
    ok26 = level5_room_26_cleared(env.get_ram())
    seams.append({"name": "0x26 west Gibdos", "ok": ok26, "ctl": c26.report(), **pin(env)})
    print("26", ok26, pin(env), flush=True)
    if not ok26:
        return "fail_26"
    hop25 = walk_west_from_26(env, assist, n)
    wait_play(env, assist, n, 0x25)
    if not (level5_in_room_25(env.get_ram()) or hop_ok(env, 0x25)):
        seams.append({"name": "0x26 west to 0x25 west Pols Voice", "ok": False, "hop": hop25, **pin(env)})
        return "fail_hop_26_to_25"
    obs, c25 = fight_spec(
        env, ROOM_25_SPEC, assist, n, controller=Level5PolsVoiceController(spec=ROOM_25_SPEC)
    )
    ok25 = level5_room_25_cleared(env.get_ram())
    seams.append({"name": "0x25 west Pols Voice", "ok": ok25, "ctl": c25.report(), **pin(env)})
    print("25", ok25, pin(env), flush=True)
    if not ok25:
        return "fail_25"
    hop24 = walk_west_from_25(env, assist, n)
    wait_play(env, assist, n, 0x24)
    door24 = level5_in_room_24(env.get_ram()) or hop_ok(env, 0x24)
    seams.append({"name": "0x24 Digdogger", "ok": door24, "hop": hop24, **pin(env)})
    print("24", door24, pin(env), flush=True)
    if not door24:
        return "fail_hop_25_to_24"
    return None


def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle_to_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    seams = []
    blocker = None
    boss = None
    start = None
    final = None
    obs = None
    tag = "l5_whistle_to_tf_stitch"
    env = make_env(GAME, "Level5Whistle", GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs = step(env, nes_idle_action(), assist, n)
        start = pin(env)
        print("START", start, flush=True)
        seams.append({"name": "0x04 Recorder / Whistle cellar", "ok": True, **start})

        leave = exit_whistle_04(env, assist, n)
        wait_play(env, assist, n, 0x05)
        ok05 = leave.get("success") or (hop_ok(env, 0x05) and read_snapshot(env.get_ram()).mode == PLAY_MODE)
        seams.append({"name": "0x05 six blue Darknuts + block stairs", "ok": ok05, "leave": {k: leave[k] for k in leave if k != "log"}, **pin(env)})
        print("05", ok05, leave.get("success"), pin(env), flush=True)
        if not ok05:
            blocker = "fail_leave_04_to_05"
        else:
            hop06 = walk_east_from_05(env, assist, n)
            wait_play(env, assist, n, 0x06)
            ok06 = hop06.get("success") or hop_ok(env, 0x06)
            seams.append({"name": "0x06 passage east of 0x05", "ok": ok06, "hop": hop06, **pin(env)})
            print("06", ok06, pin(env), flush=True)
            if not ok06:
                blocker = "fail_hop_05_to_06"
            else:
                stairs = take_block_stairs_06(env, assist, n)
                print("STAIRS06_block", stairs.get("success"), stairs.get("dest"), stairs.get("mode"), stairs.get("xy"), flush=True)
                if not stairs.get("success"):
                    stairs2 = take_center_stairs_06(env, assist, n)
                    print("STAIRS06_center", stairs2.get("success"), stairs2.get("dest"), stairs2.get("mode"), flush=True)
                    stairs = stairs2
                if not stairs.get("success"):
                    seams.append({"name": "0x06 stairs to cellar 0x07", "ok": False, "stairs": {k: stairs[k] for k in stairs if k != "log"}, **pin(env)})
                    blocker = "fail_06_stairs"
                else:
                    cellar = cellar_to_64(env, assist, n)
                    print("CELLAR", cellar.get("success"), cellar.get("dest"), flush=True)
                    ok64 = cellar.get("success") or hop_ok(env, 0x64)
                    if hop_ok(env, 0x64) and read_snapshot(env.get_ram()).mode != PLAY_MODE:
                        wait_play(env, assist, n, 0x64)
                        ok64 = hop_ok(env, 0x64)
                    seams.append({"name": "0x07 cellar from 0x64 stairs", "ok": True, **pin(env)})
                    seams.append({"name": "0x64 blue Darknuts / stairs", "ok": ok64, "cellar": {k: cellar[k] for k in cellar if k not in ("start", "log")}, **pin(env)})
                    print("64", ok64, pin(env), flush=True)
                    if not ok64:
                        blocker = "fail_cellar_to_64"
                    else:
                        hop65 = walk_east_from_64(env, assist, n)
                        wait_play(env, assist, n, 0x65)
                        ok65 = hop65.get("success") or hop_ok(env, 0x65)
                        seams.append({"name": "0x65 west Gibdos", "ok": ok65, "hop": hop65, **pin(env)})
                        print("65", ok65, pin(env), flush=True)
                        if not ok65:
                            blocker = "fail_hop_64_to_65"
                        else:
                            hop66 = exit_door(env, assist, n, "RIGHT")
                            wait_play(env, assist, n, 0x66)
                            ok66 = hop_ok(env, 0x66)
                            if not ok66:
                                hop66 = bomb_east_from_65(env, assist, n)
                                wait_play(env, assist, n, 0x66)
                                ok66 = hop66.get("success") or hop_ok(env, 0x66)
                            seams.append({"name": "0x66 3x Gibdo first key", "ok": ok66, "hop": hop66, **pin(env)})
                            print("66", ok66, pin(env), flush=True)
                            if not ok66:
                                blocker = "fail_hop_65_to_66"
                            else:
                                print("66_objs", live_objects(env), flush=True)
                                obs, c66 = fight_spec(env, ROOM_66_SPEC, assist, n)
                                ok66c = level5_room_66_cleared(env.get_ram()) or c66.success
                                print("66_clear", ok66c, c66.report().get("phase"), pin(env), flush=True)
                                seams.append({"name": "0x66 3x Gibdo first key (re-clear)", "ok": ok66c, "ctl": c66.report(), **pin(env)})
                                hop56 = walk_north_from_66(env, assist, n)
                                wait_play(env, assist, n, 0x56)
                                ok56 = hop56.get("success") or hop_ok(env, 0x56)
                                seams.append({"name": "0x56 north Dodongos", "ok": ok56, "hop": hop56, **pin(env)})
                                print("56", ok56, pin(env), flush=True)
                                if not ok56:
                                    blocker = "fail_hop_66_to_56"
                                else:
                                    blocker = from_56_to_24(env, assist, n, seams)
                                    if blocker is None:
                                        boss = fight_digdogger_and_tf(env, assist, n)
                                        print("BOSS", boss, flush=True)
                                        seams.append(
                                            {
                                                "name": "0x14 L5 triforce" if hop_ok(env, 0x14) else "0x24 Digdogger",
                                                "ok": bool(boss.get("tf_l5")),
                                                "boss": boss,
                                                **pin(env),
                                            }
                                        )
                                        if not boss.get("tf_l5"):
                                            blocker = "tf_bit_0x10_not_set"

        final = pin(env)
        shot = out / f"{tag}_final.png"
        if obs is not None:
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
    tf = final.get("triforce") if final else None
    tf_l5 = bool(final and final.get("tf_l5_bit"))
    report = {
        "ok": tf_l5 and blocker is None,
        "segment": tag,
        "continuous_emulator_session": True,
        "track": "assisted",
        "status_claim": False,
        "level5_complete_claim": False,
        "start_state": "Level5Whistle",
        "end_claim": "l5_triforce_bit_0x10" if tf_l5 else None,
        "whistle_0x065C": final.get("whistle") if final else None,
        "triforce_0x0671": tf,
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": None if boss is None else boss.get("digdogger_dead"),
        "boss": boss,
        "total_frames": n[0],
        "start": start,
        "final": final,
        "seams": seams,
        "room_sequence": [
            f"0x{s.get('screen'):02x} {s.get('name')}" for s in seams if s.get("screen") is not None
        ],
        "blocker": blocker,
        "bk2": bk2,
        "png": str(out / f"{tag}_final.png"),
        "pokes": False,
        "path_note": (
            "Whistle wing is a dead-end: 0x05 S/N/W=wall E=key. "
            "Leave 0x04 via x=176 down / x=48 up -> 0x05 east 0x06 stairs -> "
            "cellar 0x07 -> 0x64 -> 0x65 -> 0x66 -> 0x56 east 0x57 north 0x47 "
            "north 0x37 north 0x27 west 0x26 west 0x25 west 0x24. "
            "Did not claim Level5Complete unless TF bit 0x10 is real."
        ),
    }
    path = out / f"{tag}.json"
    write_json_report(path, report)
    print(
        f"wrote {path} frames={n[0]} bk2={bk2} tf={tf} tf_l5={tf_l5} "
        f"blocker={blocker} whistle={report['whistle_0x065C']}",
        flush=True,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
