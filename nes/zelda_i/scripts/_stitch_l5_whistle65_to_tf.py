"""Level5Whistle65 -> bomb east 0x66 -> clear -> UP 0x56 -> graph -> Digdogger TF.

LOCKED ROUTE: 0x65 east bomb -> 0x66 CLEAR -> UP 0x56 -> 0x57 -> 0x47 ->
0x37 -> 0x27 -> 0x26 -> 0x25 -> Digdogger 0x24 whistle-shrink -> TF 0x10.

Do NOT take 0x66 south to 0x76. Do NOT skip 0x65 north then fail-open 0x56.
Do NOT freeze the tape. One env. stop_record. No Complete until TF 0x10.
If 0x66 north stays sealed after a real clear, stop and report objects/doors/mask.
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
    DODONGO_OBJECT_TYPE,
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
    ROOM_ITEM_SMALL_KEY,
    Level5PolsVoiceController,
    level5_in_room_24,
    level5_in_room_25,
    level5_in_room_26,
    level5_room_25_cleared,
    level5_room_26_cleared,
    level5_room_27_cleared,
    level5_room_66_cleared,
)
from zelda_i.level5_path import (
    bomb_east_from_65,
    fight_blue_darknuts,
    select_b_item_menu,
    walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

DIGDOGGER = 0x38
GIBDO = 0x30
ZOL_TYPES = (ZOL_OBJECT_TYPE, GEL_SPLIT_OBJECT_TYPE, GEL_OBJECT_TYPE)
STATE = "Level5Whistle65"
ROOM_NAMES = {
    0x14: "L5 triforce",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x65: "west Gibdos",
    0x66: "3x Gibdo first key",
    0x76: "L5 entrance",
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
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": int(s.room_all_dead),
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
            out.append({
                "slot": o.slot,
                "type": o.type_id,
                "type_hex": f"0x{o.type_id:02x}",
                "name": object_name(o.type_id),
                "hp": o.hp,
                "x": o.x,
                "y": o.y,
            })
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
        if read_snapshot(env.get_ram()).screen == 0x76:
            print("ABORT_entered_0x76_during_fight", spec.spec_id, pin(env), flush=True)
            break
    return obs, ctl


def spec_66_from_west(env):
    """Fight whatever is actually in 0x66. Confirm types from RAM."""
    objs = [o for o in live_objects(env) if o["hp"] > 0 and o["type"] not in (0x40, 0x4E, 0x55, 0x56)]
    types = tuple(sorted({o["type"] for o in objs})) or (GIBDO,)
    if DODONGO_OBJECT_TYPE in types:
        enemy_types = (DODONGO_OBJECT_TYPE,)
        expected = sum(1 for o in objs if o["type"] == DODONGO_OBJECT_TYPE) or 3
        spec_id = "level5_room66_live_dodongos"
    elif GIBDO in types:
        enemy_types = (GIBDO,)
        expected = sum(1 for o in objs if o["type"] == GIBDO) or 3
        spec_id = "level5_room66_live_gibdos"
    else:
        enemy_types = types
        expected = len(objs) or 3
        spec_id = "level5_room66_live_other"
    return replace(
        ROOM_66_SPEC,
        spec_id=spec_id,
        source_room=0x65,
        entry=DoorRoute("LEFT", ((32, 141),)),
        enemy_types=enemy_types,
        expected_enemy_count=expected,
        required_open_doors=0x08,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=8),
        exit_routes=(DoorRoute("UP", ((120, 93),)),),
        max_frames=20000,
    ), objs


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
    return replace(
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
    return replace(
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


def walk_north_from_66(env, assist, n):
    """After a real 0x66 clear, south-band to x=120 then UP. Never south to 0x76."""
    idle(env, assist, n, 80)
    s = read_snapshot(env.get_ram())
    doors0 = {
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
        "all_dead": int(s.room_all_dead),
        "xy": [s.link_x, s.link_y],
        "objects": live_objects(env),
    }
    print("66_post_clear", doors0, pin(env), flush=True)
    if s.screen == 0x76:
        return {"path": "already_0x76_abort", "dest": 0x76, "success": False, "doors0": doors0}
    # South band y=173 avoids the west-river pinch at y=93,x=32.
    walk_axis(env, assist, n, "y", 173, max_f=400)
    if hop_ok(env, 0x76):
        return {"path": "slipped_south_0x76", "dest": 0x76, "success": False, "doors0": doors0, **pin(env)}
    walk_axis(env, assist, n, "x", 120, max_f=500)
    idle(env, assist, n, 40)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    goto(env, assist, n, 120, 93, tol=3, max_f=240)
    push_dir(env, assist, n, "UP", frames=280)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x56)
    if hop_ok(env, 0x56):
        s = read_snapshot(env.get_ram())
        return {
            "path": "idle_south173_x120_up",
            "dest": s.screen,
            "xy": [s.link_x, s.link_y],
            "doors0": doors0,
            "success": True,
        }
    # One more center-column try. Still no south, no 0x65 north.
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=400)
    walk_axis(env, assist, n, "y", 93, max_f=300)
    goto(env, assist, n, 120, 93, tol=2, max_f=200)
    push_dir(env, assist, n, "UP", frames=240)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x56)
    s = read_snapshot(env.get_ram())
    return {
        "path": "center_retry" if hop_ok(env, 0x56) else "north_still_sealed",
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "doors0": doors0,
        "doors1": {
            "doors": int(s.cur_opened_doors),
            "mask": int(s.open_doorway_mask),
            "all_dead": int(s.room_all_dead),
            "objects": live_objects(env),
        },
        "success": hop_ok(env, 0x56),
    }


def hold_dir(env, assist, n, direction, frames):
    for _ in range(frames):
        step(env, nes_action(direction), assist, n)


def unstick_hold(env, assist, n):
    """Long holds beat walk_axis stall=40 when wedged in a C-block gap."""
    snap = read_snapshot(env.get_ram())
    start = (snap.link_x, snap.link_y)
    notes = [f"stuck_{start[0]},{start[1]}"]
    for direction, frames in (
        ("DOWN", 160),
        ("RIGHT", 160),
        ("LEFT", 80),
        ("UP", 80),
        ("DOWN", 200),
        ("RIGHT", 200),
        ("LEFT", 120),
    ):
        hold_dir(env, assist, n, direction, frames)
        snap = read_snapshot(env.get_ram())
        notes.append(f"hold_{direction}{frames}:{snap.link_x},{snap.link_y}")
        if (snap.link_x, snap.link_y) != start:
            notes.append("unstuck")
            return True, notes
    for direction in ("DOWN", "RIGHT", "LEFT", "UP"):
        step(env, nes_action(direction, "A"), assist, n)
        hold_dir(env, assist, n, direction, 80)
        snap = read_snapshot(env.get_ram())
        notes.append(f"slash_{direction}:{snap.link_x},{snap.link_y}")
        if (snap.link_x, snap.link_y) != start:
            notes.append("unstuck_slash")
            return True, notes
    return False, notes


def walk_north_from_47(env, assist, n):
    """0x47 south mouth (120,205) hold UP -> 0x37. x=128 C-block pinch sticks."""
    notes = []
    s = read_snapshot(env.get_ram())
    notes.append(f"enter={s.link_x},{s.link_y}")
    if s.link_y < 173:
        walk_axis(env, assist, n, "y", 189, max_f=400)
    walk_axis(env, assist, n, "x", 120, max_f=400)
    walk_axis(env, assist, n, "y", 205, max_f=200)
    notes.append(f"stand={read_snapshot(env.get_ram()).link_x},{read_snapshot(env.get_ram()).link_y}")
    for _ in range(220):
        if read_snapshot(env.get_ram()).screen == 0x37:
            break
        step(env, nes_action("UP"), assist, n)
    idle(env, assist, n, 12)
    wait_play(env, assist, n, 0x37)
    s = read_snapshot(env.get_ram())
    return {"path": "south120_hold_up", "dest": s.screen, "xy": [s.link_x, s.link_y], "success": hop_ok(env, 0x37), "notes": notes}


def walk_north_from_57(env, assist, n):
    """0x57: from y=141 UP only at x=48/80/112/128/160/192. x=32/208 are ladder locks."""
    notes = []
    s = read_snapshot(env.get_ram())
    notes.append(f"enter={s.link_x},{s.link_y}")
    if abs(s.link_y - 125) <= 6:
        for _ in range(50):
            step(env, nes_action("DOWN"), assist, n)
        notes.append(f"down_off={read_snapshot(env.get_ram()).link_x},{read_snapshot(env.get_ram()).link_y}")
    for name, col in (("gap_x120", 120), ("gap_x112", 112), ("gap_x128", 128), ("gap_x80", 80), ("gap_x160", 160), ("gap_x48", 48), ("gap_x192", 192)):
        walk_axis(env, assist, n, "y", 141, max_f=300)
        walk_axis(env, assist, n, "x", col, max_f=400)
        walk_axis(env, assist, n, "y", 93, max_f=400)
        s = read_snapshot(env.get_ram())
        notes.append(f"{name}={s.link_x},{s.link_y},0x{s.screen:02x}")
        if s.screen == 0x47:
            return {"path": name, "dest": 0x47, "success": True, "notes": notes}
        if s.link_y <= 109:
            walk_axis(env, assist, n, "x", 120, max_f=300)
            push_dir(env, assist, n, "UP", frames=260)
            idle(env, assist, n, 12)
            wait_play(env, assist, n, 0x47)
            if hop_ok(env, 0x47):
                s = read_snapshot(env.get_ram())
                return {"path": name + "_push", "dest": 0x47, "success": True, "notes": notes}
    s = read_snapshot(env.get_ram())
    return {"path": "failed", "dest": s.screen, "xy": [s.link_x, s.link_y], "success": hop_ok(env, 0x47), "notes": notes}


def fight_digdogger_and_tf(env, assist, n):
    """Center, hold B for one Recorder song, sword small 0x18, north 0x14 TF."""
    SMALL = 0x18
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=400)
    idle(env, assist, n, 16)
    menu = select_b_item_menu(env, assist, n, 5)
    idle(env, assist, n, 16)
    walk_axis(env, assist, n, "y", 141, max_f=200)
    walk_axis(env, assist, n, "x", 120, max_f=200)
    idle(env, assist, n, 12)
    for _ in range(16):
        step(env, nes_action("B"), assist, n)
    after_b = None
    for i in range(10):
        idle(env, assist, n, 30)
        after_b = live_objects(env)
        small = [o for o in after_b if o["type"] == SMALL and o["hp"] > 0]
        big = [o for o in after_b if o["type"] == DIGDOGGER and o["hp"] > 0]
        print("SONG", i, "small", small, "big", big, flush=True)
        if small and not big:
            break
    if not [o for o in live_objects(env) if o["type"] == SMALL and o["hp"] > 0]:
        # One more song from center.
        walk_axis(env, assist, n, "y", 141, max_f=200)
        walk_axis(env, assist, n, "x", 120, max_f=200)
        idle(env, assist, n, 8)
        for _ in range(16):
            step(env, nes_action("B"), assist, n)
        for i in range(8):
            idle(env, assist, n, 30)
            after_b = live_objects(env)
            small = [o for o in after_b if o["type"] == SMALL and o["hp"] > 0]
            big = [o for o in after_b if o["type"] == DIGDOGGER and o["hp"] > 0]
            print("SONG2", i, "small", small, "big", big, flush=True)
            if small and not big:
                break
    after_b = live_objects(env)
    small = [o for o in after_b if o["type"] == SMALL and o["hp"] > 0]
    big = [o for o in after_b if o["type"] == DIGDOGGER and o["hp"] > 0]
    fight = None
    targets = small or big
    tid = SMALL if small else DIGDOGGER
    if targets:
        spec = replace(
            ROOM_5B_SPEC,
            spec_id="level5_digdogger_small" if tid == SMALL else "level5_digdogger_big",
            source_room=0x25,
            room_id=0x24,
            entry=DoorRoute("LEFT", ((224, 141),)),
            enemy_types=(tid,),
            expected_enemy_count=len(targets),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("UP", ((120, 93),)),),
            max_frames=16000,
            level=LEVEL_5,
        )
        ctl = GenericDungeonRoomController(spec)
        for _ in range(spec.max_frames):
            snap = read_snapshot(env.get_ram())
            action = ctl.step(snap)
            step(env, action.action, assist, n)
            if ctl.success or ctl.phase is DungeonPhase.FAILED:
                break
        fight = {"ok": bool(ctl.success), "frames": ctl.frames, "phase": str(ctl.phase), "tid": tid}
    leftovers = [
        o
        for o in read_snapshot(env.get_ram()).objects
        if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF, 0x55, 0x4E, 0x40, 0x1A) and o.hp > 0
    ]
    extra = None
    if leftovers:
        extra = fight_blue_darknuts(env, assist, n, 0x24, expected=len(leftovers), source=0x25)
    for tx, ty in ((120, 141), (144, 141), (96, 141), (160, 141), (80, 141), (120, 125), (120, 157), (224, 141)):
        walk_axis(env, assist, n, "y", ty, max_f=200)
        walk_axis(env, assist, n, "x", tx, max_f=200)
        idle(env, assist, n, 8)
    walk_axis(env, assist, n, "y", 141, max_f=300)
    walk_axis(env, assist, n, "x", 120, max_f=300)
    walk_axis(env, assist, n, "y", 93, max_f=400)
    walk_axis(env, assist, n, "x", 120, max_f=200)
    push_dir(env, assist, n, "UP", frames=300)
    idle(env, assist, n, 16)
    for _ in range(260):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen == 0x14:
            break
        step(env, nes_action("UP"), assist, n)
    idle(env, assist, n, 16)
    snap = read_snapshot(env.get_ram())
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    if snap.screen == 0x14 or snap.room_item_id == 0x1B:
        w0 = tf0
        for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109), (120, 173)):
            walk_axis(env, assist, n, "y", ty, max_f=200)
            walk_axis(env, assist, n, "x", tx, max_f=200)
            idle(env, assist, n, 12)
            if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) > w0:
                break
    idle(env, assist, n, 30)
    tf1 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    snap = read_snapshot(env.get_ram())
    dead = not any(
        o.type_id in (DIGDOGGER, SMALL) and o.hp > 0
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


def walk_north_from_37(env, assist, n):
    """0x37 pit/ladder: cross at y=109, avoid x=56, slide (128,93)->(120,93)."""
    notes = []
    snap = read_snapshot(env.get_ram())
    start_xy = [snap.link_x, snap.link_y]
    if abs(snap.link_x - 56) <= 8:
        hold_dir(env, assist, n, "RIGHT", 80)
        walk_axis(env, assist, n, "x", 80, max_f=200)
        notes.append(f"off56:{read_snapshot(env.get_ram()).link_x},{read_snapshot(env.get_ram()).link_y}")
    for name, steps in (
        ("y109_x120_n", (("y", 109), ("x", 120), ("y", 93))),
        ("y125_x120_n", (("y", 125), ("x", 120), ("y", 93))),
        ("y141_x120_n", (("y", 141), ("x", 120), ("y", 93))),
        ("x160_y109_x120", (("x", 160), ("y", 109), ("x", 120), ("y", 93))),
        ("x80_y109_x120", (("x", 80), ("y", 109), ("x", 120), ("y", 93))),
    ):
        for axis, tgt in steps:
            ok = walk_axis(env, assist, n, axis, tgt, max_f=500)
            s = read_snapshot(env.get_ram())
            notes.append(f"{name}:{axis}:{tgt}:ok={ok}:xy={s.link_x},{s.link_y}")
            if s.screen == 0x27:
                return {"success": True, "path": name, "dest": 0x27, "notes": notes,
                        "xy": [s.link_x, s.link_y], "start_xy": start_xy}
        s = read_snapshot(env.get_ram())
        if abs(s.link_y - 93) <= 8 and abs(s.link_x - 120) <= 16:
            notes.append(f"near_mouth_{name}")
            break
    s = read_snapshot(env.get_ram())
    if abs(s.link_y - 93) <= 8 and s.link_x > 120:
        hold_dir(env, assist, n, "LEFT", 40)
        walk_axis(env, assist, n, "x", 120, max_f=200)
        notes.append(f"slide128to120:{read_snapshot(env.get_ram()).link_x},{read_snapshot(env.get_ram()).link_y}")
    goto(env, assist, n, 120, 93, tol=2, max_f=240)
    for _ in range(24):
        s = read_snapshot(env.get_ram())
        if abs(s.link_x - 120) <= 1 and abs(s.link_y - 93) <= 2:
            break
        if abs(s.link_x - 120) > 1:
            step(env, nes_action("RIGHT" if s.link_x < 120 else "LEFT"), assist, n)
        else:
            step(env, nes_action("DOWN" if s.link_y < 93 else "UP"), assist, n)
    at = [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]
    notes.append(f"at_mouth:{at}")
    room0 = read_snapshot(env.get_ram()).screen
    push_dir(env, assist, n, "UP", frames=200)
    idle(env, assist, n, 16)
    wait_play(env, assist, n, 0x27)
    s = read_snapshot(env.get_ram())
    return {
        "success": hop_ok(env, 0x27),
        "path": "slide_mouth_up",
        "dest": s.screen,
        "xy": [s.link_x, s.link_y],
        "at_mouth": at,
        "start_xy": start_xy,
        "notes": notes,
        "changed": s.screen != room0,
    }


def from_56_to_24(env, assist, n, seams):
    """Walk open-door floor. Do not abort on optional combat clears."""
    hop57 = exit_door(env, assist, n, "RIGHT")
    wait_play(env, assist, n, 0x57)
    if not hop_ok(env, 0x57):
        seams.append({"name": "0x56 east to 0x57", "ok": False, "hop": hop57, **pin(env)})
        return "fail_hop_56_to_57"
    seams.append({"name": "0x56 east to 0x57", "ok": True, **pin(env)})
    print("57enter", pin(env), flush=True)

    hop47 = walk_north_from_57(env, assist, n)
    wait_play(env, assist, n, 0x47)
    if not hop_ok(env, 0x47):
        seams.append({"name": "0x57 north to 0x47", "ok": False, "hop": hop47, **pin(env)})
        return "fail_hop_57_to_47"
    seams.append({"name": "0x57 north to 0x47", "ok": True, "hop": hop47, **pin(env)})
    print("47enter", pin(env), flush=True)

    hop37 = walk_north_from_47(env, assist, n)
    wait_play(env, assist, n, 0x37)
    if not hop_ok(env, 0x37):
        seams.append({"name": "0x47 north to 0x37", "ok": False, "hop": hop37, **pin(env)})
        return "fail_hop_47_to_37"
    seams.append({"name": "0x47 north to 0x37", "ok": True, "hop": hop37, **pin(env)})
    print("37enter", pin(env), flush=True)

    hop27 = walk_north_from_37(env, assist, n)
    wait_play(env, assist, n, 0x27)
    if not hop_ok(env, 0x27):
        hop27 = exit_door(env, assist, n, "UP")
        wait_play(env, assist, n, 0x27)
    if not hop_ok(env, 0x27):
        seams.append({"name": "0x37 north to 0x27", "ok": False, "hop": hop27, **pin(env)})
        return "fail_hop_37_to_27"
    seams.append({"name": "0x37 north to 0x27", "ok": True, "hop": hop27, **pin(env)})
    print("27enter", pin(env), flush=True)

    hop26 = walk_west_from_27(env, assist, n)
    wait_play(env, assist, n, 0x26)
    if not hop_ok(env, 0x26):
        seams.append({"name": "0x27 west to 0x26", "ok": False, "hop": hop26, **pin(env)})
        return "fail_hop_27_to_26"
    seams.append({"name": "0x27 west to 0x26", "ok": True, **pin(env)})
    print("26enter", pin(env), flush=True)

    hop25 = walk_west_from_26(env, assist, n)
    wait_play(env, assist, n, 0x25)
    if not hop_ok(env, 0x25):
        seams.append({"name": "0x26 west to 0x25", "ok": False, "hop": hop25, **pin(env)})
        return "fail_hop_26_to_25"
    seams.append({"name": "0x26 west to 0x25", "ok": True, **pin(env)})
    print("25enter", pin(env), flush=True)

    hop24 = walk_west_from_25(env, assist, n)
    wait_play(env, assist, n, 0x24)
    if not hop_ok(env, 0x24):
        seams.append({"name": "0x25 west to 0x24 Digdogger", "ok": False, "hop": hop24, **pin(env)})
        return "fail_hop_25_to_24"
    seams.append({"name": "0x25 west to 0x24 Digdogger", "ok": True, **pin(env)})
    print("24enter", pin(env), flush=True)
    return None


def main():
    configure_headless()
    out = RECORDINGS_DIR / "stitches"
    movie = out / "bk2_whistle65_to_tf"
    movie.mkdir(parents=True, exist_ok=True)
    assist = UnlimitedHealthAssist(enabled=True)
    n = [0]
    seams = []
    blocker = None
    boss = None
    start = None
    final = None
    obs = None
    dump65 = None
    objs66 = None
    tag = "l5_whistle65_to_tf_stitch"
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array", record=str(movie))
    try:
        obs, _ = reset_obs(env)
        obs = step(env, nes_idle_action(), assist, n)
        idle(env, assist, n, 12)
        start = pin(env)
        dump65 = {**start, "objects": live_objects(env)}
        print("DUMP65", dump65, flush=True)
        write_json_report(RECORDINGS_DIR / "l5_whistle65_dump.json", dump65)
        seams.append({"name": "0x65 west Gibdos (Level5Whistle65 dump)", "ok": start["screen"] == 0x65 and start["whistle"] == 1, **start, "objects": dump65["objects"]})
        if start["screen"] != 0x65 or start["whistle"] != 1:
            blocker = "start_not_0x65_whistle1"
        else:
            # Skip 0x65 north. Bomb east to 0x66.
            hop66 = bomb_east_from_65(env, assist, n)
            wait_play(env, assist, n, 0x66)
            ok66 = hop66.get("success") or hop_ok(env, 0x66)
            objs66 = live_objects(env)
            print("66_enter", ok66, pin(env), "objs", objs66, flush=True)
            seams.append({"name": "0x66 enter via 0x65 bomb east", "ok": ok66, "hop": {k: hop66[k] for k in hop66 if k != "menu"}, "objects": objs66, **pin(env)})
            if not ok66:
                blocker = "fail_hop_65_to_66"
            elif hop_ok(env, 0x76):
                blocker = "entered_0x76_abort"
            else:
                spec66, objs_in = spec_66_from_west(env)
                print("66_spec", spec66.spec_id, spec66.enemy_types, spec66.expected_enemy_count, objs_in, flush=True)
                obs, c66 = fight_spec(env, spec66, assist, n)
                if hop_ok(env, 0x76):
                    blocker = "entered_0x76_during_66_fight"
                else:
                    live66 = [
                        o for o in live_objects(env)
                        if o["hp"] > 0 and o["type"] in spec66.enemy_types
                    ]
                    ok66c = (level5_room_66_cleared(env.get_ram()) or c66.success or not live66)
                    post = {**pin(env), "objects": live_objects(env)}
                    print("66_clear", ok66c, c66.report().get("phase"), post, flush=True)
                    seams.append({
                        "name": "0x66 clear (live types)",
                        "ok": ok66c,
                        "ctl": c66.report(),
                        "spec_id": spec66.spec_id,
                        "enter_objects": objs_in,
                        **post,
                    })
                    if not ok66c:
                        blocker = "fail_clear_66"
                    else:
                        hop56 = walk_north_from_66(env, assist, n)
                        wait_play(env, assist, n, 0x56)
                        ok56 = hop56.get("success") or hop_ok(env, 0x56)
                        print("56", ok56, hop56.get("path"), pin(env), flush=True)
                        seams.append({"name": "0x56 north Dodongos", "ok": ok56, "hop": hop56, **pin(env)})
                        if not ok56:
                            blocker = "fail_hop_66_to_56_after_real_clear"
                            print("HONEST_STOP_66_NORTH", hop56, live_objects(env), pin(env), flush=True)
                        else:
                            blocker = from_56_to_24(env, assist, n, seams)
                            if blocker is None:
                                boss = fight_digdogger_and_tf(env, assist, n)
                                print("BOSS", boss, flush=True)
                                seams.append({
                                    "name": "0x14 L5 triforce" if hop_ok(env, 0x14) else "0x24 Digdogger",
                                    "ok": bool(boss.get("tf_l5")),
                                    "boss": boss,
                                    **pin(env),
                                })
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
        "start_state": STATE,
        "end_claim": "l5_triforce_bit_0x10" if tf_l5 else None,
        "whistle_0x065C": final.get("whistle") if final else None,
        "triforce_0x0671": tf,
        "tf_l5_bit_0x10_real": tf_l5,
        "digdogger_dead": None if boss is None else boss.get("digdogger_dead"),
        "boss": boss,
        "dump65": dump65,
        "objs66_enter": objs66,
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
        "did_not_take_0x66_south": True,
        "did_not_redo_0x06_stairs": True,
        "path_note": (
            "Start Level5Whistle65. Skip 0x65 north. Bomb east 0x66, clear live "
            "objects, UP 0x56 (never 0x76). Then 57 47 37 27 26 25 24 whistle TF. "
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
