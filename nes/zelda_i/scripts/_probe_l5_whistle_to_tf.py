"""Level5Whistle 0x04 exit -> L5 graph -> Digdogger 0x24 -> TF 0x14 bit 0x10.

Survival / infinite-life. No door pokes. Not Clean STATUS. Stay on L5.
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon import (
    DoorRoute,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.level3_dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
from zelda_i.level5_dungeon import (
    GIBDO_OBJECT_TYPE,
    KEESE_OBJECT_TYPE,
    POLS_VOICE_OBJECT_TYPE,
    ROOM_25_SPEC,
    ROOM_26_SPEC,
    ROOM_27_SPEC,
    ROOM_66_SPEC,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level5_path import (
    cellar_other_mouth,
    exit_whistle_04,
    select_b_item_menu,
    walk_axis as path_walk_axis,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.level9_stairs import (
    dest_report,
    on_stair_tile,
    stair_transition_modes,
    walk_to_step,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)

CELLAR = (9, 10, 11, 16)
SKIP_TYPES = {0, 0xFF, 0x40, 0x4E, 0x49, 0x55, 0x68, 0x5A}


def dump(env) -> dict:
    snap = read_snapshot(env.get_ram())
    body = w.dump_live(snap, env.get_ram())
    body["dest"] = dest_report(snap)
    return body


def live_combat(snap) -> list:
    if snap.mode != PLAY_MODE:
        return []
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in SKIP_TYPES and (o.hp > 0 or o.type_id == 0x1B)
    ]


def door_hop(env, assist, total, direction: str, expect: int, *, tx: int, ty: int) -> dict:
    room0 = read_snapshot(env.get_ram()).screen
    path_walk_axis(env, assist, total, "y", ty, max_f=400)
    path_walk_axis(env, assist, total, "x", tx, max_f=400)
    push_dir(env, assist, total, direction, frames=240)
    idle(env, assist, total, 12)
    w.wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 10)
    snap = read_snapshot(env.get_ram())
    return {
        "from": f"0x{room0:02x}",
        "dir": direction,
        "dest": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "ok": snap.screen == expect and snap.mode == PLAY_MODE,
        "whistle": int(read_u8(env.get_ram(), ADDR_WHISTLE)),
    }


def take_06_stairs(env, assist, total) -> dict:
    """Walk off center, approach from south, idle on stair tile. Do not hold a dir on-tile."""
    room0 = 0x06
    log = []

    def left() -> bool:
        snap = read_snapshot(env.get_ram())
        return stair_transition_modes(snap.mode) or snap.screen != room0

    # Step off the center so the trigger can fire on re-entry.
    path_walk_axis(env, assist, total, "x", 80, max_f=300)
    path_walk_axis(env, assist, total, "y", 173, max_f=300)
    idle(env, assist, total, 8)
    log.append({"phase": "off", **dump(env)})
    if left():
        return {"ok": True, "via": "off_already", "log": log, "end": dump(env)}

    path_walk_axis(env, assist, total, "x", 120, max_f=300)
    # Walk UP onto (120,141) from the south gap.
    for _ in range(220):
        snap = read_snapshot(env.get_ram())
        if left():
            break
        if snap.link_y <= 141 and abs(snap.link_x - 120) <= 2:
            break
        if abs(snap.link_x - 120) > 2:
            w.step(env, assist, total, nes_action("RIGHT" if snap.link_x < 120 else "LEFT"))
        else:
            w.step(env, assist, total, nes_action("UP"))
    idle(env, assist, total, 90)
    log.append({"phase": "south_on", **dump(env)})
    if left():
        w.wait_play(env, assist, total, max_f=280)
        idle(env, assist, total, 12)
        return {"ok": True, "via": "south_on", "log": log, "end": dump(env)}

    # Other approaches: off then walk onto the tile, then idle.
    approaches = (
        ((120, 109), "DOWN"),
        ((80, 141), "RIGHT"),
        ((160, 141), "LEFT"),
        ((120, 157), "UP"),
        ((96, 141), "RIGHT"),
        ((144, 141), "LEFT"),
    )
    for (ax, ay), toward in approaches:
        if left():
            break
        path_walk_axis(env, assist, total, "y", ay, max_f=280)
        path_walk_axis(env, assist, total, "x", ax, max_f=280)
        idle(env, assist, total, 6)
        for _ in range(80):
            snap = read_snapshot(env.get_ram())
            if left() or on_stair_tile(snap):
                break
            frame = walk_to_step(snap, 120, 141, y_first=True, tol=2)
            if frame.reason == "walk_arrived":
                break
            w.step(env, assist, total, frame.action)
        idle(env, assist, total, 80)
        rec = {"phase": f"from_{ax}_{ay}", **dump(env)}
        log.append(rec)
        print("STAIR06", rec.get("phase"), rec.get("room_hex"), rec.get("mode"), rec.get("xy"), "stair", rec.get("stair_tile"), flush=True)
        if left():
            break

    w.wait_play(env, assist, total, max_f=280)
    idle(env, assist, total, 12)
    snap = read_snapshot(env.get_ram())
    ok = stair_transition_modes(snap.mode) or snap.screen == 0x07
    return {"ok": ok, "via": "hunt", "log": log, "end": dump(env)}


def fight_spec(env, assist, total, spec) -> dict:
    ctl = GenericDungeonRoomController(spec)
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
                print(f"KILL 0x{spec.room_id:02x} n={last_n} f={ctl.frames}", flush=True)
        action = ctl.step(snap)
        w.step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = live_combat(snap) if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success),
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "spec": spec.spec_id,
        "room": f"0x{snap.screen:02x}",
    }


def fight_if_needed(env, assist, total, room: int) -> dict:
    snap = read_snapshot(env.get_ram())
    live = live_combat(snap)
    if not live:
        return {"ok": True, "skipped": True, "start_n": 0, "room": f"0x{room:02x}"}
    types = {o.type_id for o in live}
    print("FIGHT", f"0x{room:02x}", "n", len(live), "types", sorted(types), flush=True)
    if room == 0x27:
        spec = replace(ROOM_27_SPEC, source_room=0x37, max_frames=28000)
    elif room == 0x26:
        spec = replace(ROOM_26_SPEC, source_room=0x27, max_frames=28000)
    elif room == 0x25:
        spec = replace(ROOM_25_SPEC, source_room=0x26, max_frames=28000)
    elif GIBDO_OBJECT_TYPE in types and POLS_VOICE_OBJECT_TYPE not in types:
        spec = replace(
            ROOM_66_SPEC,
            spec_id=f"l5_{room:02x}_gibdo",
            source_room=room,
            room_id=room,
            expected_enemy_count=len([o for o in live if o.type_id == GIBDO_OBJECT_TYPE]),
            max_frames=28000,
        )
    elif POLS_VOICE_OBJECT_TYPE in types:
        spec = replace(
            ROOM_25_SPEC,
            spec_id=f"l5_{room:02x}_pols",
            source_room=room,
            room_id=room,
            expected_enemy_count=len([o for o in live if o.type_id == POLS_VOICE_OBJECT_TYPE]),
            max_frames=28000,
        )
    elif ZOL_OBJECT_TYPE in types or 0x14 in types:
        spec = replace(
            ROOM_5B_SPEC,
            spec_id=f"l5_{room:02x}_zol",
            source_room=room,
            room_id=room,
            enemy_types=(ZOL_OBJECT_TYPE, 0x14, 0x15),
            expected_enemy_count=len(live),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
            max_frames=20000,
            level=5,
        )
    else:
        spec = replace(
            ROOM_5B_SPEC,
            spec_id=f"l5_{room:02x}_mix",
            source_room=room,
            room_id=room,
            enemy_types=tuple(types),
            expected_enemy_count=len(live),
            required_open_doors=0,
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
            combat=ROOM_59_SPEC.combat,
            exit_routes=(DoorRoute("UP", ((120, 93),)),),
            max_frames=28000,
            level=5,
        )
    rec = fight_spec(env, assist, total, spec)
    idle(env, assist, total, 16)
    # Walk a key if it dropped.
    snap = read_snapshot(env.get_ram())
    if snap.room_item_id == 0x19:
        path_walk_axis(env, assist, total, "y", 141, max_f=240)
        path_walk_axis(env, assist, total, "x", 120, max_f=240)
        path_walk_axis(env, assist, total, "y", 157, max_f=200)
        path_walk_axis(env, assist, total, "x", 120, max_f=200)
        idle(env, assist, total, 10)
    return rec


def fail(tag, hops, checkpoints, env, reason, room, extra=None):
    snap_dump = dump(env)
    png = w.shot(env, assist_holder[0], total_holder[0], tag + "_fail") if assist_holder else None
    body = {
        "ok": False,
        "failed_room": room,
        "reason": reason,
        "hops": hops,
        "checkpoints": checkpoints,
        "now": snap_dump,
        "screenshot": png,
        "pokes": False,
        "status_claim": None,
        "whistle_0x065C": snap_dump.get("whistle_0x065C"),
        "level": snap_dump.get("level"),
    }
    if extra:
        body.update(extra)
    w.write_dump(tag, body)
    print("STOP", reason, "room", room, "pose", [snap_dump.get("x"), snap_dump.get("y")], "mode", snap_dump.get("mode"), "tile", snap_dump.get("colliding_tile"), flush=True)
    return body


assist_holder: list = []
total_holder: list = []


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    hops = []
    checkpoints = []
    env, assist, _ = w.open_env("Level5Whistle")
    assist_holder.append(assist)
    total = [1]
    total_holder.append(total)
    try:
        idle(env, assist, total, 16)
        start = dump(env)
        png0 = w.shot(env, assist, total, "l5_whistle_confirm")
        w.write_dump(
            "l5_whistle_confirm",
            {
                "ok": start.get("room") == 0x04 and start.get("whistle_0x065C") == 1,
                "dump": start,
                "screenshot": png0,
                "pokes": False,
                "status_claim": None,
            },
        )
        print(
            "START",
            start.get("room_hex"),
            "mode",
            start.get("mode"),
            "xy",
            [start.get("x"), start.get("y")],
            "whistle",
            start.get("whistle_0x065C"),
            "tile",
            start.get("colliding_tile"),
            "stair",
            start.get("stair_tile"),
            "cellar",
            start.get("cellar_mode"),
            flush=True,
        )
        if start.get("room") != 0x04 or start.get("whistle_0x065C") != 1:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "whistle_save_bad", "0x04", extra={"start": start})

        # 2. Exit cellar 0x04 -> floor 0x05. Dest must change.
        walk = exit_whistle_04(env, assist, total)
        hops.append({"hop": "0x04_exit", **{k: walk[k] for k in walk if k != "log"}})
        print("EXIT04", walk.get("success"), "dest", f"0x{walk.get('dest'):02x}", "xy", walk.get("xy"), "mode", walk.get("mode"), flush=True)
        w.shot(env, assist, total, "l5_04_exit_floor")
        if not walk.get("left_cellar") or walk.get("dest") == 0x04:
            return fail(
                "l5_whistle_to_tf",
                hops,
                checkpoints,
                env,
                "exit_04_dest_unchanged",
                "0x04",
                extra={"exit04": walk, "pose": dump(env)},
            )
        if walk.get("dest") != 0x05:
            return fail(
                "l5_whistle_to_tf",
                hops,
                checkpoints,
                env,
                "exit_04_not_0x05",
                f"0x{walk.get('dest'):02x}",
                extra={"exit04": walk},
            )
        checkpoints.append(
            w.save_ckpt(
                env,
                "Level5WhistleFloor",
                "Level5Whistle",
                {
                    "segment": "Level5WhistleFloor",
                    "via": "0x04 ladder176 pit189 left48 UP",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": 0x05, "whistle_0x065C": 1},
            )
        )

        # 0x05 E -> 0x06
        hop = door_hop(env, assist, total, "RIGHT", 0x06, tx=224, ty=141)
        hops.append({"hop": "0x05_east", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "east_not_0x06", "0x05")
        checkpoints.append(
            w.save_ckpt(
                env,
                "Level5Whistle06",
                "Level5Whistle",
                {
                    "segment": "Level5Whistle06",
                    "via": "0x05 east",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": 0x06, "whistle_0x065C": 1},
            )
        )

        # 0x06 stairs -> cellar 0x07
        stairs = take_06_stairs(env, assist, total)
        hops.append({"hop": "0x06_stairs", "ok": stairs["ok"], "via": stairs.get("via"), "end": stairs.get("end")})
        print("STAIRS06", stairs.get("ok"), stairs.get("end", {}).get("room_hex"), stairs.get("end", {}).get("mode"), flush=True)
        w.shot(env, assist, total, "l5_06_stairs")
        if not stairs["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "stairs_not_cellar", "0x06", extra={"stairs": {k: v for k, v in stairs.items() if k != "log"}})
        # If we landed in 0x04, leave again — wrong mouth.
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x04 or (snap.mode in CELLAR and snap.screen == 0x04):
            again = exit_whistle_04(env, assist, total)
            hops.append({"hop": "0x04_reexit", **{k: again[k] for k in again if k != "log"}})
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "06_stairs_went_04", "0x06")

        # 0x07 other mouth -> 0x64
        cellar = cellar_other_mouth(env, assist, total)
        hops.append({"hop": "0x07_to_64", **{k: cellar[k] for k in cellar if k != "start"}})
        print("CELLAR07", cellar.get("success"), "dest", f"0x{cellar.get('dest'):02x}", "side", cellar.get("chose_side"), flush=True)
        snap = read_snapshot(env.get_ram())
        if snap.screen != 0x64:
            # cellar_other_mouth expects 0x06; we want the opposite of spawn, which should be 0x64 if we came from 0x06.
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "left_mouth_not_0x64", "0x07", extra={"cellar": cellar})
        checkpoints.append(
            w.save_ckpt(
                env,
                "Level5Whistle64",
                "Level5Whistle",
                {
                    "segment": "Level5Whistle64",
                    "via": "0x06 stairs 0x07 left mouth",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": 0x64, "whistle_0x065C": 1},
            )
        )

        # 0x64 E -> 0x65 (bomb hole already open this visit)
        path_walk_axis(env, assist, total, "y", 93, max_f=300)
        path_walk_axis(env, assist, total, "x", 208, max_f=400)
        hop = door_hop(env, assist, total, "RIGHT", 0x65, tx=224, ty=141)
        hops.append({"hop": "0x64_east", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "east_not_0x65", "0x64")

        # 0x65 UP -> 0x55 (diamond: y=109 then x=120)
        path_walk_axis(env, assist, total, "y", 109, max_f=300)
        hop = door_hop(env, assist, total, "UP", 0x55, tx=120, ty=93)
        hops.append({"hop": "0x65_up", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "north_not_0x55", "0x65")
        checkpoints.append(
            w.save_ckpt(
                env,
                "Level5Whistle55",
                "Level5Whistle",
                {
                    "segment": "Level5Whistle55",
                    "via": "0x65 north shutter",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": 0x55, "whistle_0x065C": 1},
            )
        )

        # 0x55 R -> 0x56
        fight_if_needed(env, assist, total, 0x55)
        hop = door_hop(env, assist, total, "RIGHT", 0x56, tx=224, ty=141)
        hops.append({"hop": "0x55_right", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "east_not_0x56", "0x55")

        # 0x56 R -> 0x57
        fight_if_needed(env, assist, total, 0x56)
        hop = door_hop(env, assist, total, "RIGHT", 0x57, tx=224, ty=141)
        hops.append({"hop": "0x56_right", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "east_not_0x57", "0x56")

        # 0x57 UP -> 0x47
        f57 = fight_if_needed(env, assist, total, 0x57)
        hops.append({"hop": "fight_57", **f57})
        hop = door_hop(env, assist, total, "UP", 0x47, tx=120, ty=93)
        hops.append({"hop": "0x57_up", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "north_not_0x47", "0x57")

        # 0x47 UP -> 0x37 (C-block pinch x=128 — stay x=120)
        f47 = fight_if_needed(env, assist, total, 0x47)
        hops.append({"hop": "fight_47", **f47})
        path_walk_axis(env, assist, total, "x", 120, max_f=300)
        hop = door_hop(env, assist, total, "UP", 0x37, tx=120, ty=93)
        hops.append({"hop": "0x47_up", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "north_not_0x37", "0x47")

        # 0x37 UP -> 0x27 (pit: cross y=109, avoid x=56)
        f37 = fight_if_needed(env, assist, total, 0x37)
        hops.append({"hop": "fight_37", **f37})
        path_walk_axis(env, assist, total, "y", 109, max_f=300)
        path_walk_axis(env, assist, total, "x", 120, max_f=300)
        hop = door_hop(env, assist, total, "UP", 0x27, tx=120, ty=93)
        hops.append({"hop": "0x37_up", **hop})
        print("HOP", hops[-1], flush=True)
        if not hop["ok"]:
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "north_not_0x27", "0x37")

        # 0x27 W -> 0x26
        f27 = fight_if_needed(env, assist, total, 0x27)
        hops.append({"hop": "fight_27", **f27})
        west = walk_west_from_27(env, assist, total)
        hops.append({"hop": "0x27_west", **{k: west[k] for k in west if k != "log"}})
        print("HOP", hops[-1], flush=True)
        if not west.get("success"):
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "west_not_0x26", "0x27")

        # 0x26 W -> 0x25
        f26 = fight_if_needed(env, assist, total, 0x26)
        hops.append({"hop": "fight_26", **f26})
        west = walk_west_from_26(env, assist, total)
        hops.append({"hop": "0x26_west", **{k: west[k] for k in west if k != "log"}})
        print("HOP", hops[-1], flush=True)
        if not west.get("success"):
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "west_not_0x25", "0x26")

        # 0x25 W -> 0x24
        f25 = fight_if_needed(env, assist, total, 0x25)
        hops.append({"hop": "fight_25", **f25})
        west = walk_west_from_25(env, assist, total)
        hops.append({"hop": "0x25_west", **{k: west[k] for k in west if k != "log"}})
        print("HOP", hops[-1], flush=True)
        if not west.get("success"):
            return fail("l5_whistle_to_tf", hops, checkpoints, env, "west_not_0x24", "0x25")
        checkpoints.append(
            w.save_ckpt(
                env,
                "Level5Whistle24",
                "Level5Whistle",
                {
                    "segment": "Level5Whistle24",
                    "via": "graph walk from 0x04",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                {"success": True, "room": 0x24, "whistle_0x065C": 1},
            )
        )
        w.shot(env, assist, total, "l5_24_arrive")

        # 4. Whistle-shrink Digdogger, sword small, heart 0x1A, north 0x14, TF bit 0x10
        at24 = dump(env)
        print("AT24", at24.get("room_hex"), "objs", [(o["type_hex"], o["hp"]) for o in at24.get("objects") or []], flush=True)
        menu = select_b_item_menu(env, assist, total, 5)
        for _ in range(5):
            w.step(env, assist, total, nes_action("B"))
            idle(env, assist, total, 50)
        idle(env, assist, total, 80)
        after_b = dump(env)
        print("WHISTLE_B", menu, "objs", [(o["type_hex"], o["hp"]) for o in after_b.get("objects") or []], flush=True)
        leftovers = live_combat(read_snapshot(env.get_ram()))
        fight = None
        if leftovers:
            types = {o.type_id for o in leftovers}
            fight = w.fight_type(env, assist, total, 0x24, next(iter(types)), expected=len(leftovers))
            idle(env, assist, total, 16)
            print("BOSS", fight.get("ok"), "end_n", fight.get("end_n"), flush=True)
        # Heart 0x1A typically east of room / (224,141)
        path_walk_axis(env, assist, total, "y", 141, max_f=300)
        path_walk_axis(env, assist, total, "x", 224, max_f=400)
        idle(env, assist, total, 16)
        path_walk_axis(env, assist, total, "x", 120, max_f=300)
        path_walk_axis(env, assist, total, "y", 141, max_f=200)
        after_heart = dump(env)
        # North shutter -> 0x14
        hop = door_hop(env, assist, total, "UP", 0x14, tx=120, ty=93)
        hops.append({"hop": "0x24_north", **hop})
        print("HOP", hops[-1], flush=True)
        tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
        if hop["ok"] or read_snapshot(env.get_ram()).screen == 0x14:
            for tx, ty in ((120, 141), (120, 125), (120, 157), (96, 141), (144, 141), (120, 109)):
                path_walk_axis(env, assist, total, "y", ty, max_f=200)
                path_walk_axis(env, assist, total, "x", tx, max_f=200)
                idle(env, assist, total, 10)
                if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) > tf0:
                    break
        idle(env, assist, total, 20)
        final = dump(env)
        tf1 = int(final.get("triforce_0x0671") or 0)
        png = w.shot(env, assist, total, "l5_24_whistle_boss")
        tf_ok = bool(tf1 & 0x10)
        if tf_ok:
            checkpoints.append(
                w.save_ckpt(
                    env,
                    "Level5Complete",
                    "Level5Whistle",
                    {
                        "segment": "Level5Complete",
                        "via": "whistle Digdogger north TF 0x14",
                        "key_poke": False,
                        "door_poke": False,
                        "bomb_count_poke": False,
                        "selected_item_poke": False,
                    },
                    {"success": True, "room": int(final.get("room") or 0), "tf": tf1, "whistle_0x065C": 1},
                )
            )
        rec = {
            "ok": tf_ok,
            "pokes": False,
            "status_claim": None,
            "track": "assisted",
            "start": start,
            "hops": hops,
            "checkpoints": checkpoints,
            "exit_dest": "0x05",
            "whistle_0x065C": final.get("whistle_0x065C"),
            "at24": at24,
            "menu": menu,
            "after_whistle": after_b,
            "fight": fight,
            "after_heart": after_heart,
            "final": final,
            "tf_in": tf0,
            "tf_out": tf1,
            "tf_l5_bit": tf_ok,
            "screenshot": png,
            "failed_room": None if tf_ok else (hop.get("dest") if not hop["ok"] else "0x14"),
            "reason": None if tf_ok else "tf_bit_0x10_not_set",
        }
        w.write_dump("l5_whistle_to_tf", rec)
        w.write_dump("l5_24_whistle_boss", rec)
        print("FINAL", "tf", hex(tf1), "bit", tf_ok, "room", final.get("room_hex"), "whistle", final.get("whistle_0x065C"), flush=True)
        return rec
    finally:
        env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"))
    print("EXIT_DEST", r.get("exit_dest") or r.get("failed_room"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("TF", r.get("tf_l5_bit"), r.get("tf_out"))
    print("FAILED", r.get("failed_room"), r.get("reason"))
    print("CKPT", r.get("checkpoints"))
    print("HOPS", r.get("hops"))
    print("status_claim", None)
