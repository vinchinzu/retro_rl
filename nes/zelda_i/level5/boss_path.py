"""Level 5 Recorder cellar → Digdogger 0x24 → TF bit 0x10.

Extracted from ``scripts/run_level5_whistle_tf``. Spine leftover is cellar
``0x04`` mode 9 ``(135,141)``. Coordinate waypoints; no BFS; no door/key
pokes; do not grant the Whistle.

    0x04 exit (ladder x=176 DOWN, pit y=189, mouth x=48 UP) → play 0x05
    east 0x06 stairs → cellar 0x07 → 0x64 east 0x65
    0x65 Gibdo clear; north shutter else bomb-east 0x66 → 0x56 → 0x57
    skip combat through 0x47/0x37/0x27/0x26/0x25
    west key 0x24, whistle-shrink 0x38→0x18, sword, north TF 0x14
"""

from __future__ import annotations

from dataclasses import replace

from retro_harness.nes import nes_action, nes_idle_action

from zelda_i.anchors import LEVEL5_TF_ROOM, TF_BIT_L5
from zelda_i.dungeon.behaviors import DIGDOGGER_SHRUNK_TYPE, DIGDOGGER_TYPE
from zelda_i.dungeon.engine import (
    DoorRoute,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon.ops import exit_door, idle, push_dir
from zelda_i.level3.dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
from zelda_i.level5.dungeon import (
    GIBDO_OBJECT_TYPE,
    LEVEL_5,
    ROOM_65_SPEC,
    ROOM_66_SPEC,
    ZOL_OBJECT_TYPE,
)
from zelda_i.level5.path import (
    CELLAR_MODES,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
    _step,
    bomb_east_from_65,
    cellar_to_64,
    exit_whistle_04,
    select_b_item_menu,
    take_block_stairs_06,
    walk_axis,
    walk_east_from_05,
    walk_east_from_64,
    walk_west_from_25,
    walk_west_from_26,
    walk_west_from_27,
)
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

HEART_CONTAINER = 0x1A
WHISTLE_B_SLOT = 5
# Doorway (224,141) + short B taps do not shrink; fireballs interrupt song.
WHISTLE_STAND = (120, 141)
SKIP_TYPES = (0, 0xFF, 0x55, 0x4E, 0x40, 0x68)

_DROP = frozenset(
    {"log", "before", "at_door", "after", "progress", "menu", "start", "reused"}
)

STOP_EXIT04 = "exit04"
STOP_ROOM24 = "room24"
STOP_TRIFORCE = "triforce"
TF_SUFFIX_STOPS = (STOP_EXIT04, STOP_ROOM24, STOP_TRIFORCE)


def _slim(rec: dict) -> dict:
    return {k: rec[k] for k in rec if k not in _DROP}


def _append(hops: list[dict], name: str, rec: dict) -> dict:
    hops.append({"hop": name, **_slim(rec)})
    return rec


def live_types(snap, types) -> list:
    return [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in types and o.hp > 0
    ]


def wait_play(env, assist, total: list[int], max_f: int = 240) -> None:
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and not snap.transitioning:
            return
        _step(env, assist, total, nes_idle_action())
    idle(env, assist, total, 8)


def door(env, assist, total: list[int], direction: str, expect: int, **kw) -> dict:
    rec = exit_door(env, assist, total, direction, **kw)
    wait_play(env, assist, total)
    snap = read_snapshot(env.get_ram())
    ok = snap.level == LEVEL_5 and snap.screen == expect and snap.mode == PLAY_MODE
    rec["ok"] = ok
    rec["dest"] = snap.screen
    rec["xy"] = [snap.link_x, snap.link_y]
    rec["mode"] = snap.mode
    return rec


def fight_ctl(env, assist, total: list[int], spec, controller_cls=GenericDungeonRoomController) -> dict:
    ctl = controller_cls(spec)
    for _ in range(spec.max_frames):
        snap = read_snapshot(env.get_ram())
        action = ctl.step(snap)
        _step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = read_snapshot(env.get_ram())
    live = (
        spec.live_enemies(snap)
        if snap.mode == PLAY_MODE and snap.screen == spec.room_id
        else []
    )
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "end_n": len(live),
        "spec": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "phase": str(ctl.phase),
    }


def grab_item(env, assist, total: list[int], pred, stands, max_each: int = 220) -> bool:
    if pred(env):
        return True
    for tx, ty in stands:
        walk_axis(env, assist, total, "y", ty, max_f=max_each)
        walk_axis(env, assist, total, "x", tx, max_f=max_each)
        idle(env, assist, total, 8)
        if pred(env):
            return True
    return pred(env)


def north_pinch(env, assist, total: list[int], expect: int) -> dict:
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


def take_stairs_06(env, assist, total: list[int]) -> dict:
    """0x06: push 0x68 north, RIGHT onto (128,141). Center idle at (120,141) never warps."""
    reused = take_block_stairs_06(env, assist, total)
    snap = read_snapshot(env.get_ram())
    ok = reused.get("success") or snap.mode in CELLAR_MODES or snap.screen == ROOM_L5_CELLAR_07
    return {
        "ok": bool(ok),
        "via": reused.get("path"),
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "tile": int(snap.colliding_tile),
    }


def _in_whistle_cellar(snap) -> bool:
    return snap.level == LEVEL_5 and snap.screen == ROOM_L5_WHISTLE_ITEM


def _fight_if_live(env, assist, total, hops, spec, types, name: str) -> bool:
    snap = read_snapshot(env.get_ram())
    n = len(live_types(snap, types))
    if not n:
        return True
    spec = replace(spec, expected_enemy_count=n, required_open_doors=0)
    fight = fight_ctl(env, assist, total, spec)
    _append(hops, name, fight)
    return bool(fight.get("ok"))


def _walk_west(env, assist, total, hops, walker, expect: int, name: str) -> bool:
    west = walker(env, assist, total)
    wait_play(env, assist, total, max_f=180)
    snap = read_snapshot(env.get_ram())
    west["dest"] = snap.screen
    west["mode"] = snap.mode
    west["success"] = snap.screen == expect and snap.mode == PLAY_MODE
    _append(hops, name, west)
    return bool(west.get("success"))


def path_exit_whistle_04(env, assist, total: list[int], hops: list[dict]) -> dict:
    """Leave leftover cellar 0x04 (135,141) mode 9 onto play 0x05."""
    snap = read_snapshot(env.get_ram())
    if snap.mode == PLAY_MODE and snap.screen == ROOM_L5_WHISTLE_05:
        hops.append({"hop": "exit_whistle_04", "ok": True, "via": "already_05"})
        return {"ok": True, "already": True, "dest": snap.screen}
    if not _in_whistle_cellar(snap):
        hops.append(
            {
                "hop": "exit_whistle_04",
                "ok": True,
                "via": f"skipped_0x{snap.screen:02x}_m{snap.mode}",
            }
        )
        return {"ok": True, "skipped": True, "dest": snap.screen}
    walk = exit_whistle_04(env, assist, total)
    rec = _append(hops, "exit_whistle_04", walk)
    rec["ok"] = bool(walk.get("success"))
    hops[-1]["ok"] = rec["ok"]
    return rec


def path_05_to_24(env, assist, total: list[int], hops: list[dict]) -> dict:
    """Play 0x05 → Digdogger 0x24. Skip combat after 0x65 until the boss."""
    snap = read_snapshot(env.get_ram())
    room = snap.screen

    if room == ROOM_L5_WHISTLE_05:
        rec = walk_east_from_05(env, assist, total)
        wait_play(env, assist, total)
        snap = read_snapshot(env.get_ram())
        rec["dest"] = snap.screen
        rec["success"] = snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE
        _append(hops, "05_east", rec)
        if not rec.get("success"):
            return {"ok": False, "failed": "east_did_not_enter_06", "room": 0x05}
        room = ROOM_L5_PASSAGE_06

    if room == ROOM_L5_PASSAGE_06:
        stairs = take_stairs_06(env, assist, total)
        _append(hops, "06_stairs", stairs)
        if not stairs.get("ok"):
            return {"ok": False, "failed": "stairs_not_taken", "room": 0x06}
        room = read_snapshot(env.get_ram()).screen

    snap = read_snapshot(env.get_ram())
    if snap.mode in CELLAR_MODES or snap.screen == ROOM_L5_CELLAR_07:
        cellar = cellar_to_64(env, assist, total)
        _append(hops, "07_left_mouth", cellar)
        snap = read_snapshot(env.get_ram())
        if snap.screen == ROOM_L5_PASSAGE_06:
            stairs = take_stairs_06(env, assist, total)
            _append(hops, "06_stairs_retry", stairs)
            snap = read_snapshot(env.get_ram())
            if snap.mode in CELLAR_MODES or snap.screen == ROOM_L5_CELLAR_07:
                cellar = cellar_to_64(env, assist, total)
                _append(hops, "07_left_mouth_retry", cellar)
                wait_play(env, assist, total)
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_L5_BLUE_64 or snap.mode != PLAY_MODE:
            return {
                "ok": False,
                "failed": "cellar_did_not_enter_64",
                "room": snap.screen,
            }
        room = ROOM_L5_BLUE_64

    if room == ROOM_L5_BLUE_64:
        rec = walk_east_from_64(env, assist, total)
        _append(hops, "64_east", rec)
        if not rec.get("success"):
            return {"ok": False, "failed": "east_did_not_enter_65", "room": 0x64}
        room = 0x65

    if room == 0x65:
        spec = replace(
            ROOM_65_SPEC,
            spec_id="level5_whistle_65_gibdos",
            source_room=0x64,
            room_id=0x65,
            entry=DoorRoute("RIGHT", ((32, 141),)),
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
            max_frames=28000,
            level=LEVEL_5,
        )
        if not _fight_if_live(
            env, assist, total, hops, spec, (GIBDO_OBJECT_TYPE,), "fight_65"
        ):
            return {"ok": False, "failed": "gibdos_not_cleared_north_shutter", "room": 0x65}
        walk_axis(env, assist, total, "y", 109, max_f=400)
        walk_axis(env, assist, total, "x", 120, max_f=400)
        walk_axis(env, assist, total, "y", 93, max_f=300)
        rec = door(env, assist, total, "UP", 0x55, x_force=120, y_force=93, push=280)
        _append(hops, "65_up", rec)
        if rec.get("ok"):
            room = 0x55
        else:
            rec = bomb_east_from_65(env, assist, total)
            _append(hops, "65_bomb_east", rec)
            if not rec.get("success"):
                return {
                    "ok": False,
                    "failed": "north_shutter_sealed_east_bomb_failed",
                    "room": 0x65,
                }
            room = 0x66

    if room == 0x66:
        spec = replace(
            ROOM_66_SPEC,
            spec_id="level5_whistle_66_gibdos",
            source_room=0x65,
            room_id=0x66,
            entry=DoorRoute("RIGHT", ((32, 141),)),
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
            max_frames=28000,
            level=LEVEL_5,
        )
        if not _fight_if_live(
            env, assist, total, hops, spec, (GIBDO_OBJECT_TYPE,), "fight_66"
        ):
            return {"ok": False, "failed": "gibdos_not_cleared", "room": 0x66}
        rec = door(env, assist, total, "UP", 0x56)
        _append(hops, "66_up", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "up_did_not_enter_56", "room": 0x66}
        room = 0x56

    if room == 0x55:
        spec = replace(
            ROOM_66_SPEC,
            spec_id="level5_whistle_55_zols",
            source_room=0x65,
            room_id=0x55,
            entry=DoorRoute("UP", ((120, 205),)),
            enemy_types=(ZOL_OBJECT_TYPE, 0x14, 0x15),
            reward=RewardSpec(kind=RewardKind.CLEAR_ONLY),
            max_frames=20000,
            level=LEVEL_5,
        )
        if not _fight_if_live(
            env, assist, total, hops, spec, (ZOL_OBJECT_TYPE, 0x14, 0x15), "fight_55"
        ):
            return {"ok": False, "failed": "zols_not_cleared_east_shutter", "room": 0x55}
        rec = door(env, assist, total, "RIGHT", 0x56)
        _append(hops, "55_east", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "east_did_not_enter_56", "room": 0x55}
        room = 0x56

    if room == 0x56:
        rec = door(env, assist, total, "RIGHT", 0x57)
        _append(hops, "56_east", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "east_did_not_enter_57", "room": 0x56}
        room = 0x57

    if room == 0x57:
        rec = door(env, assist, total, "UP", 0x47)
        _append(hops, "57_up", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "up_did_not_enter_47", "room": 0x57}
        room = 0x47

    if room == 0x47:
        rec = north_pinch(env, assist, total, 0x37)
        _append(hops, "47_up", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "up_did_not_enter_37", "room": 0x47}
        room = 0x37

    if room == 0x37:
        rec = north_pinch(env, assist, total, 0x27)
        _append(hops, "37_up", rec)
        if not rec.get("ok"):
            return {"ok": False, "failed": "up_did_not_enter_27", "room": 0x37}
        keys0 = read_snapshot(env.get_ram()).keys
        grab_item(
            env,
            assist,
            total,
            lambda e: read_snapshot(e.get_ram()).keys > keys0,
            ((120, 141), (96, 141), (144, 141), (120, 157), (80, 141), (160, 141)),
        )
        room = 0x27

    if room == 0x27:
        if not _walk_west(env, assist, total, hops, walk_west_from_27, 0x26, "27_west"):
            return {"ok": False, "failed": "west_did_not_enter_26", "room": 0x27}
        keys0 = read_snapshot(env.get_ram()).keys
        grab_item(
            env,
            assist,
            total,
            lambda e: read_snapshot(e.get_ram()).keys > keys0,
            ((224, 141), (120, 141), (96, 141), (144, 141)),
        )
        room = 0x26

    if room == 0x26:
        if not _walk_west(env, assist, total, hops, walk_west_from_26, 0x25, "26_west"):
            return {"ok": False, "failed": "west_did_not_enter_25", "room": 0x26}
        room = 0x25

    if room == 0x25:
        if not _walk_west(env, assist, total, hops, walk_west_from_25, 0x24, "25_west"):
            return {"ok": False, "failed": "west_did_not_enter_24", "room": 0x25}
        room = 0x24

    if room != 0x24:
        return {"ok": False, "failed": "not_in_24", "room": room}
    return {"ok": True, "room": 0x24}


def fight_digdogger(env, assist, total: list[int]) -> dict:
    """Mid-room recorder shrink 0x38→0x18, sword, heart, north TF 0x14."""
    walk_axis(env, assist, total, "y", WHISTLE_STAND[1], max_f=200)
    walk_axis(env, assist, total, "x", WHISTLE_STAND[0], max_f=400)
    idle(env, assist, total, 8)
    menu = select_b_item_menu(env, assist, total, WHISTLE_B_SLOT)
    shrunk = False
    for _attempt in range(4):
        for _ in range(12):
            _step(env, assist, total, nes_action("B"))
        for _ in range(20):
            idle(env, assist, total, 12)
            snap = read_snapshot(env.get_ram())
            types = [o.type_id for o in live_types(snap, tuple(range(1, 0x80)))]
            if DIGDOGGER_SHRUNK_TYPE in types or (types and DIGDOGGER_TYPE not in types):
                shrunk = True
                break
            if snap.room_item_id == HEART_CONTAINER and DIGDOGGER_TYPE not in types:
                shrunk = True
                break
        if shrunk:
            break

    idle(env, assist, total, 30)
    snap = read_snapshot(env.get_ram())
    bosses = live_types(snap, (DIGDOGGER_TYPE,))
    small = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id not in SKIP_TYPES and o.hp > 0
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
        if 1 <= o.slot <= 12 and o.type_id not in SKIP_TYPES and o.hp > 0
    ]
    killed = not leftovers and not live_types(snap, (DIGDOGGER_TYPE,))
    hc0 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1
    heart = grab_item(
        env,
        assist,
        total,
        lambda e: (((int(read_u8(e.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1) > hc0,
        (
            (120, 141),
            (120, 125),
            (96, 141),
            (144, 141),
            (120, 157),
            (80, 141),
            (160, 141),
            (224, 141),
        ),
    )
    hc1 = ((int(read_u8(env.get_ram(), ADDR_HEALTH)) >> 4) & 0x0F) + 1

    walk_axis(env, assist, total, "y", 141, max_f=300)
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 93, max_f=400)
    push_dir(env, assist, total, "UP", frames=240)
    wait_play(env, assist, total)
    snap = read_snapshot(env.get_ram())
    tf0 = int(read_u8(env.get_ram(), ADDR_TRIFORCE))
    if snap.screen == LEVEL5_TF_ROOM or snap.room_item_id in (0x1B, HEART_CONTAINER):
        grab_item(
            env,
            assist,
            total,
            lambda e: int(read_u8(e.get_ram(), ADDR_TRIFORCE)) > tf0,
            (
                (120, 141),
                (120, 125),
                (120, 157),
                (96, 141),
                (144, 141),
                (120, 109),
                (80, 141),
                (160, 141),
            ),
        )
    idle(env, assist, total, 40)
    for _ in range(400):
        if int(read_u8(env.get_ram(), ADDR_TRIFORCE)) & TF_BIT_L5:
            break
        _step(env, assist, total, nes_idle_action())
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf1 = int(read_u8(ram, ADDR_TRIFORCE))
    return {
        "ok": bool(tf1 & TF_BIT_L5),
        "menu": menu,
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "fight": None if fight is None else _slim(fight),
        "killed": killed,
        "shrunk": shrunk,
        "heart": heart,
        "hc_in": hc0,
        "hc_out": hc1,
        "tf_in": tf0,
        "tf_out": tf1,
        "tf_l5": bool(tf1 & TF_BIT_L5),
        "room": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "item": snap.room_item_id,
    }


def run_level5_tf_suffix(
    env, *, assist, frame_base: int, stop_at: str = STOP_TRIFORCE
) -> tuple[bool, int, dict]:
    """Env-stepping suffix from leftover 0x04. ``stop_at``: exit04 / room24 / triforce."""
    if stop_at not in TF_SUFFIX_STOPS:
        raise ValueError(f"unknown L5 TF stop {stop_at!r}; wired: {TF_SUFFIX_STOPS}")
    total = [int(frame_base)]
    hops: list[dict] = []

    walk = path_exit_whistle_04(env, assist, total, hops)
    if not walk.get("ok") and not walk.get("success"):
        return False, total[0], {"failed": "exit_whistle_04", "hops": hops}
    if stop_at == STOP_EXIT04:
        snap = read_snapshot(env.get_ram())
        whistle = int(read_u8(env.get_ram(), ADDR_WHISTLE))
        ok = (
            whistle >= 1
            and snap.level == LEVEL_5
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_L5_WHISTLE_05
            and not snap.transitioning
        )
        return ok, total[0], {
            "failed": None if ok else "not_play_0x05",
            "hops": hops,
            "xy": [snap.link_x, snap.link_y],
            "room": snap.screen,
            "mode": snap.mode,
        }

    path = path_05_to_24(env, assist, total, hops)
    if not path.get("ok"):
        snap = read_snapshot(env.get_ram())
        return False, total[0], {
            "failed": path.get("failed"),
            "hops": hops,
            "xy": [snap.link_x, snap.link_y],
            "room": snap.screen,
        }
    if stop_at == STOP_ROOM24:
        snap = read_snapshot(env.get_ram())
        ok = snap.level == LEVEL_5 and snap.screen == 0x24 and snap.mode == PLAY_MODE
        return ok, total[0], {"failed": None if ok else "not_in_24", "hops": hops}

    boss = fight_digdogger(env, assist, total)
    hops.append({"hop": "digdogger", **_slim(boss)})
    ok = bool(boss.get("tf_l5"))
    return ok, total[0], {
        "failed": None if ok else "triforce_bit_0x10",
        "hops": hops,
        "digdogger": boss,
    }


__all__ = [
    "HEART_CONTAINER",
    "STOP_EXIT04",
    "STOP_ROOM24",
    "STOP_TRIFORCE",
    "TF_SUFFIX_STOPS",
    "WHISTLE_B_SLOT",
    "WHISTLE_STAND",
    "door",
    "fight_ctl",
    "fight_digdogger",
    "grab_item",
    "live_types",
    "north_pinch",
    "path_05_to_24",
    "path_exit_whistle_04",
    "run_level5_tf_suffix",
    "take_stairs_06",
    "wait_play",
]
