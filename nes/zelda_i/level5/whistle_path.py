"""Level 5 whistle / bomb / cellar inbound path.

Bomb-west 0x65→0x64 center stairs → cellar 0x07 other mouth →
0x06 key-west → 0x05 clear+block stairs → 0x04 Recorder → left mouth back to 0x05.

Bomb walls are ``BombWallSpec`` rows over one ``bomb_wall`` engine.
Room specs and stop predicates remain in ``level5.dungeon``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from retro_harness.nes import nes_action

from zelda_i.dungeon.engine import (
    DoorRoute,
    DungeonPhase,
    GenericDungeonRoomController,
    RewardKind,
    RewardSpec,
)
from zelda_i.dungeon.ids import DARKNUT_OBJECT_TYPE
from zelda_i.dungeon.ops import goto
from zelda_i.dungeon.pause_select import PauseSelectController, PauseSelectPhase
from zelda_i.level3.dungeon import ROOM_59_SPEC, ROOM_5B_SPEC
from zelda_i.level5.dungeon import (
    BOMB_EAST_STAND,
    BOMB_WEST_66_STAND,
    LEVEL_5,
    ROOM_L5_BLUE_64,
    ROOM_L5_CELLAR_07,
    ROOM_L5_GIBDO_66,
    ROOM_L5_PASSAGE_06,
    ROOM_L5_WEST_65,
    ROOM_L5_WHISTLE_05,
    ROOM_L5_WHISTLE_ITEM,
)
from zelda_i.dungeon.door_hop import door_band_goal
from zelda_i.dungeon.hop_controller import stairs_step
from zelda_i.level5.path import _step, wait_ram, walk_axis
from zelda_i.level9.stairs import BLOCK_STAIRS_X, BLOCK_STAIRS_Y, PUSHABLE_BLOCK
from zelda_i.ram import ADDR_SELECTED_ITEM, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

_rs = read_snapshot

BLUE_DARKNUT_TYPE = 0x0C
CENTER_STAIRS = (120, 141)
CELLAR_MODES = (9, 10, 11, 16)
# 0x06 diamond: 0x68 rests (96,144). Push UP → (96,128). Stairs stand (96,133).
# Center 0x70–0x73 tiles are decorative and do not warp. South key is 0x16, not return.
ROOM_06_BLOCK_X = 96
_FACE = {"UP": 0x08, "DOWN": 0x04, "RIGHT": 0x01, "LEFT": 0x02}


def _play_room(screen: int):
    return lambda snap: snap.mode == PLAY_MODE and snap.screen == screen and not snap.transitioning


def _whistle_bit(env) -> int:
    return int(read_u8(env.get_ram(), ADDR_WHISTLE))


def _cellar_walk_axis(env, assist, total: list[int], axis: str, target: int, max_f: int = 700) -> bool:
    """Axis walk that survives recorder fanfare and aborts on a real 0x04 leave."""
    return walk_axis(
        env,
        assist,
        total,
        axis,
        target,
        max_f=max_f,
        stall_limit=160,
        done=lambda snap: (
            snap.mode == PLAY_MODE and snap.screen != ROOM_L5_WHISTLE_ITEM
        ),
    )


def select_b_item_menu(env, assist, total: list[int], want: int) -> dict:
    """Pause-cycle B items. want=1 bombs, want=5 recorder. No RAM poke.

    Drives the shared ``PauseSelectController`` frame-by-frame instead of
    hand-rolling the START / idle / RIGHT / idle / START cycle timings.
    """
    selected0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
    seen = [selected0]
    if selected0 == want:
        return {"used": False, "selected": selected0, "seen": seen}
    ctl = PauseSelectController(want=want)
    ctl.bind_env(env)
    menu_open = False
    while not ctl.success and not ctl.failed:
        opening = ctl.phase is PauseSelectPhase.CHECK
        closing = ctl.phase is PauseSelectPhase.CLOSE
        action = ctl.step(_rs(env.get_ram()))
        _step(env, assist, total, action.action)
        if opening:
            menu_open = True
        elif closing:
            menu_open = False
        cur = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        if cur != seen[-1]:
            seen.append(cur)
    if ctl.failed and menu_open:
        # Safety net: the shared controller fails closed, but callers here
        # rely on the pause menu always being closed on return.
        _step(env, assist, total, nes_action("START"))
        wait_ram(
            env,
            assist,
            total,
            lambda snap: snap.mode == PLAY_MODE and not snap.transitioning,
            max_frames=64,
            spec_id="close_pause",
        )
    return {
        "used": True,
        "selected_before": selected0,
        "selected_after": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
        "seen": seen,
        "preferred": want if ctl.success else selected0,
        "failed": ctl.failed,
        "fail_reason": ctl.fail_reason,
    }


def _in_cellar(snap) -> bool:
    return snap.mode in CELLAR_MODES


@dataclass(frozen=True)
class BombWallSpec:
    """One L5 bomb wall: how to reach the bricks, which way to face, dest."""

    name: str
    stand: tuple[int, int]
    face: str
    away: str
    dest_room: int
    # Fixed approach: (axis, target, max_f) walked in order.
    approach: tuple[tuple[str, int, int], ...] = ()
    # Candidate approaches probed against ``stand`` (used where the room
    # geometry can pinch); ``source_room`` aborts probing once Link left.
    probe_paths: tuple[tuple[tuple[str, int], ...], ...] = ()
    probe_max_f: int = 400
    probe_tol: int = 8
    source_room: int | None = None
    # Leave the south mouth before the south band (cleared 0x66 river).
    leave_south_mouth: bool = False


BOMB_WEST_66 = BombWallSpec(
    name="bomb_west_from_66",
    stand=BOMB_WEST_66_STAND,
    face="LEFT",
    away="RIGHT",
    dest_room=ROOM_L5_WEST_65,
    probe_paths=(
        (("y", 189), ("x", 32), ("y", 141)),
        (("y", 109), ("x", 32), ("y", 141)),
        (("x", 56), ("y", 109), ("x", 32), ("y", 141)),
    ),
    source_room=ROOM_L5_GIBDO_66,
    leave_south_mouth=True,
)

BOMB_WEST_65 = BombWallSpec(
    name="bomb_west_from_65",
    stand=(32, 141),
    face="LEFT",
    away="RIGHT",
    dest_room=ROOM_L5_BLUE_64,
    approach=(("y", 109, 400), ("x", 32, 400), ("y", 141, 400), ("x", 32, 200)),
)

BOMB_EAST_65 = BombWallSpec(
    name="bomb_east_from_65",
    stand=BOMB_EAST_STAND,
    face="RIGHT",
    away="LEFT",
    dest_room=ROOM_L5_GIBDO_66,
    approach=(
        ("y", 109, 400),
        ("x", 208, 500),
        ("y", BOMB_EAST_STAND[1], 400),
        ("x", BOMB_EAST_STAND[0], 200),
    ),
)


def bomb_wall(env, assist, total: list[int], spec: BombWallSpec) -> dict:
    """Walk to one ``BombWallSpec`` stand, drop one bomb, hold through the hole.

    Bombs come from the pause menu — no RAM poke. The dest room must change;
    the mode-gated hold at the end carries Link through the scroll.
    """
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}
    used = None
    if spec.leave_south_mouth and snap.link_y > 185:
        walk_axis(env, assist, total, "y", 189, max_f=200)
    for axis, tgt, max_f in spec.approach:
        walk_axis(env, assist, total, axis, tgt, max_f=max_f)
    for name, steps in enumerate(spec.probe_paths):
        snap = _rs(env.get_ram())
        if snap.screen == spec.dest_room and snap.mode == PLAY_MODE:
            used = f"already_{spec.dest_room:02x}_{name}"
            break
        if spec.source_room is not None and snap.screen != spec.source_room:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=spec.probe_max_f)
        snap = _rs(env.get_ram())
        if (
            abs(snap.link_x - spec.stand[0]) <= spec.probe_tol
            and abs(snap.link_y - spec.stand[1]) <= spec.probe_tol
        ):
            used = f"path_{name}"
            break
    goto(env, assist, total, spec.stand[0], spec.stand[1], tol=3, max_f=300)
    want_face = _FACE[spec.face]
    wait_ram(
        env,
        assist,
        total,
        lambda snap: snap.facing == want_face or snap.screen == spec.dest_room,
        hold=spec.face,
        max_frames=16,
        spec_id=f"face_{spec.name}",
    )
    menu = select_b_item_menu(env, assist, total, 1)
    snap = _rs(env.get_ram())
    bombs0 = int(snap.bombs)
    sx, sy = spec.stand
    _step(env, assist, total, nes_action(spec.face, "B"))

    def _away(snap) -> bool:
        if snap.screen == spec.dest_room:
            return True
        if spec.face == "LEFT":
            return snap.link_x >= sx + 12
        if spec.face == "RIGHT":
            return snap.link_x <= sx - 12
        return abs(snap.link_x - sx) + abs(snap.link_y - sy) >= 12

    wait_ram(
        env, assist, total, _away, hold=spec.away, max_frames=40, spec_id=f"away_{spec.name}",
    )
    door_bit = {"RIGHT": 0x01, "LEFT": 0x02, "DOWN": 0x04, "UP": 0x08}[spec.face]

    def _hole_open(snap) -> bool:
        return (
            snap.screen == spec.dest_room
            or bool(snap.cur_opened_doors & door_bit)
        )

    # Stand still during fuse blast (do not walk into the live bomb)
    wait_ram(
        env,
        assist,
        total,
        _hole_open,
        hold=None,
        max_frames=140,
        spec_id=f"blast_{spec.name}",
    )
    wait_ram(
        env,
        assist,
        total,
        _play_room(spec.dest_room),
        hold=spec.face,
        max_frames=600,
        spec_id=f"hole_{spec.name}",
    )
    snap = _rs(env.get_ram())
    return {
        "path": spec.name,
        "via": used,
        "start": start,
        "menu": menu,
        "bombs_in": bombs0,
        "bombs_out": int(snap.bombs),
        "bombs_spent": bombs0 - int(snap.bombs),
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": (
            snap.level == LEVEL_5
            and snap.screen == spec.dest_room
            and snap.mode == PLAY_MODE
        ),
    }


def bomb_west_from_66(env, assist, total: list[int]) -> dict:
    """Bomb the west wall of cleared 0x66. One bomb. Dest must become 0x65.

    Horizontal river at y≈141 locks sideways input on the Stepladder. Leave
    the south mouth, hold the south band y=189 to the west column, then rise
    to the bomb bricks. North-band y=109 is the fallback if the south pinch
    stalls.
    """
    return bomb_wall(env, assist, total, BOMB_WEST_66)


def bomb_west_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the west wall of cleared 0x65. One bomb. Dest must become 0x64.

    Live 0x65 has a center diamond: y=109 then x=32 then y=141, not y=141
    first. Hold LEFT through the west scroll even while SCREEN still reads 0x65.
    """
    return bomb_wall(env, assist, total, BOMB_WEST_65)


def bomb_east_from_65(env, assist, total: list[int]) -> dict:
    """Bomb the east wall of cleared 0x65. One bomb. Dest must become 0x66.

    North shutter is one-way (0x55 S=open / 0x65 N=shutter). Diamond: y=109
    then east, not y=141 first.
    """
    return bomb_wall(env, assist, total, BOMB_EAST_65)


STAIRS_LATTICE_FRAMES = 900


def take_center_stairs_64(env, assist, total: list[int]) -> dict:
    """Walk the south (then north) gap onto visible center stairs in 0x64.

    Do not hunt the east bomb hole. Success = cellar/stairs mode or room 0x07.
    """
    log = []
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "mode": snap.mode, "room": snap.screen}

    def done(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.level == LEVEL_5 and snap.screen == ROOM_L5_CELLAR_07

    paths = (
        (("y", 189), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 96), ("y", 149), ("x", 120), ("y", 141)),
        (("y", 189), ("x", 64), ("y", 141), ("x", 120)),
        (("y", 93), ("x", 80), ("y", 141), ("x", 120)),
        (("y", 189), ("x", 120), ("y", 141)),
        (("y", 173), ("x", 120), ("y", 141), ("x", 120)),
    )
    # ROM-driven first: push the block secret if it is pending, then walk
    # the lattice onto the stair tile. The hand paths below are the fallback.
    for _ in range(STAIRS_LATTICE_FRAMES):
        snap = _rs(env.get_ram())
        if done(snap) or snap.screen != ROOM_L5_BLUE_64:
            break
        step = stairs_step(env, snap)
        if step is None:
            break
        _step(env, assist, total, nes_action(step))
    for name_i, steps in enumerate(paths):
        if done(_rs(env.get_ram())):
            break
        if _rs(env.get_ram()).screen != ROOM_L5_BLUE_64:
            break
        for axis, tgt in steps:
            walk_axis(env, assist, total, axis, tgt, max_f=360)
            snap = _rs(env.get_ram())
            log.append(
                {
                    "path": name_i,
                    "step": f"{axis}:{tgt}",
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if done(snap):
                break
        if wait_ram(env, assist, total, done, max_frames=24, spec_id="stairs64_warp"):
            break
        # Nudge onto the tile; never hold LEFT (east bomb hole → 0x65).
        for direction in ("UP", "DOWN", "RIGHT"):
            if wait_ram(
                env,
                assist,
                total,
                lambda snap: done(snap) or snap.screen != ROOM_L5_BLUE_64,
                hold=direction,
                max_frames=16,
                spec_id=f"stairs64_{direction.lower()}",
            ):
                break
        if done(_rs(env.get_ram())):
            break

    wait_ram(
        env,
        assist,
        total,
        lambda snap: done(snap) or (snap.mode == PLAY_MODE and snap.screen != ROOM_L5_BLUE_64),
        max_frames=200,
        spec_id="stairs64_settle",
    )
    snap = _rs(env.get_ram())
    return {
        "path": "south_gap_center_stairs",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "cellar": _in_cellar(snap),
        "success": done(snap) and snap.screen != ROOM_L5_WEST_65,
    }


# Live L5 cellar 0x07: left mouth spawn ~(48,93); floor y=189; right climb x=192.
L5_CELLAR_FLOOR_Y = 189
L5_CELLAR_LEFT_X = 48
L5_CELLAR_RIGHT_X = 192


def cellar_other_mouth(env, assist, total: list[int]) -> dict:
    """From L5 cellar 0x07, take the opposite mouth to room 0x06. No pokes."""
    # Stair-enter sits at (128,141), then remaps to a ladder (48,93) or (192,93).
    wait_ram(
        env,
        assist,
        total,
        lambda snap: snap.mode in CELLAR_MODES and (snap.link_x <= 64 or snap.link_x >= 176),
        max_frames=180,
        spec_id="cellar07_spawn",
    )
    snap = _rs(env.get_ram())
    start = {"xy": [snap.link_x, snap.link_y], "room": snap.screen, "mode": snap.mode}
    # Left column is the 0x64 return. Floor-cross to x=192 then UP → 0x06.
    if snap.link_x <= 128:
        side = "right"
        tx = L5_CELLAR_RIGHT_X
    else:
        side = "left"
        tx = L5_CELLAR_LEFT_X
    leftover = (snap.link_x, snap.link_y)
    gx, _gy = door_band_goal("UP", leftover, (tx, 93))
    walk_axis(env, assist, total, "y", L5_CELLAR_FLOOR_Y, max_f=400)
    walk_axis(env, assist, total, "x", gx, max_f=500)
    wait_ram(
        env,
        assist,
        total,
        _play_room(ROOM_L5_PASSAGE_06),
        hold="UP",
        max_frames=400,
        spec_id="cellar07_up",
    )
    snap = _rs(env.get_ram())
    return {
        "path": "cellar_other_mouth",
        "start": start,
        "chose_side": side,
        "target_x": tx,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "success": snap.level == LEVEL_5 and snap.screen == ROOM_L5_PASSAGE_06 and snap.mode == PLAY_MODE,
    }


def key_west_to(env, assist, total: list[int], expect: int) -> dict:
    """Spend a key at the west door. No door/key poke."""
    snap = _rs(env.get_ram())
    keys0 = int(snap.keys)
    leftover = (snap.link_x, snap.link_y)
    gx, gy = door_band_goal("LEFT", leftover, (32, 141))
    walk_axis(env, assist, total, "y", gy, max_f=400)
    walk_axis(env, assist, total, "x", gx, max_f=500)
    goto(env, assist, total, gx, gy, tol=3, max_f=300)
    wait_ram(
        env,
        assist,
        total,
        _play_room(expect),
        hold="LEFT",
        max_frames=400,
        spec_id="key_west",
    )
    snap = _rs(env.get_ram())
    return {
        "path": "key_west",
        "keys_in": keys0,
        "keys_out": int(snap.keys),
        "key_spent": int(snap.keys) < keys0,
        "dest": snap.screen,
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "success": snap.level == LEVEL_5 and snap.screen == expect and snap.mode == PLAY_MODE,
    }


def fight_blue_darknuts(env, assist, total: list[int], room: int, expected: int, source: int) -> dict:
    """Reuse GenericDungeonRoomController + ROOM_5B_SPEC / ROOM_59 combat."""
    spec = replace(
        ROOM_5B_SPEC,
        spec_id=f"level5_room{room:02x}_blue_darknuts",
        source_room=source,
        room_id=room,
        entry=DoorRoute("LEFT", ((224, 141),)),
        enemy_types=(BLUE_DARKNUT_TYPE, DARKNUT_OBJECT_TYPE),
        expected_enemy_count=expected,
        required_open_doors=0,
        reward=RewardSpec(kind=RewardKind.CLEAR_ONLY, settle_all_dead=0),
        combat=ROOM_59_SPEC.combat,
        exit_routes=(DoorRoute("RIGHT", ((208, 141),)),),
        max_frames=28000,
        level=LEVEL_5,
    )
    ctl = GenericDungeonRoomController(spec)
    start_n = None
    last_n = None
    progress = []
    for _ in range(spec.max_frames):
        snap = _rs(env.get_ram())
        if snap.mode == PLAY_MODE and snap.screen == room:
            live = spec.live_enemies(snap)
            if start_n is None:
                start_n = len(live)
                last_n = start_n
                progress.append({"f": ctl.frames, "n": start_n})
            elif len(live) != last_n:
                last_n = len(live)
                progress.append({"f": ctl.frames, "n": last_n})
        action = ctl.step(snap)
        _step(env, assist, total, action.action)
        if ctl.success or ctl.phase is DungeonPhase.FAILED:
            break
    snap = _rs(env.get_ram())
    live = [
        o
        for o in snap.objects
        if 1 <= o.slot <= 12 and o.type_id in (BLUE_DARKNUT_TYPE, DARKNUT_OBJECT_TYPE) and o.hp > 0
    ] if snap.mode == PLAY_MODE else []
    return {
        "ok": bool(ctl.success) and not live,
        "frames": ctl.frames,
        "start_n": 0 if start_n is None else start_n,
        "end_n": len(live),
        "progress": progress,
        "spec_id": spec.spec_id,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
    }


def push_block_stairs(env, assist, total: list[int], room: int) -> dict:
    """Push 0x68 then stand on revealed stairs. Never treat a door exit as stairs."""
    snap = _rs(env.get_ram())
    blocks = [
        o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
    ]
    log = []
    dest = None

    def left_ok(snap) -> bool:
        if _in_cellar(snap):
            return True
        return snap.screen != room and snap.mode in (*CELLAR_MODES, PLAY_MODE) and snap.screen != ROOM_L5_WEST_65

    targets = [(b.x, b.y) for b in blocks] + [
        (96, 144),
        (112, 144),
        (80, 144),
        (120, 144),
        (128, 144),
    ]
    seen = set()
    for tx, ty in targets:
        key = (tx // 8, ty // 8)
        if key in seen:
            continue
        seen.add(key)
        snap = _rs(env.get_ram())
        if left_ok(snap):
            dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
            break
        walk_axis(env, assist, total, "y", ty, max_f=280)
        walk_axis(env, assist, total, "x", tx + 16, max_f=280)
        rec = {"stand": [tx, ty], "dirs": []}
        blocks0 = [(b.x, b.y) for b in blocks]

        def _moved(snap) -> bool:
            if left_ok(snap):
                return True
            now = [
                (o.x, o.y)
                for o in snap.objects
                if 1 <= o.slot <= 12 and o.type_id == PUSHABLE_BLOCK
            ]
            return bool(now) and now != blocks0

        for direction in ("LEFT", "UP", "DOWN", "RIGHT"):
            wait_ram(
                env,
                assist,
                total,
                _moved,
                hold=direction,
                max_frames=90,
                spec_id=f"push68_{direction.lower()}",
            )
            snap = _rs(env.get_ram())
            rec["dirs"].append(
                {
                    "dir": direction,
                    "xy": [snap.link_x, snap.link_y],
                    "mode": snap.mode,
                    "room": snap.screen,
                }
            )
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        log.append(rec)
        if dest is not None:
            break
        for sx, sy in ((tx, ty), (BLOCK_STAIRS_X, BLOCK_STAIRS_Y), CENTER_STAIRS, (120, 125)):
            walk_axis(env, assist, total, "y", sy, max_f=200)
            walk_axis(env, assist, total, "x", sx, max_f=200)
            wait_ram(env, assist, total, left_ok, max_frames=16, spec_id="stairs_stand")
            snap = _rs(env.get_ram())
            if left_ok(snap):
                dest = {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]}
                break
        if dest is not None:
            break
    snap = _rs(env.get_ram())
    return {
        "blocks_seen": [{"slot": b.slot, "x": b.x, "y": b.y} for b in blocks],
        "dest": dest,
        "end": {"room": snap.screen, "mode": snap.mode, "xy": [snap.link_x, snap.link_y]},
        "log": log,
        "success": dest is not None,
    }



def take_whistle_04(env, assist, total: list[int]) -> dict:
    """Cellar 0x04: floor y=189, short ladder x=176, left on y=141 to the Recorder."""
    w0 = _whistle_bit(env)

    def got(_snap=None) -> bool:
        return _whistle_bit(env) > w0

    walk_axis(env, assist, total, "y", 189, max_f=400)
    walk_axis(env, assist, total, "x", 176, max_f=400)
    wait_ram(
        env,
        assist,
        total,
        lambda snap: got() or (snap.link_y <= 141 and abs(snap.link_x - 176) <= 4),
        hold="UP",
        max_frames=80,
        spec_id="whistle04_climb",
    )
    walk_axis(env, assist, total, "y", 141, max_f=200)
    _s = _rs(env.get_ram())
    leftover = (_s.link_x, _s.link_y)
    gx, gy = door_band_goal("LEFT", leftover, (128, 141))
    walk_axis(env, assist, total, "y", gy, max_f=200)
    walk_axis(env, assist, total, "x", gx, max_f=300)
    wait_ram(env, assist, total, got, max_frames=24, spec_id="whistle04_item")
    if not got():
        walk_axis(env, assist, total, "x", 144, max_f=200)
        walk_axis(env, assist, total, "x", 120, max_f=200)
        wait_ram(env, assist, total, got, max_frames=20, spec_id="whistle04_hunt")
    w1 = _whistle_bit(env)
    snap = _rs(env.get_ram())
    return {
        "in": w0,
        "out": w1,
        "got": w1 > w0,
        "xy": [snap.link_x, snap.link_y],
        "room": snap.screen,
        "mode": snap.mode,
    }


def hunt_whistle(env, assist, total: list[int]) -> dict:
    """Walk item stands until ADDR_WHISTLE becomes 1.

    Room 0x04 is a side-scroll item cellar: top-down stands stay on the
    floor (y=189). Use take_whistle_04 (right short ladder -> y=141).
    """
    w0 = _whistle_bit(env)
    snap0 = _rs(env.get_ram())
    room0 = snap0.screen
    hits = []
    if room0 == ROOM_L5_WHISTLE_ITEM or snap0.mode in CELLAR_MODES:
        cellar = take_whistle_04(env, assist, total)
        hits.append({"via": "take_whistle_04", "xy": cellar.get("xy"), "value": cellar.get("out")})
        if cellar.get("got"):
            return {"in": w0, "out": cellar["out"], "got": True, "hits": hits, "via": "take_whistle_04"}
    stands = (
        (120, 141),
        (136, 141),
        (104, 141),
        (120, 125),
        (120, 157),
        (80, 141),
        (160, 141),
        (120, 109),
        (64, 117),
        (176, 117),
        (96, 165),
        (144, 165),
    )

    def got(_snap=None) -> bool:
        return _whistle_bit(env) > w0

    for tx, ty in stands:
        snap = _rs(env.get_ram())
        if snap.screen != room0 and snap.mode == PLAY_MODE:
            break
        walk_axis(env, assist, total, "y", ty, max_f=220)
        walk_axis(env, assist, total, "x", tx, max_f=220)
        wait_ram(env, assist, total, got, max_frames=16, spec_id="hunt_whistle")
        w1 = _whistle_bit(env)
        snap = _rs(env.get_ram())
        hits.append({"stand": [tx, ty], "xy": [snap.link_x, snap.link_y], "value": w1})
        if w1 > w0:
            break
    w1 = _whistle_bit(env)
    return {"in": w0, "out": w1, "got": w1 > w0, "hits": hits}


# Live 0x04 item cellar: isolated recorder alcove ~y=141, x≈112–176.
# Short ladder at x=176 drops to pit y=189. Left mouth stairs at x=48
# (spawn 48,65) return to play 0x05. Do not walk left on the alcove —
# the platform does not connect to the left column.
WHISTLE_04_LADDER_X = 176
WHISTLE_04_PIT_Y = 189
WHISTLE_04_MOUTH_X = 48


def exit_whistle_04(env, assist, total: list[int]) -> dict:
    """Leave cellar 0x04: alcove x=176 DOWN → pit y=189 → left mouth x=48 UP → 0x05.

    Failed probes walked LEFT/UP on the recorder alcove (y=141). That platform
    does not connect to the left column. Drop the short ladder first.
    """
    snap = _rs(env.get_ram())
    start = {
        "xy": [snap.link_x, snap.link_y],
        "mode": snap.mode,
        "room": snap.screen,
        "whistle": _whistle_bit(env),
    }
    log = [dict(start, tag="start")]
    x0, y0 = snap.link_x, snap.link_y

    def left_ok(s) -> bool:
        return s.mode == PLAY_MODE and s.screen != ROOM_L5_WHISTLE_ITEM

    def rec(tag: str) -> None:
        s = _rs(env.get_ram())
        log.append(
            {
                "tag": tag,
                "xy": [s.link_x, s.link_y],
                "mode": s.mode,
                "room": s.screen,
                "tile": int(s.colliding_tile),
            }
        )

    # Recorder item-get holds Link overhead. Dest is RAM xy change, not idle(n).
    thawed = wait_ram(
        env,
        assist,
        total,
        lambda s: s.link_x != x0 or s.link_y != y0 or left_ok(s),
        hold="RIGHT",
        max_frames=280,
        spec_id="thaw_04",
    )
    rec("unstick")

    def drop_ok(s) -> bool:
        return left_ok(s) or s.link_y >= WHISTLE_04_PIT_Y - 2

    # Alcove (y≈141) only drops at the short ladder x=176.
    for attempt in range(3):
        snap = _rs(env.get_ram())
        if left_ok(snap) or snap.link_y >= 170:
            break
        _cellar_walk_axis(env, assist, total, "y", 141, max_f=240)
        _cellar_walk_axis(env, assist, total, "x", WHISTLE_04_LADDER_X, max_f=700)
        rec("ladder")
        wait_ram(env, assist, total, drop_ok, hold="DOWN", max_frames=280, spec_id="drop_04")
        _cellar_walk_axis(env, assist, total, "y", WHISTLE_04_PIT_Y, max_f=400)
        rec("pit")
        snap = _rs(env.get_ram())
        if snap.link_y < 170 and abs(snap.link_x - WHISTLE_04_LADDER_X) > 4:
            log.append({"tag": f"retry_ladder_{attempt}", "xy": [snap.link_x, snap.link_y]})

    snap = _rs(env.get_ram())
    # Live RAM: only the pit (y>=170) connects to the left mouth. Do not
    # walk LEFT on the alcove — that stalls at x≈112, y=141.
    if not left_ok(snap) and snap.link_y >= 170:
        leftover = (snap.link_x, snap.link_y)
        gx, _gy = door_band_goal("UP", leftover, (WHISTLE_04_MOUTH_X, 93))
        _cellar_walk_axis(env, assist, total, "x", gx, max_f=700)
        rec("left_col")
        wait_ram(
            env,
            assist,
            total,
            left_ok,
            hold="UP",
            max_frames=400,
            spec_id="mouth_04",
        )
    rec("after_up")
    snap = _rs(env.get_ram())
    return {
        "path": "alcove_ladder176_pit189_left48",
        "start": start,
        "log": log,
        "dest": snap.screen,
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "whistle": _whistle_bit(env),
        "success": (
            snap.level == LEVEL_5
            and snap.mode == PLAY_MODE
            and snap.screen == ROOM_L5_WHISTLE_05
        ),
        "left_cellar": left_ok(snap),
        "thawed": thawed,
    }
