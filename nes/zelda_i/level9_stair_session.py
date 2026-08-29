"""Level 9 stair session: env-step primitives, dump/probe, compose-to-credits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level2_puzzles import BombWall
from zelda_i.level9_ganon import ADDR_SELECTED_ITEM, B_ITEM_ARROWS, B_ITEM_BOMBS, LEVEL9
from zelda_i.level9_patra import PATRA_MAX_FRAMES, patra_action, patra_body, patra_eyes
from zelda_i.level9_stairs import (
    BOMB_WALL_04_WEST,
    BOMB_WALL_31_WEST,
    BOMB_WEST_STAND,
    CELLAR_MODE,
    ROOM03,
    ROOM03_STAIR_X,
    ROOM03_STAIR_Y,
    ROOM04,
    ROOM11,
    ROOM13,
    ROOM21,
    ROOM21_WEST_X,
    ROOM30,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    ROOM31,
    ROOM40,
    ROOM41,
    ROOM51,
    SOUTH_DOOR_Y,
    StairLoader,
    cellar_exit_step,
    cellar_for_play_room,
    chase_sword_step,
    dest_report,
    door_column_step,
    in_stair_source,
    live_combat_objects,
    loader_avoids,
    rom_doors_report,
    rom_pair,
    rom_secret,
    room03_rom_neighbors,
    room30_rom_neighbors,
    stair_loader_for,
    stair_transition_modes,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)

BEAD = "rr-sz8.3"
TAG = "l9_stair_patra_credits_recon"
FIXTURE_SOURCE = "Level9EntranceReconFixture"
SETTLE_IDLE_FRAMES = 20
LOAD_MAX_FRAMES = 500
WALK_MAX_FRAMES = 400
PUSH_FRAMES = 40
CLEAR_MAX_FRAMES = 2500
CELLAR_MAX_FRAMES = 2800
ADDR_UPDATING = 0x0011
ADDR_SUBMODE = 0x0013
ADDR_UW_EXIT_TYPE = 0x005A
NORTH_PROBE_FRAMES = 400
CLEAR_PROBE_FRAMES = 1800
StepFn = Callable[[Any], Any]


def _assign(env: Any, address: int, value: int) -> None:
    env.unwrapped.data.memory.assign(int(address), "|u1", int(value) & 0xFF)


def _step(env: Any, action: list[int], *, assist: Any, total: list[int]):
    obs, *_ = env.step(action)
    total[0] += 1
    if assist is not None:
        assist.apply_env(env, frame=total[0])
    return obs


def _idle(env: Any, frames: int, *, assist: Any, total: list[int]):
    obs = None
    for _ in range(frames):
        obs = _step(env, nes_idle_action(), assist=assist, total=total)
    return obs


def _full_loadout() -> tuple[tuple[str, int, int], ...]:
    from zelda_i.scripts.run_level9_ganon import FULL_LOADOUT
    return FULL_LOADOUT


def _loops():
    import zelda_i.level9_stair_run as loops
    return loops


def _loader_write_rows(loader: StairLoader) -> list[dict[str, Any]]:
    rows = [
        {"name": name, "address": address, "address_hex": f"0x{address:04X}", "value": value}
        for name, address, value in _full_loadout()
    ]
    rows.extend(
        [
            {"name": "loader_level", "address": ADDR_LEVEL, "address_hex": "0x0010", "value": LEVEL9},
            {"name": "loader_mode", "address": ADDR_MODE, "address_hex": "0x0012", "value": PLAY_MODE},
            {"name": "loader_current_room", "address": ADDR_SCREEN, "address_hex": "0x00EB", "value": loader.from_room},
            {"name": "loader_next_room", "address": ADDR_NEXT_SCREEN, "address_hex": "0x00EC", "value": loader.room},
            {
                "name": "loader_link_position", "addresses": [ADDR_LINK_X, ADDR_LINK_Y],
                "address_hex": ["0x0070", "0x0084"], "values": [loader.link_x, loader.link_y],
            },
            {
                "name": "loader_door_staging",
                "addresses": [ADDR_CUR_OPENED_DOORS, ADDR_OPEN_DOORWAY_MASK],
                "address_hex": ["0x00EE", "0x033F"], "values": [0x0F, 0x0F],
            },
            {
                "name": "loader_hold_direction", "value": loader.direction,
                "from_room": loader.from_room, "to_room": loader.room, "label": loader.label,
            },
        ]
    )
    return rows


def _apply_loader(
    env: Any, loader: StairLoader, *, door_staging: bool = True, selected_item: int | None = None,
) -> None:
    for _, address, value in _full_loadout():
        _assign(env, address, value)
    pairs = [
        (ADDR_LEVEL, LEVEL9), (ADDR_MODE, PLAY_MODE),
        (ADDR_SCREEN, loader.from_room), (ADDR_NEXT_SCREEN, loader.room),
        (ADDR_LINK_X, loader.link_x), (ADDR_LINK_Y, loader.link_y),
    ]
    if door_staging:
        pairs.extend(((ADDR_CUR_OPENED_DOORS, 0x0F), (ADDR_OPEN_DOORWAY_MASK, 0x0F)))
    if selected_item is not None:
        pairs.append((ADDR_SELECTED_ITEM, int(selected_item) & 0xFF))
    for address, value in pairs:
        _assign(env, address, value)


def _hold_until_room(env: Any, loader: StairLoader, *, total: list[int], max_frames: int = LOAD_MAX_FRAMES):
    obs = None
    for _ in range(max_frames):
        obs = _step(env, nes_action(loader.direction), assist=None, total=total)
        if in_stair_source(read_snapshot(env.get_ram()), loader.room):
            return obs, True
    return obs, False


def materialize_stair_room(
    env: Any, room: int, *, total: list[int], door_staging: bool = True, selected_item: int | None = None,
) -> tuple[Any, StairLoader, bool]:
    loader = stair_loader_for(room)
    reset_obs(env)
    _apply_loader(env, loader, door_staging=door_staging, selected_item=selected_item)
    obs, loaded = _hold_until_room(env, loader, total=total)
    if loaded:
        obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
    return obs, loader, loaded


def _left_source(snap: Any, source: int) -> bool:
    return snap.screen != int(source) and not snap.transitioning and snap.mode not in (6, 7)


def _exit_cellar(env: Any, *, total: list[int], side: str, max_frames: int = CELLAR_MAX_FRAMES):
    obs = None
    start = read_snapshot(env.get_ram())
    start_room, start_xy = int(start.screen), (int(start.link_x), int(start.link_y))
    placed = False
    for i in range(max_frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.level == LEVEL9 and snap.screen != start_room and not snap.transitioning:
            return obs, snap
        if snap.mode == PLAY_MODE and snap.screen == start_room and not stair_transition_modes(start.mode) and i > 30:
            return obs, snap
        if not placed:
            moved = abs(int(snap.link_x) - start_xy[0]) > 8 or abs(int(snap.link_y) - start_xy[1]) > 8
            if snap.mode == CELLAR_MODE and not snap.transitioning and (moved or i > 90):
                placed = True
            else:
                obs = _step(env, nes_action("UP"), assist=None, total=total)
                continue
        obs = _step(env, cellar_exit_step(snap, side=side).action, assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def _walk_target(env: Any, total: list[int], x: int, y: int, frames: int = WALK_MAX_FRAMES):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if stair_transition_modes(snap.mode) or _left_source(snap, snap.screen):
            return obs, snap
        frame = walk_to_step(snap, x, y, y_first=True)
        if frame.reason == "walk_arrived":
            return obs, snap
        obs = _step(env, frame.action, assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def _door_step(
    source: int, dest: int, direction: str, *, south_y: int | None = SOUTH_DOOR_Y, west_band: int | None = None,
) -> StepFn:
    def step_fn(snap: Any):
        return door_column_step(
            snap, source=source, dest=dest, direction=direction, south_y=south_y, west_band=west_band,
        )
    return step_fn


def probe_room(
    env: Any, step_fn: StepFn, dest_room: int, frames: int, *, total: list[int],
    assist: Any = None, hold: str = "UP", source_room: int | None = None,
) -> dict[str, Any]:
    start = read_snapshot(env.get_ram())
    source = int(source_room if source_room is not None else start.screen)
    transition = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen == dest_room and snap.mode == PLAY_MODE and not snap.transitioning:
            transition = {
                "from_room": source, "direction": hold, "to_room": int(snap.screen),
                "mode": int(snap.mode), "objects": dest_report(snap)["objects"],
            }
            break
        if snap.screen != source or snap.mode != PLAY_MODE or snap.transitioning:
            _step(env, nes_action(hold), assist=assist, total=total)
            continue
        _step(env, step_fn(snap).action, assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "start_room": int(start.screen), "end_room": int(end.screen), "end_mode": int(end.mode),
        "link": {"x": end.link_x, "y": end.link_y},
        "doors": dest_report(end)["doors"], "mask": dest_report(end)["mask"],
        "entered_other_room": transition is not None,
        f"entered_0x{dest_room:02x}": int(end.screen) == dest_room,
        "transition": transition, "still_in_source": bool(in_stair_source(end, source)),
        "stuck_y": int(end.link_y),
    }


def _new_env() -> Any:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")


def _png(obs: Any, path: Any) -> str:
    save_rgb_png(obs, path)
    return str(path)


def _clear_combat(env: Any, room: int, *, total: list[int], assist: Any, frames: int = CLEAR_MAX_FRAMES) -> None:
    cooldown = 0
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if in_stair_source(snap, room) and not live_combat_objects(snap):
            break
        if not in_stair_source(snap, room):
            break
        frame, cooldown = chase_sword_step(snap, cooldown)
        _step(env, frame.action, assist=assist, total=total)
    _idle(env, 16, assist=assist, total=total)


def _clear_room21(env: Any, *, total: list[int], assist: Any = None) -> dict[str, Any]:
    cooldown = frames = 0
    for _ in range(PATRA_MAX_FRAMES):
        snap = read_snapshot(env.get_ram())
        if not in_stair_source(snap, ROOM21):
            break
        live = patra_body(snap) is not None or bool(patra_eyes(snap))
        if not live and not live_combat_objects(snap):
            break
        if live:
            action, _reason, cooldown = patra_action(snap, cooldown=cooldown)
            _step(env, action, assist=assist, total=total)
        else:
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)
        frames += 1
    _idle(env, 16, assist=assist, total=total)
    end = read_snapshot(env.get_ram())
    return {
        "frames": frames, "body_alive": patra_body(end) is not None, "eyes": len(patra_eyes(end)),
        "combat_left": len(live_combat_objects(end)), "doors": dest_report(end)["doors"],
        "screen": int(end.screen), "link": {"x": end.link_x, "y": end.link_y},
    }


def _wait_play_room(env: Any, room: int, *, total: list[int], assist: Any, frames: int = 180):
    idle = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and int(snap.screen) == room and not snap.transitioning:
            break
        idle = _step(env, nes_action("UP"), assist=assist, total=total)
    snap = read_snapshot(env.get_ram())
    if snap.mode == PLAY_MODE:
        idle = _idle(env, 16, assist=assist, total=total)
        snap = read_snapshot(env.get_ram())
    return idle, snap


def _pause_select_arrows(env: Any, *, total: list[int], assist: Any) -> tuple[int, int]:
    _idle(env, 45, assist=assist, total=total)
    _step(env, nes_action("START"), assist=assist, total=total)
    _idle(env, 40, assist=assist, total=total)
    pause_moves = 0
    for _ in range(8):
        selected = int(env.get_ram()[ADDR_SELECTED_ITEM])
        if selected == B_ITEM_ARROWS:
            break
        _step(env, nes_action("RIGHT"), assist=assist, total=total)
        _idle(env, 8, assist=assist, total=total)
        pause_moves += 1
    _step(env, nes_action("START"), assist=assist, total=total)
    _idle(env, 40, assist=assist, total=total)
    return int(env.get_ram()[ADDR_SELECTED_ITEM]), pause_moves


def _seed_west_south_03(env: Any, *, total: list[int], assist: Any) -> None:
    for x, y, y_first in ((208, 189, True), (32, 189, False)):
        for _ in range(WALK_MAX_FRAMES):
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM03:
                return
            frame = walk_to_step(snap, x, y, y_first=y_first, tol=0)
            if frame.reason == "walk_arrived":
                break
            _step(env, frame.action, assist=assist, total=total)


@dataclass
class DumpPlan:
    room: int
    rom_rooms: tuple[int, ...]
    extra: dict[str, Any] = field(default_factory=dict)
    selected_item: int | None = None
    probe_dest: int | None = None
    probe_hold: str = "UP"
    probe_south_y: int | None = SOUTH_DOOR_Y
    probe_west_band: int | None = None
    actions: tuple[str, ...] = ()
    bomb_wall: BombWall | None = None
    bomb_dest: int | None = None
    bomb_timeout: int | None = None
    lands_key: str = ""
    how_key: str = ""
    next_candidate: str = ""
    stair_poke: tuple[int, int] | None = None


def dump_room(plan: DumpPlan, *, tag: str) -> dict[str, Any]:
    loops = _loops()
    assist = UnlimitedHealthAssist(enabled=True)
    room = plan.room
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False,
        "fixture_only": True, "init_mode9": False, "room": f"0x{room:02X}",
        "rom_doors": rom_doors_report(*plan.rom_rooms), **plan.extra,
    }
    env = _new_env()
    total = [0]

    def remat(door_staging: bool = True) -> tuple[Any, StairLoader, bool]:
        nonlocal env, total
        env.close()
        env = _new_env()
        total = [0]
        return materialize_stair_room(
            env, room, total=total, door_staging=door_staging, selected_item=plan.selected_item,
        )

    def step_fn() -> StepFn:
        assert plan.probe_dest is not None
        return _door_step(
            room, plan.probe_dest, plan.probe_hold,
            south_y=plan.probe_south_y, west_band=plan.probe_west_band,
        )

    def do_probe(key: str) -> dict[str, Any]:
        assert plan.probe_dest is not None
        result = probe_room(
            env, step_fn(), plan.probe_dest, NORTH_PROBE_FRAMES + 800,
            total=total, assist=assist, hold=plan.probe_hold, source_room=room,
        )
        report[key] = result
        idle = _idle(env, 1, assist=assist, total=total)
        report[f"{key}_png"] = _png(idle, RECORDINGS_DIR / f"{tag}_{key}.png")
        dest_snap = read_snapshot(env.get_ram())
        if plan.probe_dest is not None and int(dest_snap.screen) == plan.probe_dest and dest_snap.mode == PLAY_MODE:
            report["dest_screen"] = int(dest_snap.screen)
            report["dest_mode"] = int(dest_snap.mode)
            report[plan.lands_key] = True
            report["dest_png"] = _png(idle, RECORDINGS_DIR / f"{tag}_dest.png")
            report["dest_objects"] = dest_report(dest_snap)["objects"]
        return result

    try:
        obs, loader, loaded = materialize_stair_room(
            env, room, total=total, door_staging=True, selected_item=plan.selected_item,
        )
        report["loader"] = {
            "label": loader.label, "from_room": f"0x{loader.from_room:02X}",
            "direction": loader.direction, "link_xy": [loader.link_x, loader.link_y],
            "door_staging_on_from_room": True,
        }
        report["loaded"] = loaded
        if not loaded:
            report["error"] = f"loader did not settle 0x{room:02X}"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        snap = read_snapshot(env.get_ram())
        report["settled"] = dest_report(snap)
        report["link_start"] = {"x": snap.link_x, "y": snap.link_y}
        if plan.selected_item is not None:
            report["selected_item_live"] = int(env.get_ram()[ADDR_SELECTED_ITEM])
            report["bombs_live"] = int(snap.bombs)
        report["settle_png"] = _png(obs, RECORDINGS_DIR / f"{tag}_settle.png")

        for action in plan.actions:
            if action == "tiles":
                tiles = loops.dump_room_tiles(env, total=total)
                tile_json = RECORDINGS_DIR / f"{tag}_tiles.json"
                write_json_report(tile_json, tiles)
                report["tiles"] = {k: tiles[k] for k in ("stair_hits", "mouth_hits", "tile_counts")}
                report["tile_json"] = str(tile_json)
                report["stair_hits"] = tiles["stair_hits"]
            elif action == "remat":
                _, _, ok = remat()
                if not ok:
                    report["error"] = f"rematerialize failed 0x{room:02X}"
                    return report
            elif action == "probe":
                do_probe("north_probe_uncleared" if plan.probe_hold == "UP" else "south_probe_uncleared")
            elif action == "clear":
                _clear_combat(env, room, total=total, assist=assist, frames=CLEAR_PROBE_FRAMES)
                report["after_clear"] = dest_report(read_snapshot(env.get_ram()))
                report["after_clear_png"] = _png(
                    _idle(env, 1, assist=assist, total=total), RECORDINGS_DIR / f"{tag}_after_clear.png"
                )
            elif action == "clear21":
                report["patra_clear"] = _clear_room21(env, total=total, assist=assist)
                report["after_clear"] = dest_report(read_snapshot(env.get_ram()))
            elif action == "probe_clear":
                do_probe("north_probe_cleared" if plan.probe_hold == "UP" else "south_probe_cleared")
            elif action == "bomb":
                assert plan.bomb_wall is not None
                bomb = loops.run_bomb_west(
                    env, total=total, assist=assist, wall=plan.bomb_wall,
                    dest=plan.bomb_dest or 0, stand_timeout=plan.bomb_timeout,
                )
                after, dest_obs = bomb.pop("after_bomb_obs"), bomb.pop("obs")
                report["bomb_west"] = bomb
                if after is not None:
                    report["after_bomb_png"] = _png(after, RECORDINGS_DIR / f"{tag}_after_bomb.png")
                if dest_obs is not None:
                    report["dest_png"] = _png(dest_obs, RECORDINGS_DIR / f"{tag}_dest.png")
                dest_snap = read_snapshot(env.get_ram())
                report["dest_screen"] = int(dest_snap.screen)
                report["dest_mode"] = int(dest_snap.mode)
                report[plan.lands_key] = int(dest_snap.screen) == (plan.bomb_dest or -1) and dest_snap.mode == PLAY_MODE
                report["dest_objects"] = dest_report(dest_snap)["objects"]
            elif action == "bomb_after":
                assert plan.bomb_wall is not None and plan.probe_dest is not None
                _clear_combat(env, plan.probe_dest, total=total, assist=assist)
                bomb = loops.run_bomb_west(
                    env, total=total, assist=assist, wall=plan.bomb_wall,
                    dest=plan.bomb_dest or 0, stand_timeout=plan.bomb_timeout,
                )
                bomb.pop("after_bomb_obs", None)
                dest_obs = bomb.pop("obs", None)
                report["bomb_west_after_south"] = bomb
                if dest_obs is not None:
                    report["bomb_west_dest_png"] = _png(dest_obs, RECORDINGS_DIR / f"{tag}_bomb_west_dest.png")
                bomb_snap = read_snapshot(env.get_ram())
                report["west_bomb_still_works"] = int(bomb_snap.screen) == (plan.bomb_dest or -1)
                report["bomb_dest_screen"] = int(bomb_snap.screen)
            elif action == "stairs30":
                dest = loops.take_stairs_from_source(env, ROOM30, total=total, cellar_side="right", assist=assist)
                report["walk"] = dest
                idle = _idle(env, 1, assist=assist, total=total)
                report["after_walk_png"] = _png(idle, RECORDINGS_DIR / f"{tag}_after_walk.png")
                idle, snap = _wait_play_room(env, ROOM04, total=total, assist=assist)
                report["dest_png"] = _png(idle, RECORDINGS_DIR / f"{tag}_dest.png")
                report["dest_screen"] = int(snap.screen)
                report["dest_mode"] = int(snap.mode)
                report["lands_0x04"] = int(snap.screen) == ROOM04 and snap.mode == PLAY_MODE
                report["entered_cellar_67"] = bool(dest.get("passage_entered") or int(snap.screen) == 0x67)
                report["walk_log"] = dest.get("log")
                if report["lands_0x04"]:
                    report["stair_tile"] = 0x72
                    report["stair_xy"] = [ROOM30_STAIR_X, ROOM30_STAIR_Y]
                    report["cellar_room"] = "0x67"
                    report["west_still_bomb"] = rom_pair(ROOM04, ROOM03, "w") == (4, 4)
                    report["can_reuse_compose_04"] = True
            elif action == "stairs_if_30" and report.get("lands_0x30"):
                has_block = any(int(obj.get("type_id") or 0) == 0x68 for obj in (report.get("dest_objects") or []))
                report["block_0x68_present"] = has_block
                dest = loops.take_stairs_from_source(env, ROOM30, total=total, cellar_side="right", assist=assist)
                report["stairs_probe"] = dest
                idle, snap = _wait_play_room(env, ROOM04, total=total, assist=assist)
                report["stairs_still_work"] = has_block and rom_secret(ROOM30) == 5 and (
                    int(snap.screen) in (ROOM04, 0x67) or bool(dest.get("passage_entered"))
                )
                report["stairs_dest_screen"] = int(snap.screen)
                report["stairs_dest_png"] = _png(
                    idle or _idle(env, 1, assist=assist, total=total), RECORDINGS_DIR / f"{tag}_stairs_dest.png"
                )
            elif action == "poke" and plan.stair_poke and report.get(plan.lands_key):
                px, py = plan.stair_poke
                _assign(env, ADDR_LINK_X, px)
                _assign(env, ADDR_LINK_Y, py)
                _idle(env, 1, assist=None, total=total)
                stair = read_snapshot(env.get_ram())
                rec = {"x": px, "y": py, "tile": int(stair.colliding_tile), "screen": int(stair.screen)}
                if (px, py) == (ROOM03_STAIR_X, ROOM03_STAIR_Y):
                    report["stair_tile_at_03"] = {**rec, "still_0x72": int(stair.colliding_tile) == 0x72}
                else:
                    report["stair_tile_at_30"] = {**rec, "still_0x73": int(stair.colliding_tile) == 0x73}
            elif action == "no_poke":
                obs, loader_np, loaded_np = remat(door_staging=False)
                report["no_door_poke_settle"] = {
                    "loaded": loaded_np, "loader": loader_np.label,
                    "final": compact_snapshot(read_snapshot(env.get_ram())),
                }
                if loaded_np:
                    report["no_door_poke_settle"]["settled"] = dest_report(read_snapshot(env.get_ram()))
                    report["no_door_poke_settle"]["png"] = _png(obs, RECORDINGS_DIR / f"{tag}_no_door_poke_settle.png")

        if plan.probe_dest is not None:
            uncleared = report.get("north_probe_uncleared") or report.get("south_probe_uncleared") or {}
            cleared = report.get("north_probe_cleared") or report.get("south_probe_cleared") or {}
            entered = bool(
                uncleared.get(f"entered_0x{plan.probe_dest:02x}")
                or cleared.get(f"entered_0x{plan.probe_dest:02x}")
                or report.get(plan.lands_key)
            )
            report[plan.lands_key] = bool(entered)
            door_side = "north" if plan.probe_hold == "UP" else "south"
            if plan.how_key:
                report[plan.how_key] = (
                    "already_open" if entered and (report["settled"]["doors"].get(door_side))
                    else ("open_walk" if entered and plan.probe_hold == "UP" else "shutter_after_clear" if entered else "sealed")
                )
            if not entered:
                last = cleared or uncleared
                report["dest_screen"] = last.get("end_room")
                report["dest_mode"] = last.get("end_mode")
                if plan.next_candidate:
                    report["next_candidate"] = plan.next_candidate
        elif plan.next_candidate and not report.get(plan.lands_key):
            report["next_candidate"] = plan.next_candidate
        report["ok"] = bool(loaded)
        return report
    finally:
        env.close()


def dump_room_13(*, tag: str = "l9_room13_dump") -> dict[str, Any]:
    dumped = dump_room(DumpPlan(
        ROOM13, (ROOM13, ROOM03), extra={"how_up_opens": "sealed_wall", "clean_walk": False, "clean_predecessor": False},
        probe_dest=ROOM03, probe_hold="UP", probe_south_y=None,
        actions=("probe", "remat", "clear", "probe_clear", "no_poke"),
        lands_key="entered_0x03", how_key="how_up_opens",
    ), tag=tag)
    dumped["clean_walk"] = bool(dumped.get("entered_0x03") and (dumped.get("no_door_poke_settle") or {}).get("loaded"))
    dumped["disproof"] = "0x13 north is ROM wall (1); 0x03 south is ROM wall (1)."
    return dumped


def dump_room_04(*, tag: str = "l9_room04_dump") -> dict[str, Any]:
    return dump_room(DumpPlan(
        ROOM04, (ROOM04, ROOM03), selected_item=B_ITEM_BOMBS,
        extra={
            "rom_neighbors_of_03": room03_rom_neighbors(),
            "rom_west_is_bomb": rom_pair(ROOM04, ROOM03, "w") == (4, 4),
            "rom_predecessor": rom_pair(ROOM04, ROOM03, "w") == (4, 4),
            "bomb_stand": list(BOMB_WEST_STAND), "selected_item_fixture": B_ITEM_BOMBS, "door_poke_on_03": False,
        },
        actions=("tiles", "remat", "clear", "bomb", "poke", "no_poke"),
        bomb_wall=BOMB_WALL_04_WEST, bomb_dest=ROOM03, stair_poke=(ROOM03_STAIR_X, ROOM03_STAIR_Y),
        lands_key="lands_0x03",
        next_candidate="0x04 west did not land 0x03. Next real candidate is cellar 0x67 / play 0x30.",
    ), tag=tag)


def dump_room_30(*, tag: str = "l9_room30_dump") -> dict[str, Any]:
    pair = cellar_for_play_room(ROOM30)
    return dump_room(DumpPlan(
        ROOM30, (ROOM30, ROOM40, ROOM04),
        extra={
            "rom_secret_block_stairs": rom_secret(ROOM30) == 5, "loader_avoids_04": loader_avoids(ROOM30, ROOM04),
            "cellar_pair": None if pair is None else {"cellar": f"0x{pair[0]:02X}", "mouth": pair[1]},
            "hypothesized_right_dest": "0x04", "door_poke_on_04": False,
        },
        actions=("tiles", "remat", "stairs30"), lands_key="lands_0x04",
        next_candidate="0x30 / cellar 0x67 right dest is not 0x04.",
    ), tag=tag)


def dump_room_40(*, tag: str = "l9_room40_dump") -> dict[str, Any]:
    dumped = dump_room(DumpPlan(
        ROOM40, (ROOM40, ROOM30, ROOM20, ROOM31),
        extra={
            "rom_neighbors_of_30": room30_rom_neighbors(),
            "rom_north_is_key": rom_pair(ROOM40, ROOM30, "n") == (5, 5),
            "rom_predecessor": rom_pair(ROOM40, ROOM30, "n") == (5, 5),
            "loader_avoids_30": loader_avoids(ROOM40, ROOM30), "door_poke_on_30": False,
            "cellar_67_is_successor_not_pred": True,
        },
        probe_dest=ROOM30, probe_hold="UP",
        actions=("tiles", "remat", "probe", "remat", "clear", "probe_clear", "poke", "no_poke"),
        stair_poke=(ROOM30_STAIR_X, ROOM30_STAIR_Y), lands_key="lands_0x30", how_key="how_up_opens",
        next_candidate="0x40 north did not land 0x30. Next real candidate is 0x31 west bomb.",
    ), tag=tag)
    dumped["clean_walk"] = bool(dumped.get("lands_0x30"))
    if dumped.get("dest_objects"):
        dumped["block_stairs_still_works"] = any(int(o.get("type_id") or 0) == 0x68 for o in dumped["dest_objects"])
    return dumped


def dump_room_31(*, tag: str = "l9_room31_dump") -> dict[str, Any]:
    return dump_room(DumpPlan(
        ROOM31, (ROOM31, ROOM30, ROOM41), selected_item=B_ITEM_BOMBS,
        extra={
            "rom_neighbors_of_30": room30_rom_neighbors(),
            "rom_west_is_bomb": rom_pair(ROOM31, ROOM30, "w") == (4, 4),
            "rom_predecessor": rom_pair(ROOM31, ROOM30, "w") == (4, 4),
            "loader_avoids_30": loader_avoids(ROOM31, ROOM30),
            "bomb_stand": list(BOMB_WEST_STAND), "selected_item_fixture": B_ITEM_BOMBS, "door_poke_on_30": False,
        },
        actions=("tiles", "remat", "clear", "bomb", "stairs_if_30", "no_poke"),
        bomb_wall=BOMB_WALL_31_WEST, bomb_dest=ROOM30, bomb_timeout=4000, lands_key="lands_0x30",
        next_candidate="0x31 west did not land 0x30.",
    ), tag=tag)


def dump_room_21(*, tag: str = "l9_room21_dump") -> dict[str, Any]:
    return dump_room(DumpPlan(
        ROOM21, (ROOM21, ROOM31, ROOM11), selected_item=B_ITEM_BOMBS,
        extra={
            "rom_south_is_shutter": rom_pair(ROOM21, ROOM31, "s") == (7, 0),
            "rom_predecessor": rom_pair(ROOM21, ROOM31, "s") == (7, 0),
            "loader_avoids_31": loader_avoids(ROOM21, ROOM31),
            "door_poke_on_31": False, "selected_item_fixture": B_ITEM_BOMBS,
        },
        probe_dest=ROOM31, probe_hold="DOWN", probe_west_band=ROOM21_WEST_X,
        actions=("tiles", "remat", "probe", "remat", "clear21", "probe_clear", "bomb_after", "no_poke"),
        bomb_wall=BOMB_WALL_31_WEST, bomb_dest=ROOM30, bomb_timeout=4000,
        lands_key="lands_0x31", how_key="how_south_opens",
        next_candidate="0x21 south shutter stays sealed after Patra kill. Next clean 0x31 entry: play 0x41 north.",
    ), tag=tag)


def dump_room_41(*, tag: str = "l9_room41_dump") -> dict[str, Any]:
    return dump_room(DumpPlan(
        ROOM41, (ROOM41, ROOM31, ROOM51),
        extra={
            "rom_north_is_open": rom_pair(ROOM41, ROOM31, "n") == (0, 7),
            "rom_predecessor": rom_pair(ROOM41, ROOM31, "n") == (0, 7),
            "loader_avoids_31": loader_avoids(ROOM41, ROOM31), "door_poke_on_31": False,
        },
        probe_dest=ROOM31, probe_hold="UP",
        actions=("tiles", "remat", "probe", "remat", "clear", "probe_clear", "no_poke"),
        lands_key="lands_0x31", how_key="how_north_opens",
        next_candidate="0x41 north stays sealed into 0x31. Do not treat 0x40 as next.",
    ), tag=tag)


def run_to_credits(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _loops().run_to_credits(*args, **kwargs)


def run_room04_bomb_west_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_room04_bomb_west_to_credits(**kwargs)


def run_room30_stairs_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_room30_stairs_to_credits(**kwargs)


def run_room40_key_north_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_room40_key_north_to_credits(**kwargs)


def run_room31_bomb_west_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_room31_bomb_west_to_credits(**kwargs)


def run_room21_south_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_room21_south_to_credits(**kwargs)


def run_play_source_to_credits(**kwargs: Any) -> dict[str, Any]:
    return _loops().run_play_source_to_credits(**kwargs)


def probe_sources(**kwargs: Any) -> dict[str, Any]:
    return _loops().probe_sources(**kwargs)


def probe_cellar_dest_table(**kwargs: Any) -> dict[str, Any]:
    return _loops().probe_cellar_dest_table(**kwargs)


def dump_play_rooms(**kwargs: Any) -> dict[str, Any]:
    return _loops().dump_play_rooms(**kwargs)


# Re-exports so probe scripts / CLI can keep importing session names.
def take_stairs_from_source(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _loops().take_stairs_from_source(*args, **kwargs)


def dump_room_tiles(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _loops().dump_room_tiles(*args, **kwargs)


def enter_patra_via_source_cellar(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _loops().enter_patra_via_source_cellar(*args, **kwargs)
