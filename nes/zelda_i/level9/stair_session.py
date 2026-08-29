"""Level 9 stair session: env-step primitives, dump/probe, compose-to-credits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level2.puzzles import BombWall
from zelda_i.level9.ganon import ADDR_SELECTED_ITEM, B_ITEM_ARROWS, B_ITEM_BOMBS, LEVEL9
from zelda_i.level9.patra import PATRA_MAX_FRAMES, patra_action, patra_body, patra_eyes
from zelda_i.level9.stairs import (
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
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOOK,
    ADDR_BOOMERANG,
    ADDR_BOW,
    ADDR_BRACELET,
    ADDR_CANDLE,
    ADDR_CUR_OPENED_DOORS,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_LETTER,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MAGIC_BOOMERANG,
    ADDR_MAGIC_KEY,
    ADDR_MAGIC_SHIELD,
    ADDR_MAX_BOMBS,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_POTION,
    ADDR_RAFT,
    ADDR_RING,
    ADDR_ROD,
    ADDR_RUPEES,
    ADDR_SCREEN,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
)

BEAD = "rr-sz8.3"
TAG = "l9_stair_patra_credits_recon"
FIXTURE_SOURCE = "Level9EntranceReconFixture"
# Fully loaded first-quest inventory. Fixture writes only; never route/STATUS.
FULL_LOADOUT: tuple[tuple[str, int, int], ...] = (
    ("selected_item_silver_arrows", ADDR_SELECTED_ITEM, B_ITEM_ARROWS),
    ("magical_sword", ADDR_SWORD, 3),
    ("bombs", ADDR_BOMBS, 16),
    ("silver_arrows", ADDR_ARROWS, 2),
    ("bow", ADDR_BOW, 1),
    ("red_candle", ADDR_CANDLE, 2),
    ("whistle", ADDR_WHISTLE, 1),
    ("food", ADDR_FOOD, 1),
    ("red_potion", ADDR_POTION, 2),
    ("magic_rod", ADDR_ROD, 1),
    ("raft", ADDR_RAFT, 1),
    ("book", ADDR_BOOK, 1),
    ("red_ring", ADDR_RING, 2),
    ("ladder", ADDR_LADDER, 1),
    ("magic_key", ADDR_MAGIC_KEY, 1),
    ("bracelet", ADDR_BRACELET, 1),
    ("letter", ADDR_LETTER, 1),
    ("rupees", ADDR_RUPEES, 255),
    ("keys", ADDR_KEYS, 9),
    ("health_16_full", ADDR_HEALTH, 255),
    ("triforce_all_8", ADDR_TRIFORCE, 255),
    ("wood_boomerang", ADDR_BOOMERANG, 1),
    ("magic_boomerang", ADDR_MAGIC_BOOMERANG, 1),
    ("magic_shield", ADDR_MAGIC_SHIELD, 1),
    ("max_bombs", ADDR_MAX_BOMBS, 16),
)
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


def _inventory_snapshot(ram: Any) -> dict[str, int]:
    return {
        "selected_item": int(ram[ADDR_SELECTED_ITEM]),
        "sword": int(ram[ADDR_SWORD]),
        "bombs": int(ram[ADDR_BOMBS]),
        "arrows": int(ram[ADDR_ARROWS]),
        "bow": int(ram[ADDR_BOW]),
        "ring": int(ram[ADDR_RING]),
        "magic_key": int(ram[ADDR_MAGIC_KEY]),
        "keys": int(ram[ADDR_KEYS]),
        "health": int(ram[ADDR_HEALTH]),
        "triforce": int(ram[ADDR_TRIFORCE]),
    }


def _checkpoint_result(env: Any, total: list[int]) -> dict[str, Any]:
    return {
        "ok": True,
        "frame": total[0],
        "state": compact_snapshot(read_snapshot(env.get_ram())),
    }


def _write_provenance(
    path: Path,
    *,
    source_state: str,
    phase: str,
    result: dict[str, Any],
    fixture_writes: list[dict[str, Any]],
    bead: str = BEAD,
) -> None:
    source = state_path(GAME_DIR, GAME, source_state)
    write_state_provenance(
        path,
        source_state_path=source if source.exists() else None,
        request={
            "bead": bead,
            "phase": phase,
            "track": "recon_fixture",
            "route_eligible": False,
            "fixture_only": True,
            "fixture_writes": fixture_writes,
        },
        selected_trial=result,
        natural_entry=False,
    )


def _save_checkpoint(
    env: Any,
    name: str,
    *,
    source_state: str,
    phase: str,
    result: dict[str, Any],
    fixture_writes: list[dict[str, Any]],
    bead: str = BEAD,
) -> Path:
    path = save_state(env, GAME_DIR, GAME, name)
    _write_provenance(
        path,
        source_state=source_state,
        phase=phase,
        result=result,
        fixture_writes=fixture_writes,
        bead=bead,
    )
    return path


def _loops():
    import zelda_i.level9.stair_run as loops
    return loops


def _loader_write_rows(loader: StairLoader) -> list[dict[str, Any]]:
    rows = [
        {"name": name, "address": address, "address_hex": f"0x{address:04X}", "value": value}
        for name, address, value in FULL_LOADOUT
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
    for _, address, value in FULL_LOADOUT:
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


_LOOP_EXPORTS = frozenset({
    "run_to_credits", "run_room04_bomb_west_to_credits", "run_room30_stairs_to_credits",
    "run_room40_key_north_to_credits", "run_room31_bomb_west_to_credits",
    "run_room21_south_to_credits", "run_play_source_to_credits",
    "take_stairs_from_source", "dump_room_tiles", "enter_patra_via_source_cellar",
})
_SUFFIX_EXPORTS = frozenset({
    "dump_room_13", "dump_room_04", "dump_room_30", "dump_room_40",
    "dump_room_31", "dump_room_21", "dump_room_41",
    "probe_sources", "probe_cellar_dest_table", "dump_play_rooms",
})


def __getattr__(name: str):
    if name in _LOOP_EXPORTS:
        return getattr(_loops(), name)
    if name in _SUFFIX_EXPORTS:
        import zelda_i.level9.stair_suffix as suffix
        return getattr(suffix, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
