"""Level 9 stair-taking env loops (cellar, bomb-west, play-room walk)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level2_puzzles import BombWall
from zelda_i.level9_ganon import ADDR_SELECTED_ITEM, B_ITEM_ARROWS, B_ITEM_BOMBS
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import (
    BLACK_MOUTH_TILE,
    BLOCK_PUSH_STANDS,
    BOMB_WALL_04_WEST,
    BOMB_WALL_31_WEST,
    BOMB_WEST_STAND,
    CELLAR_MODE,
    ITEM_CELLAR_MODE,
    LEVEL9_CELLAR_ROOMS,
    LEVEL9_STAIR_PAIRS,
    PATRA_STAIR_SOURCE,
    PLAY_STAIR_CANDIDATES,
    ROOM03,
    ROOM04,
    ROOM21,
    ROOM21_WEST_X,
    ROOM30,
    ROOM31,
    ROOM40,
    ROOM41,
    STAIR_STANDS,
    STAIR_TILE_HI,
    STAIR_TILE_LO,
    bomb_west_approach_step,
    cellar_dest_for,
    cellar_exit_step,
    cellar_for_play_room,
    cellar_mouth_xy,
    chase_sword_step,
    dest_report,
    in_cellar_67,
    in_patra_cellar,
    in_stair_source,
    landed_final_patra,
    live_combat_objects,
    make_bomb_west_controller,
    on_warp_tile,
    paired_stair_dest,
    play_rooms_entering_cellar,
    room03_like_like_blocks_push,
    room03_stairs_step,
    room03_west_block_pushed,
    room30_stairs_step,
    stair_loader_for,
    stair_transition_modes,
    take_stairs_step,
    walk_to_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_LINK_X, ADDR_LINK_Y, ADDR_MODE, PLAY_MODE, read_snapshot

# Env primitives live in the session module (scripts import _step/_idle there).
from zelda_i.level9_stair_session import (  # noqa: E402
    ADDR_SUBMODE,
    ADDR_UPDATING,
    ADDR_UW_EXIT_TYPE,
    BEAD,
    CLEAR_MAX_FRAMES,
    FIXTURE_SOURCE,
    NORTH_PROBE_FRAMES,
    PUSH_FRAMES,
    TAG,
    WALK_MAX_FRAMES,
    _assign,
    _clear_combat,
    _clear_room21,
    _door_step,
    _exit_cellar,
    _idle,
    _left_source,
    _loader_write_rows,
    _new_env,
    _pause_select_arrows,
    _png,
    _seed_west_south_03,
    _step,
    _wait_play_room,
    _walk_target,
    materialize_stair_room,
    probe_room,
)


def take_stairs_from_source(
    env: Any,
    source: int,
    *,
    total: list[int],
    cellar_side: str = "left",
    assist: Any = None,
    chase_types: tuple[int, ...] | None = None,
    clear_frames: int | None = None,
    room03_chase_mode: str = "early_clear",
    chase_y_min: int | None = None,
) -> dict[str, Any]:
    log: list[str] = []
    obs = None

    def current() -> Any:
        return read_snapshot(env.get_ram())

    def note(label: str) -> None:
        snap = current()
        block = next((o for o in snap.objects if o.type_id == 0x68), None)
        by = f" block=({block.x},{block.y})" if block is not None else ""
        log.append(
            f"{label}: room=0x{snap.screen:02x} mode={snap.mode} "
            f"xy=({snap.link_x},{snap.link_y}) tile=0x{snap.colliding_tile:02x}{by}"
        )

    def saw_stairs(snap: Any) -> bool:
        if int(source) == 0x03:
            return stair_transition_modes(snap.mode) or in_patra_cellar(snap)
        if int(source) == 0x30:
            return stair_transition_modes(snap.mode) or in_cellar_67(snap)
        return stair_transition_modes(snap.mode) or on_warp_tile(snap) or in_patra_cellar(snap)

    if int(source) not in (0x03, 0x30):
        for x, y in STAIR_STANDS[:7]:
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            for _ in range(WALK_MAX_FRAMES):
                snap = current()
                if saw_stairs(snap) or _left_source(snap, source):
                    break
                frame = take_stairs_step(snap, source=source, target=(x, y), push=False)
                obs = _step(env, frame.action, assist=assist, total=total)
                if frame.reason in {"on_stair_tile", "stand_on_stairs"}:
                    obs = _idle(env, 20, assist=assist, total=total)
                    break
            else:
                continue
            break
    note("after_visible_stairs")

    cooldown = 0
    clear_budget = CLEAR_MAX_FRAMES if clear_frames is None else int(clear_frames)
    for _ in range(clear_budget):
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        combat = live_combat_objects(snap)
        if chase_types is not None:
            combat = tuple(obj for obj in combat if obj.type_id in chase_types)
        if in_stair_source(snap, source) and not combat:
            break
        if chase_y_min is not None and int(snap.link_y) < int(chase_y_min):
            frame = walk_to_step(snap, 120, 189, y_first=True)
        else:
            frame, cooldown = chase_sword_step(snap, cooldown, types=chase_types)
        obs = _step(env, frame.action, assist=assist, total=total)
    if clear_budget:
        _idle(env, 12, assist=assist, total=total)
    note("after_clear")

    if int(source) == 0x03:
        cooldown = 0
        for i in range(5000):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            likes = tuple(obj for obj in live_combat_objects(snap) if obj.type_id == 0x17)
            grabbed = any(
                abs(int(obj.x) - snap.link_x) <= 8 and abs(int(obj.y) - snap.link_y) <= 8
                for obj in likes
            )
            combat = live_combat_objects(snap)
            if chase_types is not None:
                combat = tuple(obj for obj in combat if obj.type_id in chase_types)
            if room03_chase_mode == "grabbed":
                should_chase, chase_filter = grabbed, (0x17,)
            elif room03_chase_mode == "blocking":
                should_chase = room03_like_like_blocks_push(snap) and not room03_west_block_pushed(snap)
                chase_filter = (0x17,)
            else:
                should_chase = (grabbed or (combat and i < 900)) and not room03_west_block_pushed(snap)
                chase_filter = chase_types
            if should_chase:
                if chase_y_min is not None and int(snap.link_y) < int(chase_y_min):
                    frame = walk_to_step(snap, 120, 189, y_first=True)
                else:
                    frame, cooldown = chase_sword_step(snap, cooldown, types=chase_filter)
            else:
                frame = room03_stairs_step(snap)
            obs = _step(env, frame.action, assist=assist, total=total)
            if getattr(frame, "reason", "") == "stand_on_03_stairs":
                obs = _idle(env, 8, assist=assist, total=total)
                if saw_stairs(current()):
                    break
        note("after_room03_stairs")

    if int(source) == 0x30:
        for _ in range(4000):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            frame = room30_stairs_step(snap)
            obs = _step(env, frame.action, assist=assist, total=total)
            if getattr(frame, "reason", "") == "stand_on_30_stairs":
                obs = _idle(env, 12, assist=assist, total=total)
                if saw_stairs(current()):
                    break
        note("after_room30_stairs")

    skip_generic = int(source) in (0x03, 0x30)
    for x, y in BLOCK_PUSH_STANDS:
        if skip_generic:
            break
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        obs, snap = _walk_target(env, total, x, y)
        if saw_stairs(snap) or _left_source(snap, source):
            break
        for _ in range(PUSH_FRAMES):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            obs = _step(env, nes_action("LEFT"), assist=None, total=total)
        else:
            continue
        break
    note("after_block_push")

    for x, y in STAIR_STANDS:
        if skip_generic:
            break
        snap = current()
        if saw_stairs(snap) or _left_source(snap, source):
            break
        for _ in range(WALK_MAX_FRAMES):
            snap = current()
            if saw_stairs(snap) or _left_source(snap, source):
                break
            frame = take_stairs_step(snap, source=source, target=(x, y), push=False)
            obs = _step(env, frame.action, assist=None, total=total)
            if frame.reason in {"on_stair_tile", "stand_on_stairs"}:
                obs = _idle(env, 20, assist=None, total=total)
                break
        else:
            continue
        break
    note("after_stair_stands")

    snap = current()
    if (not skip_generic) and in_stair_source(snap, source) and not saw_stairs(snap):
        for y in (0x60, 0x80, 0x90, 0xA0, 0xB0):
            for x in (0x40, 0x60, 0x78, 0x90, 0xB0, 0xD0):
                snap = current()
                if saw_stairs(snap) or _left_source(snap, source):
                    break
                obs, snap = _walk_target(env, total, x, y, frames=220)
                if saw_stairs(snap) or _left_source(snap, source):
                    break
            else:
                continue
            break
        note("after_grid")

    snap = current()
    passage_entered = bool(stair_transition_modes(snap.mode) or snap.mode == CELLAR_MODE)
    if passage_entered or snap.mode in (CELLAR_MODE, ITEM_CELLAR_MODE, 10, 16):
        obs, snap = _exit_cellar(env, total=total, side=cellar_side)
        note(f"after_cellar_{cellar_side}")

    for _ in range(400):
        snap = current()
        if snap.mode == PLAY_MODE and not snap.transitioning:
            break
        if snap.mode == CELLAR_MODE and not snap.transitioning:
            obs = _step(env, cellar_exit_step(snap, side=cellar_side).action, assist=assist, total=total)
        else:
            obs = _step(env, nes_action("UP"), assist=assist, total=total)
    if snap.mode == PLAY_MODE:
        obs = _idle(env, 24, assist=None, total=total)
        snap = current()

    result = dest_report(snap)
    result["source"] = int(source)
    result["paired_hypothesis"] = paired_stair_dest(source)
    result["passage_entered"] = passage_entered or stair_transition_modes(current().mode)
    result["log"] = log
    result["ok_left_source"] = bool(_left_source(snap, source) or snap.screen != source)
    return result


def _cellar_write_rows(source: int, side: str) -> list[dict[str, Any]]:
    mouth_x, mouth_y = cellar_mouth_xy(side=side)
    return [
        {"name": "init_mode9_cellar", "address": ADDR_MODE, "address_hex": "0x0012", "value": 9,
         "note": "lets the engine run InitMode9 (fade + LayoutCellar)"},
        {"name": "init_submode", "address": ADDR_SUBMODE, "address_hex": "0x0013", "value": 0},
        {"name": "init_is_updating_mode", "address": ADDR_UPDATING, "address_hex": "0x0011", "value": 0},
        {"name": "underground_exit_type", "address": ADDR_UW_EXIT_TYPE, "address_hex": "0x005A", "value": 0},
        {
            "name": "cellar_mouth_stand",
            "addresses": [ADDR_LINK_X, ADDR_LINK_Y], "address_hex": ["0x0070", "0x0084"],
            "values": [mouth_x, mouth_y], "side": side,
            "checksubroom_dest": cellar_dest_for(source, side=side),
        },
    ]


def enter_patra_via_source_cellar(
    env: Any, source: int, *, total: list[int], side: str = "left",
) -> dict[str, Any]:
    _assign(env, ADDR_MODE, CELLAR_MODE)
    _assign(env, ADDR_SUBMODE, 0)
    _assign(env, ADDR_UPDATING, 0)
    _assign(env, ADDR_UW_EXIT_TYPE, 0)
    for i in range(400):
        _step(env, nes_idle_action(), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == CELLAR_MODE and int(env.get_ram()[ADDR_UPDATING]) != 0 and i > 20:
            break
    mouth_x, mouth_y = cellar_mouth_xy(side=side)
    _assign(env, ADDR_LINK_X, mouth_x)
    _assign(env, ADDR_LINK_Y, mouth_y)
    _idle(env, 4, assist=None, total=total)
    expected = cellar_dest_for(source, side=side)
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if expected is not None and snap.screen == expected:
            break
        _step(env, nes_action("UP"), assist=None, total=total)
    for _ in range(400):
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and expected is not None and snap.screen == expected and not snap.transitioning:
            _idle(env, 24, assist=None, total=total)
            break
        _step(env, nes_idle_action(), assist=None, total=total)
    snap = read_snapshot(env.get_ram())
    result = dest_report(snap)
    result.update({"source": int(source), "cellar_side": side, "expected_dest": expected, "passage_entered": True})
    return result


def dump_room_tiles(env: Any, *, total: list[int]) -> dict[str, Any]:
    hits: list[dict[str, int]] = []
    mouths: list[dict[str, int]] = []
    counts: Counter[int] = Counter()
    grid: list[list[int]] = []
    for y in range(0x4D, 0xDE, 8):
        row: list[int] = []
        for x in range(0x20, 0xE1, 8):
            _assign(env, ADDR_LINK_X, x)
            _assign(env, ADDR_LINK_Y, y)
            _step(env, nes_idle_action(), assist=None, total=total)
            snap = read_snapshot(env.get_ram())
            tile = int(snap.colliding_tile)
            counts[tile] += 1
            row.append(tile)
            rec = {"x": int(snap.link_x), "y": int(snap.link_y), "tile": tile}
            if STAIR_TILE_LO <= tile <= STAIR_TILE_HI:
                hits.append(rec)
            if tile == BLACK_MOUTH_TILE:
                mouths.append(rec)
        grid.append(row)
    return {
        "stair_hits": hits, "mouth_hits": mouths,
        "tile_counts": {f"0x{t:02X}": n for t, n in counts.most_common(12)},
        "grid_origin": [0x20, 0x4D], "grid_step": 8, "grid": grid,
    }


def run_bomb_west(
    env: Any, *, total: list[int], assist: Any, wall: BombWall, dest: int,
    stand_timeout: int | None = None,
) -> dict[str, Any]:
    obs = None
    for _ in range(WALK_MAX_FRAMES + 400):
        snap = read_snapshot(env.get_ram())
        if not in_stair_source(snap, wall.room):
            break
        frame = bomb_west_approach_step(snap)
        if abs(int(snap.link_x) - BOMB_WEST_STAND[0]) <= 4 and abs(int(snap.link_y) - BOMB_WEST_STAND[1]) <= 4:
            break
        obs = _step(env, frame.action, assist=assist, total=total)
    ctrl = make_bomb_west_controller(wall, stand_timeout=stand_timeout)
    after_bomb_obs = None
    for _ in range(ctrl.max_frames):
        snap = read_snapshot(env.get_ram())
        frame = ctrl.step(snap)
        obs = _step(env, frame.action, assist=assist, total=total)
        if after_bomb_obs is None and ctrl.phase.name == "PUSH":
            after_bomb_obs = obs
        if ctrl.success or ctrl.phase.name in {"DONE", "FAILED"}:
            break
    snap = read_snapshot(env.get_ram())
    return {
        "controller": ctrl.report(), "dest": dest_report(snap),
        f"entered_0x{dest:02x}": int(snap.screen) == dest,
        "still_in_source": bool(in_stair_source(snap, wall.room)),
        "obs": obs, "after_bomb_obs": after_bomb_obs,
    }


def walk_play_room_to_patra(
    env: Any, source: int, *, total: list[int], cellar_side: str = "left", assist: Any = None,
    chase_types: tuple[int, ...] | None = None, clear_frames: int | None = None,
    room03_chase_mode: str = "early_clear", chase_y_min: int | None = None,
) -> dict[str, Any]:
    dest = take_stairs_from_source(
        env, source, total=total, cellar_side=cellar_side, assist=assist,
        chase_types=chase_types, clear_frames=clear_frames,
        room03_chase_mode=room03_chase_mode, chase_y_min=chase_y_min,
    )
    snap = read_snapshot(env.get_ram())
    dest["entered_cellar_77"] = bool(in_patra_cellar(snap) or dest.get("passage_entered"))
    pair = cellar_for_play_room(source)
    dest["cellar_for_source"] = None if pair is None else {"cellar": f"0x{pair[0]:02X}", "mouth": pair[1]}
    return dest


@dataclass
class CreditsPlan:
    start_room: int
    via: str
    tag: str
    steps: tuple[str, ...]
    selected_item: int | None = B_ITEM_BOMBS
    checkpoint: str = ""
    phase: str = ""
    start_state: str = ""
    room03_chase_mode: str = "early_clear"
    chase_y_min: int | None = None
    source_label: str = ""


def run_to_credits(
    plan: CreditsPlan, *, infinite_life: bool = True, save_checkpoints: bool = False,
    tag: str | None = None, trial_i: int = 0,
) -> dict[str, Any]:
    from zelda_i.level9_stair_suffix import run_suffix_from_live_env
    from zelda_i.scripts.run_level9_ganon import _save_checkpoint

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    tag = tag or plan.tag
    writes = _loader_write_rows(stair_loader_for(plan.start_room))
    if plan.selected_item is not None:
        writes.append({
            "name": "selected_item_bombs", "address": ADDR_SELECTED_ITEM,
            "address_hex": "0x0656", "value": plan.selected_item,
        })
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False,
        "fixture_only": True, "init_mode9": False,
        "source_room": plan.source_label or f"0x{plan.start_room:02X}",
        "via": plan.via, "trial": trial_i, "tag": tag, "fixture_writes": writes,
        "runtime_controller_writes": {
            "object": 0, "room": 0, "door": 0, "inventory": 0, "progression": 0, "capacity": 0,
        },
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, plan.start_room, total=total, selected_item=plan.selected_item,
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = f"loader did not settle 0x{plan.start_room:02X}"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        report["settle_png"] = _png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png")

        def fail(msg: str) -> dict[str, Any]:
            report["error"] = msg
            return report

        for step in plan.steps:
            if step == "clear21":
                report["patra_clear"] = _clear_room21(env, total=total, assist=assist)
            elif step.startswith("clear"):
                room = {"clear31": ROOM31, "clear41": ROOM41, "clear04": ROOM04}.get(step, plan.start_room)
                _clear_combat(env, room, total=total, assist=assist, frames=800 if step == "clear04" else CLEAR_MAX_FRAMES)
            elif step.startswith("north") or step.startswith("south"):
                dest = ROOM31 if "41" in step or "21" in step else ROOM30
                hold = "DOWN" if step.startswith("south") else "UP"
                source = ROOM21 if step.startswith("south") else (ROOM41 if "41" in step else ROOM40)
                west = ROOM21_WEST_X if hold == "DOWN" else None
                probed = probe_room(
                    env, _door_step(source, dest, hold, west_band=west), dest, NORTH_PROBE_FRAMES + 800,
                    total=total, assist=assist, hold=hold, source_room=source,
                )
                report[step] = probed
                snap = read_snapshot(env.get_ram())
                _png(_idle(env, 1, assist=assist, total=total), RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
                report["dest_screen"] = int(snap.screen)
                if int(snap.screen) != dest:
                    return fail(f"{step} dest 0x{snap.screen:02X} is not 0x{dest:02X}")
            elif step in {"bomb31", "bomb04"}:
                wall = BOMB_WALL_31_WEST if step == "bomb31" else BOMB_WALL_04_WEST
                dest = ROOM30 if step == "bomb31" else ROOM03
                bomb = run_bomb_west(
                    env, total=total, assist=assist, wall=wall, dest=dest,
                    stand_timeout=4000 if step == "bomb31" else None,
                )
                after, dest_obs = bomb.pop("after_bomb_obs", None), bomb.pop("obs", None)
                key = "bomb_west_04" if step == "bomb04" and "bomb_west" in report else "bomb_west"
                report[key] = bomb
                if after is not None:
                    _png(after, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_bomb.png")
                if dest_obs is not None:
                    _png(dest_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
                snap = read_snapshot(env.get_ram())
                report["dest_screen"] = int(snap.screen)
                if int(snap.screen) != dest:
                    return fail(f"{step} dest 0x{snap.screen:02X} is not 0x{dest:02X}")
            elif step == "stairs30":
                dest = take_stairs_from_source(
                    env, ROOM30, total=total, cellar_side="right", assist=assist, chase_y_min=plan.chase_y_min,
                )
                report["walk"] = dest
                _png(_idle(env, 1, assist=assist, total=total), RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png")
                idle, snap = _wait_play_room(env, ROOM04, total=total, assist=assist)
                report["cellar_room"] = 0x67
                if int(snap.screen) != ROOM04 or snap.mode != PLAY_MODE:
                    if idle is not None:
                        _png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_stairs_dest.png")
                    return fail(f"stairs dest 0x{snap.screen:02X} mode {snap.mode} is not play 0x04")
                _png(idle, RECORDINGS_DIR / f"{tag}_t{trial_i}_stairs_dest.png")
            elif step == "pause_arrows":
                selected, moves = _pause_select_arrows(env, total=total, assist=assist)
                report["selected_item_after_pause"] = selected
                report["pause_right_moves"] = moves
                if selected != B_ITEM_ARROWS:
                    return fail(f"pause menu left selected_item={selected}, need arrows")
            elif step == "seed03":
                _seed_west_south_03(env, total=total, assist=assist)
                report["west_south"] = dest_report(read_snapshot(env.get_ram()))
            elif step == "walk03":
                dest03 = walk_play_room_to_patra(
                    env, ROOM03, total=total, cellar_side="left", assist=assist,
                    room03_chase_mode=plan.room03_chase_mode,
                )
                report["walk_04"] = dest03
                snap = read_snapshot(env.get_ram())
                idle_obs = _idle(env, 1, assist=None, total=total)
                _png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
                if not landed_final_patra(snap):
                    return fail(
                        f"stairs settled room 0x{snap.screen:02X} mode {snap.mode} "
                        f"patra={dest03.get('final_patra_live')}"
                    )
                if save_checkpoints and plan.checkpoint:
                    path = _save_checkpoint(
                        env, plan.checkpoint, source_state=FIXTURE_SOURCE, phase=plan.phase,
                        result={
                            "ok": True, "source_room": plan.start_room, "via": plan.via,
                            "room": 0x52, "final_patra_live": True, "frames": total[0],
                        },
                        fixture_writes=writes, bead=BEAD,
                    )
                    report["checkpoint_path"] = str(path)
            elif step == "suffix":
                suffix = run_suffix_from_live_env(
                    env, assist=assist, total=total, tag=tag, trial_i=trial_i, start_state=plan.start_state,
                )
                report["suffix"] = suffix
                report["ok"] = bool(suffix.get("ok"))
                report["credits_reached"] = suffix.get("credits_reached")
                report["total_frames"] = total[0]
                if not report["ok"]:
                    report["error"] = suffix.get("error") or "suffix failed"
        return report
    finally:
        env.close()


def run_room04_bomb_west_to_credits(**kwargs: Any) -> dict[str, Any]:
    return run_to_credits(CreditsPlan(
        ROOM04, "bomb_west", "l9_play04_bombwest_patra_credits_recon",
        ("clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
        checkpoint="Level9Room04BombWestReconFixture", phase="play_0x04_bomb_west_into_live_patra",
        start_state="play_0x04_bomb_west_walk",
    ), **kwargs)


def run_room30_stairs_to_credits(**kwargs: Any) -> dict[str, Any]:
    return run_to_credits(CreditsPlan(
        ROOM30, "cellar_0x67_right", "l9_play30_cellar67_patra_credits_recon",
        ("stairs30", "clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
        checkpoint="Level9Room30StairsReconFixture", phase="play_0x30_cellar_0x67_right_into_live_patra",
        start_state="play_0x30_cellar_0x67_right_walk",
    ), **kwargs)


def run_room40_key_north_to_credits(**kwargs: Any) -> dict[str, Any]:
    return run_to_credits(CreditsPlan(
        ROOM40, "key_north", "l9_play40_keynorth_patra_credits_recon",
        ("north40", "stairs30", "clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
        checkpoint="Level9Room40KeyNorthReconFixture", phase="play_0x40_key_north_into_live_patra",
        start_state="play_0x40_key_north_walk", room03_chase_mode="blocking",
    ), **kwargs)


def run_room31_bomb_west_to_credits(*, from_41: bool = False, **kwargs: Any) -> dict[str, Any]:
    if from_41:
        return run_to_credits(CreditsPlan(
            ROOM41, "north_then_bomb_west", "l9_play41_north_patra_credits_recon",
            ("clear41", "north41", "clear31", "bomb31", "stairs30", "clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
            checkpoint="Level9Room41NorthReconFixture", phase="play_0x41_north_into_live_patra",
            start_state="play_0x41_north_walk", chase_y_min=180, source_label="0x41",
        ), **kwargs)
    return run_to_credits(CreditsPlan(
        ROOM31, "bomb_west", "l9_play31_bombwest_patra_credits_recon",
        ("clear31", "bomb31", "stairs30", "clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
        checkpoint="Level9Room31BombWestReconFixture", phase="play_0x31_bomb_west_into_live_patra",
        start_state="play_0x31_bomb_west_walk",
    ), **kwargs)


def run_room21_south_to_credits(**kwargs: Any) -> dict[str, Any]:
    return run_to_credits(CreditsPlan(
        ROOM21, "south_shutter", "l9_play21_south_patra_credits_recon",
        ("clear21", "south21", "clear31", "bomb31", "stairs30", "clear04", "bomb04", "pause_arrows", "seed03", "walk03", "suffix"),
        checkpoint="Level9Room21SouthReconFixture", phase="play_0x21_south_into_live_patra",
        start_state="play_0x21_south_walk",
    ), **kwargs)


def run_play_source_to_credits(
    *, source: int, cellar_side: str = "left", infinite_life: bool = True,
    save_checkpoints: bool = False, tag: str = TAG, trial_i: int = 0,
) -> dict[str, Any]:
    from zelda_i.level9_stair_suffix import run_suffix_from_live_env
    from zelda_i.scripts.run_level9_ganon import _save_checkpoint

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    writes = _loader_write_rows(stair_loader_for(source))
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False,
        "fixture_only": True, "init_mode9": False, "source_room": f"0x{source:02X}",
        "cellar_side": cellar_side, "trial": trial_i, "tag": tag, "fixture_writes": writes,
        "runtime_controller_writes": {
            "object": 0, "room": 0, "door": 0, "inventory": 0, "progression": 0, "capacity": 0,
        },
    }
    try:
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = f"loader did not settle 0x{source:02X}"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        report["settle_png"] = _png(obs, settle_png)
        tiles = dump_room_tiles(env, total=total)
        tile_json = RECORDINGS_DIR / f"{tag}_t{trial_i}_0x{source:02x}_tiles.json"
        write_json_report(tile_json, tiles)
        report["tile_json"] = str(tile_json)
        report["stair_hits"] = tiles["stair_hits"]
        report["mouth_hits"] = tiles["mouth_hits"]
        hits = tiles["stair_hits"] or tiles["mouth_hits"]
        if hits:
            report["stair_tile"], report["stair_xy"] = hits[0]["tile"], [hits[0]["x"], hits[0]["y"]]
        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        if not loaded:
            report["error"] = f"rematerialize failed 0x{source:02X}"
            return report
        _png(obs, settle_png)
        dest = walk_play_room_to_patra(env, source, total=total, cellar_side=cellar_side, assist=assist)
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        idle_obs = _idle(env, 1, assist=None, total=total)
        report["after_walk_png"] = _png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png")
        if not landed_final_patra(snap):
            report["error"] = (
                f"walk from 0x{source:02X} settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest.get('final_patra_live')}"
            )
            report["final"] = dest
            return report
        report["patra_entry_png"] = _png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
        report["final_patra_live"] = True
        if save_checkpoints:
            path = _save_checkpoint(
                env, f"Level9Room{source:02X}StairsReconFixture", source_state=FIXTURE_SOURCE,
                phase=f"play_0x{source:02x}_stairs_into_live_patra",
                result={"ok": True, "source_room": source, "room": 0x52, "final_patra_live": True, "frames": total[0]},
                fixture_writes=writes, bead=BEAD,
            )
            report["checkpoint_path"] = str(path)
        suffix = run_suffix_from_live_env(
            env, assist=assist, total=total, tag=tag, trial_i=trial_i, start_state=f"play_0x{source:02x}_walk",
        )
        report["suffix"] = suffix
        report["ok"] = bool(suffix.get("ok"))
        report["credits_reached"] = suffix.get("credits_reached")
        report["total_frames"] = total[0]
        if not report["ok"]:
            report["error"] = suffix.get("error") or "suffix failed"
        return report
    finally:
        env.close()


def probe_sources(*, tag: str = f"{TAG}_probe", stop_on_patra: bool = True, cellar_sides: tuple[str, ...] = ("left", "right")) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False, "fixture_only": True,
        "rom_pairs": [[f"0x{a:02X}", f"0x{b:02X}"] for a, b in LEVEL9_STAIR_PAIRS],
        "sources": [], "winner": None,
    }
    env = _new_env()
    try:
        for room in LEVEL9_STAIR_SOURCES:
            room_row: dict[str, Any] = {
                "source": f"0x{room:02X}",
                "paired_hypothesis": None if paired_stair_dest(room) is None else f"0x{paired_stair_dest(room):02X}",
                "attempts": [],
            }
            winner_here = False
            for side in cellar_sides:
                env.close()
                env = _new_env()
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                attempt: dict[str, Any] = {
                    "cellar_side": side, "loader": loader.label, "from_room": loader.from_room,
                    "loaded": loaded, "frames": total[0],
                }
                if not loaded:
                    attempt["error"] = "loader did not settle"
                    attempt["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                    room_row["attempts"].append(attempt)
                    continue
                attempt["settled"] = dest_report(read_snapshot(env.get_ram()))
                attempt["settle_png"] = _png(obs, RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png")
                dest = take_stairs_from_source(env, room, total=total, cellar_side=side)
                attempt["dest"] = dest
                attempt["frames"] = total[0]
                dest_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_{side}_dest.png"
                attempt["dest_png"] = _png(_idle(env, 1, assist=None, total=total), dest_png)
                room_row["attempts"].append(attempt)
                if dest.get("landed_final_patra"):
                    room_row["winner"] = True
                    report["winner"] = {
                        "source": f"0x{room:02X}", "cellar_side": side, "dest": dest,
                        "settle_png": attempt["settle_png"], "dest_png": str(dest_png), "loader": loader.label,
                    }
                    winner_here = True
                    break
            report["sources"].append(room_row)
            if winner_here and stop_on_patra:
                report["ok"] = True
                break
    finally:
        env.close()
    if report["winner"] is None:
        report["error"] = "no stair source landed live final Patra 0x52"
    return report


def probe_cellar_dest_table(*, tag: str = f"{TAG}_dest_table") -> dict[str, Any]:
    rooms = list(LEVEL9_CELLAR_ROOMS) + [r for r in LEVEL9_STAIR_SOURCES if r not in LEVEL9_CELLAR_ROOMS]
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False, "fixture_only": True,
        "note": "cellar dest via InitMode9 + CheckSubroom mouth UP", "sources": [], "winner": None,
    }
    env = _new_env()
    try:
        for room in rooms:
            room_row: dict[str, Any] = {
                "source": f"0x{room:02X}", "in_cellar_array": room in LEVEL9_CELLAR_ROOMS,
                "rom_left": cellar_dest_for(room, side="left"), "rom_right": cellar_dest_for(room, side="right"),
                "attempts": [],
            }
            for side in ("left", "right"):
                env.close()
                env = _new_env()
                total = [0]
                obs, loader, loaded = materialize_stair_room(env, room, total=total)
                attempt: dict[str, Any] = {
                    "cellar_side": side, "loader": loader.label, "loaded": loaded,
                    "rom_dest": cellar_dest_for(room, side=side),
                }
                if not loaded:
                    attempt["error"] = "loader did not settle"
                    attempt["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                    room_row["attempts"].append(attempt)
                    continue
                attempt["settle_png"] = _png(obs, RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png")
                attempt["settle"] = dest_report(read_snapshot(env.get_ram()))
                dest = enter_patra_via_source_cellar(env, room, total=total, side=side)
                dest_png = RECORDINGS_DIR / f"{tag}_0x{room:02x}_{side}_dest.png"
                attempt["dest_png"] = _png(_idle(env, 1, assist=None, total=total), dest_png)
                attempt["dest"] = dest
                attempt["frames"] = total[0]
                room_row["attempts"].append(attempt)
                if dest.get("landed_final_patra") and report["winner"] is None:
                    report["winner"] = {"source": f"0x{room:02X}", "cellar_side": side, "dest": dest, "dest_png": str(dest_png)}
            report["sources"].append(room_row)
    finally:
        env.close()
    report["ok"] = report["winner"] is not None
    return report


def dump_play_rooms(*, rooms: tuple[int, ...] = PLAY_STAIR_CANDIDATES, tag: str = f"{TAG}_play_tiles") -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False, "fixture_only": True,
        "note": "play-room tile dump; CheckWarps walk is separate",
        "checkwarps_77": [{"play": f"0x{r:02X}", "mouth": side} for r, side in play_rooms_entering_cellar(PATRA_STAIR_SOURCE)],
        "rooms": [],
    }
    env = _new_env()
    try:
        for room in rooms:
            env.close()
            env = _new_env()
            total = [0]
            obs, loader, loaded = materialize_stair_room(env, room, total=total)
            pair = cellar_for_play_room(room)
            row: dict[str, Any] = {
                "room": f"0x{room:02X}", "loader": loader.label, "from_room": f"0x{loader.from_room:02X}",
                "loaded": loaded,
                "cellar_pair": None if pair is None else {"cellar": f"0x{pair[0]:02X}", "mouth": pair[1]},
            }
            if not loaded:
                row["error"] = "loader did not settle"
                row["final"] = compact_snapshot(read_snapshot(env.get_ram()))
                report["rooms"].append(row)
                continue
            row["settled"] = dest_report(read_snapshot(env.get_ram()))
            row["settle_png"] = _png(obs, RECORDINGS_DIR / f"{tag}_0x{room:02x}_settle.png")
            tiles = dump_room_tiles(env, total=total)
            row["tiles"] = {k: tiles[k] for k in ("stair_hits", "mouth_hits", "tile_counts", "grid_origin", "grid_step")}
            tile_json = RECORDINGS_DIR / f"{tag}_0x{room:02x}_tiles.json"
            write_json_report(tile_json, tiles)
            row["tile_json"] = str(tile_json)
            report["rooms"].append(row)
    finally:
        env.close()
    report["ok"] = any(r.get("loaded") for r in report["rooms"])
    return report

