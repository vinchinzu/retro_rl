"""Level 9 stair-run suffix: fixture, Patra→credits, play-source compose."""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import (
    ADDR_SELECTED_ITEM,
    B_ITEM_ARROWS,
    B_ITEM_BOMBS,
    GanonFightController,
    LEVEL9,
    credits_rolling,
    final_ending_screen,
    ganon_defeated,
)
from zelda_i.level9_patra import (
    FinalPatraFightController,
    NORTH_DOOR,
    PATRA_EYE_COUNT,
    PATRA_MAX_FRAMES,
    final_patra_live,
    final_patra_north_door_earned,
    patra_action,
    patra_body,
    patra_eyes,
)
from zelda_i.level9_room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9_stairs import (
    BLACK_MOUTH_TILE,
    BLOCK_PUSH_STANDS,
    CELLAR_MODE,
    ITEM_CELLAR_MODE,
    LEVEL9_CELLAR_ROOMS,
    LEVEL9_STAIR_PAIRS,
    PATRA_STAIR_SOURCE,
    PLAY_STAIR_CANDIDATES,
    STAIR_STANDS,
    STAIR_TILE_HI,
    STAIR_TILE_LO,
    StairLoader,
    cellar_dest_for,
    cellar_exit_step,
    cellar_for_play_room,
    cellar_mouth_xy,
    in_patra_cellar,
    is_patra_cellar_source,
    chase_sword_step,
    dest_report,
    room03_stairs_step,
    room03_like_like_blocks_push,
    room03_west_block_pushed,
    in_stair_source,
    landed_final_patra,
    live_combat_objects,
    on_stair_tile,
    on_warp_tile,
    paired_stair_dest,
    play_rooms_entering_cellar,
    in_room_04,
    in_room_13,
    in_room_30,
    in_room_21,
    in_room_31,
    in_room_40,
    in_room_41,
    room30_loader_avoids_04,
    room30_rom_neighbors,
    room30_rom_secret_is_block_stairs,
    room30_stairs_step,
    room30_block_secret_open,
    in_cellar_67,
    room40_is_rom_predecessor_of_30,
    room40_loader_avoids_30,
    room40_rom_north_is_key,
    room40_to_30_step,
    make_room04_bomb_west_controller,
    make_room31_bomb_west_controller,
    pause_select_next_b_item_script,
    room04_bomb_west_approach_step,
    room31_bomb_west_approach_step,
    room21_is_rom_predecessor_of_31,
    room21_loader_avoids_31,
    room21_rom_south_is_shutter,
    room21_to_31_step,
    room41_is_rom_predecessor_of_31,
    room41_loader_avoids_31,
    room41_rom_north_is_open,
    room41_to_31_step,
    room31_is_rom_predecessor_of_30,
    room31_loader_avoids_30,
    room31_rom_west_is_bomb,
    room03_rom_neighbors,
    room04_is_rom_predecessor_of_03,
    room04_rom_west_is_bomb,
    room13_is_clean_predecessor_of_03,
    room13_to_03_step,
    ROOM03,
    ROOM03_ROM_EAST,
    ROOM03_ROM_NORTH,
    ROOM03_ROM_SOUTH,
    ROOM03_ROM_WEST,
    ROOM03_PUSH_X,
    ROOM03_PUSH_Y,
    ROOM03_STAIR_X,
    ROOM03_STAIR_Y,
    ROOM04,
    ROOM04_BOMB_WEST_STAND,
    ROOM30,
    ROOM30_ROM_EAST,
    ROOM30_ROM_NORTH,
    ROOM30_ROM_SECRET,
    ROOM30_ROM_SOUTH,
    ROOM30_ROM_WEST,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    ROOM40,
    ROOM40_ROM_NORTH,
    ROOM40_ROM_SOUTH,
    ROOM40_ROM_WEST,
    ROOM40_ROM_EAST,
    ROOM20,
    ROOM20_ROM_SOUTH,
    ROOM2F,
    ROOM11,
    ROOM21,
    ROOM21_ROM_EAST,
    ROOM21_ROM_NORTH,
    ROOM21_ROM_SOUTH,
    ROOM21_ROM_WEST,
    ROOM31,
    ROOM31_BOMB_WEST_STAND,
    ROOM31_ROM_EAST,
    ROOM31_ROM_NORTH,
    ROOM31_ROM_SOUTH,
    ROOM31_ROM_WEST,
    ROOM41,
    ROOM41_ROM_EAST,
    ROOM41_ROM_NORTH,
    ROOM41_ROM_SOUTH,
    ROOM41_ROM_WEST,
    ROOM51,
    ROOM04_ROM_EAST,
    ROOM04_ROM_NORTH,
    ROOM04_ROM_SOUTH,
    ROOM04_ROM_WEST,
    ROOM13,
    ROOM13_ROM_EAST,
    ROOM13_ROM_NORTH,
    ROOM13_ROM_SOUTH,
    ROOM13_ROM_WEST,
    stair_loader_for,
    stair_transition_modes,
    take_stairs_step,
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
from zelda_i.scripts.run_level9_ganon import (
    FIXTURE_SOURCE,
    FULL_LOADOUT,
    _assign,
    _checkpoint_result,
    _collect_power_triforce,
    _enter_ganon,
    _enter_zelda,
    _idle,
    _rescue_zelda,
    _save_checkpoint,
    _step,
)
from zelda_i.scripts.run_level9_patra import _inventory_snapshot
from zelda_i.level9_stair_run import (
    _cellar_write_rows,
    _loader_write_rows,
    _run_bomb_west_04,
    dump_room_tiles,
    enter_patra_via_source_cellar,
    materialize_stair_room,
    take_stairs_from_source,
    walk_play_room_to_patra,
)

BEAD = "rr-sz8.3"
TAG = "l9_stair_patra_credits_recon"
SETTLE_IDLE_FRAMES = 20
LOAD_MAX_FRAMES = 500
WALK_MAX_FRAMES = 400
PUSH_FRAMES = 40
CLEAR_MAX_FRAMES = 2500
CELLAR_MAX_FRAMES = 2800
GRID_MAX_FRAMES = 1800
ADDR_UPDATING = 0x0011
ADDR_SUBMODE = 0x0013
ADDR_UW_EXIT_TYPE = 0x005A
NORTH_PROBE_FRAMES = 400
CLEAR_PROBE_FRAMES = 1800

def build_winning_fixture(
    *,
    source: int,
    cellar_side: str = "left",
    tag: str = TAG,
    fixture_name: str | None = None,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    name = fixture_name or f"Level9Stair{source:02X}PatraEnteredReconFixture"
    loader = stair_loader_for(source)
    writes = _loader_write_rows(loader)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "source_state": FIXTURE_SOURCE,
        "stair_source": source,
        "cellar_side": cellar_side,
        "checkpoint": name,
        "fixture_writes": writes,
    }
    try:
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        report["loader"] = used.label
        if not loaded:
            report["error"] = f"loader did not settle 0x{source:02X}"
            return report
        writes.extend(_cellar_write_rows(source, cellar_side))
        report["fixture_writes"] = writes
        if cellar_dest_for(source, side=cellar_side) == 0x52 or is_patra_cellar_source(source):
            dest = enter_patra_via_source_cellar(
                env, source, total=total, side=cellar_side
            )
        else:
            dest = take_stairs_from_source(
                env, source, total=total, cellar_side=cellar_side
            )
        snap = read_snapshot(env.get_ram())
        report["dest"] = dest
        png = RECORDINGS_DIR / f"{tag}_entered.png"
        save_rgb_png(obs if obs is not None else env.render(), png)
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, png)
        report["screenshot"] = str(png)
        report["ok"] = bool(landed_final_patra(snap))
        if report["ok"]:
            path = _save_checkpoint(
                env,
                name,
                source_state=FIXTURE_SOURCE,
                phase=f"cellar_checksubroom_0x{source:02x}_{cellar_side}_into_live_patra",
                result={
                    "ok": True,
                    "stair_source": source,
                    "room": 0x52,
                    "final_patra_live": True,
                    "patra_eye_count": len(patra_eyes(snap)),
                    "frames": total[0],
                },
                fixture_writes=writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)
        else:
            report["error"] = (
                f"stairs from 0x{source:02X} settled "
                f"room 0x{snap.screen:02X} mode {snap.mode} "
                f"patra={final_patra_live(snap)} eyes={len(patra_eyes(snap))}"
            )
        return report
    finally:
        env.close()


def run_suffix_from_fixture(
    *,
    start_state: str,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    """Reuse the proven Patra → Ganon → Zelda → credits suffix."""
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "start_state": start_state,
        "trial": trial_i,
        "tag": tag,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
        "checkpoints": [],
    }

    def checkpoint(name: str, phase: str) -> None:
        if not save_checkpoints:
            return
        path = _save_checkpoint(
            env,
            name,
            source_state=start_state,
            phase=phase,
            result=_checkpoint_result(env, total),
            fixture_writes=[],
            bead=BEAD,
        )
        report["checkpoints"].append(str(path))

    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["start"] = compact_snapshot(start)
        report["start_inventory"] = _inventory_snapshot(env.get_ram())
        if not landed_final_patra(start):
            report["error"] = (
                "expected live final Patra with eight eyes and closed north door"
            )
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")

        patra_fight = FinalPatraFightController().run(
            env, assist=assist, total=total
        )
        report["patra_fight"] = patra_fight
        if not patra_fight["ok"] or not final_patra_north_door_earned(
            read_snapshot(env.get_ram())
        ):
            report["error"] = "final Patra controller timed out before north door"
            return report
        obs = _idle(env, 45, assist=assist, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_cleared.png")
        checkpoint(
            "Level9StairPatraClearedReconFixture",
            "final_patra_north_door_earned",
        )

        obs, entered = _enter_ganon(env, assist=assist, total=total)
        report["ganon_entered"] = entered
        if not entered:
            report["error"] = "failed to enter live Ganon room 0x42"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")

        ganon_fight = GanonFightController().run(env, assist=assist, total=total)
        report["ganon_fight"] = ganon_fight
        report["runtime_controller_writes"]["inventory"] = int(
            ganon_fight["selected_item_writes"]
        )
        if (
            not ganon_fight["ok"]
            or not ganon_defeated(env.get_ram())
            or ganon_fight["selected_item_writes"] != 0
        ):
            report["error"] = "Ganon suffix failed or wrote B-item selection"
            return report
        obs = _idle(env, 1, assist=assist, total=total)
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_arrow_kill.png")

        obs, power = _collect_power_triforce(env, assist=assist, total=total)
        report["power_triforce_collected"] = power
        if not power:
            report["error"] = "Ganon died but north door did not open"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_defeated.png")

        obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
        report["zelda_room_entered"] = zelda_room
        if not zelda_room:
            report["error"] = "failed to enter live Zelda room 0x32"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")

        obs, rescued = _rescue_zelda(env, assist=assist, total=total)
        report["zelda_rescued"] = rescued
        if not rescued:
            report["error"] = "failed to clear guard fires and trigger Zelda ending"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")

        credits_frame = None
        credits_capture_frame = None
        final_frame = None
        for _ in range(12000):
            snap = read_snapshot(env.get_ram())
            if credits_frame is None and credits_rolling(snap):
                credits_frame = total[0]
                obs = _idle(env, 240, assist=assist, total=total)
                credits_capture_frame = total[0]
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_credits.png")
            if final_ending_screen(snap):
                final_frame = total[0]
                obs = _idle(env, 90, assist=assist, total=total)
                save_rgb_png(
                    obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png"
                )
                break
            obs = _step(env, nes_idle_action(), assist=assist, total=total)

        assist_report = assist.report() if assist is not None else {"enabled": False}
        report["credits_frame"] = credits_frame
        report["credits_capture_frame"] = credits_capture_frame
        report["final_screen_frame"] = final_frame
        report["credits_reached"] = credits_frame is not None
        report["final_screen_reached"] = final_frame is not None
        report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
        report["total_frames"] = total[0]
        report["assist"] = assist_report
        report["continuous_session"] = True
        report["state_loads_after_start"] = 0
        report["ok"] = bool(
            credits_frame is not None
            and final_frame is not None
            and assist_report.get("progression_writes", 0) == 0
            and assist_report.get("capacity_writes", 0) == 0
            and not any(report["runtime_controller_writes"].values())
        )
        return report
    finally:
        env.close()


def run_room04_bomb_west_to_credits(
    *,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = "l9_play04_bombwest_patra_credits_recon",
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: 0x04 bomb-west → 0x03 stairs → Patra → credits.

    Fixture inventory + 0x14 neighbor-scroll happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    0x03 doors are never poked. InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(ROOM04)
    writes = _loader_write_rows(loader)
    writes.append(
        {
            "name": "selected_item_bombs",
            "address": ADDR_SELECTED_ITEM,
            "address_hex": "0x0656",
            "value": B_ITEM_BOMBS,
        }
    )
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": "0x04",
        "via": "bomb_west",
        "trial": trial_i,
        "tag": tag,
        "fixture_writes": writes,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
        "door_poke_on_03": False,
    }
    try:
        obs, used, loaded = materialize_stair_room(
            env, ROOM04, total=total, selected_item=B_ITEM_BOMBS
        )
        report["loader"] = used.label
        report["loaded"] = loaded
        if not loaded:
            report["error"] = "loader did not settle 0x04"
            report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
            return report
        report["settled"] = dest_report(read_snapshot(env.get_ram()))
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        cooldown = 0
        for _ in range(800):
            snap = read_snapshot(env.get_ram())
            if in_room_04(snap) and not live_combat_objects(snap):
                break
            if not in_room_04(snap):
                break
            frame, cooldown = chase_sword_step(snap, cooldown)
            _step(env, frame.action, assist=assist, total=total)

        bomb = _run_bomb_west_04(env, total=total, assist=assist)
        after_bomb_obs = bomb.pop("after_bomb_obs")
        dest_obs = bomb.pop("obs")
        report["bomb_west"] = bomb
        if after_bomb_obs is not None:
            save_rgb_png(after_bomb_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_bomb.png")
        if dest_obs is not None:
            save_rgb_png(dest_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_dest.png")
        snap = read_snapshot(env.get_ram())
        report["dest_screen"] = int(snap.screen)
        if int(snap.screen) != ROOM03:
            report["error"] = (
                f"bomb-west dest 0x{snap.screen:02X} is not 0x03 "
                f"phase={bomb['controller'].get('phase')}"
            )
            return report

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
        selected = int(env.get_ram()[ADDR_SELECTED_ITEM])
        report["selected_item_after_pause"] = selected
        report["pause_right_moves"] = pause_moves
        if selected != B_ITEM_ARROWS:
            report["error"] = f"pause menu left selected_item={selected}, need arrows"
            return report

        # East hole sits on stair-Y but the diamond blocks a straight LEFT.
        # Accepted 0x03 push recovers down the west wall; seed that stand
        # (32,189) before the same take_stairs_from_source as play-source 03.
        for x, y, y_first in ((208, 189, True), (32, 189, False)):
            for _ in range(WALK_MAX_FRAMES):
                snap = read_snapshot(env.get_ram())
                if snap.screen != ROOM03:
                    break
                frame = walk_to_step(snap, x, y, y_first=y_first, tol=0)
                if frame.reason == "walk_arrived":
                    break
                _step(env, frame.action, assist=assist, total=total)
        report["west_south"] = dest_report(read_snapshot(env.get_ram()))

        dest = walk_play_room_to_patra(
            env,
            ROOM03,
            total=total,
            cellar_side="left",
            assist=assist,
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png")
        if not landed_final_patra(snap):
            report["error"] = (
                f"0x04→0x03 stairs settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest.get('final_patra_live')}"
            )
            report["final"] = dest
            return report
        save_rgb_png(idle_obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png")
        if save_checkpoints:
            path = _save_checkpoint(
                env,
                "Level9Room04BombWestReconFixture",
                source_state=FIXTURE_SOURCE,
                phase="play_0x04_bomb_west_into_live_patra",
                result={
                    "ok": True,
                    "source_room": ROOM04,
                    "via": "bomb_west",
                    "room": 0x52,
                    "final_patra_live": True,
                    "frames": total[0],
                },
                fixture_writes=writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)

        suffix = run_suffix_from_live_env(
            env,
            assist=assist,
            total=total,
            tag=tag,
            trial_i=trial_i,
            start_state="play_0x04_bomb_west_walk",
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


def run_play_source_to_credits(
    *,
    source: int,
    cellar_side: str = "left",
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    """One continuous trial: play source -> cellar 0x77 -> live Patra -> credits.

    Fixture inventory + neighbor-scroll settle happen before the walk.
    After materialize, no object / room / door / progression / capacity writes.
    InitMode9 is not used.
    """
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    loader = stair_loader_for(source)
    writes = _loader_write_rows(loader)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "init_mode9": False,
        "source_room": f"0x{source:02X}",
        "cellar_side": cellar_side,
        "trial": trial_i,
        "tag": tag,
        "fixture_writes": writes,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
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
        settle = dest_report(read_snapshot(env.get_ram()))
        report["settled"] = settle
        settle_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_settle.png"
        save_rgb_png(obs, settle_png)
        report["settle_png"] = str(settle_png)

        tiles = dump_room_tiles(env, total=total)
        tile_json = RECORDINGS_DIR / f"{tag}_t{trial_i}_0x{source:02x}_tiles.json"
        write_json_report(tile_json, tiles)
        report["tile_json"] = str(tile_json)
        report["stair_hits"] = tiles["stair_hits"]
        report["mouth_hits"] = tiles["mouth_hits"]
        if tiles["stair_hits"]:
            hit = tiles["stair_hits"][0]
            report["stair_tile"] = hit["tile"]
            report["stair_xy"] = [hit["x"], hit["y"]]
        elif tiles["mouth_hits"]:
            hit = tiles["mouth_hits"][0]
            report["stair_tile"] = hit["tile"]
            report["stair_xy"] = [hit["x"], hit["y"]]

        # Rematerialize so the tile-scan pokes are not on the acceptance path.
        env.close()
        env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
        total = [0]
        obs, used, loaded = materialize_stair_room(env, source, total=total)
        if not loaded:
            report["error"] = f"rematerialize failed 0x{source:02X}"
            return report
        save_rgb_png(obs, settle_png)

        dest = walk_play_room_to_patra(
            env, source, total=total, cellar_side=cellar_side, assist=assist
        )
        report["walk"] = dest
        snap = read_snapshot(env.get_ram())
        walk_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_after_walk.png"
        idle_obs = _idle(env, 1, assist=None, total=total)
        save_rgb_png(idle_obs, walk_png)
        report["after_walk_png"] = str(walk_png)
        if in_patra_cellar(snap) or snap.mode == CELLAR_MODE:
            cellar_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_cellar.png"
            save_rgb_png(idle_obs, cellar_png)
            report["cellar_png"] = str(cellar_png)
            report["entered_cellar_77"] = snap.screen == PATRA_STAIR_SOURCE
        if landed_final_patra(snap):
            patra_png = RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_entry.png"
            save_rgb_png(idle_obs, patra_png)
            report["patra_entry_png"] = str(patra_png)
            report["final_patra_live"] = True
        else:
            report["error"] = (
                f"walk from 0x{source:02X} settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={dest.get('final_patra_live')} "
                f"eyes={dest.get('patra_eyes')}"
            )
            report["final"] = dest
            return report

        if save_checkpoints:
            path = _save_checkpoint(
                env,
                f"Level9Room{source:02X}StairsReconFixture",
                source_state=FIXTURE_SOURCE,
                phase=f"play_0x{source:02x}_stairs_into_live_patra",
                result={
                    "ok": True,
                    "source_room": source,
                    "room": 0x52,
                    "final_patra_live": True,
                    "patra_eye_count": len(patra_eyes(snap)),
                    "frames": total[0],
                },
                fixture_writes=writes,
                bead=BEAD,
            )
            report["checkpoint_path"] = str(path)

        suffix = run_suffix_from_live_env(
            env,
            assist=assist,
            total=total,
            tag=tag,
            trial_i=trial_i,
            start_state=f"play_0x{source:02x}_walk",
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


def run_suffix_from_live_env(
    env: Any,
    *,
    assist: Any,
    total: list[int],
    tag: str,
    trial_i: int,
    start_state: str,
) -> dict[str, Any]:
    """Patra -> Ganon -> Zelda -> credits on an already-live env."""
    report: dict[str, Any] = {
        "ok": False,
        "start_state": start_state,
        "trial": trial_i,
        "runtime_controller_writes": {
            "object": 0,
            "room": 0,
            "door": 0,
            "inventory": 0,
            "progression": 0,
            "capacity": 0,
        },
    }
    start = read_snapshot(env.get_ram())
    report["start"] = compact_snapshot(start)
    if not landed_final_patra(start):
        report["error"] = "expected live final Patra with eight eyes and closed north door"
        return report
    save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")

    patra_fight = FinalPatraFightController().run(env, assist=assist, total=total)
    report["patra_fight"] = patra_fight
    if not patra_fight["ok"] or not final_patra_north_door_earned(
        read_snapshot(env.get_ram())
    ):
        report["error"] = "final Patra controller timed out before north door"
        return report
    obs = _idle(env, 45, assist=assist, total=total)
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_cleared.png")

    obs, entered = _enter_ganon(env, assist=assist, total=total)
    report["ganon_entered"] = entered
    if not entered:
        report["error"] = "failed to enter live Ganon room 0x42"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")

    ganon_fight = GanonFightController().run(env, assist=assist, total=total)
    report["ganon_fight"] = ganon_fight
    report["runtime_controller_writes"]["inventory"] = int(
        ganon_fight["selected_item_writes"]
    )
    if (
        not ganon_fight["ok"]
        or not ganon_defeated(env.get_ram())
        or ganon_fight["selected_item_writes"] != 0
    ):
        report["error"] = "Ganon suffix failed or wrote B-item selection"
        return report
    obs = _idle(env, 1, assist=assist, total=total)
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_arrow_kill.png")

    obs, power = _collect_power_triforce(env, assist=assist, total=total)
    report["power_triforce_collected"] = power
    if not power:
        report["error"] = "Ganon died but north door did not open"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_defeated.png")

    obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
    report["zelda_room_entered"] = zelda_room
    if not zelda_room:
        report["error"] = "failed to enter live Zelda room 0x32"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")

    obs, rescued = _rescue_zelda(env, assist=assist, total=total)
    report["zelda_rescued"] = rescued
    if not rescued:
        report["error"] = "failed to clear guard fires and trigger Zelda ending"
        return report
    save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")

    credits_frame = None
    credits_capture_frame = None
    final_frame = None
    for _ in range(12000):
        snap = read_snapshot(env.get_ram())
        if credits_frame is None and credits_rolling(snap):
            credits_frame = total[0]
            obs = _idle(env, 240, assist=assist, total=total)
            credits_capture_frame = total[0]
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_credits.png")
        if final_ending_screen(snap):
            final_frame = total[0]
            obs = _idle(env, 90, assist=assist, total=total)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png")
            break
        obs = _step(env, nes_idle_action(), assist=assist, total=total)

    assist_report = assist.report() if assist is not None else {"enabled": False}
    report["credits_frame"] = credits_frame
    report["credits_capture_frame"] = credits_capture_frame
    report["final_screen_frame"] = final_frame
    report["credits_reached"] = credits_frame is not None
    report["final_screen_reached"] = final_frame is not None
    report["final"] = compact_snapshot(read_snapshot(env.get_ram()))
    report["total_frames"] = total[0]
    report["assist"] = assist_report
    report["continuous_session"] = True
    report["state_loads_after_start"] = 0
    report["ok"] = bool(
        credits_frame is not None
        and final_frame is not None
        and assist_report.get("progression_writes", 0) == 0
        and assist_report.get("capacity_writes", 0) == 0
        and not any(report["runtime_controller_writes"].values())
    )
    return report


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    patra = report.get("patra_fight") or {}
    ganon = report.get("ganon_fight") or {}
    return {
        "trial": report.get("trial"),
        "ok": report.get("ok"),
        "patra_north_door": patra.get("north_door_earned"),
        "patra_frames": patra.get("frames"),
        "ganon_defeated": ganon.get("last_boss_defeated"),
        "credits_frame": report.get("credits_frame"),
        "final_screen_frame": report.get("final_screen_frame"),
        "error": report.get("error"),
    }
