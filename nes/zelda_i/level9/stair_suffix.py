"""Level 9 stair-run suffix: fixture, Patra→credits."""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon.trace import compact_snapshot
from zelda_i.level9.ganon import (
    B_ITEM_BOMBS,
    GanonFightController,
    _collect_power_triforce,
    _enter_ganon,
    _enter_zelda,
    _rescue_zelda,
    credits_rolling,
    final_ending_screen,
    ganon_defeated,
)
from zelda_i.level9.patra import (
    FinalPatraFightController,
    final_patra_live,
    final_patra_north_door_earned,
    patra_eyes,
)
from zelda_i.level9.room62 import LEVEL9_STAIR_SOURCES
from zelda_i.level9.stairs import (
    BOMB_WALL_04_WEST,
    BOMB_WALL_31_WEST,
    BOMB_WEST_STAND,
    LEVEL9_CELLAR_ROOMS,
    LEVEL9_STAIR_PAIRS,
    PATRA_STAIR_SOURCE,
    PLAY_STAIR_CANDIDATES,
    ROOM03,
    ROOM03_STAIR_X,
    ROOM03_STAIR_Y,
    ROOM04,
    ROOM11,
    ROOM13,
    ROOM20,
    ROOM21,
    ROOM21_WEST_X,
    ROOM30,
    ROOM30_STAIR_X,
    ROOM30_STAIR_Y,
    ROOM31,
    ROOM40,
    ROOM41,
    ROOM51,
    cellar_dest_for,
    cellar_for_play_room,
    dest_report,
    is_patra_cellar_source,
    landed_final_patra,
    loader_avoids,
    paired_stair_dest,
    play_rooms_entering_cellar,
    rom_pair,
    rom_secret,
    room03_rom_neighbors,
    room30_rom_neighbors,
    stair_loader_for,
)
from zelda_i.level9.stair_session import (
    BEAD,
    DumpPlan,
    FIXTURE_SOURCE,
    TAG,
    _checkpoint_result,
    _idle,
    _inventory_snapshot,
    _loader_write_rows,
    _new_env,
    _png,
    _save_checkpoint,
    _step,
    dump_room,
    enter_patra_via_source_cellar,
    materialize_stair_room,
    take_stairs_from_source,
)
from zelda_i.level9.stair_run import _cellar_write_rows, dump_room_tiles
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot


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
    writes = _loader_write_rows(stair_loader_for(source))
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False,
        "fixture_only": True, "source_state": FIXTURE_SOURCE, "stair_source": source,
        "cellar_side": cellar_side, "checkpoint": name, "fixture_writes": writes,
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
            dest = enter_patra_via_source_cellar(env, source, total=total, side=cellar_side)
        else:
            dest = take_stairs_from_source(env, source, total=total, cellar_side=cellar_side)
        snap = read_snapshot(env.get_ram())
        report["dest"] = dest
        png = RECORDINGS_DIR / f"{tag}_entered.png"
        save_rgb_png(obs if obs is not None else env.render(), png)
        save_rgb_png(_idle(env, 1, assist=None, total=total), png)
        report["screenshot"] = str(png)
        report["ok"] = bool(landed_final_patra(snap))
        if report["ok"]:
            path = _save_checkpoint(
                env, name, source_state=FIXTURE_SOURCE,
                phase=f"cellar_checksubroom_0x{source:02x}_{cellar_side}_into_live_patra",
                result={
                    "ok": True, "stair_source": source, "room": 0x52,
                    "final_patra_live": True, "patra_eye_count": len(patra_eyes(snap)),
                    "frames": total[0],
                },
                fixture_writes=writes, bead=BEAD,
            )
            report["checkpoint_path"] = str(path)
        else:
            report["error"] = (
                f"stairs from 0x{source:02X} settled room 0x{snap.screen:02X} "
                f"mode {snap.mode} patra={final_patra_live(snap)} eyes={len(patra_eyes(snap))}"
            )
        return report
    finally:
        env.close()


def run_suffix_from_live_env(
    env: Any, *, assist: Any, total: list[int], tag: str, trial_i: int, start_state: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False, "start_state": start_state, "trial": trial_i,
        "runtime_controller_writes": {
            "object": 0, "room": 0, "door": 0, "inventory": 0, "progression": 0, "capacity": 0,
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
    if not patra_fight["ok"] or not final_patra_north_door_earned(read_snapshot(env.get_ram())):
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
    report["runtime_controller_writes"]["inventory"] = int(ganon_fight["selected_item_writes"])
    if not ganon_fight["ok"] or not ganon_defeated(env.get_ram()) or ganon_fight["selected_item_writes"] != 0:
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

    credits_frame = credits_capture_frame = final_frame = None
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


def run_suffix_from_fixture(
    *,
    start_state: str,
    infinite_life: bool = True,
    save_checkpoints: bool = False,
    tag: str = TAG,
    trial_i: int = 0,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    report: dict[str, Any] = {
        "ok": False, "bead": BEAD, "track": "recon_fixture", "route_eligible": False,
        "fixture_only": True, "start_state": start_state, "trial": trial_i, "tag": tag,
        "runtime_controller_writes": {
            "object": 0, "room": 0, "door": 0, "inventory": 0, "progression": 0, "capacity": 0,
        },
        "checkpoints": [],
    }

    def checkpoint(name: str, phase: str) -> None:
        if not save_checkpoints:
            return
        path = _save_checkpoint(
            env, name, source_state=start_state, phase=phase,
            result=_checkpoint_result(env, total), fixture_writes=[], bead=BEAD,
        )
        report["checkpoints"].append(str(path))

    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["start"] = compact_snapshot(start)
        report["start_inventory"] = _inventory_snapshot(env.get_ram())
        if not landed_final_patra(start):
            report["error"] = "expected live final Patra with eight eyes and closed north door"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")
        live = run_suffix_from_live_env(
            env, assist=assist, total=total, tag=tag, trial_i=trial_i, start_state=start_state,
        )
        live.pop("start", None)
        report.update(live)
        if save_checkpoints and report.get("ok"):
            checkpoint("Level9StairPatraClearedReconFixture", "final_patra_north_door_earned")
        return report
    finally:
        env.close()


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    patra = report.get("patra_fight") or {}
    ganon = report.get("ganon_fight") or {}
    return {
        "trial": report.get("trial"), "ok": report.get("ok"),
        "patra_north_door": patra.get("north_door_earned"), "patra_frames": patra.get("frames"),
        "ganon_defeated": ganon.get("last_boss_defeated"),
        "credits_frame": report.get("credits_frame"),
        "final_screen_frame": report.get("final_screen_frame"),
        "error": report.get("error"),
    }


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
