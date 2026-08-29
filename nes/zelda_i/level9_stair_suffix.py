"""Level 9 stair-run suffix: fixture, Patra→credits."""

from __future__ import annotations

from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import (
    GanonFightController,
    credits_rolling,
    final_ending_screen,
    ganon_defeated,
)
from zelda_i.level9_patra import (
    FinalPatraFightController,
    final_patra_live,
    final_patra_north_door_earned,
    patra_eyes,
)
from zelda_i.level9_stairs import (
    cellar_dest_for,
    is_patra_cellar_source,
    landed_final_patra,
    stair_loader_for,
)
from zelda_i.level9_stair_session import (
    BEAD,
    FIXTURE_SOURCE,
    TAG,
    _idle,
    _loader_write_rows,
    _step,
    enter_patra_via_source_cellar,
    materialize_stair_room,
    take_stairs_from_source,
)
from zelda_i.level9_stair_run import _cellar_write_rows
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import (
    _checkpoint_result,
    _collect_power_triforce,
    _enter_ganon,
    _enter_zelda,
    _rescue_zelda,
    _save_checkpoint,
)
from zelda_i.scripts.run_level9_patra import _inventory_snapshot


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
