"""Run live final Patra ``0x52`` through Ganon, Zelda, and the ending.

The start state is an explicit full-loadout/room-loader recon fixture.  After
reset, the route uses controller input plus optional Survival health refill;
it does not write object slots, room state, doors, inventory, or progression.

Example::

    uv run python zelda_i/scripts/run_level9_patra.py \
      --build-fixture --infinite-life --save-state --trials 1 \
      --tag l9_patra_credits_recon
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
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
    NORTH_DOOR,
    PATRA_EYE_COUNT,
    final_patra_live,
    final_patra_north_door_earned,
    patra_eyes,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOMBS,
    ADDR_BOW,
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_MAGIC_KEY,
    ADDR_RING,
    ADDR_SELECTED_ITEM,
    ADDR_SWORD,
    ADDR_TRIFORCE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import (
    B_ITEM_ARROWS,
    FIXTURE_SOURCE,
    _checkpoint_result,
    _collect_power_triforce,
    _enter_ganon,
    _enter_zelda,
    _fixture_write_rows,
    _idle,
    _rescue_zelda,
    _save_checkpoint,
    _step,
    build_fixture,
)

BEAD = "rr-sz8.2"
FIXTURE_NAME = "Level9FinalPatraReconFixture"
TAG = "l9_patra_credits_recon"


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


def run_once(
    *,
    start_state: str = FIXTURE_NAME,
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
    writes = _fixture_write_rows(clear_final_patra=False)
    report: dict[str, Any] = {
        "ok": False,
        "bead": BEAD,
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "start_state": start_state,
        "source_state": FIXTURE_SOURCE,
        "trial": trial_i,
        "tag": tag,
        "fixture_writes_inherited": writes,
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
            fixture_writes=writes,
            bead=BEAD,
        )
        report["checkpoints"].append(str(path))

    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        start_inventory = _inventory_snapshot(env.get_ram())
        report["start"] = compact_snapshot(start)
        report["start_inventory"] = start_inventory
        if not (
            final_patra_live(start)
            and len(patra_eyes(start)) == PATRA_EYE_COUNT
            and not start.cur_opened_doors & NORTH_DOOR
            and not start.open_doorway_mask & NORTH_DOOR
        ):
            report["error"] = (
                "expected live final Patra with eight eyes and closed north door"
            )
            return report
        if start_inventory["selected_item"] != B_ITEM_ARROWS:
            report["error"] = "fixture did not preselect Silver Arrows"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_start.png")

        patra_fight = FinalPatraFightController().run(
            env,
            assist=assist,
            total=total,
        )
        report["patra_fight"] = patra_fight
        if not patra_fight["ok"] or not final_patra_north_door_earned(
            read_snapshot(env.get_ram())
        ):
            report["error"] = "final Patra controller timed out before north door"
            obs = _idle(env, 1, assist=assist, total=total)
            save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_failure.png")
            return report

        # Let the naturally earned shutter finish opening before going north.
        obs = _idle(env, 45, assist=assist, total=total)
        after_patra = read_snapshot(env.get_ram())
        after_patra_inventory = _inventory_snapshot(env.get_ram())
        report["after_patra"] = compact_snapshot(after_patra)
        report["after_patra_inventory"] = after_patra_inventory
        report["inventory_preserved_through_patra"] = (
            after_patra_inventory == start_inventory
        )
        if not report["inventory_preserved_through_patra"]:
            report["error"] = "inventory changed during final Patra combat"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_patra_cleared.png")
        checkpoint(
            "Level9FinalPatraClearedReconFixture", "final_patra_north_door_earned"
        )

        obs, entered = _enter_ganon(env, assist=assist, total=total)
        report["ganon_entered"] = entered
        if not entered:
            report["error"] = "failed to enter live Ganon room 0x42"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ganon_start.png")
        checkpoint("Level9PatraGanonReconFixture", "ganon_fight_start")

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
        checkpoint("Level9PatraGanonDefeatedReconFixture", "ganon_defeated_north_open")

        obs, zelda_room = _enter_zelda(env, assist=assist, total=total)
        report["zelda_room_entered"] = zelda_room
        if not zelda_room:
            report["error"] = "failed to enter live Zelda room 0x32"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_zelda_room.png")
        checkpoint("Level9PatraZeldaRoomReconFixture", "zelda_room_entry")

        obs, rescued = _rescue_zelda(env, assist=assist, total=total)
        report["zelda_rescued"] = rescued
        if not rescued:
            report["error"] = "failed to clear guard fires and trigger Zelda ending"
            return report
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_ending_start.png")
        checkpoint("Level9PatraEndingStartReconFixture", "ending_mode_start")

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
                checkpoint("Level9PatraCreditsReconFixture", "credits_rolling")
            if final_ending_screen(snap):
                final_frame = total[0]
                obs = _idle(env, 90, assist=assist, total=total)
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final_screen.png")
                checkpoint(
                    "Level9PatraFinalScreenReconFixture", "final_press_start_screen"
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


def _trial_summary(report: dict[str, Any]) -> dict[str, Any]:
    patra = report.get("patra_fight") or {}
    ganon = report.get("ganon_fight") or {}
    assist = report.get("assist") or {}
    return {
        "trial": report.get("trial"),
        "ok": report.get("ok"),
        "patra_north_door": patra.get("north_door_earned"),
        "patra_frames": patra.get("frames"),
        "patra_eye_counts": patra.get("eye_count_changes"),
        "patra_body_hp": patra.get("body_hp_changes"),
        "ganon_defeated": ganon.get("last_boss_defeated"),
        "selected_item_writes": ganon.get("selected_item_writes"),
        "zelda_rescued": report.get("zelda_rescued"),
        "credits_frame": report.get("credits_frame"),
        "final_screen_frame": report.get("final_screen_frame"),
        "assist_health_writes": (assist.get("health") or {}).get("writes"),
        "progression_writes": assist.get("progression_writes"),
        "capacity_writes": assist.get("capacity_writes"),
        "error": report.get("error"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-state", default=FIXTURE_NAME)
    parser.add_argument("--build-fixture", action="store_true")
    parser.add_argument("--infinite-life", action="store_true")
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()

    built = None
    if args.build_fixture:
        built = build_fixture(
            tag=f"{args.tag}_fixture",
            fixture_name=FIXTURE_NAME,
            clear_final_patra=False,
            bead=BEAD,
        )
        print(
            "FIXTURE",
            {
                "ok": built.get("ok"),
                "path": built.get("checkpoint_path"),
                "room": (built.get("room_entry") or {}).get("room"),
                "objects": len((built.get("room_entry") or {}).get("objects", [])),
            },
        )
        if not built.get("ok"):
            write_json_report(
                RECORDINGS_DIR / f"{args.tag}.json",
                {"fixture": built, "ok": False},
            )
            return 1

    trials: list[dict[str, Any]] = []
    for trial_i in range(max(1, args.trials)):
        result = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoints=args.save_state,
            tag=args.tag,
            trial_i=trial_i,
        )
        trials.append(result)
        print("TRIAL", _trial_summary(result))

    report = {
        "bead": BEAD,
        "segment": "final_patra_to_final_screen",
        "track": "recon_fixture",
        "route_eligible": False,
        "fixture_only": True,
        "fixture": built,
        "ok": all(trial.get("ok") for trial in trials),
        "trials": trials,
    }
    out = RECORDINGS_DIR / f"{args.tag}.json"
    write_json_report(out, report)
    print("REPORT", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
