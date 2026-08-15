"""Assisted: Level3Raft → Manhandla 0x4d → TF bit 0x04 (rr-vpl residual).

Directed path (LIVE 2026-08-07)::

    0x0f mode9 reverse channel + NW stairs UP → 0x69
    UP → 0x59
    BOMB_RIGHT@(192,141) → 0x5a   *** walk-RIGHT sealed post-Raft ***
    RIGHT → 0x5b
    BOMB_RIGHT@(192,141) → 0x5c (3× Darknut)
    full clear (doors raw=3) → RIGHT @ y≈141 → 0x5d
    clear Zol+Keese only (ignore invuln 0x2b) → UP → 0x4d Manhandla 0x3c
    bombs → HC → TF room (bit 0x04)

Thin runner over ``Level3BossPathController`` + ``dungeon_ops``.
Intervention: Survival (``--infinite-life``). Not Clean STATUS.
Does **not** rewrite ``run_level3_raft.py``.

``--poke-bombs N`` is RECON opt-in (default off). Durable runs should not poke.

Examples::

    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --trials 2
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --to-boss --trials 2
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --kill --poke-bombs 16
    uv run python nes/zelda_i/scripts/run_level3_to_boss.py --infinite-life --phase gate5d --tag l3_gate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, room_fields
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level3_boss_path import Level3BossPathController
from zelda_i.level3_dungeon import (
    ROOM_L3_BOSS,
    ROOM_L3_BOSS_PREP,
    level3_has_raft,
)
from zelda_i.level3_overworld import LEVEL3
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.runner import VideoTap, add_video_args, resolve_video

def _provenance(
    path: Path,
    *,
    start_state: str,
    segment: str,
    intervention: str,
    trial: dict,
) -> None:
    write_state_provenance(
        path,
        source_state_path=(
            GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
        ),
        request={
            "segment": segment,
            "natural_entry": False,
            "start_state": start_state,
            "intervention_class": intervention,
        },
        selected_trial=trial,
        natural_entry=False,
    )

def _finish(
    report: dict,
    tag: str,
    controller: Level3BossPathController,
    *,
    assist=None,
    tap=None,
) -> dict:
    report["controller"] = controller.report()
    report["path_log"].extend(
        e for e in controller.path_log if e not in report["path_log"]
    )
    report["traps"].extend(t for t in controller.traps if t not in report["traps"])
    report["notes"].extend(n for n in controller.notes if n not in report["notes"])
    if assist is not None and "assist" not in report:
        report["assist"] = assist.report()
    if tap is not None and "video" not in report:
        report["video"] = tap.close()
    out = RECORDINGS_DIR / f"{tag}_report.json"
    write_json_report(out, report)
    report["report_path"] = str(out)
    return report

def run_once(
    *,
    start_state: str = "Level3Raft",
    infinite_life: bool = True,
    to_boss: bool = True,
    kill: bool = False,
    poke_bombs: int | None = None,
    save_checkpoint: bool = False,
    tag: str = "l3_to_boss",
    phase: str = "all",
    video_path=None,
    video_config=None,
    intro_frames: int = 0,
) -> dict:
    """One assisted trial from Level3Raft toward boss / TF."""
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    total = [0]
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    controller = Level3BossPathController(poke_bombs=poke_bombs, tag=tag)
    report: dict = {
        "ok": False,
        "track": track,
        "intervention_class": intervention,
        "start_state": start_state,
        "phase": phase,
        "to_boss": to_boss,
        "kill": kill,
        "tag": tag,
        "poke_bombs": poke_bombs,
        "reached_5d": False,
        "reached_4d": False,
        "boss_beaten": False,
        "tf04": False,
        "manhandla_confirmed": False,
        "dmg_events": 0,
        "path_log": [],
        "traps": [],
        "notes": [],
    }
    tap = None

    try:
        obs, _ = reset_obs(env)
        tap = VideoTap(
            video_path,
            video_config,
            tag=tag,
            intro_summary="Survival Level3Raft -> Manhandla -> TF 0x04",
            intro_frames=intro_frames,
        )
        tap.attach(env, obs)
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        total[0] = 1
        idle(env, assist, total, 20)
        entry = room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["entry"] = entry
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_start.png")

        if not (
            entry["level"] == LEVEL3
            and (entry.get("raft") or level3_has_raft(env.get_ram()))
        ):
            report["error"] = (
                f"expected Level3Raft (raft set); got level={entry['level']} "
                f"raft={entry.get('raft')} sc={entry['sc']}"
            )
            return _finish(report, tag, controller, assist=assist, tap=tap)

        # --- path to 0x5d ---
        if phase in ("all", "to5d", "gate5d", "boss", "kill"):
            if entry["screen"] == ROOM_L3_BOSS_PREP and entry["mode"] == PLAY_MODE:
                report["path_to_5d"] = {"ok": True, "skipped": True}
                report["reached_5d"] = controller.reached_5d = True
            elif entry["screen"] == ROOM_L3_BOSS and entry["mode"] == PLAY_MODE:
                report["path_to_5d"] = {
                    "ok": True,
                    "skipped": True,
                    "already_boss": True,
                }
                report["reached_5d"] = controller.reached_5d = True
                report["reached_4d"] = controller.reached_4d = True
            else:
                p5 = controller.path_to_5d(env, assist, total)
                report["path_to_5d"] = {
                    "ok": p5.get("ok"),
                    "path_log": p5.get("path_log"),
                    "error": p5.get("error"),
                    "final": p5.get("final"),
                }
                report["path_log"].extend(p5.get("path_log") or [])
                report["traps"].extend(p5.get("traps") or [])
                report["notes"].extend(p5.get("notes") or [])
                report["reached_5d"] = bool(p5.get("ok"))
                if not p5.get("ok"):
                    report["final"] = p5.get("final")
                    report["total_frames"] = total[0]
                    report["error"] = p5.get("error")
                    return _finish(report, tag, controller, assist=assist, tap=tap)

        if phase == "to5d":
            report["ok"] = report["reached_5d"]
            report["final"] = room_fields(
                read_snapshot(env.get_ram()), env.get_ram()
            )
            report["total_frames"] = total[0]
            return _finish(report, tag, controller, assist=assist, tap=tap)

        # --- gate 0x5d → 0x4d ---
        if (
            to_boss or kill or phase in ("all", "gate5d", "boss", "kill")
        ) and not report["reached_4d"]:
            if read_snapshot(env.get_ram()).screen == ROOM_L3_BOSS:
                report["reached_4d"] = controller.reached_4d = True
            else:
                gate = controller.open_5d_up(env, assist, total)
                report["gate_5d"] = {
                    "ok": gate.get("ok"),
                    "method": gate.get("method"),
                    "clear": gate.get("clear"),
                    "attempts_n": len(gate.get("attempts") or []),
                    "attempts_tail": (gate.get("attempts") or [])[-12:],
                    "pre": gate.get("pre"),
                    "post": gate.get("post"),
                    "error": gate.get("error"),
                }
                report["reached_4d"] = bool(gate.get("ok"))
                if gate.get("ok"):
                    report["notes"].append(f"0x4d via {gate.get('method')}")
                else:
                    report["traps"].append(
                        "0x5d UP gate residual — walk/bomb approaches exhausted"
                    )

        if phase == "gate5d":
            report["ok"] = report["reached_4d"]
            report["final"] = room_fields(
                read_snapshot(env.get_ram()), env.get_ram()
            )
            report["total_frames"] = total[0]
            return _finish(report, tag, controller, assist=assist, tap=tap)

        # --- Manhandla confirm + optional kill ---
        if report["reached_4d"] or (
            read_snapshot(env.get_ram()).screen == ROOM_L3_BOSS
        ):
            report["reached_4d"] = controller.reached_4d = True
            idle(env, assist, total, 40)
            controller.confirm_manhandla(env)
            report["boss_room"] = room_fields(
                read_snapshot(env.get_ram()), env.get_ram()
            )
            report["manhandla_confirmed"] = controller.manhandla_confirmed
            for n in controller.notes:
                if n.startswith("Manhandla") and n not in report["notes"]:
                    report["notes"].append(n)

            if save_checkpoint:
                path = save_state(env, GAME_DIR, GAME, "Level3Boss")
                report["saved_boss"] = str(path)
                _provenance(
                    path,
                    start_state=start_state,
                    segment="level3_to_boss",
                    intervention=intervention,
                    trial=controller.report(),
                )

            if kill or phase in ("all", "boss", "kill"):
                fight = controller.fight_manhandla(
                    env, assist, total, max_frames=16000
                )
                report["fight"] = {
                    "ok": fight.get("ok"),
                    "tf04": fight.get("tf04"),
                    "frames": fight.get("frames"),
                    "dmg_events": fight.get("dmg_events"),
                    "error": fight.get("error"),
                    "notes": fight.get("notes"),
                    "log_tail": (fight.get("log") or [])[-15:],
                    "final": fight.get("final"),
                }
                report["dmg_events"] = int(fight.get("dmg_events") or 0)
                report["boss_beaten"] = bool(
                    fight.get("ok") and not fight.get("error")
                )
                report["tf04"] = bool(fight.get("tf04"))
                if report["tf04"] and save_checkpoint:
                    path = save_state(env, GAME_DIR, GAME, "Level3Complete")
                    report["saved_complete"] = str(path)
                    _provenance(
                        path,
                        start_state=start_state,
                        segment="level3_complete",
                        intervention=intervention,
                        trial=controller.report(),
                    )
                obs, *_ = env.step(nes_idle_action())
                total[0] += 1
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_boss_after.png")

        report["final"] = room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["total_frames"] = total[0]
        report["ok"] = bool(
            report["reached_4d"]
            or report["tf04"]
            or (report["boss_beaten"] and report["dmg_events"] > 0)
        )
        report["success_tier"] = (
            "tf04"
            if report["tf04"]
            else (
                "boss_kill"
                if report["boss_beaten"]
                else ("enter_4d" if report["reached_4d"] else "partial")
            )
        )
        return _finish(report, tag, controller, assist=assist, tap=tap)
    finally:
        if tap is not None and "video" not in report:
            report["video"] = tap.close()
        env.close()

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level3Raft")
    p.add_argument(
        "--infinite-life",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--trials", type=int, default=1)
    p.add_argument(
        "--to-boss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after entering 0x4d (default True); with --kill continues",
    )
    p.add_argument("--kill", action="store_true", help="Fight Manhandla after 0x4d")
    p.add_argument(
        "--phase",
        choices=("all", "to5d", "gate5d", "boss", "kill"),
        default="all",
        help="all=path+gate+fight; to5d/gate5d for short probes",
    )
    p.add_argument(
        "--poke-bombs",
        type=int,
        default=None,
        nargs="?",
        const=16,
        help=(
            "RECON bomb inventory poke (opt-in; default off). "
            "Pass --poke-bombs or --poke-bombs N (N=16 when flag alone)."
        ),
    )
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--tag", default="l3_to_boss")
    add_video_args(p)
    args = p.parse_args(argv)

    poke = args.poke_bombs  # None unless flag given
    video_path, video_config, intro_frames = resolve_video(
        args,
        default_path=RECORDINGS_DIR / f"{args.tag}.mp4",
    )
    kill = args.kill or args.phase in ("kill", "boss", "all")
    if args.phase == "all" and not args.kill and args.to_boss:
        kill = True
    if args.phase in ("to5d", "gate5d"):
        kill = False

    trials: list[dict] = []
    for i in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{i}"
        rep = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            to_boss=args.to_boss,
            kill=kill,
            poke_bombs=poke,
            save_checkpoint=args.save_state and i == 0,
            tag=tag,
            phase=args.phase,
            video_path=video_path,
            video_config=video_config,
            intro_frames=intro_frames,
        )
        trials.append(rep)
        print(
            f"trial{i}: ok={rep.get('ok')} tier={rep.get('success_tier')} "
            f"5d={rep.get('reached_5d')} 4d={rep.get('reached_4d')} "
            f"man={rep.get('manhandla_confirmed')} kill={rep.get('boss_beaten')} "
            f"tf04={rep.get('tf04')} dmg={rep.get('dmg_events')} "
            f"frames={rep.get('total_frames')} err={rep.get('error')}"
        )
        if rep.get("gate_5d"):
            g = rep["gate_5d"]
            print(
                f"  gate: ok={g.get('ok')} method={g.get('method')} "
                f"clear={g.get('clear')}"
            )
        for n in (rep.get("notes") or [])[:6]:
            print(f"  note: {n}")
        for t in (rep.get("traps") or [])[:6]:
            print(f"  trap: {t}")

    n4d = sum(1 for t in trials if t.get("reached_4d"))
    nkill = sum(1 for t in trials if t.get("boss_beaten"))
    ntf = sum(1 for t in trials if t.get("tf04"))
    rollup = {
        "trials": len(trials),
        "enter_4d": f"{n4d}/{len(trials)}",
        "boss_kill": f"{nkill}/{len(trials)}",
        "tf04": f"{ntf}/{len(trials)}",
        "intervention_class": "survival" if args.infinite_life else "clean",
        "poke_bombs": poke,
        "trial_summaries": [
            {
                "ok": t.get("ok"),
                "tier": t.get("success_tier"),
                "reached_5d": t.get("reached_5d"),
                "reached_4d": t.get("reached_4d"),
                "manhandla": t.get("manhandla_confirmed"),
                "boss_beaten": t.get("boss_beaten"),
                "tf04": t.get("tf04"),
                "dmg_events": t.get("dmg_events"),
                "error": t.get("error"),
                "method": (t.get("gate_5d") or {}).get("method"),
                "frames": t.get("total_frames"),
            }
            for t in trials
        ],
        "reports": [t.get("report_path") for t in trials],
    }
    out = RECORDINGS_DIR / f"{args.tag}_rollup.json"
    write_json_report(out, rollup)
    print(
        f"rollup: 4d={n4d}/{len(trials)} kill={nkill}/{len(trials)} "
        f"tf={ntf}/{len(trials)}"
    )
    print(f"rollup_path={out}")
    return 0 if n4d > 0 or nkill > 0 or ntf > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
