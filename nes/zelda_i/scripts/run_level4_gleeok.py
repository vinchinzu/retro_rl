"""Assisted dual-green: Level4GleeokEnter → Gleeok melee → HC → TF 0x08.

Uses ``Level4GleeokFightController`` (live IDs only). Survival assist for
first-pass; not Clean STATUS.

Examples::

    uv run python nes/zelda_i/scripts/run_level4_gleeok.py \\
        --infinite-life --trials 2 --save-state --tag l4_rvae_gleeok_tf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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
from zelda_i.level4_boss_combat import (
    Level4GleeokFightController,
    level4_tf08,
)
from zelda_i.level4_dungeon import ROOM_L4_GLEEOK_13
from zelda_i.level4_overworld import LEVEL4
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot

def _provenance(
    path: Path,
    *,
    start_state: str,
    segment: str,
    intervention: str,
    trial: dict[str, Any],
    trial_i: int,
) -> None:
    write_state_provenance(
        path,
        source_state_path=(
            GAME_DIR / "custom_integrations" / GAME / f"{start_state}.state"
        ),
        request={
            "bead": "rr-rvae",
            "segment": segment,
            "track": "assisted" if intervention == "survival" else "clean",
            "intervention_class": intervention,
            "trial": trial_i,
            "natural_entry": False,
            "start_state": start_state,
        },
        selected_trial=trial,
        natural_entry=False,
    )

def run_once(
    *,
    start_state: str = "Level4GleeokEnter",
    infinite_life: bool = True,
    save_checkpoint: bool = False,
    tag: str = "l4_gleeok",
    trial_i: int = 0,
) -> dict[str, Any]:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    intervention = "survival" if infinite_life else "clean"
    total = [0]
    ctl = Level4GleeokFightController(tag=f"{tag}_t{trial_i}")
    report: dict[str, Any] = {
        "ok": False,
        "track": track,
        "intervention_class": intervention,
        "start_state": start_state,
        "tag": tag,
        "trial": trial_i,
        "bead": "rr-rvae",
        "boss_beaten": False,
        "hc_collected": False,
        "tf08": False,
        "notes": [],
    }
    try:
        obs, _ = reset_obs(env)
        for i in range(3):
            obs, *_ = env.step(nes_idle_action())
            total[0] += 1
            if assist is not None:
                assist.apply_env(env, frame=total[0])
        idle(env, assist, total, 10)
        entry = room_fields(read_snapshot(env.get_ram()), env.get_ram())
        report["entry"] = entry
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_start.png")

        if not (
            entry["level"] == LEVEL4
            and entry["screen"] == ROOM_L4_GLEEOK_13
            and entry["mode"] in (PLAY_MODE, 5)
        ):
            report["error"] = (
                f"expected Level4GleeokEnter 0x13; got L{entry['level']} "
                f"sc={entry['sc']} mode={entry['mode']}"
            )
            return report

        fight = ctl.run(env, assist, total)
        report["fight"] = {
            "ok": fight.get("ok"),
            "tf08": fight.get("tf08"),
            "boss_beaten": fight.get("boss_beaten"),
            "hc_collected": fight.get("hc_collected"),
            "frames": fight.get("frames"),
            "dmg_events": fight.get("dmg_events"),
            "error": fight.get("error"),
            "notes": fight.get("notes"),
            "log_tail": (fight.get("log") or [])[-15:],
            "final": fight.get("final"),
        }
        report["notes"].extend(fight.get("notes") or [])
        report["boss_beaten"] = bool(fight.get("boss_beaten"))
        report["hc_collected"] = bool(fight.get("hc_collected"))
        report["tf08"] = bool(fight.get("tf08") or level4_tf08(env.get_ram()))
        report["ok"] = bool(fight.get("ok") and report["tf08"])
        report["total_frames"] = total[0]
        report["controller"] = ctl.report()
        final = fight.get("final") or room_fields(
            read_snapshot(env.get_ram()), env.get_ram()
        )
        report["final"] = final

        obs, *_ = env.step(nes_idle_action())
        total[0] += 1
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_t{trial_i}_final.png")

        if save_checkpoint and report["ok"]:
            name = "Level4Complete" if trial_i == 0 else f"Level4Complete_t{trial_i}"
            path = save_state(env, GAME_DIR, GAME, name)
            report["checkpoint"] = str(path)
            _provenance(
                path,
                start_state=start_state,
                segment="gleeok_fight_tf",
                intervention=intervention,
                trial={
                    "ok": True,
                    "frames": fight.get("frames"),
                    "final": final,
                    "notes": report["notes"],
                    "tf08": True,
                },
                trial_i=trial_i,
            )
            # Also keep boss-cleared mid if HC path happened (optional).
            if report["boss_beaten"] and trial_i == 0:
                # Best-effort: state is already post-TF; Level4BossCleared from
                # earlier probe may exist. Do not overwrite with TF fanfare.
                pass

        return report
    finally:
        env.close()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-state",
        default="Level4GleeokEnter",
        help="Start checkpoint (default Level4GleeokEnter)",
    )
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival assist (required for first-pass dual-green)",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--tag", default="l4_rvae_gleeok_tf")
    args = parser.parse_args()

    trials: list[dict[str, Any]] = []
    for i in range(args.trials):
        print(f"=== trial {i} ===")
        r = run_once(
            start_state=args.from_state,
            infinite_life=args.infinite_life,
            save_checkpoint=args.save_state,
            tag=args.tag,
            trial_i=i,
        )
        summary = {
            "ok": r.get("ok"),
            "tf08": r.get("tf08"),
            "frames": (r.get("fight") or {}).get("frames"),
            "hc": (r.get("final") or {}).get("heart_containers"),
            "triforce": (r.get("final") or {}).get("triforce"),
            "error": r.get("error") or (r.get("fight") or {}).get("error"),
            "notes": r.get("notes"),
        }
        print("RESULT", summary)
        trials.append(r)

    dual = all(t.get("ok") and t.get("tf08") for t in trials) and len(trials) >= 2
    report = {
        "bead": "rr-rvae",
        "segment": "gleeok_fight_hc_tf",
        "from": args.from_state,
        "dual_green": dual,
        "ok": dual or (len(trials) == 1 and trials[0].get("ok")),
        "track": "assisted" if args.infinite_life else "clean",
        "trials": [
            {
                "trial": t.get("trial"),
                "ok": t.get("ok"),
                "tf08": t.get("tf08"),
                "frames": (t.get("fight") or {}).get("frames"),
                "total_frames": t.get("total_frames"),
                "hc_collected": t.get("hc_collected"),
                "final": t.get("final"),
                "notes": t.get("notes"),
                "checkpoint": t.get("checkpoint"),
                "error": t.get("error") or (t.get("fight") or {}).get("error"),
            }
            for t in trials
        ],
        "tag": args.tag,
        "checkpoints": ["Level4Complete"] if args.save_state else [],
    }
    out = RECORDINGS_DIR / f"{args.tag}_dual.json"
    write_json_report(out, report)
    print("DUAL", dual, "→", out)
    return 0 if (dual or (len(trials) == 1 and trials[0].get("ok"))) else 1

if __name__ == "__main__":
    raise SystemExit(main())
