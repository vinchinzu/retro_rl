"""Assisted: Level2Boom → Dodongo 0x0e bomb-mouth → triforce bit 0x02 (rr-5dk).

Path knowledge lives in ``zelda_i.level2_boss_path``. This script is a thin
CLI/env wrapper (assisted-track labeling; not Clean STATUS).

Path (live 2026-08-07)::

    0x4f bomb-N → 0x3f → LEFT Moldorm 0x3e → UP ropes 0x2e clear → UP Goriya
    0x1e clear → **bomb-N @(120,101)** → boss **0x0e** → LEFT TF **0x0d**
    → south-band waypoints → ADDR_TRIFORCE & 0x02

Examples::

    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --trials 2 --save-state
    uv run python nes/zelda_i/scripts/run_level2_dodongo.py --infinite-life --from-state Level2_0E
    uv run python nes/zelda_i/scripts/run_level2_complete.py --infinite-life --trials 2 --save-state
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = _REPO_ROOT / "nes"
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.level2_boss_path import (
    BOMB_STAND_1E,
    LEVEL2_TF_BIT,
    ROOM_0E,
    ROOM_TF,
    resolve_path_start,
    run_boss_path,
    sample_snapshot,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, read_snapshot, read_u8


def run_once(
    *,
    start_state: str = "Level2Boom",
    infinite_life: bool = True,
    tag: str = "level2_dodongo",
    save_checkpoint: bool = False,
) -> dict:
    configure_headless()
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    track = "assisted" if infinite_life else "clean"
    timeline: list = []
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        env.reset()
        obs, *_ = env.step(nes_idle_action())
        if assist is not None:
            assist.apply_env(env, frame=0)
        boot = sample_snapshot(
            read_snapshot(env.get_ram()), env.get_ram(), event="boot"
        )
        timeline.append(boot)
        boot_sc = read_snapshot(env.get_ram()).screen
        start = resolve_path_start(start_state, boot_sc)

        def _save_0e() -> None:
            if save_checkpoint:
                save_state(env, GAME_DIR, GAME, "Level2_0E")

        path = run_boss_path(
            env,
            start=start,
            assist=assist,
            timeline=timeline,
            save_0e_checkpoint=_save_0e if start_state not in (
                "Level2_0E",
                "Level2_0D_PostBoss",
            )
            else None,
        )
        fight = path.get("fight") or {}
        tf_report = path.get("tf_report") or {}
        timeline = path.get("timeline") or timeline

        if not path.get("ok") and path.get("reason") and path.get("reason") != "tf_fail":
            return _fail(env, timeline, tag, track, str(path["reason"]))

        # Screenshots around fight / final when path reached boss.
        if start_state not in ("Level2_0D_PostBoss",) and boot_sc != ROOM_TF:
            try:
                obs, *_ = env.step(nes_idle_action())
                save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_after_fight.png")
            except Exception:
                pass

        timeline.append(
            {
                "event": "tf_phase",
                "ok": tf_report.get("ok"),
                "frames": tf_report.get("frames"),
                "phase": tf_report.get("phase"),
                "policy_live": tf_report.get("policy_live"),
            }
        )
        timeline.extend(tf_report.get("log") or [])

        ram = env.get_ram()
        snap = read_snapshot(ram)
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        ok = bool(tf & LEVEL2_TF_BIT)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_{'ok' if ok else 'fail'}.png")

        ck = None
        prov = None
        if ok and save_checkpoint:
            ck_path = save_state(env, GAME_DIR, GAME, "Level2Complete")
            ck = str(ck_path)
            selected_trial = {
                "ok": ok,
                "result": "TF_02",
                "triforce": tf,
                "triforce_bit_0x02": True,
                "tf_policy_live": tf_report.get("policy_live"),
                "tf_frames": tf_report.get("frames"),
                "fight_success": fight.get("success"),
                "final": sample_snapshot(snap, ram, event="final"),
                "start_state": start_state,
                "track": track,
            }
            prov = str(
                write_state_provenance(
                    ck_path,
                    source_state_path=GAME_DIR
                    / "custom_integrations"
                    / GAME
                    / f"{start_state}.state",
                    request={
                        "segment": "level2_dodongo_tf02",
                        "bead": "rr-5dk",
                        "track": track,
                        "start_state": start_state,
                        "natural_entry": False,
                    },
                    selected_trial=selected_trial,
                    natural_entry=False,
                )
            )

        out = {
            "bead": "rr-5dk",
            "result": (
                "TF_02"
                if ok
                else ("DODONGO_DEAD" if fight.get("success") else "PARTIAL")
            ),
            "ok": ok,
            "track": track,
            "intervention_class": "survival" if infinite_life else "clean",
            "start_state": start_state,
            "path_start": start.name,
            "triforce": tf,
            "triforce_bit_0x02": ok,
            "boss_room": f"0x{ROOM_0E:02x}",
            "tf_room": f"0x{ROOM_TF:02x}",
            "tf_room_note": "WEST/LEFT of boss after kill; walkthrough east residual",
            "dodongo_type": "0x32",
            "bomb_wall_1e": {
                "stand": list(BOMB_STAND_1E),
                "face": "UP",
                "to": f"0x{ROOM_0E:02x}",
            },
            "tf_policy_live": tf_report.get("policy_live"),
            "fight": {k: v for k, v in fight.items() if k != "log"},
            "timeline": timeline,
            "final": sample_snapshot(snap, ram, event="final"),
            "checkpoint": ck,
            "provenance": prov,
            "natural_entry": False,
            "status_promote": False,
            "library": "zelda_i.level2_boss_path",
            "evidence": [
                f"recordings/{tag}.json",
                f"recordings/{tag}_boss_entry.png",
                f"recordings/{tag}_{'ok' if ok else 'fail'}.png",
            ],
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
        return out
    finally:
        env.close()


def _fail(env, timeline, tag, track, reason: str) -> dict:
    s = read_snapshot(env.get_ram())
    out = {
        "bead": "rr-n5i",
        "result": "FAIL",
        "ok": False,
        "reason": reason,
        "track": track,
        "timeline": timeline,
        "final": sample_snapshot(s, env.get_ram(), event="fail"),
        "triforce_bit_0x02": False,
        "library": "zelda_i.level2_boss_path",
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", out)
    try:
        obs = env.render()
        save_rgb_png(obs, RECORDINGS_DIR / f"{tag}_fail_{reason}.png")
    except Exception:
        pass
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-state", default="Level2Boom")
    p.add_argument("--infinite-life", action="store_true", default=True)
    p.add_argument("--no-infinite-life", action="store_true")
    p.add_argument("--tag", default="level2_dodongo")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--save-state", action="store_true")
    args = p.parse_args()
    inf = not args.no_infinite_life
    results = []
    for t in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{t}"
        r = run_once(
            start_state=args.from_state,
            infinite_life=inf,
            tag=tag,
            save_checkpoint=args.save_state and t == 0,
        )
        results.append(r)
        print(
            f"trial{t}: result={r.get('result')} ok={r.get('ok')} "
            f"tf={r.get('triforce')} fight={r.get('fight', {}).get('success')} "
            f"final_sc={(r.get('final') or {}).get('sc')}"
        )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"summary: {n_ok}/{len(results)} TF 0x02")
    write_json_report(
        RECORDINGS_DIR / f"{args.tag}_summary.json",
        {
            "bead": "rr-5dk",
            "ok": n_ok == len(results) and n_ok > 0,
            "ok_count": n_ok,
            "trials": len(results),
            "results": [r.get("result") for r in results],
            "track": "assisted" if inf else "clean",
            "start_state": args.from_state,
            "triforce_bit_0x02": n_ok > 0,
            "status_promote": False,
            "natural_entry": False,
            "library": "zelda_i.level2_boss_path",
            "checkpoint": next(
                (r.get("checkpoint") for r in results if r.get("checkpoint")),
                None,
            ),
            "trial_details": [
                {
                    "trial": i,
                    "ok": r.get("ok"),
                    "result": r.get("result"),
                    "triforce": r.get("triforce"),
                    "tf_policy_live": r.get("tf_policy_live"),
                    "final": r.get("final"),
                    "checkpoint": r.get("checkpoint"),
                }
                for i, r in enumerate(results)
            ],
        },
    )


if __name__ == "__main__":
    main()
