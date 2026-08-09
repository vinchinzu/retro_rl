"""Glass Joe bout segment from Match1 / Level1.

Milestones:

- ``knockdown``: score ≥1 opponent knockdown (M2).
- ``win`` / ``bout``: full bout win — KO / TKO / decision (M3, default).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/punch_out/scripts/run_glass_joe.py --goal win --trials 3
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from punch_out.paths import GAME, GAME_DIR, RECORDINGS_DIR
from punch_out.policy import BoutMode, GlassJoePolicy
from punch_out.ram import (
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_HEALTH,
    ADDR_ROUND,
    FIGHT_BETWEEN,
    hearts,
    is_match_live,
    parse_game_state,
    stars,
)
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

DEFAULT_STATE = "Match1"
DEFAULT_GOAL = "win"
# Full bout (3 rounds + counts) needs headroom beyond the old 20k default.
DEFAULT_MAX_FRAMES = 25000
WIN_OUTCOMES = frozenset({"tko_win", "ko_win", "decision_win"})


def normalize_goal(goal: str) -> str:
    return "win" if goal == "bout" else goal


def classify_bout_outcome(
    *,
    goal: str,
    policy: GlassJoePolicy,
    opp_hp: int,
    fight: int,
    rnd: int,
) -> str | None:
    """Return a terminal outcome label, or None if the bout should continue."""
    if goal == "knockdown" and policy.opp_kd >= 1:
        return "knockdown"
    if policy.mac_kd >= 3:
        return "loss_tko"
    if goal != "win":
        return None

    # Three knockdowns → TKO (Joe stays down / between after count).
    if policy.opp_kd >= 3 and (
        fight == FIGHT_BETWEEN
        or (
            opp_hp == 0
            and policy.mode == BoutMode.WATCH_KD
            and policy.mode_t > 200
        )
    ):
        return "tko_win"

    # Verified M3 path: long count-out KO (e.g. second KD on Glass Joe).
    if (
        policy.opp_kd >= 1
        and opp_hp == 0
        and policy.mode == BoutMode.WATCH_KD
        and policy.mode_t > 650
    ):
        return "ko_win"

    # Decision: survived to post-R3 with more opp KDs.
    if (
        fight == FIGHT_BETWEEN
        and rnd >= 3
        and policy.mode_t > 120
        and policy.opp_kd > policy.mac_kd
    ):
        return "decision_win"

    return None


def _ensure_match_live(env, obs, max_wait: int = 2000):
    """If loaded pre-clock, idle until the bout clock is live.

    Returns ``(obs, frames_waited)``.
    """
    for waited in range(max_wait):
        ram = env.get_ram()
        if is_match_live(ram):
            return obs, waited
        step = env.step(nes_idle_action())
        obs = step[0] if isinstance(step, tuple) else step
    return obs, max_wait


def _maybe_open_video(out: Path, *, record: bool, obs):
    """Optionally start a FrameVideoWriter; returns writer or None."""
    if not record:
        return None
    try:
        from retro_harness.video import FrameVideoWriter
    except Exception as exc:  # pragma: no cover - optional path
        print(f"video unavailable: {exc}")
        return None
    import numpy as np

    frame = np.asarray(obs)
    if frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    path = out / "bout.mp4"
    try:
        return FrameVideoWriter(
            path,
            width=w,
            height=h,
            fps=60,
            scale=2,
            crf=20,
            preset="veryfast",
            footer=False,
            audio_rate=None,
        )
    except Exception as exc:  # pragma: no cover
        print(f"video open failed: {exc}")
        return None


def _record_terminal(
    *,
    env,
    obs,
    out: Path,
    frame: int,
    outcome: str,
    screenshots: list[str],
    saved: list[str],
    save_clear: bool,
) -> None:
    """Snapshot + optional clear-state for a terminal outcome."""
    tag = {
        "knockdown": "kd1",
        "loss_tko": "loss",
        "tko_win": "tko",
        "ko_win": "ko",
        "decision_win": "decision",
    }.get(outcome, outcome)
    png = save_rgb_png(obs, out / f"gj_{frame:04d}_{tag}.png")
    screenshots.append(png.name)
    if not save_clear:
        return
    if outcome == "knockdown":
        path = save_state(env, GAME_DIR, GAME, "GlassJoe_KD1")
        saved.append(path.name)
    elif outcome in WIN_OUTCOMES:
        path = save_state(env, GAME_DIR, GAME, "GlassJoe_Clear")
        saved.append(path.name)


def run_glass_joe(
    *,
    state_name: str = DEFAULT_STATE,
    goal: str = DEFAULT_GOAL,
    max_frames: int = DEFAULT_MAX_FRAMES,
    out_dir: Path | None = None,
    save_clear: bool = True,
    record: bool = False,
    trial: int | None = None,
) -> dict[str, Any]:
    """Load checkpoint, run GlassJoePolicy until goal or fail."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    if state_name not in available:
        if "Level1" in available:
            state_name = "Level1"
        else:
            raise SystemExit(f"missing state {state_name}; have {available[:12]}")

    goal = normalize_goal(goal)

    base = out_dir or (RECORDINGS_DIR / "glass_joe")
    out = base if trial is None else base / f"trial_{trial:02d}"
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    writer = None
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        obs, waited = _ensure_match_live(env, obs)
        writer = _maybe_open_video(out, record=record, obs=obs)

        policy = GlassJoePolicy()
        screenshots: list[str] = []
        saved: list[str] = []
        png = save_rgb_png(obs, out / "gj_0000_start.png")
        screenshots.append(png.name)
        if writer is not None:
            writer.write(obs)

        outcome = "timeout"
        end_frame = 0
        final_mac = 96
        final_opp = 96
        kd_shots = 0

        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            fa = policy.tick(ram)
            step = env.step(fa.action)
            obs = step[0] if isinstance(step, tuple) else step
            end_frame = frame
            ram = env.get_ram()
            final_mac = int(ram[ADDR_HEALTH])
            final_opp = int(ram[ADDR_OPP_HEALTH])
            fight = int(ram[ADDR_FIGHT_FLAG])
            rnd = int(ram[ADDR_ROUND])

            if writer is not None:
                writer.write(obs)

            if policy.opp_kd > kd_shots:
                kd_shots = policy.opp_kd
                png = save_rgb_png(obs, out / f"gj_{frame:04d}_kd{kd_shots}.png")
                screenshots.append(png.name)

            terminal = classify_bout_outcome(
                goal=goal,
                policy=policy,
                opp_hp=final_opp,
                fight=fight,
                rnd=rnd,
            )
            if terminal is not None:
                outcome = terminal
                _record_terminal(
                    env=env,
                    obs=obs,
                    out=out,
                    frame=frame,
                    outcome=outcome,
                    screenshots=screenshots,
                    saved=saved,
                    save_clear=save_clear,
                )
                break
        else:
            png = save_rgb_png(obs, out / f"gj_{end_frame:04d}_timeout.png")
            screenshots.append(png.name)

        state = parse_game_state(env.get_ram(), frame=end_frame)
        success = (
            outcome == "knockdown"
            if goal == "knockdown"
            else outcome in WIN_OUTCOMES
        )
        report = {
            "game": GAME,
            "state": state_name,
            "goal": goal,
            "outcome": outcome,
            "success": success,
            "frames": end_frame,
            "waited_for_clock": waited,
            "opp_kd": policy.opp_kd,
            "mac_kd": policy.mac_kd,
            "hits": policy.hits,
            "mac_health": final_mac,
            "opp_health": final_opp,
            "hearts": hearts(env.get_ram()),
            "stars": stars(env.get_ram()),
            "round": int(env.get_ram()[ADDR_ROUND]),
            "mode": state.mode.name,
            "policy_mode": policy.mode.name,
            "reasons": dict(
                sorted(policy.reasons.items(), key=lambda kv: -kv[1])[:25]
            ),
            "screenshots": screenshots,
            "saved_states": saved,
            "trial": trial,
            "video": "bout.mp4" if writer is not None else None,
        }
        write_json_report(out / "report.json", report)
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        env.close()

    print(
        f"GLASS_JOE goal={goal} outcome={outcome} success={report['success']} "
        f"frames={end_frame} opp_kd={policy.opp_kd} mac_kd={policy.mac_kd} "
        f"mac={final_mac} opp={final_opp} state={state_name}"
        + (f" trial={trial}" if trial is not None else "")
    )
    return report


def run_trials(
    *,
    trials: int,
    state_name: str,
    goal: str,
    max_frames: int,
    out_dir: Path | None,
    save_clear: bool,
    record: bool,
) -> dict[str, Any]:
    """Run N independent bouts; write summary under out_dir."""
    goal = normalize_goal(goal)
    base = out_dir or (RECORDINGS_DIR / "glass_joe")
    base.mkdir(parents=True, exist_ok=True)
    reports = []
    for i in range(1, trials + 1):
        rep = run_glass_joe(
            state_name=state_name,
            goal=goal,
            max_frames=max_frames,
            out_dir=base,
            save_clear=save_clear and i == 1,
            record=record and i == 1,
            trial=i,
        )
        reports.append(rep)
    wins = sum(1 for r in reports if r["success"])
    summary = {
        "game": GAME,
        "goal": goal,
        "state": state_name,
        "trials": trials,
        "wins": wins,
        "success_rate": wins / trials if trials else 0.0,
        "success": wins >= trials and trials > 0,
        "outcomes": [r["outcome"] for r in reports],
        "frames": [r["frames"] for r in reports],
        "opp_kd": [r["opp_kd"] for r in reports],
        "mac_kd": [r["mac_kd"] for r in reports],
        "reports": [f"trial_{i:02d}/report.json" for i in range(1, trials + 1)],
    }
    write_json_report(base / "summary.json", summary)
    print(
        f"SUMMARY trials={trials} wins={wins}/{trials} "
        f"outcomes={summary['outcomes']} success={summary['success']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument(
        "--goal",
        choices=("knockdown", "bout", "win"),
        default=DEFAULT_GOAL,
        help="knockdown = first KD; win/bout = full bout clear (M3, default)",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Independent full runs (for M3 ≥3/3 verification)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write bout.mp4 for trial 1 (requires ffmpeg)",
    )
    args = parser.parse_args()
    if args.trials > 1:
        summary = run_trials(
            trials=args.trials,
            state_name=args.state,
            goal=args.goal,
            max_frames=args.max_frames,
            out_dir=args.out_dir,
            save_clear=not args.no_save,
            record=args.record,
        )
        raise SystemExit(0 if summary["success"] else 1)

    report = run_glass_joe(
        state_name=args.state,
        goal=args.goal,
        max_frames=args.max_frames,
        out_dir=args.out_dir,
        save_clear=not args.no_save,
        record=args.record,
    )
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
