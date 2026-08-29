"""Quick headless boss-fight metrics from a save state.

Supports heal modes (default: emergency) to match the production low-assist
run and the lab.slash_lab trial runner:

  - ``none``: no HP writes (pure survival stress)
  - ``emergency``: restore to 80 when HP <= 16 (production-like)
"""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.segment_runner import configure_headless  # noqa: E402
from tmnt_iv.assist import apply_emergency_hp  # noqa: E402
from tmnt_iv.observe import HpDelta, policy_input  # noqa: E402
from tmnt_iv.paths import GAME, GAME_DIR  # noqa: E402
from tmnt_iv.policy import Stage1Policy  # noqa: E402
from tmnt_iv.ram import parse_game_state  # noqa: E402


def run_probe(
    *,
    state_name: str,
    max_frames: int = 12000,
    stop_stage_gt: int | None = None,
    heal_mode: str = "emergency",
    trace_stall: bool = False,
) -> dict[str, Any]:
    """Fight from ``state_name`` until timeout, KO, or stage advance.

    heal_mode:
      - ``none``: no HP writes (pure survival stress)
      - ``emergency``: restore to 80 when HP <= 16 (production-like)
    """
    configure_headless()
    env = make_env(GAME, state_name, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    reset_obs(env)
    start = parse_game_state(env.get_ram(), frame=0)
    meter = HpDelta.start(start.health)
    prev_lives = start.lives
    heals = 0
    reasons: dict[str, int] = {}
    boss_hp_start = int(start.extras.get("boss_hp", 0))
    final = start
    outcome = "timeout"
    stall_starts: list[dict[str, Any]] = []
    stall_x_hist: dict[str, int] = {}
    prev_cam = start.camera_x
    prev_reason = ""
    saw_form1 = any(e.kind == 0x52 for e in start.living_enemies)
    try:
        for frame in range(1, max_frames + 1):
            state = parse_game_state(env.get_ram(), frame=frame)
            final = state
            if any(e.kind == 0x52 for e in state.living_enemies):
                saw_form1 = True
            meter.note(state.health)

            # Emergency heal assist (production-like).
            if heal_mode == "emergency":
                if apply_emergency_hp(env, state.health):
                    heals += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    final = state
                    meter.note(state.health)

            if state.lives < prev_lives:
                outcome = "life_loss"
                break
            prev_lives = state.lives
            if stop_stage_gt is not None and state.stage > stop_stage_gt:
                outcome = "stage_advance"
                break
            if (
                start.boss_active
                and not state.boss_active
                and int(state.extras.get("event", 0)) in {0x0B, 0x19}
            ):
                outcome = "boss_down"
                # keep a few frames for fade
            if outcome == "boss_down" and frame > 0 and frame % 60 == 0:
                if state.stage > start.stage or int(
                    state.extras.get("event", 0)
                ) in {0x19, 0x04}:
                    outcome = "cleared"
                    break
            action, reason = policy_input(policy, state)
            reasons[reason] = reasons.get(reason, 0) + 1
            cam_delta = state.camera_x - prev_cam
            if reason.startswith("stall_") and not prev_reason.startswith("stall_"):
                sample = {
                    "frame": frame,
                    "reason": reason,
                    "x": state.player_x,
                    "y": state.player_y,
                    "cam": state.camera_x,
                    "dcam": cam_delta,
                    "event": int(state.extras.get("event", -1)),
                    "boss": int(state.boss_active),
                    "n": len(state.living_enemies),
                    "anim": int(state.extras.get("anim", -1)),
                    "form1": int(saw_form1),
                }
                if len(stall_starts) < 40:
                    stall_starts.append(sample)
                bucket = (
                    f"x{state.player_x // 16 * 16}"
                    f"_y{state.player_y // 8 * 8}"
                    f"_dcam{min(max(cam_delta, -2), 2)}"
                    f"_f1{int(saw_form1)}"
                )
                stall_x_hist[bucket] = stall_x_hist.get(bucket, 0) + 1
            prev_reason = reason
            prev_cam = state.camera_x
            if action[8]:
                outcome = "forbidden_a"
                break
            env.step(action)
        else:
            outcome = "timeout"
    finally:
        env.close()

    top = sorted(reasons.items(), key=lambda kv: -kv[1])[:12]
    report = {
        "state": state_name,
        "outcome": outcome,
        "frames": final.frame,
        "start_stage": start.stage,
        "end_stage": final.stage,
        "start_hp": start.health,
        "end_hp": final.health,
        "min_hp": meter.min_hp,
        "damage_taken": meter.damage,
        "max_hit": meter.max_hit,
        "heals": heals,
        "lives": f"{start.lives}->{final.lives}",
        "boss_hp": f"{boss_hp_start}->{int(final.extras.get('boss_hp', 0))}",
        "event": hex(int(final.extras.get("event", -1))),
        "top_reasons": top,
        "saw_form1": saw_form1,
        "stall_hist": sorted(stall_x_hist.items(), key=lambda kv: -kv[1])[:16],
    }
    if trace_stall:
        report["stall_starts"] = stall_starts
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", help="save state name")
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument(
        "--stop-stage-gt",
        type=int,
        default=None,
        help="stop when stage byte exceeds this value",
    )
    parser.add_argument(
        "--heal",
        choices=["none", "emergency"],
        default="emergency",
        help="HP assist mode (default: emergency, production-like)",
    )
    parser.add_argument(
        "--trace-stall",
        action="store_true",
        help="include per-start stall samples (x/y/cam/form-1)",
    )
    args = parser.parse_args(argv)
    report = run_probe(
        state_name=args.state,
        max_frames=args.max_frames,
        stop_stage_gt=args.stop_stage_gt,
        heal_mode=args.heal,
        trace_stall=args.trace_stall,
    )
    print(report)
    return 0 if report["outcome"] not in {"life_loss", "forbidden_a"} else 1

if __name__ == "__main__":
    raise SystemExit(main())
