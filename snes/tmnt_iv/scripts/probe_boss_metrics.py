"""Quick headless boss-fight metrics from a save state.

Supports heal modes (default: emergency) to match the production low-assist
run and the lab.slash_lab trial runner:

  - ``none``: no HP writes (pure survival stress)
  - ``emergency``: restore to 80 when HP <= 16 (production-like)
"""

from __future__ import annotations

import argparse
from typing import Any

from tmnt_iv.run.trial import (
    CLEAN_CONTRACT,
    TrialContract,
    TrialEntry,
    TrialLimits,
    TrialObjective,
    run_trial,
)


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
    if heal_mode == "emergency":
        contract = TrialContract(
            name="assisted",
            emergency_hp=True,
            iframe_hold=False,
            allowed_write_keys=frozenset({"player_hp"}),
        )
    else:
        contract = CLEAN_CONTRACT
    stall_starts: list[dict[str, Any]] = []
    stall_x_hist: dict[str, int] = {}
    prev_cam = 0
    prev_reason = ""
    saw_form1 = False
    started = False

    def on_frame(ctx: Any) -> None:
        nonlocal prev_cam, prev_reason, saw_form1, started
        state = ctx.state
        if any(e.kind == 0x52 for e in state.living_enemies):
            saw_form1 = True
        if not started:
            prev_cam = state.camera_x
            started = True
        if not trace_stall:
            prev_cam = state.camera_x
            prev_reason = ctx.reason
            return
        cam_delta = state.camera_x - prev_cam
        reason = ctx.reason
        if reason.startswith("stall_") and not prev_reason.startswith("stall_"):
            sample = {
                "frame": ctx.frame,
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

    kind: str = "stage_advance" if stop_stage_gt is not None else "boss_fade"
    result = run_trial(
        TrialEntry(kind="state", state_name=state_name),
        TrialObjective(kind=kind, stop_stage_gt=stop_stage_gt),
        contract,
        TrialLimits(max_frames=max_frames),
        on_frame=on_frame,
    )
    outcome = result.outcome
    if outcome == "forbidden_action":
        outcome = "forbidden_a"
    report = {
        "state": state_name,
        "outcome": outcome,
        "frames": result.total_frames,
        "start_stage": result.entry_stage,
        "end_stage": result.end_stage,
        "start_hp": result.start_hp,
        "end_hp": result.end_hp,
        "min_hp": result.min_hp,
        "damage_taken": result.damage_taken,
        "max_hit": result.max_hit,
        "heals": result.emergency_hp_writes,
        "lives": result.lives,
        "life_losses": result.life_losses,
        "event": hex(result.end_event),
        "top_reasons": result.top_reasons,
        "saw_form1": saw_form1,
        "stall_hist": sorted(stall_x_hist.items(), key=lambda kv: -kv[1])[:16],
        "emergency_hp_writes": result.emergency_hp_writes,
        "iframe_writes": result.iframe_writes,
        "assist": result.assist,
        "integrity": result.integrity,
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
