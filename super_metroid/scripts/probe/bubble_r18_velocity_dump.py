#!/usr/bin/env python3
"""R18: pure velocity dumps at fire / post_run / wall-approach + short WJ probe.

Not hop GREEN. Captures save-states (with velocity) from the natural CATH-04
source via lower + max-left seat, then optionally runs the product fire recipe
and logs p132 / Phase D marks.

```bash
uv run python super_metroid/scripts/probe/bubble_r18_velocity_dump.py
uv run python super_metroid/scripts/probe/bubble_r18_velocity_dump.py --from-seat \\
  super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_runway_clear_x27.state
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    save_dev_state,
)
from super_metroid.paths import GAME_DIR  # noqa: E402
from super_metroid.ram import parse_env_state, read_wram_u16  # noqa: E402
from super_metroid.routes.controller_common import hold  # noqa: E402
from super_metroid.routes.kpdr.bubble_mountain import (  # noqa: E402
    BubbleTrack,
    bubble_is_true_ground,
    bubble_lower_to_mid_pin,
    bubble_on_save_runway,
    bubble_phase_d_top_band,
)
from super_metroid.routes.kpdr.bubble_mountain_primitives import (  # noqa: E402
    bubble_double_walljump_r15,
    bubble_prepare_fire_run,
    bubble_runway_dash,
    bubble_seat_max_left_fire,
    bubble_spin_glide,
)

SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEBUG = GAME_DIR / "debug"
DEFAULT_SOURCE = SCRATCH / "post_rising_tide_to_bubble_pure.state"
ROOM_BUBBLE = 0xACB3


class DumpSession:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, frame=0, mode="nav")
        self.last_action = None
        self.segment_marks: list[dict[str, object]] = []
        self.trace: list[dict[str, int]] = []

    def mark(self, name: str) -> None:
        st = self.state
        self.segment_marks.append(
            {
                "segment": name,
                "frame": self.frame,
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "pose": int(st.pose),
                "vx": int(st.velocity_x),
                "vy": int(st.velocity_y),
                "xsub": read_wram_u16(self.env, 0x0AF8),
                "ysub": read_wram_u16(self.env, 0x0AFC),
            }
        )

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self.last_action = action
        st = self.state
        self.trace.append(
            {
                "f": self.frame,
                "x": int(st.samus_x),
                "y": int(st.samus_y),
                "p": int(st.pose),
                "vx": int(st.velocity_x),
                "vy": int(st.velocity_y),
            }
        )
        return self.state


def _snap(st: Any, env: Any | None = None) -> dict[str, int]:
    out = {
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "p": int(st.pose),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
        "room": int(st.room_id),
    }
    if env is not None:
        out["xsub"] = read_wram_u16(env, 0x0AF8)
        out["ysub"] = read_wram_u16(env, 0x0AFC)
    return out


def _metrics(trace: list[dict[str, int]]) -> dict[str, Any]:
    if not trace:
        return {}
    min_y = min(t["y"] for t in trace)
    max_x = max(t["x"] for t in trace)
    mx200 = 0
    first_wj = None
    n_p132 = 0
    saw84 = False
    top = False
    for t in trace:
        if t["y"] <= 200:
            mx200 = max(mx200, t["x"])
        if t["p"] == 132:
            n_p132 += 1
            if first_wj is None:
                first_wj = {"f": t["f"], "x": t["x"], "y": t["y"]}
        if t["p"] in (79, 83, 84):
            saw84 = True
        if t["y"] <= 200 and t["x"] >= 300:
            top = True
    return {
        "min_y": min_y,
        "max_x": max_x,
        "mx200": mx200,
        "top": top,
        "n_p132": n_p132,
        "first_wj": first_wj,
        "saw84": saw84,
        "frames": len(trace),
    }


def dump_from_source(
    source: Path,
    *,
    seat_out: Path,
    postrun_out: Path,
    wall_out: Path,
    json_out: Path,
    run_fire: bool = True,
) -> dict[str, Any]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, source)
    st = parse_env_state(env, frame=0, mode="nav")
    assist.apply(env.data, st)
    for _ in range(2):
        env.step(idle_action())
        st = parse_env_state(env, frame=0, mode="nav")
        assist.apply(env.data, st)

    sess = DumpSession(env, assist)
    track = BubbleTrack(
        label="r18dump",
        min_y=int(sess.state.samus_y),
        max_x=int(sess.state.samus_x),
    )
    sess.mark("ENTRY")
    bubble_lower_to_mid_pin(sess, track)
    sess.mark("AFTER_LOWER")

    seated = False
    if bubble_on_save_runway(sess.state) or (
        380 <= int(sess.state.samus_y) <= 430
        and 20 <= int(sess.state.samus_x) <= 100
    ):
        seated = bubble_seat_max_left_fire(sess, track)
    sess.mark("FIRE_SEAT")
    save_dev_state(env, seat_out)

    report: dict[str, Any] = {
        "source": str(source),
        "entry": sess.segment_marks[0] if sess.segment_marks else None,
        "after_lower": next(
            (m for m in sess.segment_marks if m["segment"] == "AFTER_LOWER"), None
        ),
        "fire_seat": _snap(sess.state, env),
        "seated_max_left": seated,
        "runway": bool(bubble_on_save_runway(sess.state)),
        "true_ground": bool(bubble_is_true_ground(sess.state)),
        "states": {
            "fire_seat": str(seat_out),
            "post_run": str(postrun_out),
            "wall_approach": str(wall_out),
        },
    }

    if run_fire and seated:
        fire_trace_start = len(sess.trace)
        bubble_prepare_fire_run(sess, track, y_clear=True, crouch=False)
        bubble_runway_dash(sess, track, frames=21, arm_pump=True)
        sess.mark("POST_RUN")
        save_dev_state(env, postrun_out)
        report["post_run"] = _snap(sess.state, env)

        bubble_spin_glide(sess, track, frames=83)
        sess.mark("POST_SPIN")
        save_dev_state(env, wall_out)
        report["wall_approach"] = _snap(sess.state, env)

        bubble_double_walljump_r15(sess, track)
        sess.mark("AFTER_DWJ")
        fire_trace = sess.trace[fire_trace_start:]
        metrics = _metrics(fire_trace)
        report["fire_metrics"] = metrics
        report["end"] = _snap(sess.state, env)
        report["top_reached"] = bool(track.top_reached or metrics.get("top"))
    else:
        report["fire_metrics"] = None
        report["note"] = "seat not max-left or --no-fire; dumps seat only"

    report["marks"] = sess.segment_marks
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    return report


def probe_seat(seat: Path, json_out: Path) -> dict[str, Any]:
    """Run product fire recipe from an existing seat dump."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, seat)
    st = parse_env_state(env, frame=0, mode="nav")
    assist.apply(env.data, st)
    for _ in range(2):
        env.step(idle_action())
        st = parse_env_state(env, frame=0, mode="nav")
        assist.apply(env.data, st)

    sess = DumpSession(env, assist)
    track = BubbleTrack(
        label="r18seat",
        min_y=int(sess.state.samus_y),
        max_x=int(sess.state.samus_x),
    )
    start = _snap(sess.state, env)
    from super_metroid.routes.kpdr.bubble_mountain_primitives import (
        bubble_save_runway_fire_recipe,
    )

    bubble_save_runway_fire_recipe(
        sess, track, y_clear=True, crouch=False, arm_pump=True, wj_count=2
    )
    metrics = _metrics(sess.trace)
    report = {
        "seat": str(seat),
        "start": start,
        "end": _snap(sess.state, env),
        "metrics": metrics,
        "track_top": bool(track.top_reached),
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument(
        "--from-seat",
        type=Path,
        default=None,
        help="Skip lower; probe fire recipe from this seat state only",
    )
    ap.add_argument(
        "--seat-out",
        type=Path,
        default=SCRATCH / "post_bubble_fire_seat_pure.state",
    )
    ap.add_argument(
        "--postrun-out",
        type=Path,
        default=SCRATCH / "post_bubble_fire_postrun_pure.state",
    )
    ap.add_argument(
        "--wall-out",
        type=Path,
        default=SCRATCH / "post_bubble_wall_approach_pure.state",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=DEBUG / "bubble_r18_velocity_dump.json",
    )
    ap.add_argument("--no-fire", action="store_true")
    args = ap.parse_args()

    if args.from_seat is not None:
        report = probe_seat(args.from_seat, args.json_out)
    else:
        report = dump_from_source(
            args.source,
            seat_out=args.seat_out,
            postrun_out=args.postrun_out,
            wall_out=args.wall_out,
            json_out=args.json_out,
            run_fire=not args.no_fire,
        )
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
