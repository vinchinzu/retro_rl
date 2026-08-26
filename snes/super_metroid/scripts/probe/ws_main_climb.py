#!/usr/bin/env python3
"""Powered Wrecked Ship Main Shaft → Attic (rr-kw8t hop 2).

Six in-room seams (not hop GREEN): pit_shot → grate_seat → west_super →
mid_climb → attic_seat → attic_door. Full hop GREEN is Attic gs=8 only.
Save room RIGHT is dead — skip. Do not go DOWN through the floor hatch.
https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft

```bash
# Watch: repo-wide --headed (same flag on kpdr.py pure / ./play).
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-main-to-attic \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ws_basement_to_main.state \
  --headed

# One seam (diagnostic PhaseStop, not hop GREEN):
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video
```

Default source: ``scratch/post_ws_basement_to_main.state``. Boot settle 5.
``--headed`` is ``retro_harness.headed``. Do not revert the controller on RED.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from retro_harness.headed import add_headed_flag, attach_headed, idle_headed
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env, save_dev_state
from super_metroid.hop_glance import (
    LeaveMiss,
    WS_MAIN_TO_ATTIC,
    final_from_state,
    grade_final,
)
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.ram import parse_env_state
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.controller_common import MORPH_POSES, is_morph
from super_metroid.routes.kpdr.k6.ws_main_climb import (
    play_ws_main_to_attic,
    play_ws_main_to_attic_phased,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.k6.ws_main_phases import WS_MAIN_PHASES
from super_metroid.routes.skills.geometry import PhaseStop
from super_metroid.routes.kpdr.k6.ws_main_ice import list_shaft_enemies
from super_metroid.routes.skills.charge_shot import session_beam_charge

SCRATCH = SCRATCH_STATE_DIR
DEFAULT_SOURCE = SCRATCH / "post_ws_basement_to_main.state"
DEFAULT_OUT = SCRATCH / "post_ws_main_to_attic.state"
DEFAULT_REPORT = SCRATCH / "ws_main_to_attic.json"
DEFAULT_DUAL = SCRATCH / "ws_main_to_attic_dual.json"
BOOT_SETTLE = 5
HOP = "ws_main_to_attic"


class _Sess:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist | None):
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.min_y = int(self.state.samus_y)
        self.min_y_xy = [int(self.state.samus_x), int(self.state.samus_y)]
        self.min_y_pose = int(self.state.pose)
        self.min_y_frame = 0

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        st = parse_env_state(self.env, frame=self.frame, mode="nav")
        if self.assist is not None:
            try:
                self.assist.apply(self.env.data, st)
            except Exception:  # noqa: BLE001
                try:
                    self.assist.apply(self.env, st)
                except Exception:  # noqa: BLE001
                    pass
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        y = int(self.state.samus_y)
        if y < self.min_y:
            self.min_y = y
            self.min_y_xy = [int(self.state.samus_x), y]
            self.min_y_pose = int(self.state.pose)
            self.min_y_frame = self.frame
        return self.state


def _snap(st: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    pose = int(st.pose)
    out: dict[str, Any] = final_from_state(st)
    out.update(
        {
            "morph": pose in MORPH_POSES or is_morph(pose),
            "selected": int(st.selected_item),
            "items": f"0x{int(st.collected_items):04X}",
            "beams": f"0x{int(st.collected_beams):04X}",
            "vx": int(st.velocity_x),
            "vy": int(st.velocity_y),
            "facing": int(st.facing),
            "supers": int(getattr(st, "super_missiles", 0) or 0),
            "missiles": int(getattr(st, "missiles", 0) or 0),
        }
    )
    if extra:
        out.update(extra)
    return out


def _headed_hud(env: Any) -> str:
    st = parse_env_state(env, mode="nav")
    max_hp = int(getattr(st, "max_health", 0) or 0)
    return (
        f"BOT  0x{int(st.room_id):04X} ({int(st.samus_x)},{int(st.samus_y)}) "
        f"p{int(st.pose)} gs={int(st.game_state)} hp={int(st.health)}/{max_hp}"
    )


def _run_hop(
    source: Path,
    *,
    assist: bool = True,
    settle: int = BOOT_SETTLE,
    save: Path | None = None,
    headed: bool = False,
    start_phase: str = "pit_shot",
    stop_at: str = "attic_door",
) -> dict[str, Any]:
    env = make_dev_env()
    a = UnlimitedResourcesAssist() if assist else None
    pygame_mod = None
    try:
        if a is not None:
            a.attach_env(env)
        if headed:
            pygame_mod = attach_headed(
                env, title="SM BOT: powered Main Shaft → Attic", hud=_headed_hud
            )
        boot_from_state(env, source, settle_frames=settle)
        sess = _Sess(env, a)
        boot = _snap(sess.state, {"frame": 0})
        error = None
        hop_misses: list[str] | None = None
        phase_stop = None
        sliced = start_phase != "pit_shot" or stop_at != "attic_door"
        try:
            if sliced:
                st = play_ws_main_to_attic_phased(
                    sess, start=start_phase, stop=stop_at
                )
            else:
                st = play_ws_main_to_attic(sess)
        except PhaseStop as exc:
            phase_stop = exc.phase
            st = exc.state
            error = None
        except LeaveMiss as exc:
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
            hop_misses = list(exc.misses)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
        enemies = list_shaft_enemies(sess)
        extra = {
            "frame": sess.frame,
            "charge": session_beam_charge(sess),
            "mov": int(getattr(st, "movement_type", 0) or 0),
            "killed": int(getattr(st, "enemies_killed", 0) or 0),
            "min_y": sess.min_y,
            "min_y_xy": sess.min_y_xy,
            "min_y_pose": sess.min_y_pose,
            "min_y_frame": sess.min_y_frame,
            "enemies": [
                {
                    "slot": e.slot,
                    "id": f"0x{e.enemy_id:04X}",
                    "xy": [e.x, e.y],
                    "hp": e.hp,
                    "freeze": e.freeze_timer,
                }
                for e in enemies
            ],
        }
        hop_green = error is None and ws_main_attic_settled(st)
        phase_ok = phase_stop is not None and error is None
        ok = hop_green or phase_ok
        if hop_green and save is not None:
            save_dev_state(env, save)
        leftover_state = None
        if not hop_green:
            leftover_state = SCRATCH / "ws_main_to_attic_leftover.state"
            try:
                save_dev_state(env, leftover_state)
            except Exception:  # noqa: BLE001
                leftover_state = None
            try:
                from PIL import Image

                png = SCRATCH / "ws_main_to_attic_red.png"
                Image.fromarray(env.render()).save(png)
            except Exception:  # noqa: BLE001
                png = None
        else:
            png = None
        timed = format_segment_time(sess.frame)
        final = _snap(st, extra)
        report: dict[str, Any] = {
            "success": ok,
            "error": error,
            "source": str(source),
            "boot": boot,
            "final": final,
            "frames": sess.frame,
            "time": timed,
            "saved": str(save) if ok and save is not None else None,
            "leftover_state": str(leftover_state) if leftover_state is not None else None,
            "red_png": str(png) if png is not None else None,
            "hop_green": hop_green,
            "phase_stop": phase_stop,
        }
        if not hop_green:
            report["leftover"] = dict(final)
            report["glance_misses"] = hop_misses or grade_final(final, WS_MAIN_TO_ATTIC)
        return report
    finally:
        if headed and pygame_mod is not None:
            idle_headed(env, pygame_mod)
        env.close()


def cmd_pure(args: argparse.Namespace) -> int:
    source = Path(args.source or DEFAULT_SOURCE)
    save = Path(args.out or DEFAULT_OUT)
    headed = bool(getattr(args, "headed", False))
    dual = bool(args.dual) and not headed
    runs = [
        _run_hop(
            source,
            assist=not args.no_assist,
            settle=args.settle,
            save=save if i == 0 else None,
            headed=headed and i == 0,
            start_phase=str(getattr(args, "start_phase", None) or "pit_shot"),
            stop_at=str(getattr(args, "stop_at", None) or "attic_door"),
        )
        for i in range(2 if dual else 1)
    ]
    for row in runs:
        row["command"] = "pure"
        row["hop"] = HOP
    primary = runs[0]
    timed = primary["time"]
    dual_exact = True
    if dual:
        dual_exact = (
            runs[0]["success"]
            and runs[1]["success"]
            and runs[0]["frames"] == runs[1]["frames"]
            and runs[0]["final"]["xy"] == runs[1]["final"]["xy"]
            and runs[0]["final"]["pose"] == runs[1]["final"]["pose"]
            and runs[0]["final"]["gs"] == runs[1]["final"]["gs"]
        )
        dual_report = {
            "success": all(r["success"] for r in runs) and dual_exact,
            "dual_exact": dual_exact,
            "hop": HOP,
            "runs": runs,
            "frames": primary["frames"],
            "time": timed,
        }
        if not dual_report["success"]:
            leftover = primary.get("leftover", primary.get("final"))
            dual_report["leftover"] = leftover
            dual_report["glance_misses"] = primary.get(
                "glance_misses", grade_final(primary["final"], WS_MAIN_TO_ATTIC)
            )
        dual_path = Path(args.dual_report or DEFAULT_DUAL)
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual_report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dual_path}")
    out = Path(args.report or DEFAULT_REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    hop_green = bool(primary.get("hop_green"))
    flag = "GREEN" if hop_green and dual_exact else (
        "PHASE" if primary.get("phase_stop") else "RED"
    )
    print(
        f"{flag} dual={dual_exact if dual else 'n/a'} "
        f"phase_stop={primary.get('phase_stop')} "
        f"frames={timed['frames']} seconds={timed['seconds']} "
        f"clock={timed['clock']} final={primary['final']} "
        f"saved={primary.get('saved')} err={primary['error']}"
    )
    print(f"wrote {out}")
    return 0 if primary["success"] and dual_exact else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--dual-report", type=Path, default=None)
    ap.add_argument("--settle", type=int, default=BOOT_SETTLE)
    ap.add_argument("--no-assist", action="store_true")
    ap.add_argument("--dual", action="store_true")
    ap.add_argument(
        "--start-phase",
        choices=WS_MAIN_PHASES,
        default="pit_shot",
        help="Skip earlier seams (phase dump in). Not hop GREEN.",
    )
    ap.add_argument(
        "--stop-at",
        choices=WS_MAIN_PHASES,
        default="attic_door",
        help="PhaseStop at this seam. Diagnostic only — not hop GREEN.",
    )
    add_headed_flag(ap)
    ap.add_argument(
        "--no-video",
        action="store_true",
        help="Accepted no-op (probe has no video).",
    )
    args = ap.parse_args(argv)
    args.cmd = "pure"
    return cmd_pure(args)


if __name__ == "__main__":
    raise SystemExit(main())
