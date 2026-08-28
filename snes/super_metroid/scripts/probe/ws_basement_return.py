#!/usr/bin/env python3
"""Powered Wrecked Ship Basement → Main Shaft (rr-kw8t hop 1).

Morph-roll LEFT from the Phantoon leave, jump UP the blue ceiling hatch
into 0xCAF6. Map station LEFT is dead — skip.
https://wiki.supermetroid.run/Basement

```bash
# Watch: repo-wide --headed (same flag on kpdr.py pure / ./play).
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-basement-to-main \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_phantoon_leave.state \
  --headed

QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_basement_return.py --dual --no-video
```

Default source: ``scratch/post_phantoon_leave.state``. Boot settle 5.
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
from super_metroid.hop_glance import LeaveMiss, final_from_state, grade_final
from super_metroid.leave_specs import WS_BASEMENT_TO_MAIN
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.ram import parse_env_state
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.controller_common import MORPH_POSES, is_morph
from super_metroid.combat.enemies import list_enemies
from super_metroid.combat.enemies.workrobot import stall_reason
from super_metroid.routes.kpdr.k6.ws_basement_return import (
    play_ws_basement_to_main,
    ws_basement_main_settled,
)
from super_metroid.routes.skills.charge_shot import session_beam_charge

SCRATCH = SCRATCH_STATE_DIR
DEFAULT_SOURCE = SCRATCH / "post_phantoon_leave.state"
DEFAULT_OUT = SCRATCH / "post_ws_basement_to_main.state"
DEFAULT_REPORT = SCRATCH / "ws_basement_to_main.json"
DEFAULT_DUAL = SCRATCH / "ws_basement_to_main_dual.json"
BOOT_SETTLE = 5
HOP = "ws_basement_to_main"


class _Sess:
    def __init__(self, env: Any, assist: UnlimitedResourcesAssist | None):
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")

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
) -> dict[str, Any]:
    env = make_dev_env()
    a = UnlimitedResourcesAssist() if assist else None
    pygame_mod = None
    try:
        if a is not None:
            a.attach_env(env)
        if headed:
            pygame_mod = attach_headed(
                env, title="SM BOT: post-Phantoon basement → Main", hud=_headed_hud
            )
        boot_from_state(env, source, settle_frames=settle)
        sess = _Sess(env, a)
        boot = _snap(sess.state, {"frame": 0})
        error = None
        hop_misses: list[str] | None = None
        try:
            st = play_ws_basement_to_main(sess)
        except LeaveMiss as exc:
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
            hop_misses = list(exc.misses)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
        enemies = list_enemies(sess)
        extra = {
            "frame": sess.frame,
            "charge": session_beam_charge(sess),
            "mov": int(st.movement_type),
            "killed": int(getattr(st, "enemies_killed", 0) or 0),
            "stall": stall_reason(
                int(st.samus_x),
                int(st.samus_y),
                int(st.movement_type),
                int(st.pose),
                enemies,
            ),
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
        ok = error is None and ws_basement_main_settled(st)
        if ok and save is not None:
            save_dev_state(env, save)
        leftover_state = None
        if not ok:
            leftover_state = SCRATCH / "ws_basement_to_main_leftover.state"
            try:
                save_dev_state(env, leftover_state)
            except Exception:  # noqa: BLE001
                leftover_state = None
            try:
                from PIL import Image

                png = SCRATCH / "ws_basement_to_main_red.png"
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
        }
        if not ok:
            # Leftover IS the final still. Next agent reads leftover, not the pin.
            report["leftover"] = dict(final)
            report["glance_misses"] = hop_misses or grade_final(final, WS_BASEMENT_TO_MAIN)
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
                "glance_misses", grade_final(primary["final"], WS_BASEMENT_TO_MAIN)
            )
        dual_path = Path(args.dual_report or DEFAULT_DUAL)
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual_report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dual_path}")
    out = Path(args.report or DEFAULT_REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    flag = "GREEN" if primary["success"] and dual_exact else "RED"
    print(
        f"{flag} dual={dual_exact if dual else 'n/a'} "
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
