#!/usr/bin/env python3
"""Powered Wrecked Ship Main Shaft → Attic (rr-kw8t hop 2).

``--start-phase`` / ``--stop-at`` over ``WS_MAIN_PHASES``. Full hop GREEN
is Attic gs=8 only. PhaseStop is graded against the phase seat, not dest
Attic. Pin, checkbox, and default argv:
``snes/super_metroid/docs/tasks/rr-kw8t-residual.md``.

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-main-to-attic \
  --source <pin> --headed

QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at <phase> --no-video --dual
```

``--headed`` is ``retro_harness.headed``. Do not revert the controller on RED.
https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

from retro_harness.controls import pressed_snes_buttons
from retro_harness.headed import add_headed_flag, attach_headed, idle_headed
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.combat.enemies import ATOMIC_ID, list_enemies
from super_metroid.dev.common import boot_from_state, make_dev_env, save_dev_state
from super_metroid.hop_glance import LeaveMiss, final_from_state, grade_final
from super_metroid.leave_specs import WS_MAIN_PHASE_SPECS, WS_MAIN_TO_ATTIC
from super_metroid.paths import SCRATCH_STATE_DIR
from super_metroid.plm import (
    session_ram,
    shot_block_spawns,
    snapshot_plms,
    snapshot_projectiles,
)
from super_metroid.ram import parse_env_state
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.controller_common import MORPH_POSES, is_morph
from super_metroid.routes.kpdr.wrecked_ship.ws_main_climb import (
    play_ws_main_to_attic,
    ws_main_attic_settled,
)
from super_metroid.routes.skills.geometry import PhaseStop
from super_metroid.routes.skills.charge_shot import session_beam_charge

SCRATCH = SCRATCH_STATE_DIR
DEFAULT_SOURCE = SCRATCH / "post_ws_basement_to_main.state"
DEFAULT_OUT = SCRATCH / "post_ws_main_to_attic.state"
DEFAULT_REPORT = SCRATCH / "ws_main_to_attic.json"
DEFAULT_DUAL = SCRATCH / "ws_main_to_attic_dual.json"
GRATE_SEAT_PIN = SCRATCH / "post_ws_main_grate_seat.state"
BOOT_SETTLE = 5
HOP = "ws_main_to_attic"
TRACE_FRAMES = 240
WALL_STUCK_FRAMES = 180


def phase_glance(
    phase_stop: str | None, final: dict[str, Any], error: str | None
) -> tuple[bool, list[str]]:
    """Grade a PhaseStop still against the in-room seat, not dest Attic."""
    if not phase_stop:
        return False, []
    spec = WS_MAIN_PHASE_SPECS[phase_stop]
    misses = list(grade_final(final, spec))
    return error is None and not misses, misses


def _held_pin(phase_stop: str) -> Path:
    if phase_stop == "grate_seat":
        return GRATE_SEAT_PIN
    return SCRATCH / f"post_ws_main_{phase_stop}.state"


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
        self.trace: deque[dict[str, Any]] = deque(maxlen=TRACE_FRAMES)
        self.events: list[dict[str, Any]] = []
        self._prev_plms: tuple[dict[str, int], ...] = ()
        self._prev_buttons: set[str] = set()
        self._wall_progress_frame = 0
        self._saw_wall = False
        self._saw_523 = False
        self._saw_443 = False
        self._upper_n = 0

    def step(self, action, reason: str = ""):
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
        self._note_frame(action, reason)
        return self.state

    def _note_frame(self, action, reason: str) -> None:
        st = self.state
        x, y = int(st.samus_x), int(st.samus_y)
        buttons = set(pressed_snes_buttons(action))
        ram = session_ram(self)
        projs: tuple[dict[str, int], ...] = ()
        spawned: tuple[dict[str, int], ...] = ()
        if ram is not None:
            cur = snapshot_plms(ram)
            spawned = shot_block_spawns(self._prev_plms, cur)
            self._prev_plms = cur
            projs = snapshot_projectiles(ram)
        planted_wall = (
            576 <= y <= 590
            and x >= 1228
            and abs(int(st.velocity_y)) <= 1
            and int(st.pose) in (1, 2, 3, 4, 9, 10, 137, 138)
        )
        planted_523 = 508 <= y <= 540 and abs(int(st.velocity_y)) <= 1
        planted_443 = (
            430 <= y <= 455
            and abs(int(st.velocity_y)) <= 1
            and int(st.pose) in (1, 2, 9, 10)
        )
        if planted_wall and not self._saw_wall:
            self._saw_wall = True
            self._wall_progress_frame = self.frame
            self.events.append({"kind": "wall_587", "frame": self.frame, "xy": [x, y]})
        if planted_523 and not self._saw_523:
            self._saw_523 = True
            self.events.append({"kind": "land_523", "frame": self.frame, "xy": [x, y]})
        if planted_443 and not self._saw_443:
            self._saw_443 = True
            self.events.append({"kind": "land_443", "frame": self.frame, "xy": [x, y]})
        x_edge = ("X" in buttons) != ("X" in self._prev_buttons)
        if spawned:
            self._upper_n += len(spawned)
            if planted_wall:
                self._wall_progress_frame = self.frame
            self.events.append(
                {
                    "kind": "plm",
                    "frame": self.frame,
                    "xy": [x, y],
                    "ids": [f"0x{int(r['id']):04X}" for r in spawned],
                    "pxpy": [[int(r["px"]), int(r["py"])] for r in spawned],
                    "n": self._upper_n,
                }
            )
        if projs and planted_wall:
            self._wall_progress_frame = self.frame
        if y < 580:
            self._wall_progress_frame = self.frame
        if planted_wall and self._wall_progress_frame and self._upper_n == 0:
            if self.frame - self._wall_progress_frame >= WALL_STUCK_FRAMES:
                self.events.append(
                    {
                        "kind": "wall_stuck",
                        "frame": self.frame,
                        "xy": [x, y],
                        "charge": session_beam_charge(self),
                        "buttons": sorted(buttons),
                    }
                )
                raise TimeoutError(
                    f"planted-wall deadlock at ({x},{y}) p{int(st.pose)} "
                    f"charge={session_beam_charge(self)} reason={reason}"
                )
        row = {
            "f": self.frame,
            "xy": [x, y],
            "p": int(st.pose),
            "bt": sorted(buttons),
            "why": reason,
            "ch": session_beam_charge(self),
            "x_edge": x_edge,
            "proj": len(projs),
        }
        if planted_wall or (planted_523 and x <= 1088):
            atoms = [
                {
                    "slot": e.slot,
                    "xy": [e.x, e.y],
                    "hp": e.hp,
                    "freeze": e.freeze_timer,
                }
                for e in list_enemies(self)
                if e.enemy_id == ATOMIC_ID and abs(e.y - y) <= 80
            ]
            if atoms:
                row["atomics"] = atoms
        self.trace.append(row)
        self._prev_buttons = buttons


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
        try:
            st = play_ws_main_to_attic(sess, start=start_phase, stop=stop_at)
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
        enemies = list_enemies(sess)
        extra = {
            "frame": sess.frame,
            "charge": session_beam_charge(sess),
            "mov": int(st.movement_type),
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
        extra["events"] = list(sess.events)
        extra["trace"] = list(sess.trace)
        extra["upper_n"] = sess._upper_n
        timed = format_segment_time(sess.frame)
        final = _snap(st, extra)
        hop_green = error is None and ws_main_attic_settled(st)
        if phase_stop is not None:
            phase_ok, glance_misses = phase_glance(phase_stop, final, error)
        elif stop_at != "attic_door":
            _, glance_misses = phase_glance(stop_at, final, error)
            phase_ok = False
        elif hop_misses:
            glance_misses = hop_misses
            phase_ok = False
        elif not hop_green:
            glance_misses = grade_final(final, WS_MAIN_TO_ATTIC)
            phase_ok = False
        else:
            glance_misses = []
            phase_ok = False
        ok = hop_green or phase_ok
        saved: str | None = None
        if hop_green and save is not None and not phase_stop:
            save_dev_state(env, save)
            saved = str(save)
        # Usable outgoing pin only. Observable land that PhaseStops but
        # fails phase_glance must not clobber post_ws_main_grate_seat.state.
        if phase_ok and phase_stop:
            pin = _held_pin(phase_stop)
            try:
                save_dev_state(env, pin)
                saved = str(pin)
            except Exception:  # noqa: BLE001
                pass
        leftover_state = None
        png = None
        if not ok:
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
        report: dict[str, Any] = {
            "success": ok,
            "error": error,
            "source": str(source),
            "boot": boot,
            "final": final,
            "frames": sess.frame,
            "time": timed,
            "saved": saved,
            "leftover_state": str(leftover_state) if leftover_state is not None else None,
            "red_png": str(png) if png is not None else None,
            "hop_green": hop_green,
            "phase_ok": phase_ok,
            "phase_stop": phase_stop,
            "glance_misses": glance_misses,
        }
        if not ok:
            report["leftover"] = dict(final)
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
            dual_report["glance_misses"] = primary.get("glance_misses") or []
        dual_path = Path(args.dual_report or DEFAULT_DUAL)
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual_report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dual_path}")
    out = Path(args.report or DEFAULT_REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    hop_green = bool(primary.get("hop_green"))
    phase_ok = bool(primary.get("phase_ok"))
    if hop_green and dual_exact:
        flag = "GREEN"
    elif phase_ok:
        flag = "PHASE"
    else:
        flag = "RED"
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
    from super_metroid.routes.kpdr.wrecked_ship.ws_main_geometry import WS_MAIN_PHASES

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
