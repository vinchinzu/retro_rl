#!/usr/bin/env python3
"""rr-av5s upper residual probe: mid+human850 then open-loop / WJ recipes.

Not product evidence. One-shot diagnostics for Hellway dual green.

```bash
uv run python snes/super_metroid/scripts/probe/red_upper_probe.py --mode full-human
uv run python snes/super_metroid/scripts/probe/red_upper_probe.py --mode sweep-human
uv run python snes/super_metroid/scripts/probe/red_upper_probe.py --mode wj-latch
uv run python snes/super_metroid/scripts/probe/red_upper_probe.py --mode full-from-floor
```
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import boot_from_state, make_dev_env  # noqa: E402
from super_metroid.paths import GAME_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import (  # noqa: E402
    WallJumpTiming,
    hold,
    is_wall_latch,
    select_weapon,
    settle_hold,
    unmorph,
    walljump_once,
)
from super_metroid.routes.kpdr.k5.geometry import (  # noqa: E402
    RED_FLOOR_Y,
    RED_TOP_DOOR_Y,
)
from super_metroid.routes.kpdr.k5.red_to_hellway import (  # noqa: E402
    _HUMAN_ASCENT_FULL,
    _HUMAN_FLOOR_FRAMES,
    _HUMAN_FLOOR_RLE,
    _climb_mid,
    _play_upper_rle,
)
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER  # noqa: E402
from super_metroid.routes.rle import play_script  # noqa: E402

SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_ice_bat_to_red_pure.state"


class _ProbeSession:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.best_y = 99999
        self.best_xy = (0, 0, 0)

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        y = int(self.state.samus_y)
        if y < self.best_y and int(self.state.room_id) == ROOM_RED_TOWER:
            self.best_y = y
            self.best_xy = (
                int(self.state.samus_x),
                y,
                int(self.state.pose),
            )
        return self.state


def _pin(st) -> dict:
    return {
        "room": f"0x{int(st.room_id):04X}",
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "pose": int(st.pose),
        "vy": int(st.velocity_y),
        "hellway": int(st.room_id) == ROOM_HELLWAY,
    }


def _rle_from(start: int, n: int | None = None):
    full = _HUMAN_ASCENT_FULL
    used = 0
    out: list[tuple[int, tuple[str, ...]]] = []
    for count, btns in full:
        count = int(count)
        if used + count <= start:
            used += count
            continue
        skip = max(0, start - used)
        rem = count - skip
        used += count
        if n is None:
            take = rem
        else:
            already = sum(c for c, _ in out)
            take = min(rem, n - already)
        if take > 0:
            out.append((take, tuple(btns)))
        if n is not None and sum(c for c, _ in out) >= n:
            break
    return tuple(out)


def _entry_and_mid(session: _ProbeSession, label: str = "p") -> dict:
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 6, reason=f"{label}_entry")
    for _ in range(100):
        st = session.state
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        if int(st.samus_x) <= 165 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_clear_bat")
    settle_hold(session, 6, reason=f"{label}_settle")
    _climb_mid(session, f"{label}_mid")
    return _pin(session.state)


def _to_handoff(session: _ProbeSession, label: str = "p") -> dict:
    mid = _entry_and_mid(session, label)
    if int(session.state.samus_y) >= RED_FLOOR_Y - 120:
        _play_upper_rle(session, _HUMAN_FLOOR_RLE, f"{label}_h850")
    return {"mid": mid, "after850": _pin(session.state)}


def _play_rest(session: _ProbeSession, runs, reason: str) -> None:
    if not runs:
        return
    play_script(
        session,
        runs,
        reason=reason,
        room_id=ROOM_RED_TOWER,
        stop_when=lambda st: int(st.room_id) == ROOM_HELLWAY
        or int(st.room_id) != ROOM_RED_TOWER,
        on_lag="break",
    )


def mode_human_ext(session: _ProbeSession, extra: int) -> dict:
    pre = _to_handoff(session)
    runs = _rle_from(_HUMAN_FLOOR_FRAMES, extra)
    _play_rest(session, runs, f"human_ext_{extra}")
    return {
        "mode": f"human_ext_{extra}",
        "pre": pre,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": session.best_xy,
        "frames": session.frame,
    }


def mode_full_human(session: _ProbeSession) -> dict:
    pre = _to_handoff(session)
    rest = _rle_from(_HUMAN_FLOOR_FRAMES, None)
    _play_rest(session, rest, "human_rest")
    return {
        "mode": "full_human_rest",
        "pre": pre,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": session.best_xy,
        "frames": session.frame,
        "rest_frames": sum(n for n, _ in rest),
    }


def mode_full_from_floor(session: _ProbeSession) -> dict:
    mid = _entry_and_mid(session)
    play_script(
        session,
        _HUMAN_ASCENT_FULL,
        reason="human_full",
        room_id=ROOM_RED_TOWER,
        stop_when=lambda st: int(st.room_id) == ROOM_HELLWAY
        or int(st.room_id) != ROOM_RED_TOWER,
        on_lag="break",
    )
    return {
        "mode": "full_human_from_floor",
        "mid": mid,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": session.best_xy,
        "frames": session.frame,
    }


def mode_wj_latch(session: _ProbeSession) -> dict:
    pre = _to_handoff(session)
    milestones = [{"t": "start", **_pin(session.state)}]
    last_log_y = int(session.state.samus_y)
    for cycle in range(60):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        x = int(st.samus_x)
        pose = int(st.pose)
        if y < last_log_y - 40:
            milestones.append({"cycle": cycle, **_pin(st)})
            last_log_y = y
        if y <= RED_TOP_DOOR_Y + 40:
            for _ in range(100):
                hold(session, 1, "RIGHT", "B", "A", reason="exit")
                if int(session.state.room_id) == ROOM_HELLWAY:
                    break
            break
        if pose in (29, 30, 31, 32):
            hold(session, 1, "UP", reason="unmorph_true")
            continue
        if is_wall_latch(st):
            into = "LEFT" if x >= 160 else "RIGHT"
            flip = "RIGHT" if into == "LEFT" else "LEFT"
            walljump_once(
                session,
                WallJumpTiming(
                    into=into, flip=flip, into_frames=4, amid_frames=2, flip_frames=16
                ),
                reason=f"wj_latched_{cycle}",
            )
            continue
        if x >= 200:
            hold(session, 1, "LEFT", reason="face_l")
            for _ in range(22):
                st2 = hold(session, 1, "LEFT", "A", reason="wj_r")
                if is_wall_latch(st2) or int(st2.samus_y) < y - 25:
                    break
            continue
        if x <= 55:
            hold(session, 1, "RIGHT", reason="face_r")
            for _ in range(22):
                st2 = hold(session, 1, "RIGHT", "A", reason="wj_l")
                if is_wall_latch(st2) or int(st2.samus_y) < y - 25:
                    break
            continue
        direction = "RIGHT" if x < 130 else "LEFT"
        hold(session, 2, direction, "B", reason="spin_run")
        for _ in range(30):
            st2 = hold(session, 1, direction, "B", "A", reason="spin")
            if is_wall_latch(st2):
                break
            if int(st2.samus_x) >= 210 or int(st2.samus_x) <= 50:
                break
    return {
        "mode": "wj_latch",
        "pre": pre,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": session.best_xy,
        "milestones": milestones[-12:],
        "frames": session.frame,
    }


def _boot(source: Path) -> tuple:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, source)
    for _ in range(5):
        env.step(idle_action())
        assist.apply(env.data, parse_env_state(env, mode="nav"))
    session = _ProbeSession(env, assist)
    return env, session


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=(
            "human-ext",
            "full-human",
            "full-from-floor",
            "wj-latch",
            "sweep-human",
        ),
        default="full-human",
    )
    ap.add_argument("--extra", type=int, default=1723)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "sweep-human":
        results = []
        for extra in (200, 400, 600, 900, 1200, 1723):
            env, session = _boot(args.source)
            try:
                r = mode_human_ext(session, extra)
                results.append(r)
                print(
                    f"extra={extra:4d} end=({r['end']['x']},{r['end']['y']})"
                    f"p{r['end']['pose']} room={r['end']['room']} "
                    f"best_y={r['best_y']} hell={r['end']['hellway']} "
                    f"after850=({r['pre']['after850']['x']},"
                    f"{r['pre']['after850']['y']})",
                    flush=True,
                )
            except Exception as exc:
                results.append({"mode": f"human_ext_{extra}", "error": str(exc)})
                print(f"extra={extra} ERROR {exc}", flush=True)
                traceback.print_exc()
            finally:
                env.close()
        print(json.dumps(results, indent=2))
        return

    reports = []
    for t in range(args.trials):
        env, session = _boot(args.source)
        try:
            if args.mode == "human-ext":
                r = mode_human_ext(session, args.extra)
            elif args.mode == "full-human":
                r = mode_full_human(session)
            elif args.mode == "full-from-floor":
                r = mode_full_from_floor(session)
            else:
                r = mode_wj_latch(session)
            r["trial"] = t
            reports.append(r)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as exc:
            reports.append({"trial": t, "error": str(exc)})
            print(f"ERROR trial {t}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            env.close()


if __name__ == "__main__":
    main()
