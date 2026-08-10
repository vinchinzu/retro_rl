#!/usr/bin/env python3
"""rr-av5s top gap probe: dual period-WJ path → y420→y180 / Hellway recipes.

Saves intermediate pins for fast re-entry, then sweeps top-out recipes.

```bash
uv run python snes/super_metroid/scripts/probe/red_top_gap_probe.py --mode capture
uv run python snes/super_metroid/scripts/probe/red_top_gap_probe.py --mode from-peak --recipe seat-wj
uv run python snes/super_metroid/scripts/probe/red_top_gap_probe.py --mode sweep --source-pin peak
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
from retro_harness.env import write_state_bytes  # noqa: E402
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
    RED_BOTTOM_Y,
    RED_FLOOR_Y,
    RED_TOP_DOOR_X,
    RED_TOP_DOOR_Y,
)
from super_metroid.routes.kpdr.k5.red_to_hellway import (  # noqa: E402
    _HUMAN_FLOOR_RLE,
    _UPPER_WJ_FLIP,
    _UPPER_WJ_INTO,
    _UPPER_WJ_PERIOD,
    _UPPER_WJ_PHASES,
    _climb_mid,
    _period_wj,
    _play_upper_rle,
    _seat_left_after_handoff,
)
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER  # noqa: E402

SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_ice_bat_to_red_pure.state"
PEAK_STATE = SCRATCH / "red_top_peak_probe.state"
END_STATE = SCRATCH / "red_top_end_probe.state"
SEAT_STATE = SCRATCH / "red_top_seat_probe.state"


class _ProbeSession:
    def __init__(self, env, assist: UnlimitedResourcesAssist) -> None:
        self.env = env
        self.assist = assist
        self.frame = 0
        self.state = parse_env_state(env, mode="nav")
        self.best_y = 99999
        self.best_xy = (0, 0, 0)
        self.log: list[dict] = []

    def step(self, action, reason: str = ""):
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        y = int(self.state.samus_y)
        if y < self.best_y and int(self.state.room_id) == ROOM_RED_TOWER:
            self.best_y = y
            self.best_xy = (int(self.state.samus_x), y, int(self.state.pose))
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


def _boot(source: Path) -> tuple:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    boot_from_state(env, source)
    for _ in range(5):
        env.step(idle_action())
        assist.apply(env.data, parse_env_state(env, mode="nav"))
    return env, _ProbeSession(env, assist)


def _save(env, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_state_bytes(path, env.em.get_state())


def _to_mid_handoff(session: _ProbeSession) -> dict:
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 6, reason="entry")
    for _ in range(100):
        st = session.state
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        if int(st.samus_x) <= 165 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", "B", reason="clear_bat")
    settle_hold(session, 6, reason="settle")
    mid = _climb_mid(session, "mid")
    mid_pin = _pin(session.state)
    if int(session.state.samus_y) >= RED_FLOOR_Y - 120:
        _play_upper_rle(session, _HUMAN_FLOOR_RLE, "h850")
    after = _pin(session.state)
    return {"mid": mid_pin, "after850": after, "mid_ret": str(type(mid))}


def _period_wj_instrumented(
    session: _ProbeSession,
    *,
    save_peak_env=None,
    peak_threshold: int = 450,
) -> dict:
    """Seat + alternating period WJ; save first time y<=peak_threshold."""
    phases_out: list[dict] = []
    y_h = int(session.state.samus_y)
    if 1300 <= y_h <= 1550:
        _seat_left_after_handoff(session, "seat")
    seat_pin = _pin(session.state)
    if save_peak_env is not None:
        # optional early seat save
        pass

    hold(session, 3, "LEFT", "B", reason="wj_run")
    for _ in range(12):
        st = hold(session, 1, "LEFT", "B", "A", reason="wj_j")
        if int(st.room_id) != ROOM_RED_TOWER:
            break

    peak_saved = False
    best_before = session.best_y
    for i, (side, frames, stop_y) in enumerate(_UPPER_WJ_PHASES):
        y0 = int(session.state.samus_y)
        best0 = session.best_y
        _period_wj(
            session,
            f"pwj{i}",
            side=side,
            frames=frames,
            stop_y=stop_y,
        )
        st = session.state
        pin = _pin(st)
        phases_out.append(
            {
                "i": i,
                "side": side,
                "stop_y": stop_y,
                "y0": y0,
                "end": pin,
                "best_y": session.best_y,
                "best_xy": list(session.best_xy),
                "delta_best": best0 - session.best_y if session.best_y < best0 else 0,
            }
        )
        print(
            f"  phase{i} {side:5s} stop={stop_y} "
            f"end=({pin['x']},{pin['y']})p{pin['pose']} "
            f"best={session.best_y} {session.best_xy}",
            flush=True,
        )
        if (
            not peak_saved
            and save_peak_env is not None
            and session.best_y <= peak_threshold
        ):
            _save(save_peak_env, PEAK_STATE)
            peak_saved = True
            print(f"  saved peak state best_y={session.best_y} → {PEAK_STATE}", flush=True)
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        if int(st.samus_y) <= RED_TOP_DOOR_Y + 40:
            break
        if int(st.room_id) == ROOM_HELLWAY:
            break

    return {
        "seat": seat_pin,
        "phases": phases_out,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": list(session.best_xy),
        "best_before_wj": best_before,
        "peak_saved": peak_saved,
        "frames": session.frame,
    }


# --- top-out recipes (from peak / end / seat pins) ---


def recipe_continue_wj(
    session: _ProbeSession,
    *,
    frames: int = 2400,
    period: int = 16,
    into: int = 6,
    flip: int = 8,
    start_side: str = "LEFT",
    switch_every: int = 400,
) -> dict:
    side = start_side
    last_switch = 0
    for i in range(frames):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        if y <= RED_TOP_DOOR_Y + 25:
            break
        if y >= RED_BOTTOM_Y - 80:
            break
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason="u")
            continue
        if switch_every and i - last_switch >= switch_every:
            side = "RIGHT" if side == "LEFT" else "LEFT"
            last_switch = i
        opp = "RIGHT" if side == "LEFT" else "LEFT"
        ph = i % period
        if ph < into:
            hold(session, 1, side, "A", reason="into")
        elif ph < into + flip:
            hold(session, 1, opp, "A", reason="flip")
        else:
            hold(session, 1, opp, "B", "A", reason="spin")
    return _end_report(session, "continue_wj")


def recipe_tight_wj(session: _ProbeSession, **kw) -> dict:
    # tighter period 12/4/6
    return recipe_continue_wj(
        session,
        frames=kw.get("frames", 2400),
        period=12,
        into=4,
        flip=6,
        start_side=kw.get("start_side", "LEFT"),
        switch_every=kw.get("switch_every", 300),
    )


def recipe_loose_wj(session: _ProbeSession, **kw) -> dict:
    return recipe_continue_wj(
        session,
        frames=kw.get("frames", 2400),
        period=20,
        into=8,
        flip=10,
        start_side=kw.get("start_side", "LEFT"),
        switch_every=kw.get("switch_every", 500),
    )


def recipe_latch_wj(session: _ProbeSession, *, frames: int = 2000) -> dict:
    for cycle in range(frames // 20):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_TOP_DOOR_Y + 25:
            break
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason="u")
            continue
        if is_wall_latch(st):
            into = "LEFT" if x >= 140 else "RIGHT"
            flip = "RIGHT" if into == "LEFT" else "LEFT"
            walljump_once(
                session,
                WallJumpTiming(
                    into=into, flip=flip, into_frames=4, amid_frames=2, flip_frames=14
                ),
                reason=f"latched_{cycle}",
            )
            continue
        if x >= 200:
            hold(session, 1, "LEFT", reason="face")
            for _ in range(20):
                st2 = hold(session, 1, "LEFT", "A", reason="wj_r")
                if is_wall_latch(st2) or int(st2.samus_y) < y - 20:
                    break
            continue
        if x <= 50:
            hold(session, 1, "RIGHT", reason="face")
            for _ in range(20):
                st2 = hold(session, 1, "RIGHT", "A", reason="wj_l")
                if is_wall_latch(st2) or int(st2.samus_y) < y - 20:
                    break
            continue
        direction = "RIGHT" if x < 130 else "LEFT"
        hold(session, 2, direction, "B", reason="run")
        for _ in range(28):
            st2 = hold(session, 1, direction, "B", "A", reason="spin")
            if is_wall_latch(st2):
                break
            if int(st2.samus_x) >= 210 or int(st2.samus_x) <= 45:
                break
    return _end_report(session, "latch_wj")


def recipe_seat_wj(session: _ProbeSession) -> dict:
    """Try land thin seat ~y587 x85-89 or y740 wide, then relaunch WJ."""
    # free-fall / steer toward left seat band
    for _ in range(180):
        st = session.state
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        x = int(st.samus_x)
        vy = int(st.velocity_y)
        if vy == 0 and y < 800:
            break
        # aim for x~87
        if x > 95:
            hold(session, 1, "LEFT", reason="to_seat")
        elif x < 75:
            hold(session, 1, "RIGHT", reason="to_seat")
        else:
            hold(session, 1, reason="fall")
    land = _pin(session.state)
    settle_hold(session, 8, reason="seat_s")
    for _ in range(16):
        st = session.state
        if int(st.pose) in (1, 2):
            break
        if int(st.pose) in (29, 30, 31, 32, 137, 138, 9, 10):
            hold(session, 1, "UP", reason="stand")
        else:
            break
    # hop up / WJ from seat
    hold(session, 4, "LEFT" if int(session.state.samus_x) > 128 else "RIGHT", "B", reason="run")
    for _ in range(14):
        hold(session, 1, "LEFT" if int(session.state.samus_x) > 128 else "RIGHT", "B", "A", reason="j")
    r = recipe_continue_wj(session, frames=2000, start_side="LEFT", switch_every=350)
    r["land"] = land
    r["recipe"] = "seat_wj"
    return r


def recipe_right_plat_hop(session: _ProbeSession) -> dict:
    """Bias right wall platforms (y1284/1028/740) then hop to top door."""
    for i in range(2500):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_TOP_DOOR_Y + 30:
            break
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason="u")
            continue
        # prefer right structure
        if x < 190:
            if int(st.velocity_y) != 0 or i % 20 < 12:
                hold(session, 1, "RIGHT", "B", "A", reason="r_ja")
            else:
                hold(session, 1, "RIGHT", "B", reason="r_run")
            continue
        ph = i % 16
        if ph < 6:
            hold(session, 1, "RIGHT", "A", reason="into")
        elif ph < 14:
            hold(session, 1, "LEFT", "A", reason="flip")
        else:
            hold(session, 1, "LEFT", "B", "A", reason="spin")
    # try exit
    for _ in range(120):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        if int(st.samus_y) > RED_TOP_DOOR_Y + 50:
            hold(session, 1, "RIGHT", "B", "A", reason="top_hop")
        elif int(st.samus_x) < RED_TOP_DOOR_X - 20:
            hold(session, 1, "RIGHT", "B", reason="top_run")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason="exit")
    return _end_report(session, "right_plat_hop")


def recipe_shoot_hop(session: _ProbeSession) -> dict:
    """Spray shots while WJ — clear any shot blocks near top."""
    for i in range(2200):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        y = int(st.samus_y)
        if y <= RED_TOP_DOOR_Y + 25:
            break
        if int(st.pose) in (29, 30, 31, 32):
            hold(session, 1, "UP", reason="u")
            continue
        side = "LEFT" if (i // 400) % 2 == 0 else "RIGHT"
        opp = "RIGHT" if side == "LEFT" else "LEFT"
        ph = i % 16
        if i % 40 < 4:
            hold(session, 1, "X", reason="shot")
            continue
        if ph < 6:
            hold(session, 1, side, "A", reason="into")
        elif ph < 14:
            hold(session, 1, opp, "A", reason="flip")
        else:
            hold(session, 1, opp, "B", "A", reason="spin")
    return _end_report(session, "shoot_hop")


def recipe_exit_dash(session: _ProbeSession) -> dict:
    """If already near top, just run RIGHT into Hellway."""
    for _ in range(200):
        st = session.state
        if int(st.room_id) == ROOM_HELLWAY:
            break
        if int(st.room_id) != ROOM_RED_TOWER:
            break
        if int(st.samus_y) > RED_TOP_DOOR_Y + 80:
            hold(session, 1, "RIGHT", "B", "A", reason="up")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason="exit")
    return _end_report(session, "exit_dash")


def _end_report(session: _ProbeSession, name: str) -> dict:
    # attempt exit if near door
    st = session.state
    if (
        int(st.room_id) == ROOM_RED_TOWER
        and int(st.samus_y) <= RED_TOP_DOOR_Y + 80
    ):
        for _ in range(150):
            st = session.state
            if int(st.room_id) == ROOM_HELLWAY:
                break
            if int(st.room_id) != ROOM_RED_TOWER:
                break
            hold(session, 1, "RIGHT", "B", "A", reason="exit")
    return {
        "recipe": name,
        "end": _pin(session.state),
        "best_y": session.best_y,
        "best_xy": list(session.best_xy),
        "frames": session.frame,
        "hellway": int(session.state.room_id) == ROOM_HELLWAY,
    }


RECIPES = {
    "continue_wj": recipe_continue_wj,
    "tight_wj": recipe_tight_wj,
    "loose_wj": recipe_loose_wj,
    "latch_wj": recipe_latch_wj,
    "seat_wj": recipe_seat_wj,
    "right_plat_hop": recipe_right_plat_hop,
    "shoot_hop": recipe_shoot_hop,
    "exit_dash": recipe_exit_dash,
}


def mode_capture(source: Path) -> dict:
    env, session = _boot(source)
    try:
        pre = _to_mid_handoff(session)
        print(f"handoff {pre['after850']} best={session.best_y}", flush=True)
        wj = _period_wj_instrumented(session, save_peak_env=env, peak_threshold=480)
        _save(env, END_STATE)
        print(f"saved end → {END_STATE}", flush=True)
        # if peak never saved but best is low, save now
        if not wj.get("peak_saved") and session.best_y <= 500:
            _save(env, PEAK_STATE)
            print(f"late-saved peak best={session.best_y}", flush=True)
        return {
            "mode": "capture",
            "pre": pre,
            "wj": wj,
            "end": _pin(session.state),
            "best_y": session.best_y,
            "best_xy": list(session.best_xy),
            "frames": session.frame,
            "peak_state": str(PEAK_STATE) if PEAK_STATE.exists() else None,
            "end_state": str(END_STATE),
        }
    finally:
        env.close()


def mode_from_pin(source: Path, recipe: str) -> dict:
    env, session = _boot(source)
    try:
        # small settle
        settle_hold(session, 4, reason="boot")
        start = _pin(session.state)
        print(f"start {start} recipe={recipe}", flush=True)
        fn = RECIPES[recipe]
        r = fn(session)
        r["start"] = start
        print(json.dumps(r, indent=2), flush=True)
        return r
    finally:
        env.close()


def mode_product(source: Path) -> dict:
    """Run product play_red_to_hellway."""
    from super_metroid.routes.kpdr.k5.red_to_hellway import play_red_to_hellway
    from super_metroid.routes.runtime import ControllerSession

    env, session = _boot(source)
    try:
        # wrap as ControllerSession-compatible (ProbeSession has same step API)
        try:
            play_red_to_hellway(session)  # type: ignore[arg-type]
            err = None
        except Exception as exc:
            err = str(exc)
        return {
            "mode": "product",
            "end": _pin(session.state),
            "best_y": session.best_y,
            "best_xy": list(session.best_xy),
            "frames": session.frame,
            "error": err,
            "hellway": int(session.state.room_id) == ROOM_HELLWAY,
        }
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("capture", "from-peak", "from-end", "sweep", "product"),
        default="capture",
    )
    ap.add_argument("--recipe", default="continue_wj", choices=sorted(RECIPES))
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument(
        "--source-pin",
        choices=("peak", "end", "raw"),
        default="peak",
        help="sweep/from: which pin state to boot",
    )
    args = ap.parse_args()

    if args.mode == "capture":
        r = mode_capture(args.source)
        print(json.dumps(r, indent=2, default=str))
        return

    if args.mode == "product":
        r = mode_product(args.source)
        print(json.dumps(r, indent=2))
        return

    pin_map = {
        "peak": PEAK_STATE,
        "end": END_STATE,
        "raw": args.source,
    }
    if args.mode in ("from-peak", "from-end"):
        src = PEAK_STATE if args.mode == "from-peak" else END_STATE
        if not src.exists():
            print(f"missing {src}; run --mode capture first", file=sys.stderr)
            sys.exit(2)
        r = mode_from_pin(src, args.recipe)
        print(json.dumps(r, indent=2, default=str))
        return

    if args.mode == "sweep":
        src = pin_map[args.source_pin]
        if not src.exists():
            print(f"missing {src}; run --mode capture first", file=sys.stderr)
            sys.exit(2)
        results = []
        for name in RECIPES:
            print(f"\n=== RECIPE {name} from {args.source_pin} ===", flush=True)
            try:
                r = mode_from_pin(src, name)
                results.append(r)
                print(
                    f"→ {name}: best_y={r['best_y']} end=({r['end']['x']},{r['end']['y']}) "
                    f"hell={r.get('hellway')} room={r['end']['room']}",
                    flush=True,
                )
            except Exception as exc:
                results.append({"recipe": name, "error": str(exc)})
                print(f"→ {name} ERROR {exc}", flush=True)
                traceback.print_exc()
        print(json.dumps(results, indent=2, default=str))
        return


if __name__ == "__main__":
    main()
