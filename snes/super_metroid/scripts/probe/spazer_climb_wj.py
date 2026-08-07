#!/usr/bin/env python3
"""Probe Below Spazer floor→top wall-jump climb (pure / place isolation).

Not continuous evidence. Product climb lives in routes/kpdr/spazer/climb.py.

```bash
# Natural floor entry (continuous-with-Charge)
uv run python snes/super_metroid/scripts/probe/spazer_climb_wj.py --mode floor

# Place mid shaft y=260 (probe-proven band) then double WJ
uv run python snes/super_metroid/scripts/probe/spazer_climb_wj.py --mode mid --place-y 260

# Sweep place y
uv run python snes/super_metroid/scripts/probe/spazer_climb_wj.py --mode sweep
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retro_harness.actions import buttons, idle_action  # noqa: E402
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.dev.common import (  # noqa: E402
    boot_from_state,
    make_dev_env,
    place_samus,
    save_dev_state,
)
from super_metroid.paths import GAME_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.controller_common import (  # noqa: E402
    WallJumpTiming,
    consecutive_walljumps,
    walljump_once,
)

ROOM_BELOW = 0xA408
ROOM_BAT = 0xA3DD
SCRATCH = GAME_DIR / "custom_integrations" / "SuperMetroid-Snes" / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_below_spazer_with_charge_continuous.state"
DEBUG = GAME_DIR / "debug" / "spazer_climb_wj"
TOP_Y = 160
MID_Y = 300
DOOR_SAFE_X = 48
SOLID_TOP_Y = (88, 150)
SOLID_TOP_X_MIN = 70

# Match product routes/kpdr/spazer/geometry.py mid WJ timings.
WJ_LEFT = WallJumpTiming(
    into="LEFT", flip="RIGHT", into_frames=12, amid_frames=2, flip_frames=14
)
WJ_RIGHT = WallJumpTiming(
    into="RIGHT", flip="LEFT", into_frames=12, amid_frames=2, flip_frames=12
)
WJ_PAIR = (WJ_LEFT, WJ_RIGHT)


def _on_solid_top(st) -> bool:
    return (
        int(st.room_id) == ROOM_BELOW
        and SOLID_TOP_Y[0] <= int(st.samus_y) <= SOLID_TOP_Y[1]
        and int(st.samus_x) >= SOLID_TOP_X_MIN
        and int(st.pose) in (1, 2, 9, 10)
    )


class Sess:
    def __init__(self) -> None:
        self.env = make_dev_env()
        self.assist = UnlimitedResourcesAssist()
        self.frame = 0
        self.state = parse_env_state(self.env, frame=0, mode="nav")
        self.trace: list[dict[str, int]] = []
        self.min_y = 9999
        self.min_y_xy = (0, 0)
        self.left_room = False

    def close(self) -> None:
        self.env.close()

    def boot(self, source: Path) -> None:
        boot_from_state(self.env, source)
        self.frame = 0
        self.state = parse_env_state(self.env, frame=0, mode="nav")
        self.assist.apply(self.env.data, self.state)
        for _ in range(2):
            self.hold(1)
        self.min_y = int(self.state.samus_y)
        self.min_y_xy = (int(self.state.samus_x), int(self.state.samus_y))
        self.left_room = False
        self.trace.clear()
        self._snap()

    def step(self, action, reason: str = "") -> Any:
        del reason
        self.env.step(action)
        self.frame += 1
        self.state = parse_env_state(self.env, frame=self.frame, mode="nav")
        self.assist.apply(self.env.data, self.state)
        self._snap()
        if int(self.state.room_id) != ROOM_BELOW:
            self.left_room = True
        return self.state

    def hold(self, n: int, *names: str, reason: str = "") -> Any:
        act = buttons(*names) if names else idle_action()
        for _ in range(n):
            self.step(act, reason=reason)
            if self.left_room:
                break
        return self.state

    def _snap(self) -> None:
        st = self.state
        x, y, p = int(st.samus_x), int(st.samus_y), int(st.pose)
        rid = int(st.room_id)
        self.trace.append(
            {
                "f": self.frame,
                "x": x,
                "y": y,
                "p": p,
                "room": rid,
                "door": int(st.door_transition),
            }
        )
        if rid == ROOM_BELOW and y < self.min_y:
            self.min_y = y
            self.min_y_xy = (x, y)

    def pin(self) -> dict[str, Any]:
        st = self.state
        solid = _on_solid_top(st)
        return {
            "frame": self.frame,
            "room": f"0x{int(st.room_id):04X}",
            "x": int(st.samus_x),
            "y": int(st.samus_y),
            "pose": int(st.pose),
            "door": int(st.door_transition),
            "min_y": self.min_y,
            "min_y_xy": list(self.min_y_xy),
            "top": int(st.room_id) == ROOM_BELOW and int(st.samus_y) <= TOP_Y,
            "solid_top": solid,
            "mid": int(st.room_id) == ROOM_BELOW and int(st.samus_y) <= MID_Y,
            "left_room": self.left_room or int(st.room_id) != ROOM_BELOW,
        }


def _stop_top(s: Sess):
    def _stop(st) -> bool:
        if int(st.room_id) != ROOM_BELOW:
            return True
        if _on_solid_top(st):
            return True
        if int(st.samus_y) <= 160 and int(st.samus_x) >= DOOR_SAFE_X:
            return True
        # Bat door suck — abort
        if int(st.door_transition) != 0 and int(st.samus_x) < DOOR_SAFE_X:
            return True
        return False

    return _stop


def recipe_clear_and_shot(s: Sess) -> None:
    """Off left door + a few power shots."""
    # Nudge right if hugging door
    if int(s.state.samus_x) < 55:
        s.hold(12, "RIGHT", "B", reason="clear_door")
    s.hold(4, reason="settle")
    for _ in range(6):
        s.hold(1, "X", reason="shot")
        s.hold(8, reason="shot_wait")


def recipe_floor_spin(s: Sess, *, spin: int = 28) -> None:
    """Spin-jump into left shaft; stay x>=40 if possible."""
    s.hold(4, "LEFT", reason="face_left")
    s.hold(6, "LEFT", "B", reason="run")
    s.hold(spin, "LEFT", "B", "A", reason="spin_up")
    # If rising, keep spin; bump right if door threat
    for _ in range(40):
        st = s.state
        if int(st.room_id) != ROOM_BELOW:
            return
        if int(st.samus_y) <= MID_Y:
            return
        if int(st.samus_x) < DOOR_SAFE_X:
            s.hold(1, "RIGHT", "B", "A", reason="door_bump")
        else:
            s.hold(1, "LEFT", "B", "A", reason="spin_hold")


def recipe_period_wj(s: Sess, *, frames: int = 200, period: int = 14) -> None:
    """Open-loop left-shaft period WJ (into LEFT / bounce RIGHT)."""
    stop = _stop_top(s)
    for i in range(frames):
        if stop(s.state):
            return
        st = s.state
        if int(st.samus_x) < DOOR_SAFE_X:
            s.hold(2, "RIGHT", "B", reason="door_safe")
            continue
        ph = i % period
        if ph < 4:
            s.hold(1, "LEFT", "A", reason="pwj_into")
        elif ph < 7:
            s.hold(1, "RIGHT", "A", reason="pwj_flip")
        else:
            s.hold(1, "RIGHT", "B", "A", reason="pwj_spin")


def recipe_double_wj(s: Sess, *, pair: tuple[WallJumpTiming, ...] = WJ_PAIR) -> None:
    stop = _stop_top(s)
    # Approach left wall slightly off door
    if int(s.state.samus_x) < DOOR_SAFE_X + 5:
        s.hold(8, "RIGHT", "B", reason="off_door")
    s.hold(2, "LEFT", reason="face_wall")
    consecutive_walljumps(
        s,  # type: ignore[arg-type]
        pair,
        reason="spazer_wj",
        gap_frames=2,
        stop_when=stop,
    )
    # Follow spin upward
    for _ in range(30):
        if stop(s.state):
            return
        if int(s.state.samus_x) < DOOR_SAFE_X:
            s.hold(1, "RIGHT", "B", "A", reason="follow_safe")
        else:
            s.hold(1, "LEFT", "B", "A", reason="follow_spin")


def recipe_period_crest(s: Sess, *, frames: int = 100) -> None:
    """Product crest: period over lip then settle solid top (x≥70, y∈[88,150])."""
    for i in range(frames):
        if _on_solid_top(s.state):
            return
        if int(s.state.room_id) != ROOM_BELOW:
            return
        x, y = int(s.state.samus_x), int(s.state.samus_y)
        if x >= SOLID_TOP_X_MIN and y <= 110:
            for _ in range(45):
                if _on_solid_top(s.state):
                    return
                sx = int(s.state.samus_x)
                if sx < 75:
                    s.hold(1, "RIGHT", reason="crest_land")
                elif sx > 120:
                    s.hold(1, "LEFT", reason="crest_land")
                else:
                    s.hold(1, reason="crest_land")
            return
        if y > 420:
            return
        if x < 40:
            s.hold(2, "RIGHT", "B", reason="crest_door")
            continue
        ph = i % 14
        if ph < 4:
            s.hold(1, "LEFT", "A", reason="crest_period")
        elif ph < 7:
            s.hold(1, "RIGHT", "A", reason="crest_period")
        else:
            s.hold(1, "RIGHT", "B", "A", reason="crest_period")


def recipe_mid_to_top(s: Sess) -> None:
    """From mid band: WJ height + period over-lip crest (product path)."""
    for attempt in range(5):
        if int(s.state.room_id) != ROOM_BELOW:
            return
        if _on_solid_top(s.state):
            return
        if int(s.state.samus_x) < DOOR_SAFE_X:
            s.hold(12, "RIGHT", "B", reason="off_door")
        # Two pairs then period crest (matches routes/kpdr/spazer/climb.py).
        stop = _stop_top(s)
        consecutive_walljumps(
            s,  # type: ignore[arg-type]
            WJ_PAIR + WJ_PAIR,
            reason=f"spazer_wj{attempt}",
            gap_frames=0,
            stop_when=lambda st: stop(st) or int(st.samus_y) <= 160 or _on_solid_top(st),
        )
        if _on_solid_top(s.state):
            return
        if int(s.state.samus_y) <= 200:
            recipe_period_crest(s)
        if _on_solid_top(s.state):
            return
        s.hold(3, reason="wj_gap")


def recipe_floor_to_top(s: Sess) -> None:
    recipe_clear_and_shot(s)
    # Multiple spin attempts toward mid, then WJ
    for _ in range(5):
        if int(s.state.room_id) != ROOM_BELOW:
            return
        if int(s.state.samus_y) <= MID_Y:
            break
        recipe_floor_spin(s, spin=32)
        s.hold(10, reason="land")
        if int(s.state.samus_y) > 380:
            # fell — retry from lip
            if int(s.state.samus_x) < 50:
                s.hold(10, "RIGHT", "B", reason="reseat")
    if int(s.state.samus_y) > MID_Y:
        # period WJ from low air / wall contact
        recipe_period_wj(s, frames=240)
    recipe_mid_to_top(s)
    # final period if still short
    if int(s.state.samus_y) > TOP_Y and int(s.state.room_id) == ROOM_BELOW:
        recipe_period_wj(s, frames=160)


def run_floor(source: Path) -> dict[str, Any]:
    s = Sess()
    try:
        s.boot(source)
        recipe_floor_to_top(s)
        pin = s.pin()
        pin["mode"] = "floor"
        pin["trace_len"] = len(s.trace)
        if pin["top"]:
            out = SCRATCH / "post_below_spazer_top_pure.state"
            save_dev_state(s.env, out)
            pin["saved"] = str(out)
        return pin
    finally:
        s.close()


def run_mid(source: Path, place_y: int, place_x: int = 55) -> dict[str, Any]:
    s = Sess()
    try:
        s.boot(source)
        place_samus(s.env, place_x, place_y)
        for _ in range(4):
            s.hold(1)
        recipe_mid_to_top(s)
        pin = s.pin()
        pin["mode"] = "mid"
        pin["place"] = [place_x, place_y]
        pin["trace_len"] = len(s.trace)
        return pin
    finally:
        s.close()


def run_sweep(source: Path) -> dict[str, Any]:
    rows = []
    for y in (240, 260, 280, 300, 320, 340):
        for x in (48, 55, 65):
            pin = run_mid(source, place_y=y, place_x=x)
            rows.append(pin)
            print(
                f"place x={x} y={y} → min_y={pin['min_y']} "
                f"end=({pin['x']},{pin['y']})p{pin['pose']} top={pin['top']}"
            )
    return {"mode": "sweep", "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )
    ap.add_argument(
        "--mode",
        choices=("floor", "mid", "sweep", "period"),
        default="floor",
    )
    ap.add_argument("--place-y", type=int, default=260)
    ap.add_argument("--place-x", type=int, default=55)
    args = ap.parse_args()
    DEBUG.mkdir(parents=True, exist_ok=True)
    source = args.source
    if not source.is_file():
        print(f"missing source: {source}", file=sys.stderr)
        return 2

    if args.mode == "floor":
        pin = run_floor(source)
    elif args.mode == "mid":
        pin = run_mid(source, place_y=args.place_y, place_x=args.place_x)
    elif args.mode == "period":
        s = Sess()
        try:
            s.boot(source)
            recipe_clear_and_shot(s)
            recipe_floor_spin(s)
            recipe_period_wj(s, frames=300)
            pin = s.pin()
            pin["mode"] = "period"
        finally:
            s.close()
    else:
        pin = run_sweep(source)

    out = DEBUG / f"{args.mode}_pin.json"
    out.write_text(json.dumps(pin, indent=2) + "\n")
    print(json.dumps(pin, indent=2))
    print(f"wrote {out}")
    # Success = solid top land (node 4), not merely air y≤160.
    if isinstance(pin, dict) and pin.get("solid_top"):
        return 0
    if args.mode == "sweep":
        tops = [r for r in pin.get("rows", []) if r.get("solid_top")]
        return 0 if tops else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
