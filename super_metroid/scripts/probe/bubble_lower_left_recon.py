#!/usr/bin/env python3
"""Diagnostic place-grid + short controller recon: Bubble lower-left → save door.

Development-only. Boots the CATH-04 pure Bubble entry, places Samus on a left-
column (x,y) grid, settles, and records which cells are standable. Optionally
runs short charged-HJ climbs from each solid shelf toward the mid-iso pin
(~x105 y370). Not pure-green evidence; no progression / capacity / door writes.
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
    place_samus,
)
from super_metroid.ram import parse_env_state  # noqa: E402

ROOM_BUBBLE = 0xACB3
DEFAULT_SOURCE = (
    ROOT
    / "super_metroid"
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
    / "post_rising_tide_to_bubble_pure.state"
)
DEFAULT_OUTPUT = ROOT / "super_metroid" / "debug" / "bubble_lower_left_recon.json"

# Mid-iso / save-door pin class (post_bubble_mid_climb_pure).
PIN_X = (77, 160)
PIN_Y = (350, 400)
STAND_PIN_POSES = frozenset({1, 2, 9, 10, 25, 26, 27, 28})

# Left-column place grid: entry y~637 → pin y~370. x near save-door column.
DEFAULT_XS = (70, 90, 105, 120, 140, 180, 220)
DEFAULT_YS = (640, 600, 560, 520, 500, 480, 460, 440, 420, 400, 385, 370, 355)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pin_dict(state: Any) -> dict[str, object]:
    return {
        "room": f"0x{state.room_id:04X}",
        "pose": int(state.pose),
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "vx": int(state.velocity_x),
        "vy": int(state.velocity_y),
        "door_transition": int(state.door_transition),
    }


def _on_pin(state: Any) -> bool:
    return (
        abs(int(state.velocity_y)) <= 2
        and int(state.pose) in STAND_PIN_POSES
        and PIN_X[0] <= int(state.samus_x) <= PIN_X[1]
        and PIN_Y[0] <= int(state.samus_y) <= PIN_Y[1]
    )


def _settle(env: Any, assist: UnlimitedResourcesAssist, frames: int = 40) -> Any:
    state = parse_env_state(env, frame=0, mode="full")
    for i in range(frames):
        env.step(idle_action())
        state = parse_env_state(env, frame=i + 1, mode="full")
        assist.apply(env.data, state)
    return state


def place_trial(
    source: Path,
    x: int,
    y: int,
    settle: int = 45,
) -> dict[str, object]:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        state = _settle(env, assist, 12)
        place_samus(env, x, y)
        state = _settle(env, assist, settle)
        grounded = (
            abs(int(state.velocity_y)) <= 1
            and int(state.pose) in STAND_PIN_POSES | {37, 38}
        )
        return {
            "place": {"x": x, "y": y},
            "final": pin_dict(state),
            "grounded": grounded,
            "on_pin": _on_pin(state),
            "dy": int(state.samus_y) - y,
            "dx": int(state.samus_x) - x,
            "same_room": int(state.room_id) == ROOM_BUBBLE,
            "assist": assist.report(),
        }
    finally:
        env.close()


def climb_from_place(
    source: Path,
    x: int,
    y: int,
    *,
    frames: int = 420,
    dir_bias: str = "LEFT",
) -> dict[str, object]:
    """Place on a shelf, charged HJ with left/right bias toward pin."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        _settle(env, assist, 12)
        place_samus(env, x, y)
        state = _settle(env, assist, 30)
        start = pin_dict(state)
        min_y = int(state.samus_y)
        max_x = int(state.samus_x)
        pinned = False
        for frame in range(frames):
            if int(state.room_id) != ROOM_BUBBLE:
                break
            if _on_pin(state):
                pinned = True
                break
            sx, sy = int(state.samus_x), int(state.samus_y)
            min_y = min(min_y, sy)
            max_x = max(max_x, sx)
            # Hard-avoid left door plunge / SC right.
            if sx < 55:
                names = ("RIGHT", "B")
            elif sx > 400:
                names = ("LEFT", "B")
            elif abs(int(state.velocity_y)) <= 1 and int(state.pose) in (
                1,
                2,
                9,
                10,
            ):
                # Charge then HJ.
                for _ in range(10):
                    env.step(buttons("A"))
                    state = parse_env_state(env, frame=frame, mode="full")
                    assist.apply(env.data, state)
                d = "RIGHT" if sx < 95 else (dir_bias if sx > 130 else "RIGHT")
                for _ in range(40):
                    env.step(buttons(d, "B", "A"))
                    state = parse_env_state(env, frame=frame, mode="full")
                    assist.apply(env.data, state)
                    min_y = min(min_y, int(state.samus_y))
                    max_x = max(max_x, int(state.samus_x))
                    if _on_pin(state):
                        pinned = True
                        break
                if pinned:
                    break
                continue
            else:
                d = "RIGHT" if sx < 90 else ("LEFT" if sx > 140 else dir_bias)
                names = (d, "B", "A")
            env.step(buttons(*names))
            state = parse_env_state(env, frame=frame + 1, mode="full")
            assist.apply(env.data, state)

        return {
            "place": {"x": x, "y": y},
            "start": start,
            "final": pin_dict(state),
            "min_y": min_y,
            "max_x": max_x,
            "pinned": pinned,
            "on_pin_final": _on_pin(state),
            "frames": frame + 1 if frames else 0,
            "assist": assist.report(),
        }
    finally:
        env.close()


def natural_left_column(
    source: Path,
    *,
    frames: int = 2000,
    strategy: str = "wall_hug_hj",
) -> dict[str, object]:
    """Controller-only climb from natural CATH-04 entry (no place)."""
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        boot_from_state(env, source)
        state = parse_env_state(env, frame=0, mode="full")
        assist.apply(env.data, state)
        start = pin_dict(state)
        min_y = int(state.samus_y)
        max_x = int(state.samus_x)
        samples: list[dict[str, object]] = []
        pinned = False
        for frame in range(frames):
            if int(state.room_id) != ROOM_BUBBLE:
                break
            if _on_pin(state):
                pinned = True
                samples.append({"frame": frame, **pin_dict(state), "event": "pin"})
                break
            sx, sy = int(state.samus_x), int(state.samus_y)
            min_y = min(min_y, sy)
            max_x = max(max_x, sx)
            if frame % 30 == 0:
                samples.append({"frame": frame, **pin_dict(state)})

            if sx < 55:
                names: tuple[str, ...] = ("RIGHT", "B")
            elif sx > 280 and sy > 400:
                # Pull back from cavity mid-right shelves.
                names = ("LEFT", "B", "A")
            elif strategy == "wall_hug_hj":
                # Prefer x∈[70,130] while climbing.
                if abs(int(state.velocity_y)) <= 1 and int(state.pose) in (
                    1,
                    2,
                    9,
                    10,
                ):
                    for _ in range(8):
                        env.step(buttons("A"))
                        state = parse_env_state(env, frame=frame, mode="full")
                        assist.apply(env.data, state)
                    d = "RIGHT" if sx < 85 else ("LEFT" if sx > 125 else "RIGHT")
                    for _ in range(36):
                        env.step(buttons(d, "B", "A"))
                        state = parse_env_state(env, frame=frame, mode="full")
                        assist.apply(env.data, state)
                        min_y = min(min_y, int(state.samus_y))
                        max_x = max(max_x, int(state.samus_x))
                        if _on_pin(state):
                            pinned = True
                            break
                    if pinned:
                        break
                    continue
                d = "RIGHT" if sx < 85 else ("LEFT" if sx > 125 else "LEFT")
                phase = frame % 24
                if phase < 16:
                    names = (d, "B", "A")
                elif phase < 20:
                    names = (d, "B", "X")
                else:
                    names = (d, "B")
            elif strategy == "left_wall_wj":
                # Spin / WJ against left cavity while holding near left.
                d = "LEFT" if sx > 100 else "RIGHT"
                phase = frame % 12
                if phase < 2:
                    names = (d, "B")
                elif phase < 4:
                    opp = "RIGHT" if d == "LEFT" else "LEFT"
                    names = (opp, "A")
                else:
                    names = (d, "B", "A")
            else:
                names = ("RIGHT", "B", "A")

            env.step(buttons(*names))
            state = parse_env_state(env, frame=frame + 1, mode="full")
            assist.apply(env.data, state)

        return {
            "strategy": strategy,
            "start": start,
            "final": pin_dict(state),
            "min_y": min_y,
            "max_x": max_x,
            "pinned": pinned,
            "on_pin_final": _on_pin(state),
            "frames": frame + 1 if frames else 0,
            "samples": samples[-40:],
            "assist": assist.report(),
        }
    finally:
        env.close()


def run_probe(
    source: Path,
    output: Path,
    *,
    xs: tuple[int, ...],
    ys: tuple[int, ...],
    climb_from_solid: bool,
    natural: bool,
) -> dict[str, object]:
    if not source.exists():
        raise FileNotFoundError(source)

    places = [place_trial(source, x, y) for y in ys for x in xs]
    solid = [
        p
        for p in places
        if p["grounded"]
        and p["same_room"]
        and abs(int(p["dy"])) <= 24  # landed near place y (on a shelf)
    ]
    # Cluster solid shelves by rounded y.
    by_y: dict[int, list[dict[str, object]]] = {}
    for p in solid:
        key = int(round(int(p["final"]["y"]) / 8) * 8)  # type: ignore[index]
        by_y.setdefault(key, []).append(p)

    climbs: list[dict[str, object]] = []
    if climb_from_solid:
        # Climb from best solid cell per y band (prefer pin x band).
        for y_key in sorted(by_y.keys(), reverse=True):
            cands = by_y[y_key]
            cands = sorted(
                cands,
                key=lambda p: abs(int(p["final"]["x"]) - 105),  # type: ignore[index]
            )
            best = cands[0]
            fx = int(best["final"]["x"])  # type: ignore[index]
            fy = int(best["final"]["y"])  # type: ignore[index]
            climbs.append(climb_from_place(source, fx, fy, frames=500))

    naturals: list[dict[str, object]] = []
    if natural:
        for strategy in ("wall_hug_hj", "left_wall_wj"):
            naturals.append(natural_left_column(source, frames=2200, strategy=strategy))

    report: dict[str, object] = {
        "kind": "bubble_lower_left_recon",
        "developmentOnly": True,
        "source": display_path(source),
        "pinTarget": {"x": list(PIN_X), "y": list(PIN_Y)},
        "grid": {"xs": list(xs), "ys": list(ys)},
        "placeTrials": places,
        "solidShelves": [
            {
                "place": p["place"],
                "final": p["final"],
                "on_pin": p["on_pin"],
            }
            for p in solid
        ],
        "solidYBands": {str(k): len(v) for k, v in sorted(by_y.items())},
        "climbsFromSolid": climbs,
        "naturalClimbs": naturals,
        "nonClaims": [
            "Diagnostic recon only; not pure-green evidence",
            "Not continuous evidence and no STATUS promotion",
            "place_samus is development-only geometry probe",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-climb", action="store_true")
    parser.add_argument("--no-natural", action="store_true")
    args = parser.parse_args(argv)

    report = run_probe(
        args.source,
        args.output,
        xs=DEFAULT_XS,
        ys=DEFAULT_YS,
        climb_from_solid=not args.no_climb,
        natural=not args.no_natural,
    )
    solid = report["solidShelves"]
    print(f"source={report['source']}")
    print(f"solid_shelves={len(solid)}")  # type: ignore[arg-type]
    print(f"y_bands={report['solidYBands']}")
    for p in solid:  # type: ignore[assignment]
        f = p["final"]  # type: ignore[index]
        print(
            f"  place={p['place']} → xy=({f['x']},{f['y']}) "  # type: ignore[index]
            f"pose={f['pose']} on_pin={p['on_pin']}"  # type: ignore[index]
        )
    for c in report["climbsFromSolid"]:  # type: ignore[assignment]
        print(
            f"climb place={c['place']} min_y={c['min_y']} max_x={c['max_x']} "  # type: ignore[index]
            f"pinned={c['pinned']} final={c['final']}"  # type: ignore[index]
        )
    for n in report["naturalClimbs"]:  # type: ignore[assignment]
        print(
            f"natural {n['strategy']}: min_y={n['min_y']} max_x={n['max_x']} "  # type: ignore[index]
            f"pinned={n['pinned']} final={n['final']}"  # type: ignore[index]
        )
    print(f"wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
