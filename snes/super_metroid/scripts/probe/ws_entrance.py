#!/usr/bin/env python3
"""Wrecked Ship Entrance → Main Shaft pure hop (rr-ahjo).

Unpowered 4-screen hallway. Dash right, tank Coverns, beam the blue door.
https://wiki.supermetroid.run/Wrecked_Ship_Entrance

```bash
uv run python snes/super_metroid/scripts/probe/ws_entrance.py bench
uv run python snes/super_metroid/scripts/probe/ws_entrance.py dump
uv run python snes/super_metroid/scripts/probe/ws_entrance.py pure
uv run python snes/super_metroid/scripts/probe/ws_entrance.py pure --dual
```

Default source: ``scratch/post_ws_poweron.state`` (standing p1 gs=8).
Boot settle 5. No free-place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from retro_harness.actions import buttons
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env, save_dev_state
from super_metroid.paths import GAME_DIR
from super_metroid.ram import parse_env_state
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.wrecked_ship import ws_entrance as ws

SCRATCH = GAME_DIR / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_ws_poweron.state"
DEFAULT_OUT = SCRATCH / "post_ws_entrance_to_main.state"
DEFAULT_REPORT = SCRATCH / "ws_entrance_to_main.json"
DEFAULT_DUAL = SCRATCH / "ws_entrance_to_main_dual.json"
BOOT_SETTLE = 5


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
    out: dict[str, Any] = {
        "room": f"0x{int(st.room_id):04X}",
        "xy": [int(st.samus_x), int(st.samus_y)],
        "pose": int(st.pose),
        "gs": int(st.game_state),
        "dt": int(st.door_transition),
        "items": f"0x{int(st.collected_items):04X}",
        "beams": f"0x{int(st.collected_beams):04X}",
        "selected": int(st.selected_item),
        "max_pb": int(st.max_power_bombs),
        "health": int(st.health),
    }
    if extra:
        out.update(extra)
    return out


def _enemies(env: Any, n: int = 8) -> list[dict[str, Any]]:
    ram = env.get_ram()
    rows: list[dict[str, Any]] = []
    for i in range(n):
        eid = int(ram[0x0F78 + i * 0x40]) | (int(ram[0x0F79 + i * 0x40]) << 8)
        x = int(ram[0x0F7A + i * 0x40]) | (int(ram[0x0F7B + i * 0x40]) << 8)
        y = int(ram[0x0F7E + i * 0x40]) | (int(ram[0x0F7F + i * 0x40]) << 8)
        hp = int(ram[0x0F8C + i * 0x40]) | (int(ram[0x0F8D + i * 0x40]) << 8)
        if eid or hp:
            rows.append({"i": i, "id": f"0x{eid:04X}", "x": x, "y": y, "hp": hp})
    return rows


def _run_hop(
    source: Path,
    *,
    assist: bool = True,
    settle: int = BOOT_SETTLE,
    save: Path | None = None,
) -> dict[str, Any]:
    env = make_dev_env()
    a = UnlimitedResourcesAssist() if assist else None
    try:
        boot_from_state(env, source, settle_frames=settle)
        sess = _Sess(env, a)
        boot = _snap(sess.state, {"frame": 0})
        error = None
        try:
            st = ws.play_ws_entrance_to_main(sess)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
        ok = error is None and ws.ws_entrance_main_settled(st)
        if ok and save is not None:
            save_dev_state(env, save)
        timed = format_segment_time(sess.frame)
        return {
            "success": ok,
            "error": error,
            "source": str(source),
            "boot": boot,
            "final": _snap(st, {"frame": sess.frame}),
            "frames": sess.frame,
            "time": timed,
            "saved": str(save) if ok and save is not None else None,
        }
    finally:
        env.close()


def cmd_bench(args: argparse.Namespace) -> int:
    source = Path(args.source or DEFAULT_SOURCE)
    report = _run_hop(
        source,
        assist=not args.no_assist,
        settle=args.settle,
        save=Path(args.out) if args.out else None,
    )
    report["command"] = "bench"
    report["hop"] = "ws_entrance_to_main"
    out = Path(args.report or DEFAULT_REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    timed = report["time"]
    flag = "GREEN" if report["success"] else "RED"
    print(
        f"{flag} frames={timed['frames']} seconds={timed['seconds']} "
        f"clock={timed['clock']} final={report['final']} err={report['error']}"
    )
    print(f"wrote {out}")
    return 0 if report["success"] else 1


def cmd_dump(args: argparse.Namespace) -> int:
    source = Path(args.source or DEFAULT_SOURCE)
    env = make_dev_env()
    a = UnlimitedResourcesAssist() if not args.no_assist else None
    try:
        boot_from_state(env, source, settle_frames=args.settle)
        sess = _Sess(env, a)
        samples = [
            {
                **_snap(sess.state, {"frame": 0}),
                "enemies": _enemies(env),
            }
        ]
        dash = buttons("RIGHT", "B")
        door_band = None
        for i in range(args.frames):
            st = sess.step(dash)
            take = i % 20 == 19 or st.samus_x >= 900 or st.room_id != ws.ROOM_WS_ENTRANCE
            if take:
                row = {**_snap(st, {"frame": sess.frame}), "enemies": _enemies(env)}
                samples.append(row)
            if door_band is None and st.samus_x >= 900:
                door_band = samples[-1]
            if st.room_id != ws.ROOM_WS_ENTRANCE:
                break
        report = {
            "command": "dump",
            "source": str(source),
            "door_band": door_band,
            "final": samples[-1],
            "samples": samples,
        }
        out = Path(args.report or SCRATCH / "ws_entrance_to_main_dump.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"door_band": door_band, "final": samples[-1]}, indent=2))
        print(f"wrote {out}")
        return 0
    finally:
        env.close()


def cmd_pure(args: argparse.Namespace) -> int:
    source = Path(args.source or DEFAULT_SOURCE)
    save = Path(args.out or DEFAULT_OUT)
    runs = [
        _run_hop(
            source,
            assist=not args.no_assist,
            settle=args.settle,
            save=save if i == 0 else None,
        )
        for i in range(2 if args.dual else 1)
    ]
    for row in runs:
        row["command"] = "pure"
        row["hop"] = "ws_entrance_to_main"
    primary = runs[0]
    timed = primary["time"]
    dual_exact = True
    if args.dual:
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
            "runs": runs,
            "frames": primary["frames"],
            "time": timed,
        }
        dual_path = Path(args.dual_report or DEFAULT_DUAL)
        dual_path.parent.mkdir(parents=True, exist_ok=True)
        dual_path.write_text(json.dumps(dual_report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {dual_path}")
    out = Path(args.report or DEFAULT_REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    flag = "GREEN" if primary["success"] and dual_exact else "RED"
    print(
        f"{flag} dual={dual_exact if args.dual else 'n/a'} "
        f"frames={timed['frames']} seconds={timed['seconds']} "
        f"clock={timed['clock']} final={primary['final']} "
        f"saved={primary.get('saved')} err={primary['error']}"
    )
    print(f"wrote {out}")
    ok = primary["success"] and dual_exact
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(pp: argparse.ArgumentParser) -> None:
        pp.add_argument("--source", type=Path, default=None)
        pp.add_argument("--out", type=Path, default=None)
        pp.add_argument("--report", type=Path, default=None)
        pp.add_argument("--settle", type=int, default=BOOT_SETTLE)
        pp.add_argument("--no-assist", action="store_true")

    p_bench = sub.add_parser("bench", help="Run product hop once and write JSON")
    _common(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    p_dump = sub.add_parser("dump", help="RIGHT+B RAM walk (door x / Coverns)")
    _common(p_dump)
    p_dump.add_argument("--frames", type=int, default=200)
    p_dump.set_defaults(func=cmd_dump)

    p_pure = sub.add_parser("pure", help="Product hop; --dual for exact frame match")
    _common(p_pure)
    p_pure.add_argument("--dual", action="store_true")
    p_pure.add_argument("--dual-report", type=Path, default=None)
    p_pure.set_defaults(func=cmd_pure)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
