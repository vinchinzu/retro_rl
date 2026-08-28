#!/usr/bin/env python3
"""Wrecked Ship Main Shaft → Basement pure hop (rr-4btp).

Unpowered first visit toward Phantoon. Descend stairs (not attic, not save),
shoot floor pipes, morph, Super the green floor hatch, drop into 0xCC6F.
https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft

```bash
uv run python snes/super_metroid/scripts/probe/ws_main.py bench
uv run python snes/super_metroid/scripts/probe/ws_main.py dump
uv run python snes/super_metroid/scripts/probe/ws_main.py pure
uv run python snes/super_metroid/scripts/probe/ws_main.py pure --dual
```

Default source: ``scratch/post_ws_entrance_to_main.state`` (p9 gs=8).
Boot settle 5. No free-place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env, save_dev_state
from super_metroid.paths import GAME_DIR
from super_metroid.ram import parse_env_state
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.controller_common import MORPH_POSES, is_morph
from super_metroid.routes.kpdr import wrecked_ship as ws
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_SAVE

SCRATCH = GAME_DIR / "scratch"
DEFAULT_SOURCE = SCRATCH / "post_ws_entrance_to_main.state"
DEFAULT_OUT = SCRATCH / "post_ws_main_to_basement.state"
DEFAULT_REPORT = SCRATCH / "ws_main_to_basement.json"
DEFAULT_DUAL = SCRATCH / "ws_main_to_basement_dual.json"
DEFAULT_BEFORE = SCRATCH / "ws_main_to_basement_before.json"
DEFAULT_DUMP = SCRATCH / "ws_main_to_basement_dump.json"
BOOT_SETTLE = 5

# Wrong-room halt (do not enter). Save 0xCE8A, Attic 0xCA52, back to Entrance.

# Save door is on the right of the entry platform. Dash RIGHT hits it.
SAVE_DOOR_X = 1240


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
    out: dict[str, Any] = {
        "room": f"0x{int(st.room_id):04X}",
        "xy": [int(st.samus_x), int(st.samus_y)],
        "pose": pose,
        "gs": int(st.game_state),
        "dt": int(st.door_transition),
        "morph": pose in MORPH_POSES or is_morph(pose),
        "selected": int(st.selected_item),
        "items": f"0x{int(st.collected_items):04X}",
        "beams": f"0x{int(st.collected_beams):04X}",
        "max_pb": int(st.max_power_bombs),
        "health": int(st.health),
        "vx": int(st.velocity_x),
        "vy": int(st.velocity_y),
        "facing": int(st.facing),
        "movement": int(st.movement_type),
        "door_ptr": f"0x{int(st.door_def_ptr):04X}",
        "floor_y": int(st.samus_y) if int(st.velocity_y) == 0 else None,
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


def _plms(env: Any, n: int = 24) -> dict[str, Any]:
    """Best-effort PLM ID + block index (diagnostic; not a validated open-state)."""
    ram = env.get_ram()
    width = int(ram[0x07A5]) | (int(ram[0x07A6]) << 8)
    height = int(ram[0x07A7]) | (int(ram[0x07A8]) << 8)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        pid = int(ram[0x1C37 + i * 2]) | (int(ram[0x1C38 + i * 2]) << 8)
        block = int(ram[0x1C87 + i * 2]) | (int(ram[0x1C88 + i * 2]) << 8)
        if not pid:
            continue
        bx = block % width if width else None
        by = block // width if width else None
        rows.append(
            {
                "i": i,
                "id": f"0x{pid:04X}",
                "block": block,
                "bx": bx,
                "by": by,
                "px": (bx * 16 + 8) if bx is not None else None,
                "py": (by * 16 + 8) if by is not None else None,
            }
        )
    return {"room_width": width, "room_height": height, "plms": rows}


def _rich(env: Any, st: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = _snap(st, extra)
    row["enemies"] = _enemies(env)
    row["plms"] = _plms(env)
    return row


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
            st = ws.play_ws_main_to_basement(sess)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            st = sess.state
        ok = error is None and ws.ws_main_basement_settled(st)
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
    report["hop"] = "ws_main_to_basement"
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


def _halt_reason(st: Any, *, start_y: int, idle_done: bool) -> str | None:
    room = int(st.room_id)
    x = int(st.samus_x)
    y = int(st.samus_y)
    if room == ROOM_WS_ATTIC:
        return "attic_0xCA52"
    if room == ROOM_WS_SAVE:
        return "save_0xCE8A"
    if room == ws.ROOM_WS_ENTRANCE:
        return "back_entrance_0xCA08"
    if room == ws.ROOM_WS_BASEMENT:
        return "basement_0xCC6F"
    if room != ws.ROOM_WS_MAIN:
        return f"wrong_room_0x{room:04X}"
    if y < start_y - 40:
        return "went_up"
    if x >= SAVE_DOOR_X and y <= start_y + 40:
        return "save_door_x_band"
    if idle_done and int(st.door_transition) != 0:
        return "door_transition"
    return None


def cmd_dump(args: argparse.Namespace) -> int:
    """DOWN+RIGHT stair walk from the pin. Halt at first miss. No free-place."""
    source = Path(args.source or DEFAULT_SOURCE)
    env = make_dev_env()
    a = UnlimitedResourcesAssist() if not args.no_assist else None
    try:
        boot_from_state(env, source, settle_frames=args.settle)
        sess = _Sess(env, a)
        start_y = int(sess.state.samus_y)
        samples = [_rich(env, sess.state, {"frame": 0, "phase": "boot"})]
        halt = None
        btn_names = tuple(
            b.strip().upper() for b in str(args.buttons).split(",") if b.strip()
        )
        if args.screenshot:
            from PIL import Image

            rgb = env.render()
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(args.screenshot)
        # Observe leftover p9 momentum before committing to the walk.
        idle = idle_action()
        for i in range(min(20, args.frames)):
            st = sess.step(idle)
            if i == 19 or int(st.samus_y) != start_y or int(st.room_id) != ws.ROOM_WS_MAIN:
                samples.append(_rich(env, st, {"frame": sess.frame, "phase": "idle"}))
            halt = _halt_reason(st, start_y=start_y, idle_done=False)
            if halt:
                break
        descend = buttons(*btn_names) if btn_names else idle_action()
        y_progress = start_y
        if halt is None:
            for i in range(args.frames):
                st = sess.step(descend)
                y = int(st.samus_y)
                take = (
                    i % 15 == 14
                    or y >= y_progress + 32
                    or int(st.room_id) != ws.ROOM_WS_MAIN
                    or int(st.samus_x) >= SAVE_DOOR_X
                    or (st.pose in MORPH_POSES)
                )
                if take:
                    samples.append(
                        _rich(env, st, {"frame": sess.frame, "phase": "down_right"})
                    )
                    if y > y_progress:
                        y_progress = y
                halt = _halt_reason(st, start_y=start_y, idle_done=True)
                if halt:
                    if not take:
                        samples.append(
                            _rich(
                                env,
                                st,
                                {"frame": sess.frame, "phase": "down_right_halt"},
                            )
                        )
                    break
                # Miss: still on the entry platform after ~3s and x in save band.
                if i == 200 and y <= start_y + 24:
                    halt = "no_stair_progress"
                    samples.append(
                        _rich(env, st, {"frame": sess.frame, "phase": "miss"})
                    )
                    break
        report = {
            "command": "dump",
            "hop": "ws_main_to_basement",
            "source": str(source),
            "hypothesis": f"{btn_names} from pin; halt at first miss",
            "screenshot": str(args.screenshot) if args.screenshot else None,
            "halt": halt,
            "boot": samples[0],
            "final": samples[-1],
            "samples": samples,
        }
        out = Path(args.report or DEFAULT_DUMP)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "halt": halt,
                    "boot_xy": samples[0]["xy"],
                    "final": {
                        k: samples[-1].get(k)
                        for k in ("room", "xy", "pose", "gs", "morph", "floor_y")
                    },
                    "n_samples": len(samples),
                },
                indent=2,
            )
        )
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
        row["hop"] = "ws_main_to_basement"
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

    p_dump = sub.add_parser("dump", help="RAM walk (stairs / floor door). Halt at first miss.")
    _common(p_dump)
    p_dump.add_argument("--frames", type=int, default=400)
    p_dump.add_argument(
        "--buttons",
        default="DOWN,RIGHT",
        help="Comma-separated hold for the walk (default DOWN,RIGHT)",
    )
    p_dump.add_argument("--screenshot", type=Path, default=None)
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
