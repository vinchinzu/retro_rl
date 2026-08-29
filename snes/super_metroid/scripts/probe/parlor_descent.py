#!/usr/bin/env python3
"""Parlor first-descent probe: capture enter pin, seed vs moonfall bench.

Same pin both rows. Moonwalk is a file option (``$09E4``); the moonfall
row pokes it on. Leave proof is RAM + JSON, not an MP4.

```bash
# Power-on → Parlor ordinary (long). Writes scratch/parlor_descent_enter.state
uv run python snes/super_metroid/scripts/probe/parlor_descent.py capture

# Faster practice pin: door-warp landing→Parlor (awake parlor; seed desyncs)
uv run python snes/super_metroid/scripts/probe/parlor_descent.py capture-warp

# Idle dump
uv run python snes/super_metroid/scripts/probe/parlor_descent.py dump --frames 200

# One policy from the pin
uv run python snes/super_metroid/scripts/probe/parlor_descent.py strategy --policy seed
uv run python snes/super_metroid/scripts/probe/parlor_descent.py strategy --policy moonfall

# Before/after from the same pin
uv run python snes/super_metroid/scripts/probe/parlor_descent.py bench

# Diagnostics
uv run python snes/super_metroid/scripts/probe/parlor_descent.py trace --shots snes/super_metroid/scratch/parlor_descent_shots
uv run python snes/super_metroid/scripts/probe/parlor_descent.py lip
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.combat.probe import (
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.common import door_warp, save_dev_state
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.progression import MORPH_GRAPH
from super_metroid.ram import ADDR_MOONWALK, parse_state, set_moonwalk
from super_metroid.room_timer import format_segment_time
from retro_harness.env import make_env
from super_metroid.routes.kpdr.early_spine import (
    play_boot_to_ceres,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
    play_landing_to_parlor,
    play_parlor_to_climb,
)
from super_metroid.routes.kpdr.parlor_descent import (
    parlor_moonfall_action,
    play_parlor_to_climb_moonfall,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PARLOR
from super_metroid.routes.runtime import RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.moonfall import is_airborne, is_moonfalling, is_moonwalking

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "parlor_descent_enter.state"
DEFAULT_MOONWALK = SCRATCH_STATE_DIR / "parlor_descent_enter_moonwalk.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "parlor_descent_bench.json"
# Landing Site left door into Parlor top-right (maps/path_room_board.json).
DOOR_LANDING_TO_PARLOR = 0x8916
DEFAULT_WARP_SOURCE = SCRATCH_STATE_DIR / "full_start_v1_morph.state"

_NAMED_STATES: dict[str, Path] = {
    "enter": DEFAULT_ENTRY,
    "entry": DEFAULT_ENTRY,
    "parlor_descent_enter": DEFAULT_ENTRY,
    "moonwalk": DEFAULT_MOONWALK,
}


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    # Product parlor seed starts in the landing→parlor door (gs=11). Extra
    # idle settle desyncs the hash-pinned seed.
    return open_state_env(
        state_path,
        settle=0,
        missing_hint="Capture first: parlor_descent.py capture",
    )


def _snapshot(session: RouteSession, extra: dict | None = None) -> dict[str, object]:
    state = session.state
    env = getattr(session, "env", None)
    moonwalk = int(state.moonwalk)
    if env is not None:
        moonwalk = _u16(env.get_ram(), ADDR_MOONWALK)
    out: dict[str, object] = {
        "room_id_hex": f"0x{state.room_id:04X}",
        "samus_x": state.samus_x,
        "samus_y": state.samus_y,
        "pose": state.pose,
        "facing": state.facing,
        "movement_type": state.movement_type,
        "vertical_direction": state.vertical_direction,
        "velocity_y": state.velocity_y,
        "game_state": state.game_state,
        "moonwalk": moonwalk,
        "collected_items": f"0x{state.collected_items:04X}",
    }
    if extra:
        out.update(extra)
    return out


def _make_session(env) -> RouteSession:
    assist = UnlimitedAmmoAssist(enabled=True)
    return RouteSession(env, writer=None, assist=assist, graph=MORPH_GRAPH)


def _dump_parlor_pins(env, session: RouteSession, out: Path, moon_path: Path) -> dict:
    save_dev_state(env, out)
    set_moonwalk(env, True)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    save_dev_state(env, moon_path)
    return {
        "saved_state": str(out),
        "moonwalk_state": str(moon_path),
        "entry": _snapshot(session),
        "frame": session.frame,
        "timing": format_segment_time(session.frame),
    }


def cmd_capture(args: argparse.Namespace) -> int:
    """Power-on → Parlor ordinary (first descent; planet not awake)."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    try:
        session = _make_session(env)
        play_boot_to_ceres(session)
        play_ceres_outbound_to_ridley(session)
        play_ceres_escape_to_landing(session)
        play_landing_to_parlor(session)
        if session.state.room_id != ROOM_PARLOR or session.state.game_state != 8:
            report = {
                "command": "capture",
                "success": False,
                "outcome": "not_in_parlor",
                "final": _snapshot(session),
                "frame": session.frame,
            }
            write_json_report(report, Path(args.report) if args.report else None)
            return 1
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        moon_path = Path(args.moonwalk_state) if args.moonwalk_state else DEFAULT_MOONWALK
        body = _dump_parlor_pins(env, session, out, moon_path)
        report = {
            "command": "capture",
            "success": True,
            "notes": "Power-on → Parlor ordinary. Moonwalk pin is the same seat + $09E4=1.",
            **body,
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0
    finally:
        env.close()


def cmd_capture_warp(args: argparse.Namespace) -> int:
    """Door-warp into Parlor from a Zebes pin (practice; not continuous evidence)."""
    source = Path(args.source)
    env, loaded = _open_env(source)
    try:
        session = _make_session(env)
        door_warp(env, DOOR_LANDING_TO_PARLOR, expected_room=ROOM_PARLOR)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        if session.state.room_id != ROOM_PARLOR or session.state.game_state != 8:
            report = {
                "command": "capture-warp",
                "success": False,
                "outcome": "warp_missed_parlor",
                "source": loaded,
                "final": _snapshot(session),
            }
            write_json_report(report, Path(args.report) if args.report else None)
            return 1
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        moon_path = Path(args.moonwalk_state) if args.moonwalk_state else DEFAULT_MOONWALK
        body = _dump_parlor_pins(env, session, out, moon_path)
        report = {
            "command": "capture-warp",
            "success": True,
            "source": loaded,
            "door": hex(DOOR_LANDING_TO_PARLOR),
            "notes": (
                "Warp into Parlor via landing door 0x8916. Practice pin, not "
                "power-on evidence. Moonwalk twin has $09E4=1."
            ),
            **body,
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0
    finally:
        env.close()


def _kin(session: RouteSession, extra: dict | None = None) -> dict[str, object]:
    s = session.state
    out: dict[str, object] = {
        "f": session.frame,
        "x": s.samus_x,
        "y": s.samus_y,
        "pose": s.pose,
        "mt": s.movement_type,
        "vd": s.vertical_direction,
        "vy": s.velocity_y,
        "vx": s.velocity_x,
        "face": s.facing,
        "gs": s.game_state,
        "room": f"0x{s.room_id:04X}",
        "mw": int(s.moonwalk),
        "mf": int(is_moonfalling(s)),
        "mwalk": int(is_moonwalking(s)),
        "air": int(is_airborne(s)),
        "kb": int(is_knockback(s)),
    }
    if extra:
        out.update(extra)
    return out


def _save_shot(env, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(env.render()).save(path)


def cmd_dump(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    try:
        session = _make_session(env)
        samples = []
        for i in range(args.frames):
            if i == 0 or i + 1 == args.frames or i % 30 == 0:
                samples.append(_snapshot(session, {"i": i, "frame": session.frame}))
            session.step(idle_action(), "dump_idle")
        report = {
            "command": "dump",
            "state": loaded,
            "entry": samples[0] if samples else _snapshot(session),
            "samples": samples,
            "final": _snapshot(session),
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0
    finally:
        env.close()


def _run_policy(env, policy: str) -> tuple[RouteSession, dict]:
    session = _make_session(env)
    session.parlor_moonfall = policy == "moonfall"  # type: ignore[attr-defined]
    if session.state.room_id != ROOM_PARLOR:
        return session, {
            "success": False,
            "outcome": "wrong_room",
            "room_id_hex": f"0x{session.state.room_id:04X}",
        }
    entry = _snapshot(session)
    start = session.frame
    try:
        if policy == "moonfall":
            play_parlor_to_climb_moonfall(session, restore_moonwalk=True)
        else:
            play_parlor_to_climb(session)
    except Exception as exc:
        frames = session.frame - start
        return session, {
            "success": False,
            "outcome": f"{type(exc).__name__}: {exc}",
            "entry": entry,
            "final": _snapshot(session),
            "timing": format_segment_time(frames),
        }
    if session.state.room_id == ROOM_CLIMB and session.state.game_state != 8:
        try:
            session.wait_until(
                lambda s: s.room_id == ROOM_CLIMB and s.game_state == 8,
                timeout=180,
                reason="climb_gs8_settle",
            )
        except TimeoutError:
            pass
    frames = session.frame - start
    ok = session.state.room_id == ROOM_CLIMB and session.state.game_state == 8
    return session, {
        "success": ok,
        "outcome": "climb" if ok else f"ended_0x{session.state.room_id:04X}_gs{session.state.game_state}",
        "entry": entry,
        "final": _snapshot(session),
        "timing": format_segment_time(frames),
    }


def cmd_strategy(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    try:
        _session, body = _run_policy(env, args.policy)
        report = {
            "command": "strategy",
            "state": loaded,
            "policy": args.policy,
            **body,
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0 if body.get("success") else 1
    finally:
        env.close()


def cmd_bench(args: argparse.Namespace) -> int:
    state_path = _resolve_state(args.state)
    rows: dict[str, dict] = {}
    for policy in ("seed", "moonfall"):
        env, loaded = _open_env(state_path)
        try:
            _session, body = _run_policy(env, policy)
            rows[policy] = {"state": loaded, **body}
        finally:
            env.close()
    seed_f = int(rows["seed"].get("timing", {}).get("frames") or 0)
    moon_f = int(rows["moonfall"].get("timing", {}).get("frames") or 0)
    delta = moon_f - seed_f
    report = {
        "command": "bench",
        "state": str(state_path),
        "before": rows["seed"],
        "after": rows["moonfall"],
        "delta": {
            **format_segment_time(abs(delta)),
            "frames_signed": delta,
            "faster": "moonfall" if delta < 0 else ("seed" if delta > 0 else "tie"),
        },
        "same_pin": True,
        "notes": "Negative delta frames = moonfall faster. Practice, not continuous evidence.",
    }
    out = Path(args.report) if args.report else DEFAULT_REPORT
    write_json_report(report, out)
    return 0 if rows["seed"].get("success") and rows["moonfall"].get("success") else 1


def cmd_trace(args: argparse.Namespace) -> int:
    env, loaded = _open_env(_resolve_state(args.state))
    shots_dir = Path(args.shots) if args.shots else None
    try:
        session = _make_session(env)
        set_moonwalk(env, True)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        from super_metroid.routes.kpdr.parlor_descent import ParlorMoonfallTrack

        track = ParlorMoonfallTrack()
        samples: list[dict] = []
        events: list[dict] = []
        max_y = session.state.samus_y
        max_vy = 0
        mf_frames = 0
        last_phase = track.phase
        if shots_dir:
            _save_shot(env, shots_dir / "f000_start.png")
        for i in range(args.frames):
            st = session.state
            max_y = max(max_y, int(st.samus_y))
            max_vy = max(max_vy, int(st.velocity_y))
            if is_moonfalling(st):
                mf_frames += 1
            names, track = parlor_moonfall_action(st, track)
            row = _kin(session, {"phase": track.phase, "btns": list(names)})
            if (
                i == 0
                or i + 1 == args.frames
                or i % args.stride == 0
                or track.phase != last_phase
                or is_knockback(st)
            ):
                samples.append(row)
            if track.phase != last_phase:
                events.append({**row, "event": f"phase_{track.phase}"})
                last_phase = track.phase
                if shots_dir:
                    _save_shot(env, shots_dir / f"f{session.frame:04d}_{track.phase}.png")
            action = buttons(*names) if names else idle_action()
            session.step(action, f"trace_{track.phase}")
            if st.room_id == ROOM_CLIMB:
                events.append({**_kin(session), "event": "climb"})
                break
            if track.phase == "done":
                break
        if shots_dir:
            _save_shot(env, shots_dir / f"f{session.frame:04d}_final.png")
        report = {
            "command": "trace",
            "state": loaded,
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
            "max_y": max_y,
            "max_vy": max_vy,
            "moonfall_frames": mf_frames,
            "final": _kin(session, {"phase": track.phase}),
            "events": events,
            "samples": samples,
            "shots": str(shots_dir) if shots_dir else None,
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0 if session.state.room_id == ROOM_CLIMB else 1
    finally:
        env.close()


def cmd_lip(args: argparse.Namespace) -> int:
    """Run left on the top corridor; report last grounded x before the shaft drop."""
    env, loaded = _open_env(_resolve_state(args.state))
    try:
        session = _make_session(env)
        set_moonwalk(env, True)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        last_ground: dict | None = None
        samples: list[dict] = []
        landed = False
        for i in range(args.frames):
            st = session.state
            y = int(st.samus_y)
            if not is_airborne(st):
                landed = True
                last_ground = _kin(session)
                names = ("LEFT", "B", "X")
            else:
                names = ("LEFT", "A", "B") if landed else ("LEFT", "B")
            if i == 0 or i % 10 == 0 or (landed and is_airborne(st) and y > 200):
                samples.append(_kin(session, {"btns": list(names), "landed": int(landed)}))
            session.step(buttons(*names), "lip")
            if landed and is_airborne(session.state) and int(session.state.samus_y) > 220:
                samples.append(_kin(session, {"event": "walked_off"}))
                break
        report = {
            "command": "lip",
            "state": loaded,
            "last_grounded": last_ground,
            "final": _kin(session),
            "samples": samples,
            "frames": session.frame,
            "timing": format_segment_time(session.frame),
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0 if last_ground else 1
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("--save-state", default=str(DEFAULT_ENTRY))
    cap.add_argument("--moonwalk-state", default=str(DEFAULT_MOONWALK))
    cap.add_argument("--report")
    cap.set_defaults(func=cmd_capture)

    warp = sub.add_parser("capture-warp")
    warp.add_argument("--source", default=str(DEFAULT_WARP_SOURCE))
    warp.add_argument("--save-state", default=str(DEFAULT_ENTRY))
    warp.add_argument("--moonwalk-state", default=str(DEFAULT_MOONWALK))
    warp.add_argument("--report")
    warp.set_defaults(func=cmd_capture_warp)

    dump = sub.add_parser("dump")
    dump.add_argument("--state", default="enter")
    dump.add_argument("--frames", type=int, default=200)
    dump.add_argument("--report")
    dump.set_defaults(func=cmd_dump)

    st = sub.add_parser("strategy")
    st.add_argument("--state", default="enter")
    st.add_argument("--policy", choices=("seed", "moonfall"), default="moonfall")
    st.add_argument("--report")
    st.set_defaults(func=cmd_strategy)

    bench = sub.add_parser("bench")
    bench.add_argument("--state", default="enter")
    bench.add_argument("--report", default=str(DEFAULT_REPORT))
    bench.set_defaults(func=cmd_bench)

    tr = sub.add_parser("trace")
    tr.add_argument("--state", default="enter")
    tr.add_argument("--frames", type=int, default=800)
    tr.add_argument("--stride", type=int, default=15)
    tr.add_argument("--shots")
    tr.add_argument("--report")
    tr.set_defaults(func=cmd_trace)

    lip = sub.add_parser("lip")
    lip.add_argument("--state", default="enter")
    lip.add_argument("--frames", type=int, default=400)
    lip.add_argument("--report")
    lip.set_defaults(func=cmd_lip)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
