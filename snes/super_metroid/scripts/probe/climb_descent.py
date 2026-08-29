#!/usr/bin/env python3
"""Climb first-descent probe: capture enter pin, seed vs moonfall bench.

Same pin both rows. Moonwalk is a file option (``$09E4``); the moonfall
row pokes it on. Leave proof is RAM + JSON, not an MP4.

```bash
# Power-on → Climb ordinary (long). Writes scratch/climb_descent_enter.state
uv run python snes/super_metroid/scripts/probe/climb_descent.py capture

# Faster practice pin: door-warp parlor→Climb (not continuous evidence)
uv run python snes/super_metroid/scripts/probe/climb_descent.py capture-warp

# Idle dump
uv run python snes/super_metroid/scripts/probe/climb_descent.py dump --frames 200

# One policy from the pin
uv run python snes/super_metroid/scripts/probe/climb_descent.py strategy --policy seed
uv run python snes/super_metroid/scripts/probe/climb_descent.py strategy --policy moonfall

# Before/after from the same pin
uv run python snes/super_metroid/scripts/probe/climb_descent.py bench

# Diagnostics (practice pin)
uv run python snes/super_metroid/scripts/probe/climb_descent.py lip
uv run python snes/super_metroid/scripts/probe/climb_descent.py trace --shots snes/super_metroid/scratch/climb_descent_shots
uv run python snes/super_metroid/scripts/probe/climb_descent.py search
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.combat.probe import (
    open_state_env,
    resolve_named_state,
    write_json_report,
)
from super_metroid.dev.common import door_warp, save_dev_state
from super_metroid.paths import GAME, GAME_DIR, SCRATCH_STATE_DIR
from super_metroid.progression import MORPH_GRAPH
from retro_harness.actions import buttons, idle_action
from super_metroid.ram import ADDR_MOONWALK, parse_state, set_moonwalk
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.climb_descent import (
    BOTTOM_Y,
    FALL_X,
    climb_moonfall_action,
    play_climb_to_pit_moonfall,
)
from super_metroid.routes.kpdr.early_spine import (
    play_boot_to_ceres,
    play_ceres_escape_to_landing,
    play_ceres_outbound_to_ridley,
    play_climb_to_pit,
    play_landing_to_parlor,
    play_parlor_to_climb,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CLIMB, ROOM_PIT
from super_metroid.routes.runtime import RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.skills.moonfall import is_airborne, is_moonfalling, is_moonwalking
from retro_harness.env import make_env

DEFAULT_ENTRY = SCRATCH_STATE_DIR / "climb_descent_enter.state"
DEFAULT_MOONWALK = SCRATCH_STATE_DIR / "climb_descent_enter_moonwalk.state"
DEFAULT_REPORT = GAME_DIR / "scratch" / "climb_descent_bench.json"
# Parlor bottom-left vertical door into Climb (maps/path_room_board.json).
DOOR_PARLOR_TO_CLIMB = 0x898E
# Zebes in-game source for warp capture (morph owned; no Hi-Jump). Ceres
# power-on capture is preferred when the elev escape is green.
DEFAULT_WARP_SOURCE = (
    SCRATCH_STATE_DIR / "full_start_v1_morph.state"
)

_NAMED_STATES: dict[str, Path] = {
    "enter": DEFAULT_ENTRY,
    "entry": DEFAULT_ENTRY,
    "climb_descent_enter": DEFAULT_ENTRY,
    "moonwalk": DEFAULT_MOONWALK,
}


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _resolve_state(name: str) -> Path:
    return resolve_named_state(name, _NAMED_STATES)


def _open_env(state_path: Path):
    return open_state_env(
        state_path,
        missing_hint="Capture first: climb_descent.py capture",
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
    # Match morph continuous: ammo-only assist. Energy stays natural (Ceres).
    assist = UnlimitedAmmoAssist(enabled=True)
    return RouteSession(env, writer=None, assist=assist, graph=MORPH_GRAPH)


def cmd_capture(args: argparse.Namespace) -> int:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    try:
        session = _make_session(env)
        play_boot_to_ceres(session)
        play_ceres_outbound_to_ridley(session)
        play_ceres_escape_to_landing(session)
        play_landing_to_parlor(session)
        play_parlor_to_climb(session)
        if session.state.room_id != ROOM_CLIMB or session.state.game_state != 8:
            report = {
                "command": "capture",
                "success": False,
                "outcome": "not_in_climb",
                "final": _snapshot(session),
                "frame": session.frame,
            }
            write_json_report(report, Path(args.report) if args.report else None)
            return 1
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        moon_path = Path(args.moonwalk_state) if args.moonwalk_state else DEFAULT_MOONWALK
        body = _dump_climb_pins(env, session, out, moon_path)
        report = {
            "command": "capture",
            "success": True,
            "notes": "Power-on → Climb ordinary. Moonwalk pin is the same seat + $09E4=1.",
            **body,
        }
        write_json_report(report, Path(args.report) if args.report else None)
        return 0
    finally:
        env.close()


def _dump_climb_pins(env, session: RouteSession, out: Path, moon_path: Path) -> dict:
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


def cmd_capture_warp(args: argparse.Namespace) -> int:
    """Door-warp into Climb from a Zebes pin (practice; not continuous evidence)."""
    source = Path(args.source)
    env, loaded = _open_env(source)
    try:
        session = _make_session(env)
        door_warp(env, DOOR_PARLOR_TO_CLIMB, expected_room=ROOM_CLIMB)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        if session.state.room_id != ROOM_CLIMB or session.state.game_state != 8:
            report = {
                "command": "capture-warp",
                "success": False,
                "outcome": "warp_missed_climb",
                "source": loaded,
                "final": _snapshot(session),
            }
            write_json_report(report, Path(args.report) if args.report else None)
            return 1
        out = Path(args.save_state) if args.save_state else DEFAULT_ENTRY
        moon_path = Path(args.moonwalk_state) if args.moonwalk_state else DEFAULT_MOONWALK
        body = _dump_climb_pins(env, session, out, moon_path)
        report = {
            "command": "capture-warp",
            "success": True,
            "source": loaded,
            "door": hex(DOOR_PARLOR_TO_CLIMB),
            "notes": (
                "Warp into Climb via parlor door 0x898E. Practice pin, not "
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
        "e0x": int(s.enemy0_x),
        "e0y": int(s.enemy0_y),
        "e0hp": int(s.enemy0_hp),
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
    session.climb_moonfall = policy == "moonfall"  # type: ignore[attr-defined]
    if session.state.room_id != ROOM_CLIMB:
        return session, {
            "success": False,
            "outcome": "wrong_room",
            "room_id_hex": f"0x{session.state.room_id:04X}",
        }
    entry = _snapshot(session)
    start = session.frame
    try:
        if policy == "moonfall":
            play_climb_to_pit_moonfall(session, restore_moonwalk=True)
        else:
            play_climb_to_pit(session)
    except Exception as exc:
        frames = session.frame - start
        return session, {
            "success": False,
            "outcome": f"{type(exc).__name__}: {exc}",
            "entry": entry,
            "final": _snapshot(session),
            "timing": format_segment_time(frames),
        }
    if session.state.room_id == ROOM_PIT and session.state.game_state != 8:
        try:
            session.wait_until(
                lambda s: s.room_id == ROOM_PIT and s.game_state == 8,
                timeout=180,
                reason="pit_gs8_settle",
            )
        except TimeoutError:
            pass
    frames = session.frame - start
    ok = session.state.room_id == ROOM_PIT and session.state.game_state == 8
    return session, {
        "success": ok,
        "outcome": "pit" if ok else f"ended_0x{session.state.room_id:04X}_gs{session.state.game_state}",
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
    """Per-frame kinematics of the current RAM moonfall policy."""
    env, loaded = _open_env(_resolve_state(args.state))
    shots_dir = Path(args.shots) if args.shots else None
    try:
        session = _make_session(env)
        set_moonwalk(env, True)
        session.state = parse_state(env.get_ram(), frame=session.frame)
        from super_metroid.routes.kpdr.climb_descent import ClimbMoonfallTrack

        track = ClimbMoonfallTrack()
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
            names, track = climb_moonfall_action(st, track)
            row = _kin(session, {"phase": track.phase, "btns": list(names)})
            if (
                i == 0
                or i + 1 == args.frames
                or i % args.stride == 0
                or track.phase != last_phase
                or is_knockback(st)
                or (not is_airborne(st) and int(st.samus_y) > 90)
            ):
                samples.append(row)
            if track.phase != last_phase:
                events.append({**row, "event": f"phase_{track.phase}"})
                last_phase = track.phase
                if shots_dir:
                    _save_shot(env, shots_dir / f"f{session.frame:04d}_{track.phase}.png")
            if is_knockback(st) and shots_dir and i % 15 == 0:
                _save_shot(env, shots_dir / f"f{session.frame:04d}_kb.png")
            action = buttons(*names) if names else idle_action()
            session.step(action, f"trace_{track.phase}")
            if st.room_id == ROOM_PIT:
                events.append({**_kin(session), "event": "pit"})
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
        return 0 if session.state.room_id == ROOM_PIT else 1
    finally:
        env.close()



def cmd_lip(args: argparse.Namespace) -> int:
    """Moonwalk right until airborne; report last grounded x (start-platform lip)."""
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
            if not is_airborne(st):
                landed = True
                last_ground = _kin(session)
                names = ("RIGHT", "X", "L")
            else:
                names = ("X", "L", "A") if not landed else ("RIGHT", "X", "L")
            if i == 0 or i % 10 == 0 or (landed and is_airborne(st)):
                samples.append(_kin(session, {"btns": list(names), "landed": int(landed)}))
            session.step(buttons(*names), "lip")
            if landed and is_airborne(session.state):
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


def _setup_then_fall(
    session: RouteSession,
    *,
    jump_x: int,
    walk_max: int,
    jump_hold: int,
    spin: int,
    clear_y: int,
    steer: str,
    max_frames: int,
    drop_right: bool = False,
    weave_left_y: int = 0,
    weave_right_y: int = 0,
) -> dict:
    """Open-loop moonfall setup used by search. Never LEFT-steers before clear_y
    unless a weave window is set."""
    set_moonwalk(session.env, True)
    session.state = parse_state(session.env.get_ram(), frame=session.frame)
    max_y = int(session.state.samus_y)
    max_vy = 0
    mf_frames = 0
    jump_at: dict | None = None
    first_ground: dict | None = None
    first_kb: dict | None = None
    landed = False
    jumped = False
    jump_left = 0
    spin_left = 0
    walk_held = 0
    for _ in range(max_frames):
        st = session.state
        x, y = int(st.samus_x), int(st.samus_y)
        max_y = max(max_y, y)
        max_vy = max(max_vy, int(st.velocity_y))
        if is_moonfalling(st):
            mf_frames += 1
        if is_knockback(st) and first_kb is None:
            first_kb = _kin(session)
        if st.room_id == ROOM_PIT and st.game_state == 8:
            return {
                "success": True,
                "outcome": "pit",
                "jump_x": jump_x,
                "steer": steer,
                "frames": session.frame,
                "timing": format_segment_time(session.frame),
                "max_y": max_y,
                "max_vy": max_vy,
                "moonfall_frames": mf_frames,
                "jump_at": jump_at,
                "first_ground": first_ground,
                "first_kb": first_kb,
                "final": _kin(session),
            }
        if st.room_id != ROOM_CLIMB:
            names = ("LEFT", "X")
        elif not landed:
            if is_airborne(st):
                # Door-drop: keep RIGHT so we may skip the start platform.
                names = ("RIGHT", "X", "L", "A") if drop_right else ("X", "L", "A")
            else:
                landed = True
                names = ("RIGHT", "X", "L")
        elif not jumped:
            walk_held += 1
            if is_airborne(st) or x >= jump_x or walk_held >= walk_max:
                jumped = True
                jump_left = jump_hold
                jump_at = _kin(session)
                names = ("RIGHT", "X", "L", "A")
            else:
                names = ("RIGHT", "X", "L")
        elif jump_left > 0:
            jump_left -= 1
            if jump_left == 0:
                spin_left = spin
            names = ("RIGHT", "X", "L", "A")
        elif spin_left > 0:
            spin_left -= 1
            names = ("RIGHT", "A")
        elif (not is_airborne(st)) and y < BOTTOM_Y:
            if first_ground is None:
                first_ground = _kin(session)
            names = ("RIGHT",) if x < FALL_X else ()
        elif y >= BOTTOM_Y:
            names = ("LEFT", "X") if x > 80 else ("LEFT", "X")
        elif weave_left_y and weave_left_y <= y < (weave_right_y or 9999):
            names = ("LEFT",)
        elif y < clear_y:
            # Residual: do not LEFT-steer the first ~200px of fall.
            if steer == "left":
                names = ("LEFT",)
            elif steer in ("right", "wall"):
                names = ("RIGHT",)
            else:
                names = ()
        elif y >= 1800:
            names = ("LEFT", "X", "L") if x > 120 else ("LEFT", "X")
        elif steer == "wall":
            if x < FALL_X - 12:
                names = ("RIGHT",)
            elif x > FALL_X + 12:
                names = ()
            else:
                names = ()
        elif steer == "right":
            names = ("RIGHT",)
        else:
            names = ()
        session.step(buttons(*names) if names else idle_action(), "search")
    return {
        "success": False,
        "outcome": "timeout" if first_kb is None else "knockback",
        "jump_x": jump_x,
        "steer": steer,
        "frames": session.frame,
        "timing": format_segment_time(session.frame),
        "max_y": max_y,
        "max_vy": max_vy,
        "moonfall_frames": mf_frames,
        "jump_at": jump_at,
        "first_ground": first_ground,
        "first_kb": first_kb,
        "final": _kin(session),
    }



def cmd_search(args: argparse.Namespace) -> int:
    """Grid jump_x × steer. Same pin each cell. Print a compact table."""
    state_path = _resolve_state(args.state)
    jump_xs = [int(p) for p in str(args.jump_x).split(",") if p.strip()]
    steers = [s.strip() for s in str(args.steer).split(",") if s.strip()]
    rows: list[dict] = []
    for jump_x in jump_xs:
        for steer in steers:
            env, loaded = _open_env(state_path)
            try:
                session = _make_session(env)
                body = _setup_then_fall(
                    session,
                    jump_x=jump_x,
                    walk_max=args.walk_max,
                    jump_hold=args.jump_hold,
                    spin=args.spin,
                    clear_y=args.clear_y,
                    steer=steer,
                    max_frames=args.frames,
                    drop_right=bool(args.drop_right),
                    weave_left_y=int(args.weave_left_y),
                    weave_right_y=int(args.weave_right_y),
                )
                body["state"] = loaded
                rows.append(body)
            finally:
                env.close()
            jx = body.get("jump_at") or {}
            fg = body.get("first_ground") or {}
            print(
                f"jx={jump_x:3d} steer={steer:5s} ok={int(body['success'])} "
                f"out={body['outcome']:10s} f={body['frames']:4d} "
                f"max_y={body['max_y']:4d} max_vy={body['max_vy']:3d} "
                f"mf={body['moonfall_frames']:3d} "
                f"jump@({jx.get('x')},{jx.get('y')}) "
                f"gnd@({fg.get('x')},{fg.get('y')})"
            )
    best = max(
        rows,
        key=lambda r: (
            int(r["success"]),
            int(r["max_y"]),
            int(r["moonfall_frames"]),
            -int(r["frames"]),
        ),
    )
    report = {
        "command": "search",
        "state": str(state_path),
        "rows": rows,
        "best": {
            "jump_x": best["jump_x"],
            "steer": best["steer"],
            "success": best["success"],
            "outcome": best["outcome"],
            "max_y": best["max_y"],
            "max_vy": best["max_vy"],
            "moonfall_frames": best["moonfall_frames"],
            "frames": best["frames"],
        },
        "notes": "Practice grid, not continuous evidence. Negative LEFT before clear_y.",
    }
    out = Path(args.report) if args.report else GAME_DIR / "scratch" / "climb_descent_search.json"
    # Compact stdout; full JSON on disk only.
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"best={report['best']} wrote={out}")
    return 0 if any(r["success"] for r in rows) else 1


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
    tr.add_argument("--frames", type=int, default=400)
    tr.add_argument("--stride", type=int, default=15)
    tr.add_argument("--shots")
    tr.add_argument("--report")
    tr.set_defaults(func=cmd_trace)

    lip = sub.add_parser("lip")
    lip.add_argument("--state", default="enter")
    lip.add_argument("--frames", type=int, default=180)
    lip.add_argument("--report")
    lip.set_defaults(func=cmd_lip)

    sr = sub.add_parser("search")
    sr.add_argument("--state", default="enter")
    sr.add_argument("--jump-x", default="360,370,380,390,400,410,420,430")
    sr.add_argument("--steer", default="right,none,wall")
    sr.add_argument("--walk-max", type=int, default=120)
    sr.add_argument("--jump-hold", type=int, default=3)
    sr.add_argument("--spin", type=int, default=4)
    sr.add_argument("--clear-y", type=int, default=280)
    sr.add_argument("--frames", type=int, default=500)
    sr.add_argument("--drop-right", action="store_true")
    sr.add_argument("--weave-left-y", type=int, default=0)
    sr.add_argument("--weave-right-y", type=int, default=0)
    sr.add_argument("--report")
    sr.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
