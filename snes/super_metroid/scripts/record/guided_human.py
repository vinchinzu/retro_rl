#!/usr/bin/env python3
"""Human path recorder with on-screen route guide (Cathedral spine default).

Loads a pure/continuous scratch source, draws the KPDR guide polyline in the
current room (camera-projected), and records human inputs + position trace to
``super_metroid/tasks/<name>.json`` for faster pure-controller iteration.

Harvest-style task JSON lives in shared :mod:`retro_harness.task_recording`;
the guide overlay uses :mod:`retro_harness.path_overlay`.

```bash
# Cathedral left lip → Rising Tide → Bubble → Bat (default)
uv run python snes/super_metroid/scripts/record/guided_human.py

# Named task + route preset
uv run python snes/super_metroid/scripts/record/guided_human.py \\
  --name cathedral_to_bat_v1 --route cathedral-to-bat

# Start at Bubble (CATH-04 pure source)
uv run python snes/super_metroid/scripts/record/guided_human.py \\
  --from bubble --route bubble-to-bat --name bubble_human

# Post-Torizo Parlor — Flyway door → Alcatraz LEFT wall-jump shaft (guide on)
uv run python snes/super_metroid/scripts/record/guided_human.py \\
  --from parlor --route parlor-left --name parlor_left_human

# List start presets / routes
uv run python snes/super_metroid/scripts/record/guided_human.py --list
```

Controls (PlaySession defaults + recording):
  F5 / F1  Save recording + end state, exit
  ESC / Q  Cancel without saving
  [ ]      Speed · TAB turbo · F1–F4 checkpoints
  Unlimited energy/ammo assists on by default (human practice only)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, globals().get('_SNES_IMPORT_ROOT', ROOT)):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from retro_harness.actions import idle_action  # noqa: E402
from retro_harness.env import make_env, read_state_bytes  # noqa: E402
from retro_harness.path_overlay import (  # noqa: E402
    draw_guide_path,
    draw_player_marker,
    nearest_waypoint_index,
    transform_from_session_ctx,
)
from retro_harness.play_session import PlaySession  # noqa: E402
from retro_harness.runtime import step_env  # noqa: E402
from retro_harness.task_recording import (  # noqa: E402
    RecordedTask,
    pressed_buttons,
    summarize_position_trace,
)
from super_metroid.assist import UnlimitedResourcesAssist  # noqa: E402
from super_metroid.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from super_metroid.ram import parse_env_state  # noqa: E402
from super_metroid.routes.kpdr.guide_paths import (  # noqa: E402
    ROUTE_PRESETS,
    guide_for_room,
)

# Layer 1 camera scroll (WRAM) — same as place_samus in dev/common.py.
ADDR_CAMERA_X = 0x0911
ADDR_CAMERA_Y = 0x0915

SCRATCH = INTEGRATION_DIR / "scratch"
TASKS_DIR = GAME_DIR / "tasks"

# Start presets: short name → relative state under SuperMetroid-Snes/
START_PRESETS: dict[str, tuple[str, str]] = {
    "cathedral": (
        "scratch/post_cathedral_entrance_to_cathedral_pure.state",
        "Cathedral left lip (CATH-02 pure successor)",
    ),
    "cathedral-entrance": (
        "scratch/post_business_to_cathedral_entrance_pure.state",
        "Cathedral Entrance left lip (CATH-01 pure successor)",
    ),
    "rising-tide": (
        "scratch/post_cathedral_to_rising_tide_pure.state",
        "Rising Tide left entry (CATH-03 pure successor)",
    ),
    "bubble": (
        "scratch/post_rising_tide_to_bubble_pure.state",
        "Bubble Mountain entry (CATH-04 pure source)",
    ),
    "business": (
        "scratch/post_business_continuous.state",
        "Business Center continuous tip",
    ),
    "parlor": (
        "scratch/post_torizo_parlor_continuous.state",
        "Post-Bomb-Torizo Parlor at Flyway door (~968,651) — left climb demo",
    ),
    "post-torizo": (
        "scratch/post_torizo_parlor_continuous.state",
        "Alias of parlor (post-BT Flyway door pin)",
    ),
}


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _resolve_state(arg: str) -> Path:
    if arg in START_PRESETS:
        rel, _ = START_PRESETS[arg]
        return INTEGRATION_DIR / rel
    path = Path(arg)
    if path.is_file():
        return path.resolve()
    # Allow stem under scratch/ or integration root.
    candidates = [
        SCRATCH / f"{arg}.state",
        SCRATCH / arg,
        INTEGRATION_DIR / f"{arg}.state",
        INTEGRATION_DIR / arg,
        GAME_DIR / arg,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Start state not found: {arg}")


def _trace_row(env, frame: int, action) -> dict[str, object]:
    state = parse_env_state(env, frame=frame, mode="nav")
    return {
        "frame": frame,
        "x": int(state.samus_x),
        "y": int(state.samus_y),
        "room": int(state.room_id),
        "room_hex": f"0x{int(state.room_id):04X}",
        "pose": int(state.pose),
        "vx": int(state.velocity_x),
        "vy": int(state.velocity_y),
        "buttons": pressed_buttons(action),
        "energy": int(state.health),
        "missiles": int(state.missiles),
        "supers": int(state.super_missiles),
        "pbs": int(state.power_bombs),
        "selected": int(state.selected_item),
        "door_transition": int(state.door_transition),
        "phase": state.phase.value if hasattr(state.phase, "value") else str(state.phase),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from",
        dest="start",
        default="cathedral",
        help=(
            "Start preset name or path/stem "
            f"(presets: {', '.join(START_PRESETS)}; default: cathedral)"
        ),
    )
    parser.add_argument(
        "--route",
        default=None,
        choices=sorted(ROUTE_PRESETS),
        help=(
            "Guide route preset (waypoints drawn per room). "
            "Default: parlor-left when --from parlor/post-torizo, "
            "else cathedral-to-bat"
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Task name under super_metroid/tasks/ (default: guided_<route>_<ts>)",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--no-assist",
        action="store_true",
        help="Disable unlimited energy/ammo (harder practice)",
    )
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Record without drawing the route line",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List start presets and routes, then exit",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=TASKS_DIR,
        help=f"Task JSON directory (default: {TASKS_DIR})",
    )
    args = parser.parse_args()

    if args.list:
        print("Start presets:")
        for key, (rel, desc) in START_PRESETS.items():
            path = INTEGRATION_DIR / rel
            mark = "OK" if path.is_file() else "MISSING"
            print(f"  {key:22s} [{mark}] {desc}")
            print(f"    {path}")
        print("\nRoute presets:")
        for key, guides in sorted(ROUTE_PRESETS.items()):
            rooms = " → ".join(g.name or f"0x{g.room_id:04X}" for g in guides)
            print(f"  {key:22s} {rooms}")
        return 0

    # Sensible default route from start pin (parlor → Alcatraz left WJ guide).
    if args.route is None:
        if args.start in ("parlor", "post-torizo"):
            args.route = "parlor-left"
        else:
            args.route = "cathedral-to-bat"

    try:
        state_path = _resolve_state(args.start)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not state_path.is_file():
        print(f"ERROR: state missing: {state_path}", file=sys.stderr)
        return 1

    task_name = args.name or f"guided_{args.route}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_path = out_dir / f"{task_name}.json"
    end_state_paths = [
        out_dir / f"{task_name}_end.state",
        SCRATCH / f"{task_name}_end.state",
    ]

    route_guides = ROUTE_PRESETS[args.route]
    route_room_ids = {g.room_id for g in route_guides}

    state_bytes = read_state_bytes(state_path)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")

    assist = UnlimitedResourcesAssist(
        unlimited_energy=not args.no_assist,
        unlimited_ammo=not args.no_assist,
    )
    task = RecordedTask(
        name=task_name,
        start_state=str(state_path.relative_to(INTEGRATION_DIR))
        if state_path.is_relative_to(INTEGRATION_DIR)
        else str(state_path),
    )
    task.metadata["route"] = args.route
    task.metadata["guide_rooms"] = [
        {"room_id": f"0x{g.room_id:04X}", "name": g.name, "points": len(g.points)}
        for g in route_guides
    ]
    task.metadata["source_path"] = str(state_path)

    live: dict[str, object] = {
        "room": 0,
        "x": 0,
        "y": 0,
        "guide_name": "",
        "nearest": None,
        "cam_x": 0,
        "cam_y": 0,
    }
    saved = {"ok": False}

    def on_step(obs, reward, done, info) -> None:
        del obs, reward, done, info
        action = session.last_action_post_sanitize
        frame = session.frame_count
        # Assist after human step (same pattern as pure probe sessions).
        st = parse_env_state(env, frame=frame, mode="nav")
        assist.apply(env.data, st)
        row = _trace_row(env, frame - 1 if frame > 0 else 0, action)
        task.append_frame(action, trace_row=row)
        live["room"] = row["room"]
        live["x"] = row["x"]
        live["y"] = row["y"]
        ram = env.get_ram()
        live["cam_x"] = _u16(ram, ADDR_CAMERA_X)
        live["cam_y"] = _u16(ram, ADDR_CAMERA_Y)
        guide = guide_for_room(int(row["room"]))
        if guide is not None and guide.room_id in route_room_ids:
            live["guide_name"] = guide.name
            live["nearest"] = nearest_waypoint_index(guide.points, int(row["x"]), int(row["y"]))
        else:
            live["guide_name"] = ""
            live["nearest"] = None

    def on_hud(info) -> list[str]:
        del info
        room = int(live["room"] or 0)
        lines = [
            f"[REC] {task_name}  F5=save  ESC=cancel",
            f"room=0x{room:04X}  xy=({live['x']},{live['y']})  frames={len(task.frames)}",
        ]
        if live["guide_name"]:
            lines.append(f"guide: {live['guide_name']}  wp={live['nearest']}")
        else:
            lines.append("guide: (no polyline for this room)")
        return lines

    def on_overlay(pg, ctx) -> None:
        if args.no_guide:
            return
        room = int(live["room"] or 0)
        guide = guide_for_room(room)
        if guide is None or guide.room_id not in route_room_ids:
            return
        transform = transform_from_session_ctx(
            ctx,
            camera_x=int(live["cam_x"] or 0),
            camera_y=int(live["cam_y"] or 0),
        )
        surface = ctx.get("screen")
        font = ctx.get("font")
        if surface is None:
            return
        draw_guide_path(
            pg,
            surface,
            guide.points,
            transform,
            color=guide.color,
            width=2,
            radius=5,
            highlight_index=live["nearest"] if isinstance(live["nearest"], int) else 0,
            font=font,
            draw_labels=True,
        )
        draw_player_marker(
            pg,
            surface,
            int(live["x"] or 0),
            int(live["y"] or 0),
            transform,
        )

    def on_key_down(key: int) -> bool:
        # F5/F1: finalize recording (PlaySession F5 normally only quicksaves).
        import pygame

        if key in (pygame.K_F5, pygame.K_F1):
            _finalize(save=True)
            session.running = False
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            print("[REC] cancelled")
            session.running = False
            return True
        return False

    def _finalize(*, save: bool) -> None:
        if not save or saved["ok"]:
            return
        if not task.frames:
            print("[REC] nothing recorded")
            return
        try:
            task.end_state_data = env.em.get_state()
        except Exception as exc:
            print(f"[REC] end-state capture failed: {exc}")
        task.metadata.update(
            summarize_position_trace(frames=task.frames, trace=task.trace, room_key="room")
        )
        task.metadata["assist"] = {
            "unlimited_energy": not args.no_assist,
            "unlimited_ammo": not args.no_assist,
            "telemetry": assist.telemetry.to_dict() if hasattr(assist, "telemetry") else {},
        }
        task.recorded_at = datetime.now().isoformat()
        task.save(task_path, end_state_paths=end_state_paths)
        # Mirror a lightweight pointer under recordings/ for discoverability.
        rec_ptr = RECORDINGS_DIR / "human_tasks" / f"{task_name}.json"
        try:
            if task_path.resolve() != rec_ptr.resolve():
                rec_ptr.parent.mkdir(parents=True, exist_ok=True)
                if not rec_ptr.exists():
                    rec_ptr.symlink_to(task_path.resolve())
        except OSError:
            pass
        saved["ok"] = True
        print(f"[REC] saved {task_path} ({len(task.frames)} frames)")
        for p in end_state_paths:
            print(f"[REC] end state → {p}")

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME,
        scale=args.scale,
        title=f"Guided REC: {task_name} [{args.route}]",
        bot=None,
        action_size=12,
        base_fps=60,
    )
    session.on_step = on_step
    session.on_hud = on_hud
    session.on_overlay = on_overlay
    session.on_key_down = on_key_down
    session.on_close = lambda: None

    def _seed_live_from_env() -> None:
        boot = parse_env_state(env, mode="nav")
        live["room"] = int(boot.room_id)
        live["x"] = int(boot.samus_x)
        live["y"] = int(boot.samus_y)
        ram0 = env.get_ram()
        live["cam_x"] = _u16(ram0, ADDR_CAMERA_X)
        live["cam_y"] = _u16(ram0, ADDR_CAMERA_Y)
        g0 = guide_for_room(int(boot.room_id))
        if g0 is not None and g0.room_id in route_room_ids:
            live["guide_name"] = g0.name
            live["nearest"] = nearest_waypoint_index(
                g0.points, int(boot.samus_x), int(boot.samus_y)
            )
        else:
            live["guide_name"] = ""
            live["nearest"] = None

    # PlaySession.run() always env.reset() first — re-inject cathedral source
    # after that reset so we actually start on the chosen pure state.
    import retro_harness.play_session as _ps_mod

    _orig_reset = _ps_mod.reset_env

    def _reset_then_boot(e):
        obs, info = _orig_reset(e)
        e.em.set_state(state_bytes)
        for _ in range(8):
            obs, _r, _t, _tr, info = step_env(e, idle_action())
        _seed_live_from_env()
        print(
            f"[BOOT] room=0x{int(live['room']):04X} "
            f"xy=({live['x']},{live['y']}) from {state_path.name}"
        )
        return obs, info

    print("=" * 60)
    print(f"GUIDED HUMAN RECORD  route={args.route}")
    print(f"  start: {state_path}")
    print(f"  task:  {task_path}")
    print(f"  guide: {'ON' if not args.no_guide else 'OFF'}  assist={'OFF' if args.no_assist else 'ON'}")
    print("  F5/F1 = save recording · ESC/Q = cancel")
    print("=" * 60)

    _ps_mod.reset_env = _reset_then_boot
    try:
        session.run()
    finally:
        _ps_mod.reset_env = _orig_reset
        # session.run already closes env; ignore double-close

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
