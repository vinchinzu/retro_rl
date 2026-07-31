#!/usr/bin/env python3
"""Interactive inspect / record at the natural SM→Z3 portal door.

Default state is ``PortalRedDoor``: still **Super Metroid**, Samus at the
Parlor bottom-right red door (door ``$8976``), missiles selected. Walk in
yourself — that is the natural first portal (tewtal: map-station door →
Fortune Teller cave ``$0122``). Do **not** poke Z3 module/RAM to fake control.

```bash
# Capture still-SM door state, then play
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --save-png --save-state
uv run python smz3/scripts/play_portal.py

# Rebuild door state then open window
uv run python smz3/scripts/play_portal.py --refresh

# Settled Fortune Teller overworld (after natural portal)
uv run python smz3/scripts/play_portal.py --state PortalSettled
```

Controls (windowed):
  arrows = D-pad   Z=B  X=A  A=Y  S=X   TAB=turbo   [/]=speed
  Open portal: select missiles (assist already did), face RIGHT, shoot (S/X), walk in
  F5 = quicksave   F6 = re-save named state + meta
  F9 = dump combo snapshot JSONL
  R  = reload start state
  ESC = quit (writes recording if --record / any input)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Prefer a real display for interactive play; headless only when asked.
if "SDL_VIDEODRIVER" not in os.environ and os.environ.get("HEADLESS", "").lower() not in (
    "1",
    "true",
    "yes",
):
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ["SDL_VIDEODRIVER"] = "wayland"
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")

from retro_harness.env import make_env, save_state, state_path  # noqa: E402
from retro_harness.play_session import PlaySession  # noqa: E402
from smz3.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from smz3.portal_route import (  # noqa: E402
    PORTAL_RED_DOOR_STATE,
    PORTAL_RESIDUE_STATE,
    STOP_AFTER_PORTAL,
    STOP_AT_RED_DOOR,
    run_landing_to_portal,
)
from smz3.ram import snapshot_env  # noqa: E402
from smz3.world import detect_world  # noqa: E402


def _stop_for_state(state_name: str) -> str:
    if state_name == PORTAL_RESIDUE_STATE:
        return STOP_AFTER_PORTAL
    return STOP_AT_RED_DOOR


def _refresh_state(state_name: str, *, missile_assist: bool) -> Path:
    """Drive power-on → checkpoint and write the named save state."""
    from smz3.boot import make_boot_env

    prev_driver = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = make_boot_env(render_mode="rgb_array")
    try:
        env.reset()
        stop = _stop_for_state(state_name)
        result = run_landing_to_portal(
            env,
            close=False,
            grant_missile_assist=missile_assist,
            stop=stop,
        )
        if not result.ok:
            raise SystemExit(f"drive failed ({stop}): {result.detail}")
        path = save_state(env, GAME_DIR, GAME, state_name)
        snap = snapshot_env(env, frame=result.frames)
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "state": state_name,
            "path": str(path),
            "stop": stop,
            "goal": result.goal,
            "portal_started": result.portal_started,
            "z3_settled": result.z3_settled,
            "frames": result.frames,
            "snapshot": snap.to_dict(),
            "detail": result.detail,
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "natural_portal": (
                "At red door: shoot missiles + RIGHT. Door $8976 → cave $0122. "
                "No Z3 RAM pokes after teleport."
            ),
        }
        meta_path = RECORDINGS_DIR / f"{state_name}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[refresh] ok goal={result.goal} portal_started={result.portal_started} "
            f"settled={result.z3_settled}"
        )
        print(f"[refresh] state -> {path}")
        print(f"[refresh] meta  -> {meta_path}")
        return path
    finally:
        env.close()
        if prev_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = prev_driver
        # For interactive window after dummy drive
        if os.environ.get("HEADLESS", "").lower() not in ("1", "true", "yes"):
            if os.environ.get("WAYLAND_DISPLAY"):
                os.environ["SDL_VIDEODRIVER"] = "wayland"
            else:
                os.environ["SDL_VIDEODRIVER"] = "x11"


def _write_recording(
    path: Path,
    *,
    state_name: str,
    raw_buttons: list[list[int]],
    final_snap: dict[str, Any],
    notes: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "smz3_portal_play",
        "state": state_name,
        "frames": len(raw_buttons),
        "raw_buttons": raw_buttons,
        "final_snapshot": final_snap,
        "notes": notes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "button_order": [
            "B",
            "Y",
            "SELECT",
            "START",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "X",
            "L",
            "R",
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[record] {path} ({len(raw_buttons)} frames)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state",
        default=PORTAL_RED_DOOR_STATE,
        help=(
            f"Integration state (default: {PORTAL_RED_DOOR_STATE} = still SM at door; "
            f"{PORTAL_RESIDUE_STATE} = black post-teleport hang)"
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run landing→checkpoint and overwrite the state before play",
    )
    parser.add_argument(
        "--no-missile-assist",
        action="store_true",
        help="With --refresh: do not grant missiles",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Always write a raw_buttons JSON on exit",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Explicit recording path (default: recordings/portal_play_<ts>.json)",
    )
    parser.add_argument(
        "--idle-frames",
        type=int,
        default=0,
        help="Headless: step this many idle frames then exit (implies HEADLESS)",
    )
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args(argv)

    if not (INTEGRATION_DIR / "rom.sfc").exists():
        print(
            "Missing integration ROM; run smz3/scripts/wire_integration_rom.py",
            file=sys.stderr,
        )
        return 1

    if args.idle_frames > 0:
        os.environ["HEADLESS"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    sp = state_path(GAME_DIR, GAME, args.state)
    if args.refresh or not sp.exists():
        if not sp.exists() and not args.refresh:
            print(f"[play] missing {sp}; driving route to create it…")
        _refresh_state(args.state, missile_assist=not args.no_missile_assist)
        if not state_path(GAME_DIR, GAME, args.state).exists():
            print(f"failed to create state {args.state}", file=sys.stderr)
            return 1

    env = make_env(
        game=GAME,
        state=args.state,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
    )

    recorded: list[list[int]] = []
    dump_count = 0
    recording_written = False

    if args.idle_frames > 0:
        import numpy as np
        from retro_harness.snes import idle_action

        env.reset()
        idle = idle_action(dtype=np.int8)
        for i in range(args.idle_frames):
            env.step(idle)
            recorded.append([int(x) for x in idle.tolist()])
            if i == 0 or (i + 1) % 60 == 0 or i + 1 == args.idle_frames:
                snap = snapshot_env(env, frame=i + 1)
                print(
                    f"idle {i + 1}/{args.idle_frames} "
                    f"sm=0x{snap.sm_room_id:04X} xy=({snap.sm_samus_x},{snap.sm_samus_y}) "
                    f"mod=${snap.z3_module:02X} cave=${snap.z3_room_id:04X} "
                    f"sm_ctrl={snap.sm_controllable} z3_ctrl={snap.z3_controllable}"
                )
        snap = snapshot_env(env, frame=args.idle_frames)
        rec_path = args.record_path or (
            RECORDINGS_DIR
            / f"portal_play_idle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        _write_recording(
            rec_path,
            state_name=args.state,
            raw_buttons=recorded,
            final_snap=snap.to_dict(),
            notes=f"idle_frames={args.idle_frames}",
        )
        from PIL import Image

        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        png = RECORDINGS_DIR / f"portal_play_idle_{args.state}.png"
        Image.fromarray(env.render()).save(png)
        print(f"png: {png}")
        env.close()
        return 0

    def on_hud(_info: dict) -> list[str]:
        snap = snapshot_env(env, frame=session.frame_count)
        world = detect_world(snap)
        lines = [
            f"state={args.state} world={world.value}",
            (
                f"sm room=0x{snap.sm_room_id:04X} "
                f"xy=({snap.sm_samus_x},{snap.sm_samus_y}) ctrl={snap.sm_controllable}"
            ),
            (
                f"z3 mod=${snap.z3_module:02X} sub=${snap.z3_submodule:02X} "
                f"cave=${snap.z3_room_id:04X} link=({snap.z3_link_x},{snap.z3_link_y}) "
                f"z3_ctrl={snap.z3_controllable}"
            ),
            f"rec={len(recorded)}  NATURAL: missiles + RIGHT into red door",
            "F5=quicksave F6=named-state F9=dump R=reload ESC=quit",
        ]
        if snap.sm_controllable and snap.sm_room_id == 0x92FD:
            lines.append("SM parlor — open red door, walk in (no RAM pokes)")
        if snap.z3_module == 0x0F and not snap.z3_controllable:
            lines.append("PORTAL FIRED → module $0F hang (force-blank; settle blocked)")
        if snap.z3_controllable:
            lines.append("SETTLED Link control!")
        return lines

    def on_step(_obs, _reward, _done, _info) -> None:
        action = session.last_action_post_sanitize
        recorded.append([int(x) for x in action])

    def on_key_down(key: int) -> bool:
        nonlocal dump_count
        import pygame

        if key == pygame.K_F6:
            path = save_state(env, GAME_DIR, GAME, args.state)
            snap = snapshot_env(env, frame=session.frame_count)
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            meta_path = RECORDINGS_DIR / f"{args.state}_meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "state": args.state,
                        "path": str(path),
                        "snapshot": snap.to_dict(),
                        "frame": session.frame_count,
                        "saved_from": "play_portal F6",
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"[F6] named state -> {path}")
            return True
        if key == pygame.K_F9:
            dump_count += 1
            snap = snapshot_env(env, frame=session.frame_count)
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            dump_path = RECORDINGS_DIR / "portal_play_dumps.jsonl"
            line = {
                "n": dump_count,
                "play_frame": session.frame_count,
                "snapshot": snap.to_dict(),
                "world": detect_world(snap).value,
            }
            with dump_path.open("a") as fh:
                fh.write(json.dumps(line) + "\n")
            print(
                f"[F9] dump #{dump_count} sm=0x{snap.sm_room_id:04X} "
                f"mod=${snap.z3_module:02X} cave=${snap.z3_room_id:04X} "
                f"sm_ctrl={snap.sm_controllable} z3_ctrl={snap.z3_controllable} "
                f"-> {dump_path}"
            )
            return True
        return False

    def on_reset() -> None:
        recorded.clear()
        print("[R] recording buffer cleared; reloading start state")

    def on_close() -> None:
        nonlocal recording_written
        if recording_written:
            return
        had_input = any(any(b) for b in recorded)
        if not (args.record or had_input):
            print("[record] skipped (no non-idle input; pass --record to force)")
            return
        try:
            snap = snapshot_env(env, frame=session.frame_count).to_dict()
        except Exception:
            snap = {}
        rec_path = args.record_path or (
            RECORDINGS_DIR
            / f"portal_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        _write_recording(
            rec_path,
            state_name=args.state,
            raw_buttons=recorded,
            final_snap=snap,
            notes="interactive natural portal play",
        )
        recording_written = True

    session = PlaySession(
        env,
        game_dir=str(GAME_DIR),
        game=GAME,
        scale=args.scale,
        title=f"SMZ3 portal — {args.state}",
    )
    session.on_hud = on_hud
    session.on_step = on_step
    session.on_key_down = on_key_down
    session.on_reset = on_reset
    session.on_close = on_close

    print(f"Playing state: {args.state} ({sp})")
    if args.state == PORTAL_RED_DOOR_STATE:
        print(
            "Still SM at Parlor red door. Natural portal: missiles + RIGHT into door."
        )
        print("Destination (combo table): Fortune Teller cave $0122 → OW $35.")
    elif args.state == "PortalSettled":
        print("Settled Z3 overworld at Fortune Teller (screen $35). Walk with D-pad.")
    else:
        print("Post-teleport residue mid-handoff; prefer PortalSettled or PortalRedDoor.")
    print(
        "Focus this window: F9=dump RAM; ESC or Q quits (saves button log if you moved)."
    )
    print("Terminal ESC is ignored — use the game window, or Ctrl+C.")
    from smz3.paths import SHARED_Z3_JP_ROM

    if not SHARED_Z3_JP_ROM.is_file():
        print(
            "WARN: missing roms/zelda3_jp.sfc (ALttP JP 1.0). "
            "Combo built with USA Zelda hangs black after the portal.",
            file=sys.stderr,
        )
    session.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
