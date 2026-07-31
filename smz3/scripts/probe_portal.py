"""Probe: power-on → Parlor red door (default) or through natural portal.

Default stops **before** teleport so the frame is still Super Metroid (visible).
Use ``--through-portal`` to walk in and wait for Z3 settle (module ``$09`` OW).

```bash
# Preferred: still SM at red door + save state for interactive play
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --save-png --save-state
uv run python smz3/scripts/play_portal.py

# Through portal → Fortune Teller OW (needs ALttP JP 1.0 combo)
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --through-portal --save-png --save-state
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import save_state  # noqa: E402
from smz3.boot import make_boot_env  # noqa: E402
from smz3.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from smz3.portal_route import (  # noqa: E402
    PORTAL_RED_DOOR_STATE,
    PORTAL_RESIDUE_STATE,
    PORTAL_SETTLED_STATE,
    STOP_AFTER_PORTAL,
    STOP_AT_RED_DOOR,
    run_landing_to_portal,
)
from smz3.portals import early_portal, room_name  # noqa: E402
from smz3.ram import snapshot_env  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument(
        "--through-portal",
        action="store_true",
        help=(
            "Walk into the red door (natural SM→Z3 teleport). "
            "Default is stop at the door still in SM (playable)."
        ),
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Write PortalRedDoor.state (default stop) or PortalResidue.state",
    )
    parser.add_argument(
        "--state-name",
        default=None,
        help="Override save-state base name",
    )
    parser.add_argument(
        "--no-missile-assist",
        action="store_true",
        help="Do not grant missiles (red door will not open without ammo)",
    )
    args = parser.parse_args(argv)

    stop = STOP_AFTER_PORTAL if args.through_portal else STOP_AT_RED_DOOR
    # Name resolved after run when through-portal (Settled vs Residue).
    state_name = args.state_name
    save_state_flag = args.save_state or args.save_png

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if not (INTEGRATION_DIR / "rom.sfc").exists():
        print(
            "Missing integration ROM; run smz3/scripts/wire_integration_rom.py",
            file=sys.stderr,
        )
        return 1

    env = make_boot_env(render_mode="rgb_array")
    try:
        env.reset()
        result = run_landing_to_portal(
            env,
            close=False,
            grant_missile_assist=not args.no_missile_assist,
            stop=stop,
        )
        payload = result.to_dict()

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"ok: {result.ok}")
            print(f"goal: {result.goal}")
            print(f"stop: {stop}")
            print(f"portal_started: {result.portal_started}")
            print(f"z3_settled: {result.z3_settled}")
            print(f"frames: {result.frames} (boot {result.boot_frames})")
            print(f"world: {result.world.value}")
            print(f"detail: {result.detail}")
            snap = result.final_snapshot
            if snap is not None:
                print(
                    f"sm: room=0x{snap.sm_room_id:04X} "
                    f"xy=({snap.sm_samus_x},{snap.sm_samus_y}) "
                    f"ctrl={snap.sm_controllable}"
                )
            if result.z3_module is not None:
                print(
                    f"z3: module=${result.z3_module:02X} "
                    f"room=${result.z3_room_id:04X}"
                    if result.z3_room_id is not None
                    else f"z3: module=${result.z3_module:02X}"
                )
            print("visits:")
            for v in result.visits:
                print(
                    f"  0x{v.room_id:04X} {room_name(v.room_id):24s} "
                    f"enter={v.enter_frame} leave={v.leave_frame} "
                    f"dwell={v.dwell_frames}"
                )
            p = early_portal()
            print(
                f"portal catalog: door {p.sm_door_ptr:#06x} → {p.z3_name} "
                f"(cave {p.z3_cave_id:#06x})"
            )
            if stop == STOP_AT_RED_DOOR and result.ok:
                print(
                    "note: still SM at Parlor red door — shoot missiles + RIGHT "
                    "to take the natural portal (do not poke Z3 RAM)"
                )
            if result.z3_settled:
                print(
                    "note: Z3 settled — Fortune Teller OW (natural portal complete)"
                )
            elif result.portal_started and not result.z3_settled:
                print(
                    "note: portal residue only — Link not settled "
                    "(see docs/EARLY_ROOMS.md; needs JP 1.0 + longer settle wait)"
                )

        if state_name is None:
            if stop == STOP_AT_RED_DOOR:
                state_name = PORTAL_RED_DOOR_STATE
            elif result.z3_settled:
                state_name = PORTAL_SETTLED_STATE
            else:
                state_name = PORTAL_RESIDUE_STATE

        if args.save_png:
            from PIL import Image

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            if stop == STOP_AT_RED_DOOR:
                png_name = "m3_portal_red_door.png"
            elif result.z3_settled:
                png_name = "m3_portal_settled.png"
            else:
                png_name = "m3_landing_to_portal.png"
            path = RECORDINGS_DIR / png_name
            Image.fromarray(env.render()).save(path)
            print(f"png: {path}")

        if save_state_flag and result.ok:
            state_path = save_state(env, GAME_DIR, GAME, state_name)
            snap = snapshot_env(env, frame=result.frames)
            meta = {
                "state": state_name,
                "path": str(state_path),
                "stop": stop,
                "goal": result.goal,
                "portal_started": result.portal_started,
                "z3_settled": result.z3_settled,
                "frames": result.frames,
                "snapshot": snap.to_dict(),
                "detail": result.detail,
                "natural_portal": (
                    "Shoot missiles (X) while holding RIGHT into the red door. "
                    "Combo remaps door $8976 → Fortune Teller cave $0122. "
                    "No post-teleport RAM pokes."
                ),
                "play": f"uv run python smz3/scripts/play_portal.py --state {state_name}",
            }
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            meta_path = RECORDINGS_DIR / f"{state_name}_meta.json"
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"state: {state_path}")
            print(f"meta: {meta_path}")
            print(
                f"play: uv run python smz3/scripts/play_portal.py --state {state_name}"
            )
        elif save_state_flag and not result.ok:
            print("state: skipped (route not ok)", file=sys.stderr)

        return 0 if result.ok else 2
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
