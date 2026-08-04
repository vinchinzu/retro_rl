"""M2 probe: power-on → Landing Site → Parlor with room timeout.

  SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py
  SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py --save-png --json
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

from smz3.boot import make_boot_env  # noqa: E402
from smz3.early_route import run_landing_to_parlor  # noqa: E402
from smz3.paths import INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from smz3.portals import early_portal, portals_to_dict  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--list-portals", action="store_true")
    args = parser.parse_args(argv)

    if args.list_portals:
        print(json.dumps(portals_to_dict(), indent=2))
        return 0

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
        result = run_landing_to_parlor(env, close=False)
        payload = result.to_dict()
        payload["early_portal"] = early_portal().to_dict()

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"ok: {result.ok}")
            print(f"frames: {result.frames} (boot {result.boot_frames})")
            print(f"world: {result.world.value}")
            print(f"detail: {result.detail}")
            print("visits:")
            for v in result.visits:
                print(
                    f"  0x{v.room_id:04X} {room_label(v.room_id):24s} "
                    f"enter={v.enter_frame} leave={v.leave_frame} "
                    f"dwell={v.dwell_frames}"
                )
            p = early_portal()
            print(
                f"next portal: {p.sm_name} door {p.sm_door_ptr:#06x} → "
                f"{p.z3_name} (needs missiles + red door from Parlor)"
            )

        if args.save_png and result.final_snapshot is not None:
            from PIL import Image

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            path = RECORDINGS_DIR / "m2_landing_to_parlor.png"
            Image.fromarray(env.render()).save(path)
            print(f"png: {path}")

        return 0 if result.ok else 2
    finally:
        env.close()


def room_label(room_id: int) -> str:
    from smz3.portals import room_name

    return room_name(room_id)


if __name__ == "__main__":
    raise SystemExit(main())
