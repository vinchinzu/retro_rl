"""Power-on → first controllable SM frame on the SMZ3 combo ROM.

  SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_boot.py
  SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_boot.py --save-png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from smz3.boot import boot_to_controllable, make_boot_env  # noqa: E402
from smz3.paths import INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from smz3.world import detect_world  # noqa: E402

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=3600,
        help="Abort after this many emulator frames",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Write recordings/boot_controllable.png on success",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print BootResult as JSON",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    link = INTEGRATION_DIR / "rom.sfc"
    if not link.exists():
        print(
            f"Missing {link}; run: uv run python smz3/scripts/wire_integration_rom.py",
            file=sys.stderr,
        )
        return 1

    env = make_boot_env(render_mode="rgb_array")
    try:
        env.reset()
        result = boot_to_controllable(env, max_frames=args.max_frames, close=False)
        world = detect_world(result.snapshot)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"ok: {result.ok}")
            print(f"frames: {result.frames}")
            print(f"world: {world.value}")
            print(f"detail: {result.detail}")
            snap = result.snapshot
            print(
                f"sm: gs={snap.sm_game_state} room=0x{snap.sm_room_id:04X} "
                f"area={snap.sm_area_index} hp={snap.sm_health} "
                f"xy=({snap.sm_samus_x},{snap.sm_samus_y})"
            )
            print(
                f"z3: module=0x{snap.z3_module:02X} sub=0x{snap.z3_submodule:02X} "
                f"(ignored while SM owns WRAM)"
            )

        if result.ok and args.save_png:
            from PIL import Image

            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            img = env.render()
            path = RECORDINGS_DIR / "boot_controllable.png"
            Image.fromarray(img).save(path)
            print(f"png: {path}")

        return 0 if result.ok else 2
    finally:
        env.close()

if __name__ == "__main__":
    raise SystemExit(main())
