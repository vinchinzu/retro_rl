"""Headless power-on → FirstPlay.state for SM Rando (vanilla ROM).

```bash
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.make_boot
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.make_boot --json
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Abort after this many emulator frames",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip recordings/boot_first_play.png snapshot",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print BootResult as JSON",
    )
    parser.add_argument(
        "--setup-rom",
        action="store_true",
        help="Run setup_rom first if rom.sfc is missing",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from sm_rando.boot import create_first_play_state
    from sm_rando.paths import INTEGRATION_DIR

    link = INTEGRATION_DIR / "rom.sfc"
    if not link.exists():
        if args.setup_rom:
            from sm_rando.scripts.setup_rom import setup_rom

            setup_rom()
        else:
            print(
                f"Missing {link}; run: uv run python -m sm_rando.scripts.setup_rom",
                file=sys.stderr,
            )
            return 1

    kwargs: dict = {"save_png": not args.no_png}
    if args.max_frames is not None:
        kwargs["max_frames"] = args.max_frames

    result = create_first_play_state(**kwargs)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"ok: {result.ok}")
        print(f"frames: {result.frames}")
        print(f"game_state: {result.game_state}")
        print(f"room_id: 0x{result.room_id:04X}")
        print(f"detail: {result.detail}")
        if result.state_path:
            print(f"state: {result.state_path}")

    return 0 if result.ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
