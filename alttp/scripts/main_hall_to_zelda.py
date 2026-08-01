"""CLI: main hall (room 0x61) west exit / Zelda path (dev state default).

Thin wrapper — prefer ``alttp/scripts/room_engine.py`` for new rooms::

  uv run python alttp/scripts/room_engine.py show room_61
  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \\
      --edge west_to_0x60 --state CastleMain --overlay

Legacy::

  SDL_VIDEODRIVER=dummy uv run python alttp/scripts/main_hall_to_zelda.py --overlay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from alttp.opening_route.main_hall_to_zelda import run_from_main_hall  # noqa: E402
from alttp.paths import RECORDINGS_DIR  # noqa: E402
from alttp.room_map import load_room_map  # noqa: E402
from alttp.room_sense import overlay_from_env  # noqa: E402
from alttp.startup import build_boot_env  # noqa: E402
from alttp.primitives import settle_control  # noqa: E402


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main(argv: list[str] | None = None) -> int:
    _configure_headless()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="CastleMain", help="Save state name")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=RECORDINGS_DIR / "main_hall_to_zelda.json",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=RECORDINGS_DIR / "main_hall_to_zelda.png",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Write start/end overlays with enemy sprite boxes",
    )
    args = parser.parse_args(argv)

    out_dir = RECORDINGS_DIR / "probe_main_hall"
    out_dir.mkdir(parents=True, exist_ok=True)
    room_map = load_room_map("room_61")

    env = build_boot_env(args.state)
    try:
        env.reset()  # type: ignore[attr-defined]
        settle_control(env)
        if args.overlay:
            img = overlay_from_env(
                env,
                include_enemies=True,
                points=room_map.points,
            )
            overlay_path = out_dir / f"{args.state}_overlay_start.png"
            Image.fromarray(img).save(overlay_path)
            print(f"Wrote {overlay_path}")

        result = run_from_main_hall(env, source="state_load_dev")
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(env.render())).save(args.screenshot)  # type: ignore[attr-defined]
        print(f"Wrote {args.screenshot}")
        if args.overlay:
            img = overlay_from_env(
                env,
                include_enemies=True,
                points=room_map.points,
            )
            end_path = out_dir / f"{args.state}_overlay_end.png"
            Image.fromarray(img).save(end_path)
            print(f"Wrote {end_path}")
    finally:
        env.close()  # type: ignore[attr-defined]

    report = result.to_report("main_hall_to_zelda")
    report["cli"] = {"state": args.state}
    report["roomMap"] = room_map.compact_summary()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.json_out}")
    print(
        f"ok={result.ok} phase={result.phase} frames={result.frames} "
        f"blocker={result.blocker!r}"
    )
    print(f"acceptance={result.acceptance}")

    if result.phase == "left_main_hall_west":
        return 0
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
