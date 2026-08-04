"""Probe: Link's House OW → enter door → open chest (+ optional video).

Loads ``PortalSettled``, runs outdoor Fortune Teller → $2C, then
``house_route`` map path into the house and opens the chest.

```bash
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_house.py --video --save-png
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smz3.boot import make_boot_env  # noqa: E402
from smz3.house_route import run_links_house_chest  # noqa: E402
from smz3.outdoor_route import run_fortune_teller_to_links_house  # noqa: E402
from smz3.paths import GAME_DIR, INTEGRATION, INTEGRATION_DIR, RECORDINGS_DIR  # noqa: E402
from smz3.portal_route import (  # noqa: E402
    PORTAL_SETTLED_STATE,
    STOP_AFTER_PORTAL,
    run_landing_to_portal,
)
from smz3.ram import snapshot_env  # noqa: E402
from smz3.recording import (  # noqa: E402
    RecordingEnv,
    configure_headless,
    probe_frame_size,
    wrap_recording,
)
from retro_harness.video import FrameVideoWriter  # noqa: E402

DEFAULT_VIDEO = RECORDINGS_DIR / "links_house_chest.mp4"
DEFAULT_PNG = RECORDINGS_DIR / "m3_links_house_chest.png"
DEFAULT_JSON = RECORDINGS_DIR / "links_house_chest.json"


def _make_settled_env(*, state: str, render_mode: str = "rgb_array") -> Any:
    from retro_harness.env import make_env

    return make_env(INTEGRATION, state, GAME_DIR, render_mode=render_mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=PORTAL_SETTLED_STATE)
    parser.add_argument(
        "--through-portal",
        action="store_true",
        help="Power-on → portal settle instead of loading a save state",
    )
    parser.add_argument(
        "--skip-outdoor",
        action="store_true",
        help="Assume already on Link's House OW (or indoors); skip outdoor leg",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--video",
        nargs="?",
        const=str(DEFAULT_VIDEO),
        default=None,
        help=f"Record MP4 (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--report", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args(argv)
    configure_headless()

    if not (INTEGRATION_DIR / "rom.sfc").exists():
        print(
            "Missing integration ROM; run smz3/scripts/wire_integration_rom.py",
            file=sys.stderr,
        )
        return 1

    state_path = INTEGRATION_DIR / f"{args.state}.state"
    if not args.through_portal and not state_path.exists():
        print(
            f"Missing {state_path}; run probe_portal.py --through-portal --save-state "
            "or pass --through-portal",
            file=sys.stderr,
        )
        return 1

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    writer: FrameVideoWriter | None = None
    env: Any = None
    video_path: Path | None = None

    try:
        if args.video is not None:
            video_path = Path(args.video)
            probe = (
                make_boot_env(render_mode="rgb_array")
                if args.through_portal
                else _make_settled_env(state=args.state)
            )
            try:
                height, width = probe_frame_size(probe)
            finally:
                probe.close()
            writer = FrameVideoWriter(
                video_path,
                width=width,
                height=height,
                fps=args.fps,
                scale=args.scale,
            )

        if args.through_portal:
            env = wrap_recording(make_boot_env(render_mode="rgb_array"), writer)
            env.reset()
            portal = run_landing_to_portal(env, close=False, stop=STOP_AFTER_PORTAL)
            if not portal.z3_settled:
                payload = {
                    "ok": False,
                    "phase": "portal",
                    "portal": portal.to_dict(),
                }
                args.report.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps(payload, indent=2))
                return 2
        else:
            env = wrap_recording(_make_settled_env(state=args.state), writer)
            env.reset()

        outdoor_report: dict[str, Any] | None = None
        if not args.skip_outdoor:
            outdoor = run_fortune_teller_to_links_house(env, start_frame=0)
            outdoor_report = outdoor.to_dict()
            if not outdoor.ok:
                payload = {
                    "ok": False,
                    "phase": "outdoor",
                    "outdoor": outdoor_report,
                }
                args.report.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps(payload, indent=2))
                return 2

        house = run_links_house_chest(env, start_frame=0)
        snap = house.final_snapshot or snapshot_env(env)

        report: dict[str, Any] = {
            **house.to_dict(),
            "state": None if args.through_portal else args.state,
            "through_portal": args.through_portal,
            "outdoor": outdoor_report,
            "video": str(video_path) if video_path is not None else None,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"ok: {house.ok}")
            print(f"detail: {house.detail}")
            print(f"entered: {house.entered} chest: {house.chest_opened}")
            print(
                f"max_hp: {house.max_hp_before} -> {house.max_hp_after} "
                f"lamp={house.lamp} delta={house.inventory_delta}"
            )
            print(f"frames: {house.frames}")
            print(
                f"final: indoors={snap.z3_indoors} room=${snap.z3_room_id:04X} "
                f"xy=({snap.z3_link_x},{snap.z3_link_y})"
            )
            if video_path is not None:
                print(f"video: {video_path}")

        if args.save_png or args.video is not None:
            from PIL import Image

            obs = None
            if isinstance(env, RecordingEnv) and env.last_obs is not None:
                obs = env.last_obs
            else:
                obs = env.render()
            if obs is not None:
                args.png.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(obs)).save(args.png)
                report["png"] = str(args.png)
                print(f"png: {args.png}")

        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report: {args.report}")
        return 0 if house.ok else 2
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
