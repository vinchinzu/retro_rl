"""Probe: Fortune Teller OW → Link's House (no sword) + optional MP4.

Loads ``PortalSettled`` (or re-runs portal settle), drives outdoor nav that
avoids combat by fleeing enemies, and optionally records video proof.

```bash
# From settled Z3 overworld checkpoint
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_outdoor.py --video --save-png

# Refresh PortalSettled then outdoor leg
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_outdoor.py --through-portal --video
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
from smz3.outdoor_route import (  # noqa: E402
    FORTUNE_TELLER_SCREEN,
    LINKS_HOUSE_OW_SCREEN,
    run_fortune_teller_to_links_house,
)
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

DEFAULT_VIDEO = RECORDINGS_DIR / "fortune_to_links_house.mp4"
DEFAULT_PNG = RECORDINGS_DIR / "m3_links_house_ow.png"
DEFAULT_JSON = RECORDINGS_DIR / "fortune_to_links_house.json"


def _make_settled_env(*, state: str, render_mode: str = "rgb_array") -> Any:
    from retro_harness.env import make_env

    return make_env(INTEGRATION, state, GAME_DIR, render_mode=render_mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        default=PORTAL_SETTLED_STATE,
        help=f"Save state to load (default: {PORTAL_SETTLED_STATE})",
    )
    parser.add_argument(
        "--through-portal",
        action="store_true",
        help="Power-on → portal settle instead of loading a save state",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON result")
    parser.add_argument(
        "--save-png",
        action="store_true",
        help=f"Write final frame PNG (default path: {DEFAULT_PNG})",
    )
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG, help="PNG output path")
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
    outdoor_frames_before = 0

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
                    "detail": "portal did not settle; outdoor skipped",
                }
                args.report.write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps(payload, indent=2))
                return 2
            outdoor_frames_before = portal.frames
        else:
            env = wrap_recording(_make_settled_env(state=args.state), writer)
            env.reset()

        result = run_fortune_teller_to_links_house(env, start_frame=0)
        snap = result.final_snapshot or snapshot_env(env)

        report = {
            **result.to_dict(),
            "state": None if args.through_portal else args.state,
            "through_portal": args.through_portal,
            "outdoor_start_frame": outdoor_frames_before,
            "target_screen": f"0x{LINKS_HOUSE_OW_SCREEN:02X}",
            "fortune_teller_screen": f"0x{FORTUNE_TELLER_SCREEN:02X}",
            "video": str(video_path) if video_path is not None else None,
            "no_sword": True,
            "strategy": "corridor_nav + flee_hostiles",
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"ok: {result.ok}")
            print(f"detail: {result.detail}")
            print(f"frames: {result.frames}")
            print(f"screens: {report['screens_visited']}")
            print(f"fled_frames: {result.fled_frames}")
            print(
                f"final: screen=${snap.z3_screen_id:02X} "
                f"xy=({snap.z3_link_x},{snap.z3_link_y}) "
                f"ctrl={snap.z3_controllable}"
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
        return 0 if result.ok else 2
    finally:
        if writer is not None:
            video_frames = writer.frames_written
            writer.close()
            if video_path is not None:
                print(f"video_frames: {video_frames}")
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
