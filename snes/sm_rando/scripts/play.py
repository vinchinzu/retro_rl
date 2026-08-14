#!/usr/bin/env python3
"""Interactive SM Rando play — FirstPlay + record by default.

Controls (windowed):
  arrows = D-pad   Z=B  X=A  A=Y  S=X   TAB=turbo   [/]=speed
  F5 = quicksave into snes/sm_rando/custom_integrations/SMRando-Snes/
  R = reload start state   ESC = quit

```bash
./play
uv run python -m sm_rando.scripts.play
uv run python -m sm_rando.scripts.play --no-record
uv run python -m sm_rando.scripts.play --rebuild-boot
HEADLESS=1 SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.play --max-frames 30
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _configure_display(*, headless: bool) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")
        os.environ.setdefault("HEADLESS", "1")
        return
    if "SDL_VIDEODRIVER" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["SDL_VIDEODRIVER"] = "wayland"
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Save state name (default: FirstPlay)",
    )
    parser.add_argument(
        "--seed",
        default="demo",
        help="Seed label for run manifest (default: demo)",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Dummy SDL / no window (also HEADLESS=1)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable MP4 recording (recording is ON by default)",
    )
    parser.add_argument(
        "--rebuild-boot",
        action="store_true",
        help="Force re-run power-on boot and overwrite FirstPlay.state",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Quit after N frames (useful for headless smoke)",
    )
    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Play super_metroid SuperMetroid-Snes instead of SMRando-Snes",
    )
    parser.add_argument(
        "--record-stride",
        type=int,
        default=2,
        help="Write every Nth frame to MP4 (default: 2)",
    )
    args = parser.parse_args(argv)

    headless = args.headless or os.environ.get("HEADLESS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    _configure_display(headless=headless)

    from retro_harness.env import make_env
    from retro_harness.play_session import PlaySession
    from retro_harness.play_spine import RunManifest, fun_hud_lines, utc_now_iso
    from sm_rando.paths import (
        FIRST_PLAY_STATE,
        GAME,
        GAME_DIR,
        INTEGRATION_DIR,
        RECORDINGS_DIR,
    )

    if args.vanilla:
        from sm_rando.play import vanilla_skill_play

        manifest = vanilla_skill_play(
            state=args.state or "FirstAction",
            seed=args.seed or "vanilla",
            scale=args.scale,
            headless=headless or None,
        )
        path = manifest.meta.get("manifest_path")
        print(f"outcome={manifest.outcome} frames={manifest.frames} manifest={path}")
        return 0

    # Ensure ROM + FirstPlay for SMRando-Snes path.
    if not (INTEGRATION_DIR / "rom.sfc").exists():
        from sm_rando.scripts.setup_rom import setup_rom

        setup_rom()

    from sm_rando.boot import ensure_first_play_state

    state_name = args.state or FIRST_PLAY_STATE
    if state_name == FIRST_PLAY_STATE or args.rebuild_boot:
        try:
            ensure_first_play_state(rebuild=args.rebuild_boot)
        except Exception as exc:  # noqa: BLE001
            print(f"boot/ensure FirstPlay failed: {exc}", file=sys.stderr)
            return 2

    state_path = INTEGRATION_DIR / f"{state_name}.state"
    if not state_path.is_file():
        print(f"Missing state: {state_path}", file=sys.stderr)
        return 1

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = RECORDINGS_DIR / f"play_{stamp}.mp4"
    manifest_path = RECORDINGS_DIR / f"play_{stamp}.json"

    manifest = RunManifest(
        game=GAME,
        package="sm_rando",
        started_at=utc_now_iso(),
        seed=args.seed,
        start_state=state_name,
        mode="record" if not args.no_record else "play",
        meta={"title": f"SM Rando · {state_name}"},
    )
    manifest.add_milestone("first_play")

    env = make_env(
        game=GAME,
        state=state_name,
        game_dir=GAME_DIR,
        render_mode="rgb_array",
    )

    writer: Any = None
    frame_stride = max(1, args.record_stride)
    frames_seen = 0
    # SNES framebuffer; refined on first on_step if needed.
    frame_h, frame_w = 224, 256

    try:
        if not args.no_record:
            from retro_harness.video import FrameVideoWriter

            writer = FrameVideoWriter(
                video_path,
                width=frame_w,
                height=frame_h,
                fps=30 if frame_stride > 1 else 60,
                scale=2,
                crf=20,
                preset="veryfast",
            )
            manifest.video_path = str(video_path)

        session = PlaySession(
            env,
            game_dir=str(GAME_DIR),
            game=GAME,
            scale=args.scale,
            title=f"SM Rando · {state_name}",
            headless=headless,
            action_size=12,
        )

        def _hud(info: dict) -> list[str]:
            frame = int(info.get("frame", session.frame_count) or 0)
            manifest.frames = max(manifest.frames, frame)
            return fun_hud_lines(
                package="sm_rando",
                seed=args.seed,
                frame=frame,
                milestone="first_play / Ceres",
                extra=["F5=save · rec on" if writer else "F5=save · no-record"],
            )

        def _on_step(obs: Any, reward: float, done: bool, info: dict) -> None:
            del reward, done
            nonlocal frames_seen
            manifest.frames = max(manifest.frames, session.frame_count)
            info["frame"] = session.frame_count
            if writer is not None:
                rgb = np.asarray(obs)
                if rgb.ndim != 3 or rgb.shape[-1] != 3:
                    rendered = env.render()
                    if rendered is None:
                        return
                    rgb = np.asarray(rendered)
                if frames_seen % frame_stride == 0:
                    writer.write(rgb)
                frames_seen += 1
            if args.max_frames is not None and session.frame_count >= args.max_frames:
                session.running = False

        session.on_hud = _hud
        session.on_step = _on_step  # type: ignore[method-assign]
        # PlaySession.run closes env in its finally block.
        session.run()
        manifest.outcome = "session_end"
    except SystemExit:
        manifest.outcome = "exit"
        raise
    except Exception as exc:  # noqa: BLE001
        manifest.outcome = "error"
        manifest.notes.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001
                manifest.notes.append(f"video_close: {exc}")
        manifest.meta["recorded_at"] = datetime.now(timezone.utc).isoformat()
        manifest.meta["integration"] = str(INTEGRATION_DIR)
        summary = {
            **manifest.to_dict(),
            "manifest_path": str(manifest_path),
            "video_path": str(video_path) if not args.no_record else None,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        manifest.meta["manifest_path"] = str(manifest_path)

    print(
        f"outcome={manifest.outcome} frames={manifest.frames} "
        f"manifest={manifest_path}"
        + (f" video={video_path}" if not args.no_record else "")
    )
    return 0 if manifest.outcome in {"session_end", "exit"} else 1

if __name__ == "__main__":
    raise SystemExit(main())
