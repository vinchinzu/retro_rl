#!/usr/bin/env python3
"""Interactive play + record for ALTTP Rando (JP 1.0).

Defaults to FirstPlay (first controllable frame — no title/name menus).

```bash
./play
uv run python -m alttp_rando.scripts.play
uv run python -m alttp_rando.scripts.play --no-record
uv run python -m alttp_rando.scripts.play --rebuild-boot
uv run python -m alttp_rando.scripts.play --vanilla
```

F5 saves into ``custom_integrations/ALTTPRando-Snes/``.
Recordings (MP4 + JSON) go under ``recordings/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ALTTP Rando play / record")
    parser.add_argument(
        "--state",
        default=None,
        help="Save state name (default: FirstPlay for rando, LinksHouseWake for --vanilla)",
    )
    parser.add_argument("--seed", default=None, help="Seed label for manifest")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip MP4 recording (JSON manifest still written)",
    )
    parser.add_argument(
        "--rebuild-boot",
        action="store_true",
        help="Force recreate FirstPlay.state before play",
    )
    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="Play vanilla ALTTP (USA) skills under this package spine",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recorded video FPS (stride applied from 60Hz)",
    )
    args = parser.parse_args(argv)

    from alttp_rando.paths import (
        FIRST_PLAY_STATE,
        GAME,
        GAME_DIR,
        INTEGRATION_DIR,
        RECORDINGS_DIR,
    )
    from alttp_rando.play import vanilla_skill_play
    from retro_harness.play_session import PlaySession
    from retro_harness.video import FrameVideoWriter

    if args.vanilla:
        state = args.state or "LinksHouseWake"
        manifest = vanilla_skill_play(
            state=state,
            seed=args.seed or "vanilla",
            scale=args.scale,
            headless=True if args.headless else None,
        )
        path = manifest.meta.get("manifest_path")
        print(f"outcome={manifest.outcome} frames={manifest.frames} manifest={path}")
        return 0

    # Ensure JP ROM + FirstPlay.
    from alttp_rando.scripts.setup_rom import main as setup_main

    rc = setup_main()
    if rc != 0:
        return rc

    from alttp_rando.boot import ensure_first_play_state

    try:
        ensure_first_play_state(rebuild=args.rebuild_boot)
    except RuntimeError as exc:
        print(f"boot failed: {exc}", file=sys.stderr)
        return 1

    state = args.state or FIRST_PLAY_STATE
    state_file = INTEGRATION_DIR / f"{state}.state"
    if not state_file.is_file():
        print(f"Missing state: {state_file}", file=sys.stderr)
        return 1

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_label = args.seed or "jp_vanilla"
    video_path = RECORDINGS_DIR / f"play_{seed_label}_{stamp}.mp4"
    json_path = RECORDINGS_DIR / f"play_{seed_label}_{stamp}.json"

    writer: FrameVideoWriter | None = None
    frames_written = 0
    frame_stride = max(1, 60 // max(1, args.fps))
    session_frames = {"n": 0}

    def _session_factory(env: Any, **kwargs: Any) -> PlaySession:
        nonlocal writer, frames_written
        session = PlaySession(env, **kwargs)

        if not args.no_record:
            # Size writer from first obs after reset — hook on_step after start.
            # Probe one render for dimensions.
            try:
                obs = env.render()
                if obs is None:
                    # reset not yet called; defer until first on_step
                    dims = None
                else:
                    rgb = np.asarray(obs)
                    dims = (int(rgb.shape[0]), int(rgb.shape[1]))
            except Exception:
                dims = None

            if dims is not None:
                h, w = dims
                writer = FrameVideoWriter(
                    video_path,
                    width=w,
                    height=h,
                    fps=args.fps,
                    scale=max(1, args.scale),
                )

            original_step = session.on_step

            def _on_step(obs: Any, reward: float, done: bool, info: dict) -> None:
                nonlocal writer, frames_written
                session_frames["n"] = session.frame_count
                if callable(original_step):
                    try:
                        original_step(obs, reward, done, info)
                    except TypeError:
                        # Some hooks only take info.
                        pass
                if args.no_record:
                    return
                rgb = np.asarray(obs)
                if rgb.ndim != 3 or rgb.shape[-1] != 3:
                    rendered = env.render()
                    if rendered is None:
                        return
                    rgb = np.asarray(rendered)
                if writer is None:
                    writer = FrameVideoWriter(
                        video_path,
                        width=int(rgb.shape[1]),
                        height=int(rgb.shape[0]),
                        fps=args.fps,
                        scale=max(1, args.scale),
                    )
                if session.frame_count % frame_stride == 0:
                    writer.write(rgb)
                    frames_written = writer.frames_written

            session.on_step = _on_step  # type: ignore[method-assign]

            original_close = session.on_close

            def _on_close() -> None:
                nonlocal writer
                if callable(original_close):
                    original_close()
                if writer is not None:
                    writer.close()
                    writer = None

            session.on_close = _on_close  # type: ignore[method-assign]

        return session

    # run_play_spine doesn't accept session_factory; use play_game + manifest.
    from retro_harness.play_spine import (
        RunManifest,
        configure_display,
        default_manifest_path,
        fun_hud_lines,
        utc_now_iso,
    )
    from retro_harness.live_play import play_game

    is_headless = args.headless or (
        os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
    )
    configure_display(headless=is_headless)

    manifest = RunManifest(
        game=GAME,
        package="alttp_rando",
        started_at=utc_now_iso(),
        seed=seed_label,
        start_state=state,
        mode="play",
        meta={"title": f"ALTTP Rando · {state}", "jp_rom": True},
    )
    manifest.add_milestone("first_play")

    def _hud(info: dict) -> list[str]:
        frame = int(info.get("frame", 0) or 0)
        manifest.frames = max(manifest.frames, frame)
        return fun_hud_lines(
            package="alttp_rando",
            seed=seed_label,
            frame=frame,
            milestone="FirstPlay",
            extra=["F5 save · JP 1.0"],
        )

    try:
        play_game(
            game=GAME,
            state=state,
            game_dir=GAME_DIR,
            title=f"ALTTP Rando · {state}",
            scale=args.scale,
            on_hud=_hud,
            session_factory=_session_factory,
            headless=is_headless,
        )
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
            frames_written = writer.frames_written
            writer.close()
            writer = None
        manifest.frames = max(manifest.frames, session_frames["n"])
        if not args.no_record and video_path.is_file():
            manifest.video_path = str(video_path)
            manifest.mode = "record"
        # Dual write: play_*.json + spine-style path.
        report = manifest.to_dict()
        report["video_frames"] = frames_written
        report["integration_dir"] = str(INTEGRATION_DIR)
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        spine = default_manifest_path(
            RECORDINGS_DIR, package="alttp_rando", seed=seed_label
        )
        manifest.meta["manifest_path"] = str(json_path)
        manifest.meta["spine_path"] = str(spine)
        manifest.write(spine)
        print(f"manifest={json_path}")
        if not args.no_record and video_path.is_file():
            print(f"video={video_path} frames={frames_written}")

    print(f"outcome={manifest.outcome} frames={manifest.frames}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
