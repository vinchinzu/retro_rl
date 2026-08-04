"""Render a full route to MP4."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from retro_harness.platformer.level_config import get_level_config
from retro_harness.platformer.route.models import (
    RouteConfig,
    find_best_recording,
    load_recording_data,
)



def record_route_video(
    route: RouteConfig,
    output: str,
    *,
    scale: int = 3,
    completion_hold: int = 60,
) -> None:
    """Render the full route as a single MP4.

    Each segment is played in its own env from its saved state.
    Segments are stitched together with a brief label overlay between them.
    """
    from retro_harness.platformer.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS
    from retro_harness.platformer.record_video import draw_text
    from retro_harness.env import make_env

    # First pass: figure out video dimensions from first segment
    seg0 = route.segments[0]
    cfg0 = get_level_config(seg0.config_id)
    env0 = make_env(cfg0.game_name, cfg0.start_state, cfg0.game_dir, render_mode="rgb_array")
    obs0, _ = env0.reset()
    h, w = obs0.shape[0], obs0.shape[1]
    env0.close()

    out_h, out_w = h * scale, w * scale
    cumulative_frames = 0

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb24", "-r", "60",
        "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]

    print(f"Recording route: {route.display_name} ({len(route.segments)} segments)")
    print(f"Output: {output} at {out_w}x{out_h}")

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_frame(f):
        try:
            proc.stdin.write(f.tobytes())
        except BrokenPipeError:
            pass

    def write_title_card(text: str, frames: int = 90):
        """Write a black frame with centered text for N frames."""
        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        cx = out_w // 2 - len(text) * 5 // 2
        cy = out_h // 2 - 3
        draw_text(frame, text, cx, cy, (255, 255, 255), shadow=False)
        for _ in range(frames):
            write_frame(frame)

    for i, seg in enumerate(route.segments):
        try:
            config = get_level_config(seg.config_id)
        except KeyError:
            print(f"  [{i}] {seg.label}: CONFIG ERROR, skipping")
            continue

        # Find recording
        if seg.recording:
            rec_path = Path(seg.recording)
            if not rec_path.is_absolute():
                rec_path = config.runs_dir / seg.recording
        else:
            rec_path = find_best_recording(config)

        if rec_path is None or not rec_path.exists():
            print(f"  [{i}] {seg.label}: NO RECORDING, skipping")
            continue

        actions, is_raw = load_recording_data(rec_path)
        action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS

        # Title card between segments
        write_title_card(seg.label or config.display_name)

        env = make_env(config.game_name, config.start_state, config.game_dir, render_mode="rgb_array")
        obs, _ = env.reset()
        action_size = env.action_space.shape[0]

        schema = config.ram_schema
        ram = env.get_ram()
        initial_values = schema.read(ram)
        config.apply_computed(initial_values)
        initial_lives = initial_values.get("lives")
        _main_ids = {config.target_level_id} | set(config.level_id_aliases)
        max_progress = 0.0
        seg_frames = 0

        print(f"  [{i}] {seg.label}: {len(actions)} frames from {rec_path.name}")

        for frame_idx, action in enumerate(actions):
            if is_raw:
                buttons = list(action)
            else:
                buttons = action_index_to_buttons(action, action_table)
            if len(buttons) < action_size:
                buttons = buttons + [0] * (action_size - len(buttons))
            elif len(buttons) > action_size:
                buttons = buttons[:action_size]

            obs, *_ = env.step(np.array(buttons, dtype=np.int8))

            ram = env.get_ram()
            values = schema.read(ram)
            config.apply_computed(values)
            level_id = values.get("level_id", 0)
            lives = values.get("lives", 0)
            in_sub = level_id != 0 and level_id not in _main_ids

            if not in_sub:
                px = float(values.get("player_x", values.get("camera_x", 0)))
                if px > max_progress:
                    max_progress = px

            # Check completion
            completed = False
            if config.completion_signal == "level_id_change":
                if level_id not in _main_ids and level_id != 0:
                    if (max_progress >= config.completion_min_progress
                            and (not config.completion_level_ids or level_id in config.completion_level_ids)
                            and level_id not in config.completion_exclude_ids):
                        completed = True
            elif config.completion_signal == "ram_flag":
                flag_val = values.get(config.completion_ram_key, None)
                if (flag_val is not None
                        and flag_val == config.completion_ram_value
                        and max_progress >= config.completion_min_progress):
                    completed = True

            died = initial_lives is not None and lives < initial_lives

            # Scale and annotate frame
            frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
            secs = (cumulative_frames + frame_idx) / 60.0
            draw_text(frame, f"F:{cumulative_frames + frame_idx}", 4, 4)
            draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
            draw_text(frame, seg.label or config.display_name, 4, 20, (100, 255, 100))

            if completed:
                draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
            elif died:
                draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))

            write_frame(frame)
            seg_frames = frame_idx + 1

            if completed:
                for _ in range(completion_hold):
                    obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                    frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                    draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
                    write_frame(frame)
                break

            if died:
                for _ in range(60):
                    obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                    frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                    draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))
                    write_frame(frame)
                break

        cumulative_frames += seg_frames
        env.close()

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()

    out_path = Path(output)
    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nDone! {output} ({size_mb:.1f} MB, {cumulative_frames}f / {cumulative_frames/60:.1f}s)")
