#!/usr/bin/env python3
"""Watch the full speedrun route (Landing Site → Bomb Torizo) in a pygame window
or encode to MP4 video.

Usage:
    # Watch live in pygame window
    uv run python -m super_metroid_rl.scripts.watch_full_route

    # Encode to MP4 video
    uv run python -m super_metroid_rl.scripts.watch_full_route --video route.mp4

    # Adjust scale
    uv run python -m super_metroid_rl.scripts.watch_full_route --scale 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retro_harness.env import make_env

LEVEL_ID_ADDR = 0x079B
PLAYER_X_ADDR = 0x0AF6
PLAYER_Y_ADDR = 0x0AFA
HEALTH_ADDR = 0x09C2

ROOM_NAMES = {
    0x91F8: "Landing Site",
    0x92FD: "Parlor",
    0x96BA: "Climb",
    0x975C: "Pit Room",
    0x97B5: "BB Elev Hallway",
    0x9E9F: "Morph Ball Room",
    0x9F11: "Construction Zone",
    0xA107: "First Missile Room",
    0x9804: "Bomb Torizo Room",
    0x9879: "Flyway",
}

# Full route: (state_name, segment_file_relative_to_runs_dir, display_label)
FULL_ROUTE = [
    ("ZebesStart",           "segments/seg00_landing_site.json",        "1/15 Landing Site"),
    ("seg01_0x92FD",         "segments/seg01_parlor.json",              "2/15 Parlor (descent)"),
    ("seg02_0x96BA",         "segments/seg02_climb.json",               "3/15 Climb (descent)"),
    ("seg03_0x975C",         "segments/seg03_pit_room.json",            "4/15 Pit Room (descent)"),
    ("seg04_0x97B5_fixed",   "segments/seg04_bb_elev_hallway.json",     "5/15 BB Elevator (descent)"),
    ("seg05_0x9E9F_auto",    "hc_morph_collect/hillclimb_raw_best.json", "6/15 Morph Ball Room [HC]"),
    ("seg06_0x9F11",         "segments/seg06_construction_zone.json",   "7/15 Construction Zone"),
    ("seg07_0xA107",         "segments/seg07_first_missile_room.json",  "8/15 First Missile Room"),
    ("seg08_0x9F11",         "hc_constr_ret/hillclimb_raw_best.json",  "9/15 Construction (return) [HC]"),
    ("seg09_0x9E9F",         "hc_morph_ret/hillclimb_raw_best.json",   "10/15 Morph Ball (return) [HC]"),
    ("seg10_0x97B5",         "segments/seg10_bb_elev_hallway.json",     "11/15 BB Elevator (return)"),
    ("seg11_0x975C",         "segments_return/seg00_pit_room.json",     "12/15 Pit Room (return)"),
    ("seg01_0x96BA",         "segments_return/seg01_climb.json",        "13/15 Climb (return)"),
    ("seg02_0x92FD",         "segments_return/seg02_parlor.json",       "14/15 Parlor → Flyway"),
    ("seg03_0x9879",         "segments_return/seg03_flyway.json",       "15/15 Flyway → Torizo"),
]


def read_u16(ram, addr):
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


# Bitmap font for video HUD (same as record_video.py)
FONT = {
    '0': ["0110", "1001", "1001", "1001", "0110"],
    '1': ["0010", "0110", "0010", "0010", "0111"],
    '2': ["0110", "1001", "0010", "0100", "1111"],
    '3': ["1110", "0001", "0110", "0001", "1110"],
    '4': ["1001", "1001", "1111", "0001", "0001"],
    '5': ["1111", "1000", "1110", "0001", "1110"],
    '6': ["0110", "1000", "1110", "1001", "0110"],
    '7': ["1111", "0001", "0010", "0100", "0100"],
    '8': ["0110", "1001", "0110", "1001", "0110"],
    '9': ["0110", "1001", "0111", "0001", "0110"],
    'A': ["0110", "1001", "1111", "1001", "1001"],
    'B': ["1110", "1001", "1110", "1001", "1110"],
    'C': ["0111", "1000", "1000", "1000", "0111"],
    'D': ["1110", "1001", "1001", "1001", "1110"],
    'E': ["1111", "1000", "1110", "1000", "1111"],
    'F': ["1111", "1000", "1110", "1000", "1000"],
    'G': ["0111", "1000", "1011", "1001", "0111"],
    'H': ["1001", "1001", "1111", "1001", "1001"],
    'I': ["0111", "0010", "0010", "0010", "0111"],
    'K': ["1001", "1010", "1100", "1010", "1001"],
    'L': ["1000", "1000", "1000", "1000", "1111"],
    'M': ["10001", "11011", "10101", "10001", "10001"],
    'N': ["1001", "1101", "1011", "1001", "1001"],
    'O': ["0110", "1001", "1001", "1001", "0110"],
    'P': ["1110", "1001", "1110", "1000", "1000"],
    'R': ["1110", "1001", "1110", "1010", "1001"],
    'S': ["0111", "1000", "0110", "0001", "1110"],
    'T': ["1110", "0100", "0100", "0100", "0100"],
    'U': ["1001", "1001", "1001", "1001", "0110"],
    'V': ["1001", "1001", "1001", "0110", "0010"],
    'W': ["10001", "10001", "10101", "11011", "10001"],
    'X': ["1001", "0110", "0110", "0110", "1001"],
    'Y': ["1001", "1001", "0110", "0010", "0010"],
    'Z': ["1111", "0010", "0100", "1000", "1111"],
    '+': ["0000", "0100", "1110", "0100", "0000"],
    '-': ["0000", "0000", "1110", "0000", "0000"],
    ' ': ["0000", "0000", "0000", "0000", "0000"],
    ':': ["0000", "0100", "0000", "0100", "0000"],
    '/': ["0001", "0010", "0100", "1000", "0000"],
    '.': ["0000", "0000", "0000", "0000", "0100"],
    '(': ["0010", "0100", "0100", "0100", "0010"],
    ')': ["0100", "0010", "0010", "0010", "0100"],
}


def draw_text(frame, text, x, y, color=(255, 255, 255), shadow=True):
    h, w = frame.shape[:2]
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            cx += 5
            continue
        for gy, row in enumerate(glyph):
            for gx, pixel in enumerate(row):
                if pixel == '1':
                    py, px = y + gy, cx + gx
                    if 0 <= py < h and 0 <= px < w:
                        if shadow:
                            sy, sx = py + 1, px + 1
                            if 0 <= sy < h and 0 <= sx < w:
                                frame[sy, sx] = (0, 0, 0)
                        frame[py, px] = color
        cx += len(glyph[0]) + 1 if glyph else 5


BUTTON_NAMES = ["B", "Y", "Sel", "Sta", "U", "D", "L", "R", "A", "X", "L1", "R1"]


def button_label(buttons):
    pressed = [BUTTON_NAMES[i] for i in range(min(len(buttons), 12)) if buttons[i]]
    return "+".join(pressed) if pressed else "-"


def main():
    parser = argparse.ArgumentParser(description="Watch full speedrun route")
    parser.add_argument("--video", "-o", help="Output MP4 path (omit for pygame window)")
    parser.add_argument("--scale", type=int, default=3, help="Display scale")
    args = parser.parse_args()

    game_dir = PROJECT_ROOT / "super_metroid_rl"
    state_dir = game_dir / "custom_integrations" / "SuperMetroid-Snes"
    runs_dir = game_dir / "optimizer" / "runs" / "sm_landing_site"

    # Load all segment data upfront
    all_segments = []
    total_frames = 0
    for state_name, seg_rel, label in FULL_ROUTE:
        seg_path = runs_dir / seg_rel
        if not seg_path.exists():
            print(f"MISSING: {seg_rel}")
            return 1
        seg_data = json.loads(seg_path.read_text())
        raw = seg_data["raw_buttons"]
        all_segments.append((state_name, raw, label))
        total_frames += len(raw)

    print(f"Full route: {len(all_segments)} segments, {total_frames} frames ({total_frames/60:.1f}s)")

    # Create env once
    env = make_env(
        game="SuperMetroid-Snes",
        state=FULL_ROUTE[0][0],
        game_dir=str(game_dir),
        render_mode="rgb_array",
    )
    obs, _ = env.reset()
    h, w = obs.shape[:2]
    out_h, out_w = h * args.scale, w * args.scale

    # Select workaround state
    _select_prev = False
    _select_val = 0
    _has_selected_item = None

    if args.video:
        # === VIDEO MODE ===
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "rgb24", "-r", "60",
            "-i", "-",
            "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            args.video,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        global_frame = 0
        for seg_idx, (state_name, raw_buttons, label) in enumerate(all_segments):
            # Load state
            state_path = state_dir / f"{state_name}.state"
            with gzip.open(state_path, "rb") as f:
                state_bytes = f.read()
            env.em.set_state(state_bytes)

            ram = env.get_ram()
            room = read_u16(ram, LEVEL_ID_ADDR)
            room_name = ROOM_NAMES.get(room, f"0x{room:04X}")
            print(f"  [{seg_idx+1}/{len(all_segments)}] {room_name}: {len(raw_buttons)}f")

            for f_idx, buttons in enumerate(raw_buttons):
                action_arr = np.array(buttons, dtype=np.int8)
                action_size = env.action_space.shape[0]
                if len(buttons) < action_size:
                    padded = np.zeros(action_size, dtype=np.int8)
                    padded[:len(buttons)] = buttons
                    action_arr = padded

                # Select workaround
                if len(buttons) > 2 and buttons[2]:
                    if not _select_prev:
                        if _has_selected_item is None:
                            try:
                                env.unwrapped.data.lookup_value("selected_item")
                                _has_selected_item = True
                            except Exception:
                                _has_selected_item = False
                        if _has_selected_item:
                            _select_val ^= 1
                            try:
                                env.unwrapped.data.set_value("selected_item", _select_val)
                            except Exception:
                                pass
                    _select_prev = True
                else:
                    _select_prev = False

                obs, *_ = env.step(action_arr)

                ram = env.get_ram()
                hp = read_u16(ram, HEALTH_ADDR)
                px = read_u16(ram, PLAYER_X_ADDR)
                py = read_u16(ram, PLAYER_Y_ADDR)
                cur_room = read_u16(ram, LEVEL_ID_ADDR)
                cur_name = ROOM_NAMES.get(cur_room, f"0x{cur_room:04X}")

                # Scale frame
                frame = np.repeat(np.repeat(obs, args.scale, axis=0), args.scale, axis=1).copy()

                # HUD
                secs = global_frame / 60.0
                btn = button_label(buttons)
                draw_text(frame, f"F:{global_frame} T:{secs:.1f}S", 4, 4, (255, 255, 255))
                draw_text(frame, f"{cur_name}  HP:{hp}", 4, 12, (200, 200, 200))
                draw_text(frame, f"BTN:{btn}", 4, 20, (100, 255, 100))
                draw_text(frame, label, 4, 28, (255, 200, 100))

                try:
                    proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break

                global_frame += 1

        # Hold last frame for 2 seconds
        for _ in range(120):
            obs, *_ = env.step(np.zeros(env.action_space.shape[0], dtype=np.int8))
            frame = np.repeat(np.repeat(obs, args.scale, axis=0), args.scale, axis=1).copy()
            draw_text(frame, "ROUTE COMPLETE", 4, 4, (0, 255, 0))
            secs = global_frame / 60.0
            draw_text(frame, f"F:{global_frame} T:{secs:.1f}S", 4, 12, (255, 255, 255))
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break

        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        proc.wait()

        env.close()

        out_path = Path(args.video)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\nDone! {args.video} ({size_mb:.1f} MB, {global_frame} frames, {global_frame/60:.1f}s)")

    else:
        # === PYGAME WINDOW MODE ===
        import os
        # Wayland: use wayland driver, fall back to x11
        if "WAYLAND_DISPLAY" in os.environ:
            os.environ.setdefault("SDL_VIDEODRIVER", "wayland")
        else:
            os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        import pygame

        pygame.init()
        screen = pygame.display.set_mode(
            (out_w, out_h), pygame.SWSURFACE
        )
        pygame.display.set_caption("Super Metroid Speedrun: Landing Site → Bomb Torizo")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)

        running = True
        global_frame = 0
        turbo = False

        for seg_idx, (state_name, raw_buttons, label) in enumerate(all_segments):
            if not running:
                break

            # Load state
            state_path = state_dir / f"{state_name}.state"
            with gzip.open(state_path, "rb") as f:
                state_bytes = f.read()
            env.em.set_state(state_bytes)

            ram = env.get_ram()
            room = read_u16(ram, LEVEL_ID_ADDR)
            room_name = ROOM_NAMES.get(room, f"0x{room:04X}")
            print(f"  [{seg_idx+1}/{len(all_segments)}] {room_name}: {len(raw_buttons)}f")

            for f_idx, buttons in enumerate(raw_buttons):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_TAB:
                            turbo = not turbo
                if not running:
                    break

                action_arr = np.array(buttons, dtype=np.int8)
                action_size = env.action_space.shape[0]
                if len(buttons) < action_size:
                    padded = np.zeros(action_size, dtype=np.int8)
                    padded[:len(buttons)] = buttons
                    action_arr = padded

                # Select workaround
                if len(buttons) > 2 and buttons[2]:
                    if not _select_prev:
                        if _has_selected_item is None:
                            try:
                                env.unwrapped.data.lookup_value("selected_item")
                                _has_selected_item = True
                            except Exception:
                                _has_selected_item = False
                        if _has_selected_item:
                            _select_val ^= 1
                            try:
                                env.unwrapped.data.set_value("selected_item", _select_val)
                            except Exception:
                                pass
                    _select_prev = True
                else:
                    _select_prev = False

                obs, *_ = env.step(action_arr)
                global_frame += 1

                # Render (skip some frames in turbo)
                if turbo and f_idx % 4 != 0:
                    continue

                ram = env.get_ram()
                hp = read_u16(ram, HEALTH_ADDR)
                cur_room = read_u16(ram, LEVEL_ID_ADDR)
                cur_name = ROOM_NAMES.get(cur_room, f"0x{cur_room:04X}")

                surf = pygame.surfarray.make_surface(obs.swapaxes(0, 1))
                screen.blit(pygame.transform.scale(surf, screen.get_size()), (0, 0))

                secs = global_frame / 60.0
                btn = button_label(buttons)
                turbo_tag = " [TURBO]" if turbo else ""
                lines = [
                    f"F{global_frame}/{total_frames} ({secs:.1f}s) | {cur_name} | HP:{hp} | {btn}{turbo_tag}",
                    label,
                ]
                for i, line in enumerate(lines):
                    text = font.render(line, True, (255, 255, 0))
                    screen.blit(text, (4, 4 + i * 18))

                pygame.display.flip()
                clock.tick(0 if turbo else 60)

        # Hold at end
        if running:
            end_font = pygame.font.SysFont("monospace", 24)
            for _ in range(180):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        running = False
                        break
                if not running:
                    break
                text = end_font.render(
                    f"ROUTE COMPLETE  {global_frame}f  ({global_frame/60:.1f}s)",
                    True, (0, 255, 0),
                )
                screen.blit(text, (out_w // 2 - text.get_width() // 2, out_h // 2))
                pygame.display.flip()
                clock.tick(60)

        pygame.quit()
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
