"""Replay a spine button tape from power-on: no controller, no lookahead, no load.

    uv run python nes/zelda_i/scripts/replay_tape.py nes/zelda_i/recordings/<tag>.tape.npz
    uv run python nes/zelda_i/scripts/replay_tape.py <tape> --no-video   # sync check only

``run_survival_spine.py`` writes ``recordings/<tag>.tape.npz``: every button
vector its live session played through ``env.step``. Rollout lookahead steps
``env.em`` and restores it, so none of its frames are on the tape. Playing the
tape into a fresh power-on is the recording path: it runs at emulator speed,
and the audit sees zero state loads and zero RAM writes. The replay is in sync
when its final RAM hashes to the one the live run wrote beside the tape.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from retro_harness.audit import AuditCapabilities, AuditedEnv
from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import (
    TAPE_SUFFIX,
    VideoTap,
    add_video_args,
    load_tape,
    ram_crc,
    resolve_video,
    unpack_buttons,
)
from zelda_i.spine.survival import spine_final_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", type=Path)
    parser.add_argument("--tag", default=None, help="Default: <tape tag>_replay")
    add_video_args(parser, default_on=True)
    args = parser.parse_args(argv)

    configure_headless()
    tape = load_tape(args.tape)
    meta = tape.meta
    if meta.get("resumed_from"):
        # The tape starts at a loaded save point, not at power-on.
        parser.error(f"tape resumed from {meta['resumed_from']!r}; replay needs a power-on tape")
    source = str(meta.get("tag") or args.tape.name.removesuffix(TAPE_SUFFIX))
    tag = args.tag or f"{source}_replay"
    video_path, video_config, intro = resolve_video(
        args, default_path=RECORDINGS_DIR / f"{tag}.mp4"
    )
    clean = bool(meta.get("clean"))
    tap = VideoTap(
        video_path,
        video_config,
        tag=tag,
        intro_summary="Continuous power-on, first quest, first file",
        intro_frames=intro,
        intervention="Clean: no refills, no RAM writes" if clean else "Survival infinite-life",
        transition_pngs=False,
    )
    frames = len(tape.buttons)
    # Frame (1-based) whose system RAM first differs from the live run's.
    first_desync: int | None = None
    desynced = 0
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        env = AuditedEnv(env, capabilities=AuditCapabilities.all("zelda_i.replay_tape"))
        tap.attach(env, obs)
        started = time.perf_counter()
        for index, (bits, crc) in enumerate(zip(tape.buttons, tape.crcs)):
            obs, *_ = env.step(unpack_buttons(int(bits)))
            if ram_crc(env.get_ram()) != int(crc):
                desynced += 1
                first_desync = first_desync or index + 1
        wall_s = time.perf_counter() - started
        audit = env.audit()
        final_ram = np.array(env.get_ram(), dtype=np.uint8)
        screenshot = RECORDINGS_DIR / f"{tag}_final.png"
        save_rgb_png(obs, screenshot)
    finally:
        try:
            video = tap.close()
        except Exception:
            tap.abort()
            video = {"path": None}
        env.close()

    in_sync = desynced == 0 and len(tape.crcs) == frames
    # Final bytes that differ from the live run's, cart WRAM included (its
    # $6000/$652D differ between any two processes and are not a desync).
    diff = [
        f"${i:04X}:{int(a):02X}->{int(b):02X}"
        for i, (a, b) in enumerate(zip(tape.ram, final_ram))
        if a != b
    ]
    loads = int(audit.mid_run_loads or 0)
    writes = int(audit.ram_writes or 0)
    report = {
        "ok": bool(in_sync and not loads and not writes and meta.get("ok")),
        "tape": str(args.tape),
        "source": meta,
        "frames": frames,
        "in_sync": in_sync,
        "first_desync_frame": first_desync,
        "desynced_frames": desynced,
        "final_ram_diff": diff[:64],
        "set_state_count": loads,
        "ram_writes": writes,
        "wall_s": round(wall_s, 1),
        "x_realtime": round(frames / 60.0 / wall_s, 2) if wall_s else None,
        "final": spine_final_fields(read_snapshot(final_ram), final_ram),
        "screenshot": str(screenshot),
        "video": video,
    }
    write_json_report(RECORDINGS_DIR / f"{tag}.json", report)
    print(
        f"replay {tag}: ok={report['ok']} in_sync={in_sync} "
        f"first_desync={first_desync} frames={frames} set_state={loads} "
        f"ram_writes={writes} wall={report['wall_s']}s "
        f"({report['x_realtime']}x real time) video={video.get('path')}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
