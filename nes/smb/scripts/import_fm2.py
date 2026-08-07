"""Import FCEUX ``.fm2`` movies into ``nes9_rle`` SMB seeds + optional verify.

Primary reference: HappyLee warps TASVideos #1715M (04:57.31 power-on /
04:54.032 RTA). Community RTA-rules movies also work.

```bash
# Parse + write continuous seed from vendored HappyLee movie
uv run python -m smb.scripts.import_fm2 \\
  nes/smb/tas/ref/happylee_warps_1715M.fm2 \\
  --out nes/smb/models/smb_happylee_warps_raw.json

# Power-on verify under stable-retro (no L+R sanitize)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.import_fm2 \\
  nes/smb/tas/ref/happylee_warps_1715M.fm2 --verify --max-frames 20000

# Control-relative 1-2 W4 slice (after HappyLee 1-1 natural predecessor)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.import_fm2 --verify-1-2-slice
uv run python -m smb.scripts.import_fm2 --export-1-2-slice

# Control-relative 4-1 + 4-2 → W8 (after HL W4)
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.scripts.import_fm2 --verify-4-1-4-2-slice
uv run python -m smb.scripts.import_fm2 --export-4-1-slice --export-4-2-slice
```

Important: TAS frames use Left+Right. Replay must **not** call
``sanitize_action`` / directional conflict stripping.

Playbook: ``nes/smb/docs/TAS_ADAPT.md``. Prefer per-level slices over full
power-on FM2 (fceumm blackout desyncs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

from smb.paths import GAME_DIR, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle, expand_nes9_rle
from smb.ram import read_snapshot, reached_ending
from smb.tas.fm2 import fm2_to_nes9_frames, frames_to_nes9_rle_payload, parse_fm2
from smb.timing import NTSC_FPS, build_timing_block, format_time


DEFAULT_REF = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2"


def _action9(frame: list[int]) -> np.ndarray:
    action = np.zeros(9, dtype=np.int8)
    for j in range(min(9, len(frame))):
        action[j] = int(frame[j])
    return action


def _probe_11_progress(
    frames: list[list[int]],
    *,
    max_frames: int = 2500,
    pad_before: int = 0,
    skip_movie: int = 0,
) -> dict[str, Any]:
    """Power-on probe: max x / death in early 1-1 under pad/skip alignment.

    ``pad_before`` inserts idle frames before the movie (if our blackout is
    shorter). ``skip_movie`` drops leading movie frames (if ours is longer).
    """
    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0
    from smb.ram import PLAYER_STATE_DYING

    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    idle = np.zeros(9, dtype=np.int8)
    for _ in range(pad_before):
        env.step(idle)

    body = frames[skip_movie:]
    limit = min(len(body), max_frames)
    start_lives: int | None = None
    control_frame: int | None = None
    max_x = 0
    death_frame: int | None = None
    x_at: dict[int, int] = {}

    for i in range(limit):
        env.step(_action9(body[i]))
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=i + 1 + pad_before)
        if start_lives is None and int(snap.oper_mode) == 1 and 0 <= int(snap.lives) <= 8:
            # Prefer first frame with real on-foot state (post blackout).
            if int(snap.player_state) in (0x08, 0x00) and 0 < int(snap.player_x) < 200:
                start_lives = int(snap.lives)
                control_frame = i + 1 + pad_before
        if start_lives is None:
            continue
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
            if (i + 1) % 100 == 0:
                x_at[i + 1 + pad_before] = px
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death_frame = i + 1 + pad_before
            break
        # 1-1 flagpole region
        if px >= 3000:
            break

    env.close()
    return {
        "pad_before": pad_before,
        "skip_movie": skip_movie,
        "control_frame": control_frame,
        "max_player_x": max_x,
        "death_frame": death_frame,
        "x_samples": x_at,
        "survived_to_flag_zone": max_x >= 3000 and death_frame is None,
    }


def search_alignment(
    frames: list[list[int]],
    *,
    skip_range: range | None = None,
    pad_range: range | None = None,
    max_frames: int = 2500,
) -> dict[str, Any]:
    """Brute small pad/skip grid; rank by max 1-1 x (flag zone wins)."""
    skip_range = skip_range or range(0, 80, 5)
    pad_range = pad_range or range(0, 1, 1)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for pad in pad_range:
        for skip in skip_range:
            tr = _probe_11_progress(
                frames, max_frames=max_frames, pad_before=pad, skip_movie=skip
            )
            trials.append(tr)
            score = int(tr["max_player_x"])
            if tr.get("survived_to_flag_zone"):
                score += 100_000
            if best is None or score > int(best.get("_score", -1)):
                best = {**tr, "_score": score}
    return {"best": best, "trials": trials, "n_trials": len(trials)}


def _verify_poweron(
    frames: list[list[int]],
    *,
    max_frames: int | None = None,
    pad_idle: int = 200,
    pad_before: int = 0,
    skip_movie: int = 0,
) -> dict[str, Any]:
    """Replay frames from ``env.reset()``; report ending + milestones."""
    from retro_harness.env import make_env
    from smb.paths import GAME_DIR, GAME_V0
    from smb.ram import PLAYER_STATE_DYING

    # Power-on: NONE state (same as run_warp_finish --mode poweron).
    game_name = GAME_V0
    env = make_env(game_name, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()
    idle = np.zeros(9, dtype=np.int8)
    for _ in range(pad_before):
        env.step(idle)

    body = frames[skip_movie:]
    limit = len(body) if max_frames is None else min(len(body), max_frames)
    exits: list[dict[str, Any]] = []
    last_world: int | None = None
    last_level: int | None = None
    ending_frame: int | None = None
    death = False
    death_frame: int | None = None
    max_x = 0
    start_lives: int | None = None
    control_frame: int | None = None
    snap0 = read_snapshot(env.get_ram(), frame=0)

    for i in range(limit):
        # Do NOT sanitize L+R — TAS depends on simultaneous left+right.
        env.step(_action9(body[i]))
        ram = env.get_ram()
        abs_f = i + 1 + pad_before
        snap = read_snapshot(ram, frame=abs_f)
        # Ignore title/garbage RAM until oper_mode=playing and lives look real.
        if start_lives is None and int(snap.oper_mode) == 1 and 0 <= int(snap.lives) <= 8:
            if int(snap.player_state) in (0x08, 0x00) and 0 < int(snap.player_x) < 500:
                start_lives = int(snap.lives)
                control_frame = abs_f
        if start_lives is None:
            continue
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        w, lvl = int(snap.world), int(snap.level)
        if last_world is not None and (w, lvl) != (last_world, last_level):
            exits.append(
                {
                    "frame": abs_f,
                    "from": f"{last_world + 1}-{last_level + 1}",
                    "to": f"{w + 1}-{lvl + 1}",
                    "player_x": px,
                    "oper_mode": int(snap.oper_mode),
                }
            )
        last_world, last_level = w, lvl
        if reached_ending(ram, start_lives=start_lives):
            ending_frame = abs_f
            break
        if int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING:
            death = True
            death_frame = abs_f
            break

    # optional idle pad after movie ends (axe settle)
    if ending_frame is None and not death and start_lives is not None:
        for j in range(pad_idle):
            env.step(idle)
            ram = env.get_ram()
            if reached_ending(ram, start_lives=start_lives):
                ending_frame = limit + pad_before + j + 1
                break

    snap_end = read_snapshot(env.get_ram(), frame=ending_frame or limit + pad_before)
    env.close()

    report: dict[str, Any] = {
        "game_name": game_name,
        "movie_frames_played": limit,
        "pad_before": pad_before,
        "skip_movie": skip_movie,
        "ending_frame": ending_frame,
        "completed": ending_frame is not None,
        "death": death,
        "death_frame": death_frame,
        "exits": exits,
        "max_player_x": max_x,
        "start_lives": start_lives,
        "control_frame": control_frame,
        "start_snapshot": _snap_dict(snap0),
        "end_snapshot": _snap_dict(snap_end),
        "note": (
            "fceumm/stable-retro often desyncs FCEUX power-on movies; "
            "use --align-search then --skip-movie / --pad-before."
        ),
    }
    if ending_frame is not None:
        report["rta_note"] = (
            "ending_frame is power-on movie length (includes boot); "
            f"NTSC time ≈ {format_time(ending_frame, NTSC_FPS)}"
        )
        report["tasvideos_anchor_frames"] = 17_868
        report["delta_vs_happylee_1715"] = ending_frame - 17_868
    return report


def _snap_dict(snap: Any) -> dict[str, Any]:
    keys = (
        "world",
        "level",
        "level_id",
        "player_x",
        "player_y",
        "lives",
        "oper_mode",
        "player_state",
        "timer",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(snap, k):
            try:
                out[k] = int(getattr(snap, k))
            except (TypeError, ValueError):
                out[k] = getattr(snap, k)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "fm2",
        type=Path,
        nargs="?",
        default=DEFAULT_REF,
        help=f"path to .fm2 (default: {DEFAULT_REF})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write nes9_rle JSON seed",
    )
    p.add_argument(
        "--route-id",
        default="smb_fm2_import",
        help="route_id field in written seed",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="power-on replay under stable-retro and report ending",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="cap verify playback length",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write verify report JSON",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="print FM2 summary and exit (no write/verify)",
    )
    p.add_argument(
        "--align-search",
        action="store_true",
        help="grid-search skip_movie (and pad) for max 1-1 x under fceumm",
    )
    p.add_argument(
        "--skip-movie",
        type=int,
        default=0,
        help="drop this many leading movie frames before replay",
    )
    p.add_argument(
        "--pad-before",
        type=int,
        default=0,
        help="idle frames after env.reset before movie starts",
    )
    p.add_argument(
        "--skip-max",
        type=int,
        default=80,
        help="align-search: max skip_movie (step 5)",
    )
    p.add_argument(
        "--verify-1-2-slice",
        action="store_true",
        help=(
            "natural HL 1-1 → surface control → FM2 body → W4 "
            "(default indices from smb.tas.slice)"
        ),
    )
    p.add_argument(
        "--export-1-2-slice",
        action="store_true",
        help="write models/smb_1_2_happylee_slice.json from verified FM2 range",
    )
    p.add_argument(
        "--search-1-2",
        action="store_true",
        help="grid-search FM2 start indices for W4 after natural HL 1-1",
    )
    p.add_argument(
        "--1-2-start",
        type=int,
        default=None,
        dest="slice_1_2_start",
        help="FM2 start index for 1-2 body (default: verified constant)",
    )
    p.add_argument(
        "--1-2-start-min",
        type=int,
        default=2080,
        dest="slice_1_2_start_min",
    )
    p.add_argument(
        "--1-2-start-max",
        type=int,
        default=2140,
        dest="slice_1_2_start_max",
    )
    p.add_argument(
        "--verify-4-1-4-2-slice",
        action="store_true",
        help=(
            "natural HL chain → 4-1 control → FM2 4-1 → 4-2 control → FM2 → W8 "
            "(default indices from smb.tas.slice)"
        ),
    )
    p.add_argument(
        "--export-4-1-slice",
        action="store_true",
        help="write models/smb_4_1_happylee_slice.json from verified FM2 range",
    )
    p.add_argument(
        "--export-4-2-slice",
        action="store_true",
        help="write models/smb_4_2_happylee_slice.json from verified FM2 range",
    )
    p.add_argument(
        "--search-4-1",
        action="store_true",
        help="grid-search FM2 start indices for 4-1 clear after HL W4",
    )
    p.add_argument(
        "--search-4-2",
        action="store_true",
        help="grid-search FM2 start indices for W8 after HL 4-1",
    )
    p.add_argument(
        "--4-1-start",
        type=int,
        default=None,
        dest="slice_4_1_start",
        help="FM2 start index for 4-1 body (default: verified constant)",
    )
    p.add_argument(
        "--4-2-start",
        type=int,
        default=None,
        dest="slice_4_2_start",
        help="FM2 start index for 4-2 body (default: verified constant)",
    )
    p.add_argument(
        "--4-1-start-min",
        type=int,
        default=3880,
        dest="slice_4_1_start_min",
    )
    p.add_argument(
        "--4-1-start-max",
        type=int,
        default=4020,
        dest="slice_4_1_start_max",
    )
    p.add_argument(
        "--4-2-start-min",
        type=int,
        default=6100,
        dest="slice_4_2_start_min",
    )
    p.add_argument(
        "--4-2-start-max",
        type=int,
        default=6250,
        dest="slice_4_2_start_max",
    )
    p.add_argument(
        "--verify-8-1-8-2-slice",
        action="store_true",
        help="HL chain → W8 → 8-1 → 8-2 → 8-3 load (probe-verified indices)",
    )
    p.add_argument(
        "--export-8-1-slice",
        action="store_true",
        help="write models/smb_8_1_happylee_slice.json",
    )
    p.add_argument(
        "--export-8-2-slice",
        action="store_true",
        help="write models/smb_8_2_happylee_slice.json",
    )
    p.add_argument(
        "--search-8-3",
        action="store_true",
        help="grid-search FM2 starts for 8-3 → 8-4 leave after HL 8-2",
    )
    p.add_argument(
        "--verify-w8-tail",
        action="store_true",
        help="from 8-1 control, continuous FM2 tail until ending/death",
    )
    p.add_argument(
        "--8-1-start",
        type=int,
        default=None,
        dest="slice_8_1_start",
    )
    p.add_argument(
        "--8-2-start",
        type=int,
        default=None,
        dest="slice_8_2_start",
    )
    p.add_argument(
        "--8-3-start-min",
        type=int,
        default=13000,
        dest="slice_8_3_start_min",
    )
    p.add_argument(
        "--8-3-start-max",
        type=int,
        default=13600,
        dest="slice_8_3_start_max",
    )
    p.add_argument(
        "--8-3-step",
        type=int,
        default=1,
        dest="slice_8_3_step",
    )
    p.add_argument(
        "--8-3-lead-max",
        type=int,
        default=0,
        dest="slice_8_3_lead_max",
        help="also try 0..N lead idle frames before each 8-3 FM2 start",
    )
    args = p.parse_args(argv)

    fm2_path = args.fm2 if args.fm2.exists() else DEFAULT_REF

    # Control-relative slice ops (may not need full movie summary first)
    if args.verify_1_2_slice or args.export_1_2_slice or args.search_1_2:
        from smb.tas.slice import (
            HL_1_2_FM2_START,
            HL_1_2_W4_FRAMES,
            export_1_2_slice,
            search_1_2_offsets,
            verify_1_2_natural_chain,
        )

        start = args.slice_1_2_start if args.slice_1_2_start is not None else HL_1_2_FM2_START
        if args.search_1_2:

            def _progress(tr: Any) -> None:
                if tr.w4 or (tr.max_x or 0) > 500:
                    print(
                        f"  si={tr.start_idx} max_x={tr.max_x} "
                        f"w4={tr.w4} death={tr.death}",
                        flush=True,
                    )

            report = search_1_2_offsets(
                fm2_path=fm2_path,
                start_min=args.slice_1_2_start_min,
                start_max=args.slice_1_2_start_max,
                progress=_progress,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_1_2_offset_search.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("best") else 1

        if args.export_1_2_slice:
            # Prefer length from a quick verify when possible
            w4 = HL_1_2_W4_FRAMES
            payload = export_1_2_slice(
                fm2_path=fm2_path,
                start_idx=start,
                w4_frames=w4,
                out_path=args.out,
            )
            print(json.dumps({k: payload[k] for k in payload if k != "segments"}, indent=2))
            print(f"wrote {payload.get('_path')} frames={payload['num_frames']}", file=sys.stderr)

        if args.verify_1_2_slice:
            report = verify_1_2_natural_chain(
                fm2_path=fm2_path,
                start_idx=start,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_1_2_slice_verify.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("success") else 1

        return 0

    if (
        args.verify_4_1_4_2_slice
        or args.export_4_1_slice
        or args.export_4_2_slice
        or args.search_4_1
        or args.search_4_2
    ):
        from smb.tas.slice import (
            HL_4_1_FM2_START,
            HL_4_1_LEAVE_FRAMES,
            HL_4_2_FM2_START,
            HL_4_2_W8_FRAMES,
            export_4_1_slice,
            export_4_2_slice,
            search_4_1_offsets,
            search_4_2_offsets,
            verify_4_1_4_2_natural_chain,
        )

        si41 = (
            args.slice_4_1_start
            if args.slice_4_1_start is not None
            else HL_4_1_FM2_START
        )
        si42 = (
            args.slice_4_2_start
            if args.slice_4_2_start is not None
            else HL_4_2_FM2_START
        )

        if args.search_4_1:

            def _p41(tr: Any) -> None:
                if tr.w4 or (tr.max_x or 0) > 800:
                    print(
                        f"  41 si={tr.start_idx} max_x={tr.max_x} "
                        f"leave={tr.w4} death={tr.death}",
                        flush=True,
                    )

            report = search_4_1_offsets(
                fm2_path=fm2_path,
                start_min=args.slice_4_1_start_min,
                start_max=args.slice_4_1_start_max,
                progress=_p41,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_4_1_offset_search.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("best") else 1

        if args.search_4_2:

            def _p42(tr: Any) -> None:
                if tr.w4 or (tr.max_x or 0) > 400:
                    print(
                        f"  42 si={tr.start_idx} max_x={tr.max_x} "
                        f"w8={tr.w4} ug={tr.ug} death={tr.death}",
                        flush=True,
                    )

            report = search_4_2_offsets(
                fm2_path=fm2_path,
                start_4_1=si41,
                start_min=args.slice_4_2_start_min,
                start_max=args.slice_4_2_start_max,
                progress=_p42,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_4_2_offset_search.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("best") else 1

        if args.export_4_1_slice:
            payload = export_4_1_slice(
                fm2_path=fm2_path,
                start_idx=si41,
                leave_frames=HL_4_1_LEAVE_FRAMES,
                out_path=args.out if args.export_4_1_slice and not args.export_4_2_slice else None,
            )
            print(
                json.dumps(
                    {k: payload[k] for k in payload if k != "segments"}, indent=2
                )
            )
            print(
                f"wrote {payload.get('_path')} frames={payload['num_frames']}",
                file=sys.stderr,
            )

        if args.export_4_2_slice:
            out42 = args.out if args.export_4_2_slice and not args.export_4_1_slice else None
            payload = export_4_2_slice(
                fm2_path=fm2_path,
                start_idx=si42,
                w8_frames=HL_4_2_W8_FRAMES,
                out_path=out42,
            )
            print(
                json.dumps(
                    {k: payload[k] for k in payload if k != "segments"}, indent=2
                )
            )
            print(
                f"wrote {payload.get('_path')} frames={payload['num_frames']}",
                file=sys.stderr,
            )

        if args.verify_4_1_4_2_slice:
            report = verify_4_1_4_2_natural_chain(
                fm2_path=fm2_path,
                start_4_1=si41,
                start_4_2=si42,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_4_1_4_2_slice_verify.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("success") else 1

        return 0

    if (
        args.verify_8_1_8_2_slice
        or args.export_8_1_slice
        or args.export_8_2_slice
        or args.search_8_3
        or args.verify_w8_tail
    ):
        from smb.tas.slice import (
            HL_8_1_FM2_START,
            HL_8_2_FM2_START,
            export_8_1_slice,
            export_8_2_slice,
            search_8_3_offsets,
            verify_8_1_8_2_natural_chain,
            verify_continuous_tail_from_8_1,
        )

        si81 = (
            args.slice_8_1_start
            if args.slice_8_1_start is not None
            else HL_8_1_FM2_START
        )
        si82 = (
            args.slice_8_2_start
            if args.slice_8_2_start is not None
            else HL_8_2_FM2_START
        )

        if args.export_8_1_slice:
            payload = export_8_1_slice(fm2_path=fm2_path, start_idx=si81)
            print(
                json.dumps(
                    {k: payload[k] for k in payload if k != "segments"}, indent=2
                )
            )
            print(
                f"wrote {payload.get('_path')} frames={payload['num_frames']}",
                file=sys.stderr,
            )

        if args.export_8_2_slice:
            payload = export_8_2_slice(fm2_path=fm2_path, start_idx=si82)
            print(
                json.dumps(
                    {k: payload[k] for k in payload if k != "segments"}, indent=2
                )
            )
            print(
                f"wrote {payload.get('_path')} frames={payload['num_frames']}",
                file=sys.stderr,
            )

        if args.verify_8_1_8_2_slice:
            report = verify_8_1_8_2_natural_chain(
                fm2_path=fm2_path,
                start_8_1=si81,
                start_8_2=si82,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_8_1_8_2_slice_verify.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("success") else 1

        if args.verify_w8_tail:
            report = verify_continuous_tail_from_8_1(
                fm2_path=fm2_path,
                start_idx=si81,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_w8_tail_verify.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("success") else 1

        if args.search_8_3:

            def _p83(tr: Any) -> None:
                if tr.w4 or (tr.max_x or 0) > 600:
                    print(
                        f"  83 si={tr.start_idx} max_x={tr.max_x} "
                        f"leave={tr.w4} death={tr.death}",
                        flush=True,
                    )

            report = search_8_3_offsets(
                fm2_path=fm2_path,
                start_8_1=si81,
                start_8_2=si82,
                start_min=args.slice_8_3_start_min,
                start_max=args.slice_8_3_start_max,
                step=max(1, args.slice_8_3_step),
                lead_idles=range(0, max(0, args.slice_8_3_lead_max) + 1),
                progress=_p83,
            )
            print(json.dumps(report, indent=2))
            rp = args.report or (
                RECORDINGS_DIR / "tas_import" / "happylee_8_3_offset_search.json"
            )
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {rp}", file=sys.stderr)
            return 0 if report.get("n_hits") else 1

        return 0

    if not args.fm2.exists():
        print(f"missing fm2: {args.fm2}", file=sys.stderr)
        return 2

    movie = parse_fm2(args.fm2)
    summary = movie.summary()
    print(json.dumps(summary, indent=2))

    if args.summary_only:
        return 0

    frames = movie.frames
    out = args.out
    if out is None and not args.verify and not args.align_search:
        out = MODELS_DIR / f"{args.route_id}.json"

    if out is not None:
        payload = frames_to_nes9_rle_payload(
            frames,
            route_id=args.route_id,
            source=str(args.fm2),
            extra={
                "fm2_summary": summary,
                "start_state": "power_on",
                "settle_frames": 0,
                "note": (
                    "Raw FM2 import including boot/title frames. "
                    "Do not sanitize L+R. Verify with --verify before promote. "
                    "fceumm may need --skip-movie / --pad-before alignment."
                ),
            },
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} ({payload['num_frames']} frames)", file=sys.stderr)

    if args.align_search:
        align = search_alignment(
            frames,
            skip_range=range(0, max(1, args.skip_max) + 1, 5),
            pad_range=range(0, 1, 1),
            max_frames=args.max_frames or 2500,
        )
        print(json.dumps(align, indent=2))
        report_path = args.report or (
            RECORDINGS_DIR / "tas_import" / f"{args.fm2.stem}_align.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(align, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {report_path}", file=sys.stderr)
        return 0

    if args.verify:
        report = _verify_poweron(
            frames,
            max_frames=args.max_frames,
            pad_before=args.pad_before,
            skip_movie=args.skip_movie,
        )
        print(json.dumps(report, indent=2))
        report_path = args.report
        if report_path is None:
            report_path = (
                RECORDINGS_DIR / "tas_import" / f"{args.fm2.stem}_verify.json"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {report_path}", file=sys.stderr)
        return 0 if report.get("completed") else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
