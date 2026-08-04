"""Fold and optimize the continuous Level1_1-to-ending controller seed.

Removes the mid-1-2 state splice. The resulting policy starts at published
``Level1_1`` after a fixed idle phase-align (14 frames), then runs controller-
only through 1-1, 1-2 warp, 4-1, 4-2, and all World 8 exits to the 8-4 ending.

Source:

- Prelude: session ``20260429_214207`` branch 5 frames ``[0:2042]`` (1-1 clear
  through the mid-1-2 transition that matches the folded suffix phase).
- Baseline suffix: ``smb_warp_mid_to_ending.json`` (19,963 frames).
- Fast 4-2 fragment: ``smb_4_2_fast_w8.json`` (2,375 frames), verified from
  the real predecessor state.
- Deterministic natural-entry splices remove the accidental 1-2 Start pause
  and phase-align each later level without emulator-state loads.

```bash
uv run python -m smb.scripts.fold_continuous_policy
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smb.paths import FULLGAME_RECORDINGS_DIR
from smb.policy import (
    DEFAULT_CONTINUOUS_SEED,
    DEFAULT_FAST_4_2_SEED,
    DEFAULT_WARP_SUFFIX_SEED,
    compress_nes9_rle,
    expand_nes9_rle,
    load_nes9_rle_seed,
)

SOURCE_SESSION = "20260429_214207"
PRELUDE_BRANCH_ID = 5
PRELUDE_END = 2042
# Idle frames after Level1_1 load before replaying the continuous seed.
# Verified: settle 13/14/34/35 reach World 4; 14 matches suffix frame count.
CHAIN_SETTLE_FRAMES = 14
EXPECTED_PRELUDE = 2042
EXPECTED_SUFFIX = 19_963
EXPECTED_BASELINE = EXPECTED_PRELUDE + EXPECTED_SUFFIX
EXPECTED_FAST_4_2 = 2_375
EXPECTED_TOTAL = 21_731

# Frame-local first-pipe repair.  The raw 20260429 prelude wall-slides into
# the pipe; this replacement leaves the preceding state untouched, lands on
# the pipe lip at speed, and rejoins the historical DOWN-enter tail at frame
# 468.  Keep this source-owned rather than relying on the already-published
# seed as an implicit patch: ``build_continuous_seed`` must be reproducible.
PIPE_FIX_START = 310
PIPE_FIX_END = 468
PIPE_FIX_SEGMENTS = (
    ({"b": [1, 0, 0, 0, 0, 0, 0, 1, 0], "n": 2}),
    ({"b": [1, 0, 0, 0, 0, 0, 0, 1, 1], "n": 50}),
    ({"b": [0, 0, 0, 0, 0, 0, 0, 1, 0], "n": 9}),
    ({"b": [0, 0, 0, 0, 0, 0, 1, 0, 0], "n": 1}),
    ({"b": [0, 0, 0, 0, 0, 0, 0, 0, 0], "n": 96}),
)
PIPE_FIX_FRAMES = expand_nes9_rle(
    {"format": "nes9_rle", "segments": list(PIPE_FIX_SEGMENTS)}
)

# Zero-based, exclusive slice anchors in the 22,005-frame baseline. These
# boundaries were captured at natural level-entry states, then the complete
# result was re-verified from Level1_1 and power-on.
PAUSE_PREFIX_END = 3_981
PAUSE_RESUME_START = 4_278
WORLD4_CONTROL_END = 4_560
WORLD4_ACTION_START = 4_619
WORLD42_CONTROL_END = 7_264
# The robust fragment and baseline reach identical player physics here:
# optimized route frame 7,854 == baseline frame 8,059. Switching input tails
# avoids the repeated vine attempts and reaches World 8 153 frames sooner.
FAST_42_PREFIX_END = 946
BASELINE_42_TAIL_START = 8_059
BASELINE_42_TAIL_END = 9_335
WORLD8_TRANSITION_IDLE = 219
WORLD8_ACTION_START = 9_575
WORLD82_CONTROL_END = 13_188
WORLD82_PHASE_IDLE = 5
WORLD82_ACTION_START = 13_205
WORLD83_CONTROL_END = 16_352
WORLD83_PHASE_IDLE = 15
WORLD83_ACTION_START = 16_375
WORLD84_CONTROL_END = 18_572
WORLD84_PHASE_IDLE = 3
WORLD84_ACTION_START = 18_603


def _load_branch_frames(
    session_dir: Path,
    branch_id: int,
    end: int,
) -> list[list[int]]:
    path = session_dir / "branches" / f"branch_{branch_id:03d}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("raw_buttons") or []
    if len(raw) < end:
        raise ValueError(
            f"{path} has {len(raw)} raw frames; prelude requires {end}"
        )
    return [[int(b) for b in frame[:9]] for frame in raw[:end]]


def build_continuous_seed(
    *,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    session_id: str = SOURCE_SESSION,
    suffix_seed: Path = DEFAULT_WARP_SUFFIX_SEED,
    fast_4_2_seed: Path = DEFAULT_FAST_4_2_SEED,
) -> dict[str, Any]:
    """Return the continuous Level1_1→ending RLE seed."""
    session_dir = recordings_dir / session_id
    prelude = _load_branch_frames(session_dir, PRELUDE_BRANCH_ID, PRELUDE_END)
    if len(prelude) != EXPECTED_PRELUDE:
        raise ValueError(
            f"prelude has {len(prelude)} frames; expected {EXPECTED_PRELUDE}"
        )
    if len(PIPE_FIX_FRAMES) != PIPE_FIX_END - PIPE_FIX_START:
        raise ValueError("pipe-fix fragment does not cover its declared window")
    prelude[PIPE_FIX_START:PIPE_FIX_END] = PIPE_FIX_FRAMES

    suffix_data = load_nes9_rle_seed(suffix_seed)
    suffix = expand_nes9_rle(suffix_data)
    if len(suffix) != EXPECTED_SUFFIX:
        raise ValueError(
            f"suffix has {len(suffix)} frames; expected {EXPECTED_SUFFIX}"
        )

    baseline = prelude + suffix
    if len(baseline) != EXPECTED_BASELINE:
        raise ValueError(
            f"baseline path has {len(baseline)} frames; "
            f"expected {EXPECTED_BASELINE}"
        )

    fast_4_2_data = load_nes9_rle_seed(fast_4_2_seed)
    fast_4_2 = expand_nes9_rle(fast_4_2_data)
    if len(fast_4_2) != EXPECTED_FAST_4_2:
        raise ValueError(
            f"fast 4-2 fragment has {len(fast_4_2)} frames; "
            f"expected {EXPECTED_FAST_4_2}"
        )

    idle = [[0] * 9]
    frames = (
        baseline[:PAUSE_PREFIX_END]
        + baseline[PAUSE_RESUME_START:WORLD4_CONTROL_END]
        + baseline[WORLD4_ACTION_START:WORLD42_CONTROL_END]
        + fast_4_2[:FAST_42_PREFIX_END]
        + baseline[BASELINE_42_TAIL_START:BASELINE_42_TAIL_END]
        + idle * WORLD8_TRANSITION_IDLE
        + baseline[WORLD8_ACTION_START:WORLD82_CONTROL_END]
        + idle * WORLD82_PHASE_IDLE
        + baseline[WORLD82_ACTION_START:WORLD83_CONTROL_END]
        + idle * WORLD83_PHASE_IDLE
        + baseline[WORLD83_ACTION_START:WORLD84_CONTROL_END]
        + idle * WORLD84_PHASE_IDLE
        + baseline[WORLD84_ACTION_START:]
    )
    if len(frames) != EXPECTED_TOTAL:
        raise ValueError(
            f"optimized path has {len(frames)} frames; expected {EXPECTED_TOTAL}"
        )

    return {
        "format": "nes9_rle",
        "route_id": "smb_warp_any_percent",
        "level_id": "smb_1_1_to_ending",
        "start_state": "Level1_1",
        "settle_frames": CHAIN_SETTLE_FRAMES,
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(frames),
        "verified_completed": True,
        "target": "world_8_4_ending",
        "source": (
            f"optimized continuous controller path: session {session_id} "
            f"branch {PRELUDE_BRANCH_ID}[:{PRELUDE_END}] + folded warp "
            "suffix + natural-entry 4-2 fragment; accidental 1-2 pause "
            "removed; no mid-attempt state load"
        ),
        "source_session": session_id,
        "source_fragments": [
            {
                "branch_id": PRELUDE_BRANCH_ID,
                "start": 0,
                "end": PRELUDE_END,
                "frames": len(prelude),
                "milestone": "1-1 clear through mid-1-2 transition",
            },
            {
                "seed": str(suffix_seed.name),
                "frames": len(suffix),
                "milestone": "baseline 1-2 warp through 8-4 ending",
            },
            {
                "seed": str(fast_4_2_seed.name),
                "frames": len(fast_4_2),
                "milestone": "natural-entry 4-2 underground through World 8",
            },
        ],
        "optimization": {
            "baseline_frames": EXPECTED_BASELINE,
            "optimized_frames": len(frames),
            "frames_saved": EXPECTED_BASELINE - len(frames),
            "changes": [
                "removed accidental Start pause in 1-2",
                (
                    "replaced 4-2 approach with a natural-entry hybrid at an "
                    "identical player-physics state"
                ),
                "phase-aligned World 8 level entries with controller idle only",
                (
                    "1-1 first pipe: top-land at full speed (js=312,jh=50), "
                    "coast+brake, stand on pipe top, rejoin original DOWN enter "
                    "at frame 468 (eliminates x=898 wall-slide)"
                ),
            ],
            "pipe_fix": {
                "js": 312,
                "jh": 50,
                "dd_air": 9,
                "brake_left": 1,
                "rejoin": 468,
                "keep_until": 300,
                "behavior": "land_on_top_brake_stand_rejoin_original_enter",
            },
        },
        "verification": {
            "start_lives": 2,
            "final_lives": 2,
            "deaths": 0,
            "final_world": 7,
            "final_level": 3,
            "final_oper_mode": 2,
            "ending_stable_idle_frames": 120,
            "mid_attempt_state_loads": 0,
        },
        "notes": (
            "Published start is Level1_1. After the documented settle frames, "
            "all progress through the 8-4 ending is controller input with no "
            "emulator-state reload. The same seed is power-on Clean with the "
            "documented fixed boot and settle phases. First pipe: lands on top "
            "near x=920 (no side-hit at x=898), brakes, DOWN-enters via original "
            "phase-aligned tail."
        ),
        "segments": compress_nes9_rle(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=FULLGAME_RECORDINGS_DIR,
    )
    parser.add_argument("--session", default=SOURCE_SESSION)
    parser.add_argument(
        "--suffix-seed",
        type=Path,
        default=DEFAULT_WARP_SUFFIX_SEED,
    )
    parser.add_argument(
        "--fast-4-2-seed",
        type=Path,
        default=DEFAULT_FAST_4_2_SEED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONTINUOUS_SEED,
    )
    args = parser.parse_args()

    seed = build_continuous_seed(
        recordings_dir=args.recordings_dir,
        session_id=args.session,
        suffix_seed=args.suffix_seed,
        fast_4_2_seed=args.fast_4_2_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(seed, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {seed['num_frames']} frames, "
        f"{len(seed['segments'])} RLE segments, "
        f"settle={seed['settle_frames']}"
    )


if __name__ == "__main__":
    main()
