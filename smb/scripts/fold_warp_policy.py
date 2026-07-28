"""Fold the successful checkpoint-practice path into one replay seed.

The source session used save states while discovering the route. Each selected
branch fragment begins at the exact state produced by the preceding fragment,
so concatenating the controller frames removes all of those practice reloads.
The resulting policy has one published start state (mid-1-2) and no state load
through the World 8-4 ending.

Run after relocating or replacing the source practice session:

    uv run python -m smb.scripts.fold_warp_policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smb.paths import FULLGAME_RECORDINGS_DIR
from smb.policy import DEFAULT_WARP_SUFFIX_SEED, compress_nes9_rle

SOURCE_SESSION = "20260429_214207"
EXPECTED_FRAMES = 19_963

# (branch id, exclusive raw_buttons end, milestone reached by/within fragment)
SOURCE_FRAGMENTS: tuple[tuple[int, int, str], ...] = (
    (6, 5199, "1-2 warp, 4-1, late 4-2"),
    (7, 2458, "World 8 entry and 8-1 x=284"),
    (45, 419, "8-1 x=1213"),
    (53, 3085, "8-1 exit and 8-2 natural entry"),
    (56, 589, "8-2 x=1303"),
    (62, 1413, "8-2 x=2964"),
    (76, 1168, "8-2 exit and 8-3 natural entry"),
    (77, 311, "8-3 x=713"),
    (82, 229, "8-3 x=1274"),
    (85, 390, "8-3 x=2179"),
    (97, 3270, "8-3 exit and first four 8-4 areas"),
    (100, 1053, "8-4 final exterior x=4152"),
    (108, 379, "8-4 ending"),
)


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
            f"{path} has {len(raw)} raw frames; fragment requires {end}"
        )
    return [[int(b) for b in frame[:9]] for frame in raw[:end]]


def build_folded_seed(
    *,
    recordings_dir: Path = FULLGAME_RECORDINGS_DIR,
    session_id: str = SOURCE_SESSION,
) -> dict[str, Any]:
    """Return the self-contained RLE seed assembled from source branches."""
    session_dir = recordings_dir / session_id
    frames: list[list[int]] = []
    fragment_rows: list[dict[str, Any]] = []
    for branch_id, end, milestone in SOURCE_FRAGMENTS:
        fragment = _load_branch_frames(session_dir, branch_id, end)
        frames.extend(fragment)
        fragment_rows.append(
            {
                "branch_id": branch_id,
                "start": 0,
                "end": end,
                "frames": len(fragment),
                "milestone": milestone,
            }
        )

    if len(frames) != EXPECTED_FRAMES:
        raise ValueError(
            f"folded path has {len(frames)} frames; expected {EXPECTED_FRAMES}"
        )

    return {
        "format": "nes9_rle",
        "route_id": "smb_warp_any_percent",
        "level_id": "smb_1_2_to_ending",
        "start_state": "Level1_2_WarpMid",
        "game_name": "SuperMarioBros-Nes-v0",
        "num_frames": len(frames),
        "verified_completed": True,
        "target": "world_8_4_ending",
        "source": (
            f"folded controller-only suffix from session {session_id}; "
            "practice state loads removed"
        ),
        "source_session": session_id,
        "source_fragments": fragment_rows,
        "verification": {
            "start_lives": 2,
            "final_lives": 2,
            "deaths": 0,
            "final_world": 7,
            "final_level": 3,
            "final_oper_mode": 2,
            "ending_stable_idle_frames": 180,
        },
        "notes": (
            "Development suffix starts at the disclosed mid-1-2 state. "
            "After that initial state, all progress through 8-4 uses controller "
            "input with no emulator-state reload."
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
        "--output",
        type=Path,
        default=DEFAULT_WARP_SUFFIX_SEED,
    )
    args = parser.parse_args()

    seed = build_folded_seed(
        recordings_dir=args.recordings_dir,
        session_id=args.session,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Compact encoding keeps the 20k-frame RLE policy reviewable in git.
    args.output.write_text(
        json.dumps(seed, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: {seed['num_frames']} frames, "
        f"{len(seed['segments'])} RLE segments"
    )


if __name__ == "__main__":
    main()
