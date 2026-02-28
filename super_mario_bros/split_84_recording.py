"""Split the 8-4 full recording into per-segment recordings.

Uses the same transition frames as extract_84_states.py.

Usage:
    uv run python super_mario_bros/split_84_recording.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "optimizer" / "runs"
RECORDING = RUNS / "smb_8_4" / "recording_001.json"
RAW_RECORDING = RUNS / "smb_8_4" / "recording_001_raw.json"

# Segment boundaries: (seg_num, start_frame, end_frame_exclusive)
# Based on transition frames from trace and extract_84_states.py
SEGMENTS = [
    (1, 0, 790),       # Castle start, state starts at frame 0
    (2, 790, 1125),    # Pipe section, state at frame 790
    (3, 1125, 1895),   # Castle maze, state at frame 1125
    (4, 1895, 3030),   # Underwater, state at frame 1895
    (5, 3030, None),   # Final castle + Bowser, state at frame 3030
]


def main():
    with open(RECORDING) as f:
        data = json.load(f)
    actions = data["actions"]

    with open(RAW_RECORDING) as f:
        raw_data = json.load(f)
    raw_buttons = raw_data["raw_buttons"]
    raw_pre_sanitize = raw_data.get("raw_buttons_pre_sanitize")

    print(f"Full recording: {len(actions)} frames")
    if raw_pre_sanitize is not None and len(raw_pre_sanitize) != len(raw_buttons):
        print(
            "Warning: raw_buttons_pre_sanitize length mismatch; "
            "skipping pre-sanitize split export."
        )
        raw_pre_sanitize = None

    for seg, start, end in SEGMENTS:
        seg_dir = RUNS / f"smb_8_4_{seg}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        seg_actions = actions[start:end]
        seg_raw = raw_buttons[start:end]
        seg_raw_pre = (
            raw_pre_sanitize[start:end] if raw_pre_sanitize is not None else None
        )

        # Save action-index recording
        out = seg_dir / "recording_000.json"
        with open(out, "w") as f:
            json.dump({
                "actions": seg_actions,
                "metadata": {
                    "level": f"smb_8_4_{seg}",
                    "source": "split_from_8_4_full",
                    "total_frames": len(seg_actions),
                    "parent_start_frame": start,
                },
            }, f)

        # Save raw recording
        raw_out = seg_dir / "recording_000_raw.json"
        raw_payload = {
            "raw_buttons": seg_raw,
            "actions": seg_actions,
            "metadata": {
                "level": f"smb_8_4_{seg}",
                "source": "split_from_8_4_full",
                "total_frames": len(seg_raw),
            },
        }
        if seg_raw_pre is not None:
            raw_payload["raw_buttons_pre_sanitize"] = seg_raw_pre
        with open(raw_out, "w") as f:
            json.dump(raw_payload, f)

        print(f"  seg{seg}: frames {start}-{end or len(actions)} ({len(seg_actions)} frames) -> {out}")

    print("\nDone! Verify with:")
    for seg, _, _ in SEGMENTS:
        print(f"  uv run python -m super_mario_bros.optimizer -l smb_8_4_{seg} verify --actions {RUNS}/smb_8_4_{seg}/recording_000.json")


if __name__ == "__main__":
    main()
