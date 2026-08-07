#!/usr/bin/env python3
"""Extract rough button timelines from a Super Metroid reference VOD.

Thin wrapper around ``yt_ref.py extract`` / ``yt_ref_lib.extract_buttons``.

Prefer the unified CLI::

    uv run python snes/super_metroid/scripts/tools/yt_ref.py chunk \\
      --start 1338 --end 1351 --name moat_shinespark --spark

Legacy direct form (still works)::

    uv run python snes/super_metroid/scripts/tools/yt_input_extract.py \\
      --start 200 --end 210 -o smoke_morph10s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_SNES = Path(__file__).resolve().parents[3]
_TOOLS = Path(__file__).resolve().parent
for _p in (ROOT, _SNES, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import YT_DEFAULT_REF_ID  # noqa: E402
import yt_ref_lib as lib  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=YT_DEFAULT_REF_ID, help="Video id (default Kentroid KPDR)")
    ap.add_argument("--video", type=Path, default=None, help="Override video path")
    ap.add_argument("--layout", type=Path, default=None, help="Override layout.json")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--segments", type=Path, default=None)
    ap.add_argument("--segment-id", type=str, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("-o", "--out", type=Path, required=True, help="Output stem")
    args = ap.parse_args()

    ws = lib.RefWorkspace.resolve(args.ref)
    layout = (
        json_load(args.layout)
        if args.layout
        else ws.load_layout()
    )
    video = args.video or ws.video_path
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")

    if args.segment_id:
        start, end, seg = ws.resolve_segment(args.segment_id, args.segments)
    else:
        if args.start is None or args.end is None:
            raise SystemExit("need --start/--end or --segment-id")
        start, end = lib.parse_time_token(args.start), lib.parse_time_token(args.end)
        seg = None
    if end <= start:
        raise SystemExit("end must be > start")

    result = lib.extract_buttons(
        video, layout, start_s=start, end_s=end, stride=max(1, args.stride)
    )
    if seg:
        result["segment"] = {
            "id": seg.get("id"),
            "label": seg.get("label"),
            "project_tip": seg.get("project_tip"),
        }
    paths = lib.write_extract_outputs(result, args.out)
    duty = lib.duty_cycle(result["frames"], result["button_order"])
    print(f"[yt_input_extract] samples={result['n_samples']} → {paths['json']}", flush=True)
    print(f"[yt_input_extract] edges={len(result['press_events'])} → {paths['edges']}", flush=True)
    print(f"[yt_input_extract] duty_cycle={duty}", flush=True)
    sys.exit(0)


def json_load(path: Path) -> dict:
    import json

    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
