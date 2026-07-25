"""Record and verify the continuous power-on through Bomb Torizo proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.start_to_bombs import (  # noqa: E402
    default_artifact_paths,
    run_start_to_bombs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_video, default_report = default_artifact_paths()
    parser.add_argument("--video", type=Path, default=default_video)
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-unlimited-ammo", action="store_true")
    args = parser.parse_args()
    report = run_start_to_bombs(
        video_path=None if args.no_video else args.video,
        report_path=args.report,
        unlimited_ammo=not args.no_unlimited_ammo,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
