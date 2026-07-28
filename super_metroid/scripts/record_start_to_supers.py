#!/usr/bin/env python3
"""Record continuous power-on through Spore Super Missile collect.

```bash
uv run python super_metroid/scripts/record_start_to_supers.py --no-video
uv run python super_metroid/scripts/record_start_to_supers.py
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.start_to_supers import (  # noqa: E402
    default_artifact_paths,
    run_start_to_supers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_video, default_report = default_artifact_paths()
    parser.add_argument("--video", type=Path, default=default_video)
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-unlimited-energy", action="store_true")
    parser.add_argument("--no-unlimited-ammo", action="store_true")
    args = parser.parse_args()
    report = run_start_to_supers(
        video_path=None if args.no_video else args.video,
        report_path=args.report,
        unlimited_energy=not args.no_unlimited_energy,
        unlimited_ammo=not args.no_unlimited_ammo,
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
