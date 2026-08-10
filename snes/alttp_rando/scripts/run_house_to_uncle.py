"""Record FirstPlay → uncle fighter sword on ALTTPRando-Snes and write evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from alttp_rando.house_to_uncle import (
    HOUSE_TO_UNCLE_EVIDENCE,
    HOUSE_TO_UNCLE_REPORT,
    run_house_to_uncle_from_first_play,
    write_evidence_sidecar,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=HOUSE_TO_UNCLE_REPORT,
        help="JSON report path under recordings/",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=HOUSE_TO_UNCLE_EVIDENCE,
        help="Evidence sidecar path under recordings/",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip writing the evidence sidecar (report only).",
    )
    args = parser.parse_args(argv)

    report = run_house_to_uncle_from_first_play(report_path=args.report)
    final = report.segment.snapshot
    tag = "GREEN" if report.success else "RED"
    print(
        f"[{tag}] outcome={report.outcome} frames={report.total_frames} "
        f"room=0x{final.room_base_id:02X} sword={final.has_fighter_sword} "
        f"clean={report.to_dict().get('clean_chain')}"
    )
    print(f"  report: {args.report}")
    if not report.success:
        print(f"  blocker: {report.segment.blocker}")
        return 1
    if not args.no_evidence:
        payload = write_evidence_sidecar(
            report_path=args.report,
            evidence_path=args.evidence,
        )
        print(f"  evidence: {args.evidence}")
        print(f"  digest: {payload['source_report_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
