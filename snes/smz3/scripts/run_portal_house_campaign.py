"""Run the SMZ3 multi-seed portal→house S/T campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smz3.portal_house_campaign import (  # noqa: E402
    CAMPAIGN_LEDGER,
    CAMPAIGN_REPORT,
    CLASSIC_REPORT,
    campaign_summary,
    run_portal_house_campaign,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dry", "live"),
        default="dry",
        help="dry = audited fixture harness dry-run; live = real SMZ3-Snes tip",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Published seed ids (default: 1337 1338 1339)",
    )
    parser.add_argument(
        "--success-threshold",
        type=int,
        default=None,
        help="S in S/T (default: 2, or seed count if smaller)",
    )
    parser.add_argument("--budget", type=int, default=None, help="Frame budget per seed")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=CAMPAIGN_LEDGER,
        help="Atomic campaign ledger path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=CAMPAIGN_REPORT,
        help="Campaign report JSON path",
    )
    parser.add_argument(
        "--classic-report",
        type=Path,
        default=CLASSIC_REPORT,
        help="Classic seed_robustness_report projection (claimable only)",
    )
    parser.add_argument(
        "--no-classic",
        action="store_true",
        help="Skip classic seed-robustness projection even when claimable",
    )
    parser.add_argument(
        "--publish-docs",
        action="store_true",
        help="Also write committed dry report under snes/smz3/docs/",
    )
    parser.add_argument(
        "--seeds-root",
        type=Path,
        default=None,
        help="Directory for fixture seed packages (default: snes/smz3/seeds)",
    )
    args = parser.parse_args(argv)

    result = run_portal_house_campaign(
        mode=args.mode,
        seeds=args.seeds,
        success_threshold=args.success_threshold,
        budget=args.budget,
        seeds_root=args.seeds_root,
        ledger_path=args.ledger,
        report_path=args.report,
        classic_report_path=None if args.no_classic else args.classic_report,
        write_classic=not args.no_classic,
        publish_docs_report=args.publish_docs,
    )
    summary = campaign_summary(result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if result.claimable and result.threshold_met:
        print(
            f"[GREEN] S/T threshold met: {result.successes}/{result.config.seed_count} "
            f"(required {result.config.success_threshold})"
        )
        return 0
    if result.claimable:
        print(
            f"[YELLOW] claimable but below threshold: "
            f"{result.successes}/{result.config.seed_count} "
            f"(required {result.config.success_threshold})"
        )
        return 1
    print(
        f"[RED] non-claimable campaign "
        f"(infra_errors={result.infra_error_count}, "
        f"successes={result.successes}/{result.config.seed_count})"
    )
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
