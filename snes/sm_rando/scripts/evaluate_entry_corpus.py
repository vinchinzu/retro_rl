"""Evaluate the real vanilla Landing policy on train and held-out states."""

from __future__ import annotations

from sm_rando.entry_corpus import (
    LANDING_BASELINE_REPORT,
    evaluate_structured_landing_baseline,
)


def main() -> int:
    report = evaluate_structured_landing_baseline()
    metrics = report["metrics"]
    print(
        f"train={metrics['train']['successes']}/{metrics['train']['attempts']} "
        f"eval={metrics['eval']['successes']}/{metrics['eval']['attempts']} "
        f"gap={metrics['generalization_gap']:+.3f} "
        f"report={LANDING_BASELINE_REPORT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
