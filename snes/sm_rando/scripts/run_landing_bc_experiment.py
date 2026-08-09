"""Train and held-out evaluate the Landing wait-to-handoff behavior clone."""

from __future__ import annotations

from sm_rando.landing_bc import LANDING_BC_REPORT, run_landing_bc_experiment


def main() -> int:
    report = run_landing_bc_experiment()
    metrics = report["metrics"]
    print(
        f"train={metrics['train']['successes']}/{metrics['train']['attempts']} "
        f"eval={metrics['eval']['successes']}/{metrics['eval']['attempts']} "
        f"gap={metrics['generalization_gap']:+.3f} "
        f"baseline={report['structured_baseline_eval_rate']:.3f} "
        f"report={LANDING_BC_REPORT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
