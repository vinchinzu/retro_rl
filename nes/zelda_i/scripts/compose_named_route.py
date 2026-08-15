"""Print a would-run plan for a published Zelda I NamedRoute.

No path logic and no emulator: bind existing L1/L2 controllers, optionally
classify ``--from-state`` via route_eligible, and write a JSON report.

Examples::

    uv run python nes/zelda_i/scripts/compose_named_route.py --route level1_complete
    uv run python nes/zelda_i/scripts/compose_named_route.py \\
        --route level2_prefix --from-state Level1ExitOverworld
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zelda_i.paths import INTEGRATION_DIR
from zelda_i.route_composer import describe_named_route
from zelda_i.route_eligible import classify
from zelda_i.runner import add_common_args, write_report


def compose_report(
    *,
    route: str,
    from_state: str | None = None,
    infinite_life: bool = False,
    tag: str = "compose",
) -> dict:
    """Env+assist+report wrapper around the composer plan (no emulator)."""
    plan = describe_named_route(route)
    payload: dict = {
        "ok": plan["unbound_count"] == 0,
        "would_run": plan["unbound_count"] == 0,
        "route": route,
        "infinite_life": infinite_life,
        "from_state": from_state,
        "plan": plan,
        "assist": {"enabled": infinite_life, "progression_writes": 0},
    }
    if from_state:
        sidecar = Path(INTEGRATION_DIR) / f"{from_state}.provenance.json"
        verdict = classify(from_state, sidecar if sidecar.is_file() else None)
        payload["from_state_eligibility"] = verdict.to_dict()
        if not verdict.eligible:
            payload["ok"] = False
            payload["would_run"] = False
            payload["blocked_by"] = "from_state_not_route_eligible"
    payload["tag"] = tag
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_tag="compose")
    parser.add_argument(
        "--route",
        default="level1_complete",
        help="NamedRoute id or alias (L1/L2 first consumer)",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write JSON under recordings/ (still no emulator)",
    )
    args = parser.parse_args(argv)
    payload = compose_report(
        route=args.route,
        from_state=args.from_state,
        infinite_life=args.infinite_life,
        tag=args.tag,
    )
    if args.write_report:
        out = write_report("compose_named_route", payload, tag=args.tag)
        payload["report_path"] = str(out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
