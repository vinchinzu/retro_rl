"""Route definition and evaluation for multi-segment speedruns.

A route is an ordered list of segments, each referencing a registered
LevelConfig.  evaluate_route() runs each segment independently using
the Evaluator, giving reliable per-segment results without emulator
state leakage.

Usage:
    from retro_harness.platformer.route import (
        RouteConfig, evaluate_route, get_platformer_route,
    )

    route = get_platformer_route("smb_any_percent")
    results = evaluate_route(route)
"""

from retro_harness.platformer.route.models import (
    RouteSegment,
    RouteConfig,
    ROUTE_REGISTRY,
    register_route,
    get_platformer_route,
    get_route,
    list_routes,
    find_best_recording,
    _load_practice_seeds,
    load_recording_data,
)
from retro_harness.platformer.route.evaluate import (
    SegmentResult,
    RouteResult,
    evaluate_route,
)
from retro_harness.platformer.route.chain_live import (
    ChainLiveSegmentResult,
    ChainLiveResult,
    _run_neuro_live,
    chain_live,
)
from retro_harness.platformer.route.chain_optimize import (
    _find_alignment,
    chain_optimize,
)
from retro_harness.platformer.route.video import record_route_video

__all__ = [
    "RouteSegment",
    "RouteConfig",
    "ROUTE_REGISTRY",
    "register_route",
    "get_platformer_route",
    "get_route",
    "list_routes",
    "find_best_recording",
    "_load_practice_seeds",
    "load_recording_data",
    "SegmentResult",
    "RouteResult",
    "evaluate_route",
    "ChainLiveSegmentResult",
    "ChainLiveResult",
    "_run_neuro_live",
    "chain_live",
    "_find_alignment",
    "chain_optimize",
    "record_route_video",
]
