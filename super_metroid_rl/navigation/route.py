"""Pre-computed speedrun route from Landing Site to Bomb Torizo.

Defines the sequence of rooms and abilities available at each step.
Used by waypoint_gen.py to auto-generate waypoints for each segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RouteStep:
    """One segment of the speedrun route."""
    segment_id: str         # Matches level_id in platformer_common registry
    entry_room_id: int      # Room Samus starts in
    exit_room_id: int       # Room Samus exits to (0 = item collect, stays in room)
    abilities: set[str] = field(default_factory=set)


# The full Landing Site → Bomb Torizo route
SPEEDRUN_ROUTE: list[RouteStep] = [
    # Descent: Landing Site → Morph Ball
    RouteStep("sm_landing_site",       0x91F8, 0x92FD, set()),
    RouteStep("sm_parlor_descent",     0x92FD, 0x96BA, set()),
    RouteStep("sm_climb_descent",      0x96BA, 0x975C, set()),
    RouteStep("sm_pit_room_descent",   0x975C, 0x97B5, set()),
    RouteStep("sm_elevator_descent",   0x97B5, 0x9E9F, set()),
    RouteStep("sm_morph_ball_collect", 0x9E9F, 0,      set()),

    # Return: Morph Ball → Bomb Torizo
    RouteStep("sm_morph_ball_return",  0x9E9F, 0x97B5, {"morph_ball"}),
    RouteStep("sm_elevator_return",    0x97B5, 0x975C, {"morph_ball"}),
    RouteStep("sm_pit_room_return",    0x975C, 0x96BA, {"morph_ball"}),
    RouteStep("sm_climb_return",       0x96BA, 0x92FD, {"morph_ball"}),
    RouteStep("sm_parlor_to_flyway",   0x92FD, 0x9879, {"morph_ball"}),
    RouteStep("sm_flyway_to_torizo",   0x9879, 0x9804, {"morph_ball", "missile"}),
]


def get_route_step(segment_id: str) -> RouteStep | None:
    """Look up a route step by segment ID."""
    for step in SPEEDRUN_ROUTE:
        if step.segment_id == segment_id:
            return step
    return None


def route_summary() -> str:
    """Human-readable route summary."""
    lines = ["Super Metroid Speedrun Route: Landing Site → Bomb Torizo", ""]
    for i, step in enumerate(SPEEDRUN_ROUTE):
        abilities_str = ", ".join(sorted(step.abilities)) if step.abilities else "none"
        exit_str = f"0x{step.exit_room_id:04X}" if step.exit_room_id else "collect"
        lines.append(
            f"  {i+1:2d}. {step.segment_id:<25s}  "
            f"0x{step.entry_room_id:04X} → {exit_str}  "
            f"[{abilities_str}]"
        )
    return "\n".join(lines)
