"""Collision-guided seed synthesizer for Super Metroid speedrun segments.

Generates approximate button sequences from collision data and room geometry.
These are rough seeds (won't complete rooms perfectly) but give the hill
climber a fundamentally different starting path than a recording.

Usage:
    uv run python -m super_metroid_rl.navigation.seed_synth --segment parlor_descent --preview
    uv run python -m super_metroid_rl.navigation.seed_synth --segment climb_descent --output seed.json
    uv run python -m super_metroid_rl.navigation.seed_synth --list
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from super_metroid_rl.navigation.map_data import WorldData, load_world
from super_metroid_rl.navigation.room_analyzer import RoomCollision
from super_metroid_rl.navigation.route import get_route_step, SPEEDRUN_ROUTE

# ---------------------------------------------------------------------------
# SNES button indices (retro env order) and pre-built button frames
# ---------------------------------------------------------------------------
_B, _Y, _SELECT, _START, _UP, _DOWN, _LEFT, _RIGHT, _A, _X, _L, _R = range(12)

# Commonly used button combinations as 12-element arrays
NOTHING = [0] * 12
F_LEFT = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
F_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
F_RUN_LEFT = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]    # B + LEFT
F_RUN_RIGHT = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]   # B + RIGHT
F_JUMP = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]         # A
F_JUMP_LEFT = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]    # A + LEFT
F_JUMP_RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]   # A + RIGHT
F_DJUMP_LEFT = [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]   # B + A + LEFT
F_DJUMP_RIGHT = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  # B + A + RIGHT
F_DOWN = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
F_UP = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# SM physics approximations (pixels per frame at steady state)
FALL_SPEED = 6       # terminal velocity falling
WALK_SPEED = 3       # walking speed
RUN_SPEED = 5        # dash speed
JUMP_FRAMES = 25     # frames of upward motion in a full jump


def _hold(frame: list[int], count: int) -> list[list[int]]:
    """Repeat a button frame for count frames."""
    return [list(frame) for _ in range(max(1, count))]


def _idle(count: int) -> list[list[int]]:
    """Empty frames (no buttons pressed)."""
    return [list(NOTHING) for _ in range(max(1, count))]


# ---------------------------------------------------------------------------
# Per-segment synthesis parameters
# ---------------------------------------------------------------------------

@dataclass
class SegmentHint:
    """Hints for how to synthesize a seed for a segment."""
    segment_type: Literal["fall", "horizontal", "complex", "elevator", "ascent"]
    entry: tuple[int, int]
    exit: tuple[int, int]
    direction: Literal["left", "right", "up", "down"]


SEGMENT_HINTS: dict[str, SegmentHint] = {
    # Descent phase
    "sm_landing_site":       SegmentHint("horizontal", (1100, 900),  (8, 1152),    "left"),
    "sm_parlor_descent":     SegmentHint("complex",    (1272, 139),  (393, 1248),  "down"),
    "sm_climb_descent":      SegmentHint("fall",       (400, 50),    (490, 2100),  "down"),
    "sm_pit_room_descent":   SegmentHint("fall",       (200, 64),    (400, 400),   "down"),
    "sm_elevator_descent":   SegmentHint("elevator",   (128, 64),    (128, 400),   "down"),
    "sm_morph_ball_collect": SegmentHint("horizontal", (64, 128),    (400, 128),   "right"),
    # Missile detour
    "sm_morph_to_construction":      SegmentHint("horizontal", (64, 128),   (400, 128),  "right"),
    "sm_construction_to_missile":    SegmentHint("ascent",     (128, 400),  (128, 64),   "up"),
    "sm_missile_to_construction":    SegmentHint("horizontal", (200, 128),  (64, 128),   "left"),
    "sm_construction_to_morph":      SegmentHint("fall",       (128, 64),   (128, 400),  "down"),
    # Return phase
    "sm_morph_ball_return":  SegmentHint("horizontal", (400, 400),  (64, 64),     "left"),
    "sm_elevator_return":    SegmentHint("elevator",   (128, 400),  (128, 64),    "up"),
    "sm_pit_room_return":    SegmentHint("horizontal", (400, 300),  (8, 128),     "left"),
    "sm_climb_return":       SegmentHint("ascent",     (500, 2187), (340, 67),    "up"),
    "sm_parlor_to_flyway":   SegmentHint("horizontal", (384, 1260), (1260, 900),  "right"),
    "sm_flyway_to_torizo":   SegmentHint("horizontal", (64, 128),   (256, 128),   "right"),
}


# ---------------------------------------------------------------------------
# Main synthesis entry point
# ---------------------------------------------------------------------------

def synthesize_seed(
    segment_id: str,
    world: WorldData | None = None,
) -> list[list[int]]:
    """Generate a synthetic button sequence from collision data.

    Args:
        segment_id: Level config ID (e.g. "sm_parlor_descent" or "parlor_descent")
        world: Loaded world data (optional, loaded on demand)

    Returns:
        List of 12-element raw button arrays.
    """
    # Ensure SM level configs are registered so get_level_config works
    import platformer_common.levels.super_metroid  # noqa: F401

    if not segment_id.startswith("sm_"):
        segment_id = "sm_" + segment_id

    hint = SEGMENT_HINTS.get(segment_id)
    if hint is None:
        raise ValueError(
            f"No synthesis hint for segment: {segment_id}\n"
            f"Available: {sorted(SEGMENT_HINTS)}"
        )

    step = get_route_step(segment_id)
    room_id = step.entry_room_id if step else 0

    if hint.segment_type == "fall":
        return _synth_fall(segment_id, room_id, hint, world)
    elif hint.segment_type == "horizontal":
        return _synth_horizontal(segment_id, room_id, hint, world)
    elif hint.segment_type == "complex":
        return _synth_complex(segment_id, room_id, hint, world)
    elif hint.segment_type == "elevator":
        return _synth_elevator(hint)
    elif hint.segment_type == "ascent":
        return _synth_ascent(segment_id, room_id, hint, world)
    else:
        raise ValueError(f"Unknown segment type: {hint.segment_type}")


# ---------------------------------------------------------------------------
# Type-specific synthesis
# ---------------------------------------------------------------------------

def _synth_fall(
    segment_id: str,
    room_id: int,
    hint: SegmentHint,
    world: WorldData | None,
) -> list[list[int]]:
    """Synthesize seed for a vertical fall room.

    Uses room_analyzer's BFS fall path to find the column sequence with
    fewest platform landings, then generates horizontal correction holds.
    """
    # Try collision-guided optimal path
    try:
        rc = RoomCollision.load(room_id)
        entry_col = hint.entry[0] // 16
        exit_col = hint.exit[0] // 16
        col_start = max(0, min(entry_col, exit_col) - 12)
        col_end = min(rc.width_blocks, max(entry_col, exit_col) + 12)
        row_start = max(0, hint.entry[1] // 16)
        row_end = min(rc.height_blocks, hint.exit[1] // 16 + 1)

        waypoints = rc.find_optimal_fall_path(col_start, col_end, row_start, row_end)
        if waypoints and len(waypoints) >= 2:
            return _waypoints_to_fall_buttons(waypoints)
    except FileNotFoundError:
        pass

    # Fallback: simple fall with horizontal drift
    return _simple_fall_buttons(hint)


def _waypoints_to_fall_buttons(waypoints: list[tuple[int, int]]) -> list[list[int]]:
    """Convert pixel waypoints from a fall path into button sequences.

    Between each pair of waypoints, calculates the time from vertical
    distance (gravity) and applies horizontal correction as needed.
    """
    buttons: list[list[int]] = []

    for i in range(len(waypoints) - 1):
        px1, py1 = waypoints[i]
        px2, py2 = waypoints[i + 1]

        dx = px2 - px1
        dy = py2 - py1

        # Frames needed for vertical distance
        if dy > 0:
            v_frames = max(int(dy / FALL_SPEED), 1)
        else:
            v_frames = max(int(abs(dy) / 3), 1)

        if abs(dx) < 16:
            # Straight fall — no horizontal input needed
            buttons.extend(_idle(v_frames))
        else:
            # Need horizontal correction while falling
            h_frames_needed = int(abs(dx) / WALK_SPEED)
            dir_frame = F_LEFT if dx < 0 else F_RIGHT

            # Hold direction for the needed correction, then release
            hold = min(h_frames_needed, v_frames)
            buttons.extend(_hold(dir_frame, hold))
            if v_frames > hold:
                buttons.extend(_idle(v_frames - hold))

    return buttons


def _simple_fall_buttons(hint: SegmentHint) -> list[list[int]]:
    """Fallback: simple fall with constant horizontal drift."""
    dx = hint.exit[0] - hint.entry[0]
    dy = hint.exit[1] - hint.entry[1]
    total_frames = max(int(abs(dy) / FALL_SPEED), 30)

    if abs(dx) > 16:
        dir_frame = F_LEFT if dx < 0 else F_RIGHT
        correction = min(int(abs(dx) / WALK_SPEED), total_frames // 2)
        return _hold(dir_frame, correction) + _idle(total_frames - correction)
    return _idle(total_frames)


def _synth_horizontal(
    segment_id: str,
    room_id: int,
    hint: SegmentHint,
    world: WorldData | None,
) -> list[list[int]]:
    """Synthesize seed for a horizontal traversal.

    Runs in the target direction with periodic jumps when the collision
    grid shows obstacles ahead at foot height.
    """
    entry_x, entry_y = hint.entry
    exit_x, exit_y = hint.exit
    dx = exit_x - entry_x
    dy = exit_y - entry_y
    going_right = dx > 0
    run_frame = F_RUN_RIGHT if going_right else F_RUN_LEFT
    djump_frame = F_DJUMP_RIGHT if going_right else F_DJUMP_LEFT

    # If significant vertical component, use waypoint-based synthesis
    if abs(dy) > 128:
        return _synth_from_config_waypoints(segment_id, hint)

    # Try collision-aware horizontal synthesis
    obstacle_cols = _scan_obstacles_ahead(room_id, hint, world)

    total_dist = max(abs(dx), abs(dy))
    run_frames = max(int(total_dist / RUN_SPEED), 60)

    buttons: list[list[int]] = []
    current_x = entry_x

    while len(buttons) < run_frames:
        # Check if we're near an obstacle column
        approx_col = int(current_x) // 16
        near_obstacle = any(abs(approx_col - oc) < 2 for oc in obstacle_cols)

        if near_obstacle:
            # Jump over obstacle
            buttons.extend(_hold(djump_frame, 15))
            buttons.extend(_hold(run_frame, 10))
            current_x += (15 + 10) * RUN_SPEED * (1 if going_right else -1)
        else:
            # Run straight
            chunk = min(30, run_frames - len(buttons))
            buttons.extend(_hold(run_frame, chunk))
            current_x += chunk * RUN_SPEED * (1 if going_right else -1)

    return buttons


def _scan_obstacles_ahead(
    room_id: int,
    hint: SegmentHint,
    world: WorldData | None,
) -> list[int]:
    """Scan collision grid for obstacle columns along the horizontal path.

    Returns a list of block columns where solid blocks exist at foot height.
    """
    obstacles: list[int] = []
    room = world.rooms.get(room_id) if world else None
    if room is None:
        return obstacles

    going_right = hint.exit[0] > hint.entry[0]
    foot_row = hint.entry[1] // 16
    # Also check one row above (chest height) and one below (ground)
    check_rows = [max(0, foot_row - 1), foot_row, min(room.height_blocks - 1, foot_row + 1)]

    start_col = hint.entry[0] // 16
    end_col = hint.exit[0] // 16
    if not going_right:
        start_col, end_col = end_col, start_col

    from super_metroid_rl.navigation.room_analyzer import BLOCKING
    for col in range(start_col, min(end_col + 1, room.width_blocks)):
        for row in check_rows:
            if 0 <= row < room.height_blocks and room.collision[row][col] in BLOCKING:
                obstacles.append(col)
                break

    return obstacles


def _synth_complex(
    segment_id: str,
    room_id: int,
    hint: SegmentHint,
    world: WorldData | None,
) -> list[list[int]]:
    """Synthesize seed for a complex multi-screen room (e.g. Parlor 5x5).

    Uses the level config's waypoints for coarse path, then generates
    motion between each pair.
    """
    return _synth_from_config_waypoints(segment_id, hint)


def _synth_from_config_waypoints(
    segment_id: str,
    hint: SegmentHint,
) -> list[list[int]]:
    """Generate buttons following waypoints from the level config."""
    # Load waypoints from registered level config
    try:
        from platformer_common.level_config import get_level_config
        config = get_level_config(segment_id)
        waypoints = config.waypoints if config.waypoints else None
    except (KeyError, ImportError):
        waypoints = None

    if not waypoints:
        waypoints = [
            (float(hint.entry[0]), float(hint.entry[1])),
            (float(hint.exit[0]), float(hint.exit[1])),
        ]

    buttons: list[list[int]] = []

    for i in range(len(waypoints) - 1):
        px1, py1 = waypoints[i]
        px2, py2 = waypoints[i + 1]
        dx = px2 - px1
        dy = py2 - py1

        going_right = dx > 0
        run_frame = F_RUN_RIGHT if going_right else F_RUN_LEFT
        djump_frame = F_DJUMP_RIGHT if going_right else F_DJUMP_LEFT
        jump_frame = F_JUMP_RIGHT if going_right else F_JUMP_LEFT

        if abs(dy) > 32 and dy > 0:
            # Falling section — horizontal movement while falling
            fall_frames = max(int(dy / FALL_SPEED), 5)
            if abs(dx) > 32:
                h_frames = min(int(abs(dx) / WALK_SPEED), fall_frames)
                dir_frame = F_RIGHT if going_right else F_LEFT
                buttons.extend(_hold(dir_frame, h_frames))
                if fall_frames > h_frames:
                    buttons.extend(_idle(fall_frames - h_frames))
            else:
                buttons.extend(_idle(fall_frames))

        elif abs(dy) > 32 and dy < 0:
            # Ascending section — jump while moving horizontally
            height = abs(dy)
            jumps = max(int(height / 64), 1)
            for _ in range(jumps):
                buttons.extend(_hold(djump_frame, JUMP_FRAMES))
                buttons.extend(_hold(run_frame, 8))

        elif abs(dx) > 16:
            # Horizontal section — run
            h_frames = max(int(abs(dx) / RUN_SPEED), 5)
            buttons.extend(_hold(run_frame, h_frames))

        else:
            # Small adjustment
            buttons.extend(_idle(5))

    return buttons


def _synth_elevator(hint: SegmentHint) -> list[list[int]]:
    """Synthesize seed for an elevator segment.

    Step onto the platform and wait for it to travel.
    """
    going_down = hint.direction == "down"

    buttons: list[list[int]] = []
    # Walk to elevator platform
    dir_frame = F_DOWN if going_down else F_UP
    buttons.extend(_hold(dir_frame, 30))
    # Wait for elevator (200-400 frames typical)
    buttons.extend(_idle(350))
    return buttons


def _synth_ascent(
    segment_id: str,
    room_id: int,
    hint: SegmentHint,
    world: WorldData | None,
) -> list[list[int]]:
    """Synthesize seed for an ascending (climbing) room.

    For tall rooms like Climb return, generates alternating wall-jump
    patterns following the level config waypoints.
    """
    # Use waypoints if available for guided ascent
    try:
        from platformer_common.level_config import get_level_config
        config = get_level_config(segment_id)
        if config.waypoints:
            return _ascent_from_waypoints(config.waypoints)
    except (KeyError, ImportError):
        pass

    # Fallback: zigzag wall-jump pattern
    return _zigzag_ascent(hint)


def _ascent_from_waypoints(
    waypoints: list[tuple[float, float]],
) -> list[list[int]]:
    """Generate ascent buttons following level config waypoints.

    Between each pair, alternates between running and jumping to
    gain height while following the horizontal path.
    """
    buttons: list[list[int]] = []

    for i in range(len(waypoints) - 1):
        px1, py1 = waypoints[i]
        px2, py2 = waypoints[i + 1]
        dx = px2 - px1
        dy = py2 - py1

        going_right = dx >= 0
        djump = F_DJUMP_RIGHT if going_right else F_DJUMP_LEFT
        run = F_RUN_RIGHT if going_right else F_RUN_LEFT

        if dy < -32:
            # Need to gain height
            height = abs(dy)
            jumps_needed = max(int(height / 64), 1)
            h_per_jump = abs(dx) / max(jumps_needed, 1)

            for j in range(jumps_needed):
                # Jump + direction
                buttons.extend(_hold(djump, JUMP_FRAMES))
                # Land + run to next position
                land_frames = max(int(h_per_jump / RUN_SPEED), 5)
                buttons.extend(_hold(run, land_frames))
        elif abs(dx) > 16:
            # Horizontal movement
            h_frames = max(int(abs(dx) / RUN_SPEED), 5)
            buttons.extend(_hold(run, h_frames))
        else:
            buttons.extend(_idle(5))

    return buttons


def _zigzag_ascent(hint: SegmentHint) -> list[list[int]]:
    """Fallback: zigzag wall-jump pattern for ascending."""
    dy = hint.entry[1] - hint.exit[1]
    num_jumps = max(int(dy / 64), 5)

    buttons: list[list[int]] = []
    for j in range(num_jumps):
        going_right = (j % 2 == 0)
        run = F_RUN_RIGHT if going_right else F_RUN_LEFT
        djump = F_DJUMP_RIGHT if going_right else F_DJUMP_LEFT

        # Run toward wall
        buttons.extend(_hold(run, 12))
        # Jump off wall
        buttons.extend(_hold(djump, JUMP_FRAMES))
        # Brief transition
        buttons.extend(_idle(3))

    return buttons


# ---------------------------------------------------------------------------
# Preview and CLI
# ---------------------------------------------------------------------------

def preview_seed(segment_id: str, world: WorldData | None = None) -> str:
    """Preview a synthetic seed: show path info and button summary."""
    if not segment_id.startswith("sm_"):
        segment_id = "sm_" + segment_id

    hint = SEGMENT_HINTS.get(segment_id)
    if not hint:
        return f"No hint for segment: {segment_id}"

    buttons = synthesize_seed(segment_id, world)

    total_left = sum(1 for f in buttons if f[_LEFT])
    total_right = sum(1 for f in buttons if f[_RIGHT])
    total_jump = sum(1 for f in buttons if f[_A])
    total_dash = sum(1 for f in buttons if f[_B])
    total_idle = sum(1 for f in buttons if not any(f))

    lines = [
        f"Segment: {segment_id}",
        f"Type: {hint.segment_type}",
        f"Entry: ({hint.entry[0]}, {hint.entry[1]})",
        f"Exit:  ({hint.exit[0]}, {hint.exit[1]})",
        f"Direction: {hint.direction}",
        f"Synthesized: {len(buttons)} frames ({len(buttons) / 60:.1f}s)",
        f"Buttons: LEFT={total_left} RIGHT={total_right} JUMP={total_jump} "
        f"DASH={total_dash} IDLE={total_idle}",
    ]

    # Show collision grid overlay if available
    step = get_route_step(segment_id)
    if step:
        try:
            rc = RoomCollision.load(step.entry_room_id)
            lines.append("")
            lines.append(rc.render_ascii(
                path_pixels=[hint.entry, hint.exit],
            ))
        except FileNotFoundError:
            lines.append("(No collision data available)")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collision-guided seed synthesizer")
    parser.add_argument("--segment", help="Segment ID (e.g. parlor_descent)")
    parser.add_argument("--preview", action="store_true", help="Show preview (no file output)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--all", action="store_true", help="Synthesize all segments")
    parser.add_argument("--list", action="store_true", help="List available segments")
    args = parser.parse_args()

    if args.list:
        print("Available segments for synthesis:")
        for sid in sorted(SEGMENT_HINTS):
            h = SEGMENT_HINTS[sid]
            print(f"  {sid:<35s}  type={h.segment_type:<12s}  dir={h.direction}")
        return

    world = None
    try:
        world = load_world()
        print(f"Loaded map data ({len(world.rooms)} rooms)")
    except Exception:
        print("(No map data at /tmp/sm_export, using fallback synthesis)")

    if args.all:
        print(f"\nSynthesizing seeds for all {len(SEGMENT_HINTS)} segments:\n")
        for sid in SPEEDRUN_ROUTE:
            seg_id = sid.segment_id
            if seg_id not in SEGMENT_HINTS:
                continue
            buttons = synthesize_seed(seg_id, world)
            print(f"  {seg_id:<35s}  {len(buttons):>5d} frames  ({len(buttons) / 60:.1f}s)")
        return

    if not args.segment:
        parser.print_help()
        return

    seg = args.segment
    if not seg.startswith("sm_"):
        seg = "sm_" + seg

    if args.preview:
        print(preview_seed(seg, world))
        return

    buttons = synthesize_seed(seg, world)
    print(f"Synthesized {len(buttons)} frames ({len(buttons) / 60:.1f}s) for {seg}")

    output = args.output or f"seed_synth_{seg}.json"
    data = {
        "raw_buttons": buttons,
        "num_frames": len(buttons),
        "source": "seed_synth",
        "segment_id": seg,
    }
    Path(output).write_text(json.dumps(data, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
