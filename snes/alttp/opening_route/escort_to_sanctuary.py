"""Escort rescued Zelda from B3 through the real stairs, mantle, and sewers.

All geometry lives in owning room maps under ``escort_*``. Controller play
opens the key chest, earns the rat key, moves the mantle and block, and pulls
the right lever. Reaching Sanctuary is incomplete until the priest's story
sequence sets progress 2 and returns control with Zelda delivered.
"""

from __future__ import annotations

from collections.abc import Callable

from alttp import primitives
from alttp.opening_route.room_engine import in_room
from alttp.ram import (
    SRAM_PROGRESS, SRAM_PROGRESS_FLAGS, AlttpSnapshot, wram_index,
)
from alttp.room_map import load_room_map
from alttp.route_report import RoutePhaseResult, SegmentResult, phase_from_primitive
from alttp.startup import snapshot_env


def evaluate_acceptance(
    snapshot: AlttpSnapshot, *, progress: int = 0, progress_flags: int = 0,
) -> dict[str, bool]:
    """Fail closed on arrival or follower dismissal before the story commits."""
    checks = {
        "in_sanctuary": snapshot.in_sanctuary,
        "has_control": snapshot.has_control,
        "zelda_delivered": snapshot.follower == 0,
        "escape_story_complete": progress == 2 and bool(progress_flags & 0x04),
    }
    return {**checks, "sanctuary_completed": all(checks.values())}


def _acceptance(env: object) -> dict[str, bool]:
    ram = env.get_ram()  # type: ignore[attr-defined]
    return evaluate_acceptance(
        snapshot_env(env), progress=int(ram[wram_index(SRAM_PROGRESS)]),
        progress_flags=int(ram[wram_index(SRAM_PROGRESS_FLAGS)]),
    )


def run_from_escort(env: object, *, source: str = "state_load_dev") -> SegmentResult:
    """Play the whole escort from the rescued Zelda cell predecessor.

    Survival health refills belong to the caller's audited wrapper. This
    segment never writes RAM, restores a state, or grants an item.
    """
    phases: list[RoutePhaseResult] = []

    def record(label: str, result: primitives.PrimitiveResult | primitives.CombatResult):
        phases.append(phase_from_primitive(label, result))
        if not result.ok:
            raise RuntimeError(result.reason)
        return result

    def settle(label: str, limit: int = 480):
        return record(label, primitives.settle_control(env, max_frames=limit))

    def walk(room: int, label: str, tolerance: int = 1):
        point = load_room_map(f"room_{room:02x}").point("escort_" + label)
        if point is None:
            raise RuntimeError(f"missing escort geometry room_{room:02x}:{label}")
        waypoint = primitives.Waypoint(
            point.x, point.y, tolerance=tolerance, room=room, label=label,
        )
        frames = 0
        for _ in range(4):
            moved = primitives.move_to(env, waypoint, step_size=1, max_frames=1000)
            frames += moved.frames
            if moved.ok or moved.snapshot.has_control:
                break
            ready = primitives.settle_control(env, max_frames=480)
            frames += ready.frames
            if not ready.ok:
                moved = ready
                break
        record(label, primitives.PrimitiveResult(moved.ok, moved.reason, frames, moved.snapshot))
        if not snapshot_env(env).has_control:
            settle(label + "_settle")

    def path(room: int, *labels: str, tolerance: int = 1):
        for label in labels:
            walk(room, label, tolerance)

    def push(label: str, button: str, predicate: Callable[[AlttpSnapshot], bool], limit: int = 600):
        record(label, primitives.move_until(
            env, (button,), predicate, max_frames=limit, step_size=1,
        ))
        settle(label + "_settle")

    def door(label: str, button: str, room: int):
        push(label, button, lambda s: in_room(s, room))

    def script(label: str, buttons: tuple[str, ...], frames: int):
        record(label, primitives.run_script(env, ((buttons, frames),)))
        if not snapshot_env(env).has_control:
            settle(label + "_settle")

    blocker = ""
    try:
        settle("escort_entry")
        snap = snapshot_env(env)
        if _acceptance(env)["sanctuary_completed"]:
            pass
        elif snap.in_sanctuary:
            record("sanctuary_story", primitives.interact_until(
                env, lambda s: _acceptance(env)["escape_story_complete"] and s.follower == 0,
                max_cycles=500,
            ))
            settle("sanctuary_dialogue", 1600)
        else:
            if not (snap.in_zelda_cell and snap.has_zelda_follower and snap.has_lamp
                    and 336 <= snap.link_x <= 368 and 4140 <= snap.link_y <= 4224):
                raise RuntimeError("escort requires rescued Zelda and lamp at the real B3 cell exit")
            path(0x80, "cell_exit", "west_hall", "west_corner", "stair_hall", "stair_mouth", tolerance=0)
            door("b3_to_b2", "UP", 0x70)
            walk(0x70, "stair_mouth", 0)
            door("b2_to_b1", "UP", 0x71)
            path(0x71, "spiral_exit", "west_stair", tolerance=0)
            push("b1_stair_up", "DOWN", lambda s: s.layer == 0 and s.has_control)
            path(0x71, "north_hall", "south_stair", tolerance=0)
            door("b1_south_to_81", "DOWN", 0x81)
            path(0x81, "north_hall", "west_corner", "west_south", "east_corner", "east_south", "east_door")
            door("81_east_to_82", "RIGHT", 0x82)
            path(0x82, "south_east", "pit_corner", "bridge_south", "bridge_north", "north_corner", "north_hall", "north_door_align", "north_door")
            door("82_north_to_72", "UP", 0x72)
            path(0x72, "south_hall", "stair_foot", "stair_head", "center_hall", "north_hall", "chest_south", "chest_west", "chest_north", "spiral_mouth")
            door("b1_to_castle", "UP", 0x01)
            path(0x01, "east_hall", "west_door")
            door("castle_west_to_50", "LEFT", 0x50)
            path(0x50, "north_hall", "south_door")
            door("castle_south_to_60", "DOWN", 0x60)
            walk(0x60, "east_hall")
            door("castle_east_to_61", "RIGHT", 0x61)
            path(0x61, "west_hall", "west_corner", "north_corner", "north_door")
            door("hall_north_to_throne", "UP", 0x51)
            path(0x51, "mantle_dialogue", "mantle_south", "mantle_west", "mantle_push", tolerance=0)
            # The mantle requires cardinal RIGHT velocity 24; diagonals are 16.
            script("move_mantle", ("RIGHT",), 340)
            walk(0x51, "mantle_dialogue", 0)
            door("throne_to_sewers", "UP", 0x41)
            path(0x41, "south_hall", "east_corner", "spiral_mouth")
            door("dark_spiral_to_42", "UP", 0x42)
            path(0x42, "east_hall", "north_stair")
            door("snake_stair_to_32", "UP", 0x32)
            path(0x32, "cross_center", "chest_south", "key_chest", tolerance=0)
            script("face_key_chest", ("UP",), 2)
            record("sewer_chest_key", primitives.interact_until(
                env, lambda s: s.num_keys >= 1, max_cycles=40,
            ))
            # Small-key chests briefly freeze walking without HoldUpItem.
            script("chest_key_settle", ("NONE",), 80)
            path(0x32, "chest_south", "cross_center", "north_door")
            door("key_door_to_22", "UP", 0x22)
            path(0x22, "south_hall", "west_door")
            door("water_hall_to_21", "LEFT", 0x21)
            path(0x21, "south_hall", "internal_north_door", "upper_center", "key_hall", "key_rat")
            record("earn_rat_key", primitives.fight_nearby(
                env, sprite_types=(0x6D,), room=0x21, max_distance=320,
                attack_distance=50, max_cycles=450,
                stop_when=lambda e: bool(primitives.sprites_of_type(e, (0xE4,))),
            ))
            record("collect_rat_key", primitives.collect_nearby(
                env, (0xE4,), max_distance=400, max_frames=600,
            ))
            if snapshot_env(env).num_keys < 1:
                raise RuntimeError("sewer rat key was not earned")
            path(0x21, "key_hall", "east_hall", "north_key_door")
            door("rat_key_door_to_11", "UP", 0x11)
            path(0x11, "south_hall", "east_corner", "east_hall", "north_hall", "block_approach", tolerance=0)
            push("push_middle_block", "UP", lambda s: s.link_y <= 600, 400)
            path(0x11, "block_west", "block_north", "spiral_mouth", tolerance=0)
            door("last_spiral_to_02", "UP", 0x02)
            path(0x02, "east_hall", "north_center", "south_hall", "lever_south", "lever_pull", tolerance=0)
            script("face_right_lever", ("UP",), 12)
            script("grip_right_lever", ("A",), 5)
            script("pull_right_lever", ("A", "DOWN"), 100)
            lever = next((s for s in primitives.sprites_of_type(env, (0x04,))), None)
            if lever is None or int(env.get_ram()[0xDC0 + lever.slot]) != 13:  # type: ignore[attr-defined]
                raise RuntimeError("right lever did not complete its pull")
            path(0x02, "lever_south", "south_hall", tolerance=0)
            door("lever_door_to_sanctuary", "DOWN", 0x12)
            # Control-looking frames occur while the priest still holds Link.
            record("sanctuary_story", primitives.interact_until(
                env, lambda s: _acceptance(env)["escape_story_complete"] and s.follower == 0,
                max_cycles=500,
            ))
            settle("sanctuary_dialogue", 1600)
        if not _acceptance(env)["sanctuary_completed"]:
            raise RuntimeError("Sanctuary story acceptance incomplete")
    except RuntimeError as exc:
        blocker = str(exc)
    snap = snapshot_env(env)
    acc = _acceptance(env)
    return SegmentResult(
        ok=not blocker and acc["sanctuary_completed"],
        phase="sanctuary_completed" if not blocker else (phases[-1].phase if phases else "entry"),
        frames=sum(p.frames for p in phases), snapshot=snap, phases=phases,
        source=source, acceptance=acc, blocker=blocker,
        notes=["Escort geometry is authoritative in maps/room_*.json (escort_* points).",
               "Small keys, mantle, block, lever, and final story flags are earned through play.",
               "Health assistance, if any, is supplied and audited by the caller."],
    )


def run_from_state(state_name: str = "DevSanctuary_Zelda", *, close: bool = True) -> SegmentResult:
    """Development diagnostic; a pin load is not a continuous-route claim."""
    from alttp.opening_route.runner import run_from_state as _run_from_state
    return _run_from_state(state_name, run_from_escort, close=close)
