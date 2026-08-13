"""Ceres Ridley (Baby Ridley) policy: tank tail hits until the escape starts.

Public RTA source: wiki.supermetroid.run/Ridley#Ceres_Station (2025-12-24).

Ceres Ridley has no useful HP bar for Any%. The escape timer starts when
Samus energy drops **below 30**. Optimal is five first-frame **tail** hits:

1. Run to the right wall as Ridley appears (tail tip is the fat hitbox).
2. After the third hit he hovers — jump to keep eating the tail.
3. After the fourth hit, nudge slightly left so the tail stays on-screen
   for the fifth hit when i-frames expire.
4. Done when ``timer_type == 3`` (Ceres escape) and health < 30.

Shooting him 100 times also ends the fight and is strictly slower.

Product previously idled at the left door until Ridley wandered over
(``ceres_ridley_natural_countdown``, ~3296f / ~54.8s). Tail-tank is the
replacement body. Energy assist is already suspended on Ceres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from super_metroid.ram import (
    ADDR_INVINCIBILITY_TIMER,
    ADDR_KNOCKBACK_TIMER,
    GS_DEAD,
    SuperMetroidState,
)
from super_metroid.room_timer import format_segment_time
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_RIDLEY
from super_metroid.routes.skills.geometry import POSE_KNOCKBACK
from super_metroid.routes.runtime import ControllerSession, hold

ENERGY_LEAVE = 30
# Right-wall seat (live dump; Mode 7 room ~256 px). Tune via probe dump.
WALL_X_MIN = 220
WALL_X_MAX = 244
NUDGE_X = 214
JUMP_AFTER_HITS = 3
NUDGE_AFTER_HITS = 4
JUMP_HOLD_FRAMES = 24
# After energy is low enough, start the left-door run (escape door).
DOOR_X = 48


@dataclass(frozen=True)
class CeresRidleyStrategy:
    """Tail-tank (default) or idle-wait baseline."""

    policy: str = "tail_tank"
    wall_x_min: int = WALL_X_MIN
    wall_x_max: int = WALL_X_MAX
    nudge_x: int = NUDGE_X
    jump_after_hits: int = JUMP_AFTER_HITS
    nudge_after_hits: int = NUDGE_AFTER_HITS
    energy_leave: int = ENERGY_LEAVE
    door_x: int = DOOR_X
    max_fight_frames: int = 6_000
    jump_hold_frames: int = JUMP_HOLD_FRAMES


@dataclass
class CeresRidleyEvidence:
    """Measured Ceres Ridley encounter (countdown start is the win)."""

    start_frame: int
    end_frame: int
    policy: str
    start_health: int
    end_health: int
    hits: int
    hit_frames: list[int] = field(default_factory=list)
    timer_type: int = 0
    escape_timer_seconds: int = 0
    final_x: int = 0
    final_y: int = 0
    outcome: str = "timeout"

    @property
    def action_frames(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict[str, object]:
        timing = format_segment_time(self.action_frames)
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "action_frames": self.action_frames,
            "seconds": timing["seconds"],
            "clock": timing["clock"],
            "ntsc_fps": timing["ntsc_fps"],
            "policy": self.policy,
            "start_health": self.start_health,
            "end_health": self.end_health,
            "hits": self.hits,
            "hit_frames": list(self.hit_frames),
            "timer_type": self.timer_type,
            "escape_timer_seconds": self.escape_timer_seconds,
            "final_x": self.final_x,
            "final_y": self.final_y,
            "outcome": self.outcome,
        }


def countdown_started(
    state: SuperMetroidState,
    *,
    energy_leave: int = ENERGY_LEAVE,
) -> bool:
    """True when Ridley has left and the Ceres escape timer is running."""
    return (
        state.room_id == ROOM_CERES_RIDLEY
        and state.timer_type == 3
        and state.health < energy_leave
    )


def is_knockback(state: SuperMetroidState, knockback_timer: int = 0) -> bool:
    """True during Ceres knockback poses or while the KB timer is live."""
    return int(state.pose) in POSE_KNOCKBACK or int(knockback_timer) > 0


def fight_ceres_ridley_action(
    state: SuperMetroidState,
    *,
    hits_taken: int,
    frames_since_hit: int = 0,
    invuln: int = 0,
    knockback_timer: int = 0,
    strategy: CeresRidleyStrategy = CeresRidleyStrategy(),
) -> tuple[str, ...]:
    """One-frame tail-tank (or idle wait). Pure; no session."""
    if strategy.policy == "wait" or countdown_started(
        state, energy_leave=strategy.energy_leave
    ):
        return ()
    if is_knockback(state, knockback_timer):
        return ()

    x = int(state.samus_x)
    if x > 60_000:
        return ()

    # Approach / hold the right wall until the hover phase.
    if hits_taken < strategy.jump_after_hits:
        if x < strategy.wall_x_min:
            return ("RIGHT", "B")
        if x > strategy.wall_x_max:
            return ("LEFT",)
        return ()

    # Third hit done: Ridley hovers. Jump to keep the tail overlapping.
    if hits_taken < strategy.nudge_after_hits:
        names: list[str] = []
        if x < strategy.wall_x_min:
            names.append("RIGHT")
        elif x > strategy.wall_x_max:
            names.append("LEFT")
        if frames_since_hit < strategy.jump_hold_frames or invuln == 0:
            names.append("A")
        return tuple(names)

    # Energy already below the leave line: Ridley is leaving. Run to the
    # left door. Do not leave the wall on hit-count alone — weak hits do
    # not drop energy under 30.
    if int(state.health) < strategy.energy_leave:
        if x > strategy.door_x:
            return ("LEFT", "B")
        return ()

    names = []
    if x > strategy.nudge_x:
        names.append("LEFT")
    elif x < strategy.nudge_x - 8:
        names.append("RIGHT")
    if frames_since_hit < strategy.jump_hold_frames or invuln == 0:
        names.append("A")
    return tuple(names)


def _u16(ram, addr: int) -> int:
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _timers(session: ControllerSession) -> tuple[int, int]:
    env = getattr(session, "env", None)
    if env is None:
        return 0, 0
    ram = env.get_ram()  # type: ignore[attr-defined]
    return _u16(ram, ADDR_INVINCIBILITY_TIMER), _u16(ram, ADDR_KNOCKBACK_TIMER)


def fight_terminal(
    state: SuperMetroidState,
    *,
    energy_leave: int = ENERGY_LEAVE,
) -> str | None:
    """Countdown, death, or room leave. ``None`` means the fight continues."""
    if countdown_started(state, energy_leave=energy_leave):
        return "ceres_ridley_countdown"
    if int(state.game_state) in GS_DEAD:
        return "death"
    if int(state.room_id) != ROOM_CERES_RIDLEY:
        return "left_room"
    return None


def require_ceres_ridley_countdown(evidence: CeresRidleyEvidence) -> None:
    """Product path: escape must have started. Probe/bench keep the evidence."""
    if evidence.outcome != "ceres_ridley_countdown":
        raise TimeoutError(
            f"Ceres Ridley did not start escape ({evidence.outcome})"
        )


def play_ceres_ridley_fight(
    session: ControllerSession,
    *,
    strategy: CeresRidleyStrategy = CeresRidleyStrategy(),
) -> CeresRidleyEvidence:
    """Run wait or tail-tank until countdown, timeout, or room leave."""
    start = session.frame
    start_health = int(session.state.health)
    hits = 0
    hit_frames: list[int] = []
    prev_health = start_health
    frames_since_hit = 0
    outcome = "timeout"

    if session.state.room_id != ROOM_CERES_RIDLEY:
        raise RuntimeError(
            f"Ceres Ridley expected room 0x{ROOM_CERES_RIDLEY:04X}, "
            f"got 0x{session.state.room_id:04X}"
        )

    for _ in range(strategy.max_fight_frames):
        state = session.state
        terminal = fight_terminal(state, energy_leave=strategy.energy_leave)
        if terminal == "ceres_ridley_countdown":
            outcome = terminal
            break
        if terminal == "death":
            raise TimeoutError(f"Ceres Ridley death during fight: {state}")
        if terminal == "left_room":
            outcome = terminal
            break

        invuln, kb = _timers(session)
        names = fight_ceres_ridley_action(
            state,
            hits_taken=hits,
            frames_since_hit=frames_since_hit,
            invuln=invuln,
            knockback_timer=kb,
            strategy=strategy,
        )
        reason = (
            "ceres_ridley_natural_countdown"
            if strategy.policy == "wait"
            else "ceres_ridley_tail_tank"
        )
        if names:
            hold(session, 1, *names, reason=reason)
        else:
            hold(session, 1, reason=reason)

        post = session.state
        health = int(post.health)
        frames_since_hit += 1
        if 0 < health < prev_health:
            hits += 1
            hit_frames.append(session.frame)
            frames_since_hit = 0
        prev_health = health
        post_term = fight_terminal(post, energy_leave=strategy.energy_leave)
        if post_term == "ceres_ridley_countdown":
            outcome = post_term
            break
        if post_term == "death":
            raise TimeoutError(f"Ceres Ridley death during fight: {post}")

    end_state = session.state
    return CeresRidleyEvidence(
        start_frame=start,
        end_frame=session.frame,
        policy=strategy.policy,
        start_health=start_health,
        end_health=int(end_state.health),
        hits=hits,
        hit_frames=hit_frames,
        timer_type=int(end_state.timer_type),
        escape_timer_seconds=int(end_state.escape_timer_seconds),
        final_x=int(end_state.samus_x),
        final_y=int(end_state.samus_y),
        outcome=outcome,
    )
