"""Stage 1 composer: hold RIGHT from Stage1.state to the first house door."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import BehaviorNode, NodeStatus, TickResult
from retro_harness.env import get_available_states, make_env
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameState
from retro_harness.segment_runner import (
    SegmentOutcome,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)

from magical_quest.paths import GAME, GAME_DIR, RECORDINGS_DIR, STAGE1_STATE
from magical_quest.ram import first_door_reached, parse_game_state

DOOR_HOLD_FRAMES = 20
FIRST_DOOR_MAX_FRAMES = 1200


class Stage1Policy(BehaviorNode):
    """Walk right until the 1-1 house door is held with HP remaining."""

    name = "Stage1Policy"

    def __init__(self, *, door_hold_frames: int = DOOR_HOLD_FRAMES) -> None:
        self.door_hold_frames = door_hold_frames
        self._door_hold = 0

    def reset(self) -> None:
        """Clear the door-hold counter for a fresh attempt."""
        self._door_hold = 0

    def tick(self, state: GameState) -> TickResult:
        if state.player_dead or state.health <= 0:
            return TickResult(
                status=NodeStatus.FAILURE,
                action=FrameAction(idle_action(), "dead"),
                reason="dead",
            )
        if first_door_reached(state):
            self._door_hold += 1
            if self._door_hold >= self.door_hold_frames:
                return TickResult(
                    status=NodeStatus.SUCCESS,
                    action=FrameAction(idle_action(), "first_door"),
                    reason="first_door",
                )
            return TickResult(
                status=NodeStatus.RUNNING,
                action=FrameAction(buttons("RIGHT"), "door_hold"),
                reason="door_hold",
            )
        self._door_hold = 0
        return TickResult(
            status=NodeStatus.RUNNING,
            action=FrameAction(buttons("RIGHT"), "walk_right"),
            reason="walk_right",
        )


def run_first_room(
    *,
    max_frames: int = FIRST_DOOR_MAX_FRAMES,
    state_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Stage1.state and hold RIGHT until the first door or a death."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or STAGE1_STATE
    if chosen not in available:
        raise FileNotFoundError(f"{chosen}.state is missing; run scripts/boot_probe.py")
    out = out_dir or RECORDINGS_DIR / "first_door"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    reasons: dict[str, int] = {}
    outcome = SegmentOutcome.TIMEOUT
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        state = parse_game_state(env.get_ram())
        start = snapshot_state(state)
        screenshots = [save_rgb_png(obs, out / "first_door_0000_start.png").name]
        frame = 0
        for frame in range(1, max_frames + 1):
            tick = policy.tick(state)
            reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
            action = tick.action.action if tick.action is not None else idle_action()
            obs, *_rest = env.step(action)
            state = parse_game_state(env.get_ram(), frame=frame)
            if tick.status is NodeStatus.SUCCESS:
                outcome = SegmentOutcome.SUCCESS
                break
            if tick.status is NodeStatus.FAILURE or state.player_dead:
                outcome = SegmentOutcome.DEATH
                break
        screenshots.append(
            save_rgb_png(obs, out / f"first_door_{frame:04d}_end.png").name
        )
        report: dict[str, Any] = {
            "outcome": outcome.name.lower(),
            "success": outcome is SegmentOutcome.SUCCESS,
            "frames": frame,
            "start_state": chosen,
            "end_health": state.health,
            "min_health_ok": state.health > 0,
            "at_first_door": bool(state.extras.get("at_first_door")),
            "player_x": state.player_x,
            "player_y": state.player_y,
            "reason_counts": dict(sorted(reasons.items())),
            "screenshots": screenshots,
            "extras": {"start": start, "end": snapshot_state(state)},
        }
        report_path = write_json_report(out / "first_door.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def main() -> int:
    """CLI entry: RIGHT from Stage1.state to the first door."""
    report = run_first_room()
    print(
        f"{report['outcome']} frames={report['frames']} "
        f"hp={report['end_health']} x={report['player_x']} y={report['player_y']}"
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
