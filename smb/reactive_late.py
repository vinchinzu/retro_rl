"""Control-relative late-game repairs for the shifted SMB warp route.

The M8 seed is still the source of the route's reliable late-game movement,
but an earlier 1-1/1-2 saving changes the emulator phase seen by World 8.
These two controllers are deliberately addressed from *natural control* at
8-3 and 8-4, rather than from an absolute frame in the full-run seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any

from smb.policy import DEFAULT_CONTINUOUS_SEED, expand_nes9_rle, load_nes9_rle_seed
from smb.ram import SmbSnapshot
from smb.reactive_route import level_control_gate, snapshot_fingerprint
from smb.routes import ROUTE_WARP_ANY_PERCENT

ButtonFrame = list[int]

# These slices start on the first controllable frame in a verified M8 replay.
# The 8-3 controller retains its complete 2,206-frame transition envelope:
# 8-4 does not become controllable until that envelope completes.
M8_83_START = 16_112
M8_84_START = 18_318

RUN: ButtonFrame = [1, 0, 0, 0, 0, 0, 0, 1, 0]
RUN_JUMP: ButtonFrame = [1, 0, 0, 0, 0, 0, 0, 1, 1]

# State-local corrections found from the shifted, real 8-3 predecessor.  The
# patch spans are stage-relative and end-exclusive; all other movement is M8
# source material.  In particular, this is not World-4 idle padding.
PATCHES: dict[str, tuple[tuple[int, int, ButtonFrame], ...]] = {
    "8-3": (
        (1_070, 1_089, RUN_JUMP),
        (1_202, 1_218, RUN),
        (1_230, 1_248, RUN_JUMP),
        (1_253, 1_274, RUN),
    ),
    # The shifted 8-4 route reaches Bowser/axe; this jump window clears the
    # final contact and retains the original transition timing.
    "8-4": ((3_360, 3_388, RUN_JUMP),),
}


@cache
def _m8_frames() -> tuple[tuple[int, ...], ...]:
    """Load immutable M8 source frames once for the late controllers."""
    return tuple(
        tuple(int(button) for button in frame)
        for frame in expand_nes9_rle(load_nes9_rle_seed(DEFAULT_CONTINUOUS_SEED))
    )


def stage_frames(stage_id: str) -> list[ButtonFrame]:
    """Return a fresh, patched frame sequence for one late-stage controller."""
    frames = _m8_frames()
    if stage_id == "8-3":
        source = frames[M8_83_START:M8_84_START]
    elif stage_id == "8-4":
        source = frames[M8_84_START:]
    else:
        raise KeyError(f"no late-stage controller for {stage_id!r}")

    controller = [list(frame) for frame in source]
    for start, end, replacement in PATCHES[stage_id]:
        if not 0 <= start < end <= len(controller):
            raise ValueError(
                f"{stage_id} patch [{start}, {end}) outside {len(controller)} frames"
            )
        controller[start:end] = [list(replacement) for _ in range(end - start)]
    return controller


@dataclass
class LateRouteController:
    """Replay the repaired 8-3 then 8-4 controllers from natural control.

    ``observe`` advances the current segment after its action was applied and
    verifies the 8-3 → 8-4 handoff.  Completion of 8-4 itself remains the
    route ending contract in :func:`smb.ram.reached_ending`.
    """

    stage_id: str = "8-3"
    _frames: list[ButtonFrame] = field(default_factory=lambda: stage_frames("8-3"))
    index: int = 0
    starts: dict[str, dict[str, int]] = field(default_factory=dict)
    completed: list[dict[str, Any]] = field(default_factory=list)
    failure: str | None = None

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self._frames)

    @property
    def current_frame_count(self) -> int:
        return len(self._frames)

    def begin(self, snap: SmbSnapshot) -> None:
        """Record and validate the natural 8-3 control entry."""
        gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[6])
        if not gate.matches(snap):
            raise ValueError("late controller must begin at natural 8-3 control")
        self.starts[self.stage_id] = snapshot_fingerprint(snap)

    def next_frame(self) -> ButtonFrame:
        if self.failure is not None:
            raise RuntimeError(self.failure)
        if self.exhausted:
            raise RuntimeError(f"{self.stage_id} controller exhausted")
        return list(self._frames[self.index])

    def observe(self, snap: SmbSnapshot) -> str | None:
        """Record a post-action state and return a stage handoff if one opens."""
        if self.exhausted:
            raise RuntimeError(f"{self.stage_id} controller exhausted")
        self.index += 1
        if not self.exhausted or self.stage_id != "8-3":
            return None

        gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[7])
        if not gate.matches(snap):
            self.failure = "8-3 controller did not hand off at natural 8-4 control"
            raise RuntimeError(self.failure)
        self.completed.append(
            {
                "stage_id": "8-3",
                "frames": self.index,
                "successor_entry": snapshot_fingerprint(snap),
            }
        )
        self.stage_id = "8-4"
        self._frames = stage_frames("8-4")
        self.index = 0
        self.starts[self.stage_id] = snapshot_fingerprint(snap)
        return self.stage_id

    def report(self) -> dict[str, Any]:
        return {
            "active_stage": self.stage_id,
            "frames_in_active_stage": self.index,
            "active_stage_total": len(self._frames),
            "starts": dict(self.starts),
            "completed": list(self.completed),
            "failure": self.failure,
            "patches": {
                stage_id: [
                    {"start": start, "end": end, "buttons": list(buttons)}
                    for start, end, buttons in patches
                ]
                for stage_id, patches in PATCHES.items()
            },
        }
