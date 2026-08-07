"""Control-relative late-game repairs for the shifted SMB warp route.

The M8 seed is still the source of the route's reliable late-game movement,
but an earlier 1-1/1-2 saving changes the emulator phase seen by World 8.
These controllers are deliberately addressed from *natural control* at 8-3
and 8-4, rather than from an absolute frame in the full-run seed.

After the 1-2 −97f polish + 8-2 +1 lead retime (2026-08-05), 8-3 needs two
lead idles before the patched M8 body; 8-4 keeps the Bowser/axe patch with
no lead. Handoff to 8-4 is on first natural control (not forced exhaust) so
leftover transition frames do not burn the entry phase.
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
M8_83_START = 16_112
M8_84_START = 18_318
M8_83_LEN = M8_84_START - M8_83_START  # 2206

IDLE: ButtonFrame = [0, 0, 0, 0, 0, 0, 0, 0, 0]
RUN: ButtonFrame = [1, 0, 0, 0, 0, 0, 0, 1, 0]
RUN_JUMP: ButtonFrame = [1, 0, 0, 0, 0, 0, 0, 1, 1]

# Lead idles prepended after patches (M8-relative patch indices stay valid).
# Found on natural 8-3 control after 8-2 +1-idle body (1-2 −97f path).
LEAD_IDLES: dict[str, int] = {
    "8-3": 2,
    "8-4": 0,
}

# State-local corrections; spans are stage-relative (pre-lead) and end-exclusive.
# All other movement is M8 source material — not World-4 idle padding.
PATCHES: dict[str, tuple[tuple[int, int, ButtonFrame], ...]] = {
    "8-3": (
        (1_070, 1_089, RUN_JUMP),
        (1_202, 1_218, RUN),
        (1_230, 1_248, RUN_JUMP),
        (1_253, 1_274, RUN),
    ),
    # Bowser/axe jump window on natural 8-4 control after the 8-3 lead retime.
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
    """Return a fresh, patched (+ optional lead idle) late-stage controller."""
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
    lead = int(LEAD_IDLES.get(stage_id, 0))
    if lead:
        controller = [list(IDLE) for _ in range(lead)] + controller
    return controller


@dataclass
class LateRouteController:
    """Replay the repaired 8-3 then 8-4 controllers from natural control.

    ``observe`` advances the current segment after its action was applied and
    hands off to 8-4 on first natural 8-4 control (or fails if the 8-3 body
    exhausts without control). Completion of 8-4 itself remains the route
    ending contract in :func:`smb.ram.reached_ending`.
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

    def _handoff_to_8_4(self, snap: SmbSnapshot) -> str:
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

    def observe(self, snap: SmbSnapshot) -> str | None:
        """Record a post-action state and return a stage handoff if one opens."""
        if self.exhausted:
            raise RuntimeError(f"{self.stage_id} controller exhausted")
        self.index += 1
        if self.stage_id != "8-3":
            return None

        gate = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[7])
        if gate.matches(snap):
            # Prefer first natural control — leftover 8-3 transition frames
            # after control can burn the 8-4 entry phase.
            return self._handoff_to_8_4(snap)
        if self.exhausted:
            self.failure = "8-3 controller did not hand off at natural 8-4 control"
            raise RuntimeError(self.failure)
        return None

    def report(self) -> dict[str, Any]:
        return {
            "active_stage": self.stage_id,
            "frames_in_active_stage": self.index,
            "active_stage_total": len(self._frames),
            "starts": dict(self.starts),
            "completed": list(self.completed),
            "failure": self.failure,
            "lead_idles": dict(LEAD_IDLES),
            "patches": {
                stage_id: [
                    {"start": start, "end": end, "buttons": list(buttons)}
                    for start, end, buttons in patches
                ]
                for stage_id, patches in PATCHES.items()
            },
        }
