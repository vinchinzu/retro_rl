"""Continuous power-on to Morph Ball runner.

This first route reuses old room recordings only after reaching their natural
room entries.  Ceres itself is replayed continuously and the two former
checkpoint gaps are closed by deterministic phase-alignment spans discovered
from the natural power-on session.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.paths import GAME, GAME_DIR, POLICY_DIR, RECORDINGS_DIR
from super_metroid.progression import (
    START_TO_MORPH_GRAPH,
    ObservedTransition,
    RoomProgressionGraph,
)
from super_metroid.ram import MORPH_BALL_MASK, SuperMetroidState, parse_state
from super_metroid.video import FrameVideoWriter

Action = np.ndarray


@dataclass(frozen=True)
class ActionSpan:
    names: tuple[str, ...]
    frames: int
    reason: str

    @property
    def action(self) -> Action:
        return buttons(*self.names) if self.names else idle_action()


@dataclass(frozen=True)
class Split:
    split_id: str
    frame: int
    room_id: int


@dataclass
class RunReport:
    schema_version: int
    success: bool
    outcome: str
    total_frames: int
    final_state: dict[str, object]
    splits: list[Split]
    transitions: list[ObservedTransition]
    action_reasons: Counter[str]
    assist: dict[str, object]
    state_loads: int
    progression_writes: int
    video_path: str | None
    source_policy: str
    rom_sha256: str
    start_state: str
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "outcome": self.outcome,
            "total_frames": self.total_frames,
            "final_state": self.final_state,
            "splits": [asdict(split) for split in self.splits],
            "transitions": [asdict(transition) for transition in self.transitions],
            "action_reasons": dict(self.action_reasons),
            "assist": self.assist,
            "state_loads": self.state_loads,
            "progression_writes": self.progression_writes,
            "video_path": self.video_path,
            "source_policy": self.source_policy,
            "rom_sha256": self.rom_sha256,
            "start_state": self.start_state,
            "generated_at": self.generated_at,
        }


def _boot_spans() -> list[ActionSpan]:
    spans = [
        ActionSpan((), 2100, "boot_title_wait"),
        ActionSpan(("A",), 10, "boot_title_confirm"),
        ActionSpan((), 120, "boot_file_menu_wait"),
        ActionSpan(("A",), 10, "boot_file_confirm"),
        ActionSpan((), 300, "boot_prologue_wait"),
        ActionSpan(("A",), 10, "boot_prologue_confirm"),
        ActionSpan((), 30, "boot_prologue_settle"),
    ]
    for _ in range(69):
        spans.append(ActionSpan(("A",), 10, "boot_intro_mash"))
        spans.append(ActionSpan((), 110, "boot_intro_wait"))
    return spans


def _ceres_outbound_spans() -> list[ActionSpan]:
    raw = (
        (("RIGHT", "A"), 24),
        (("RIGHT",), 120),
        (("LEFT",), 120),
        (("RIGHT", "B"), 240),
        ((), 60),
        (("RIGHT",), 24),
        (("RIGHT", "B"), 24),
        (("RIGHT", "B", "A"), 24),
        (("RIGHT", "A"), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT",), 24),
        (("RIGHT", "B"), 24),
        ((), 12),
        (("RIGHT",), 24),
        ((), 140),
        (("RIGHT",), 160),
        (("LEFT",), 120),
        (("RIGHT", "B"), 96),
        ((), 120),
        (("RIGHT", "B"), 216),
        ((), 150),
        (("RIGHT", "B"), 240),
        ((), 200),
    )
    return [ActionSpan(names, frames, "ceres_outbound") for names, frames in raw]


def _ceres_escape_spans() -> list[ActionSpan]:
    raw = (
        (("LEFT", "A"), 40, "ceres_ridley_exit"),
        (("LEFT",), 1000, "ceres_reverse_rooms"),
        ((), 20, "ceres_magnet_phase_align"),
        (("A",), 16, "ceres_magnet_climb"),
        (("RIGHT", "A"), 124, "ceres_magnet_climb"),
        (("LEFT", "A"), 60, "ceres_magnet_climb"),
        (("LEFT",), 320, "ceres_magnet_exit"),
        (("LEFT", "A"), 40, "ceres_falling_room"),
        (("LEFT",), 380, "ceres_falling_room"),
        (("LEFT", "A"), 70, "ceres_elevator_lower_ledge"),
    )
    return [ActionSpan(names, frames, reason) for names, frames, reason in raw]


def _ceres_shaft_spans() -> list[ActionSpan]:
    raw = (
        (("LEFT", "A"), 70),
        ((), 7),
        (("RIGHT", "A"), 67),
        ((), 1),
        (("LEFT", "A"), 67),
        ((), 19),
        (("RIGHT", "A"), 66),
        ((), 8),
        (("RIGHT", "A"), 86),
        (("LEFT", "A"), 83),
        (("RIGHT", "A"), 72),
        ((), 3),
        (("LEFT", "A"), 38),
        (("LEFT",), 25),
    )
    return [ActionSpan(names, frames, "ceres_elevator_climb") for names, frames in raw]


def _load_room_seed(index: int, name: str) -> list[Action]:
    path = POLICY_DIR / f"seg{index:02d}_{name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [np.asarray(action, dtype=np.int8) for action in payload["raw_buttons"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Session:
    def __init__(
        self,
        env: object,
        *,
        writer: FrameVideoWriter | None,
        assist: UnlimitedAmmoAssist,
        graph: RoomProgressionGraph = START_TO_MORPH_GRAPH,
    ) -> None:
        self.env = env
        self.writer = writer
        self.assist = assist
        self.graph = graph
        self.frame = 0
        self.info: dict[str, object] = {}
        self.state = parse_state(env.get_ram(), frame=0)
        self.action_reasons: Counter[str] = Counter()
        self.transitions: list[ObservedTransition] = []
        # Reset WRAM is not initialized to a meaningful room.  Do not turn the
        # first real room into an "unknown -> Ceres" graph transition.
        self._last_room = (
            self.state.room_id if self.state.room_id in self.graph.rooms else 0
        )

    def step(self, action: Action, reason: str) -> SuperMetroidState:
        obs, _, _, _, self.info = self.env.step(action)
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)
        self.assist.apply(self.env.data, self.state)
        self.action_reasons[reason] += 1
        if self.writer is not None:
            self.writer.write(obs)
        room = self.state.room_id
        if (
            room in self.graph.rooms
            and self._last_room
            and room != self._last_room
        ):
            self.transitions.append(
                self.graph.observe_transition(
                    self.frame,
                    self._last_room,
                    room,
                )
            )
        if room in self.graph.rooms:
            self._last_room = room
        return self.state

    def span(self, span: ActionSpan) -> None:
        action = span.action
        for _ in range(span.frames):
            self.step(action, span.reason)

    def spans(self, spans: list[ActionSpan]) -> None:
        for span in spans:
            self.span(span)

    def wait_until(
        self,
        predicate: Callable[[SuperMetroidState], bool],
        *,
        timeout: int,
        reason: str,
    ) -> int:
        for waited in range(timeout + 1):
            if predicate(self.state):
                return waited
            self.step(idle_action(), reason)
        raise TimeoutError(f"{reason} timed out at frame {self.frame}: {self.state}")

    def raw_actions(self, actions: list[Action], reason: str) -> None:
        for action in actions:
            self.step(action, reason)


def play_start_to_morph(session: _Session, splits: list[Split]) -> None:
    """Play the accepted power-on prefix without owning the environment."""
    session.spans(_boot_spans())
    if not (session.state.room_id == 0xDF45 and session.state.game_state == 8):
        raise RuntimeError(f"boot missed first Ceres control: {session.state}")
    splits.append(Split("first_ceres_control", session.frame, session.state.room_id))

    session.spans(_ceres_outbound_spans())
    session.wait_until(
        lambda state: state.room_id == 0xE0B5
        and state.timer_type == 3
        and state.health <= 27,
        timeout=6_000,
        reason="ceres_ridley_natural_countdown",
    )
    splits.append(Split("ridley_countdown", session.frame, session.state.room_id))

    session.spans(_ceres_escape_spans())
    if session.state.room_id != 0xDF45:
        raise RuntimeError(f"Ceres reverse route missed elevator: {session.state}")
    session.wait_until(
        lambda state: state.room_id == 0xDF45
        and state.samus_y == 571
        and state.pose == 2,
        timeout=120,
        reason="ceres_lower_ledge_settle",
    )
    session.spans(_ceres_shaft_spans())
    session.wait_until(
        lambda state: state.room_id == 0x91F8 and state.game_state == 8,
        timeout=3_000,
        reason="zebes_landing_transition",
    )
    # The ship animation pauses once at y=1137, then finally settles at
    # y=1088. Requiring 30 consecutive final-settle frames prevents premature
    # playback of the imported Landing Site seed.
    stable = 0
    for _ in range(1_200):
        if session.state.samus_y == 1088:
            stable += 1
            if stable >= 30:
                break
        else:
            stable = 0
        session.step(idle_action(), "zebes_ship_final_settle")
    else:
        raise TimeoutError(f"Zebes ship never reached final settle: {session.state}")
    splits.append(Split("zebes_landing", session.frame, session.state.room_id))

    segment_specs = (
        (0, "landing_site", 0x92FD),
        (1, "parlor", 0x96BA),
        (2, "climb", 0x975C),
        (3, "pit_room", 0x97B5),
        (4, "bb_elev_hallway", 0x9E9F),
        (5, "morph_ball_room", None),
    )
    for index, name, target_room in segment_specs:
        if index == 4:
            session.span(ActionSpan((), 17, "elevator_seed_alignment"))
        source_room = session.state.room_id
        actions = _load_room_seed(index, name)
        session.raw_actions(actions, f"seed_{name}")

        if index == 0 and session.state.room_id == source_room:
            for _ in range(90):
                session.step(buttons("LEFT", "X"), "landing_door_adapter_shoot")
                if session.state.game_state == 11:
                    break
            for _ in range(180):
                session.step(buttons("LEFT"), "landing_door_adapter_enter")
                if session.state.room_id != source_room:
                    break

        if target_room is not None and session.state.room_id != target_room:
            if session.state.game_state != 11:
                raise RuntimeError(
                    f"seed {name} missed 0x{target_room:04X}: {session.state}"
                )
            session.wait_until(
                lambda state, target=target_room: state.room_id == target,
                timeout=180,
                reason=f"seed_{name}_transition_settle",
            )

    if not session.state.collected_items & MORPH_BALL_MASK:
        raise RuntimeError(f"Morph Ball was not acquired: {session.state}")
    splits.append(Split("morph_ball", session.frame, session.state.room_id))


def run_start_to_morph(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_ammo: bool = True,
) -> RunReport:
    """Run one continuous reset session and stop after natural Morph Ball."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    writer: FrameVideoWriter | None = None
    splits: list[Split] = []
    assist = UnlimitedAmmoAssist(enabled=unlimited_ammo)
    outcome = "runner_error"
    try:
        obs, _ = env.reset()
        if video_path is not None:
            writer = FrameVideoWriter(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
            )
            writer.write(obs)
        session = _Session(env, writer=writer, assist=assist)
        play_start_to_morph(session, splits)
        outcome = "morph_ball_acquired"
        success = True
    except Exception:
        success = False
        raise
    finally:
        final_state = parse_state(env.get_ram(), frame=getattr(locals().get("session"), "frame", 0))
        if writer is not None:
            writer.close()
        env.close()

    report = RunReport(
        schema_version=1,
        success=success,
        outcome=outcome,
        total_frames=session.frame,
        final_state=final_state.to_dict(),
        splits=splits,
        transitions=session.transitions,
        action_reasons=session.action_reasons,
        assist=assist.report(),
        state_loads=0,
        progression_writes=assist.telemetry.progression_writes,
        video_path=str(Path(video_path).resolve()) if video_path is not None else None,
        source_policy="power-on Ceres policy + imported natural-entry room seeds",
        rom_sha256=_sha256(GAME_DIR.parent / "roms" / "SuperMetroid.sfc"),
        start_state="power_on/retro.State.NONE",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report


def default_artifact_paths() -> tuple[Path, Path]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        RECORDINGS_DIR / "start_to_morph.mp4",
        RECORDINGS_DIR / "start_to_morph.json",
    )
