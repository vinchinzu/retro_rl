"""Shared continuous-route runtime: session, evidence helpers, run harness.

Route-specific play logic stays in :mod:`super_metroid.routes.continuous`
and the controllers; this module owns session bookkeeping and report plumbing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol

import numpy as np

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env
from super_metroid.paths import GAME, GAME_DIR, MAPS_DIR, RECORDINGS_DIR, SHARED_ROM
from super_metroid.policy import SegmentEvidence
from super_metroid.progression import ObservedTransition, RoomProgressionGraph
from super_metroid.ram import GameplayPhase, SuperMetroidState, parse_state
from super_metroid.room_timer import RoomTimer
from super_metroid.video import FrameVideoWriter

Action = np.ndarray

ROUTE_PLAN_PATH = MAPS_DIR / "post_torizo_to_spore_spawn_plan.json"


class ControllerSession(Protocol):
    """Minimal surface controllers need (implemented by :class:`RouteSession`)."""

    frame: int
    state: SuperMetroidState

    def step(self, action: np.ndarray, reason: str) -> SuperMetroidState: ...


class AssistLike(Protocol):
    telemetry: Any

    def apply(self, data: Any, state: SuperMetroidState) -> None: ...

    def report(self) -> dict[str, object]: ...


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


@dataclass(frozen=True)
class ProgressEvent:
    event_id: str
    frame: int
    room_id: int
    before: int
    after: int


@dataclass
class ContinuousRunReport:
    """Unified continuous-run report (schema fields grow with route depth).

    ``kind`` selects optional keys for JSON compatibility with prior baselines:
    ``morph`` | ``bombs`` | ``spore`` | ``supers``.
    """

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
    source_policy: str
    rom_sha256: str
    start_state: str
    generated_at: str
    kind: str = "bombs"
    error: str | None = None
    encoded_frames: int = 0
    progress_events: list[ProgressEvent] = field(default_factory=list)
    segments: list[SegmentEvidence] = field(default_factory=list)
    integrity: dict[str, object] | None = None
    video: dict[str, object] | None = None
    video_path: str | None = None
    route_plan: dict[str, object] | None = None
    policy_sources: dict[str, object] | None = None
    boss: Any = None
    super_collect: Any = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
            "source_policy": self.source_policy,
            "rom_sha256": self.rom_sha256,
            "start_state": self.start_state,
            "generated_at": self.generated_at,
        }
        if self.kind == "morph":
            payload["video_path"] = self.video_path
            return payload

        payload["error"] = self.error
        payload["encoded_frames"] = self.encoded_frames
        payload["progress_events"] = [asdict(event) for event in self.progress_events]
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        payload["integrity"] = self.integrity
        payload["video"] = self.video
        # Spore+ tips carry boss/route plan; Super+ tips also carry super_collect.
        # Post-supers kinds (red_tower, bat, …) must not drop these fields.
        if self.kind not in ("morph", "bombs"):
            payload["route_plan"] = self.route_plan or {}
            payload["policy_sources"] = self.policy_sources or {}
            payload["boss"] = self.boss.to_dict() if self.boss is not None else None
        if self.kind not in ("morph", "bombs", "spore"):
            payload["super_collect"] = (
                self.super_collect.to_dict() if self.super_collect is not None else None
            )
        return payload


@dataclass
class PlayContext:
    session: RouteSession
    splits: list[Split]
    segments: list[SegmentEvidence]


PlayFn = Callable[[PlayContext], None]


class RouteSession:
    """Power-on session with room-graph transitions and inventory evidence."""

    def __init__(
        self,
        env: object,
        *,
        writer: FrameVideoWriter | None,
        assist: AssistLike,
        graph: RoomProgressionGraph,
        room_timer: RoomTimer | None = None,
    ) -> None:
        self.env = env
        self.writer = writer
        self.assist = assist
        self.graph = graph
        self.room_timer = room_timer
        self.frame = 0
        self.info: dict[str, object] = {}
        self.state = parse_state(env.get_ram(), frame=0)  # type: ignore[attr-defined]
        self.action_reasons: Counter[str] = Counter()
        self.transitions: list[ObservedTransition] = []
        self.progress_events: list[ProgressEvent] = []
        self.bomb_torizo_activation_frame: int | None = None
        self.bomb_torizo_defeat_frame: int | None = None
        self.bomb_torizo_peak_hp = 0
        # Reset WRAM is not a meaningful room; skip first unknown → Ceres edge.
        self._last_room = (
            self.state.room_id if self.state.room_id in self.graph.rooms else 0
        )
        # Opt-in: seed the shared RoomTimer from the power-on sample.
        if self.room_timer is not None:
            self.room_timer.observe(self.state)

    def step(self, action: Action, reason: str) -> SuperMetroidState:
        previous = self.state
        obs, _, _, _, self.info = self.env.step(action)  # type: ignore[attr-defined]
        self.frame += 1
        self.state = parse_state(self.env.get_ram(), frame=self.frame)  # type: ignore[attr-defined]
        self.assist.apply(self.env.data, self.state)  # type: ignore[attr-defined]
        self.action_reasons[reason] += 1
        if self.writer is not None:
            self.writer.write(obs)
        if self.room_timer is not None:
            self.room_timer.observe(self.state)

        room = self.state.room_id
        if room in self.graph.rooms and self._last_room and room != self._last_room:
            self.transitions.append(
                self.graph.observe_transition(self.frame, self._last_room, room)
            )
        if room in self.graph.rooms:
            self._last_room = room

        self._track_inventory(previous, self.state)
        self._track_bomb_torizo(previous, self.state)
        return self.state

    def _track_inventory(
        self,
        previous: SuperMetroidState,
        state: SuperMetroidState,
    ) -> None:
        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            return
        if 0 <= previous.max_missiles < state.max_missiles <= 230:
            self.progress_events.append(
                ProgressEvent(
                    "max_missiles",
                    self.frame,
                    state.room_id,
                    previous.max_missiles,
                    state.max_missiles,
                )
            )
        new_items = state.collected_items & ~previous.collected_items
        if new_items:
            self.progress_events.append(
                ProgressEvent(
                    "collected_items",
                    self.frame,
                    state.room_id,
                    previous.collected_items,
                    state.collected_items,
                )
            )
        if 0 < previous.max_health < state.max_health <= 1499:
            self.progress_events.append(
                ProgressEvent(
                    "max_health",
                    self.frame,
                    state.room_id,
                    previous.max_health,
                    state.max_health,
                )
            )
        if 0 <= previous.max_super_missiles < state.max_super_missiles <= 50:
            self.progress_events.append(
                ProgressEvent(
                    "max_super_missiles",
                    self.frame,
                    state.room_id,
                    previous.max_super_missiles,
                    state.max_super_missiles,
                )
            )

    def _track_bomb_torizo(
        self,
        previous: SuperMetroidState,
        state: SuperMetroidState,
    ) -> None:
        if state.room_id != 0x9804:
            return
        self.bomb_torizo_peak_hp = max(self.bomb_torizo_peak_hp, state.enemy0_hp)
        if self.bomb_torizo_activation_frame is None and state.enemy0_hp >= 800:
            self.bomb_torizo_activation_frame = self.frame
        if (
            self.bomb_torizo_activation_frame is not None
            and self.bomb_torizo_defeat_frame is None
            and previous.enemy0_hp > 0
            and state.enemy0_hp == 0
        ):
            self.bomb_torizo_defeat_frame = self.frame

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


def hold(
    session: ControllerSession,
    frames: int,
    *names: str,
    reason: str,
) -> SuperMetroidState:
    """Hold a button combo for ``frames`` steps (controller helper)."""
    action = buttons(*names) if names else idle_action()
    state = session.state
    for _ in range(frames):
        state = session.step(action, reason)
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_evidence(path: Path, expected_frames: int) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    actual_frames = int(stream["nb_read_frames"])
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
        "duration_seconds": float(stream["duration"]),
        "frames": actual_frames,
        "expected_frames": expected_frames,
        "frame_count_matches": actual_frames == expected_frames,
    }


def first_progress_event(
    events: list[ProgressEvent],
    event_id: str,
    after: int,
) -> ProgressEvent:
    return next(
        event for event in events if event.event_id == event_id and event.after == after
    )


def split_for_transition(
    transitions: list[ObservedTransition],
    split_id: str,
    source: int,
    target: int,
) -> Split:
    """Record the transition just taken by the current route hop.

    A continuous route can revisit the same doorway pair (Warehouse→Business
    before Hi-Jump and again after the Varia return).  The latest matching
    observation is the hop that just completed; choosing the first corrupts
    split ordering and correctly fails integrity.
    """
    transition = next(
        item
        for item in reversed(transitions)
        if item.source_room_id == source and item.target_room_id == target
    )
    return Split(split_id, transition.frame, transition.target_room_id)


def route_plan_evidence(plan_path: Path = ROUTE_PLAN_PATH) -> dict[str, object]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    source = dict(payload["source"])
    evidence: dict[str, object] = {
        "path": str(plan_path.resolve()),
        "sha256": sha256_file(plan_path),
        "plan_id": payload["planId"],
        "status": payload["status"],
        "room_path": payload["roomPath"],
        "acceptance_warning": payload["acceptanceWarning"],
    }
    for key in ("editorNavPath", "referenceRoutePath"):
        path = Path(str(source[key]))
        evidence[key] = str(path.resolve())
        evidence[f"{key}Sha256"] = sha256_file(path)
    evidence["editorNavDeclaredSha256"] = source["editorNavSha256"]
    evidence["referenceRouteDeclaredSha256"] = source["referenceRouteSha256"]
    return evidence


def module_source_evidence(*paths: Path) -> dict[str, object]:
    return {
        path.stem: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in paths
    }


def splits_ordered(splits: list[Split], required: tuple[str, ...]) -> bool:
    frames = {split.split_id: split.frame for split in splits}
    return all(
        frames.get(left, 10**12) < frames.get(right, -1)
        for left, right in zip(required, required[1:])
    )


def all_transitions_known(transitions: list[ObservedTransition]) -> bool:
    return bool(transitions) and all(
        transition.edge_id is not None for transition in transitions
    )


def assist_integrity(
    assist: AssistLike,
    *,
    require_deaths_zero: bool = False,
) -> dict[str, bool]:
    flags = {
        "state_loads_zero": True,
        "progression_writes_zero": assist.telemetry.progression_writes == 0,
        "capacity_writes_zero": assist.telemetry.capacity_writes == 0,
    }
    if require_deaths_zero:
        flags["deaths_zero"] = assist.telemetry.deaths == 0
    return flags


def evaluate_integrity(
    *,
    required_splits: tuple[str, ...],
    splits: list[Split],
    transitions: list[ObservedTransition],
    final_conditions: dict[str, bool],
    assist: AssistLike,
    video_evidence_payload: dict[str, object] | None,
    require_deaths_zero: bool = False,
    require_transitions: bool = True,
) -> tuple[bool, dict[str, object]]:
    split_frames = {split.split_id: split.frame for split in splits}
    order_ok = splits_ordered(splits, required_splits)
    if require_transitions:
        transitions_ok = all_transitions_known(transitions)
    else:
        transitions_ok = all(
            transition.edge_id is not None for transition in transitions
        )
    assist_flags = assist_integrity(assist, require_deaths_zero=require_deaths_zero)
    video_ok = video_evidence_payload is None or bool(
        video_evidence_payload["frame_count_matches"]
    )
    integrity: dict[str, object] = {
        "all_transitions_known": transitions_ok,
        "split_order_valid": order_ok,
        "required_splits_present": all(
            split_id in split_frames for split_id in required_splits
        ),
        "final_conditions": final_conditions,
        **assist_flags,
        "video_frame_count_matches": video_ok,
    }
    success = all(
        (
            transitions_ok,
            order_ok,
            all(final_conditions.values()),
            assist.telemetry.progression_writes == 0,
            assist.telemetry.capacity_writes == 0,
            (not require_deaths_zero) or assist.telemetry.deaths == 0,
            video_ok,
        )
    )
    return success, integrity


@dataclass
class ContinuousRunResult:
    session: RouteSession
    splits: list[Split]
    segments: list[SegmentEvidence]
    final_state: SuperMetroidState
    encoded_frames: int
    video_evidence: dict[str, object] | None
    checkpoint_state: bytes | None
    failure: Exception | None
    outcome: str


def run_continuous(
    *,
    play: PlayFn,
    assist: AssistLike,
    graph: RoomProgressionGraph,
    video_path: str | Path | None = None,
    success_outcome: str = "ok",
    room_timer: RoomTimer | None = None,
    capture_checkpoint: bool = False,
) -> ContinuousRunResult:
    """Power-on once, run ``play``, always close env/writer.

    ``room_timer`` is optional instrumentation only: it observes each frame via
    the shared :class:`~super_metroid.room_timer.RoomTimer` and never feeds
    integrity, assists, or route decisions.
    """
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    writer: FrameVideoWriter | None = None
    splits: list[Split] = []
    segments: list[SegmentEvidence] = []
    session: RouteSession | None = None
    encoded_frames = 0
    outcome = "runner_error"
    failure: Exception | None = None
    video_payload: dict[str, object] | None = None
    checkpoint_state: bytes | None = None

    try:
        obs, _ = env.reset()
        if video_path is not None:
            writer = FrameVideoWriter(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
            )
            writer.write(obs)
        session = RouteSession(
            env,
            writer=writer,
            assist=assist,
            graph=graph,
            room_timer=room_timer,
        )
        play(PlayContext(session, splits, segments))
        outcome = success_outcome
    except Exception as exc:
        failure = exc
        outcome = f"failed:{type(exc).__name__}"
    finally:
        final_frame = session.frame if session is not None else 0
        final_state = parse_state(env.get_ram(), frame=final_frame)
        # Preserve the emulator snapshot in memory until finish_report has
        # accepted the run.  Callers may then write a reusable source state
        # without ever checkpointing a failed/integrity-red attempt.
        if capture_checkpoint and failure is None:
            checkpoint_state = env.em.get_state()  # type: ignore[attr-defined]
        if writer is not None:
            encoded_frames = writer.frames
            writer.close()
        env.close()

    assert session is not None
    if video_path is not None:
        video_payload = video_evidence(Path(video_path), encoded_frames)

    return ContinuousRunResult(
        session=session,
        splits=splits,
        segments=segments,
        final_state=final_state,
        encoded_frames=encoded_frames,
        video_evidence=video_payload,
        checkpoint_state=checkpoint_state,
        failure=failure,
        outcome=outcome,
    )


def finish_report(
    result: ContinuousRunResult,
    *,
    schema_version: int,
    required_splits: tuple[str, ...],
    final_conditions: dict[str, bool],
    source_policy: str,
    report_path: str | Path | None,
    route_label: str,
    kind: str = "bombs",
    require_deaths_zero: bool = False,
    require_transitions: bool = True,
    route_plan: dict[str, object] | None = None,
    policy_sources: dict[str, object] | None = None,
    boss: Any = None,
    super_collect: Any = None,
) -> ContinuousRunReport:
    """Build integrity + report, write optional JSON, raise on failure."""
    assist = result.session.assist
    if result.failure is None:
        success, integrity = evaluate_integrity(
            required_splits=required_splits,
            splits=result.splits,
            transitions=result.session.transitions,
            final_conditions=final_conditions,
            assist=assist,
            video_evidence_payload=result.video_evidence,
            require_deaths_zero=require_deaths_zero,
            require_transitions=require_transitions,
        )
        outcome = result.outcome if success else "failed:integrity"
    else:
        success = False
        outcome = result.outcome
        integrity = {
            "all_transitions_known": False,
            "split_order_valid": False,
            "required_splits_present": False,
            "final_conditions": final_conditions,
            **assist_integrity(assist, require_deaths_zero=require_deaths_zero),
            "video_frame_count_matches": (
                result.video_evidence is None
                or bool(result.video_evidence["frame_count_matches"])
            ),
        }

    report = ContinuousRunReport(
        schema_version=schema_version,
        success=success,
        outcome=outcome,
        kind=kind,
        error=str(result.failure) if result.failure is not None else None,
        total_frames=result.session.frame,
        encoded_frames=result.encoded_frames,
        final_state=result.final_state.to_dict(),
        splits=result.splits,
        progress_events=result.session.progress_events,
        transitions=result.session.transitions,
        segments=result.segments,
        action_reasons=result.session.action_reasons,
        assist=assist.report(),
        integrity=integrity,
        route_plan=route_plan,
        policy_sources=policy_sources,
        boss=boss,
        super_collect=super_collect,
        state_loads=0,
        progression_writes=assist.telemetry.progression_writes,
        video=result.video_evidence,
        source_policy=source_policy,
        rom_sha256=sha256_file(SHARED_ROM),
        start_state="power_on/retro.State.NONE",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
    if result.failure is not None:
        raise RuntimeError(
            f"{route_label} run failed; report={report_path}"
        ) from result.failure
    if not success:
        raise RuntimeError(f"{route_label} integrity failed; report={report_path}")
    return report


def default_artifacts(stem: str) -> tuple[Path, Path]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR / f"{stem}.mp4", RECORDINGS_DIR / f"{stem}.json"
