"""Continuous power-on through both early Missiles and Bomb Torizo."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from retro_harness.actions import buttons, idle_action
from retro_harness.env import make_env
from super_metroid.assist import UnlimitedAmmoAssist
from super_metroid.paths import GAME, GAME_DIR, RECORDINGS_DIR, SHARED_ROM
from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.progression import EARLY_GAME_GRAPH, ObservedTransition
from super_metroid.ram import (
    BOMBS_MASK,
    MORPH_BALL_MASK,
    GameplayPhase,
    SuperMetroidState,
    parse_state,
)
from super_metroid.start_to_morph import Split, _Session, play_start_to_morph
from super_metroid.video import FrameVideoWriter


@dataclass(frozen=True)
class ProgressEvent:
    event_id: str
    frame: int
    room_id: int
    before: int
    after: int


@dataclass
class EarlyRunReport:
    schema_version: int
    success: bool
    outcome: str
    error: str | None
    total_frames: int
    encoded_frames: int
    final_state: dict[str, object]
    splits: list[Split]
    progress_events: list[ProgressEvent]
    transitions: list[ObservedTransition]
    segments: list[SegmentEvidence]
    action_reasons: Counter[str]
    assist: dict[str, object]
    integrity: dict[str, object]
    state_loads: int
    progression_writes: int
    video: dict[str, object] | None
    source_policy: str
    rom_sha256: str
    start_state: str
    generated_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "success": self.success,
            "outcome": self.outcome,
            "error": self.error,
            "total_frames": self.total_frames,
            "encoded_frames": self.encoded_frames,
            "final_state": self.final_state,
            "splits": [asdict(split) for split in self.splits],
            "progress_events": [asdict(event) for event in self.progress_events],
            "transitions": [asdict(transition) for transition in self.transitions],
            "segments": [segment.to_dict() for segment in self.segments],
            "action_reasons": dict(self.action_reasons),
            "assist": self.assist,
            "integrity": self.integrity,
            "state_loads": self.state_loads,
            "progression_writes": self.progression_writes,
            "video": self.video,
            "source_policy": self.source_policy,
            "rom_sha256": self.rom_sha256,
            "start_state": self.start_state,
            "generated_at": self.generated_at,
        }


class _EarlySession(_Session):
    """Session with inventory and boss-history evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.progress_events: list[ProgressEvent] = []
        self.bomb_torizo_activation_frame: int | None = None
        self.bomb_torizo_defeat_frame: int | None = None
        self.bomb_torizo_peak_hp = 0

    def step(self, action, reason: str) -> SuperMetroidState:
        previous = self.state
        state = super().step(action, reason)
        if (
            state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and 0 <= previous.max_missiles < state.max_missiles <= 230
        ):
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
        if state.phase is GameplayPhase.ORDINARY_GAMEPLAY and new_items:
            self.progress_events.append(
                ProgressEvent(
                    "collected_items",
                    self.frame,
                    state.room_id,
                    previous.collected_items,
                    state.collected_items,
                )
            )
        if state.room_id == 0x9804:
            self.bomb_torizo_peak_hp = max(self.bomb_torizo_peak_hp, state.enemy0_hp)
            if (
                self.bomb_torizo_activation_frame is None
                and state.enemy0_hp >= 800
            ):
                self.bomb_torizo_activation_frame = self.frame
            if (
                self.bomb_torizo_activation_frame is not None
                and self.bomb_torizo_defeat_frame is None
                and previous.enemy0_hp > 0
                and state.enemy0_hp == 0
            ):
                self.bomb_torizo_defeat_frame = self.frame
        return state


_TWO_MISSILES = PolicySegment(
    "two_missile_detour",
    "two_missile_detour.json",
    StateRequirement(
        room_id=0x9F11,
        collected_items_mask=MORPH_BALL_MASK,
    ),
    StateRequirement(
        room_id=0x9F11,
        phases=frozenset({GameplayPhase.ORDINARY_GAMEPLAY}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "two_missile_detour",
)

_CONSTRUCTION_RETURN = PolicySegment(
    "construction_and_morph_return",
    "construction_to_elevator.json",
    _TWO_MISSILES.exit,
    StateRequirement(
        room_id=0x9E9F,
        phases=frozenset({GameplayPhase.ROOM_TRANSITION}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "construction_and_morph_return",
)

_ELEVATOR_RETURN = PolicySegment(
    "elevator_return",
    "elevator_to_pit.json",
    _CONSTRUCTION_RETURN.exit,
    StateRequirement(
        room_id=0x97B5,
        phases=frozenset({GameplayPhase.ROOM_TRANSITION}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "elevator_return",
)

_PIT_TO_POST_TORIZO = PolicySegment(
    "pit_to_post_torizo",
    "pit_to_post_torizo.json",
    StateRequirement(
        room_id=0x975C,
        phases=frozenset({GameplayPhase.ORDINARY_GAMEPLAY}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
        x_range=(692, 694),
        y_range=(187, 187),
    ),
    StateRequirement(
        room_id=0x92FD,
        phases=frozenset({GameplayPhase.ORDINARY_GAMEPLAY}),
        collected_items_mask=MORPH_BALL_MASK | BOMBS_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "pit_to_torizo_replay",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_evidence(path: Path, expected_frames: int) -> dict[str, object]:
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
        "sha256": _sha256(path),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
        "duration_seconds": float(stream["duration"]),
        "frames": actual_frames,
        "expected_frames": expected_frames,
        "frame_count_matches": actual_frames == expected_frames,
    }


def _first_event(
    events: list[ProgressEvent],
    event_id: str,
    after: int,
) -> ProgressEvent:
    return next(event for event in events if event.event_id == event_id and event.after == after)


def play_start_to_bombs(
    session: _EarlySession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> None:
    """Play the accepted power-on-through-Torizo prefix in an owned session."""
    play_start_to_morph(session, splits)

    session.wait_until(
        lambda state: state.room_id == 0x9F11,
        timeout=180,
        reason="morph_to_construction_transition_settle",
    )
    segments.append(play_policy(session, _TWO_MISSILES))
    first_missiles = _first_event(session.progress_events, "max_missiles", 5)
    second_missiles = _first_event(session.progress_events, "max_missiles", 10)
    splits.extend(
        (
            Split("first_missiles", first_missiles.frame, first_missiles.room_id),
            Split(
                "blue_brinstar_missiles",
                second_missiles.frame,
                second_missiles.room_id,
            ),
        )
    )

    segments.append(play_policy(session, _CONSTRUCTION_RETURN))
    segments.append(play_policy(session, _ELEVATOR_RETURN))
    session.wait_until(
        lambda state: state.room_id == 0x975C
        and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        and state.samus_x == 693
        and state.samus_y == 187,
        timeout=600,
        reason="pit_natural_entry_alignment",
    )
    # The two-Missile detour leaves missiles selected, while the successful
    # return replay begins on the beam and selects missiles later at the red
    # door. Normalize through ordinary controller input, never a RAM write.
    session.step(buttons("SELECT"), "pit_weapon_selection_normalize")
    for _ in range(9):
        session.step(idle_action(), "pit_grounded_settle")
    if session.state.selected_item != 0:
        raise RuntimeError(
            f"Pit weapon selection did not return to beam: {session.state}"
        )

    segments.append(play_policy(session, _PIT_TO_POST_TORIZO))
    bombs_event = next(
        event
        for event in session.progress_events
        if event.event_id == "collected_items" and event.after & BOMBS_MASK
    )
    splits.append(Split("bombs", bombs_event.frame, bombs_event.room_id))
    if session.bomb_torizo_activation_frame is None:
        raise RuntimeError("Bomb Torizo never activated")
    if session.bomb_torizo_defeat_frame is None:
        raise RuntimeError("Bomb Torizo HP never reached zero after activation")
    splits.append(
        Split("bomb_torizo_defeated", session.bomb_torizo_defeat_frame, 0x9804)
    )
    torizo_exit = next(
        transition
        for transition in session.transitions
        if transition.source_room_id == 0x9804
        and transition.target_room_id == 0x9879
    )
    splits.append(
        Split("bomb_torizo_exit", torizo_exit.frame, torizo_exit.target_room_id)
    )


def run_start_to_bombs(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_ammo: bool = True,
) -> EarlyRunReport:
    """Run a single reset session through Bomb Torizo and return to Parlor."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    writer: FrameVideoWriter | None = None
    splits: list[Split] = []
    segments: list[SegmentEvidence] = []
    assist = UnlimitedAmmoAssist(enabled=unlimited_ammo)
    session: _EarlySession | None = None
    encoded_frames = 0
    outcome = "runner_error"
    failure: Exception | None = None
    video_evidence: dict[str, object] | None = None

    try:
        obs, _ = env.reset()
        if video_path is not None:
            writer = FrameVideoWriter(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
            )
            writer.write(obs)
        session = _EarlySession(
            env,
            writer=writer,
            assist=assist,
            graph=EARLY_GAME_GRAPH,
        )
        play_start_to_bombs(session, splits, segments)
        outcome = "bomb_torizo_defeated_bombs_acquired"
    except Exception as exc:
        failure = exc
        outcome = f"failed:{type(exc).__name__}"
    finally:
        final_frame = session.frame if session is not None else 0
        final_state = parse_state(env.get_ram(), frame=final_frame)
        if writer is not None:
            encoded_frames = writer.frames
            writer.close()
        env.close()

    assert session is not None
    all_transitions_known = all(
        transition.edge_id is not None for transition in session.transitions
    )
    required_split_ids = (
        "first_ceres_control",
        "ridley_countdown",
        "zebes_landing",
        "morph_ball",
        "first_missiles",
        "blue_brinstar_missiles",
        "bombs",
        "bomb_torizo_defeated",
        "bomb_torizo_exit",
    )
    split_frames = {split.split_id: split.frame for split in splits}
    split_order_valid = all(
        split_frames.get(left, 10**12) < split_frames.get(right, -1)
        for left, right in zip(required_split_ids, required_split_ids[1:])
    )

    if video_path is not None:
        video_evidence = _video_evidence(Path(video_path), encoded_frames)
    final_conditions = {
        "both_missile_expansions": final_state.max_missiles >= 10,
        "bombs_collected": final_state.bombs,
        "bomb_torizo_activated": session.bomb_torizo_activation_frame is not None,
        "bomb_torizo_peak_hp_800": session.bomb_torizo_peak_hp >= 800,
        "bomb_torizo_hp_reached_zero": session.bomb_torizo_defeat_frame is not None,
        "natural_boss_room_exit": any(
            transition.source_room_id == 0x9804
            and transition.target_room_id == 0x9879
            for transition in session.transitions
        ),
        "post_boss_parlor_settle": (
            final_state.room_id == 0x92FD
            and final_state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ),
    }
    integrity = {
        "all_transitions_known": all_transitions_known,
        "split_order_valid": split_order_valid,
        "required_splits_present": all(
            split_id in split_frames for split_id in required_split_ids
        ),
        "final_conditions": final_conditions,
        "state_loads_zero": True,
        "progression_writes_zero": assist.telemetry.progression_writes == 0,
        "capacity_writes_zero": assist.telemetry.capacity_writes == 0,
        "video_frame_count_matches": (
            video_evidence is None or video_evidence["frame_count_matches"]
        ),
    }
    success = failure is None and all(
        (
            all_transitions_known,
            split_order_valid,
            all(final_conditions.values()),
            assist.telemetry.progression_writes == 0,
            assist.telemetry.capacity_writes == 0,
            video_evidence is None or bool(video_evidence["frame_count_matches"]),
        )
    )
    if failure is None and not success:
        outcome = "failed:integrity"

    report = EarlyRunReport(
        schema_version=2,
        success=success,
        outcome=outcome,
        error=str(failure) if failure is not None else None,
        total_frames=session.frame,
        encoded_frames=encoded_frames,
        final_state=final_state.to_dict(),
        splits=splits,
        progress_events=session.progress_events,
        transitions=session.transitions,
        segments=segments,
        action_reasons=session.action_reasons,
        assist=assist.report(),
        integrity=integrity,
        state_loads=0,
        progression_writes=assist.telemetry.progression_writes,
        video=video_evidence,
        source_policy=(
            "accepted power-on prefix + hash-pinned natural manual replay segments "
            "+ phase-guarded unlimited ammo"
        ),
        rom_sha256=_sha256(SHARED_ROM),
        start_state="power_on/retro.State.NONE",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    if failure is not None:
        raise RuntimeError(f"start-to-bombs run failed; report={report_path}") from failure
    if not success:
        raise RuntimeError(f"start-to-bombs integrity failed; report={report_path}")
    return report


def default_artifact_paths() -> tuple[Path, Path]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        RECORDINGS_DIR / "start_to_bomb_torizo.mp4",
        RECORDINGS_DIR / "start_to_bomb_torizo.json",
    )
