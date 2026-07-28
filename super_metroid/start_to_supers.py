"""Continuous power-on runner through Spore Super Missile collect.

Composes the accepted start-to-Spore-Spawn prefix with
``post_spore_controller.play_super_room_collect``. No save-state loads after
power-on; resource assists only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from retro_harness.env import make_env
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.paths import GAME, GAME_DIR, MAPS_DIR, RECORDINGS_DIR, SHARED_ROM
from super_metroid.policy import SegmentEvidence
from super_metroid.post_spore_controller import (
    SuperCollectEvidence,
    play_super_room_collect,
)
from super_metroid.progression import ObservedTransition, START_TO_SPORE_SPAWN_GRAPH
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK, GameplayPhase, parse_state
from super_metroid.spore_spawn_controller import (
    SporeSpawnEvidence,
    play_post_torizo_to_spore_spawn,
)
from super_metroid.start_to_bombs import (
    ProgressEvent,
    _sha256,
    _video_evidence,
    play_start_to_bombs,
)
from super_metroid.start_to_morph import Split
from super_metroid.start_to_spore_spawn import (
    ROUTE_PLAN_PATH,
    _SporeSession,
    _route_plan_evidence,
    _split_for_transition,
)
from super_metroid.video import FrameVideoWriter

CONTROLLER_PATH = Path(__file__).with_name("post_spore_controller.py")
SPORE_CONTROLLER_PATH = Path(__file__).with_name("spore_spawn_controller.py")
PREFIX_PATH = Path(__file__).with_name("start_to_bombs.py")


@dataclass
class SupersRunReport:
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
    boss: SporeSpawnEvidence | None
    super_collect: SuperCollectEvidence | None
    action_reasons: Counter[str]
    assist: dict[str, object]
    integrity: dict[str, object]
    route_plan: dict[str, object]
    policy_sources: dict[str, object]
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
            "boss": self.boss.to_dict() if self.boss is not None else None,
            "super_collect": (
                self.super_collect.to_dict() if self.super_collect is not None else None
            ),
            "action_reasons": dict(self.action_reasons),
            "assist": self.assist,
            "integrity": self.integrity,
            "route_plan": self.route_plan,
            "policy_sources": self.policy_sources,
            "state_loads": self.state_loads,
            "progression_writes": self.progression_writes,
            "video": self.video,
            "source_policy": self.source_policy,
            "rom_sha256": self.rom_sha256,
            "start_state": self.start_state,
            "generated_at": self.generated_at,
        }


class _SupersSession(_SporeSession):
    """Spore session plus Super Missile capacity evidence."""

    def step(self, action, reason: str):
        previous = self.state
        state = super().step(action, reason)
        if (
            state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and 0 <= previous.max_super_missiles < state.max_super_missiles <= 50
        ):
            self.progress_events.append(
                ProgressEvent(
                    "max_super_missiles",
                    self.frame,
                    state.room_id,
                    previous.max_super_missiles,
                    state.max_super_missiles,
                )
            )
        return state


def run_start_to_supers(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
) -> SupersRunReport:
    """Power-on through natural Super Missile collect in room ``0x9B5B``."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    writer: FrameVideoWriter | None = None
    splits: list[Split] = []
    segments: list[SegmentEvidence] = []
    assist = UnlimitedResourcesAssist(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    session: _SupersSession | None = None
    boss: SporeSpawnEvidence | None = None
    super_collect: SuperCollectEvidence | None = None
    encoded_frames = 0
    outcome = "runner_error"
    failure: Exception | None = None
    video_evidence: dict[str, object] | None = None
    route_plan = _route_plan_evidence()

    try:
        obs, _ = env.reset()
        if video_path is not None:
            writer = FrameVideoWriter(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
            )
            writer.write(obs)
        session = _SupersSession(
            env,
            writer=writer,
            assist=assist,
            graph=START_TO_SPORE_SPAWN_GRAPH,
        )
        play_start_to_bombs(session, splits, segments)
        boss = play_post_torizo_to_spore_spawn(session)

        energy = next(
            event
            for event in session.progress_events
            if event.event_id == "max_health" and event.after >= 199
        )
        splits.append(Split("terminator_energy_tank", energy.frame, energy.room_id))
        splits.extend(
            (
                _split_for_transition(
                    session.transitions,
                    "green_brinstar_main_shaft",
                    0x9938,
                    0x9AD9,
                ),
                Split("spore_spawn_activated", boss.activation_frame, 0x9DC7),
                Split("spore_spawn_defeated", boss.defeat_frame, 0x9DC7),
                _split_for_transition(
                    session.transitions,
                    "spore_spawn_exit",
                    0x9DC7,
                    0x9B5B,
                ),
            )
        )

        # Natural Super collect in the Spore Super room.
        if session.state.room_id != 0x9B5B:
            raise RuntimeError(
                f"expected Super room 0x9B5B after Spore exit, got "
                f"0x{session.state.room_id:04X}"
            )
        if session.state.max_super_missiles != 0:
            raise RuntimeError(
                f"expected zero Super capacity at Super-room entry, got "
                f"{session.state.max_super_missiles}"
            )
        super_collect = play_super_room_collect(session)
        splits.append(
            Split(
                "spore_supers_collected",
                super_collect.collect_frame,
                session.state.room_id,
            )
        )
        outcome = "spore_supers_collected"
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
        "terminator_energy_tank",
        "green_brinstar_main_shaft",
        "spore_spawn_activated",
        "spore_spawn_defeated",
        "spore_spawn_exit",
        "spore_supers_collected",
    )
    split_frames = {split.split_id: split.frame for split in splits}
    split_order_valid = all(
        split_frames.get(left, 10**12) < split_frames.get(right, -1)
        for left, right in zip(required_split_ids, required_split_ids[1:])
    )
    all_transitions_known = bool(session.transitions) and all(
        transition.edge_id is not None for transition in session.transitions
    )
    if video_path is not None:
        video_evidence = _video_evidence(Path(video_path), encoded_frames)

    final_conditions = {
        "both_missile_expansions": final_state.max_missiles >= 10,
        "morph_and_bombs": (
            final_state.collected_items & (MORPH_BALL_MASK | BOMBS_MASK)
            == MORPH_BALL_MASK | BOMBS_MASK
        ),
        "terminator_energy_tank": final_state.max_health >= 199,
        "spore_spawn_activated_at_960_hp": boss is not None and boss.peak_hp >= 960,
        "spore_spawn_hp_reached_zero": boss is not None and 0 in boss.observed_hp,
        "natural_spore_room_exit": any(
            transition.source_room_id == 0x9DC7
            and transition.target_room_id == 0x9B5B
            for transition in session.transitions
        ),
        "super_missiles_collected": final_state.max_super_missiles >= 5,
        "super_collect_in_super_room": (
            super_collect is not None
            and super_collect.max_super_missiles >= 5
            and super_collect.final_room_id == 0x9B5B
        ),
        "post_super_ordinary": (
            final_state.room_id == 0x9B5B
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
        "deaths_zero": assist.telemetry.deaths == 0,
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
            assist.telemetry.deaths == 0,
            video_evidence is None or bool(video_evidence["frame_count_matches"]),
        )
    )
    if failure is None and not success:
        outcome = "failed:integrity"

    report = SupersRunReport(
        schema_version=1,
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
        boss=boss,
        super_collect=super_collect,
        action_reasons=session.action_reasons,
        assist=assist.report(),
        integrity=integrity,
        route_plan=route_plan,
        policy_sources={
            "accepted_prefix_module": {
                "path": str(PREFIX_PATH.resolve()),
                "sha256": _sha256(PREFIX_PATH),
            },
            "post_torizo_controller": {
                "path": str(SPORE_CONTROLLER_PATH.resolve()),
                "sha256": _sha256(SPORE_CONTROLLER_PATH),
            },
            "post_spore_controller": {
                "path": str(CONTROLLER_PATH.resolve()),
                "sha256": _sha256(CONTROLLER_PATH),
            },
            "route_plan": {
                "path": str(ROUTE_PLAN_PATH.resolve()),
                "sha256": _sha256(ROUTE_PLAN_PATH),
            },
        },
        state_loads=0,
        progression_writes=assist.telemetry.progression_writes,
        video=video_evidence,
        source_policy=(
            "accepted power-on prefix + Spore controller + post-Spore Super "
            "controller + phase-guarded current resources"
        ),
        rom_sha256=_sha256(SHARED_ROM),
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
    if failure is not None:
        raise RuntimeError(
            f"start-to-Supers run failed; report={report_path}"
        ) from failure
    if not success:
        raise RuntimeError(
            f"start-to-Supers integrity failed; report={report_path}"
        )
    return report


def default_artifact_paths() -> tuple[Path, Path]:
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        RECORDINGS_DIR / "start_to_supers.mp4",
        RECORDINGS_DIR / "start_to_supers.json",
    )
