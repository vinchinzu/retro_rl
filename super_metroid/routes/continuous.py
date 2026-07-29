"""Continuous power-on route: Morph → Bombs/Torizo → Spore → Supers.

One chain, one module. Each ``play_*`` extends the previous prefix; each
``run_*`` powers on, plays through that milestone, and writes a report.
Shared session/report harness lives in :mod:`super_metroid.routes.runtime`.
Controllers (movement/combat only) stay in ``*_controller.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedAmmoAssist, UnlimitedResourcesAssist
from super_metroid.paths import POLICY_DIR, SHARED_ROM
from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    START_TO_MORPH_GRAPH,
    START_TO_SPORE_SPAWN_GRAPH,
)
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK, GameplayPhase
from super_metroid.routes.post_spore_controller import (
    SuperCollectEvidence,
    play_super_room_collect,
)
from super_metroid.routes.runtime import (
    BOMBS_PREFIX_SPLITS,
    ROUTE_PLAN_PATH,
    SPORE_EXIT_SPLITS,
    SUPERS_SPLITS,
    Action,
    ActionSpan,
    ContinuousRunReport,
    PlayContext,
    ProgressEvent,
    RouteSession,
    Split,
    default_artifacts,
    finish_report,
    first_progress_event,
    route_plan_evidence,
    run_continuous,
    sha256_file,
    split_for_transition,
    video_evidence,
)
from super_metroid.routes.spore_spawn_controller import (
    SporeSpawnEvidence,
    play_post_torizo_to_spore_spawn,
)

# ---------------------------------------------------------------------------
# Paths / back-compat aliases
# ---------------------------------------------------------------------------

_THIS = Path(__file__)
SPORE_CONTROLLER_PATH = _THIS.with_name("spore_spawn_controller.py")
POST_SPORE_CONTROLLER_PATH = _THIS.with_name("post_spore_controller.py")
# Historical name used in supers reports / tests.
CONTROLLER_PATH = POST_SPORE_CONTROLLER_PATH

RunReport = ContinuousRunReport
EarlyRunReport = ContinuousRunReport
SporeRunReport = ContinuousRunReport
SupersRunReport = ContinuousRunReport
_Session = RouteSession
_EarlySession = RouteSession
_SporeSession = RouteSession
_SupersSession = RouteSession
_sha256 = sha256_file
_video_evidence = video_evidence
_route_plan_evidence = route_plan_evidence
_split_for_transition = split_for_transition

__all__ = [
    "ActionSpan",
    "Split",
    "ProgressEvent",
    "ContinuousRunReport",
    "RunReport",
    "EarlyRunReport",
    "SporeRunReport",
    "SupersRunReport",
    "CONTROLLER_PATH",
    "ROUTE_PLAN_PATH",
    "play_start_to_morph",
    "play_start_to_bombs",
    "play_start_to_spore_spawn",
    "play_start_to_supers",
    "run_start_to_morph",
    "run_start_to_bombs",
    "run_start_to_spore_spawn",
    "run_start_to_supers",
    "default_morph_artifact_paths",
    "default_bombs_artifact_paths",
    "default_spore_artifact_paths",
    "default_supers_artifact_paths",
    "default_artifact_paths",
]


# ===========================================================================
# Morph Ball (power-on → Morph)
# ===========================================================================


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


def play_start_to_morph(session: RouteSession, splits: list[Split]) -> None:
    """Power-on through natural Morph Ball collect."""
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
        session.raw_actions(_load_room_seed(index, name), f"seed_{name}")

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
) -> ContinuousRunReport:
    """Power-on once; stop after Morph Ball."""
    assist = UnlimitedAmmoAssist(enabled=unlimited_ammo)

    def play(ctx: PlayContext) -> None:
        play_start_to_morph(ctx.session, ctx.splits)

    result = run_continuous(
        play=play,
        assist=assist,
        graph=START_TO_MORPH_GRAPH,
        video_path=video_path,
        success_outcome="morph_ball_acquired",
    )
    if result.failure is not None:
        raise result.failure

    report = ContinuousRunReport(
        schema_version=1,
        success=True,
        outcome=result.outcome,
        kind="morph",
        total_frames=result.session.frame,
        final_state=result.final_state.to_dict(),
        splits=result.splits,
        transitions=result.session.transitions,
        action_reasons=result.session.action_reasons,
        assist=assist.report(),
        state_loads=0,
        progression_writes=assist.telemetry.progression_writes,
        video_path=str(Path(video_path).resolve()) if video_path is not None else None,
        source_policy="power-on Ceres policy + imported natural-entry room seeds",
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
    return report


# ===========================================================================
# Bombs / Bomb Torizo
# ===========================================================================

_TWO_MISSILES = PolicySegment(
    "two_missile_detour",
    "two_missile_detour.json",
    StateRequirement(room_id=0x9F11, collected_items_mask=MORPH_BALL_MASK),
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


def play_start_to_bombs(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> None:
    """Morph prefix + two Missiles + Bomb Torizo + Parlor settle."""
    play_start_to_morph(session, splits)

    session.wait_until(
        lambda state: state.room_id == 0x9F11,
        timeout=180,
        reason="morph_to_construction_transition_settle",
    )
    segments.append(play_policy(session, _TWO_MISSILES))
    first_missiles = first_progress_event(session.progress_events, "max_missiles", 5)
    second_missiles = first_progress_event(session.progress_events, "max_missiles", 10)
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
) -> ContinuousRunReport:
    """Power-on once; stop after Bomb Torizo exit into Parlor."""
    assist = UnlimitedAmmoAssist(enabled=unlimited_ammo)

    def play(ctx: PlayContext) -> None:
        play_start_to_bombs(ctx.session, ctx.splits, ctx.segments)

    result = run_continuous(
        play=play,
        assist=assist,
        graph=EARLY_GAME_GRAPH,
        video_path=video_path,
        success_outcome="bomb_torizo_defeated_bombs_acquired",
    )
    final = result.final_state
    session = result.session
    return finish_report(
        result,
        schema_version=2,
        kind="bombs",
        required_splits=BOMBS_PREFIX_SPLITS,
        final_conditions={
            "both_missile_expansions": final.max_missiles >= 10,
            "bombs_collected": final.bombs,
            "bomb_torizo_activated": session.bomb_torizo_activation_frame is not None,
            "bomb_torizo_peak_hp_800": session.bomb_torizo_peak_hp >= 800,
            "bomb_torizo_hp_reached_zero": session.bomb_torizo_defeat_frame is not None,
            "natural_boss_room_exit": any(
                t.source_room_id == 0x9804 and t.target_room_id == 0x9879
                for t in session.transitions
            ),
            "post_boss_parlor_settle": (
                final.room_id == 0x92FD
                and final.phase is GameplayPhase.ORDINARY_GAMEPLAY
            ),
        },
        source_policy=(
            "accepted power-on prefix + hash-pinned natural manual replay segments "
            "+ phase-guarded unlimited ammo"
        ),
        report_path=report_path,
        route_label="start-to-bombs",
        require_transitions=False,
    )


# ===========================================================================
# Spore Spawn exit
# ===========================================================================


def play_start_to_spore_spawn(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> SporeSpawnEvidence:
    """Bombs prefix + post-Torizo controller through natural Spore exit."""
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
            split_for_transition(
                session.transitions, "green_brinstar_main_shaft", 0x9938, 0x9AD9
            ),
            Split("spore_spawn_activated", boss.activation_frame, 0x9DC7),
            Split("spore_spawn_defeated", boss.defeat_frame, 0x9DC7),
            split_for_transition(
                session.transitions, "spore_spawn_exit", 0x9DC7, 0x9B5B
            ),
        )
    )
    return boss


def run_start_to_spore_spawn(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
) -> ContinuousRunReport:
    """Power-on once; stop in Super room after Spore Spawn exit."""
    assist = UnlimitedResourcesAssist(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    boss_box: dict[str, SporeSpawnEvidence | None] = {"boss": None}
    plan = route_plan_evidence()

    def play(ctx: PlayContext) -> None:
        boss_box["boss"] = play_start_to_spore_spawn(
            ctx.session, ctx.splits, ctx.segments
        )

    result = run_continuous(
        play=play,
        assist=assist,
        graph=START_TO_SPORE_SPAWN_GRAPH,
        video_path=video_path,
        success_outcome="spore_spawn_defeated_and_exited",
    )
    boss = boss_box["boss"]
    final = result.final_state
    return finish_report(
        result,
        schema_version=1,
        kind="spore",
        required_splits=SPORE_EXIT_SPLITS,
        final_conditions={
            "both_missile_expansions": final.max_missiles >= 10,
            "morph_and_bombs": (
                final.collected_items & (MORPH_BALL_MASK | BOMBS_MASK)
                == MORPH_BALL_MASK | BOMBS_MASK
            ),
            "terminator_energy_tank": final.max_health >= 199,
            "spore_spawn_activated_at_960_hp": boss is not None and boss.peak_hp >= 960,
            "spore_spawn_hp_reached_zero": boss is not None and 0 in boss.observed_hp,
            "vulnerable_mouth_states_observed": (
                boss is not None and bool(boss.vulnerable_spritemaps)
            ),
            "natural_spore_room_exit": any(
                t.source_room_id == 0x9DC7 and t.target_room_id == 0x9B5B
                for t in result.session.transitions
            ),
            "post_spore_room_settle": (
                final.room_id == 0x9B5B
                and final.phase is GameplayPhase.ORDINARY_GAMEPLAY
            ),
        },
        source_policy=(
            "accepted power-on prefix + checked read-only post-Torizo controller "
            "+ editor-precalculated room plan + phase-guarded current resources"
        ),
        report_path=report_path,
        route_label="start-to-Spore-Spawn",
        require_deaths_zero=True,
        route_plan=plan,
        policy_sources={
            "continuous_route_module": {
                "path": str(_THIS.resolve()),
                "sha256": sha256_file(_THIS),
            },
            "post_torizo_controller": {
                "path": str(SPORE_CONTROLLER_PATH.resolve()),
                "sha256": sha256_file(SPORE_CONTROLLER_PATH),
            },
        },
        boss=boss,
    )


# ===========================================================================
# Super Missile collect (current continuous baseline)
# ===========================================================================


def play_start_to_supers(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Spore-exit prefix, then natural Super collect in room ``0x9B5B``."""
    boss = play_start_to_spore_spawn(session, splits, segments)

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
    return boss, super_collect


def run_start_to_supers(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
) -> ContinuousRunReport:
    """Power-on once through natural Super Missile collect (STATUS baseline)."""
    assist = UnlimitedResourcesAssist(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    box: dict[str, object] = {"boss": None, "super_collect": None}
    plan = route_plan_evidence()

    def play(ctx: PlayContext) -> None:
        boss, super_collect = play_start_to_supers(
            ctx.session, ctx.splits, ctx.segments
        )
        box["boss"] = boss
        box["super_collect"] = super_collect

    result = run_continuous(
        play=play,
        assist=assist,
        graph=START_TO_SPORE_SPAWN_GRAPH,
        video_path=video_path,
        success_outcome="spore_supers_collected",
    )
    boss = box["boss"]
    super_collect = box["super_collect"]
    final = result.final_state
    return finish_report(
        result,
        schema_version=1,
        kind="supers",
        required_splits=SUPERS_SPLITS,
        final_conditions={
            "both_missile_expansions": final.max_missiles >= 10,
            "morph_and_bombs": (
                final.collected_items & (MORPH_BALL_MASK | BOMBS_MASK)
                == MORPH_BALL_MASK | BOMBS_MASK
            ),
            "terminator_energy_tank": final.max_health >= 199,
            "spore_spawn_activated_at_960_hp": (
                isinstance(boss, SporeSpawnEvidence) and boss.peak_hp >= 960
            ),
            "spore_spawn_hp_reached_zero": (
                isinstance(boss, SporeSpawnEvidence) and 0 in boss.observed_hp
            ),
            "natural_spore_room_exit": any(
                t.source_room_id == 0x9DC7 and t.target_room_id == 0x9B5B
                for t in result.session.transitions
            ),
            "super_missiles_collected": final.max_super_missiles >= 5,
            "super_collect_in_super_room": (
                isinstance(super_collect, SuperCollectEvidence)
                and super_collect.max_super_missiles >= 5
                and super_collect.final_room_id == 0x9B5B
            ),
            "post_super_ordinary": (
                final.room_id == 0x9B5B
                and final.phase is GameplayPhase.ORDINARY_GAMEPLAY
            ),
        },
        source_policy=(
            "accepted power-on prefix + Spore controller + post-Spore Super "
            "controller + phase-guarded current resources"
        ),
        report_path=report_path,
        route_label="start-to-Supers",
        require_deaths_zero=True,
        route_plan=plan,
        policy_sources={
            "continuous_route_module": {
                "path": str(_THIS.resolve()),
                "sha256": sha256_file(_THIS),
            },
            "post_torizo_controller": {
                "path": str(SPORE_CONTROLLER_PATH.resolve()),
                "sha256": sha256_file(SPORE_CONTROLLER_PATH),
            },
            "post_spore_controller": {
                "path": str(POST_SPORE_CONTROLLER_PATH.resolve()),
                "sha256": sha256_file(POST_SPORE_CONTROLLER_PATH),
            },
            "route_plan": {
                "path": str(ROUTE_PLAN_PATH.resolve()),
                "sha256": sha256_file(ROUTE_PLAN_PATH),
            },
        },
        boss=boss if isinstance(boss, SporeSpawnEvidence) else None,
        super_collect=(
            super_collect if isinstance(super_collect, SuperCollectEvidence) else None
        ),
    )


# ===========================================================================
# Artifact paths
# ===========================================================================


def default_morph_artifact_paths() -> tuple[Path, Path]:
    return default_artifacts("start_to_morph")


def default_bombs_artifact_paths() -> tuple[Path, Path]:
    return default_artifacts("start_to_bomb_torizo")


def default_spore_artifact_paths() -> tuple[Path, Path]:
    return default_artifacts("start_to_spore_spawn")


def default_supers_artifact_paths() -> tuple[Path, Path]:
    return default_artifacts("start_to_supers")


# Primary baseline alias (most scripts want Supers).
default_artifact_paths = default_supers_artifact_paths
