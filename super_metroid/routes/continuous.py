"""Continuous power-on route: Morph → … → Warehouse → Hi-Jump → Kraid → Varia.

One chain, one module. Each ``play_*`` extends the previous prefix; each
``run_*`` powers on, plays through that milestone, and writes a report.
Shared session/report harness lives in :mod:`super_metroid.routes.runtime`.
Controllers (movement/combat only) stay in ``*_controller.py`` / ``kpdr/``.
Segment/hop contracts: :mod:`super_metroid.routes.segment`.

Verified continuous tip: KPDR K2.6 Warehouse Entrance. Hijump / kraid / varia
tips are wired for natural-entry attachment (promote STATUS after integrity).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedAmmoAssist, UnlimitedResourcesAssist
from super_metroid.paths import POLICY_DIR, ROOM_TIMINGS_DIR, SHARED_ROM
from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    START_TO_BAT_GRAPH,
    START_TO_BELOW_SPAZER_GRAPH,
    START_TO_HIJUMP_GRAPH,
    START_TO_KRAID_GRAPH,
    START_TO_MORPH_GRAPH,
    START_TO_RED_TOWER_GRAPH,
    START_TO_SPORE_SPAWN_GRAPH,
    START_TO_VARIA_GRAPH,
    START_TO_WAREHOUSE_GRAPH,
    RoomProgressionGraph,
)
from super_metroid.ram import (
    BOMBS_MASK,
    HI_JUMP_MASK,
    MORPH_BALL_MASK,
    VARIA_MASK,
    GameplayPhase,
    SuperMetroidState,
)
from super_metroid.room_timer import RoomTimer
from super_metroid.routes.kpdr import (
    SuperCollectEvidence,
    play_baby_kraid_to_eye,
    play_bat_to_below_spazer,
    play_below_spazer_to_west,
    play_big_pink_into_main_shaft,
    play_big_pink_to_ghz,
    play_business_to_hj_shaft,
    play_business_to_warehouse,
    play_east_to_warehouse,
    play_eye_to_kraid,
    play_farming_to_big_pink,
    play_ghz_to_noob,
    play_glass_to_east,
    play_hj_room_collect,
    play_hj_room_to_shaft,
    play_hj_shaft_to_business,
    play_hj_shaft_to_hj_room,
    play_kihunter_to_baby_kraid,
    play_kraid_entry_to_varia,
    play_noob_to_red_tower,
    play_red_tower_to_bat,
    play_super_room_collect,
    play_super_room_to_farming,
    play_warehouse_to_business,
    play_warehouse_to_zeela_with_hijump,
    play_west_to_glass,
    play_zeela_to_kihunter,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_FARMING,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_VARIA,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.catalog import (
    BAT_SPLITS,
    BELOW_SPAZER_SPLITS,
    BOMBS_PREFIX_SPLITS,
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    HIJUMP_SPLITS,
    KRAID_SPLITS,
    RED_TOWER_SPLITS,
    SPORE_EXIT_SPLITS,
    SUPERS_SPLITS,
    VARIA_SPLITS,
    WAREHOUSE_SPLITS,
    ContinuousTip,
    get_continuous_tip,
    list_continuous_tips,
    register_continuous_segments,
)
from super_metroid.routes.runtime import (
    ROUTE_PLAN_PATH,
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
# Paths / report aliases kept for published scripts + tests
# ---------------------------------------------------------------------------

_THIS = Path(__file__)
SPORE_CONTROLLER_PATH = _THIS.with_name("spore_spawn_controller.py")
# Super collect lives under kpdr/; thin re-export kept for report path stability.
POST_SPORE_CONTROLLER_PATH = _THIS.with_name("post_spore_controller.py")
KPDR_SUPER_ROOM_PATH = _THIS.parent / "kpdr" / "super_room.py"
# Historical name used in supers reports / tests.
CONTROLLER_PATH = POST_SPORE_CONTROLLER_PATH

# Alias: continuous reports used to be per-milestone type names.
RunReport = ContinuousRunReport
EarlyRunReport = ContinuousRunReport
SporeRunReport = ContinuousRunReport
SupersRunReport = ContinuousRunReport
RedTowerRunReport = ContinuousRunReport

__all__ = [
    "ActionSpan",
    "Split",
    "ProgressEvent",
    "ContinuousRunReport",
    "RunReport",
    "EarlyRunReport",
    "SporeRunReport",
    "SupersRunReport",
    "RedTowerRunReport",
    "CONTROLLER_PATH",
    "ROUTE_PLAN_PATH",
    "BOMBS_PREFIX_SPLITS",
    "SPORE_EXIT_SPLITS",
    "SUPERS_SPLITS",
    "RED_TOWER_SPLITS",
    "BAT_SPLITS",
    "BELOW_SPAZER_SPLITS",
    "WAREHOUSE_SPLITS",
    "HIJUMP_SPLITS",
    "KRAID_SPLITS",
    "VARIA_SPLITS",
    "CONTINUOUS_TIPS",
    "DEFAULT_CONTINUOUS_TIP",
    "ContinuousTip",
    "get_continuous_tip",
    "list_continuous_tips",
    "play_start_to_morph",
    "play_start_to_bombs",
    "play_start_to_spore_spawn",
    "play_start_to_supers",
    "play_start_to_red_tower",
    "play_start_to_bat",
    "play_start_to_below_spazer",
    "play_start_to_warehouse",
    "play_start_to_hijump",
    "play_start_to_kraid",
    "play_start_to_varia",
    "run_start_to_morph",
    "run_start_to_bombs",
    "run_start_to_spore_spawn",
    "run_start_to_supers",
    "run_start_to_red_tower",
    "run_start_to_bat",
    "run_start_to_below_spazer",
    "run_start_to_warehouse",
    "run_start_to_hijump",
    "run_start_to_kraid",
    "run_start_to_varia",
    "run_to",
    "default_tip_artifact_paths",
    "default_tip_room_timing_path",
    "default_artifact_paths",
    "RouteHop",
    "play_hops",
    "run_post_supers_tip",
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


def write_room_timing_artifact(
    timer: RoomTimer,
    *,
    path: str | Path,
    source: str,
    route_outcome: str,
    total_frames: int,
    success: bool | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Finalize an opt-in session timer and write under ``recordings/room_timings/``.

    Instrumentation only — never part of continuous integrity evaluation.
    """
    timer.finalize(frame=total_frames)
    payload_extra: dict[str, object] = {
        "mode": "continuous_route",
        "route_outcome": route_outcome,
        "total_frames": total_frames,
    }
    if success is not None:
        payload_extra["route_success"] = success
    if extra:
        payload_extra.update(extra)
    report = timer.report(source=source, extra=payload_extra)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_start_to_supers(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Super Missile collect (STATUS baseline).

    ``room_timing_path`` is opt-in: when set, the shared :class:`RoomTimer`
    observes every frame and a separate timing JSON is written. Timing never
    affects assist, integrity, or route decisions.
    """
    assist = UnlimitedResourcesAssist(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    box: dict[str, object] = {"boss": None, "super_collect": None}
    plan = route_plan_evidence()
    timer = RoomTimer() if room_timing_path is not None else None

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
        room_timer=timer,
    )
    boss = box["boss"]
    super_collect = box["super_collect"]
    final = result.final_state
    report: ContinuousRunReport | None = None
    try:
        report = finish_report(
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
                "kpdr_super_room": {
                    "path": str(KPDR_SUPER_ROOM_PATH.resolve()),
                    "sha256": sha256_file(KPDR_SUPER_ROOM_PATH),
                },
                "route_plan": {
                    "path": str(ROUTE_PLAN_PATH.resolve()),
                    "sha256": sha256_file(ROUTE_PLAN_PATH),
                },
            },
            boss=boss if isinstance(boss, SporeSpawnEvidence) else None,
            super_collect=(
                super_collect
                if isinstance(super_collect, SuperCollectEvidence)
                else None
            ),
        )
        return report
    finally:
        # Always persist opt-in timing (including partial runs / integrity fails).
        if timer is not None and room_timing_path is not None:
            write_room_timing_artifact(
                timer,
                path=room_timing_path,
                source="start_to_supers",
                route_outcome=(
                    report.outcome if report is not None else result.outcome
                ),
                total_frames=(
                    report.total_frames
                    if report is not None
                    else result.session.frame
                ),
                success=report.success if report is not None else False,
                extra={
                    "report_path": (
                        str(report_path) if report_path is not None else None
                    ),
                    "video_path": (
                        str(video_path) if video_path is not None else None
                    ),
                },
            )


# ===========================================================================
# Post-Supers KPDR tips: hop tables + one shared runner
# ===========================================================================
# Room-by-room tip extension recipe (do not copy another run_start_to_*):
#   1. Pure controller in routes/kpdr/ (+ KPDR_SEGMENTS registry).
#   2. Graph edges/milestones in progression.py.
#   3. Split ids in catalog.py (+ ContinuousTip + NamedRoute).
#   4. Append RouteHop(s) under the tip play_* here; run_* stays a thin
#      run_post_supers_tip(...) wrapper.
#   5. Wire tip_id → run_* in run_to() + register_continuous_segments.


@dataclass(frozen=True)
class RouteHop:
    """One controller leg: play, record split, assert destination room."""

    split_id: str
    play: Callable[[RouteSession], Any]
    from_room: int
    to_room: int
    room_label: str
    #: When False, skip door-edge split lookup (in-room milestones only).
    use_transition_split: bool = True
    #: Optional extra assert after play (before room-id check).
    after: Callable[[RouteSession], None] | None = None


def play_hops(
    session: RouteSession,
    splits: list[Split],
    hops: Sequence[RouteHop],
) -> None:
    """Run ordered hops, appending splits and failing fast on wrong room."""
    for hop in hops:
        hop.play(session)
        if hop.after is not None:
            hop.after(session)
        if hop.use_transition_split:
            splits.append(
                split_for_transition(
                    session.transitions,
                    hop.split_id,
                    hop.from_room,
                    hop.to_room,
                )
            )
        else:
            splits.append(Split(hop.split_id, session.frame, session.state.room_id))
        if session.state.room_id != hop.to_room:
            raise RuntimeError(
                f"expected {hop.room_label} 0x{hop.to_room:04X}, got "
                f"0x{session.state.room_id:04X}"
            )


def _require_big_pink_main_shaft(session: RouteSession) -> None:
    if session.state.room_id != ROOM_BIG_PINK or session.state.samus_x > 750:
        raise RuntimeError(
            f"Big Pink main shaft not reached: room=0x{session.state.room_id:04X} "
            f"x={session.state.samus_x}"
        )


# K1 legs after Super collect (Charge return intentionally omitted).
_RED_TOWER_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "super_to_farming",
        play_super_room_to_farming,
        0x9B5B,
        ROOM_FARMING,
        "farming",
    ),
    RouteHop(
        "farming_to_big_pink",
        play_farming_to_big_pink,
        ROOM_FARMING,
        ROOM_BIG_PINK,
        "Big Pink",
    ),
    RouteHop(
        "big_pink_main",
        play_big_pink_into_main_shaft,
        ROOM_BIG_PINK,
        ROOM_BIG_PINK,
        "Big Pink main shaft",
        use_transition_split=False,
        after=_require_big_pink_main_shaft,
    ),
    RouteHop(
        "big_pink_to_ghz",
        play_big_pink_to_ghz,
        ROOM_BIG_PINK,
        ROOM_GHZ,
        "GHZ",
    ),
    RouteHop(
        "ghz_to_noob",
        play_ghz_to_noob,
        ROOM_GHZ,
        ROOM_NOOB,
        "Noob",
    ),
    RouteHop(
        "noob_to_red_tower",
        play_noob_to_red_tower,
        ROOM_NOOB,
        ROOM_RED_TOWER,
        "Red Tower",
    ),
)

_BAT_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "red_tower_to_bat",
        play_red_tower_to_bat,
        ROOM_RED_TOWER,
        ROOM_BAT,
        "Bat Room",
    ),
)

_BELOW_SPAZER_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "bat_to_below_spazer",
        play_bat_to_below_spazer,
        ROOM_BAT,
        ROOM_BELOW_SPAZER,
        "Below Spazer",
    ),
)

_WAREHOUSE_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "below_spazer_to_west",
        play_below_spazer_to_west,
        ROOM_BELOW_SPAZER,
        ROOM_WEST_TUNNEL,
        "West Tunnel",
    ),
    RouteHop(
        "west_to_glass",
        play_west_to_glass,
        ROOM_WEST_TUNNEL,
        ROOM_GLASS,
        "Glass Tunnel",
    ),
    RouteHop(
        "glass_to_east",
        play_glass_to_east,
        ROOM_GLASS,
        ROOM_EAST_TUNNEL,
        "East Tunnel",
    ),
    RouteHop(
        "east_to_warehouse",
        play_east_to_warehouse,
        ROOM_EAST_TUNNEL,
        ROOM_WAREHOUSE,
        "Warehouse Entrance",
    ),
)


def _require_hijump_collected(session: RouteSession) -> None:
    if not session.state.collected_items & HI_JUMP_MASK:
        raise RuntimeError(
            f"Hi-Jump not collected: items=0x{session.state.collected_items:04X}"
        )


def _require_varia_collected(session: RouteSession) -> None:
    if not session.state.collected_items & VARIA_MASK:
        raise RuntimeError(
            f"Varia not collected: items=0x{session.state.collected_items:04X}"
        )


# K2.7–K2.10: Warehouse elevator → Business → HJ shaft → HJ room collect.
_HIJUMP_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "warehouse_to_business",
        play_warehouse_to_business,
        ROOM_WAREHOUSE,
        ROOM_BUSINESS,
        "Business Center",
    ),
    RouteHop(
        "business_to_hj_shaft",
        play_business_to_hj_shaft,
        ROOM_BUSINESS,
        ROOM_HJ_SHAFT,
        "Hi-Jump shaft",
    ),
    RouteHop(
        "hj_shaft_to_hj_room",
        play_hj_shaft_to_hj_room,
        ROOM_HJ_SHAFT,
        ROOM_HJ,
        "Hi-Jump Room",
    ),
    RouteHop(
        "hijump_collected",
        play_hj_room_collect,
        ROOM_HJ,
        ROOM_HJ,
        "Hi-Jump collect",
        use_transition_split=False,
        after=_require_hijump_collected,
    ),
)

# K2.11–K2.18: HJ return → Warehouse → Zeela → … → natural Kraid entry.
_KRAID_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "hj_room_to_shaft",
        play_hj_room_to_shaft,
        ROOM_HJ,
        ROOM_HJ_SHAFT,
        "Hi-Jump shaft return",
    ),
    RouteHop(
        "hj_shaft_to_business",
        play_hj_shaft_to_business,
        ROOM_HJ_SHAFT,
        ROOM_BUSINESS,
        "Business Center return",
    ),
    RouteHop(
        "business_to_warehouse",
        play_business_to_warehouse,
        ROOM_BUSINESS,
        ROOM_WAREHOUSE,
        "Warehouse return",
    ),
    RouteHop(
        "warehouse_to_zeela",
        play_warehouse_to_zeela_with_hijump,
        ROOM_WAREHOUSE,
        ROOM_ZEELA,
        "Warehouse Zeela",
    ),
    RouteHop(
        "zeela_to_kihunter",
        play_zeela_to_kihunter,
        ROOM_ZEELA,
        ROOM_WAREHOUSE_KIHUNTER,
        "Warehouse Kihunter",
    ),
    RouteHop(
        "kihunter_to_baby_kraid",
        play_kihunter_to_baby_kraid,
        ROOM_WAREHOUSE_KIHUNTER,
        ROOM_BABY_KRAID,
        "Baby Kraid",
    ),
    RouteHop(
        "baby_kraid_to_eye",
        play_baby_kraid_to_eye,
        ROOM_BABY_KRAID,
        ROOM_KRAID_EYE,
        "Kraid Eye Door",
    ),
    RouteHop(
        "eye_to_kraid",
        play_eye_to_kraid,
        ROOM_KRAID_EYE,
        ROOM_KRAID,
        "Kraid's Room",
    ),
)

# K3: fight + rear exit + Varia PLM (multi-room; graph has kraid→varia edge).
_VARIA_HOPS: tuple[RouteHop, ...] = (
    RouteHop(
        "kraid_to_varia",
        play_kraid_entry_to_varia,
        ROOM_KRAID,
        ROOM_VARIA,
        "Varia Suit Room",
        use_transition_split=False,
        after=_require_varia_collected,
    ),
)

_KPDR_POLICY_SOURCES: dict[str, object] = {
    "continuous_route_module": {
        "path": str(_THIS.resolve()),
        "sha256": sha256_file(_THIS),
    },
    "post_torizo_controller": {
        "path": str(SPORE_CONTROLLER_PATH.resolve()),
        "sha256": sha256_file(SPORE_CONTROLLER_PATH),
    },
    "kpdr_super_room": {
        "path": str(KPDR_SUPER_ROOM_PATH.resolve()),
        "sha256": sha256_file(KPDR_SUPER_ROOM_PATH),
    },
    "kpdr_package": {
        "path": str((_THIS.parent / "kpdr").resolve()),
        "note": "K1/K2 segment controllers under routes/kpdr/",
    },
}


def _post_supers_final_conditions(
    final: SuperMetroidState,
    boss: object,
    *,
    room_id: int,
    entry_key: str,
    ordinary_key: str,
    extra: dict[str, bool] | None = None,
) -> dict[str, bool]:
    """Shared Super+ inventory checks plus tip-room ordinary gameplay."""
    conditions = {
        "both_missile_expansions": final.max_missiles >= 10,
        "morph_and_bombs": (
            final.collected_items & (MORPH_BALL_MASK | BOMBS_MASK)
            == MORPH_BALL_MASK | BOMBS_MASK
        ),
        "terminator_energy_tank": final.max_health >= 199,
        "super_missiles_collected": final.max_super_missiles >= 5,
        "spore_spawn_hp_reached_zero": (
            isinstance(boss, SporeSpawnEvidence) and 0 in boss.observed_hp
        ),
        entry_key: (
            final.room_id == room_id
            and final.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ),
        ordinary_key: (
            final.room_id == room_id
            and final.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and final.game_state == 8
        ),
    }
    if extra:
        conditions.update(extra)
    return conditions


def run_post_supers_tip(
    play_fn: Callable[
        [RouteSession, list[Split], list[SegmentEvidence]],
        tuple[SporeSpawnEvidence, SuperCollectEvidence],
    ],
    *,
    graph: RoomProgressionGraph,
    kind: str,
    required_splits: tuple[str, ...],
    final_room: int,
    success_outcome: str,
    route_label: str,
    source_policy: str,
    timing_source: str,
    entry_condition_key: str,
    ordinary_condition_key: str,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
    extra_final_conditions: Callable[[SuperMetroidState], dict[str, bool]]
    | None = None,
) -> ContinuousRunReport:
    """Shared power-on harness for every Super+ continuous tip."""
    assist = UnlimitedResourcesAssist(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    box: dict[str, object] = {"boss": None, "super_collect": None}
    plan = route_plan_evidence()
    timer = RoomTimer() if room_timing_path is not None else None

    def play(ctx: PlayContext) -> None:
        boss, super_collect = play_fn(ctx.session, ctx.splits, ctx.segments)
        box["boss"] = boss
        box["super_collect"] = super_collect

    result = run_continuous(
        play=play,
        assist=assist,
        graph=graph,
        video_path=video_path,
        success_outcome=success_outcome,
        room_timer=timer,
    )
    boss = box["boss"]
    super_collect = box["super_collect"]
    final = result.final_state
    extra = extra_final_conditions(final) if extra_final_conditions else None
    report: ContinuousRunReport | None = None
    try:
        report = finish_report(
            result,
            schema_version=1,
            kind=kind,
            required_splits=required_splits,
            final_conditions=_post_supers_final_conditions(
                final,
                boss,
                room_id=final_room,
                entry_key=entry_condition_key,
                ordinary_key=ordinary_condition_key,
                extra=extra,
            ),
            source_policy=source_policy,
            report_path=report_path,
            route_label=route_label,
            require_deaths_zero=True,
            route_plan=plan,
            policy_sources=dict(_KPDR_POLICY_SOURCES),
            boss=boss if isinstance(boss, SporeSpawnEvidence) else None,
            super_collect=(
                super_collect
                if isinstance(super_collect, SuperCollectEvidence)
                else None
            ),
        )
        return report
    finally:
        if timer is not None and room_timing_path is not None:
            write_room_timing_artifact(
                timer,
                path=room_timing_path,
                source=timing_source,
                route_outcome=(
                    report.outcome if report is not None else result.outcome
                ),
                total_frames=(
                    report.total_frames
                    if report is not None
                    else result.session.frame
                ),
                success=report.success if report is not None else False,
                extra={
                    "report_path": (
                        str(report_path) if report_path is not None else None
                    ),
                    "video_path": (
                        str(video_path) if video_path is not None else None
                    ),
                },
            )


def play_start_to_red_tower(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Supers prefix + natural Super exit through Red Tower entry.

    Charge Beam is intentionally skipped (side trip; conventional return not
    route-ready). Path: Super room → farming → Big Pink → crest/main shaft →
    GHZ → Noob → Red Tower.
    """
    boss, super_collect = play_start_to_supers(session, splits, segments)
    play_hops(session, splits, _RED_TOWER_HOPS)
    return boss, super_collect


def run_start_to_red_tower(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Red Tower entry (KPDR K1 tip)."""
    return run_post_supers_tip(
        play_start_to_red_tower,
        graph=START_TO_RED_TOWER_GRAPH,
        kind="red_tower",
        required_splits=RED_TOWER_SPLITS,
        final_room=ROOM_RED_TOWER,
        success_outcome="red_tower_entry",
        route_label="start-to-Red-Tower",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1 controllers "
            "(Super→farm→Big Pink main→GHZ→Noob→Red) + phase-guarded resources"
        ),
        timing_source="start_to_red_tower",
        entry_condition_key="natural_red_tower_entry",
        ordinary_condition_key="post_red_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
    )


def play_start_to_bat(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Red Tower prefix + natural Red Tower descent into Bat Room."""
    boss, super_collect = play_start_to_red_tower(session, splits, segments)
    play_hops(session, splits, _BAT_HOPS)
    return boss, super_collect


def run_start_to_bat(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Bat Room entry (KPDR K2.0 tip)."""
    return run_post_supers_tip(
        play_start_to_bat,
        graph=START_TO_BAT_GRAPH,
        kind="bat",
        required_splits=BAT_SPLITS,
        final_room=ROOM_BAT,
        success_outcome="bat_room_entry",
        route_label="start-to-Bat-Room",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2.0 "
            "controllers (Super→farm→Big Pink main→GHZ→Noob→Red→Bat) + "
            "phase-guarded resources"
        ),
        timing_source="start_to_bat",
        entry_condition_key="natural_bat_room_entry",
        ordinary_condition_key="post_bat_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
    )


def play_start_to_below_spazer(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Bat prefix + natural three-platform crossing into Below Spazer."""
    boss, super_collect = play_start_to_bat(session, splits, segments)
    play_hops(session, splits, _BELOW_SPAZER_HOPS)
    return boss, super_collect


def run_start_to_below_spazer(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Below Spazer entry (KPDR K2.1 tip)."""
    return run_post_supers_tip(
        play_start_to_below_spazer,
        graph=START_TO_BELOW_SPAZER_GRAPH,
        kind="below_spazer",
        required_splits=BELOW_SPAZER_SPLITS,
        final_room=ROOM_BELOW_SPAZER,
        success_outcome="below_spazer_entry",
        route_label="start-to-Below-Spazer",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Red→Bat→Below Spazer) + phase-guarded resources"
        ),
        timing_source="start_to_below_spazer",
        entry_condition_key="natural_below_spazer_entry",
        ordinary_condition_key="post_below_spazer_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
    )


def play_start_to_warehouse(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Below Spazer prefix + natural tunnel chain into Warehouse Entrance."""
    boss, super_collect = play_start_to_below_spazer(session, splits, segments)
    play_hops(session, splits, _WAREHOUSE_HOPS)
    return boss, super_collect


def run_start_to_warehouse(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Warehouse Entrance (KPDR K2.6 tip)."""
    return run_post_supers_tip(
        play_start_to_warehouse,
        graph=START_TO_WAREHOUSE_GRAPH,
        kind="warehouse",
        required_splits=WAREHOUSE_SPLITS,
        final_room=ROOM_WAREHOUSE,
        success_outcome="warehouse_entry",
        route_label="start-to-Warehouse-Entrance",
        source_policy=(
            "accepted power-on prefix + Spore controller + KPDR K1/K2 "
            "controllers (…→Bat→Below Spazer→West→Glass→East→Warehouse) + "
            "phase-guarded resources"
        ),
        timing_source="start_to_warehouse",
        entry_condition_key="natural_warehouse_entry",
        ordinary_condition_key="post_warehouse_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
    )


def play_start_to_hijump(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Warehouse prefix + natural Hi-Jump Boots collect."""
    boss, super_collect = play_start_to_warehouse(session, splits, segments)
    play_hops(session, splits, _HIJUMP_HOPS)
    return boss, super_collect


def run_start_to_hijump(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Hi-Jump collect (KPDR K2.10)."""
    return run_post_supers_tip(
        play_start_to_hijump,
        graph=START_TO_HIJUMP_GRAPH,
        kind="hijump",
        required_splits=HIJUMP_SPLITS,
        final_room=ROOM_HJ,
        success_outcome="hijump_collected",
        route_label="start-to-Hi-Jump",
        source_policy=(
            "accepted Warehouse continuous prefix + KPDR Hi-Jump controllers "
            "(Warehouse→Business→shaft→HJ room collect) + phase-guarded resources"
        ),
        timing_source="start_to_hijump",
        entry_condition_key="natural_hijump_room",
        ordinary_condition_key="post_hijump_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
        extra_final_conditions=lambda final: {
            "hi_jump_collected": bool(final.collected_items & HI_JUMP_MASK),
        },
    )


def play_start_to_kraid(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Hi-Jump prefix + return + Warehouse approach into natural Kraid entry."""
    boss, super_collect = play_start_to_hijump(session, splits, segments)
    play_hops(session, splits, _KRAID_HOPS)
    return boss, super_collect


def run_start_to_kraid(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Kraid room entry (KPDR K2.18)."""
    return run_post_supers_tip(
        play_start_to_kraid,
        graph=START_TO_KRAID_GRAPH,
        kind="kraid",
        required_splits=KRAID_SPLITS,
        final_room=ROOM_KRAID,
        success_outcome="kraid_entry",
        route_label="start-to-Kraid-entry",
        source_policy=(
            "accepted Warehouse prefix + Hi-Jump collect/return + KPDR Kraid "
            "approach controllers + phase-guarded resources"
        ),
        timing_source="start_to_kraid",
        entry_condition_key="natural_kraid_entry",
        ordinary_condition_key="post_kraid_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
        extra_final_conditions=lambda final: {
            "hi_jump_collected": bool(final.collected_items & HI_JUMP_MASK),
        },
    )


def play_start_to_varia(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Kraid-entry prefix + fight + rear exit + natural Varia collect."""
    boss, super_collect = play_start_to_kraid(session, splits, segments)
    play_hops(session, splits, _VARIA_HOPS)
    return boss, super_collect


def run_start_to_varia(
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Varia collect (KPDR K3)."""
    return run_post_supers_tip(
        play_start_to_varia,
        graph=START_TO_VARIA_GRAPH,
        kind="varia",
        required_splits=VARIA_SPLITS,
        final_room=ROOM_VARIA,
        success_outcome="varia_collected",
        route_label="start-to-Varia",
        source_policy=(
            "accepted Kraid-entry continuous chain + combat.kraid fight/Varia "
            "policy + phase-guarded resources"
        ),
        timing_source="start_to_varia",
        entry_condition_key="natural_varia_room",
        ordinary_condition_key="post_varia_ordinary",
        video_path=video_path,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
        extra_final_conditions=lambda final: {
            "hi_jump_collected": bool(final.collected_items & HI_JUMP_MASK),
            "varia_collected": bool(final.collected_items & VARIA_MASK),
        },
    )


# ===========================================================================
# Tip dispatch + artifact paths
# ===========================================================================
# One CLI entry (scripts/record/continuous.py --to <tip>) calls run_to().
# Extend via RouteHop tables + ContinuousTip — not a new start_to_*.py file.


def _resolve_tip(tip: str | ContinuousTip | None = None) -> ContinuousTip:
    if tip is None:
        return get_continuous_tip(DEFAULT_CONTINUOUS_TIP)
    if isinstance(tip, ContinuousTip):
        return tip
    return get_continuous_tip(tip)


def default_tip_artifact_paths(tip: str | ContinuousTip | None = None) -> tuple[Path, Path]:
    """Video/report paths for a continuous tip (default: current tip)."""
    return default_artifacts(_resolve_tip(tip).artifact_stem)


def default_tip_room_timing_path(tip: str | ContinuousTip | None = None) -> Path:
    """Opt-in room-timing JSON path for a tip (gitignored)."""
    resolved = _resolve_tip(tip)
    ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    return ROOM_TIMINGS_DIR / f"{resolved.artifact_stem}_room_timing.json"


def default_artifact_paths() -> tuple[Path, Path]:
    """Video/report paths for the current continuous tip (Warehouse)."""
    return default_tip_artifact_paths()


def run_to(
    tip: str = DEFAULT_CONTINUOUS_TIP,
    *,
    video_path: str | Path | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
) -> ContinuousRunReport:
    """Power-on once through a named continuous tip (``--to`` target).

    Tips compose as prefixes:
    morph ⊂ … ⊂ warehouse ⊂ hijump ⊂ kraid ⊂ varia.
    Room-timing is only accepted for tips that declare support in the catalog.
    Default tip remains verified Warehouse until later tips are integrity-green.
    """
    resolved = get_continuous_tip(tip)
    runners: dict[str, Callable[..., ContinuousRunReport]] = {
        "morph": run_start_to_morph,
        "bombs": run_start_to_bombs,
        "spore": run_start_to_spore_spawn,
        "supers": run_start_to_supers,
        "red_tower": run_start_to_red_tower,
        "bat": run_start_to_bat,
        "below_spazer": run_start_to_below_spazer,
        "warehouse": run_start_to_warehouse,
        "hijump": run_start_to_hijump,
        "kraid": run_start_to_kraid,
        "varia": run_start_to_varia,
    }
    runner = runners[resolved.tip_id]
    kwargs: dict[str, object] = {
        "video_path": video_path,
        "report_path": report_path,
        "unlimited_ammo": unlimited_ammo,
    }
    if resolved.supports_unlimited_energy:
        kwargs["unlimited_energy"] = unlimited_energy
    elif not unlimited_energy:
        raise ValueError(
            f"tip {resolved.tip_id!r} has no unlimited-energy assist "
            f"(energy assist starts at spore+)"
        )
    if room_timing_path is not None:
        if not resolved.supports_room_timing:
            raise ValueError(
                f"tip {resolved.tip_id!r} does not support room timing "
                f"(supported: supers+ tips with supports_room_timing)"
            )
        kwargs["room_timing_path"] = room_timing_path
    return runner(**kwargs)


register_continuous_segments(
    {
        "morph": play_start_to_morph,
        "bombs": play_start_to_bombs,
        "spore": play_start_to_spore_spawn,
        "supers": play_start_to_supers,
        "red_tower": play_start_to_red_tower,
        "bat": play_start_to_bat,
        "below_spazer": play_start_to_below_spazer,
        "warehouse": play_start_to_warehouse,
        "hijump": play_start_to_hijump,
        "kraid": play_start_to_kraid,
        "varia": play_start_to_varia,
        # Historical segment keys (reports / probes).
        "start_to_morph": play_start_to_morph,
        "start_to_bombs": play_start_to_bombs,
        "start_to_spore_spawn": play_start_to_spore_spawn,
        "start_to_supers": play_start_to_supers,
        "start_to_red_tower": play_start_to_red_tower,
        "start_to_bat": play_start_to_bat,
        "start_to_below_spazer": play_start_to_below_spazer,
        "start_to_warehouse": play_start_to_warehouse,
        "start_to_hijump": play_start_to_hijump,
        "start_to_kraid": play_start_to_kraid,
        "start_to_varia": play_start_to_varia,
        "run_start_to_morph": run_start_to_morph,
        "run_start_to_bombs": run_start_to_bombs,
        "run_start_to_spore_spawn": run_start_to_spore_spawn,
        "run_start_to_supers": run_start_to_supers,
        "run_start_to_red_tower": run_start_to_red_tower,
        "run_start_to_bat": run_start_to_bat,
        "run_start_to_below_spazer": run_start_to_below_spazer,
        "run_start_to_warehouse": run_start_to_warehouse,
        "run_start_to_hijump": run_start_to_hijump,
        "run_start_to_kraid": run_start_to_kraid,
        "run_start_to_varia": run_start_to_varia,
        "run_to": run_to,
    }
)
