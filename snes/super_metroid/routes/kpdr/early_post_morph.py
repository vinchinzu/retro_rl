"""Bombs → Spore → Supers continuous spine — SpineHop orchestration.

**Model (same as Morph / Super+):** ordered :class:`~super_metroid.routes.kpdr.spine.SpineHop`
rows with multi-split bookkeeping on hop ``after`` hooks. TipSpec rows register
these spines; :func:`~super_metroid.routes.tips.play_tip` walks the parent chain
and runs :func:`~super_metroid.routes.tips.play_hops`.

**Unchanged timing surface:** hash-pinned :class:`~super_metroid.policy.PolicySegment`
JSON paths and StateRequirements, Bomb Torizo policy combat, Spore open-loop
controller holds, and Super-room collect controller are **not** retuned — only
the composition order is SpineHop rows calling those helpers.

**Graph edges / milestones** for bombs→spore stay hand-authored in
``progression/data.py`` (this wave does not relocate DoorEdges). Split ids for
continuous reports remain the historical event-based names
(:data:`~super_metroid.routes.catalog.BOMBS_PREFIX_SPLITS` etc.), not one hop_id
per intermediate leg.
"""

from __future__ import annotations

from collections.abc import Sequence

from retro_harness.actions import buttons, idle_action
from super_metroid.policy import (
    PolicySegment,
    SegmentEvidence,
    StateRequirement,
    play_policy,
)
from super_metroid.ram import BOMBS_MASK, MORPH_BALL_MASK, GameplayPhase
from super_metroid.routes.controller_common import select_weapon
from super_metroid.routes.kpdr.early_spine import play_morph_hops
from super_metroid.routes.kpdr.room_ids import (
    ROOM_BLUE_BRINSTAR_ELEVATOR,
    ROOM_CONSTRUCTION,
    ROOM_GREEN_MAIN_SHAFT,
    ROOM_MORPH,
    ROOM_PARLOR,
    ROOM_PIT,
    ROOM_SUPER,
)
from super_metroid.routes.kpdr.spine import SpineHop
from super_metroid.routes.kpdr.super_collect import (
    SuperCollectEvidence,
    play_super_room_collect,
)
from super_metroid.routes.runtime import (
    RouteSession,
    Split,
    first_progress_event,
    split_for_transition,
)
from super_metroid.routes.kpdr.spore_spawn import (
    SporeSpawnEvidence,
    play_main_shaft_to_spore_spawn,
    play_parlor_to_main_shaft,
)

__all__ = [
    "BOMBS_SPINE",
    "SPORE_SPINE",
    "SUPERS_SPINE",
    "play_bombs_hops",
    "play_spore_hops",
    "play_supers_hops",
    "validate_bombs_spine",
    "validate_spore_spine",
    "validate_supers_spine",
    # Policy segments (public for tests / evidence introspection)
    "TWO_MISSILES",
    "CONSTRUCTION_RETURN",
    "ELEVATOR_RETURN",
    "PIT_TO_POST_TORIZO",
]


# ---------------------------------------------------------------------------
# Hash-pinned policy segments (byte-identical requirements / JSON paths)
# ---------------------------------------------------------------------------


TWO_MISSILES = PolicySegment(
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

CONSTRUCTION_RETURN = PolicySegment(
    "construction_and_morph_return",
    "construction_to_elevator.json",
    TWO_MISSILES.exit,
    StateRequirement(
        room_id=0x9E9F,
        phases=frozenset({GameplayPhase.ROOM_TRANSITION}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "construction_and_morph_return",
)

ELEVATOR_RETURN = PolicySegment(
    "elevator_return",
    "elevator_to_pit.json",
    CONSTRUCTION_RETURN.exit,
    StateRequirement(
        room_id=0x97B5,
        phases=frozenset({GameplayPhase.ROOM_TRANSITION}),
        collected_items_mask=MORPH_BALL_MASK,
        minimum_ammo_capacities=(10, 0, 0),
    ),
    "elevator_return",
)

PIT_TO_POST_TORIZO = PolicySegment(
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

# Hash-pinned exit tail after a hybrid Bomb Torizo fight (door open + Flyway +
# Parlor settle). Measured: last 2000 frames of pit_to_post_torizo.json reach
# parlor from post-boss-bit BT room under both assist and clean.
PIT_TO_POST_TORIZO_EXIT_TAIL_START = -2000

# BT room / pre-combat spritemaps (duplicated to avoid combat↔routes import cycle).
_ROOM_BOMB_TORIZO = 0x9804
_STATUE_SPRITEMAP = 0x87D0
_SPAWN_SPRITEMAP = 0x804F


# ---------------------------------------------------------------------------
# Hop play callables (session-only; policy bytes / open-loop unchanged)
# ---------------------------------------------------------------------------


def play_two_missile_detour(session: RouteSession) -> SegmentEvidence:
    """Morph room → Construction settle + two-Missile detour."""
    session.wait_until(
        lambda state: state.room_id == ROOM_CONSTRUCTION,
        timeout=180,
        reason="morph_to_construction_transition_settle",
    )
    return play_policy(session, TWO_MISSILES)


def play_construction_return(session: RouteSession) -> SegmentEvidence:
    """Construction zone → Morph via elevator path (return segment)."""
    return play_policy(session, CONSTRUCTION_RETURN)


def play_elevator_return(session: RouteSession) -> SegmentEvidence:
    """Blue Brinstar elevator → Pit approach transition."""
    return play_policy(session, ELEVATOR_RETURN)


def play_pit_natural_entry(session: RouteSession) -> None:
    """Align Pit natural entry and normalize weapon to beam.

    Assisted two-Missile detour exits with missiles selected (sel=1); clean
    often exits on beam (sel=0) after ammo runs dry. One fixed SELECT when
    already on beam would arm missiles and break the climb entry. Match the
    historical 1+9 settle budget either way.
    """
    session.wait_until(
        lambda state: state.room_id == ROOM_PIT
        and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        and state.samus_x == 693
        and state.samus_y == 187,
        timeout=600,
        reason="pit_natural_entry_alignment",
    )
    if session.state.selected_item != 0:
        session.step(buttons("SELECT"), "pit_weapon_selection_normalize")
    else:
        session.step(idle_action(), "pit_weapon_already_beam")
    for _ in range(9):
        session.step(idle_action(), "pit_grounded_settle")
    if session.state.selected_item != 0:
        select_weapon(session, 0)
        for _ in range(9):
            session.step(idle_action(), "pit_grounded_settle")
    if session.state.selected_item != 0:
        raise RuntimeError(
            f"Pit weapon selection did not return to beam: {session.state}"
        )


def _assist_ammo_enabled(session: RouteSession) -> bool:
    """True when unlimited ammo assist is actively refilling missiles."""
    assist = getattr(session, "assist", None)
    if assist is None:
        return False
    if hasattr(assist, "enabled"):
        return bool(assist.enabled)
    report = assist.report() if hasattr(assist, "report") else {}
    return bool(report.get("unlimited_ammo_enabled", report.get("enabled", False)))


def _bt_fight_ready(state) -> bool:
    """Natural BT activation: in-room, bombs held, combat spritemap, full HP bar."""
    if state.room_id != _ROOM_BOMB_TORIZO:
        return False
    if not (state.collected_items & BOMBS_MASK):
        return False
    if state.enemy0_spritemap in (_STATUE_SPRITEMAP, _SPAWN_SPRITEMAP, 0):
        return False
    return state.enemy0_hp >= 800


def play_pit_to_post_torizo(session: RouteSession) -> SegmentEvidence:
    """Pit natural entry → Bomb Torizo + Bombs + Parlor settle.

    Assisted path keeps the full hash-pinned replay (116 missile refill writes
    in the accepted baseline). Clean path (ammo assist off) hybrids:

    1. Policy until natural BT activation
    2. :func:`play_bomb_torizo_fight` with the clean kite defaults
    3. Hash-pinned exit tail through Flyway → Parlor

    Assisted baselines are never overwritten; clean only writes ``*_clean``
    stems. Combat imports are lazy to avoid routes↔combat cycles at import time.
    """
    if _assist_ammo_enabled(session):
        return play_policy(session, PIT_TO_POST_TORIZO)

    # Lazy import: combat.natural_entry → continuous → early_post_morph cycle.
    from super_metroid.combat.bomb_torizo import (
        BombTorizoStrategy,
        play_bomb_torizo_fight,
    )

    # Clean hybrid: policy prefix → structured fight → policy exit tail.
    prefix = play_policy(
        session,
        PIT_TO_POST_TORIZO,
        stop_when=_bt_fight_ready,
        require_exit=False,
    )
    if not _bt_fight_ready(session.state):
        raise RuntimeError(
            "clean pit_to_post_torizo: policy prefix ended without BT activation; "
            f"state room=0x{session.state.room_id:04X} hp={session.state.enemy0_hp} "
            f"sm=0x{session.state.enemy0_spritemap:04X} items=0x{session.state.collected_items:04X}"
        )

    fight = play_bomb_torizo_fight(
        session,
        strategy=BombTorizoStrategy(),
        require_active=True,
        require_boss_bit=True,
    )
    if fight.outcome != "bomb_torizo_defeated":
        raise RuntimeError(
            f"clean Bomb Torizo fight failed: {fight.outcome} "
            f"min_hp={fight.min_enemy_hp} final_hp={fight.final_enemy_hp} "
            f"frames={fight.action_frames}"
        )

    tail = play_policy(
        session,
        PIT_TO_POST_TORIZO,
        require_exit=True,
        action_slice=slice(PIT_TO_POST_TORIZO_EXIT_TAIL_START, None),
    )
    # Merge evidence so report still shows one segment with combined frames.
    return SegmentEvidence(
        segment_id=prefix.segment_id,
        policy_path=prefix.policy_path,
        policy_sha256=prefix.policy_sha256,
        source_sha256=prefix.source_sha256,
        source_slice=f"{prefix.source_slice}+clean_bt_hybrid",
        start_frame=prefix.start_frame,
        end_frame=tail.end_frame,
        action_frames=tail.end_frame - prefix.start_frame,
        max_identical_navigation_frames=max(
            prefix.max_identical_navigation_frames,
            tail.max_identical_navigation_frames,
        ),
        start_button_frames=prefix.start_button_frames + tail.start_button_frames,
        opposite_direction_frames=(
            prefix.opposite_direction_frames + tail.opposite_direction_frames
        ),
        entry_state=prefix.entry_state,
        exit_state=tail.exit_state,
    )


def play_parlor_to_green_main(session: RouteSession) -> None:
    """Post-Torizo Parlor → Green Brinstar Main Shaft (Spore controller)."""
    play_parlor_to_main_shaft(session)


def play_main_shaft_to_spore_exit(session: RouteSession) -> SporeSpawnEvidence:
    """Main Shaft → Spore fight → natural Super-room exit."""
    return play_main_shaft_to_spore_spawn(session)


def play_super_missile_collect(session: RouteSession) -> SuperCollectEvidence:
    """Natural Super Missile collect in room ``0x9B5B``."""
    if session.state.room_id != ROOM_SUPER:
        raise RuntimeError(
            f"expected Super room 0x{ROOM_SUPER:04X} after Spore exit, got "
            f"0x{session.state.room_id:04X}"
        )
    if session.state.max_super_missiles != 0:
        raise RuntimeError(
            f"expected zero Super capacity at Super-room entry, got "
            f"{session.state.max_super_missiles}"
        )
    return play_super_room_collect(session)


# ---------------------------------------------------------------------------
# Historical split bookkeeping (report schema; hop after hooks)
# ---------------------------------------------------------------------------


def _append_missile_splits(session: RouteSession, splits: list[Split]) -> None:
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


def _append_bomb_torizo_splits(session: RouteSession, splits: list[Split]) -> None:
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
        if transition.source_room_id == 0x9804 and transition.target_room_id == 0x9879
    )
    splits.append(
        Split("bomb_torizo_exit", torizo_exit.frame, torizo_exit.target_room_id)
    )


def _append_spore_route_splits(session: RouteSession, splits: list[Split]) -> None:
    energy = next(
        event
        for event in session.progress_events
        if event.event_id == "max_health" and event.after >= 199
    )
    splits.append(Split("terminator_energy_tank", energy.frame, energy.room_id))
    splits.append(
        split_for_transition(
            session.transitions, "green_brinstar_main_shaft", 0x9938, 0x9AD9
        )
    )


def _append_spore_boss_splits(
    session: RouteSession,
    splits: list[Split],
    boss: SporeSpawnEvidence,
) -> None:
    splits.extend(
        (
            Split("spore_spawn_activated", boss.activation_frame, 0x9DC7),
            Split("spore_spawn_defeated", boss.defeat_frame, 0x9DC7),
            split_for_transition(
                session.transitions, "spore_spawn_exit", 0x9DC7, 0x9B5B
            ),
        )
    )


def _after_two_missile_detour(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    del result  # policy SegmentEvidence; multi-splits come from progress events
    _append_missile_splits(session, splits)


def _after_pit_to_post_torizo(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    del result
    _append_bomb_torizo_splits(session, splits)


def _after_parlor_to_main_shaft(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    del result
    _append_spore_route_splits(session, splits)


def _after_main_shaft_to_spore_exit(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    if not isinstance(result, SporeSpawnEvidence):
        raise RuntimeError(
            f"main_shaft_to_spore_exit expected SporeSpawnEvidence, got {type(result)!r}"
        )
    _append_spore_boss_splits(session, splits, result)


def _after_supers_collect(
    session: RouteSession, splits: list[Split], result: object = None
) -> None:
    if not isinstance(result, SuperCollectEvidence):
        raise RuntimeError(
            f"spore_supers_collected expected SuperCollectEvidence, got {type(result)!r}"
        )
    # Use collect_frame (not session.frame); play_hops skips auto hop_id split.
    splits.append(
        Split("spore_supers_collected", result.collect_frame, session.state.room_id)
    )


# ---------------------------------------------------------------------------
# Spines (orchestration only; tip_id tags the continuous tip that owns the leg)
# ---------------------------------------------------------------------------

BOMBS_SPINE: tuple[SpineHop, ...] = (
    SpineHop(
        "two_missile_detour",
        play_two_missile_detour,
        ROOM_MORPH,
        ROOM_CONSTRUCTION,
        "Construction Zone (two missiles)",
        "bombs",
        use_transition_split=False,
        after=_after_two_missile_detour,
        policy_id="two_missile_detour",
    ),
    SpineHop(
        "construction_return",
        play_construction_return,
        ROOM_CONSTRUCTION,
        ROOM_MORPH,
        "Morph Ball Room (return)",
        "bombs",
        use_transition_split=False,
        policy_id="construction_and_morph_return",
    ),
    SpineHop(
        "elevator_return",
        play_elevator_return,
        ROOM_MORPH,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        "Blue Brinstar Elevator (return)",
        "bombs",
        use_transition_split=False,
        policy_id="elevator_return",
    ),
    SpineHop(
        "pit_natural_entry",
        play_pit_natural_entry,
        ROOM_BLUE_BRINSTAR_ELEVATOR,
        ROOM_PIT,
        "Pit natural entry",
        "bombs",
        use_transition_split=False,
        policy_id="pit_natural_entry",
    ),
    SpineHop(
        "pit_to_post_torizo",
        play_pit_to_post_torizo,
        ROOM_PIT,
        ROOM_PARLOR,
        "Parlor (post Bomb Torizo)",
        "bombs",
        use_transition_split=False,
        after=_after_pit_to_post_torizo,
        policy_id="pit_to_torizo_replay",
    ),
)

SPORE_SPINE: tuple[SpineHop, ...] = (
    SpineHop(
        "parlor_to_main_shaft",
        play_parlor_to_green_main,
        ROOM_PARLOR,
        ROOM_GREEN_MAIN_SHAFT,
        "Green Brinstar Main Shaft",
        "spore",
        use_transition_split=False,
        after=_after_parlor_to_main_shaft,
        policy_id="post_torizo_controller",
    ),
    SpineHop(
        "main_shaft_to_spore_exit",
        play_main_shaft_to_spore_exit,
        ROOM_GREEN_MAIN_SHAFT,
        ROOM_SUPER,
        "Spore Super room exit",
        "spore",
        use_transition_split=False,
        after=_after_main_shaft_to_spore_exit,
        policy_id="post_torizo_controller",
    ),
)

SUPERS_SPINE: tuple[SpineHop, ...] = (
    SpineHop(
        "spore_supers_collected",
        play_super_missile_collect,
        ROOM_SUPER,
        ROOM_SUPER,
        "Super Missile collect",
        "supers",
        use_transition_split=False,
        after=_after_supers_collect,
        policy_id="super_room_collect",
    ),
)


# ---------------------------------------------------------------------------
# Tip runners (back-compat; prefer play_tip(tip_id) for new call sites)
# ---------------------------------------------------------------------------


def play_bombs_hops(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
    spine: Sequence[SpineHop] = BOMBS_SPINE,
) -> None:
    """Morph prefix + bombs SpineHop legs through post-Torizo Parlor settle."""
    from super_metroid.routes.tips import play_hops

    play_morph_hops(session, splits)
    play_hops(session, splits, spine, segments=segments)


def play_spore_hops(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
    spine: Sequence[SpineHop] = SPORE_SPINE,
) -> SporeSpawnEvidence:
    """Bombs prefix + Spore SpineHop legs through natural Super-room exit."""
    from super_metroid.routes.tips import play_hops

    play_bombs_hops(session, splits, segments)
    result = play_hops(session, splits, spine, segments=segments)
    if not isinstance(result, SporeSpawnEvidence):
        raise RuntimeError(
            f"Spore spine expected SporeSpawnEvidence, got {type(result)!r}"
        )
    return result


def play_supers_hops(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
    spine: Sequence[SpineHop] = SUPERS_SPINE,
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Spore-exit prefix + Super collect SpineHop."""
    from super_metroid.routes.tips import play_hops

    boss = play_spore_hops(session, splits, segments)
    result = play_hops(session, splits, spine, segments=segments)
    if not isinstance(result, SuperCollectEvidence):
        raise RuntimeError(
            f"Supers spine expected SuperCollectEvidence, got {type(result)!r}"
        )
    return boss, result


# ---------------------------------------------------------------------------
# Dev validation (import-time)
# ---------------------------------------------------------------------------


def validate_bombs_spine(spine: Sequence[SpineHop] = BOMBS_SPINE) -> None:
    hop_ids: set[str] = set()
    for hop in spine:
        if hop.tip_id != "bombs":
            raise RuntimeError(f"Bombs spine hop {hop.hop_id!r} tip_id={hop.tip_id!r}")
        if hop.hop_id in hop_ids:
            raise RuntimeError(f"Duplicate bombs hop_id {hop.hop_id!r}")
        hop_ids.add(hop.hop_id)
    required = {
        "two_missile_detour",
        "construction_return",
        "elevator_return",
        "pit_natural_entry",
        "pit_to_post_torizo",
    }
    missing = required - hop_ids
    if missing:
        raise RuntimeError(f"Bombs spine missing hops: {sorted(missing)}")


def validate_spore_spine(spine: Sequence[SpineHop] = SPORE_SPINE) -> None:
    hop_ids = [hop.hop_id for hop in spine]
    if hop_ids != ["parlor_to_main_shaft", "main_shaft_to_spore_exit"]:
        raise RuntimeError(f"Spore spine hop order unexpected: {hop_ids}")
    for hop in spine:
        if hop.tip_id != "spore":
            raise RuntimeError(f"Spore spine hop {hop.hop_id!r} tip_id={hop.tip_id!r}")


def validate_supers_spine(spine: Sequence[SpineHop] = SUPERS_SPINE) -> None:
    hop_ids = [hop.hop_id for hop in spine]
    if hop_ids != ["spore_supers_collected"]:
        raise RuntimeError(f"Supers spine hop order unexpected: {hop_ids}")
    for hop in spine:
        if hop.tip_id != "supers":
            raise RuntimeError(f"Supers spine hop {hop.hop_id!r} tip_id={hop.tip_id!r}")


validate_bombs_spine()
validate_spore_spine()
validate_supers_spine()
