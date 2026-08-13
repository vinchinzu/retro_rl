"""Early continuous tips: power-on Morph → Bombs → Spore → Supers.

**Morph / Bombs / Spore / Supers** register real SpineHop deltas
(:data:`~super_metroid.routes.kpdr.early_spine.MORPH_SPINE`,
:data:`~super_metroid.routes.kpdr.early_post_morph.BOMBS_SPINE`, …) with a
``parent_tip_id`` chain. Play and run both go through
:func:`~super_metroid.routes.tips.play_tip` / :func:`~super_metroid.routes.tips.run_tip`.

Finish shapes differ via TipSpec plugins (``assist_mode``, ``final_conditions_fn``,
``source_policy_fn``). All tips use the generic :func:`~super_metroid.routes.tips.run_tip`
path.

One tip table: :mod:`super_metroid.routes.tips` (no EarlyTipSpec type).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from super_metroid.policy import SegmentEvidence
from super_metroid.progression import (
    EARLY_GAME_GRAPH,
    MORPH_GRAPH,
    SPORE_GRAPH,
)
from super_metroid.ram import (
    BOMBS_MASK,
    MORPH_BALL_MASK,
    GameplayPhase,
    SuperMetroidState,
)
from super_metroid.routes.kpdr import SuperCollectEvidence
from super_metroid.routes.kpdr.early_post_morph import (
    BOMBS_SPINE,
    SPORE_SPINE,
    SUPERS_SPINE,
)
from super_metroid.routes.kpdr.early_spine import MORPH_SPINE
from super_metroid.routes.catalog import (
    BOMBS_PREFIX_SPLITS,
    SPORE_EXIT_SPLITS,
    SUPERS_SPLITS,
)
from super_metroid.routes.runtime import (
    ROUTE_PLAN_PATH,
    ContinuousRunReport,
    RouteSession,
    Split,
    hashed_policy_source,
)
from super_metroid.routes.tips import FinishCtx, TipSpec, play_tip, register_tips, run_tip
from super_metroid.video import VideoCaptureConfig
from super_metroid.routes.kpdr.spore_spawn import SporeSpawnEvidence

_THIS = Path(__file__)
KPDR_PACKAGE_PATH = _THIS.parent / "kpdr"
# Real controller body under kpdr/spore_spawn.py.
SPORE_CONTROLLER_PATH = KPDR_PACKAGE_PATH / "spore_spawn.py"
KPDR_SUPER_ROOM_PATH = KPDR_PACKAGE_PATH / "super_collect.py"
# Historical CONTROLLER_PATH name → Super-collect implementation (shim deleted).
CONTROLLER_PATH = KPDR_SUPER_ROOM_PATH
CONTINUOUS_MODULE_PATH = _THIS.with_name("continuous.py")

__all__ = [
    "CONTROLLER_PATH",
    "SPORE_CONTROLLER_PATH",
    "KPDR_SUPER_ROOM_PATH",
    "TipSpec",
    "EARLY_TIP_SPECS",
    "EARLY_TIP_BY_ID",
    "play_morph",
    "run_morph",
    "play_bombs",
    "run_bombs",
    "play_spore",
    "run_spore",
    "play_supers",
    "run_supers",
    "early_prefix_conditions",
    "spore_boss_conditions",
]


# ---------------------------------------------------------------------------
# Shared final-condition fragments (spore / supers / Super+)
# ---------------------------------------------------------------------------


def early_prefix_conditions(final: SuperMetroidState) -> dict[str, bool]:
    """Inventory gates shared from Spore tip through Super+ tips."""
    return {
        "both_missile_expansions": final.max_missiles >= 10,
        "morph_and_bombs": (
            final.collected_items & (MORPH_BALL_MASK | BOMBS_MASK)
            == MORPH_BALL_MASK | BOMBS_MASK
        ),
        "terminator_energy_tank": final.max_health >= 199,
    }


def spore_boss_conditions(
    boss: SporeSpawnEvidence | None,
    *,
    transitions: list,
    require_vulnerable: bool = False,
) -> dict[str, bool]:
    """Spore fight / exit gates used by spore + supers reports."""
    out: dict[str, bool] = {
        "spore_spawn_hp_reached_zero": boss is not None and 0 in boss.observed_hp,
        "natural_spore_room_exit": any(
            t.source_room_id == 0x9DC7 and t.target_room_id == 0x9B5B
            for t in transitions
        ),
    }
    if boss is not None:
        out["spore_spawn_activated_at_960_hp"] = boss.peak_hp >= 960
    else:
        out["spore_spawn_activated_at_960_hp"] = False
    if require_vulnerable:
        out["vulnerable_mouth_states_observed"] = bool(
            boss is not None and boss.vulnerable_spritemaps
        )
    return out


def _early_policy_sources(*keys: str) -> dict[str, object]:
    """Hashed policy sources for spore/supers continuous reports."""
    sources: dict[str, object] = {
        "continuous_route_module": hashed_policy_source(CONTINUOUS_MODULE_PATH),
        "post_torizo_controller": hashed_policy_source(SPORE_CONTROLLER_PATH),
    }
    if "post_spore" in keys:
        # Super collect lives under routes/kpdr (shim post_spore_controller removed).
        sources["post_spore_controller"] = hashed_policy_source(KPDR_SUPER_ROOM_PATH)
    if "super_room" in keys:
        sources["kpdr_super_room"] = hashed_policy_source(KPDR_SUPER_ROOM_PATH)
    if "route_plan" in keys:
        sources["route_plan"] = hashed_policy_source(ROUTE_PLAN_PATH)
    return sources


def _spore_policy_sources() -> dict[str, object]:
    return _early_policy_sources()


def _supers_policy_sources() -> dict[str, object]:
    return _early_policy_sources("post_spore", "super_room", "route_plan")


# ---------------------------------------------------------------------------
# Final-condition plugins (wired on EARLY_TIP_SPECS)
# ---------------------------------------------------------------------------


def morph_final_conditions(ctx: FinishCtx) -> dict[str, bool]:
    return {
        "morph_collected": bool(ctx.final.collected_items & MORPH_BALL_MASK),
    }


def bombs_final_conditions(ctx: FinishCtx) -> dict[str, bool]:
    final = ctx.final
    session = ctx.result.session
    return {
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
    }


def bombs_source_policy(clean: bool) -> str:
    if clean:
        return (
            "accepted power-on prefix + hash-pinned natural manual replay segments "
            "+ Clean (no ammo refill)"
        )
    return (
        "accepted power-on prefix + hash-pinned natural manual replay segments "
        "+ phase-guarded unlimited ammo"
    )


def spore_final_conditions(ctx: FinishCtx) -> dict[str, bool]:
    boss = ctx.boss if isinstance(ctx.boss, SporeSpawnEvidence) else None
    conditions = early_prefix_conditions(ctx.final)
    conditions.update(
        spore_boss_conditions(
            boss,
            transitions=ctx.result.session.transitions,
            require_vulnerable=True,
        )
    )
    conditions["post_spore_room_settle"] = (
        ctx.final.room_id == 0x9B5B
        and ctx.final.phase is GameplayPhase.ORDINARY_GAMEPLAY
    )
    return conditions


def supers_final_conditions(ctx: FinishCtx) -> dict[str, bool]:
    boss = ctx.boss if isinstance(ctx.boss, SporeSpawnEvidence) else None
    super_collect = (
        ctx.super_collect
        if isinstance(ctx.super_collect, SuperCollectEvidence)
        else None
    )
    conditions = early_prefix_conditions(ctx.final)
    conditions.update(
        spore_boss_conditions(
            boss,
            transitions=ctx.result.session.transitions,
        )
    )
    conditions.update(
        {
            "super_missiles_collected": ctx.final.max_super_missiles >= 5,
            "super_collect_in_super_room": (
                super_collect is not None
                and super_collect.max_super_missiles >= 5
                and super_collect.final_room_id == 0x9B5B
            ),
            "post_super_ordinary": (
                ctx.final.room_id == 0x9B5B
                and ctx.final.phase is GameplayPhase.ORDINARY_GAMEPLAY
            ),
        }
    )
    return conditions


# ===========================================================================
# Public play / run wrappers (thin → play_tip / run_tip)
# ===========================================================================


def play_morph(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence] | None = None,
) -> Any:
    """Power-on through natural Morph Ball collect (``MORPH_SPINE`` via play_tip)."""
    segs = segments if segments is not None else []
    return play_tip("morph", session, splits, segs)


def run_morph(
    *,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once; stop after Morph Ball.

    Morph never used energy assist historically; ``unlimited_energy`` is accepted
    for uniform CLI/clean wiring and is a no-op (ammo-only assist).
    """
    return run_tip(
        "morph",
        video_path=video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        require_clean_resources=require_clean_resources,
    )


def play_bombs(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> Any:
    """Morph prefix + two Missiles + Bomb Torizo + Parlor settle (``BOMBS_SPINE``)."""
    return play_tip("bombs", session, splits, segments)


def run_bombs(
    *,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once; stop after Bomb Torizo exit into Parlor.

    Bombs historically used ammo-only assist; ``unlimited_energy`` is accepted
    for uniform CLI/clean wiring (no-op — energy refill starts at spore+).
    """
    return run_tip(
        "bombs",
        video_path=video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        require_clean_resources=require_clean_resources,
    )


def play_spore(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> SporeSpawnEvidence:
    """Bombs prefix + post-Torizo controller through natural Spore exit (``SPORE_SPINE``)."""
    result = play_tip("spore", session, splits, segments)
    if not isinstance(result.boss, SporeSpawnEvidence):
        raise RuntimeError(
            f"play_spore expected SporeSpawnEvidence boss, got {type(result.boss)!r}"
        )
    return result.boss


def run_spore(
    *,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once; stop in Super room after Spore Spawn exit."""
    return run_tip(
        "spore",
        video_path=video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        require_clean_resources=require_clean_resources,
    )


def play_supers(
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> tuple[SporeSpawnEvidence, SuperCollectEvidence]:
    """Spore-exit prefix, then natural Super collect (``SUPERS_SPINE``)."""
    result = play_tip("supers", session, splits, segments)
    if not isinstance(result.boss, SporeSpawnEvidence) or not isinstance(
        result.super_collect, SuperCollectEvidence
    ):
        raise RuntimeError(
            "play_supers expected TipPlayResult with SporeSpawnEvidence boss and "
            f"SuperCollectEvidence super_collect, got boss={type(result.boss)!r} "
            f"super_collect={type(result.super_collect)!r}"
        )
    return result.boss, result.super_collect


def run_supers(
    *,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once through natural Super Missile collect (STATUS baseline).

    ``room_timing_path`` is opt-in: when set, the shared :class:`RoomTimer`
    observes every frame and a separate timing JSON is written. Timing never
    affects assist, integrity, or route decisions.
    """
    return run_tip(
        "supers",
        video_path=video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
        require_clean_resources=require_clean_resources,
    )


# ===========================================================================
# Early rows on the unified TipSpec table
# ===========================================================================
# Play: parent_tip_id chain + SpineHop deltas via play_tip.
# Run: assist_mode + final_conditions_fn (+ source/policy plugins) via run_tip.


EARLY_TIP_SPECS: tuple[TipSpec, ...] = (
    TipSpec(
        tip_id="morph",
        parent_tip_id=None,
        hops=MORPH_SPINE,
        graph=MORPH_GRAPH,
        kind="morph",
        required_splits=("morph_ball",),
        success_outcome="morph_ball_acquired",
        route_label="morph",
        source_policy=(
            "power-on Ceres policy + imported natural-entry room seeds"
        ),
        display_name="Power-on → Morph Ball",
        description="Ceres → Zebes Morph collect.",
        assist_mode="ammo",
        require_deaths_zero=False,
        require_transitions=False,
        include_route_plan=False,
        include_policy_sources=False,
        emit_flat_video_path=True,
        final_conditions_fn=morph_final_conditions,
    ),
    TipSpec(
        tip_id="bombs",
        parent_tip_id="morph",
        hops=BOMBS_SPINE,
        graph=EARLY_GAME_GRAPH,
        kind="bombs",
        required_splits=BOMBS_PREFIX_SPLITS,
        success_outcome="bomb_torizo_defeated_bombs_acquired",
        route_label="bombs",
        display_name="Power-on → Bomb Torizo exit",
        description="Morph prefix through natural Bomb Torizo clear.",
        aliases=("bomb_torizo", "torizo"),
        assist_mode="ammo",
        schema_version=2,
        require_deaths_zero=False,
        require_transitions=False,
        include_route_plan=False,
        include_policy_sources=False,
        final_conditions_fn=bombs_final_conditions,
        source_policy_fn=bombs_source_policy,
    ),
    TipSpec(
        tip_id="spore",
        parent_tip_id="bombs",
        hops=SPORE_SPINE,
        graph=SPORE_GRAPH,
        kind="spore",
        required_splits=SPORE_EXIT_SPLITS,
        success_outcome="spore_spawn_defeated_and_exited",
        route_label="spore",
        source_policy=(
            "accepted power-on prefix + checked read-only post-Torizo controller "
            "+ editor-precalculated room plan + phase-guarded current resources"
        ),
        display_name="Power-on → Spore Spawn exit",
        description="Bombs prefix through natural Spore exit into Super room.",
        aliases=("spore_spawn",),
        supports_unlimited_energy=True,
        final_conditions_fn=spore_final_conditions,
        policy_sources_fn=_spore_policy_sources,
    ),
    TipSpec(
        tip_id="supers",
        parent_tip_id="spore",
        hops=SUPERS_SPINE,
        graph=SPORE_GRAPH,
        kind="supers",
        required_splits=SUPERS_SPLITS,
        success_outcome="spore_supers_collected",
        route_label="supers",
        source_policy=(
            "accepted power-on prefix + Spore controller + post-Spore Super "
            "controller + phase-guarded current resources"
        ),
        timing_source="supers",
        display_name="Power-on → Spore Super Missiles",
        description="Spore prefix through natural Super Missile collect.",
        aliases=("super",),
        supports_room_timing=True,
        supports_unlimited_energy=True,
        final_conditions_fn=supers_final_conditions,
        policy_sources_fn=_supers_policy_sources,
    ),
)

EARLY_TIP_BY_ID: dict[str, TipSpec] = {
    spec.tip_id: spec for spec in EARLY_TIP_SPECS
}

register_tips(EARLY_TIP_SPECS)
