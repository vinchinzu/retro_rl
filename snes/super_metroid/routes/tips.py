"""Unified continuous tip model — one TipSpec for morph → bat_cave.

There is **no** EarlyTipSpec / PostSupersTipSpec split. Every continuous tip is
a :class:`TipSpec` row:

* **Early tips** (morph→supers) register real ``hops`` + ``parent_tip_id`` and
  play through :func:`play_tip` / :func:`play_hops`. Finish shapes differ via
  ``assist_mode`` + ``final_conditions_fn`` plugins.
* **Super+ tips** parent through ``supers`` and use :func:`play_tip` /
  :func:`run_tip` with :class:`~super_metroid.routes.kpdr.spine.SpineHop` deltas
  and spine entry/ordinary condition keys.

Canonical hop runner is :func:`play_hops` on ``SpineHop`` (no RouteHop layer).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from super_metroid.assist import UnlimitedAmmoAssist, UnlimitedResourcesAssist
from super_metroid.hop_glance import (
    LeaveMiss,
    LeaveSpec,
    final_from_state,
    grade_final,
    parse_room,
)
from super_metroid.policy import SegmentEvidence
from super_metroid.progression import RoomProgressionGraph
from super_metroid.ram import HI_JUMP_MASK, VARIA_MASK, GameplayPhase, SuperMetroidState
from super_metroid.room_timer import RoomTimer
from super_metroid.routes.kpdr.spine import SpineHop
from super_metroid.routes.runtime import (
    ContinuousRunReport,
    ContinuousRunResult,
    PlayContext,
    RouteSession,
    Split,
    finish_report,
    hashed_policy_source,
    persist_room_timing,
    resolve_clean_resources,
    route_plan_evidence,
    run_continuous,
    split_for_transition,
)
from super_metroid.video import VideoCaptureConfig

from retro_harness.env import write_state_bytes

# Filled by early_continuous + hops at import of continuous (see register_*).
TIP_SPECS: tuple["TipSpec", ...] = ()
TIP_BY_ID: dict[str, "TipSpec"] = {}

AssistMode = Literal["ammo", "resources"]

__all__ = [
    "AssistMode",
    "FinishCtx",
    "TipPlayResult",
    "TipSpec",
    "TIP_SPECS",
    "TIP_BY_ID",
    "get_tip",
    "register_tips",
    "LeaveMiss",
    "play_hops",
    "play_tip",
    "run_tip",
    "run_to_tip",
]


@dataclass(frozen=True)
class TipPlayResult:
    """Typed evidence from :func:`play_tip` (parent chain + hop delta)."""

    last: Any = None  # last non-None hop play return
    boss: Any = None  # SporeSpawnEvidence or similar
    super_collect: Any = None  # SuperCollectEvidence


@dataclass(frozen=True)
class FinishCtx:
    """Inputs for TipSpec final-condition / source-policy plugins."""

    final: SuperMetroidState
    result: ContinuousRunResult
    boss: object | None
    super_collect: object | None
    clean: bool


@dataclass(frozen=True)
class TipSpec:
    """One continuous power-on tip (CLI ``--to`` / ``run_to`` target).

    Spine-driven tips set ``parent_tip_id`` + ``hops`` (SpineHop delta). Play
    always goes through :func:`play_tip`. Run goes through :func:`run_tip` for
    every tip: early finish shapes use ``assist_mode`` + ``final_conditions_fn``;
    Super+ uses entry/ordinary keys.

    CLI identity (display, aliases, capability flags) lives here; catalog
    :class:`~super_metroid.routes.catalog.ContinuousTip` and
    :class:`~retro_harness.adventure.routes.NamedRoute` are derived views.
    """

    tip_id: str
    parent_tip_id: str | None
    hops: tuple[SpineHop, ...]
    graph: RoomProgressionGraph
    kind: str
    required_splits: tuple[str, ...]
    success_outcome: str
    route_label: str
    source_policy: str = ""
    timing_source: str = ""
    entry_condition_key: str = ""
    ordinary_condition_key: str = ""
    final_room: int | None = None
    require_hi_jump: bool = False
    require_varia: bool = False
    # --- CLI identity / capability (source of truth for ContinuousTip) ---
    display_name: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    supports_room_timing: bool = False
    supports_unlimited_energy: bool = False
    supports_checkpoint: bool = False
    # Assist: ammo-only (morph/bombs) vs full energy+ammo (spore+).
    assist_mode: AssistMode = "resources"
    # finish_report knobs (early morph/bombs differ from Super+).
    schema_version: int = 1
    require_deaths_zero: bool = True
    require_transitions: bool = True
    include_route_plan: bool = True
    include_policy_sources: bool = True
    # Morph historical JSON emits flat ``video_path`` instead of only ``video``.
    emit_flat_video_path: bool = False
    # When set, replaces spine entry/ordinary condition path.
    final_conditions_fn: Callable[[FinishCtx], dict[str, bool]] | None = None
    # Dynamic source_policy (bombs clean vs assisted); else ``source_policy``.
    source_policy_fn: Callable[[bool], str] | None = None
    # Early spore/supers policy hashes; None + include → Super+ kpdr sources.
    policy_sources_fn: Callable[[], dict[str, object]] | None = None

    @property
    def artifact_stem(self) -> str:
        """Recording basename under ``recordings/`` (always ``tip_id``)."""
        return self.tip_id


def register_tips(specs: Sequence[TipSpec], *, replace: bool = False) -> None:
    """Merge tip rows into the module-level table (order preserved).

    After each merge, rebuilds catalog ContinuousTip / NamedRoute views so CLI
    identity stays derived from TipSpec (no parallel meta tables).
    """
    global TIP_SPECS, TIP_BY_ID
    if replace:
        TIP_SPECS = tuple(specs)
    else:
        existing = {s.tip_id: s for s in TIP_SPECS}
        for spec in specs:
            existing[spec.tip_id] = spec
        # Stable order: previous order, then new tip_ids (latest value wins).
        ordered: list[TipSpec] = []
        seen: set[str] = set()
        for s in list(TIP_SPECS) + list(specs):
            if s.tip_id in seen:
                continue
            seen.add(s.tip_id)
            ordered.append(existing[s.tip_id])
        TIP_SPECS = tuple(ordered)
    TIP_BY_ID = {s.tip_id: s for s in TIP_SPECS}
    from super_metroid.routes.catalog import rebuild_from_tip_specs

    rebuild_from_tip_specs()


def get_tip(tip_id: str) -> TipSpec:
    try:
        return TIP_BY_ID[tip_id]
    except KeyError as exc:
        known = ", ".join(TIP_BY_ID) or "(none registered)"
        raise KeyError(f"Unknown continuous tip {tip_id!r}. Known: {known}") from exc


def _invoke_after(
    hop: SpineHop,
    session: RouteSession,
    splits: list[Split],
    result: Any,
) -> None:
    """Call hop.after with a single signature: (session, splits, result)."""
    after = hop.after
    if after is None:
        return
    after(session, splits, result)


def _raise_if_leave_misses(hop: SpineHop, leftover: dict[str, Any]) -> None:
    """Glance dest when ``hop.leave`` is set; dest-room check otherwise.

    Always raise :class:`LeaveMiss` with leftover populated. Never drop the still.
    """
    spec = hop.leave
    if isinstance(spec, LeaveSpec):
        misses = grade_final(leftover, spec)
    else:
        got = parse_room(leftover.get("room", leftover.get("room_id", 0)))
        if got == hop.to_room:
            return
        misses = [f"room 0x{got:04X} != 0x{hop.to_room:04X}"]
    if misses:
        raise LeaveMiss(
            hop.hop_id,
            leftover,
            misses,
            room_label=hop.room_label,
            to_room=hop.to_room,
        )


def play_hops(
    session: RouteSession,
    splits: list[Split],
    hops: Sequence[SpineHop],
    segments: list[SegmentEvidence] | None = None,
) -> Any:
    """Run ordered SpineHop legs: play → after → split → glance leave.

    Optional ``segments`` collects :class:`SegmentEvidence` return values from
    policy hops (early bombs path). ``after`` takes
    ``(session, splits, result)`` for multi-split bookkeeping. Returns the last
    non-``None`` play result (boss / collect evidence for early tips).

    Failed hops raise :class:`LeaveMiss` with ``.leftover`` (the still).
    """
    last: Any = None
    for hop in hops:
        result = hop.play(session)
        if result is not None:
            last = result
        if segments is not None and isinstance(result, SegmentEvidence):
            segments.append(result)
        _invoke_after(hop, session, splits, result)
        # Skip auto hop_id split when after already recorded that split_id
        # (e.g. supers collect uses collect_frame, not session.frame).
        if hop.hop_id not in {s.split_id for s in splits}:
            if hop.use_transition_split:
                splits.append(
                    split_for_transition(
                        session.transitions,
                        hop.hop_id,
                        hop.from_room,
                        hop.to_room,
                    )
                )
            else:
                splits.append(
                    Split(hop.hop_id, session.frame, session.state.room_id)
                )
        leftover = final_from_state(session.state)
        _raise_if_leave_misses(hop, leftover)
    return last


def _merge_tip_play_results(
    parent_result: TipPlayResult | None,
    hop_result: Any,
) -> TipPlayResult:
    """Combine parent + hop evidence into typed :class:`TipPlayResult` fields."""
    base = parent_result if parent_result is not None else TipPlayResult()
    if hop_result is None:
        return base

    # Local imports: avoid tips ↔ early controller cycles at module load.
    from super_metroid.routes.kpdr.spore_spawn import SporeSpawnEvidence
    from super_metroid.routes.kpdr.super_collect import SuperCollectEvidence

    boss = base.boss
    super_collect = base.super_collect
    if isinstance(hop_result, SporeSpawnEvidence):
        boss = hop_result
    elif isinstance(hop_result, SuperCollectEvidence):
        super_collect = hop_result
    elif isinstance(hop_result, TipPlayResult):
        return TipPlayResult(
            last=hop_result.last if hop_result.last is not None else base.last,
            boss=hop_result.boss if hop_result.boss is not None else base.boss,
            super_collect=(
                hop_result.super_collect
                if hop_result.super_collect is not None
                else base.super_collect
            ),
        )
    return TipPlayResult(last=hop_result, boss=boss, super_collect=super_collect)


def play_tip(
    tip_id: str,
    session: RouteSession,
    splits: list[Split],
    segments: list[SegmentEvidence],
) -> TipPlayResult:
    """Play a tip by id: parent chain + SpineHop delta."""
    spec = get_tip(tip_id)

    parent_result: TipPlayResult | None = None
    if spec.parent_tip_id is not None:
        parent_result = play_tip(spec.parent_tip_id, session, splits, segments)
    hop_result: Any = None
    if spec.hops:
        hop_result = play_hops(session, splits, spec.hops, segments=segments)
    return _merge_tip_play_results(parent_result, hop_result)


def _extra_final_conditions_for_spec(
    spec: TipSpec,
) -> Callable[[SuperMetroidState], dict[str, bool]] | None:
    if not spec.require_hi_jump and not spec.require_varia:
        return None

    def extra(final: SuperMetroidState) -> dict[str, bool]:
        out: dict[str, bool] = {}
        if spec.require_hi_jump:
            out["hi_jump_collected"] = bool(final.collected_items & HI_JUMP_MASK)
        if spec.require_varia:
            out["varia_collected"] = bool(final.collected_items & VARIA_MASK)
        return out

    return extra


def _spine_final_conditions(
    final: SuperMetroidState,
    boss: object,
    *,
    room_id: int,
    entry_key: str,
    ordinary_key: str,
    early_prefix: Callable[[SuperMetroidState], dict[str, bool]],
    extra: dict[str, bool] | None = None,
) -> dict[str, bool]:
    from super_metroid.routes.kpdr.spore_spawn import SporeSpawnEvidence

    conditions = early_prefix(final)
    conditions.update(
        {
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
    )
    if extra:
        conditions.update(extra)
    return conditions


def _assist_for_spec(
    spec: TipSpec,
    *,
    unlimited_energy: bool,
    unlimited_ammo: bool,
) -> UnlimitedAmmoAssist | UnlimitedResourcesAssist:
    if spec.assist_mode == "ammo":
        # Morph/bombs: energy flag accepted for CLI uniformity but is a no-op.
        return UnlimitedAmmoAssist(enabled=unlimited_ammo)
    if spec.assist_mode == "resources":
        return UnlimitedResourcesAssist(
            unlimited_energy=unlimited_energy,
            unlimited_ammo=unlimited_ammo,
        )
    raise ValueError(f"Unknown assist_mode {spec.assist_mode!r} on tip {spec.tip_id!r}")


_THIS = Path(__file__)
_KPDR_POLICY_SOURCES: dict[str, object] | None = None


def _kpdr_policy_sources() -> dict[str, object]:
    global _KPDR_POLICY_SOURCES
    if _KPDR_POLICY_SOURCES is None:
        kpdr = _THIS.parent / "kpdr"
        _KPDR_POLICY_SOURCES = {
            "continuous_route_module": hashed_policy_source(
                _THIS.with_name("continuous.py")
            ),
            "post_torizo_controller": hashed_policy_source(kpdr / "spore_spawn.py"),
            "kpdr_super_room": hashed_policy_source(kpdr / "super_collect.py"),
            "kpdr_package": {
                "path": str(kpdr.resolve()),
                "note": "K1/K2 segment controllers under routes/kpdr/",
            },
        }
    return dict(_KPDR_POLICY_SOURCES)


def _policy_sources_for_spec(spec: TipSpec) -> dict[str, object] | None:
    if not spec.include_policy_sources:
        return None
    if spec.policy_sources_fn is not None:
        return spec.policy_sources_fn()
    return _kpdr_policy_sources()


def _source_policy_for_spec(spec: TipSpec, *, clean: bool) -> str:
    if spec.source_policy_fn is not None:
        return spec.source_policy_fn(clean)
    return spec.source_policy


def _final_conditions_for_spec(
    spec: TipSpec,
    ctx: FinishCtx,
) -> dict[str, bool]:
    if spec.final_conditions_fn is not None:
        return spec.final_conditions_fn(ctx)

    from super_metroid.routes.early_continuous import early_prefix_conditions

    if spec.final_room is None:
        raise RuntimeError(f"Tip {spec.tip_id!r} missing final_room for spine conditions")
    if not spec.entry_condition_key or not spec.ordinary_condition_key:
        raise RuntimeError(
            f"Tip {spec.tip_id!r} missing entry/ordinary condition keys "
            f"(or set final_conditions_fn)"
        )
    extra_fn = _extra_final_conditions_for_spec(spec)
    extra = extra_fn(ctx.final) if extra_fn else None
    return _spine_final_conditions(
        ctx.final,
        ctx.boss,
        room_id=spec.final_room,
        entry_key=spec.entry_condition_key,
        ordinary_key=spec.ordinary_condition_key,
        early_prefix=early_prefix_conditions,
        extra=extra,
    )


def run_tip(
    tip_id: str,
    *,
    env_factory: Callable[[], Any] | None = None,
    rom_path: str | Path | None = None,
    video_path: str | Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = None,
    unlimited_energy: bool = True,
    unlimited_ammo: bool = True,
    room_timing_path: str | Path | None = None,
    state_output: str | Path | None = None,
    require_clean_resources: bool | None = None,
) -> ContinuousRunReport:
    """Power-on once through a tip (assist + condition plugins on TipSpec)."""
    from super_metroid.routes.kpdr import SuperCollectEvidence
    from super_metroid.routes.kpdr.spore_spawn import SporeSpawnEvidence

    spec = get_tip(tip_id)

    clean = resolve_clean_resources(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        require_clean_resources=require_clean_resources,
    )
    assist = _assist_for_spec(
        spec,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    play_result = TipPlayResult()
    plan = route_plan_evidence() if spec.include_route_plan else None
    timer = RoomTimer() if room_timing_path is not None else None

    def play(ctx: PlayContext) -> None:
        nonlocal play_result
        play_result = play_tip(tip_id, ctx.session, ctx.splits, ctx.segments)

    result = run_continuous(
        play=play,
        assist=assist,
        graph=spec.graph,
        env_factory=env_factory,
        video_path=video_path,
        video_config=video_config,
        success_outcome=spec.success_outcome,
        room_timer=timer,
        capture_checkpoint=state_output is not None,
    )
    boss = play_result.boss
    super_collect = play_result.super_collect
    finish_ctx = FinishCtx(
        final=result.final_state,
        result=result,
        boss=boss,
        super_collect=super_collect,
        clean=clean,
    )
    report: ContinuousRunReport | None = None
    try:
        report = finish_report(
            result,
            schema_version=spec.schema_version,
            kind=spec.kind,
            required_splits=spec.required_splits,
            final_conditions=_final_conditions_for_spec(spec, finish_ctx),
            source_policy=_source_policy_for_spec(spec, clean=clean),
            **({"rom_path": rom_path} if rom_path is not None else {}),
            report_path=report_path,
            route_label=spec.route_label,
            require_deaths_zero=spec.require_deaths_zero,
            require_transitions=spec.require_transitions,
            require_clean_resources=clean,
            route_plan=plan,
            policy_sources=_policy_sources_for_spec(spec),
            boss=boss if isinstance(boss, SporeSpawnEvidence) else None,
            super_collect=(
                super_collect
                if isinstance(super_collect, SuperCollectEvidence)
                else None
            ),
        )
        if spec.emit_flat_video_path:
            # Historical morph JSON emits ``video_path`` (not only ``video``).
            if video_path is not None:
                report.video_path = str(Path(video_path).resolve())
            elif result.video_evidence is not None:
                path = result.video_evidence.get("path")
                if path is not None:
                    report.video_path = str(path)
        if state_output is not None:
            if result.checkpoint_state is None:
                raise RuntimeError(
                    f"{spec.route_label} accepted without a checkpoint snapshot"
                )
            write_state_bytes(state_output, result.checkpoint_state)
        return report
    finally:
        persist_room_timing(
            timer=timer,
            room_timing_path=room_timing_path,
            source=spec.timing_source or spec.tip_id,
            report=report,
            result=result,
            report_path=report_path,
            video_path=video_path,
        )


def run_to_tip(
    tip_id: str,
    **kwargs: Any,
) -> ContinuousRunReport:
    """Power-on through a tip via :func:`run_tip`."""
    return run_tip(tip_id, **kwargs)
