"""Boss strategy / segment contracts for continuous composition.

Unifies the Torizo and Kraid patterns so every future boss is a
``BossSegment`` that continuous runners and the progression graph can
consume like any other hop. See ``docs/BOSS_PIPELINE.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from super_metroid.combat.features import BossCatalogEntry, get_boss_catalog
from super_metroid.policy import StateRequirement
from super_metroid.progression import ProgressCondition
from super_metroid.routes.runtime import ControllerSession


@dataclass(frozen=True)
class BossEvidence:
    """Uniform evidence for one boss Segment play.

    Compatible in spirit with ``SegmentEvidence`` / hop reports: start/end
    frames, outcome string, success flag, and free-form detail dict for
    boss-specific metrics (body zero frame, phases, item collect, …).
    """

    boss_id: str
    start_frame: int
    end_frame: int
    outcome: str
    success: bool
    final_room_id: int
    boss_defeated: bool
    detail: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "bossId": self.boss_id,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "actionFrames": self.end_frame - self.start_frame,
            "outcome": self.outcome,
            "success": self.success,
            "finalRoomId": self.final_room_id,
            "finalRoomIdHex": f"0x{self.final_room_id:04X}",
            "bossDefeated": self.boss_defeated,
            "detail": self.detail,
        }

    @classmethod
    def from_parts(
        cls,
        *,
        boss_id: str,
        start_frame: int,
        end_frame: int,
        outcome: str,
        success: bool,
        final_room_id: int,
        boss_defeated: bool,
        **detail: object,
    ) -> BossEvidence:
        return cls(
            boss_id=boss_id,
            start_frame=start_frame,
            end_frame=end_frame,
            outcome=outcome,
            success=success,
            final_room_id=final_room_id,
            boss_defeated=boss_defeated,
            detail=dict(detail),
        )


@runtime_checkable
class BossStrategy(Protocol):
    """Deterministic full-knowledge boss controller.

    Implementations live in ``combat/<boss>.py``. Continuous claims require
    natural entry — no door-warp, no forged boss/item RAM.
    """

    @property
    def boss_id(self) -> str: ...

    @property
    def catalog(self) -> BossCatalogEntry: ...

    @property
    def entry(self) -> StateRequirement: ...

    def play(self, session: ControllerSession) -> BossEvidence: ...


@dataclass(frozen=True)
class BossSegment:
    """Adapter: :class:`BossStrategy` → continuous Segment surface.

    Mirrors ``ControllerSegment`` / ``PolicySegmentAdapter`` in
    ``routes/segment.py`` so tip runners can register boss closeouts the
    same way as room hops.
    """

    strategy: BossStrategy
    segment_id: str | None = None
    exit_condition: ProgressCondition | None = None
    label: str = ""

    @property
    def id(self) -> str:
        return self.segment_id or f"boss_{self.strategy.boss_id}"

    @property
    def entry(self) -> StateRequirement:
        return self.strategy.entry

    @property
    def catalog(self) -> BossCatalogEntry:
        return self.strategy.catalog

    def play(self, session: ControllerSession) -> BossEvidence:
        entry = self.strategy.entry
        failures = entry.failures(session.state)
        if failures:
            raise RuntimeError(
                f"{self.id}: entry check failed: {'; '.join(failures)}"
            )
        evidence = self.strategy.play(session)
        if self.exit_condition is not None and not self.exit_condition.matches(
            session.state
        ):
            raise RuntimeError(
                f"{self.id}: exit ProgressCondition not met "
                f"(outcome={evidence.outcome})"
            )
        return evidence


@dataclass(frozen=True)
class CallableBossStrategy:
    """Lightweight :class:`BossStrategy` from a play function + catalog id.

    Useful when an existing ``play_*`` controller (e.g. Kraid fight+varia)
    should register as a BossSegment without a full class rewrite.
    """

    boss_id: str
    play_fn: Callable[[ControllerSession], BossEvidence]
    entry_requirement: StateRequirement
    catalog_entry: BossCatalogEntry | None = None
    success_outcomes: frozenset[str] = field(
        default_factory=lambda: frozenset({"defeated", "collected", "success"})
    )

    @property
    def catalog(self) -> BossCatalogEntry:
        if self.catalog_entry is not None:
            return self.catalog_entry
        return get_boss_catalog(self.boss_id)

    @property
    def entry(self) -> StateRequirement:
        return self.entry_requirement

    def play(self, session: ControllerSession) -> BossEvidence:
        return self.play_fn(session)


def _wrap_simple_fight(
    *,
    boss_id: str,
    room_id: int,
    catalog_fn: Callable[[], BossCatalogEntry],
    play_fn: Callable[[ControllerSession], Any],
    success: Callable[[Any], bool],
    boss_defeated: Callable[[Any], bool] | None = None,
    outcome: Callable[[Any], str] | None = None,
) -> CallableBossStrategy:
    """Shared adapter: fight evidence object → :class:`BossEvidence`."""

    def play(session: ControllerSession) -> BossEvidence:
        result = play_fn(session)
        ok = success(result)
        return BossEvidence(
            boss_id=boss_id,
            start_frame=int(result.start_frame),
            end_frame=int(result.end_frame),
            outcome=str(outcome(result) if outcome else result.outcome),
            success=ok,
            final_room_id=session.state.room_id,
            boss_defeated=bool(boss_defeated(result) if boss_defeated else ok),
            detail=result.to_dict(),
        )

    return CallableBossStrategy(
        boss_id=boss_id,
        play_fn=play,
        entry_requirement=StateRequirement(room_id=room_id),
        catalog_entry=catalog_fn(),
    )


def wrap_kraid_as_boss_strategy() -> CallableBossStrategy:
    """Kraid fight→Varia as a BossStrategy (living template)."""
    from super_metroid.combat.features import kraid_catalog
    from super_metroid.combat.kraid import ROOM_KRAID, play_kraid_fight_to_varia

    catalog = kraid_catalog()

    def play(session: ControllerSession) -> BossEvidence:
        result = play_kraid_fight_to_varia(session)
        payload = result.to_dict()
        ok = bool(payload.get("success"))
        return BossEvidence(
            boss_id="kraid",
            start_frame=int(result.fight.start_frame),
            end_frame=int(result.varia.end_frame),
            outcome=str(result.varia.outcome if ok else result.fight.outcome),
            success=ok,
            final_room_id=int(result.varia.final_room_id),
            boss_defeated=bool(result.fight.boss_bit_set),
            detail=payload,
        )

    return CallableBossStrategy(
        boss_id="kraid",
        play_fn=play,
        entry_requirement=StateRequirement(room_id=ROOM_KRAID),
        catalog_entry=catalog,
    )


def wrap_ceres_ridley_as_boss_strategy() -> CallableBossStrategy:
    """Ceres Ridley tail-tank (countdown start is the win)."""
    from super_metroid.combat.ceres_ridley import (
        ROOM_CERES_RIDLEY,
        play_ceres_ridley_fight,
    )
    from super_metroid.combat.features import ceres_ridley_catalog

    return _wrap_simple_fight(
        boss_id="ceres_ridley",
        room_id=ROOM_CERES_RIDLEY,
        catalog_fn=ceres_ridley_catalog,
        play_fn=play_ceres_ridley_fight,
        success=lambda r: r.outcome == "ceres_ridley_countdown",
        boss_defeated=lambda r: r.outcome == "ceres_ridley_countdown",
    )


def wrap_bomb_torizo_as_boss_strategy() -> CallableBossStrategy:
    """Bomb Torizo fight as a BossStrategy (activation expected)."""
    from super_metroid.combat.bomb_torizo import (
        ROOM_BOMB_TORIZO,
        play_bomb_torizo_fight,
    )
    from super_metroid.combat.features import bomb_torizo_catalog

    return _wrap_simple_fight(
        boss_id="bomb_torizo",
        room_id=ROOM_BOMB_TORIZO,
        catalog_fn=bomb_torizo_catalog,
        play_fn=play_bomb_torizo_fight,
        success=lambda r: r.outcome == "bomb_torizo_defeated"
        or (r.final_enemy_hp == 0 and r.defeat_frame is not None),
    )


def wrap_spore_spawn_as_boss_strategy() -> CallableBossStrategy:
    """Spore Spawn left-ledge missile policy (Clean / no-assist)."""
    from super_metroid.combat.features import spore_spawn_catalog
    from super_metroid.combat.spore_spawn import (
        ROOM_SPORE_SPAWN,
        play_spore_spawn_fight,
    )

    return _wrap_simple_fight(
        boss_id="spore_spawn",
        room_id=ROOM_SPORE_SPAWN,
        catalog_fn=spore_spawn_catalog,
        play_fn=play_spore_spawn_fight,
        success=lambda r: r.outcome == "spore_spawn_defeated",
        boss_defeated=lambda r: r.defeat_frame is not None,
    )


def wrap_phantoon_as_boss_strategy() -> CallableBossStrategy:
    """Phantoon doppler fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.features import phantoon_catalog
    from super_metroid.combat.phantoon import ROOM_PHANTOON
    from super_metroid.combat.phantoon_doppler import play_phantoon_doppler_fight

    return _wrap_simple_fight(
        boss_id="phantoon",
        room_id=ROOM_PHANTOON,
        catalog_fn=phantoon_catalog,
        play_fn=play_phantoon_doppler_fight,
        success=lambda r: r.outcome == "phantoon_defeated",
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_botwoon_as_boss_strategy() -> CallableBossStrategy:
    """Botwoon development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.botwoon import ROOM_BOTWOON, play_botwoon_fight
    from super_metroid.combat.features import botwoon_catalog

    return _wrap_simple_fight(
        boss_id="botwoon",
        room_id=ROOM_BOTWOON,
        catalog_fn=botwoon_catalog,
        play_fn=play_botwoon_fight,
        success=lambda r: r.outcome == "botwoon_defeated",
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_draygon_as_boss_strategy() -> CallableBossStrategy:
    """Draygon development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.draygon import ROOM_DRAYGON, play_draygon_fight
    from super_metroid.combat.features import draygon_catalog

    return _wrap_simple_fight(
        boss_id="draygon",
        room_id=ROOM_DRAYGON,
        catalog_fn=draygon_catalog,
        play_fn=play_draygon_fight,
        success=lambda r: r.outcome == "draygon_defeated",
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_ridley_as_boss_strategy() -> CallableBossStrategy:
    """Ridley development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.features import ridley_catalog
    from super_metroid.combat.ridley import ROOM_RIDLEY, play_ridley_fight

    return _wrap_simple_fight(
        boss_id="ridley",
        room_id=ROOM_RIDLEY,
        catalog_fn=ridley_catalog,
        play_fn=play_ridley_fight,
        success=lambda r: r.outcome == "ridley_defeated",
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_mother_brain_as_boss_strategy() -> CallableBossStrategy:
    """Mother Brain development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.features import mother_brain_catalog
    from super_metroid.combat.mother_brain import (
        ROOM_MOTHER_BRAIN,
        play_mother_brain_fight,
    )

    return _wrap_simple_fight(
        boss_id="mother_brain",
        room_id=ROOM_MOTHER_BRAIN,
        catalog_fn=mother_brain_catalog,
        play_fn=play_mother_brain_fight,
        success=lambda r: bool(r.event_set or r.boss_bit_set),
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_crocomire_as_boss_strategy() -> CallableBossStrategy:
    """Crocomire development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.crocomire import ROOM_CROCOMIRE, play_crocomire_fight
    from super_metroid.combat.features import crocomire_catalog

    return _wrap_simple_fight(
        boss_id="crocomire",
        room_id=ROOM_CROCOMIRE,
        catalog_fn=crocomire_catalog,
        play_fn=play_crocomire_fight,
        success=lambda r: r.outcome == "pushed",
        boss_defeated=lambda r: bool(r.boss_bit_set),
    )


def wrap_golden_torizo_as_boss_strategy() -> CallableBossStrategy:
    """Golden Torizo development fight as a BossStrategy; continuous is deferred."""
    from super_metroid.combat.features import golden_torizo_catalog
    from super_metroid.combat.golden_torizo import (
        ROOM_GOLDEN_TORIZO,
        play_golden_torizo_fight,
    )

    return _wrap_simple_fight(
        boss_id="golden_torizo",
        room_id=ROOM_GOLDEN_TORIZO,
        catalog_fn=golden_torizo_catalog,
        play_fn=play_golden_torizo_fight,
        success=lambda r: r.outcome == "golden_torizo_defeated",
    )


def strategy_summary(strategy: BossStrategy) -> dict[str, Any]:
    """Compact dict for probe reports / docs export."""
    cat = strategy.catalog
    return {
        "bossId": strategy.boss_id,
        "name": cat.name,
        "roomId": cat.room_id,
        "roomIdHex": f"0x{cat.room_id:04X}",
        "maxHp": cat.max_hp,
        "primaryWeapon": cat.primary_weapon,
        "closeout": cat.closeout,
        "continuousStatus": cat.continuous_status,
        "kpdrPriority": cat.kpdr_priority,
    }
