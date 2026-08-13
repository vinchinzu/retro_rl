"""Map Rando tech → bot builder skill registry.

Room optimization and hop hill-climb should prefer **named tech** from
:mod:`super_metroid.rooms.tech_catalog` (sm-json-data + Map Rando difficulty)
and resolve callables here.

Only techs with a registered builder are "executable builders". Green-path
shine/walljump already live in their modules; this table is the index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from super_metroid.rooms.tech_catalog import builder_targets, tech_by_name

# Lazy imports avoided at module level for heavy skills — registry stores
# dotted paths resolved on demand.


@dataclass(frozen=True)
class BuilderSkill:
    """One executable builder linked to a Map Rando tech name."""

    tech: str
    callable_path: str
    difficulty: str
    status: str  # green | partial | experimental
    summary: str

    def resolve(self) -> Callable[..., Any]:
        module_path, _, attr = self.callable_path.rpartition(".")
        if not module_path:
            raise ValueError(f"bad callable_path: {self.callable_path}")
        import importlib

        mod = importlib.import_module(module_path)
        fn = getattr(mod, attr)
        if not callable(fn):
            raise TypeError(f"{self.callable_path} is not callable")
        return fn


# Core + try builders the bot can invoke today (or thin wrappers thereof).
BUILDER_SKILLS: tuple[BuilderSkill, ...] = (
    # --- Implicit / Basic core ---
    BuilderSkill(
        tech="canDash",
        callable_path="super_metroid.routes.skills.basic_moves.dash",
        difficulty="Implicit",
        status="green",
        summary="Hold dash + direction",
    ),
    BuilderSkill(
        tech="canStopOnADime",
        callable_path="super_metroid.routes.skills.basic_moves.stop_on_a_dime",
        difficulty="Implicit",
        status="partial",
        summary="Angle hold to kill momentum",
    ),
    BuilderSkill(
        tech="canTrivialMidAirMorph",
        callable_path="super_metroid.routes.skills.basic_moves.mid_air_morph",
        difficulty="Implicit",
        status="green",
        summary="Morph mid-air (trivial)",
    ),
    BuilderSkill(
        tech="canMidAirMorph",
        callable_path="super_metroid.routes.skills.basic_moves.mid_air_morph",
        difficulty="Basic",
        status="green",
        summary="Morph mid-air",
    ),
    BuilderSkill(
        tech="canWallJump",
        callable_path="super_metroid.routes.skills.walljump.walljump_once",
        difficulty="Basic",
        status="green",
        summary="Single wall-jump pulse",
    ),
    BuilderSkill(
        tech="canShinespark",
        callable_path="super_metroid.routes.skills.shinespark.charge_store_activate",
        difficulty="Basic",
        status="green",
        summary="Full charge → store → activate",
    ),
    BuilderSkill(
        tech="canEscapeEnemyGrab",
        callable_path="super_metroid.routes.skills.knockback.escape_kb",
        difficulty="Implicit",
        status="partial",
        summary="Knockback / grab escape",
    ),
    # --- Medium try ---
    BuilderSkill(
        tech="canCrouchJump",
        callable_path="super_metroid.routes.skills.basic_moves.crouch_jump",
        difficulty="Medium",
        status="partial",
        summary="Crouch then jump",
    ),
    BuilderSkill(
        tech="canDownGrab",
        callable_path="super_metroid.routes.skills.basic_moves.down_grab",
        difficulty="Medium",
        status="partial",
        summary="Hold DOWN to grab ledge",
    ),
    BuilderSkill(
        tech="canSpeedyJump",
        callable_path="super_metroid.routes.skills.basic_moves.speedy_jump",
        difficulty="Medium",
        status="partial",
        summary="Dash then jump",
    ),
    BuilderSkill(
        tech="canPreciseWallJump",
        callable_path="super_metroid.routes.skills.walljump.walljump_once",
        difficulty="Medium",
        status="green",
        summary="Tight-timing wall-jump (same pulse, tight policy)",
    ),
    BuilderSkill(
        tech="canConsecutiveWallJump",
        callable_path="super_metroid.routes.skills.walljump.consecutive_walljumps",
        difficulty="Medium",
        status="green",
        summary="Chained wall-jumps",
    ),
    BuilderSkill(
        tech="canHorizontalShinespark",
        callable_path="super_metroid.routes.skills.shinespark.activate_shinespark",
        difficulty="Medium",
        status="green",
        summary="Horizontal spark activation",
    ),
    BuilderSkill(
        tech="canShinechargeMovement",
        callable_path="super_metroid.routes.skills.shinespark.charge_until_boost",
        difficulty="Medium",
        status="green",
        summary="Charge while moving to full echoes",
    ),
    BuilderSkill(
        tech="canMidairShinespark",
        callable_path="super_metroid.routes.skills.shinespark.store_then_spin_unspin_activate",
        difficulty="Medium",
        status="green",
        summary="Store, spin, unspin, mid-air activate",
    ),
    BuilderSkill(
        tech="canIBJ",
        callable_path="super_metroid.routes.skills.morph_bomb.morph_bomb_hole_climb",
        difficulty="Medium",
        status="partial",
        summary="Vertical morph bomb climb (IBJ-like; not general height IBJ)",
    ),
    BuilderSkill(
        tech="canBombHorizontally",
        callable_path="super_metroid.routes.skills.morph_bomb.morph_roll_to_window",
        difficulty="Medium",
        status="partial",
        summary="Morph roll / bomb window (horizontal bias partial)",
    ),
)


def builder_skill(tech: str) -> BuilderSkill | None:
    for skill in BUILDER_SKILLS:
        if skill.tech == tech:
            return skill
    return None


def list_builder_skills(
    *,
    difficulty: str | None = None,
    status: str | None = None,
) -> list[BuilderSkill]:
    rows = list(BUILDER_SKILLS)
    if difficulty is not None:
        rows = [s for s in rows if s.difficulty == difficulty]
    if status is not None:
        rows = [s for s in rows if s.status == status]
    return rows


def registered_tech_names() -> frozenset[str]:
    return frozenset(s.tech for s in BUILDER_SKILLS)


def builder_gap_report() -> dict[str, Any]:
    """Compare catalog builder targets vs registered skills."""
    targets = builder_targets()
    registered = registered_tech_names()
    missing = [t.name for t in targets if t.name not in registered]
    extra = sorted(registered - {t.name for t in targets})
    by_diff: dict[str, list[str]] = {}
    for name in missing:
        node = tech_by_name(name)
        d = node.difficulty if node else "?"
        by_diff.setdefault(d, []).append(name)
    return {
        "registered": sorted(registered),
        "registered_count": len(registered),
        "target_count": len(targets),
        "unregistered_targets": missing,
        "unregistered_by_difficulty": by_diff,
        "extra_registered": extra,
    }


__all__ = [
    "BUILDER_SKILLS",
    "BuilderSkill",
    "builder_gap_report",
    "builder_skill",
    "list_builder_skills",
    "registered_tech_names",
]
