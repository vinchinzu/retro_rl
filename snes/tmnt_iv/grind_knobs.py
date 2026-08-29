"""Whitelist of tunables for local-model grinding experiments.

Production defaults match CombatProfile, SlashTactics, and form-2.
The local grind runner overrides a subset via ``override_knobs`` for
short probe trials; winners are ported by hand into production code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class GrindKnobs:
    """Integer-only knobs a dumb local model may tweak within bounds."""

    # Shared combat poke band
    attack_range: int = 56
    min_range: int = 8
    standoff: int = 24
    attack_hold: int = 2
    attack_gap: int = 5

    # Slash hybrid whiplash — grind on RaphFullHardBoss5 (char 8).
    # spin_dodge_adx 40 is a strong Raph probe KEEP (6,765f / 226 / 3 vs
    # 11,386 / 478 / 6) but continuous dry-runs regressed total damage via
    # later-stage RNG (spin40 → 5,474; spin44 → 5,152; spin52 → 4,667).
    # Keep 52 until a full-route re-tune absorbs the path change.
    slash_spin_dodge_adx: int = 52
    slash_spin_dodge_ady: int = 22
    slash_claw_dodge_adx: int = 44
    slash_punish_approach_adx: int = 54
    slash_back_attack_adx: int = 8
    slash_punish_cycle: int = 48
    slash_punish_cross: int = 10
    slash_approach_band: int = 48
    slash_cross_frames: int = 22
    slash_cross_frames_low: int = 16
    slash_attack_frames: int = 36
    slash_attack_frames_low: int = 40
    slash_low_hp: int = 48

    # Technodrome pink Foot / tank charge — RaphFullHardStage4 baseline
    # 30,379f / 886 dmg. hit_frames=8 and charge_min=30 both regress Raph.
    blocker_retreat_frames: int = 40
    blocker_retreat_dx: int = 55
    blocker_charge_min: int = 34
    blocker_charge_dx: int = 22
    blocker_charge_timeout: int = 70
    blocker_hit_frames: int = 10

    # Super Shredder form-2: offset hold / drop-window chip / hop-behind.
    # drop_chip_frames = brief Y mash after 0xEE/0xFE drop;
    # behind_hop_frames = B-hop duration when overlapping his lane;
    # blind_offset_timeout = aura-up wait fallback when anim is blind.
    # Do not retune Slash here.
    shredder_drop_chip_frames: int = 56
    shredder_behind_hop_frames: int = 12
    shredder_blind_offset_timeout: int = 48
    shredder_attack_adx: int = 48
    shredder_space_adx: int = 20


# Bounds prevent catastrophic nonsense from a 12B model.
KNOB_BOUNDS: dict[str, tuple[int, int]] = {
    "attack_range": (32, 96),
    "min_range": (4, 24),
    "standoff": (8, 48),
    "attack_hold": (1, 6),
    "attack_gap": (1, 12),
    "slash_spin_dodge_adx": (24, 80),
    "slash_spin_dodge_ady": (8, 40),
    "slash_claw_dodge_adx": (20, 72),
    "slash_punish_approach_adx": (32, 96),
    "slash_back_attack_adx": (4, 24),
    "slash_punish_cycle": (24, 80),
    "slash_punish_cross": (4, 24),
    "slash_approach_band": (24, 80),
    "slash_cross_frames": (8, 40),
    "slash_cross_frames_low": (6, 32),
    "slash_attack_frames": (16, 64),
    "slash_attack_frames_low": (16, 64),
    "slash_low_hp": (16, 96),
    "blocker_retreat_frames": (16, 80),
    "blocker_retreat_dx": (32, 96),
    "blocker_charge_min": (16, 60),
    "blocker_charge_dx": (10, 40),
    "blocker_charge_timeout": (40, 120),
    "blocker_hit_frames": (4, 24),
    "shredder_drop_chip_frames": (16, 80),
    "shredder_behind_hop_frames": (8, 40),
    "shredder_blind_offset_timeout": (16, 64),
    "shredder_attack_adx": (40, 120),
    "shredder_space_adx": (8, 48),
}

_ACTIVE = GrindKnobs()


def active_knobs() -> GrindKnobs:
    """Return the currently active knob set (production defaults unless overridden)."""
    return _ACTIVE


def knobs_as_dict(knobs: GrindKnobs | None = None) -> dict[str, int]:
    """Serialize knobs to a plain int dict."""
    return {k: int(v) for k, v in asdict(knobs or _ACTIVE).items()}


def clamp_knob_patch(patch: Mapping[str, Any]) -> dict[str, int]:
    """Keep only known keys and clamp each value into ``KNOB_BOUNDS``."""
    cleaned: dict[str, int] = {}
    for key, raw in patch.items():
        if key not in KNOB_BOUNDS:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        lo, hi = KNOB_BOUNDS[key]
        cleaned[key] = max(lo, min(hi, value))
    return cleaned


def merge_knobs(
    base: GrindKnobs,
    patch: Mapping[str, Any],
) -> GrindKnobs:
    """Return ``base`` with a clamped whitelist patch applied."""
    cleaned = clamp_knob_patch(patch)
    if not cleaned:
        return base
    return replace(base, **cleaned)


@contextmanager
def override_knobs(patch: Mapping[str, Any]) -> Iterator[GrindKnobs]:
    """Temporarily replace the process-wide active knobs."""
    global _ACTIVE
    previous = _ACTIVE
    next_knobs = merge_knobs(previous, patch)
    _ACTIVE = next_knobs
    try:
        yield next_knobs
    finally:
        _ACTIVE = previous


def knob_field_names() -> tuple[str, ...]:
    """Stable list of knob names for prompts and validation."""
    return tuple(f.name for f in fields(GrindKnobs))


def focus_knob_names(focus: str) -> list[str]:
    """Whitelist knobs for a probe target, plus shared combat poke keys."""
    if focus in {"technodrome_tank", "tokka_rahzar"}:
        prefix = "blocker_"
    elif focus == "super_shredder":
        prefix = "shredder_"
    else:
        prefix = "slash_"
    names = [k for k in KNOB_BOUNDS if k.startswith(prefix)]
    for shared in ("attack_range", "standoff", "attack_gap"):
        if shared in KNOB_BOUNDS and shared not in names:
            names.append(shared)
    return names


__all__ = [
    "GrindKnobs",
    "KNOB_BOUNDS",
    "active_knobs",
    "clamp_knob_patch",
    "focus_knob_names",
    "knob_field_names",
    "knobs_as_dict",
    "merge_knobs",
    "override_knobs",
]
