"""Integer constants for the base-policy tactics (Slash, Technodrome, form-2).

These used to be overridden by an LLM knob-grinding loop (deleted
2026-11-14). Safety is now the lookahead guard's job (``tmnt_iv.lookahead``);
these only shape how the base policy makes progress.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TacticKnobs:
    """Base-policy tactic constants."""

    # Shared combat poke band
    attack_range: int = 56
    min_range: int = 8
    standoff: int = 24
    attack_hold: int = 2
    attack_gap: int = 5

    # Slash hybrid whiplash, tuned on RaphFullHardBoss5 (char 8).
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



_ACTIVE = TacticKnobs()


def active_knobs() -> TacticKnobs:
    """The production knob set."""
    return _ACTIVE


__all__ = ["TacticKnobs", "active_knobs"]
