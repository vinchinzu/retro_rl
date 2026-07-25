"""Curriculum tier presets for MK1 speedrun training."""

from __future__ import annotations

# Difficulty tiers: (state_prefixes, weight, label)
# SNES MK1 has 2 endurance rounds. Goro = Endurance2B alias.
LIUKANG_TIERS_FULL = [
    (["Fight", "Match2", "Match3"], 0.10, "Easy (M1-M3)"),
    (["Match4", "Match5", "Match6"], 0.15, "Medium (M4-M6)"),
    (["Match7"], 0.10, "Mirror (M7)"),
    (["Endurance1", "Endurance1B", "Endurance2"], 0.20, "Endurance"),
    (["Goro"], 0.25, "Goro (sub-boss)"),
    (["ShangTsung"], 0.20, "Shang Tsung (final)"),
]

LIUKANG_TIERS_LADDER = [
    (["Fight", "Match2", "Match3"], 0.25, "Easy (M1-M3)"),
    (["Match4", "Match5", "Match6"], 0.35, "Medium (M4-M6)"),
    (["Match7"], 0.20, "Mirror (M7)"),
    (["Endurance1", "Endurance1B", "Endurance2"], 0.10, "Endurance"),
    (["Goro"], 0.05, "Goro (sub-boss)"),
    (["ShangTsung"], 0.05, "Shang Tsung (final)"),
]

LIUKANG_TIERS_ENDURANCE = [
    (["Fight", "Match2", "Match3"], 0.05, "Easy (M1-M3)"),
    (["Match4", "Match5", "Match6"], 0.05, "Medium (M4-M6)"),
    (["Match7"], 0.05, "Mirror (M7)"),
    (["Endurance1", "Endurance1B", "Endurance2"], 0.60, "Endurance"),
    (["Goro"], 0.15, "Goro (sub-boss)"),
    (["ShangTsung"], 0.10, "Shang Tsung (final)"),
]

LIUKANG_TIERS_BOSS = [
    (["Fight", "Match2", "Match3"], 0.05, "Easy (M1-M3)"),
    (["Match4", "Match5", "Match6"], 0.05, "Medium (M4-M6)"),
    (["Match7"], 0.05, "Mirror (M7)"),
    (["Endurance1", "Endurance1B", "Endurance2"], 0.15, "Endurance"),
    (["Goro"], 0.35, "Goro (sub-boss)"),
    (["ShangTsung"], 0.35, "Shang Tsung (final)"),
]

CURRICULUM_TIERS: dict[str, list[tuple[list[str], float, str]]] = {
    "full": LIUKANG_TIERS_FULL,
    "ladder": LIUKANG_TIERS_LADDER,
    "endurance": LIUKANG_TIERS_ENDURANCE,
    "boss": LIUKANG_TIERS_BOSS,
}


def get_liukang_tiers(curriculum: str) -> list[tuple[list[str], float, str]]:
    """Return tier weights for the selected curriculum preset."""
    if curriculum not in CURRICULUM_TIERS:
        valid = ", ".join(sorted(CURRICULUM_TIERS))
        raise ValueError(f"Unknown curriculum {curriculum!r}; choose one of: {valid}")
    return CURRICULUM_TIERS[curriculum]
