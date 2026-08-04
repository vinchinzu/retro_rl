"""Filesystem constants for TMNT IV."""

from __future__ import annotations

from pathlib import Path

from retro_harness.artifacts import clean_artifact_stem
from retro_harness.game_layout import game_paths

_paths = game_paths(__file__, "TMNTIV-Snes")
GAME_DIR = _paths.game_dir
REPO_ROOT = _paths.repo_root
INTEGRATION = _paths.integration
GAME = INTEGRATION
INTEGRATION_DIR = _paths.integration_dir
RECORDINGS_DIR = _paths.recordings_dir
ROMS_DIR = _paths.roms_dir
DOCS_DIR = _paths.docs_dir

# Continuous full-run artifact basenames (see docs/CLEAN_TRACK.md).
ASSISTED_FULL_RUN_STEM = "tmnt_iv_full_hard_credits"
ASSISTED_FULL_RUN_DRY_REPORT = "tmnt_iv_full_hard_dry_run.json"
CLEAN_FULL_RUN_STEM = "tmnt_iv_full_hard_clean"

STAGE1_STATE = "Stage1"
STAGE1_CLEAR_STATE = "Stage1_Clear"
STAGE1_BEFORE_BOSS_STATE = "Stage1_BeforeBoss"
STAGE2_STATE = "Stage2"
STAGE2_CLEAR_STATE = "Stage2_Clear"
STAGE2_BEFORE_BOSS_STATE = "Boss2"
STAGE3_STATE = "Stage3"
STAGE3_CLEAR_STATE = "Stage3_Clear"
STAGE3_BEFORE_BOSS_STATE = "Boss3"
STAGE4_STATE = "Stage4"
STAGE4_CLEAR_STATE = "Stage4_Clear"
STAGE4_BEFORE_BOSS_STATE = "Boss4"
STAGE5_STATE = "Stage5"
STAGE5_CLEAR_STATE = "Stage5_Clear"
STAGE5_BEFORE_BOSS_STATE = "Boss5"
STAGE6_STATE = "Stage6"
STAGE6_CLEAR_STATE = "Stage6_Clear"
STAGE6_BEFORE_BOSS_STATE = "Boss6"
STAGE7_STATE = "Stage7"
STAGE7_CLEAR_STATE = "Stage7_Clear"
STAGE7_BEFORE_BOSS_STATE = "Boss7"
STAGE8_STATE = "Stage8"
STAGE8_CLEAR_STATE = "Stage8_Clear"
STAGE8_BEFORE_BOSS_STATE = "Boss8"
STAGE9_STATE = "Stage9"
STAGE9_CLEAR_STATE = "Stage9_Clear"
STAGE9_BEFORE_BOSS_STATE = "Boss9"
ENDING_STATE = "Ending"


# Re-export shared helper (canonical: retro_harness.artifacts.clean_artifact_stem).


def default_full_run_paths(
    *,
    clean: bool = False,
    dry_run: bool = False,
) -> tuple[Path, Path]:
    """Default video/report paths for continuous full hard runs.

    Assisted defaults stay ``tmnt_iv_full_hard_credits.*`` (dry-run renames the
    report to ``tmnt_iv_full_hard_dry_run.json`` — historical name, not
    ``{stem}_dry_run``). Clean defaults use the ``tmnt_iv_full_hard_clean``
    stem and never equal assisted basenames.
    """
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if clean:
        stem = CLEAN_FULL_RUN_STEM  # already ends with _clean
        video = RECORDINGS_DIR / f"{stem}.mp4"
        report = (
            RECORDINGS_DIR / f"{stem}_dry_run.json"
            if dry_run
            else RECORDINGS_DIR / f"{stem}.json"
        )
        return video, report

    video = RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.mp4"
    if dry_run:
        report = RECORDINGS_DIR / ASSISTED_FULL_RUN_DRY_REPORT
    else:
        report = RECORDINGS_DIR / f"{ASSISTED_FULL_RUN_STEM}.json"
    return video, report
