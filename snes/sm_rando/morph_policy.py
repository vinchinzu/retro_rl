"""Run the SM Rando integration from power-on through Morph Ball."""

from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env
from sm_rando.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR
from super_metroid.routes.continuous import ContinuousRunReport, run_tip
from super_metroid.video import VideoCaptureConfig

MORPH_POLICY_VIDEO = RECORDINGS_DIR / "policy_to_morph.mp4"
MORPH_POLICY_REPORT = RECORDINGS_DIR / "policy_to_morph.json"


def _build_env():
    """Build the actual SMRando-Snes emulator integration at power-on."""
    return make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")


def run_morph_policy(
    *,
    video_path: str | Path | None = MORPH_POLICY_VIDEO,
    video_config: VideoCaptureConfig | None = None,
    report_path: str | Path | None = MORPH_POLICY_REPORT,
) -> ContinuousRunReport:
    """Execute the shared policy on ``SMRando-Snes`` through first item.

    The current integration ROM is the documented vanilla substrate. The run
    starts from emulator power-on, performs no state loads or resource assists,
    and stops only after the Morph Ball acquisition is observed.
    """
    integration_rom = INTEGRATION_DIR / "rom.sfc"
    if not integration_rom.is_file():
        raise FileNotFoundError(
            "SMRando-Snes ROM is not configured; run "
            "`uv run python -m sm_rando.scripts.setup_rom`"
        )
    return run_tip(
        "morph",
        env_factory=_build_env,
        rom_path=integration_rom,
        video_path=video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=False,
        unlimited_ammo=False,
        require_clean_resources=True,
    )


__all__ = [
    "MORPH_POLICY_REPORT",
    "MORPH_POLICY_VIDEO",
    "run_morph_policy",
]
