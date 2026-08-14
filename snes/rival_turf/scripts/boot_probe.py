"""Boot Rival Turf! from reset and save a fight-ready Stage1.state."""

from __future__ import annotations

from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe
from rival_turf.menus import boot_to_stage1_script
from rival_turf.paths import GAME, GAME_DIR, RECORDINGS_DIR
from rival_turf.ram import parse_game_state


def _extras(state) -> str:
    return f"pos=({state.player_x},{state.player_y})"


CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_stage1_script,
    parse_state=parse_game_state,
    state_name="Stage1",
    screenshot_name="boot_stage1.png",
    label="FIGHT_READY",
    post_script_button="RIGHT",
    post_script_frames=360,
    extras_fmt=_extras,
)


def run_probe(*, approach_frames: int = 360, save_stage1: bool = True) -> int:
    """Reach Stage 1, approach its opening combat lock, and save it."""
    return run_boot_probe(CFG, save=save_stage1, post_script_frames=approach_frames)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=None, approach_default=360))


if __name__ == "__main__":
    main()
