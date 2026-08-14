"""Boot The Magical Quest from reset and save Stage1.state."""

from __future__ import annotations

from magical_quest.menus import boot_to_stage1_script
from magical_quest.paths import GAME, GAME_DIR, RECORDINGS_DIR
from magical_quest.ram import parse_game_state
from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe


def _extras(state) -> str:
    return f"x={state.player_x} progress={state.camera_x}"


CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_stage1_script,
    parse_state=parse_game_state,
    state_name="Stage1",
    screenshot_name="boot_stage1.png",
    label="LEVEL_READY",
    extras_fmt=_extras,
)


def run_probe(*, save_stage1: bool = True) -> int:
    """Reach the first controllable room and optionally save it."""
    return run_boot_probe(CFG, save=save_stage1)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=None))


if __name__ == "__main__":
    main()
