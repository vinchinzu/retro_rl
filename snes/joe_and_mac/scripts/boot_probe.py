"""Boot Joe & Mac from reset and save a controllable Stage 1 state."""

from __future__ import annotations

from joe_and_mac.menus import boot_to_stage1_script
from joe_and_mac.paths import GAME, GAME_DIR, RECORDINGS_DIR
from joe_and_mac.ram import parse_game_state
from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe


def _extras(state) -> str:
    return f"progress={state.extras['horizontal_progress']}"


CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_stage1_script,
    parse_state=parse_game_state,
    state_name="Stage1",
    screenshot_name="boot_stage1.png",
    label="STAGE1",
    extras_fmt=_extras,
)


def run_probe(*, save_stage1: bool = True) -> int:
    """Reach Stage 1, verify gameplay, and save the checkpoint."""
    return run_boot_probe(CFG, save=save_stage1)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=None))


if __name__ == "__main__":
    main()
