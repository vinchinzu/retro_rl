"""Boot Super Mario Bros. (NES) from reset and save a controllable Level1 state."""

from __future__ import annotations

from smb.menus import boot_to_level1_script
from smb.paths import GAME, GAME_DIR, RECORDINGS_DIR
from smb.ram import is_level1_ready, parse_game_state
from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe

CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_level1_script,
    parse_state=parse_game_state,
    is_ready=is_level1_ready,
    min_frame=200,
    stable_frames=20,
    motion_check=True,
)


def run_probe(*, save_level1: bool = True, walk_frames: int = 30) -> int:
    """Reach Level 1, verify readiness, optionally walk, and save checkpoint."""
    return run_boot_probe(CFG, save=save_level1, walk_frames=walk_frames)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=30))


if __name__ == "__main__":
    main()
