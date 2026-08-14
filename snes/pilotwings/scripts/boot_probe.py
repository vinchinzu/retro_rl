"""Boot Pilotwings from reset and save an airborne Lesson 1 checkpoint."""

from __future__ import annotations

from pilotwings.menus import boot_to_lesson1_plane_script
from pilotwings.paths import GAME, GAME_DIR, RECORDINGS_DIR
from pilotwings.ram import parse_game_state
from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe


def _extras(state) -> str:
    return f"altitude={state.extras['altitude']}"


CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_lesson1_plane_script,
    parse_state=parse_game_state,
    state_name="Lesson1Plane",
    screenshot_name="boot_lesson1_plane.png",
    label="LESSON1_PLANE",
    extras_fmt=_extras,
)


def run_probe(*, save_lesson: bool = True) -> int:
    """Reach the first light-plane lesson, verify flight, and save it."""
    return run_boot_probe(CFG, save=save_lesson)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=None))


if __name__ == "__main__":
    main()
