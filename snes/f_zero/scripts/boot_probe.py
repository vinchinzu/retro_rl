"""Boot F-Zero from reset and save a Mute City race-start state."""

from __future__ import annotations

from f_zero.menus import boot_to_mute_city_script
from f_zero.paths import GAME, GAME_DIR, RECORDINGS_DIR
from f_zero.ram import parse_game_state
from retro_harness.boot_probe import BootProbeConfig, main_boot_probe, run_boot_probe


def _extras(state) -> str:
    return f"speed_raw={state.extras['speed_raw']} lateral={state.extras['lateral']}"


CFG = BootProbeConfig(
    game=GAME,
    game_dir=GAME_DIR,
    recordings_dir=RECORDINGS_DIR,
    script=boot_to_mute_city_script,
    parse_state=parse_game_state,
    state_name="MuteCity",
    screenshot_name="boot_mute_city.png",
    label="RACE_READY",
    extras_fmt=_extras,
)


def run_probe(*, save_race: bool = True) -> int:
    """Reach the Mute City countdown and optionally save it."""
    return run_boot_probe(CFG, save=save_race)


def main() -> None:
    """CLI entry point."""
    raise SystemExit(main_boot_probe(CFG, walk_default=None))


if __name__ == "__main__":
    main()
