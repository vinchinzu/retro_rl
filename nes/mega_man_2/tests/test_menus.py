from __future__ import annotations

from mega_man_2.menus import (
    BOOT_MAX_FRAMES,
    CURSOR_HEAT,
    boot_to_heat_man_script,
    boot_to_level1_script,
)


def test_boot_script_is_bounded_and_nonempty() -> None:
    script = list(boot_to_level1_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_start" in reasons


def test_boot_heat_script_is_bounded_and_navigates() -> None:
    script = list(boot_to_heat_man_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_start" in reasons
    assert "boot_nav_heat" in reasons
    assert "boot_confirm" in reasons
    assert "boot_enter_stage" in reasons
    assert CURSOR_HEAT == 8
