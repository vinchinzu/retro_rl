from __future__ import annotations

from f_zero.menus import BOOT_SCRIPT_FRAMES, boot_to_mute_city_script


def test_boot_script_has_verified_length_and_finishes_idle() -> None:
    script = list(boot_to_mute_city_script())

    assert len(script) == BOOT_SCRIPT_FRAMES
    assert script[-1].reason == "boot_wait"
    assert {frame.reason for frame in script} == {
        "boot_start",
        "boot_confirm",
        "boot_confirm_alt",
        "boot_wait",
    }

