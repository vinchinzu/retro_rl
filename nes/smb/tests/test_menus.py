from __future__ import annotations

from smb.menus import BOOT_MAX_FRAMES, boot_to_level1_script, boot_to_ready, idle_n


def test_boot_helpers_are_library_entry_points() -> None:
    assert callable(boot_to_ready)
    assert callable(idle_n)


def test_boot_script_is_bounded_and_nonempty() -> None:
    script = list(boot_to_level1_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_start" in reasons
