from __future__ import annotations

from joe_and_mac.menus import BOOT_SCRIPT_FRAMES, boot_to_stage1_script


def test_boot_script_has_verified_length_and_stage_wait_tail() -> None:
    script = list(boot_to_stage1_script())

    assert len(script) == BOOT_SCRIPT_FRAMES
    assert script[-1].reason == "stage_wait"
    assert {frame.reason for frame in script} == {
        "boot_start",
        "boot_confirm",
        "boot_confirm_alt",
        "boot_wait",
        "map_up",
        "map_wait",
        "map_right",
        "map_select_node",
        "map_confirm",
        "stage_wait",
    }
