from __future__ import annotations

from rival_turf.menus import BOOT_SCRIPT_FRAMES, boot_to_stage1_script


def test_boot_script_has_verified_length_and_input_pulses() -> None:
    script = list(boot_to_stage1_script())

    assert len(script) == BOOT_SCRIPT_FRAMES
    reasons = {frame.reason for frame in script}
    assert reasons == {"boot_start", "boot_confirm", "boot_wait"}
    assert script[-1].reason == "boot_wait"

