from __future__ import annotations

from zelda_i.menus import BOOT_MAX_FRAMES, BOOT_PERIOD, boot_to_level1_script


def test_boot_script_is_bounded_and_nonempty() -> None:
    script = list(boot_to_level1_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_start" in reasons


def test_boot_period_is_fast_open_loop() -> None:
    """Period 50–60 reaches ready under ~700f; 180 was ~1749f; 40 fails debounce."""
    assert 45 <= BOOT_PERIOD <= 60
    script = list(boot_to_level1_script())
    # One full cycle must include START / A / SELECT holds and idle gaps.
    cycle = script[:BOOT_PERIOD]
    by_reason = {}
    for frame in cycle:
        by_reason.setdefault(frame.reason, 0)
        by_reason[frame.reason] += 1
    assert by_reason.get("boot_start", 0) >= 6
    assert by_reason.get("boot_confirm", 0) >= 4
    assert by_reason.get("boot_select", 0) >= 3
    assert by_reason.get("boot_wait", 0) >= 20
