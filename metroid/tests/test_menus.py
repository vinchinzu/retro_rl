from __future__ import annotations

from metroid.menus import boot_to_level1_script


def test_boot_script_yields_actions() -> None:
    actions = list(boot_to_level1_script())
    assert len(actions) > 100
    reasons = {a.reason for a in actions}
    assert "boot_start" in reasons
