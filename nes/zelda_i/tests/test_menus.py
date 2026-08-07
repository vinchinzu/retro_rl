from __future__ import annotations

from zelda_i.menus import (
    BOOT_MAX_FRAMES,
    BOOT_PERIOD,
    boot_compact_first_slot_script,
    boot_to_level1_script,
)


def test_boot_script_is_bounded_and_nonempty() -> None:
    script = list(boot_to_level1_script())
    assert 0 < len(script) <= BOOT_MAX_FRAMES
    reasons = {frame.reason for frame in script}
    assert "boot_wait" in reasons
    assert "boot_title_start" in reasons or "boot_start" in reasons


def test_compact_boot_selects_first_slot_without_file_select() -> None:
    """Compact path: title → slot 1 START only; SELECT only on name entry."""
    compact = list(boot_compact_first_slot_script())
    assert 80 <= len(compact) <= 120
    reasons = [fa.reason for fa in compact]
    assert "boot_title_start" in reasons
    assert "boot_slot1" in reasons
    assert "boot_name_letter" in reasons
    assert "boot_name_confirm" in reasons
    assert "boot_begin" in reasons
    # No open-loop file-menu SELECT spam before name entry.
    pre_name = []
    for fa in compact:
        if fa.reason == "boot_name_cursor":
            break
        pre_name.append(fa.reason)
    assert "boot_select" not in pre_name
    assert all(r != "boot_select" for r in pre_name)


def test_boot_period_fallback_bounds() -> None:
    """Fallback period still in the verified debounce window."""
    assert 45 <= BOOT_PERIOD <= 60


def test_full_script_includes_compact_prefix() -> None:
    script = list(boot_to_level1_script())
    compact = list(boot_compact_first_slot_script())
    assert [fa.reason for fa in script[: len(compact)]] == [
        fa.reason for fa in compact
    ]
