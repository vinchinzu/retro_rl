"""Unit locks for Below→Bat sill pin (Ice→Moat compose handoff)."""

from super_metroid.routes.kpdr.k5.below_to_bat import _on_bat_right_sill
from super_metroid.routes.kpdr.k5.geometry import (
    BAT_SILL_X_MAX,
    BAT_SILL_X_MIN,
    BAT_SILL_Y_MAX,
    BAT_SILL_Y_MIN,
)


def test_bat_sill_window_covers_dual_green_pin() -> None:
    assert BAT_SILL_X_MIN <= 472 <= BAT_SILL_X_MAX
    assert BAT_SILL_Y_MIN <= 139 <= BAT_SILL_Y_MAX


def test_on_bat_right_sill_rejects_mid_platform_morph() -> None:
    from types import SimpleNamespace

    ok = SimpleNamespace(samus_x=472, samus_y=139, velocity_y=0, pose=12)
    mid_morph = SimpleNamespace(samus_x=411, samus_y=190, velocity_y=0, pose=42)
    assert _on_bat_right_sill(ok) is True
    assert _on_bat_right_sill(mid_morph) is False
