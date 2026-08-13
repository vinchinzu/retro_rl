"""Unit locks for the Moat shinespark controller surface."""

from __future__ import annotations

from super_metroid.routes.kpdr import moat as moat_mod


def test_moat_scaffold_exports() -> None:
    assert moat_mod.ROOM_MOAT == 0x95FF
    assert moat_mod.ROOM_KIHUNTER == 0x948C
    assert moat_mod.ROOM_WEST_OCEAN == 0x93FE
    for name in (
        "play_moat_cross",
        "play_moat_shinespark",
        "play_leave_moat_to_kihunter",
        "play_open_kihunter_moat_door",
        "play_clear_kihunter_room",
        "play_kihunter_pre_spark_pin",
        "play_kihunter_charge_store",
    ):
        assert callable(getattr(moat_mod, name)), name
