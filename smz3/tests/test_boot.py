"""Boot helper unit tests (synthetic; live probe is scripts/probe_boot.py)."""

from __future__ import annotations

from smz3.boot import BootResult, LANDING_SITE_ROOM_ID
from smz3.ram import ComboSnapshot
from smz3.world import ActiveWorld, context_for


def _snap(**overrides: object) -> ComboSnapshot:
    base = dict(
        frame=0,
        sm_game_state=8,
        sm_room_id=LANDING_SITE_ROOM_ID,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=99,
        sm_max_health=99,
        sm_samus_x=1152,
        sm_samus_y=1088,
        sm_pose=0,
        z3_module=151,
        z3_submodule=0,
        z3_indoors=0,
        z3_room_id=0,
        z3_screen_id=0,
        z3_link_x=0,
        z3_link_y=0,
    )
    base.update(overrides)
    return ComboSnapshot(**base)  # type: ignore[arg-type]


def test_boot_result_dict_landing_site() -> None:
    snap = _snap(frame=868)
    world = ActiveWorld.SUPER_METROID
    result = BootResult(
        ok=True,
        frames=868,
        snapshot=snap,
        world=world,
        context=context_for(world),
        detail="test",
    )
    d = result.to_dict()
    assert d["ok"] is True
    assert d["frames"] == 868
    assert d["landing_site"] is True
    assert d["sm_controllable"] is True
    assert d["world"] == "super_metroid"


def test_landing_site_constant() -> None:
    assert LANDING_SITE_ROOM_ID == 0x91F8
