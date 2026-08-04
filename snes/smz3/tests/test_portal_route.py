"""Portal route unit tests (no emulator)."""

from __future__ import annotations

from smz3.portal_route import (
    PORTAL_RED_DOOR_STATE,
    PORTAL_RESIDUE_STATE,
    RED_DOOR_X,
    RED_DOOR_Y_MAX,
    RED_DOOR_Y_MIN,
    STOP_AT_RED_DOOR,
    STOP_CHOICES,
    PortalSegmentResult,
    is_portal_residue,
    load_parlor_policy_buttons,
)
from smz3.portals import FORTUNE_TELLER_CAVE_ID, early_portal
from smz3.ram import ComboSnapshot
from smz3.world import ActiveWorld


def _snap(**overrides: object) -> ComboSnapshot:
    base = dict(
        frame=0,
        sm_game_state=8,
        sm_room_id=0x92FD,
        sm_area_index=0,
        sm_door_transition=0,
        sm_health=99,
        sm_max_health=99,
        sm_samus_x=480,
        sm_samus_y=880,
        sm_pose=1,
        z3_module=0x97,
        z3_submodule=0,
        z3_indoors=0,
        z3_room_id=0,
        z3_screen_id=0,
        z3_link_x=0,
        z3_link_y=0,
    )
    base.update(overrides)
    return ComboSnapshot(**base)  # type: ignore[arg-type]


def test_red_door_band() -> None:
    assert RED_DOOR_Y_MIN < RED_DOOR_Y_MAX
    assert 400 < RED_DOOR_X < 600
    assert PORTAL_RED_DOOR_STATE == "PortalRedDoor"
    assert PORTAL_RESIDUE_STATE == "PortalResidue"
    assert STOP_AT_RED_DOOR in STOP_CHOICES


def test_parlor_policy_loads() -> None:
    btns = load_parlor_policy_buttons()
    assert len(btns) > 100
    assert len(btns[0]) == 12


def test_portal_residue_detection() -> None:
    assert is_portal_residue(
        _snap(
            sm_game_state=61560,
            sm_room_id=0,
            sm_health=0,
            z3_module=0x0F,
            z3_room_id=FORTUNE_TELLER_CAVE_ID,
            z3_indoors=1,
        )
    )
    assert not is_portal_residue(_snap())


def test_portal_result_dict() -> None:
    result = PortalSegmentResult(
        ok=True,
        goal="landing_to_portal",
        frames=5000,
        boot_frames=900,
        world=ActiveWorld.ALTTP,
        detail="portal residue",
        portal_started=True,
        z3_module=0x0F,
        z3_room_id=0x0122,
        z3_settled=False,
        assist_used=["missile_red_door"],
    )
    d = result.to_dict()
    assert d["portal_started"] is True
    assert d["z3_settled"] is False
    assert d["z3_room_id"] == "0x0122"
    assert d["portal"]["sm_door_ptr"] == "0x8976"
    assert d["assist_used"] == ["missile_red_door"]
    assert d["assist"][0]["assist_id"] == "missile_red_door"
    assert early_portal().sm_door_ptr == 0x8976
