"""Unit tests for continuous Segment / ContinuousSession contracts."""

from __future__ import annotations

from super_metroid.routes.catalog import (
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
    list_continuous_tips,
)
from super_metroid.routes.segment import (
    ContinuousSession,
    ControllerSegment,
    segment_from_kpdr,
)


def test_default_continuous_tip_is_verified_bat_cave() -> None:
    assert DEFAULT_CONTINUOUS_TIP == "bat_cave"
    assert get_continuous_tip("bat_cave").tip_id == "bat_cave"
    assert get_continuous_tip("k4_4").tip_id == "bat_cave"
    assert get_continuous_tip("frog").tip_id == "frog"
    assert get_continuous_tip("kraid").tip_id == "kraid"
    assert get_continuous_tip("hijump").tip_id == "hijump"


def test_hijump_kraid_varia_tips_registered() -> None:
    ids = {t.tip_id for t in CONTINUOUS_TIPS}
    assert {"hijump", "kraid", "varia", "warehouse"}.issubset(ids)
    # Prefix order: warehouse before hijump before kraid before varia
    ordered = [t.tip_id for t in list_continuous_tips()]
    assert ordered.index("warehouse") < ordered.index("hijump")
    assert ordered.index("hijump") < ordered.index("kraid")
    assert ordered.index("kraid") < ordered.index("varia")


def test_controller_segment_from_kpdr_registry() -> None:
    seg = segment_from_kpdr(
        "warehouse_to_business",
        entry_room=0xA6A1,
        exit_room=0xA7DE,
    )
    assert isinstance(seg, ControllerSegment)
    assert seg.id == "warehouse_to_business"
    assert callable(seg.play_fn)


def test_continuous_session_run_to_dispatch() -> None:
    session = ContinuousSession(tip="warehouse")
    # Facade resolves tip without binding a live env when only inspecting tip.
    assert session.tip == "warehouse"
    # segment adapters exist for composed packages used by continuous hops
    assert segment_from_kpdr("eye_to_kraid").id == "eye_to_kraid"
    assert segment_from_kpdr("kraid_entry_to_varia").id == "kraid_entry_to_varia"
