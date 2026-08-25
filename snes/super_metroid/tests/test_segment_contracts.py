"""Unit tests for practice Segment adapters (not the continuous hop runner)."""

from __future__ import annotations

from super_metroid.policy import PolicySegment, StateRequirement
from super_metroid.routes.catalog import (
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
    list_continuous_tips,
)
from super_metroid.routes.segment import (
    ControllerSegment,
    PolicySegmentAdapter,
    Segment,
    segment_from_kpdr,
)


def test_default_continuous_tip_is_verified_phantoon() -> None:
    assert DEFAULT_CONTINUOUS_TIP == "phantoon"
    assert get_continuous_tip("phantoon").tip_id == "phantoon"
    assert get_continuous_tip("ice").tip_id == "ice"
    assert get_continuous_tip("wave").tip_id == "wave"
    assert get_continuous_tip("speed").tip_id == "speed"
    assert get_continuous_tip("k4_5").tip_id == "speed"
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
    assert isinstance(seg, Segment)
    assert seg.id == "warehouse_to_business"
    assert callable(seg.play_fn)
    assert seg.entry_room == 0xA6A1
    assert seg.exit_room == 0xA7DE


def test_segment_from_kpdr_composed_packages() -> None:
    assert segment_from_kpdr("eye_to_kraid").id == "eye_to_kraid"
    assert segment_from_kpdr("kraid_entry_to_varia").id == "kraid_entry_to_varia"


def test_policy_segment_adapter_wraps_policy() -> None:
    policy = PolicySegment(
        segment_id="unit_policy_seg",
        filename="unit_policy_seg.json",
        entry=StateRequirement(),
        exit=StateRequirement(),
        expected_policy_id="deadbeef",
    )
    adapter = PolicySegmentAdapter(segment=policy)
    assert isinstance(adapter, Segment)
    assert adapter.id == "unit_policy_seg"
    assert adapter.entry is policy.entry
    assert adapter.exit is policy.exit
