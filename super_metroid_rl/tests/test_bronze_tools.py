"""No-ROM tests for Super Metroid Bronze tooling helpers."""

from pathlib import Path

from platformer_common.auto_state import NavStep, parse_nav_string
from super_metroid_rl.bronze_tools import (
    get_boot_macro_expectation,
    get_boot_macro_steps,
    inspect_export,
    list_boot_macros,
    repeat_nav_steps,
)


def test_repeat_nav_steps_repeats_in_order():
    steps = [
        NavStep(buttons=[3], hold_frames=10, wait_frames=20),
        NavStep(buttons=[8], hold_frames=5, wait_frames=0),
    ]

    repeated = repeat_nav_steps(steps, 3)

    assert repeated == steps * 3


def test_repeat_nav_steps_rejects_invalid_repeat():
    try:
        repeat_nav_steps([], 0)
    except ValueError as exc:
        assert "repeat must be >=" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for repeat=0")


def test_parse_nav_string_accepts_wait_steps():
    steps = parse_nav_string("WAIT:0:2100 A:10:120")

    assert steps == [
        NavStep(buttons=[], hold_frames=0, wait_frames=2100),
        NavStep(buttons=[8], hold_frames=10, wait_frames=120),
    ]


def test_none_to_start_boot_macro_shape():
    assert "none_to_start" in list_boot_macros()

    steps = get_boot_macro_steps("none_to_start")

    assert steps[0] == NavStep(buttons=[], hold_frames=0, wait_frames=2100)
    assert steps[1] == NavStep(buttons=[8], hold_frames=10, wait_frames=120)
    assert steps[2] == NavStep(buttons=[8], hold_frames=10, wait_frames=300)
    assert steps[3] == NavStep(buttons=[8], hold_frames=10, wait_frames=30)
    assert len(steps) == 73
    assert steps[-1] == NavStep(buttons=[8], hold_frames=10, wait_frames=110)
    assert get_boot_macro_expectation("none_to_start") == (0xDF45, 8)


def test_ceres_start_to_ridley_ground_macro_shape():
    assert "ceres_start_to_ridley_ground" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_start_to_ridley_ground")

    assert steps == [
        NavStep(buttons=[7, 8], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7], hold_frames=120, wait_frames=0),
        NavStep(buttons=[6], hold_frames=120, wait_frames=0),
        NavStep(buttons=[7, 0], hold_frames=240, wait_frames=60),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7, 0], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7, 0, 8], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[7, 0], hold_frames=24, wait_frames=12),
        NavStep(buttons=[7], hold_frames=24, wait_frames=0),
        NavStep(buttons=[], hold_frames=0, wait_frames=140),
        NavStep(buttons=[7], hold_frames=160, wait_frames=0),
        NavStep(buttons=[6], hold_frames=120, wait_frames=0),
        NavStep(buttons=[7, 0], hold_frames=96, wait_frames=0),
        NavStep(buttons=[], hold_frames=0, wait_frames=120),
        NavStep(buttons=[7, 0], hold_frames=216, wait_frames=0),
        NavStep(buttons=[], hold_frames=0, wait_frames=150),
        NavStep(buttons=[7, 0], hold_frames=240, wait_frames=0),
        NavStep(buttons=[], hold_frames=0, wait_frames=200),
    ]
    assert get_boot_macro_expectation("ceres_start_to_ridley_ground") == (0xE0B5, 8)


def test_ceres_ridley_ground_to_27hp_wait_state_macro_shape():
    assert "ceres_ridley_ground_to_27hp_wait_state" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_ridley_ground_to_27hp_wait_state")

    assert steps == [NavStep(buttons=[], hold_frames=0, wait_frames=2321)]
    assert get_boot_macro_expectation("ceres_ridley_ground_to_27hp_wait_state") == (0xE0B5, 8)


def test_ceres_ridley_ground_27hp_to_elevator_room_macro_shape():
    assert "ceres_ridley_ground_27hp_to_elevator_room" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_ridley_ground_27hp_to_elevator_room")

    assert steps[0] == NavStep(buttons=[], hold_frames=0, wait_frames=540)
    assert steps[1:] == [
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=1000, wait_frames=0),
        NavStep(buttons=[8], hold_frames=16, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=124, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=60, wait_frames=0),
        NavStep(buttons=[6], hold_frames=320, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=380, wait_frames=0),
    ]
    assert get_boot_macro_expectation("ceres_ridley_ground_27hp_to_elevator_room") == (0xDF45, 8)


def test_ceres_pretrigger_to_elevator_room_macro_shape():
    assert "ceres_pretrigger_to_elevator_room" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_pretrigger_to_elevator_room")

    assert steps == [
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=1000, wait_frames=0),
        NavStep(buttons=[8], hold_frames=16, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=124, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=60, wait_frames=0),
        NavStep(buttons=[6], hold_frames=320, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=380, wait_frames=0),
    ]
    assert get_boot_macro_expectation("ceres_pretrigger_to_elevator_room") == (0xDF45, 8)


def test_ceres_ridley_appeared_to_elevator_room_macro_shape():
    assert "ceres_ridley_appeared_to_elevator_room" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_ridley_appeared_to_elevator_room")

    assert steps[0] == NavStep(buttons=[], hold_frames=0, wait_frames=1888)
    assert steps[1:] == [
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=1000, wait_frames=0),
        NavStep(buttons=[8], hold_frames=16, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=124, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=60, wait_frames=0),
        NavStep(buttons=[6], hold_frames=320, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=40, wait_frames=0),
        NavStep(buttons=[6], hold_frames=380, wait_frames=0),
    ]
    assert get_boot_macro_expectation("ceres_ridley_appeared_to_elevator_room") == (0xDF45, 8)


def test_ceres_elevator_countdown_to_lowerledge_macro_shape():
    assert "ceres_elevator_countdown_to_lowerledge" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_elevator_countdown_to_lowerledge")

    assert steps == [NavStep(buttons=[6, 8], hold_frames=70, wait_frames=0)]
    assert get_boot_macro_expectation("ceres_elevator_countdown_to_lowerledge") == (0xDF45, 8)


def test_ceres_lowerledge_to_landing_site_macro_shape():
    assert "ceres_lowerledge_to_landing_site" in list_boot_macros()

    steps = get_boot_macro_steps("ceres_lowerledge_to_landing_site")

    assert steps == [
        NavStep(buttons=[6, 8], hold_frames=94, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=80, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=80, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=80, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=100, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=70, wait_frames=0),
        NavStep(buttons=[7, 8], hold_frames=90, wait_frames=0),
        NavStep(buttons=[6, 8], hold_frames=50, wait_frames=0),
        NavStep(buttons=[], hold_frames=0, wait_frames=2500),
    ]
    assert get_boot_macro_expectation("ceres_lowerledge_to_landing_site") == (0x91F8, 8)


def test_inspect_export_detects_smedit_layout(tmp_path: Path):
    (tmp_path / "nav_graph.json").write_text('{"nodes": [], "edges": []}')
    rooms_dir = tmp_path / "rooms"
    rooms_dir.mkdir()
    (rooms_dir / "room_91F8.json").write_text("{}")

    audit = inspect_export(tmp_path)

    assert audit.exists is True
    assert audit.layout == "smedit_export"
    assert audit.has_nav_graph is True
    assert audit.has_rooms_dir is True
    assert audit.room_file_count == 1
    assert audit.node_count == 0
    assert audit.edge_count == 0


def test_inspect_export_detects_refs_layout(tmp_path: Path):
    (tmp_path / "region").mkdir()
    (tmp_path / "connection").mkdir()

    audit = inspect_export(tmp_path)

    assert audit.exists is True
    assert audit.layout == "refs_sm_json_data"
    assert audit.has_nav_graph is False
    assert audit.has_rooms_dir is False
    assert "SMEDIT export" in (audit.error or "")
