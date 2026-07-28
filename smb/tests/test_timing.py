"""Unit tests for SMB TAS / RTA timing contracts."""

from __future__ import annotations

from smb.timing import (
    CONTRACTS,
    NTSC_FPS,
    PUBLIC_REFERENCES,
    build_timing_block,
    contract_result,
    format_time,
    format_time_mmss,
    frames_from_time_string,
    rta_frames,
    segment_splits,
    tasvideos_frames,
    timing_from_poweron_report,
)


def test_format_time_happylee_published() -> None:
    # TASVideos #1715: 17868 frames @ NTSC → 04:57.31
    assert format_time(17_868, NTSC_FPS).startswith("4:57.3")
    assert format_time_mmss(17_868, NTSC_FPS).startswith("04:57.3")


def test_frames_from_rta_note_roundtrips() -> None:
    frames = frames_from_time_string("04:54.032", NTSC_FPS)
    assert frames == PUBLIC_REFERENCES["happylee_warps_rta_note"]["frames"]
    assert abs(frames / NTSC_FPS - (4 * 60 + 54.032)) < 1.0 / NTSC_FPS


def test_tasvideos_and_rta_frame_math() -> None:
    assert rta_frames(policy_frames_to_ending=21_731) == 21_731
    assert (
        tasvideos_frames(
            boot_frames=350,
            settle_frames=16,
            policy_frames_to_ending=21_731,
        )
        == 350 + 16 + 21_731
    )


def test_segment_splits_policy_clock() -> None:
    milestones = [
        {"exit_id": "1-1", "frame": 1974, "world": 0, "level": 1, "lives": 2},
        {"exit_id": "1-2", "frame": 4044, "world": 3, "level": 0, "lives": 2},
    ]
    rows = segment_splits(milestones, clock_offset=0, fps=NTSC_FPS)
    assert rows[0]["seg_frames"] == 1974
    assert rows[1]["seg_frames"] == 4044 - 1974
    assert rows[1]["clock_frame"] == 4044
    abs_rows = segment_splits(milestones, clock_offset=366, fps=NTSC_FPS)
    assert abs_rows[0]["clock_frame"] == 366 + 1974


def test_contract_result_delta_vs_happylee() -> None:
    ours = 21_731
    row = contract_result("rta_any_percent", frames=ours)
    assert row["contract_id"] == "rta_any_percent"
    assert row["frames"] == ours
    assert row["delta_frames"] == ours - PUBLIC_REFERENCES["happylee_warps_rta_note"]["frames"]
    assert row["delta_time"].startswith("+")


def test_build_timing_block_poweron() -> None:
    milestones = [
        {"exit_id": "1-1", "frame": 1974, "world": 0, "level": 1, "lives": 2},
        {"exit_id": "8-4", "frame": 21731, "world": 7, "level": 3, "lives": 2},
    ]
    timing = build_timing_block(
        mode="poweron",
        boot_frames=350,
        settle_frames=16,
        policy_frames_to_ending=21_731,
        milestones=milestones,
    )
    assert set(timing["contracts"]) == {
        "rta_any_percent",
        "policy_seed",
        "tasvideos_poweron",
    }
    assert timing["contracts"]["rta_any_percent"]["frames"] == 21_731
    assert timing["contracts"]["tasvideos_poweron"]["frames"] == 22_097
    assert len(timing["comparisons"]) == 2
    # Contracts registry is complete.
    assert set(CONTRACTS) >= set(timing["contracts"])


def test_timing_from_poweron_report_shape() -> None:
    report = {
        "mode": "poweron",
        "stages": {
            "boot": {"frames": 350, "settle_frames": 16},
            "continuous": {
                "policy_frames": 21_731,
                "milestones": [
                    {
                        "exit_id": "1-1",
                        "frame": 1974,
                        "world": 0,
                        "level": 1,
                        "lives": 2,
                    },
                    {
                        "exit_id": "8-4",
                        "frame": 21731,
                        "world": 7,
                        "level": 3,
                        "lives": 2,
                    },
                ],
            },
        },
    }
    timing = timing_from_poweron_report(report)
    assert timing["contracts"]["tasvideos_poweron"]["frames"] == 22_097
    rta = timing["contracts"]["rta_any_percent"]
    assert rta["public"]["published_time"] == "04:54.032"
    assert rta["delta_frames"] > 0
