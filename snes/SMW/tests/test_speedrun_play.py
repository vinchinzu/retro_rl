"""Unit coverage for the SMW fresh-game speedrun recorder helpers."""

from __future__ import annotations

import json

from SMW.speedrun_play import SpeedrunRecorder, normalize_buttons, read_smw_ram_values


def test_normalize_buttons_pads_truncates_and_bools() -> None:
    assert normalize_buttons([1, 0, 2], size=5) == [1, 0, 1, 0, 0]
    assert normalize_buttons([1, 1, 1, 1], size=2) == [1, 1]
    assert normalize_buttons(None, size=3) == [0, 0, 0]


def test_read_smw_ram_values_decodes_unsigned_and_signed_fields() -> None:
    ram = bytearray(0x2000)
    ram[0x0100] = 0x14
    ram[0x13BF] = 0x29
    ram[0x001A] = 0x34
    ram[0x001B] = 0x12
    ram[0x00D1] = 0x78
    ram[0x00D2] = 0x56
    ram[0x007A] = 0xFE
    ram[0x007B] = 0xFF
    ram[0x0DBE] = 0xFF

    values = read_smw_ram_values(ram)

    assert values["game_mode"] == 0x14
    assert values["translevel"] == 0x29
    assert values["camera_x"] == 0x1234
    assert values["player_x"] == 0x5678
    assert values["player_x_speed"] == -2
    assert values["lives"] == -1


def test_recorder_writes_branch_frames_and_summary(tmp_path) -> None:
    recorder = SpeedrunRecorder(tmp_path, session_id="test_session", trace_every=1)

    recorder.start_branch("initial", state_name="NONE")
    recorder.record_frame(
        action_idx=1,
        raw=[0, 1] + [0] * 10,
        raw_pre=[0, 1] + [0] * 10,
        ram={"game_mode": 0x14, "translevel": 0x29},
    )
    recorder.finish_branch("session_end")

    branch_path = tmp_path / "branches" / "branch_001.json"
    payload = json.loads(branch_path.read_text(encoding="utf-8"))
    frames = (tmp_path / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert payload["actions"] == [1]
    assert payload["raw_buttons"][0][1] == 1
    assert payload["metadata"]["state_name"] == "NONE"
    assert json.loads(frames[0])["ram"]["translevel"] == 0x29
    assert recorder.branch_summaries[0]["frames"] == 1
