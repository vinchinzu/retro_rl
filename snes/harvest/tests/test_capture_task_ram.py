from __future__ import annotations

import unittest

import numpy as np

from utils.capture_task_ram import (
    coalesce_action_runs,
    coalesce_frame_windows,
    parse_address,
    parse_range,
    parse_watch_field_args,
    parse_watch_section_args,
    watch_value,
    state_addr_to_env_addr,
    summarize_watch,
)
from utils.capture_harvest_mode_ram import derive_harvest_metrics


class CaptureTaskRamHelperTests(unittest.TestCase):
    def test_parse_address_accepts_hex_and_decimal(self) -> None:
        self.assertEqual(parse_address("0x019A"), 0x019A)
        self.assertEqual(parse_address("410"), 410)

    def test_parse_range_requires_ascending_bounds(self) -> None:
        self.assertEqual(parse_range("0x10:0x1f"), (0x10, 0x1F))
        with self.assertRaises(ValueError):
            parse_range("0x20:0x1f")

    def test_state_addr_to_env_addr_offsets_high_wram_only(self) -> None:
        self.assertEqual(state_addr_to_env_addr(0x11F27), 0x15F27)
        self.assertEqual(state_addr_to_env_addr(0x098C), 0x098C)

    def test_parse_watch_field_args_resolves_harvest_state_keys(self) -> None:
        watches = parse_watch_field_args(["eve_hearts"], watches={})
        self.assertEqual(watches, {0x15F27: "eve_hearts"})

    def test_watch_value_decodes_named_u16_scalar_fields(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        addr = state_addr_to_env_addr(0x11F27)
        ram[addr] = 0x34
        ram[addr + 1] = 0x12

        self.assertEqual(watch_value(ram, addr, "eve_hearts"), 0x1234)

    def test_watch_value_scales_money_fields_to_gold(self) -> None:
        ram = np.zeros(0x20000, dtype=np.uint8)
        addr = state_addr_to_env_addr(0x11F07)
        ram[addr] = 192
        ram[addr + 1] = 0
        ram[addr + 2] = 0

        self.assertEqual(watch_value(ram, addr, "shipping_money"), 1920)

    def test_parse_watch_section_args_adds_all_romance_fields(self) -> None:
        watches = parse_watch_section_args(["Romance"], watches={})
        self.assertEqual(
            {label for label in watches.values()},
            {"maria_hearts", "ann_hearts", "nina_hearts", "ellen_hearts", "eve_hearts"},
        )

    def test_coalesce_frame_windows_groups_adjacent_frames(self) -> None:
        self.assertEqual(
            coalesce_frame_windows([3, 4, 5, 9, 11, 12]),
            [
                {"start": 3, "end": 5, "length": 3},
                {"start": 9, "end": 9, "length": 1},
                {"start": 11, "end": 12, "length": 2},
            ],
        )

    def test_coalesce_action_runs_splits_on_button_changes_and_gaps(self) -> None:
        frames = [
            [0] * 12,
            [0, 0, 0, 0, 1] + [0] * 7,
            [0, 0, 0, 0, 1] + [0] * 7,
            [0, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 3,
            [0] * 12,
            [0, 0, 0, 0, 0, 0, 0, 0, 1] + [0] * 3,
        ]
        self.assertEqual(
            coalesce_action_runs(frames),
            [
                {"start": 1, "end": 2, "length": 2, "buttons": ["Up"]},
                {"start": 3, "end": 3, "length": 1, "buttons": ["Up", "A"]},
                {"start": 5, "end": 5, "length": 1, "buttons": ["A"]},
            ],
        )

    def test_coalesce_action_runs_can_offset_task_frames(self) -> None:
        frames = [
            [0] * 12,
            [0, 0, 0, 0, 1] + [0] * 7,
            [0, 0, 0, 0, 1] + [0] * 7,
        ]
        self.assertEqual(
            coalesce_action_runs(frames, frame_offset=10),
            [{"start": 11, "end": 12, "length": 2, "buttons": ["Up"]}],
        )

    def test_summarize_watch_uses_task_frame_offset(self) -> None:
        row = summarize_watch(
            addr=0x15F27,
            label="eve_hearts",
            base_value=0,
            final_value=2,
            values=[0, 0, 2, 2],
            frame_offset=7429,
        )
        self.assertEqual(
            row["change_windows"],
            [{"start": 7431, "end": 7432, "length": 2}],
        )

    def test_derive_harvest_metrics_estimates_deposit_count(self) -> None:
        summary = {
            "watch_summary": [
                {
                    "label": "shipping_money",
                    "base": 0,
                    "final": 1920,
                    "unique_values": [0, 80, 160, 240, 320, 1920],
                }
            ]
        }

        metrics = derive_harvest_metrics(summary)

        self.assertEqual(metrics["shipping_money_delta"], 1920)
        self.assertEqual(metrics["estimated_shipping_unit"], 80)
        self.assertEqual(metrics["estimated_deposit_count"], 24)


if __name__ == "__main__":
    unittest.main()
