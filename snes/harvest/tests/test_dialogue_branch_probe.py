from __future__ import annotations

import unittest

from utils.dialogue_branch_probe import build_branch_frames, parse_override_spec


class DialogueBranchProbeTests(unittest.TestCase):
    def test_parse_override_spec_accepts_none_buttons(self) -> None:
        override = parse_override_spec("flattering@10-12=none")
        self.assertEqual(override.branch, "flattering")
        self.assertEqual((override.start_frame, override.end_frame), (10, 12))
        self.assertEqual(override.buttons, ())

    def test_parse_override_spec_accepts_multiple_buttons(self) -> None:
        override = parse_override_spec("recorded@20-21=Left,A")
        self.assertEqual(override.buttons, ("Left", "A"))

    def test_build_branch_frames_rewrites_target_window(self) -> None:
        frames = [[0] * 12 for _ in range(15)]
        override = parse_override_spec("alt@11-12=Right,A")
        updated = build_branch_frames(
            all_frames=frames,
            anchor_frame=10,
            end_frame=15,
            overrides=[override],
        )
        self.assertEqual(updated[0], [0] * 12)
        self.assertEqual(updated[1][7], 1)
        self.assertEqual(updated[1][8], 1)
        self.assertEqual(updated[2][7], 1)
        self.assertEqual(updated[2][8], 1)


if __name__ == "__main__":
    unittest.main()
