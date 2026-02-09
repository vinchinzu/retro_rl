import unittest

from donkey_kong_country.autosplit import LevelStartDetector, read_level_timer_frames


class TestAutoSplit(unittest.TestCase):
    def test_read_level_timer_frames(self):
        ram = [0] * 0x50
        # frames word at 0x46 -> 120 frames, minutes word at 0x48 -> 2 minutes
        ram[0x46] = 120 & 0xFF
        ram[0x47] = (120 >> 8) & 0xFF
        ram[0x48] = 2
        ram[0x49] = 0
        total = read_level_timer_frames(ram, frames_offset=0x46, minutes_offset=0x48)
        self.assertEqual(total, 2 * 60 * 60 + 120)

    def test_level_start_detector(self):
        det = LevelStartDetector(min_moving_frames=2)
        frames = [0, 0, 1, 2, 3]
        moved = [False, True, False, False, False]
        starts = [det.update(f, m) for f, m in zip(frames, moved)]
        self.assertEqual(starts, [False, False, False, True, True])

    def test_level_start_detector_reset(self):
        det = LevelStartDetector(min_moving_frames=2)
        for value in (0, 1, 2):
            det.update(value, True)
        det.reset()
        starts = [det.update(v, True) for v in (0, 1, 2)]
        self.assertEqual(starts, [False, False, True])


if __name__ == "__main__":
    unittest.main()
