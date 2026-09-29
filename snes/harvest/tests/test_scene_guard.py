from __future__ import annotations

import unittest

import numpy as np

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.scene import SceneGuard


def _ram(lock=0, text=0x1234, cursor=0):
    ram = np.zeros(0x24000, dtype=np.uint8)
    base = LIVE_RAM_WRAM_OFFSET if len(ram) > 0x20000 else 0
    ram[field_spec("tilemap").address] = 0x00
    ram[field_spec("input_lock").address] = lock
    for name, val in (("player_x", 136), ("player_y", 424), ("dialog_text_id", text)):
        a = field_spec(name).address + (base if name.startswith("player") else 0)
        ram[a], ram[a + 1] = val & 0xFF, val >> 8
    ram[field_spec("dialog_menu_cursor").address] = cursor
    return ram


class SceneGuardTests(unittest.TestCase):
    def test_normal_scene_is_left_alone(self) -> None:
        guard = SceneGuard(quiet_frames=3)
        ram = _ram(lock=1, text=0)
        self.assertTrue(all(guard.observe(ram, i) is None for i in range(20)))

    def test_unowned_dialogue_is_advanced_and_logged(self) -> None:
        guard = SceneGuard(quiet_frames=5, bound=50)
        ram = _ram(lock=0, text=0x0321)
        out = [guard.observe(ram, i) for i in range(12)]
        self.assertIsNone(out[0])
        self.assertIn(("a",), out)
        self.assertEqual(guard.log[0]["text_id"], 0x0321)

    def test_changing_scene_is_not_stolen(self) -> None:
        guard = SceneGuard(quiet_frames=5)
        for i in range(30):
            self.assertIsNone(guard.observe(_ram(lock=0, text=100 + i), i))

    def test_pressing_child_owns_the_scene(self) -> None:
        guard = SceneGuard(quiet_frames=5, bound=50)
        ram = _ram(lock=0, text=0x0368)
        out = [guard.observe(ram, i, child_pressed=True) for i in range(200)]
        self.assertTrue(all(o is None for o in out))

    def test_plain_dialogue_never_gets_b(self) -> None:
        guard = SceneGuard(quiet_frames=2, bound=100)
        ram = _ram(lock=0, text=0x0368)
        out = [guard.observe(ram, i) for i in range(100)]
        self.assertNotIn(("b",), out)

    def test_menu_gets_b_first(self) -> None:
        guard = SceneGuard(quiet_frames=2, bound=30)
        ram = _ram(lock=0, text=0x10, cursor=1)
        out = [guard.observe(ram, i) for i in range(6)]
        self.assertIn(("b",), out)

    def test_bounded_then_gives_up(self) -> None:
        guard = SceneGuard(quiet_frames=2, bound=10)
        ram = _ram(lock=0)
        out = [guard.observe(ram, i) for i in range(60)]
        self.assertIsNone(out[-1])
        self.assertLessEqual(sum(o is not None for o in out), 11)


if __name__ == "__main__":
    unittest.main()
