"""Unit tests for ReturnHome thrash/recover pure helpers (no emu)."""

from __future__ import annotations

import unittest

import numpy as np

from harvest.planner.tasks.home_recover import (
    RecoverKind,
    decide_child_failure,
    decide_enter_house_failure,
    decide_exit_to_farm_failure,
    decide_nav_failure,
    enter_fail_south_recovery_actions,
    exit_to_farm_recover_actions,
)
from harvest.tasks.nav import Point


class TestExitToFarmRecoverActions(unittest.TestCase):
    def test_multi_face_mash_length_and_faces(self) -> None:
        actions = exit_to_farm_recover_actions()
        # 4 faces × (4 face + 8 mash + 4 idle) = 64
        self.assertEqual(len(actions), 64)
        self.assertTrue(all(isinstance(a, np.ndarray) for a in actions))
        # First face block is down (button index 5 in SNES layout via make_action)
        # Just ensure non-zero buttons appear in each face segment.
        for face_i in range(4):
            base = face_i * 16
            face_frames = actions[base : base + 4]
            mash_frames = actions[base + 4 : base + 12]
            idle_frames = actions[base + 12 : base + 16]
            self.assertTrue(any(int(a.sum()) > 0 for a in face_frames))
            self.assertTrue(all(int(a.sum()) > 0 for a in mash_frames))
            self.assertTrue(all(int(a.sum()) == 0 for a in idle_frames))


class TestEnterFailSouthRecovery(unittest.TestCase):
    def test_none_when_not_north_of_stand(self) -> None:
        front = Point(136, 424)
        self.assertIsNone(
            enter_fail_south_recovery_actions(Point(136, 424), front)
        )
        self.assertIsNone(
            enter_fail_south_recovery_actions(Point(136, 410), front)
        )  # y == front.y - 14 >= front.y - 16

    def test_actions_when_north_of_stand(self) -> None:
        front = Point(136, 424)
        actions = enter_fail_south_recovery_actions(Point(136, 400), front)
        self.assertIsNotNone(actions)
        assert actions is not None
        # 10 left + 40 down-B + 12 right + 24 down-B + 8 idle
        self.assertEqual(len(actions), 10 + 40 + 12 + 24 + 8)


class TestDecideExitToFarm(unittest.TestCase):
    def test_queue_mash_under_limit(self) -> None:
        d = decide_exit_to_farm_failure(
            retries=0, retry_limit=3, reason="dialogue timeout"
        )
        self.assertEqual(d.kind, RecoverKind.QUEUE_EXIT_MASH)

    def test_fail_when_exhausted(self) -> None:
        d = decide_exit_to_farm_failure(
            retries=3, retry_limit=3, reason="dialogue timeout"
        )
        self.assertEqual(d.kind, RecoverKind.FAIL_EXIT)


class TestDecideEnterHouse(unittest.TestCase):
    def test_south_when_north_of_stand(self) -> None:
        front = Point(136, 424)
        d = decide_enter_house_failure(
            pos=Point(136, 390),
            front=front,
            enter_retries=0,
            reason="enter timeout",
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.kind, RecoverKind.RETRY_ENTER_SOUTH)

    def test_restart_when_at_or_south_of_stand(self) -> None:
        front = Point(136, 424)
        d = decide_enter_house_failure(
            pos=Point(136, 424),
            front=front,
            enter_retries=1,
            reason="enter timeout",
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.kind, RecoverKind.RETRY_ENTER_RESTART)

    def test_hands_flag_from_reason(self) -> None:
        front = Point(136, 424)
        d = decide_enter_house_failure(
            pos=Point(136, 424),
            front=front,
            enter_retries=0,
            reason="hands not clear before enter",
        )
        assert d is not None
        self.assertTrue(d.hands_not_clear)

    def test_none_when_retries_exhausted(self) -> None:
        front = Point(136, 424)
        d = decide_enter_house_failure(
            pos=Point(136, 390),
            front=front,
            enter_retries=4,
            reason="enter timeout",
        )
        self.assertIsNone(d)


class TestDecideNavFailure(unittest.TestCase):
    def test_force_enter_near_door(self) -> None:
        # D12 residual (118,486) vs front (136,424)
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(118, 486),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="multi_nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.FORCE_ENTER)

    def test_force_enter_lateral_near_door_gate_b(self) -> None:
        """Gate B D5 soak: multi_nav timeout at (190,423) vs front (136,424).

        dx=54 was outside the old dx<=48 band → HARD_FAIL terminal return_home.
        """
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(190, 423),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="multi_nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.FORCE_ENTER)

    def test_south_escape_fence_latitude(self) -> None:
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(774, 521),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="multi_nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.SOUTH_ESCAPE)
        self.assertTrue(d.far_east)
        self.assertFalse(d.escape_from_drop)

    def test_south_escape_deep_south(self) -> None:
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(102, 726),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="multi_nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.SOUTH_ESCAPE)
        self.assertFalse(d.far_east)

    def test_drop_spot_south_escape(self) -> None:
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_drop_spot",
            pos=Point(102, 726),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.SOUTH_ESCAPE)
        self.assertTrue(d.escape_from_drop)

    def test_hands_full_drop_retry(self) -> None:
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(400, 500),
            front=front,
            hands_clear=False,
            drop_attempts=2,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.RETRY_DROP_THEN_NAV)

    def test_hands_full_budget_exhausted_hard_fail_start_phase(self) -> None:
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(400, 500),
            front=front,
            hands_clear=False,
            drop_attempts=10,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="nav timeout",
        )
        self.assertEqual(d.kind, RecoverKind.HARD_FAIL)
        self.assertTrue(d.clear_task)
        self.assertEqual(d.set_phase, "start")

    def test_mid_yard_renav(self) -> None:
        # South of door by >24, north of fence (~y<504), within dx 80.
        front = Point(136, 424)
        d = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(150, 470),
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="soft timeout",
        )
        # dy=46 <= 80 and dx=14 <= 48 → FORCE_ENTER wins over mid-yard.
        self.assertEqual(d.kind, RecoverKind.FORCE_ENTER)

        # Just outside near-door box but still mid-yard (dy=81).
        d2 = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(150, 505),  # y=505 is south_of_fence (FENCE_PX_Y+8=504)
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="soft timeout",
        )
        # Fence latitude → south escape, not mid-yard.
        self.assertEqual(d2.kind, RecoverKind.SOUTH_ESCAPE)

        d3 = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(180, 490),  # dy=66 > 80? 490-424=66 <= 80 → force enter
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="soft timeout",
        )
        self.assertEqual(d3.kind, RecoverKind.FORCE_ENTER)

        # Mid-yard: dy > 80, not fence, dx <= 80.
        d4 = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(150, 510),  # south of fence actually
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="soft timeout",
        )
        self.assertEqual(d4.kind, RecoverKind.SOUTH_ESCAPE)

        # True mid-yard: y between front+24 and fence-ish, dy > 80.
        # front.y+81 = 505 is fence edge. Use front.y+81 with x offset
        # but fence is at 504... so mid-yard north of fence means y < 504.
        # dy > 80 means y > 504. Contradiction for front.y=424.
        # front.y + 81 = 505 >= 504 → always fence when dy>80.
        # So mid-yard only when dy is between 25 and 80 exclusive of force box.
        # Force: dx<=48 and dy<=80. Mid-yard needs dx > 48 and dx <= 80,
        # pos.y > front.y+24, not south_of_fence.
        d5 = decide_nav_failure(
            phase="nav_house_front",
            pos=Point(200, 460),  # dx=64, dy=36; near-door fails dx; mid-yard ok
            front=front,
            hands_clear=True,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=2,
            south_escape_attempts=0,
            south_escape_limit=4,
            reason="soft timeout",
        )
        self.assertEqual(d5.kind, RecoverKind.MID_YARD_RENAV)


class TestDecideChildFailure(unittest.TestCase):
    def test_exit_phase_dispatches(self) -> None:
        d = decide_child_failure(
            phase="exit_to_farm",
            pos=Point(0, 0),
            front=Point(136, 424),
            reason="dialogue",
            hands_clear=True,
            exit_to_farm_retries=1,
            exit_to_farm_retry_limit=3,
            enter_retries=0,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
        )
        self.assertEqual(d.kind, RecoverKind.QUEUE_EXIT_MASH)

    def test_enter_exhausted_hard_fail(self) -> None:
        d = decide_child_failure(
            phase="enter_house",
            pos=Point(136, 424),
            front=Point(136, 424),
            reason="timeout",
            hands_clear=True,
            exit_to_farm_retries=0,
            exit_to_farm_retry_limit=3,
            enter_retries=4,
            drop_attempts=0,
            drop_attempt_limit=10,
            offstand_corrections=0,
            south_escape_attempts=0,
            south_escape_limit=4,
        )
        self.assertEqual(d.kind, RecoverKind.HARD_FAIL)


if __name__ == "__main__":
    unittest.main()
