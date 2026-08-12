"""Unit tests for pure cow geometry helpers (no ROM)."""

from __future__ import annotations

import unittest

from harvest.tasks.cow_geometry import (
    COW_FEED_SPOTS,
    COW_TALK_FACE,
    COW_TALK_STAND,
    body_side_stand_candidates,
    count_fed_trough_flags,
    cow_body_tile,
    cow_interact_pixel,
    cow_push_escape_tile,
    face_for_cow_at_stand,
    facing_tile,
    feed_route_for_spot,
    fodder_route_from,
    geometric_fallback_stands,
    is_adjacent_to_cow_tile,
    left_cow_lane_x,
    next_unfed_spot,
    preferred_cow_stands,
    talk_route_to,
)
from harvest.tasks.cow_care import (
    left_feed_spot_action,
    left_trough_return_action,
    run_to_pixel_axis,
)


class CowGeometryTests(unittest.TestCase):
    def test_facing_tile_defaults_left(self) -> None:
        self.assertEqual(facing_tile((10, 17), "left"), (9, 17))
        self.assertEqual(facing_tile((10, 17), "up"), (10, 16))
        self.assertEqual(facing_tile((10, 17), "unknown"), (9, 17))

    def test_adjacency_head_and_body(self) -> None:
        cow = (10, 16)
        self.assertTrue(is_adjacent_to_cow_tile((11, 16), "left", cow))
        self.assertTrue(is_adjacent_to_cow_tile((11, 17), "left", cow))  # body
        self.assertFalse(is_adjacent_to_cow_tile((11, 17), "down", cow))

    def test_preferred_stands_wall_side_order(self) -> None:
        stands = preferred_cow_stands(3, 12)
        self.assertEqual(stands[0], ((4, 13), "left"))
        self.assertEqual(stands[1], ((4, 12), "left"))

    def test_preferred_stands_right_aisle(self) -> None:
        stands = preferred_cow_stands(13, 10)
        self.assertEqual(stands[0], ((12, 10), "right"))
        self.assertIn(((14, 10), "left"), stands)

    def test_body_side_candidates_center(self) -> None:
        self.assertEqual(
            body_side_stand_candidates(11, 14),
            [
                ((12, 15), "left"),
                ((10, 15), "right"),
                ((12, 14), "left"),
                ((10, 14), "right"),
            ],
        )

    def test_geometric_fallback_right_cow(self) -> None:
        out = geometric_fallback_stands(
            14,
            9,
            set(),
            current=(11, 21),
            current_face="left",
        )
        self.assertEqual(out[0], ((13, 9), "right"))

    def test_feed_route_and_next_spot(self) -> None:
        spot = COW_FEED_SPOTS[0]
        self.assertEqual(feed_route_for_spot(spot), ((9, 11), spot.stand))
        flags = COW_FEED_SPOTS[0].flag | COW_FEED_SPOTS[1].flag
        self.assertEqual(next_unfed_spot(flags, goal=4), COW_FEED_SPOTS[2])
        self.assertEqual(count_fed_trough_flags(flags, goal=4), 2)

    def test_fodder_and_talk_routes(self) -> None:
        self.assertEqual(talk_route_to((10, 17))[0], (11, 21))
        route = fodder_route_from((10, 15))
        self.assertEqual(route[0], (11, 15))

    def test_cow_interact_pixel_left_clamp(self) -> None:
        # Left/center cows clamp interact x; right-side cows keep offset.
        px = cow_interact_pixel((100, 200), "left", tool=False, cow_tile=(8, 12))
        self.assertEqual(px, (113, 200))  # min(100+13, 163)
        right = cow_interact_pixel((200, 200), "left", tool=False, cow_tile=(12, 12))
        self.assertEqual(right, (213, 200))

    def test_face_for_cow_at_stand(self) -> None:
        self.assertEqual(face_for_cow_at_stand((9, 16), (10, 16)), "right")
        self.assertEqual(
            face_for_cow_at_stand(
                COW_TALK_STAND,
                (20, 20),
                talk_stand=COW_TALK_STAND,
                talk_face="up",
            ),
            "up",
        )
        self.assertEqual(face_for_cow_at_stand((5, 5), None), COW_TALK_FACE)

    def test_push_escape_and_body(self) -> None:
        cow = (10, 16)
        self.assertEqual(cow_body_tile(cow), (10, 17))
        self.assertEqual(cow_push_escape_tile(cow, (9, 16), "right"), (11, 16))
        self.assertIsNone(cow_push_escape_tile(cow, (9, 16), "left"))

    def test_left_cow_lane_x(self) -> None:
        self.assertEqual(left_cow_lane_x(200), 38)
        self.assertEqual(left_cow_lane_x(320), 55)


class CowCarePureTests(unittest.TestCase):
    def test_run_to_pixel_axis_arrives(self) -> None:
        self.assertIsNone(run_to_pixel_axis((10, 10), (11, 10), tolerance=2))
        action = run_to_pixel_axis((10, 10), (40, 10), x_first=True)
        self.assertIsNotNone(action)

    def test_left_feed_spot_on_target(self) -> None:
        spot = COW_FEED_SPOTS[0]
        self.assertIsNone(
            left_feed_spot_action(spot, spot.interact_px[0], spot.interact_px[1])
        )

    def test_left_trough_return_outside_band(self) -> None:
        self.assertIsNone(left_trough_return_action(200, 184))


if __name__ == "__main__":
    unittest.main()
