"""Unit tests for reusable path segments and reactive mountain berry."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from harvest.core.ram_catalog import field_spec
from harvest.maps.map_config import (
    ROUTES,
    SEGMENTS,
    compose_routes,
    farm_coords_look_like_path,
    find_landmark,
    mountain_downhill_escape,
    path_coords_leaked,
    segment_waypoints,
    slice_route_from_position,
)
from harvest.core.game_clock import ClockTimeline, compare_frame_benches
from harvest.planner.day_phase_berry import (
    GRAPE_BAIL_HOUR_NO_SHOP,
    GRAPE_BAIL_HOUR_SHOP_DAY,
    grape_bail_hour_for_day,
    grape_day_spec,
    mountain_berry_count_for_day,
    mountain_berry_phase,
)
from harvest.planner.day_phase_catalog import MOUNTAIN_BERRY_PHASE, PHASE_SEQUENCES
from harvest.planner.day_phase_types import PhaseKind
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
from harvest.tasks.mountain_berry import (
    BERRY_NAV_SEGMENTS,
    TOWN_NAV_SEGMENTS,
    MountainBerryTask,
    face_toward_grape,
    first_remaining_segment,
    format_segment_time,
    go_to_town_waypoints,
    is_mountain_forage,
    mountain_corridor_segments,
    on_grape_pixel,
)
from harvest.tasks.mountain_grape_ship import MountainGrapeShipTask, ROUTE_NAME
from harvest.tasks.harvest_task import ADDR_SHIPPING_MONEY
from retro_harness import TaskResult, TaskStatus

from day_plan_test_helpers import make_transition_world, set_player_pos

ADDR_HELD = field_spec("held_item").address


class PathSegmentTests(unittest.TestCase):
    def test_town_and_mountain_share_farm_to_path(self) -> None:
        shared = SEGMENTS["farm_to_path"]
        self.assertEqual(shared[0].tilemap, 0x00)
        self.assertEqual(shared[0].target_px, (137, 375))
        self.assertFalse(shared[0].is_exit)
        self.assertEqual(shared[1].target_px, (136, 392))
        self.assertIn((136, 424), [wp.target_px for wp in shared])
        self.assertIn((72, 424), [wp.target_px for wp in shared])
        self.assertNotIn((72, 392), [wp.target_px for wp in shared])
        self.assertNotIn((8, 392), [wp.target_px for wp in shared])
        self.assertNotIn((136, 440), [wp.target_px for wp in shared])
        self.assertNotIn((72, 440), [wp.target_px for wp in shared])
        self.assertNotIn((120, 408), [wp.target_px for wp in shared])
        farm_exit = next(wp for wp in shared if wp.is_exit)
        self.assertEqual(farm_exit.target_px, (40, 424))
        self.assertEqual(farm_exit.exit_direction, "left")
        self.assertIsNone(farm_exit.run_direction)
        self.assertFalse(farm_exit.force_run)
        self.assertLess(farm_exit.radius, 8)
        self.assertEqual(shared[-1].tilemap, 0x0C)
        self.assertEqual(shared[-1].target_px, (132, 128))
        self.assertEqual(shared[-1].run_direction, "left")
        self.assertTrue(shared[-1].force_run)

        town = segment_waypoints(*TOWN_NAV_SEGMENTS)
        mountain = segment_waypoints("farm_to_path", "path_to_mountain")
        self.assertEqual(town[: len(shared)], shared)
        self.assertEqual(mountain[: len(shared)], shared)
        self.assertEqual(town, go_to_town_waypoints())
        self.assertEqual(town, ROUTES["farm_to_town"])
        self.assertEqual(town, ROUTES["go_to_town"])
        self.assertEqual(mountain, ROUTES["farm_to_mountain"])

    def test_path_to_town_and_mountain_diverge_at_crossroads(self) -> None:
        self.assertEqual(SEGMENTS["path_to_town"][-1].exit_direction, "left")
        self.assertEqual(SEGMENTS["path_to_mountain"][0].target_px, (132, 128))
        self.assertEqual(SEGMENTS["path_to_mountain"][-1].exit_direction, "up")
        self.assertEqual(SEGMENTS["path_to_farm"][0].target_px, (132, 128))
        self.assertEqual(SEGMENTS["path_to_farm"][-1].exit_direction, "right")
        self.assertNotEqual(
            SEGMENTS["path_to_town"][-1].target_px,
            SEGMENTS["path_to_mountain"][-1].target_px,
        )

    def test_farm_to_path_skips_farm_gate(self) -> None:
        hops = [wp.target_px for wp in SEGMENTS["farm_to_path"]]
        self.assertNotIn((232, 128), hops)
        self.assertEqual(hops[-1], (132, 128))

    def test_leaked_path_coords_start_at_crossroads(self) -> None:
        self.assertTrue(path_coords_leaked(10, 422))
        self.assertTrue(path_coords_leaked(314, 740))
        self.assertFalse(path_coords_leaked(232, 128))
        self.assertFalse(path_coords_leaked(132, 30))
        for name, leaked in (
            ("path_to_mountain", (10, 422)),
            ("path_to_farm", (314, 740)),
        ):
            sliced = slice_route_from_position(
                list(SEGMENTS[name]), leaked[0], leaked[1], tilemap=0x0C
            )
            self.assertEqual(sliced[0].target_px, (132, 128), name)
        ship = slice_route_from_position(
            list(ROUTES[ROUTE_NAME]), 314, 740, tilemap=0x0C
        )
        self.assertEqual(ship[0].target_px, (132, 128))
        self.assertEqual(ship[-1].target_px, (136, 456))

    def test_farm_path_gate_pixels_start_at_west_gate_not_north_shed(self) -> None:
        self.assertTrue(farm_coords_look_like_path(244, 118))
        self.assertTrue(farm_coords_look_like_path(232, 128))
        self.assertFalse(farm_coords_look_like_path(80, 424))
        self.assertFalse(farm_coords_look_like_path(136, 456))
        ship = slice_route_from_position(
            list(ROUTES[ROUTE_NAME]), 244, 118, tilemap=0x00
        )
        self.assertEqual(ship[0].target_px, (80, 424))
        self.assertEqual(ship[-1].target_px, (136, 456))
        self.assertFalse(farm_coords_look_like_path(*ship[0].target_px))

    def test_compose_routes_matches_named_full_route(self) -> None:
        composed = compose_routes(
            SEGMENTS["farm_to_path"],
            SEGMENTS["path_to_mountain"],
            SEGMENTS["mountain_entry_to_first_berry"],
        )
        self.assertEqual(composed, ROUTES["farm_to_first_mountain_berry"])
        self.assertIsNone(composed[-1].action_on_arrive)
        self.assertEqual(composed[-1].target_px, (326, 409))

    def test_first_berry_landmark_matches_route_stand(self) -> None:
        found = find_landmark("first_berry", tilemap_id=0x10)
        self.assertIsNotNone(found)
        _tm, landmark = found
        stand = ROUTES["mountain_entry_to_first_berry"][-1]
        self.assertEqual(landmark.tile, (20, 25))
        self.assertEqual(stand.target_px, (326, 409))
        hops = SEGMENTS["mountain_entry_to_first_berry"]
        # Cliff blocks due-north from land; recorded gap is east to x=32.
        # Spa pull-up (328, 688) is spa-only — grape inbound stays land→east.
        self.assertEqual(hops[0].target_px, (328, 728))
        self.assertEqual(hops[1].target_px, (424, 712))
        self.assertGreaterEqual(max(wp.target_px[0] for wp in hops), 500)
        self.assertEqual(hops[-1].target_px, (326, 409))

    def test_approach_only_succeeds_at_stand_without_held_grape(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainBerryTask(approach_only=True, pick_attempts=0)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("unverified", result.reason or "")

    def test_face_toward_grape_from_northwest_stand(self) -> None:
        world = make_transition_world(0x10, current_tile=(19, 24))
        set_player_pos(world.ram, 316, 399)
        self.assertIn(face_toward_grape(world.ram), {"right", "down"})
        self.assertFalse(on_grape_pixel(world.ram))
        set_player_pos(world.ram, 326, 409)
        self.assertEqual(face_toward_grape(world.ram), "down")
        self.assertTrue(on_grape_pixel(world.ram))

    def test_pick_at_stand_faces_down_not_bush(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("pick", task.phase_text)
        self.assertIn("pick attempt=1", result.reason or "")

    def test_eat_box_with_held_grape_does_not_count_as_talk(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        world.ram[ADDR_HELD] = 0x03
        world.ram[ADDR_INPUT_LOCK] = 2
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        task._picks = 1
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertNotIn("talked", result.reason or "")

    def test_mountain_dialogue_without_grape_is_talk_fail(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        world.ram[ADDR_INPUT_LOCK] = 2
        task = MountainBerryTask(approach_only=False, pick_attempts=2)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.FAILURE)
        self.assertIn("talked on mountain", result.reason or "")


class MountainBerrySelectTests(unittest.TestCase):
    def test_segment_choice_follows_live_tilemap(self) -> None:
        self.assertEqual(first_remaining_segment(0x00), "farm_to_path")
        self.assertIsNone(
            first_remaining_segment(
                0x00, ("path_to_mountain", "mountain_entry_to_first_berry")
            )
        )
        self.assertEqual(first_remaining_segment(0x0C), "path_to_mountain")
        self.assertEqual(first_remaining_segment(0x10), "mountain_entry_to_first_berry")
        self.assertIsNone(first_remaining_segment(0x15))

    def test_already_holding_grapes_on_mountain_succeeds(self) -> None:
        world = make_transition_world(0x10, current_tile=(32, 43))
        world.ram[ADDR_HELD] = 0x03
        task = MountainBerryTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("grapes", result.reason or "")

    def test_bin_start_clips_farm_to_path_off_house_north(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 136, 456)
        task = MountainBerryTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "farm_to_path")
        hops = [wp.target_px for wp in task._child.waypoints]
        self.assertNotEqual(hops[0], (137, 375))
        self.assertNotIn((232, 128), hops)

    def test_leaked_path_pose_settles_before_crossroads_nav(self) -> None:
        world = make_transition_world(0x0C, current_tile=(0, 26))
        set_player_pos(world.ram, 10, 422)
        task = MountainBerryTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "path_settle")
        self.assertIn("settle", result.reason or "")
        set_player_pos(world.ram, 232, 128)
        result = task.step(world)
        self.assertEqual(task.phase_text, "path_to_mountain")
        hops = [wp.target_px for wp in task._child.waypoints]
        self.assertEqual(hops[0], (132, 128))
        self.assertNotIn((232, 128), hops)

    def test_farm_to_path_clips_once_path_coords_settle(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 24))
        set_player_pos(world.ram, 136, 392)
        task = MountainBerryTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "farm_to_path")
        world.ram[ADDR_TILEMAP] = 0x0C
        set_player_pos(world.ram, 10, 422)
        result = task.step(world)
        self.assertEqual(task.phase_text, "path_settle")
        set_player_pos(world.ram, 232, 128)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "path_to_mountain")
        hops = [wp.target_px for wp in task._child.waypoints]
        self.assertEqual(hops[0], (132, 128))

    def test_house_start_arms_exit_to_farm(self) -> None:
        world = make_transition_world(0x15, current_tile=(8, 12))
        set_player_pos(world.ram, 136, 120)
        task = MountainBerryTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "exit_to_farm")
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)

    def test_forage_ids_are_decomp_held_items(self) -> None:
        self.assertTrue(is_mountain_forage(0x03))
        self.assertTrue(is_mountain_forage(0x08))
        self.assertFalse(is_mountain_forage(0x0D))
        self.assertFalse(is_mountain_forage(0x00))

    def test_day_plan_sequence_is_registered(self) -> None:
        phases = PHASE_SEQUENCES["mountain_berry"]
        names = [p.phase for p in phases]
        self.assertIn("EXIT_TO_FARM", names)
        self.assertIn("MOUNTAIN_BERRY", names)
        self.assertEqual(MOUNTAIN_BERRY_PHASE.kind, PhaseKind.MOUNTAIN_BERRY)
        self.assertEqual(BERRY_NAV_SEGMENTS[0], "farm_to_path")
        self.assertTrue(MOUNTAIN_BERRY_PHASE.params["ship"])

    def test_corridor_segments_ignore_pick_keep_window(self) -> None:
        samples = [
            {"frame": 600, "tilemap": 0x0C, "x": 132, "y": 30, "held_item": 0},
            {"frame": 634, "tilemap": 0x10, "x": 137, "y": 10, "held_item": 0},
            {"frame": 1650, "tilemap": 0x10, "x": 326, "y": 409, "held_item": 0},
            {"frame": 1800, "tilemap": 0x10, "x": 326, "y": 409, "held_item": 3},
            {"frame": 1810, "tilemap": 0x10, "x": 326, "y": 430, "held_item": 3},
            {"frame": 2200, "tilemap": 0x0C, "x": 314, "y": 740, "held_item": 3},
        ]
        segs = mountain_corridor_segments(samples)
        self.assertEqual(segs["mountain_entry_to_grape"]["frames"], 1016)
        self.assertEqual(segs["grape_to_mountain_exit"]["frames"], 390)
        self.assertEqual(segs["pick_keep"]["frames"], 160)
        clock = format_segment_time(1016)
        self.assertEqual(clock["seconds"], 16.933)
        self.assertEqual(clock["clock"], "00:16.93")

    def test_corridor_samples_build_hour_timeline_and_frame_delta(self) -> None:
        samples = [
            {"frame": 0, "hour": 6, "minute": 8, "tilemap": 0x15, "x": 128, "y": 200},
            {"frame": 662, "hour": 7, "minute": 12, "tilemap": 0x10, "x": 137, "y": 10},
            {"frame": 1650, "hour": 8, "minute": 0, "tilemap": 0x10, "x": 326, "y": 409},
            {"frame": 2373, "hour": 9, "minute": 4, "tilemap": 0x0C, "x": 314, "y": 740},
            {"frame": 3224, "hour": 10, "minute": 11, "tilemap": 0x00, "x": 135, "y": 456},
        ]
        timeline = ClockTimeline.from_samples(samples)
        hours = [mark.clock.hour for mark in timeline.hour_marks()]
        self.assertEqual(hours, [6, 7, 8, 9, 10])
        self.assertEqual(timeline.end.frame, 3224)
        faster = compare_frame_benches(3224, 3154)
        self.assertEqual(faster["delta_frames"], -70)
        self.assertTrue(faster["faster"])

    def test_grape_return_route_drops_south_cliff_and_ends_at_real_bin(self) -> None:
        route = ROUTES[ROUTE_NAME]
        self.assertEqual(route[0].target_px, (326, 409))
        cliff = SEGMENTS["first_berry_to_mountain_exit"]
        self.assertEqual(cliff[0].target_px, (326, 409))
        self.assertEqual(cliff[1].run_direction, "down")
        self.assertTrue(cliff[1].force_run)
        self.assertEqual(cliff[1].target_px, (328, 568))
        mountain_exit = next(wp for wp in route if wp.is_exit and wp.tilemap == 0x10)
        self.assertEqual(mountain_exit.exit_direction, "down")
        path_exit = next(wp for wp in route if wp.is_exit and wp.tilemap == 0x0C)
        self.assertEqual(path_exit.exit_direction, "right")
        self.assertEqual(route[-1].tilemap, 0x00)
        self.assertEqual(route[-1].target_px, (8 * 16 + 8, 28 * 16 + 8))
        self.assertEqual(route[-1].action_on_arrive, "press_a")
        self.assertEqual(route[-1].action_face, "down")
        # Cliff return is the short south drop, not the 16-hop inbound reverse.
        self.assertLess(len(cliff), len(SEGMENTS["mountain_entry_to_first_berry"]))
        inbound = SEGMENTS["mountain_entry_to_first_berry"]
        forced = {
            wp.target_px: wp.run_direction
            for wp in inbound
            if wp.force_run
        }
        self.assertEqual(forced[(520, 712)], "right")
        self.assertEqual(forced[(520, 632)], "up")
        self.assertEqual(forced[(328, 568)], "left")
        self.assertEqual(forced[(240, 488)], "left")
        self.assertEqual(forced[(312, 360)], "right")
        # The outbound cliff's (520,712) is a plain waypoint: forcing it down
        # pinned the grape *return* (which slices this same list) at ~(505,633)
        # into the carpenter terrace wall (run10). The x=20 drop above stays forced.
        land = next(wp for wp in cliff if wp.target_px == (520, 712))
        self.assertIsNone(land.run_direction)
        self.assertFalse(land.force_run)

    def test_downhill_escape_from_carpenter_skips_terrace(self) -> None:
        hops = mountain_downhill_escape(474, 630, tilemap=0x10)
        self.assertEqual(hops[0].target_px, (520, 712))
        # 960585b3's force-run on this shared outbound waypoint pinned the
        # grape return at ~(505,633); it is a plain waypoint now.
        self.assertFalse(hops[0].force_run)
        self.assertNotIn((520, 632), [wp.target_px for wp in hops])
        self.assertEqual(hops[-1].target_px, (312, 744))

    def test_downhill_escape_from_grape_stand_keeps_cliff_drop(self) -> None:
        hops = mountain_downhill_escape(326, 409, tilemap=0x10)
        self.assertEqual(hops[0].target_px, (328, 568))
        self.assertTrue(hops[0].force_run)  # x=20 grape cliff drop stays forced

    def test_grape_ship_postcondition_requires_empty_hands_and_shipping_delta(self) -> None:
        world = make_transition_world(0x00, current_tile=(61, 60))
        set_player_pos(world.ram, 61 * 16 + 8, 60 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        task = MountainGrapeShipTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "return_to_bin")
        self.assertFalse(task._child.allow_opportunistic_clear)

        task._phase = "verify"
        task._child = None
        world.ram[ADDR_HELD] = 0
        world.ram[ADDR_SHIPPING_MONEY] = 6
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("0->60", result.reason or "")

    def _grape_at_bin(self, world) -> None:
        world.ram[ADDR_HELD] = 0
        world.ram[ADDR_SHIPPING_MONEY] = int(world.ram[ADDR_SHIPPING_MONEY]) + 6

    def test_target_count_two_loops_back_for_a_second_grape(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)

        # First grape reaches the bin: not done yet — forage again.
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.shipped_count, 1)
        self.assertEqual(task.phase_text, "pick")
        self.assertIn("returning for next grape", result.reason or "")

        # Second grape reaches the bin: run is SUCCESS with two shipped.
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.shipped_count, 2)

    def test_second_grape_failure_is_best_effort_after_one_shipped(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        task.step(world)  # ship #1, loop back to pick
        self.assertEqual(task.shipped_count, 1)

        # The second forage nav dies — one grape already shipped, so SUCCESS.
        task._step_count = task.timeout + 1
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.shipped_count, 1)
        self.assertIn("stopped early", result.reason or "")

    def test_unknown_tilemap_settles_instead_of_path_to_mountain(self) -> None:
        world = make_transition_world(0x0C, current_tile=(0, 26))
        set_player_pos(world.ram, 10, 422)
        task = MountainBerryTask()
        task.reset(world)
        result = task.step(world)
        self.assertEqual(task.phase_text, "path_settle")
        world.ram[ADDR_TILEMAP] = 0x57
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "path_settle")
        self.assertIn("tilemap settle", result.reason or "")
        world.ram[ADDR_TILEMAP] = 0x0C
        set_player_pos(world.ram, 232, 128)
        result = task.step(world)
        self.assertEqual(task.phase_text, "path_to_mountain")

    def test_return_pin_retries_downhill_escape(self) -> None:
        world = make_transition_world(0x10, current_tile=(29, 39))
        set_player_pos(world.ram, 474, 630)
        world.ram[ADDR_HELD] = 0x03
        task = MountainGrapeShipTask()
        task.reset(world)
        self.assertEqual(task.phase_text, "return_to_bin")

        class FailNav:
            def step(self, _world):
                return TaskResult(
                    status=TaskStatus.FAILURE,
                    reason="soft_solid pin held=0x03 pos=(474,630) stasis=23",
                )

        task._child = FailNav()
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "return_to_bin")
        self.assertIn("downhill retry", result.reason or "")
        hops = [wp.target_px for wp in task._child.waypoints]
        self.assertEqual(hops[0], (520, 712))
        self.assertNotIn((520, 632), hops[:3])

    def test_second_grape_bails_when_shop_hour_hits(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        world.ram[field_spec("hour").address] = 12
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.shipped_count, 1)
        self.assertIn("shop window", result.reason or "")

    def test_second_grape_continues_when_bail_hour_is_raised(self) -> None:
        """A raised shop_bail_hour must actually reach the post-ship check.

        The step() guard used to be a hard-coded ``hour >= 12`` that shadowed
        ``shop_bail_hour``, so the field was dead and the second grape was
        abandoned as soon as the first landed (run11 D3-D7).
        """
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        world.ram[field_spec("hour").address] = 11
        task = MountainGrapeShipTask(target_count=2, shop_bail_hour=12)
        task.reset(world)
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.shipped_count, 1)
        self.assertIn("returning for next grape", result.reason or "")

    def test_second_loop_refused_when_it_cannot_land_before_the_deadline(self) -> None:
        """Pre-flight gate: never start a loop that ends past hard_return_hour."""
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        world.ram[field_spec("hour").address] = 13
        task = MountainGrapeShipTask(
            target_count=2, shop_bail_hour=15, loop_hours=4, hard_return_hour=16
        )
        task.reset(world)
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.shipped_count, 1)
        self.assertIn("would land after", result.reason or "")

    def test_late_hour_does_not_abandon_a_loop_in_flight_off_farm(self) -> None:
        """The mid-flight guard must not strand the farmer on mountain 0x10.

        grapefix_d3_d9 D3: the run reported SUCCESS from mountain 0x10, so
        NAV_FARM_EXIT then failed its map lock and the day lost both the seed
        purchase and CROP_ESTABLISH.
        """
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        world.ram[ADDR_HELD] = 0x03
        world.ram[field_spec("hour").address] = 17
        task = MountainGrapeShipTask(target_count=2, hard_return_hour=16)
        task.reset(world)
        task._shipped = 1
        result = task.step(world)
        self.assertNotEqual(
            result.status,
            TaskStatus.SUCCESS,
            "must not report success while still on the mountain",
        )

    def test_late_hour_stops_between_loops_when_back_on_the_farm(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[field_spec("hour").address] = 17
        task = MountainGrapeShipTask(target_count=2, hard_return_hour=16)
        task.reset(world)
        task._shipped = 1
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("past return deadline", result.reason or "")

    def test_failed_second_pick_walks_home_before_reporting_success(self) -> None:
        """No grape at the stand on loop 2 must not end the run on 0x10.

        This is the common case, not the rare one: the second loop revisits
        the same ``first_berry`` landmark, which has no same-day respawn.
        """
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._shipped = 1
        task._phase = "pick"

        class _FailingPick:
            def reset(self, _world) -> None:
                pass

            def step(self, _world) -> TaskResult:
                return TaskResult(status=TaskStatus.FAILURE, reason="no forage")

        task._child = _FailingPick()
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.phase_text, "walk_back")
        self.assertIn("walking back", result.reason or "")

    def test_timeout_off_farm_walks_home_instead_of_succeeding(self) -> None:
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._shipped = 1
        task._step_count = task.timeout
        result = task.step(world)
        self.assertNotEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.phase_text, "walk_back")

    def test_walk_back_arrival_off_farm_does_not_report_success(self) -> None:
        """MultiMapNav can report arrived while still on 0x10; do not trust it."""
        world = make_transition_world(0x10, current_tile=(20, 25))
        set_player_pos(world.ram, 326, 409)
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._shipped = 1
        task._phase = "walk_back"
        task._bail_reason = "pick: no forage"

        class _FakeArrived:
            def step(self, _world) -> TaskResult:
                return TaskResult(status=TaskStatus.SUCCESS, reason="arrived")

        task._child = _FakeArrived()
        result = task.step(world)
        self.assertNotEqual(
            result.status,
            TaskStatus.SUCCESS,
            "must not report success while still on the mountain",
        )

    def test_default_bail_hour_still_stops_the_second_grape_at_ten(self) -> None:
        world = make_transition_world(0x00, current_tile=(8, 28))
        set_player_pos(world.ram, 8 * 16 + 8, 28 * 16 + 8)
        world.ram[ADDR_HELD] = 0x03
        world.ram[field_spec("hour").address] = 10
        task = MountainGrapeShipTask(target_count=2)
        task.reset(world)
        task._phase = "verify"
        task._child = None
        self._grape_at_bin(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertEqual(task.shipped_count, 1)


class GrapeBailHourPolicyTests(unittest.TestCase):
    def test_d2_keeps_the_conservative_ten(self) -> None:
        spec = grape_day_spec(2)
        self.assertEqual(spec.count, 1)
        self.assertEqual(spec.bail_hour, 10)
        self.assertEqual(grape_bail_hour_for_day(2), 10)

    def test_no_shop_day_allows_a_later_second_loop(self) -> None:
        spec = grape_day_spec(7)
        self.assertEqual(spec.count, 2)
        self.assertEqual(spec.bail_hour, GRAPE_BAIL_HOUR_NO_SHOP)

    def test_restock_day_still_allows_the_second_grape(self) -> None:
        """2 grapes then shop is ROM-proven (13:12 / 16:08). Bail must be
        after the ~10:00 first-grape landing so loop 2 actually starts.
        grapefix_d3_d9 used bail 9 on restock days and never started it.
        """
        spec = grape_day_spec(3)
        earliest_landing = 10
        self.assertEqual(spec.count, 2)
        self.assertEqual(spec.bail_hour, GRAPE_BAIL_HOUR_NO_SHOP)
        self.assertGreater(spec.bail_hour, earliest_landing)
        self.assertLessEqual(spec.bail_hour + 4, 16)

    def test_harvest_day_does_not_force_a_second_loop(self) -> None:
        spec = grape_day_spec(9, has_harvest=True)
        self.assertEqual(spec.count, 1)
        self.assertEqual(spec.bail_hour, GRAPE_BAIL_HOUR_SHOP_DAY)
        self.assertEqual(
            grape_bail_hour_for_day(9, has_harvest=True),
            GRAPE_BAIL_HOUR_SHOP_DAY,
        )
        self.assertEqual(mountain_berry_count_for_day(9, has_harvest=True), 1)
        self.assertEqual(mountain_berry_count_for_day(9, has_harvest=False), 2)

    def test_phase_spec_carries_the_bail_hour_to_the_task(self) -> None:
        spec = mountain_berry_phase(count=2, shop_bail_hour=13)
        self.assertEqual(spec.params["shop_bail_hour"], 13)
        self.assertEqual(spec.params["count"], 2)

    def test_phase_spec_omits_the_bail_hour_when_unset(self) -> None:
        spec = mountain_berry_phase(count=1)
        self.assertNotIn("shop_bail_hour", spec.params)


if __name__ == "__main__":
    unittest.main()
