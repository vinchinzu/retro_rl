"""Unit tests for named pond corridor refill selection + crop completion."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from harvest.maps.map_config import (
    FARM_MAIN_POND_STANDS,
    FARM_POND_ACCESS_STAGING_TILES,
    FARM_POND_REFILL_CORRIDOR,
    farm_pond_refill_primary_stand,
    player_in_west_plant_pocket,
)
from harvest.tasks.crop_planter import CropWaterTask
from harvest.tasks.farm_clearer import ADDR_INPUT_LOCK, ADDR_MAP, ADDR_TILEMAP, ADDR_TOOL, ADDR_X, ADDR_Y, MAP_WIDTH
from harvest.tasks.skills import farm_nav_to_pond_refill_skill, farm_pond_refill_face
from harvest.tasks.water_refill import (
    corridor_needs_fence_open,
    crop_completion_status,
    is_no_work_reason,
    select_main_pond_refill,
    select_staging_stand,
)
from retro_harness import TaskStatus


def _blank_ram() -> np.ndarray:
    return np.zeros(ADDR_MAP + MAP_WIDTH * MAP_WIDTH, dtype=np.uint8)


def _set_tile(ram: np.ndarray, tx: int, ty: int, tile_id: int) -> None:
    ram[ADDR_MAP + ty * MAP_WIDTH + tx] = tile_id


def _set_player_tile(ram: np.ndarray, tile: tuple[int, int]) -> None:
    px = tile[0] * 16 + 8
    py = tile[1] * 16 + 8
    ram[ADDR_X] = px & 0xFF
    ram[ADDR_X + 1] = (px >> 8) & 0xFF
    ram[ADDR_Y] = py & 0xFF
    ram[ADDR_Y + 1] = (py >> 8) & 0xFF


class PondCorridorConfigTests(unittest.TestCase):
    def test_primary_stand_is_south_lip(self) -> None:
        stand, face = farm_pond_refill_primary_stand()
        self.assertEqual(stand, (32, 34))
        self.assertEqual(face, "up")
        self.assertEqual(FARM_MAIN_POND_STANDS[0], (stand, face))

    def test_corridor_steps_named(self) -> None:
        self.assertIn("stage_west_of_fence", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("open_fence_row_y31", FARM_POND_REFILL_CORRIDOR)
        self.assertIn("fill_at_main_pond", FARM_POND_REFILL_CORRIDOR)

    def test_west_pocket_predicate(self) -> None:
        self.assertTrue(player_in_west_plant_pocket((13, 27)))
        self.assertTrue(player_in_west_plant_pocket((12, 29)))
        self.assertFalse(player_in_west_plant_pocket((32, 34)))
        self.assertFalse(player_in_west_plant_pocket((10, 40)))

    def test_pond_nav_skill_targets_primary_stand(self) -> None:
        skill = farm_nav_to_pond_refill_skill()
        self.assertEqual(skill.target_px, (32 * 16 + 8, 34 * 16 + 8))
        self.assertEqual(farm_pond_refill_face(), "up")


class RefillSelectionTests(unittest.TestCase):
    def test_select_main_pond_prefers_pathable_stand(self) -> None:
        def find_path(start, goal):
            if goal == (33, 30):
                return [start, goal]
            return None

        hit = select_main_pond_refill((20, 25), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (33, 30))
        self.assertEqual(hit.face, "down")
        self.assertEqual(hit.source, "main_pond_corridor")

    def test_select_main_pond_skips_bad_stands(self) -> None:
        def find_path(start, goal):
            return [start, goal]

        hit = select_main_pond_refill(
            (20, 25),
            find_path,
            bad_stands={(32, 34), (33, 34)},
        )
        self.assertIsNotNone(hit)
        assert hit is not None
        # Nearest remaining north-lip stand from (20,25) among corridor list.
        self.assertIn(hit.stand, {(32, 30), (33, 30), (34, 30)})
        self.assertNotIn(hit.stand, {(32, 34), (33, 34)})

    def test_select_main_pond_prefers_nearest_pathable(self) -> None:
        """North of pond: north lip must beat far south lip when both pathable."""

        def find_path(start, goal):
            return [start, goal]

        hit = select_main_pond_refill((33, 28), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (33, 30))
        self.assertEqual(hit.face, "down")

    def test_select_staging_from_west_pocket(self) -> None:
        def find_path(start, goal):
            if goal == (12, 29):
                return [start, goal]
            return None

        hit = select_staging_stand((13, 27), find_path)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.stand, (12, 29))
        self.assertEqual(hit.source, "staging")

    def test_corridor_needs_fence_when_pond_blocked(self) -> None:
        def no_path(start, goal):
            return None

        self.assertTrue(
            corridor_needs_fence_open(
                (13, 27),
                no_path,
                blocking_fences=[(15, 31), (16, 31)],
            )
        )
        self.assertFalse(
            corridor_needs_fence_open(
                (13, 27),
                no_path,
                blocking_fences=[],
            )
        )

    def test_start_refill_commits_main_pond_when_pathable(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        # Open farm dirt so pathfinder can walk west pocket → pond.
        for ty in range(20, 40):
            for tx in range(10, 40):
                _set_tile(ram, tx, ty, 0x01)
        # F0 pond water cells that stands face (south lip face up → y=33).
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        _set_player_tile(ram, (20, 28))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        # Seed a water step so refill has remaining work.
        task._water_steps = [((12, 25), (12, 26), "up")]
        task._water_index = 0
        task._plot_phase = "water"
        task._plots = [(12, 25)]
        task._plot_index = 0

        task._start_refill(ram)

        self.assertEqual(task._plot_phase, "refill")
        self.assertIn(task._refill_pond_tile, {s for s, _ in FARM_MAIN_POND_STANDS})
        self.assertEqual(task._state, "navigate")
        # Stand must face preferred fill water on the live map.
        from harvest.tasks.crop_planter import edge_water_tile_id, REFILL_PREFERRED_WATER_TILES

        wid = edge_water_tile_id(ram, task._refill_pond_tile, task._refill_pond_face)
        self.assertIn(wid, REFILL_PREFERRED_WATER_TILES)


class CropCompletionTests(unittest.TestCase):
    def test_no_work_reason_helper(self) -> None:
        self.assertTrue(is_no_work_reason("no_work: water-only; no dry crop tiles"))
        self.assertTrue(is_no_work_reason("no_work"))
        self.assertFalse(is_no_work_reason("planted=1 watered=3"))
        self.assertFalse(is_no_work_reason(None))

    def test_water_mode_fails_when_dry_crops_unwatered(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=3,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "failure")
        self.assertIn("dry_crops=3", reason)

    def test_water_mode_fails_on_refill_exhausted(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=2,
            refill_exhausted=True,
            had_seed_stock=False,
        )
        self.assertEqual(status, "failure")
        self.assertIn("refill exhausted", reason)

    def test_water_mode_no_work_when_nothing_dry(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=0,
            dry_at_start=0,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "no_work")
        self.assertTrue(is_no_work_reason(reason))

    def test_water_mode_success_when_watered(self) -> None:
        status, reason = crop_completion_status(
            work_mode="water",
            planted=0,
            watered=3,
            dry_at_start=3,
            refill_exhausted=False,
            had_seed_stock=False,
        )
        self.assertEqual(status, "success")
        self.assertIn("watered=3", reason)

    def test_establish_fails_with_seed_but_no_plant(self) -> None:
        status, reason = crop_completion_status(
            work_mode="establish",
            planted=0,
            watered=0,
            dry_at_start=0,
            refill_exhausted=False,
            had_seed_stock=True,
        )
        self.assertEqual(status, "failure")
        self.assertIn("planted=0", reason)

    def test_crop_task_fails_water_only_with_dry_tiles_and_no_progress(self) -> None:
        ram = _blank_ram()
        center = (12, 25)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                _set_tile(ram, center[0] + dx, center[1] + dy, 0x54)
        # Wall off everything so water steps cannot progress.
        for ty in range(14, 40):
            for tx in range(3, 40):
                if abs(tx - center[0]) > 1 or abs(ty - center[1]) > 1:
                    _set_tile(ram, tx, ty, 0x05)
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        _set_player_tile(ram, center)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(
            work_mode="water",
            bounds=(3, 14, 30, 40),
            max_steps_per_target=5,
            max_failures=3,
        )
        task.reset(world)

        result = None
        for _ in range(80):
            result = task.step(world)
            if result.status != TaskStatus.RUNNING:
                break

        self.assertIsNotNone(result)
        assert result is not None
        # Either failure (dry remain) or timeout-ish failure — not silent SUCCESS.
        if result.status == TaskStatus.SUCCESS:
            self.assertTrue(
                is_no_work_reason(result.reason),
                msg=f"unexpected success reason: {result.reason}",
            )
            # no_work only valid if we never saw dry tiles
            self.assertEqual(task._dry_crop_tiles_at_start, 0)
        else:
            self.assertEqual(result.status, TaskStatus.FAILURE)


class SameDayEstablishWaterOrderTests(unittest.TestCase):
    """Day-plan crop pass order for same-day plant then water (rr-e1p)."""

    def test_crop_establish_then_water_phases_order(self) -> None:
        from harvest.planner.day_plan_phases import crop_establish_phases, crop_water_phases

        establish = [p.phase for p in crop_establish_phases()]
        water = [p.phase for p in crop_water_phases()]
        self.assertIn("CROP_ESTABLISH", establish)
        self.assertIn("ENSURE_CROP_SEEDS", establish)
        self.assertNotIn("ENSURE_WATERING_CAN", establish)
        self.assertIn("ENSURE_WATERING_CAN", water)
        self.assertIn("CROP_WATER", water)
        # Plant pass before water pass when both are scheduled.
        from harvest.planner.day_plan_phases import _crop_work_phases

        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = _crop_work_phases(
            has_harvest=False,
            has_waterable=True,
            has_seeds=True,
            is_rainy=False,
            late_day=False,
            policy=DayPlannerPolicy(),
        )
        names = [p.phase for p in phases]
        if "CROP_ESTABLISH" in names and "CROP_WATER" in names:
            self.assertLess(names.index("CROP_ESTABLISH"), names.index("CROP_WATER"))
            self.assertLess(
                names.index("CROP_ESTABLISH"),
                names.index("ENSURE_WATERING_CAN"),
            )


class MultihopMainPondAfterGapTests(unittest.TestCase):
    """After y=31 gap open, commit multi-hop F0 even without full BFS."""

    def test_gap_open_commits_multihop_when_pond_outside_viewport(self) -> None:
        """Simulates post-fence stall ~tile(25,30): gap open, pond far.

        Full BFS to (32,34) fails because map beyond a pocket is solid (stale
        viewport). Preferred edges also blocked. Must still commit multi-hop.
        """
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        # Default solid — "stale" outside a local pocket (viewport model).
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0x05)
        # Open walkable band north of wall + one gap through y=31.
        for ty in range(28, 31):
            for tx in range(20, 31):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_tile(ram, 20, 31, 0xA1)  # single gap
        # F0 pond water present but not path-connected in full BFS (stale east).
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        # North-lip stand cells exist as islands — solid barrier at x=31 models
        # viewport-stale gap between west lip and pond (ROM stall ~(25,30)).
        for tx in (32, 33, 34):
            _set_tile(ram, tx, 30, 0xA1)
        _set_tile(ram, 31, 30, 0x05)  # barrier: no full BFS to stands

        player = (25, 30)
        _set_player_tile(ram, player)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._water_steps = [((12, 25), (12, 26), "up")]
        task._water_index = 0
        task._plot_phase = "water"
        task._plots = [(12, 25)]
        task._plot_index = 0
        # Pretend we already spent a fence-open (gap is open).
        task._fence_open_attempts = 1

        # Precondition: full path to pond stands is None (viewport/stale).
        self.assertIsNone(task._pathfinder.find_path(ram, player, (32, 34)))
        self.assertIsNone(task._pathfinder.find_path(ram, player, (32, 30)))
        self.assertTrue(task._pond_corridor_gap_open(ram))

        task._start_refill(ram)

        self.assertEqual(
            task._plot_phase,
            "refill",
            msg=f"phase={task._plot_phase} state={task._state} "
            f"stand={task._refill_pond_tile} exhausted={task._refill_exhausted}",
        )
        self.assertTrue(task._refill_multihop)
        self.assertIsNotNone(task._refill_pond_tile)
        assert task._refill_pond_tile is not None
        self.assertIn(task._refill_pond_tile, {s for s, _ in FARM_MAIN_POND_STANDS})
        self.assertEqual(task._state, "navigate")
        self.assertFalse(task._refill_exhausted)
        self.assertEqual(task._refill_pond_tile and task._refill_pond_tile[0] >= 30, True)

    def test_fence_open_success_commits_multihop(self) -> None:
        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        ram[ADDR_TOOL] = 0x10
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_tile(ram, 18, 31, 0xA1)  # gap
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        _set_player_tile(ram, (18, 30))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._water_steps = [((12, 25), (12, 26), "up")]
        task._water_index = 0
        task._plots = [(12, 25)]
        task._plot_index = 0
        task._fence_open_attempts = 1
        # Fake completed fence subtask.
        task._fence_subtask = SimpleNamespace(
            step=lambda w: SimpleNamespace(
                status=TaskStatus.SUCCESS, reason="ok", action=None
            ),
            cleared_count=1,
        )
        task._state = "fence_open"
        task._plot_phase = "open_pond"

        result = task._handle_fence_open(world)
        self.assertIsNone(result)
        self.assertEqual(task._plot_phase, "refill")
        # Still north of wall: scripted east→south charge is armed before multihop.
        self.assertTrue(
            getattr(task, "_pending_gap_charge", False)
            or task._refill_multihop
            or len(task._action_queue) > 0,
            msg="must arm east→south charge or multihop after fence open",
        )
        # Simulate charge landing on (28,32) soft-block band → west→south-lip.
        task._action_queue.clear()
        _set_player_tile(ram, (28, 32))
        task._navigator.update(ram)
        task._pending_gap_charge = True
        task._handle_navigate(ram)
        self.assertTrue(
            getattr(task, "_pending_south_lip_charge", False)
            or task._refill_multihop
            or len(task._action_queue) > 0,
            msg="(28,32) band must arm west→south-lip or multihop",
        )
        # Simulate arriving at stand after lip charge.
        task._action_queue.clear()
        task._pending_south_lip_charge = True
        _set_player_tile(ram, (32, 34))
        task._navigator.update(ram)
        task._handle_navigate(ram)
        self.assertTrue(
            task._refill_multihop or task._state == "act",
            msg="at F0 stand must commit multihop/act",
        )
        if task._refill_pond_tile is not None:
            self.assertIn(task._refill_pond_tile, {s for s, _ in FARM_MAIN_POND_STANDS})

    def test_refill_hop_goal_densifies_toward_pond(self) -> None:
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (20, 30))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop = task._refill_hop_goal(ram, (20, 30), (32, 34))
        # From y=30: east-crawl toward x≥28 then south — never empty gap charge.
        self.assertGreaterEqual(
            hop[0],
            20,
            msg=f"from y=30 must east-crawl (x≥20), got {hop}",
        )
        self.assertNotEqual(hop, (12, 32))
        hop2 = task._refill_hop_goal(ram, (20, 29), (32, 34))
        self.assertGreaterEqual(
            hop2[0],
            20,
            msg=f"from y=29 must east-crawl not gap-south, got {hop2}",
        )
        # Must not densify onto soft-block gap column.
        self.assertFalse(
            hop2[1] >= 31 and hop2[0] <= 16,
            msg=f"must not densify into gap soft-block, got {hop2}",
        )

    def test_refill_hop_goal_from_gap_east_crawl_not_south(self) -> None:
        """From cleared gap ~(13,31): escape N/E; never densify south through gap.

        Standing ON the gap soft-blocks (13,31)→(13,32) empty-handed. ROM path
        is east-crawl on y=30 to x≥28 then south.
        """
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            if tx != 13:
                _set_tile(ram, tx, 31, 0x05)
        _set_player_tile(ram, (13, 31))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop1 = task._refill_hop_goal(ram, (13, 31), (32, 34))
        self.assertLessEqual(
            hop1[1],
            30,
            msg=f"from gap tile must leave y=31, got {hop1}",
        )
        self.assertNotEqual(
            hop1[1],
            32,
            msg=f"must not densify south through gap, got {hop1}",
        )
        hop2 = task._refill_hop_goal(ram, (13, 29), (32, 34))
        self.assertFalse(
            hop2[1] >= 32 and hop2[0] <= 16,
            msg=f"must not empty-charge gap south, got {hop2}",
        )
        self.assertGreaterEqual(
            hop2[0],
            13,
            msg=f"east-crawl should push east, got {hop2}",
        )

    def test_east_south_charge_targets_past_fence_end(self) -> None:
        """Charge must run past x=29 fence wall end (not stop at 29,30)."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        _set_player_tile(ram, (25, 30))
        task._navigator.update(ram)
        task._queue_east_south_corridor_charge((25, 30))
        self.assertTrue(task._pending_gap_charge)
        self.assertGreater(len(task._action_queue), 100)
        # East-only leg while x<31: long queue with RIGHT bursts + wiggle.
        self.assertGreaterEqual(
            len(task._action_queue),
            28 * 6,
            msg="east frames must cover past fence end x=31",
        )
        # snes_action layout: up=4 down=5 left=6 right=7
        downs = sum(1 for a in task._action_queue if int(a[5]) == 1)
        rights = sum(1 for a in task._action_queue if int(a[7]) == 1)
        # Prefer RIGHT-heavy when still under fence wall.
        self.assertGreater(
            rights,
            downs,
            msg="east-only leg must prefer RIGHT over long DOWN into posts",
        )
        # Still-north at (29,30) must re-queue (prior residual).
        task._action_queue.clear()
        task._pending_gap_charge = True
        task._east_south_charges = 1
        _set_player_tile(ram, (29, 30))
        task._navigator.update(ram)
        task._handle_navigate(ram)
        self.assertTrue(
            task._pending_gap_charge or len(task._action_queue) > 0,
            msg="still_north at (29,30) must re-queue east→south past x≥31",
        )
        self.assertGreaterEqual(task._east_south_charges, 2)
        # Past fence end: south-heavy charge.
        task._action_queue.clear()
        task._queue_east_south_corridor_charge((31, 30))
        downs2 = sum(1 for a in task._action_queue if int(a[5]) == 1)
        self.assertGreaterEqual(
            downs2,
            100,
            msg="x≥31 charge must long-DOWN into pond band",
        )

    def test_soft_band_lip_charge_caps_up_avoids_gap(self) -> None:
        """Soft band ~(29,32) must not long-UP back through open fence gap."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._queue_west_south_lip_charge((29, 32))
        self.assertTrue(task._pending_south_lip_charge)
        # snes_action: up=4 down=5 left=6 right=7
        ups = sum(1 for a in task._action_queue if int(a[4]) == 1)
        rights = sum(1 for a in task._action_queue if int(a[7]) == 1)
        lefts = sum(1 for a in task._action_queue if int(a[6]) == 1)
        downs = sum(1 for a in task._action_queue if int(a[5]) == 1)
        # Gap-safe soft band: RIGHT-heavy, short LEFT, capped UP.
        self.assertGreater(rights, lefts, msg="soft band must prefer RIGHT not long LEFT")
        self.assertLessEqual(ups, 40, msg=f"UP must be capped (got {ups}) to avoid gap re-entry")
        self.assertGreater(downs, 40, msg="soft band must DOWN first off soft-block")

    def test_refill_hop_past_fence_end_forces_south(self) -> None:
        """At (31,29) densify must push south, not self-hop thrash."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop = task._refill_hop_goal(ram, (31, 29), (32, 34))
        self.assertNotEqual(hop, (31, 29), msg=f"must not self-hop, got {hop}")
        self.assertGreaterEqual(
            hop[1],
            30,
            msg=f"past fence end must densify south, got {hop}",
        )
        hop2 = task._refill_hop_goal(ram, (31, 30), (32, 34))
        self.assertNotEqual(hop2, (31, 30), msg=f"must not self-hop at (31,30), got {hop2}")
        self.assertGreaterEqual(hop2[1], 31, msg=f"must push south from (31,30), got {hop2}")

    def test_south_lip_densify_prefers_east_not_ns_oscillate(self) -> None:
        """From (24,34) must not thrash to (24,35); prefer east toward F0."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop = task._refill_hop_goal(ram, (24, 34), (32, 34))
        self.assertNotEqual(
            hop,
            (24, 35),
            msg=f"must not pure-south thrash, got {hop}",
        )
        self.assertGreater(
            hop[0],
            24,
            msg=f"from (24,34) must push east toward F0, got {hop}",
        )
        hop2 = task._refill_hop_goal(ram, (24, 35), (32, 34))
        self.assertNotEqual(
            hop2,
            (24, 34),
            msg=f"must not reverse to (24,34), got {hop2}",
        )
        self.assertGreaterEqual(
            hop2[0],
            24,
            msg=f"from (24,35) must not go west, got {hop2}",
        )

    def test_south_of_wall_densify_thrash_arms_lip_charge(self) -> None:
        """Two stalls at (24,34)→F0 must arm west→south-lip charge."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        task._plot_phase = "refill"
        task._refill_multihop = True
        # First stall only records; second arms charge.
        p1 = task._find_nav_path(ram, (24, 34), (32, 34))
        self.assertIsNotNone(p1)
        p2 = task._find_nav_path(ram, (24, 34), (32, 34))
        self.assertIsNone(
            p2,
            msg="second densify stall must return None and arm lip charge",
        )
        self.assertTrue(
            getattr(task, "_pending_south_lip_charge", False)
            or len(task._action_queue) > 0,
            msg="south densify thrash must arm west→south-lip",
        )


class FenceLocalDropTests(unittest.TestCase):
    """FenceClearLoopTask must not hard-fail when pond BFS is viewport-blocked."""

    def test_navigate_pond_falls_back_to_local_drop(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        # Wall of solid tiles — no path to pond stands.
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0x05)
        # Tiny open cell around player.
        for ty in range(28, 31):
            for tx in range(14, 17):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (15, 29))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=1, max_steps_per_fence=200)
        # Avoid loading recorded toss task from disk.
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(15, 31), tile_id=0x05)
        task._navigator.update(ram)

        # Hop may improve manhattan inside the pocket once; then local_drop.
        final_state = None
        for _ in range(8):
            result = task.step(world)
            final_state = task._state
            if final_state == "local_drop":
                break
            # Simulate walk along hop without leaving the pocket.
            task._navigator.update(ram)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(final_state, "local_drop")

    def test_local_drop_clears_when_hands_empty(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(20, 40):
            for tx in range(10, 40):
                _set_tile(ram, tx, ty, 0xA1)
        _set_player_tile(ram, (15, 29))
        # Not carrying — local_drop should count as cleared.
        ram[ADDR_PLAYER_STATE] = 0

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(max_fences=2, max_steps_per_fence=200)
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "local_drop"
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(task.cleared_count, 1)
        self.assertEqual(task._state, "scan")


class KeepAliveClearOrderTests(unittest.TestCase):
    """CLEAR_FIELD must not starve crop keep-alive water (rr-3v9)."""

    def test_outdoor_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))
        self.assertLess(names.index("ENSURE_WATERING_CAN"), names.index("CLEAR_FIELD"))

    def test_outdoor_clear_before_water_when_no_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_outdoor_day_phases

        phases = build_outdoor_day_phases(
            weekday=3,
            hour=6,
            has_harvest=False,
            has_waterable=False,
            has_seeds=False,
            has_debris=True,
            is_rainy=False,
        )
        names = [p.phase for p in phases]
        self.assertIn("CLEAR_FIELD", names)
        self.assertNotIn("CROP_WATER", names)

    def test_full_day_water_before_clear_when_dry_crops(self) -> None:
        from harvest.planner.day_plan_phases import build_day_phases
        from harvest.planner.day_phase_types import DayPlannerPolicy

        phases = build_day_phases(
            None,
            weekday=3,
            hour=6,
            has_chickens=False,
            has_cows=False,
            has_harvest=False,
            has_waterable=True,
            has_seeds=False,
            has_debris=True,
            policy=DayPlannerPolicy(
                include_chickens=False,
                include_cows=False,
                include_shop_run=False,
                include_berry_run=False,
                include_end_day=False,
            ),
        )
        names = [p.phase for p in phases]
        self.assertIn("CROP_WATER", names)
        self.assertIn("CLEAR_FIELD", names)
        self.assertLess(names.index("CROP_WATER"), names.index("CLEAR_FIELD"))


if __name__ == "__main__":
    unittest.main()


class FenceCorridorOnlyTests(unittest.TestCase):
    """corridor_only FenceClearLoop must local-drop instead of pond thrash."""

    def test_corridor_only_carry_south_from_y30(self) -> None:
        """ROM: after lift player is often on y=30; charge must still fire."""
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_player_tile(ram, (13, 30))  # approach tile after lift
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=1, max_steps_per_fence=500, corridor_only=True
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(13, 31), tile_id=0x05)
        task._navigator.update(ram)

        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False),
            msg="must carry-south charge from y=30, not immediate local_drop",
        )
        self.assertNotEqual(
            task._state,
            "local_drop",
            msg="must not local_drop before carry-south charge on y=30",
        )

    def test_corridor_only_skips_navigate_pond_to_local_drop(self) -> None:
        from harvest.tasks.fence_flow import (
            ACTION_CARRYING_BIT,
            ADDR_PLAYER_STATE,
            FenceClearLoopTask,
        )

        ram = _blank_ram()
        ram[ADDR_TILEMAP] = 0x00
        ram[ADDR_INPUT_LOCK] = 1
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for tx in range(11, 30):
            _set_tile(ram, tx, 31, 0x05)
        _set_player_tile(ram, (13, 31))
        ram[ADDR_PLAYER_STATE] = ACTION_CARRYING_BIT

        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = FenceClearLoopTask(
            max_fences=1, max_steps_per_fence=500, corridor_only=True
        )
        task._toss_task = SimpleNamespace(frames=[])
        task.reset(world)
        task._state = "navigate_pond"
        task._current = SimpleNamespace(tile=(13, 31), tile_id=0x05)
        task._navigator.update(ram)

        # First step: carry-south charge from y<=31 (not only y==31).
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False)
            or task._state == "local_drop"
            or len(task._action_queue) > 0
            or result.reason
            in (
                "corridor_only south charge",
                "corridor_only local drop",
                "corridor_only drop south of wall",
            ),
            msg=f"state={task._state} reason={result.reason}",
        )
        self.assertTrue(
            getattr(task, "_corridor_charge_done", False)
            or result.reason == "corridor_only south charge"
            or len(task._action_queue) > 0,
            msg="corridor_only must attempt carry-south before local_drop",
        )
        # Drain charge queue then expect local_drop arm (still north in unit).
        for _ in range(200):
            result = task.step(world)
            if task._state == "local_drop":
                break
        self.assertEqual(
            task._state,
            "local_drop",
            msg=f"after charge must local_drop, got {task._state}",
        )
