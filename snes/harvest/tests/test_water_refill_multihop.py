"""Unit tests for multi-hop main-pond refill after fence gap."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

# Path-stable import of sibling helpers (works under unittest and pytest importlib mode).
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
from water_refill_helpers import (
    _blank_ram,
    _set_player_tile,
    _set_tile,
)

from harvest.maps.map_config import FARM_MAIN_POND_STANDS
from harvest.tasks.crop_planter import CropWaterTask
from harvest.core.tile_catalog import (
    ADDR_INPUT_LOCK,
    ADDR_TILEMAP,
    ADDR_TOOL,
)
from retro_harness import TaskStatus


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
            getattr(task._corridor, "pending_gap_charge", False)
            or task._refill_multihop
            or len(task._action_queue) > 0,
            msg="must arm east→south charge or multihop after fence open",
        )
        # Simulate charge landing on (28,32) soft-block band → west→south-lip.
        task._action_queue.clear()
        _set_player_tile(ram, (28, 32))
        task._navigator.update(ram)
        task._corridor.pending_gap_charge = True
        task._handle_navigate(ram)
        self.assertTrue(
            getattr(task._corridor, "pending_south_lip_charge", False)
            or task._refill_multihop
            or len(task._action_queue) > 0,
            msg="(28,32) band must arm west→south-lip or multihop",
        )
        # Simulate arriving at stand after lip charge.
        task._action_queue.clear()
        task._corridor.pending_south_lip_charge = True
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
        self.assertTrue(task._corridor.pending_gap_charge)
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
        task._corridor.pending_gap_charge = True
        task._corridor.east_south_charges = 1
        _set_player_tile(ram, (29, 30))
        task._navigator.update(ram)
        task._handle_navigate(ram)
        self.assertTrue(
            task._corridor.pending_gap_charge or len(task._action_queue) > 0,
            msg="still_north at (29,30) must re-queue east→south past x≥31",
        )
        self.assertGreaterEqual(task._corridor.east_south_charges, 2)
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
        self.assertTrue(task._corridor.pending_south_lip_charge)
        # snes_action: up=4 down=5 left=6 right=7
        ups = sum(1 for a in task._action_queue if int(a[4]) == 1)
        rights = sum(1 for a in task._action_queue if int(a[7]) == 1)
        lefts = sum(1 for a in task._action_queue if int(a[6]) == 1)
        downs = sum(1 for a in task._action_queue if int(a[5]) == 1)
        # Gap-safe soft band: RIGHT-heavy, short LEFT, capped UP.
        self.assertGreater(rights, lefts, msg="soft band must prefer RIGHT not long LEFT")
        self.assertLessEqual(ups, 40, msg=f"UP must be capped (got {ups}) to avoid gap re-entry")
        self.assertGreater(downs, 40, msg="soft band must DOWN first off soft-block")
        # rr-qc9r: LEFT must stay brief so we don't land (25,34) then thrash.
        self.assertLessEqual(lefts, 20, msg=f"LEFT must be brief wiggle (got {lefts})")

    def test_south_far_lip_charge_no_up_on_y34(self) -> None:
        """From (25,34) pure east — no UP that re-enters soft (29,32)."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._queue_west_south_lip_charge((25, 34))
        self.assertTrue(task._corridor.pending_south_lip_charge)
        ups = sum(1 for a in task._action_queue if int(a[4]) == 1)
        rights = sum(1 for a in task._action_queue if int(a[7]) == 1)
        downs = sum(1 for a in task._action_queue if int(a[5]) == 1)
        self.assertGreater(rights, 100, msg="south_far/south on y=34 must long RIGHT")
        self.assertEqual(ups, 0, msg=f"y=34 charge must not UP into soft band (got {ups})")
        # Optional brief DOWN soft-break only.
        self.assertLessEqual(downs, 30, msg=f"y=34 should not long-DOWN (got {downs})")

    def test_refill_hop_from_25_34_short_east_not_direct_f0(self) -> None:
        """Densify (25,34)→F0 must hop +3/+4 east, not direct 7-tile thrash."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop = task._refill_hop_goal(ram, (25, 34), (32, 34))
        self.assertNotEqual(
            hop,
            (32, 34),
            msg=f"must not densify direct 7-tile F0, got {hop}",
        )
        self.assertGreater(
            hop[0],
            25,
            msg=f"must push east from (25,34), got {hop}",
        )
        self.assertLessEqual(
            hop[0] - 25,
            5,
            msg=f"east hop should be short intermediate, got {hop}",
        )
        self.assertGreaterEqual(hop[1], 34, msg=f"must stay on south lip, got {hop}")

    def test_soft_block_densify_south_east_not_west(self) -> None:
        """From (29,32) densify south/east toward F0 — not west to (24,32)."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        task._navigator.update(ram)
        hop = task._refill_hop_goal(ram, (29, 32), (32, 34))
        self.assertNotEqual(hop[0], 24, msg=f"must not force west first, got {hop}")
        self.assertTrue(
            hop[1] > 32 or hop[0] >= 29,
            msg=f"soft-block densify must south/east, got {hop}",
        )

    def test_south_lip_near_f0_commits_multihop_not_recharge(self) -> None:
        """At (29,35) after lip charge, multihop/act — not another RIGHT overshoot."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water")
        task.reset(world)
        _set_player_tile(ram, (29, 35))
        task._navigator.update(ram)
        task._corridor.pending_south_lip_charge = True
        task._corridor.south_lip_charges = 2
        task._fence_open_attempts = 1
        task._handle_navigate(ram)
        # Must not re-arm another long lip charge from near F0.
        self.assertLessEqual(
            task._corridor.south_lip_charges,
            2,
            msg="near F0 must not re-queue lip charge",
        )
        self.assertFalse(
            getattr(task._corridor, "pending_south_lip_charge", False),
            msg="pending lip charge must clear at near F0",
        )
        self.assertTrue(
            task._refill_multihop or task._state == "act",
            msg="near F0 must commit multihop/act",
        )

    def test_start_refill_soft_resets_exhausted_charges(self) -> None:
        """Exhausted lip charges soft-reset so next empty-can attempt can arm."""
        ram = _blank_ram()
        for ty in range(64):
            for tx in range(64):
                _set_tile(ram, tx, ty, 0xA1)
        for ty in range(31, 34):
            for tx in range(31, 35):
                _set_tile(ram, tx, ty, 0xF0)
        _set_player_tile(ram, (14, 27))
        world = SimpleNamespace(ram=ram, info={}, obs=None)
        task = CropWaterTask(work_mode="water", refill_bounds=(3, 10, 62, 60))
        task.reset(world)
        task._navigator.update(ram)
        task._corridor.south_lip_charges = 9
        task._corridor.east_south_charges = 5
        task._start_refill(ram)
        self.assertEqual(
            task._corridor.south_lip_charges,
            0,
            msg="exhausted lip charges must soft-reset on new refill",
        )
        self.assertEqual(task._corridor.east_south_charges, 0)

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
        # Short intermediate hop — not direct (32,34) 8-tile thrash.
        self.assertLessEqual(
            hop[0] - 24,
            5,
            msg=f"should densify short east hop, got {hop}",
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
            getattr(task._corridor, "pending_south_lip_charge", False)
            or len(task._action_queue) > 0,
            msg="south densify thrash must arm west→south-lip",
        )


if __name__ == "__main__":
    unittest.main()
