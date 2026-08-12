"""Unit tests for planning-stack scaffolding (skills, contracts, waypoints, advisor)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import harvest  # noqa: F401
from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.core.task_progress import task_progress_snapshot
from harvest.core.world_context import WorldContext
from harvest.maps.map_config import Waypoint, densify_waypoints
from harvest.planner.day_phase_catalog import (
    COOP_CHORES_PHASE,
    CROP_ESTABLISH_PHASE,
    CROP_WATER_PHASE,
    HARVEST_ROUTE_PHASE,
)
from harvest.planner.day_phase_types import (
    PhaseKind,
    PhaseSpec,
    TaskContract,
    evaluate_task_contract,
)
from harvest.planner.day_plan_decision import build_day_plan_decision
from harvest.planner.local_llm import (
    apply_advisor_patch,
    build_local_llm_plan_advisor_from_env,
    validate_and_apply_phase_patch,
)
from harvest.tasks.primitives import TaskSequence
from harvest.tasks.skills import (
    NavigateUntilArrivedSkill,
    NavSkill,
    PressAInteractSkill,
    coop_nav_to_feed_bin_skill,
    coop_nav_to_shipping_bin_skill,
    farm_nav_to_shipping_bin_skill,
    sequence_skills,
    talk_press_skill,
)
from retro_harness import ActionResult, TaskResult, TaskStatus


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _write_u24(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF
    ram[addr + 2] = (value >> 16) & 0xFF


def _world(**fields: int) -> SimpleNamespace:
    ram = np.zeros(0x24000, dtype=np.uint8)
    base = LIVE_RAM_WRAM_OFFSET
    defaults = {
        "tilemap": 0x00,
        "day": 2,
        "hour": 6,
        "minute": 0,
        "weekday": 1,
        "season": 0,
        "stamina": 100,
        # money display = storage * 10; store 10 → display $100
        "money": 10,
        "player_x": 136,
        "player_y": 120,
        "input_lock": 1,
    }
    defaults.update(fields)
    for key, value in defaults.items():
        spec = field_spec(key)
        addr = spec.address + (base if spec.live_offset else 0)
        if spec.kind == "u16":
            _write_u16(ram, addr, value)
        elif spec.kind == "u24":
            _write_u24(ram, addr, value)
        else:
            ram[addr] = value & 0xFF
    return SimpleNamespace(frame=1, ram=ram, info={}, obs=None)


class _ScriptedTask:
    def __init__(self, name: str, statuses: list[TaskStatus]) -> None:
        self.name = name
        self.statuses = list(statuses)
        self.resets = 0

    def reset(self, world) -> None:
        self.resets += 1

    def can_start(self, world) -> bool:
        return True

    def step(self, world) -> TaskResult:
        status = self.statuses.pop(0) if self.statuses else TaskStatus.SUCCESS
        action = ActionResult(np.zeros(12, dtype=np.int32)) if status == TaskStatus.RUNNING else None
        return TaskResult(status=status, action=action, reason=self.name)


class DensifyWaypointsTests(unittest.TestCase):
    def test_inserts_intermediate_hops_for_long_same_map_span(self) -> None:
        waypoints = [
            Waypoint(tilemap=0x00, target_px=(0, 0), radius=12),
            Waypoint(tilemap=0x00, target_px=(320, 0), radius=12),  # 20 tiles
        ]
        dense = densify_waypoints(waypoints, max_hop_tiles=7)
        self.assertGreater(len(dense), 2)
        self.assertEqual(dense[0].target_px, (0, 0))
        self.assertEqual(dense[-1].target_px, (320, 0))
        for a, b in zip(dense, dense[1:]):
            dist = max(abs(b.target_px[0] - a.target_px[0]), abs(b.target_px[1] - a.target_px[1]))
            self.assertLessEqual(dist, 7 * 16)

    def test_preserves_map_transitions_and_exits(self) -> None:
        waypoints = [
            Waypoint(tilemap=0x00, target_px=(0, 0)),
            Waypoint(tilemap=0x0C, target_px=(400, 0), is_exit=True, exit_direction="right"),
        ]
        dense = densify_waypoints(waypoints, max_hop_tiles=7)
        self.assertEqual(len(dense), 2)


class TaskContractTests(unittest.TestCase):
    def test_phase_spec_accepts_contract_fields(self) -> None:
        spec = PhaseSpec(
            phase="COOP_CHORES",
            kind=PhaseKind.COOP_CHORES,
            estimated_frames=4000,
            required_maps=(0x28,),
            failure_modes=("feed_timeout", "egg_stuck"),
        )
        self.assertEqual(spec.contract.estimated_frames, 4000)
        self.assertEqual(spec.contract.required_maps, (0x28,))
        self.assertIn("feed_timeout", spec.contract.failure_modes)

    def test_task_contract_from_mapping(self) -> None:
        c = TaskContract.from_mapping(
            {"required_ram": ["stamina"], "estimated_frames": 100}
        )
        self.assertEqual(c.required_ram, ("stamina",))
        self.assertEqual(c.estimated_frames, 100)

    def test_production_crop_phases_declare_contracts(self) -> None:
        self.assertFalse(CROP_ESTABLISH_PHASE.contract.is_empty())
        self.assertIn(0x00, CROP_ESTABLISH_PHASE.contract.required_maps)
        self.assertIn("hoe", CROP_ESTABLISH_PHASE.contract.required_tools)
        self.assertIn("watering_can", CROP_WATER_PHASE.contract.required_tools)
        self.assertIn(0x28, COOP_CHORES_PHASE.contract.required_maps)
        self.assertIn("ship_money_not_instant", HARVEST_ROUTE_PHASE.contract.failure_modes)

    def test_evaluate_task_contract_map_and_tools(self) -> None:
        ok, reasons = evaluate_task_contract(
            CROP_ESTABLISH_PHASE.contract,
            tilemap=0x00,
            tools=("hoe", "seed"),
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, ())

        ok, reasons = evaluate_task_contract(
            CROP_ESTABLISH_PHASE.contract,
            tilemap=0x28,
            tools=("hoe",),
        )
        self.assertFalse(ok)
        self.assertTrue(any(r.startswith("map_mismatch") for r in reasons))
        self.assertTrue(any(r.startswith("missing_tool:seed") for r in reasons))

    def test_evaluate_task_contract_known_ram_fields(self) -> None:
        ok, reasons = evaluate_task_contract(
            TaskContract(required_ram=("stamina", "tilemap")),
        )
        self.assertTrue(ok)
        self.assertEqual(reasons, ())

        ok, reasons = evaluate_task_contract(
            TaskContract(required_ram=("not_a_real_field_xyz",)),
        )
        self.assertFalse(ok)
        self.assertTrue(any("unknown_ram_field" in r for r in reasons))

    def test_tool_tags_from_ram_maps_carry_pair(self) -> None:
        from harvest.core.carry import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
        from harvest.core.tile_catalog import Tool
        from harvest.planner.day_phase_types import tool_tags_from_ram

        ram = np.zeros(0x1000, dtype=np.uint8)
        ram[ADDR_TOOL_SELECTED] = int(Tool.HOE)
        ram[ADDR_TOOL_BACKPACK] = 0x07  # potato seeds
        tags = tool_tags_from_ram(ram)
        self.assertEqual(tags, ("hoe", "seed"))

        ram[ADDR_TOOL_SELECTED] = int(Tool.WATERING_CAN)
        ram[ADDR_TOOL_BACKPACK] = 0
        self.assertEqual(tool_tags_from_ram(ram), ("watering_can",))

    def test_preflight_phase_contract_soft_reasons(self) -> None:
        from harvest.core.carry import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED
        from harvest.core.ram_catalog import field_spec
        from harvest.core.tile_catalog import Tool
        from harvest.planner.day_phase_types import preflight_phase_contract

        ram = np.zeros(0x20000, dtype=np.uint8)
        ram[field_spec("tilemap").address] = 0x28  # coop, not farm
        ram[ADDR_TOOL_SELECTED] = int(Tool.HOE)
        ram[ADDR_TOOL_BACKPACK] = 0  # missing seed

        result = preflight_phase_contract(CROP_ESTABLISH_PHASE, ram=ram)
        self.assertFalse(result["ok"])
        self.assertFalse(result["empty"])
        self.assertEqual(result["phase"], "CROP_ESTABLISH")
        self.assertTrue(any(r.startswith("map_mismatch") for r in result["reasons"]))
        self.assertTrue(any(r.startswith("missing_tool:seed") for r in result["reasons"]))
        self.assertIn("hoe", result["tools"])

        # Farm + hoe + seed → ok
        ram[field_spec("tilemap").address] = 0x00
        ram[ADDR_TOOL_BACKPACK] = 0x07
        ok_result = preflight_phase_contract(CROP_ESTABLISH_PHASE, ram=ram)
        self.assertTrue(ok_result["ok"])
        self.assertEqual(ok_result["reasons"], [])


class WorldContextTests(unittest.TestCase):
    def test_caches_repeated_reads_for_same_frame(self) -> None:
        # money: storage 10 → display $100 (display_multiplier=10)
        world = _world(stamina=42, money=10)
        ctx = WorldContext().bind(world)
        a = ctx.stamina(world.ram)
        b = ctx.stamina(world.ram)
        self.assertEqual(a, 42)
        self.assertEqual(b, 42)
        self.assertIn("stamina", ctx._cache)
        snap = ctx.snapshot_dict(world.ram)
        self.assertEqual(snap["stamina"], 42)
        self.assertEqual(snap["money"], 100)


class SkillCompositionTests(unittest.TestCase):
    def test_task_sequence_progress_snapshot_exposes_child(self) -> None:
        world = _world()
        first = _ScriptedTask("first", [TaskStatus.RUNNING, TaskStatus.SUCCESS])
        second = _ScriptedTask("second", [TaskStatus.SUCCESS])
        seq = TaskSequence(name="compose", tasks=[first, second], idle_between_tasks=False)
        seq.reset(world)
        seq.step(world)
        snap = seq.progress_snapshot()
        self.assertEqual(snap.task_name, "compose")
        self.assertEqual(snap.phase_text, "first")
        self.assertIsNotNone(snap.child)

    def test_sequence_skills_runs_children(self) -> None:
        world = _world()
        seq = sequence_skills(
            "demo",
            _ScriptedTask("a", [TaskStatus.SUCCESS]),
            _ScriptedTask("b", [TaskStatus.SUCCESS]),
            idle_between=False,
        )
        seq.reset(world)
        result = seq.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)

    def test_press_a_interact_skill_succeeds_without_condition(self) -> None:
        world = _world()
        skill = PressAInteractSkill(name="tap", face="up", hold_frames=1, settle_frames=0, face_frames=0)
        skill.reset(world)
        # Drain face+press sequence then SUCCESS.
        statuses = []
        for _ in range(20):
            result = skill.step(world)
            statuses.append(result.status)
            if result.status != TaskStatus.RUNNING:
                break
        self.assertEqual(statuses[-1], TaskStatus.SUCCESS)

    def test_skill_factories_bind_named_targets(self) -> None:
        feed = coop_nav_to_feed_bin_skill()
        self.assertEqual(feed.name, "coop_nav_feed_bin")
        self.assertIsInstance(feed, NavSkill)
        self.assertEqual(feed.target_px, (2 * 16 + 8, 6 * 16 + 8))

        ship = farm_nav_to_shipping_bin_skill()
        self.assertEqual(ship.name, "farm_nav_ship_bin")
        self.assertEqual(ship.target_px, (62 * 16 + 8, 60 * 16 + 8))

        talk = talk_press_skill(name="d1_ann", face="left")
        self.assertEqual(talk.name, "d1_ann")
        self.assertEqual(talk.face, "left")

    def test_coop_nav_factories_accept_host_navigate(self) -> None:
        """Production CoopChoresTask passes navigate= for specialized routing."""
        steps = {"n": 0}

        def navigate(world) -> np.ndarray | None:
            steps["n"] += 1
            if steps["n"] < 2:
                return np.zeros(12, dtype=np.int32)
            return None

        world = _world()
        feed = coop_nav_to_feed_bin_skill(navigate=navigate)
        self.assertIsInstance(feed, NavigateUntilArrivedSkill)
        self.assertEqual(feed.name, "coop_nav_feed_bin")
        feed.reset(world)
        r1 = feed.step(world)
        self.assertEqual(r1.status, TaskStatus.RUNNING)
        self.assertIsNotNone(r1.action)
        r2 = feed.step(world)
        self.assertEqual(r2.status, TaskStatus.SUCCESS)

        ship = coop_nav_to_shipping_bin_skill(navigate=lambda _w: None)
        self.assertIsInstance(ship, NavigateUntilArrivedSkill)
        self.assertEqual(ship.name, "coop_nav_ship_bin")
        ship.reset(world)
        self.assertEqual(ship.step(world).status, TaskStatus.SUCCESS)


class AdvisorApplyTests(unittest.TestCase):
    def test_default_ignores_phase_rewrites(self) -> None:
        decision = build_day_plan_decision(ram=_world().ram)
        patched = apply_advisor_patch(
            decision,
            {
                "notes": ["hello"],
                "append_phases": [
                    {"phase": "COOP_CHORES", "kind": "coop_chores"},
                ],
            },
            source="test",
            apply_validated=False,
        )
        self.assertEqual(patched.phase_names, decision.phase_names)
        self.assertIn("advisor_phase_changes_ignored", patched.notes)

    def test_validated_append_optional_phase(self) -> None:
        decision = build_day_plan_decision(ram=_world().ram)
        if "COOP_CHORES" in decision.phase_names:
            # Already present — reorder path instead.
            patched = apply_advisor_patch(
                decision,
                {"reorder_optional": ["COOP_CHORES"]},
                source="test",
                apply_validated=True,
            )
            self.assertIn("advisor_phase_patch_applied", patched.notes)
            return
        patched = apply_advisor_patch(
            decision,
            {
                "append_phases": [
                    {
                        "phase": "COOP_CHORES",
                        "kind": "coop_chores",
                        "failure_policy": "optional",
                    }
                ],
            },
            source="test",
            apply_validated=True,
        )
        self.assertIn("COOP_CHORES", patched.phase_names)
        self.assertIn("advisor_phase_patch_applied", patched.notes)

    def test_full_rewrite_rejected(self) -> None:
        decision = build_day_plan_decision(ram=_world().ram)
        validated, notes = validate_and_apply_phase_patch(
            decision, {"phase_names": ["UNSAFE"]}
        )
        self.assertIsNone(validated)
        self.assertIn("advisor_full_phase_rewrite_rejected", notes)

    def test_env_builder_reads_apply_flag(self) -> None:
        advisor = build_local_llm_plan_advisor_from_env(
            {
                "HARVEST_PLAN_LLM_URL": "http://127.0.0.1:9/api",
                "HARVEST_PLAN_LLM_APPLY": "1",
            }
        )
        self.assertIsNotNone(advisor)
        assert advisor is not None
        self.assertTrue(advisor.apply_validated)

    def test_env_builder_disabled_without_url(self) -> None:
        self.assertIsNone(build_local_llm_plan_advisor_from_env({}))


class TaskProgressChildKeysTests(unittest.TestCase):
    def test_task_progress_snapshot_uses_progress_method(self) -> None:
        world = _world()
        seq = TaskSequence(
            name="outer",
            tasks=[_ScriptedTask("inner", [TaskStatus.RUNNING])],
            idle_between_tasks=False,
        )
        seq.reset(world)
        seq.step(world)
        snap = task_progress_snapshot(seq)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.task_name, "outer")


if __name__ == "__main__":
    unittest.main()
