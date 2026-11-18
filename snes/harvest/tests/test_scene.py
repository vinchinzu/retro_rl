from __future__ import annotations

import unittest

import numpy as np
import pytest

from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec, read_ram_value
from harvest.core.scene import (
    SceneLocation,
    SceneMode,
    classify_scene,
    classify_scene_from_ram,
    in_sleep_transition,
    morning_scene_ready,
    scene_indicates_ending,
)
from harvest.core.world_snapshot import WorldSnapshot
from harvest.runtime.rom_tools import parse_save_state, resolve_state_path


def _ram(size: int = 0x24000) -> np.ndarray:
    ram = np.zeros(size, dtype=np.uint8)
    ram[field_spec("input_lock").address] = 1
    _write_u16(ram, field_spec("player_x").address + _live_base(ram), 136)
    _write_u16(ram, field_spec("player_y").address + _live_base(ram), 424)
    return ram


def _live_base(ram: np.ndarray) -> int:
    return LIVE_RAM_WRAM_OFFSET if len(ram) > 0x20000 else 0


def _write_u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


class SceneClassifierTests(unittest.TestCase):
    def test_classifies_seasonal_farm_as_normal_farm(self) -> None:
        for tilemap in (0x00, 0x01, 0x02, 0x03):
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, SceneLocation.FARM)
                self.assertIn("farm", scene.variant)

    def test_classifies_house_variants(self) -> None:
        expected = {0x15: "base", 0x16: "level1", 0x17: "level2"}
        for tilemap, variant in expected.items():
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap
                _write_u16(ram, field_spec("player_y").address + _live_base(ram), 120)

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, SceneLocation.HOUSE)
                self.assertIn(variant, scene.variant)

    def test_classifies_known_domain_locations(self) -> None:
        expected = {
            0x04: SceneLocation.TOWN,
            0x05: SceneLocation.TOWN,
            0x10: SceneLocation.MOUNTAIN,
            0x1C: SceneLocation.SHOP,
            0x24: SceneLocation.SHOP,
            0x26: SceneLocation.SHED,
            0x27: SceneLocation.BARN,
            0x28: SceneLocation.COOP,
        }
        for tilemap, location in expected.items():
            with self.subTest(tilemap=tilemap):
                ram = _ram()
                ram[field_spec("tilemap").address] = tilemap

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, location)

    def test_classifies_town_event_stale_dialogue_as_normal(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x05
        ram[field_spec("input_lock").address] = 1
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0316)
        ram[field_spec("dialog_text_mode").address] = 0x02
        _write_u16(ram, field_spec("player_x").address + _live_base(ram), 440)
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 160)

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.TOWN)

    def test_classifies_dialogue_menu_and_generic_input_lock(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("input_lock").address] = 0
        _write_u16(ram, field_spec("dialog_text_id").address, 0x03A6)

        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

        ram[field_spec("dialog_menu_cursor").address] = 2
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.MENU)

        _write_u16(ram, field_spec("dialog_text_id").address, 0)
        ram[field_spec("dialog_text_mode").address] = 0
        ram[field_spec("dialog_menu_cursor").address] = 0
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.INPUT_LOCKED)

        ram[field_spec("tilemap").address] = 0xFE
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0315)
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

        ram[field_spec("input_lock").address] = 1
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.DIALOGUE)

    def test_classifies_transition_sleep_ending_unknown_and_invalid(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("player_state").address + _live_base(ram)] = 0x80
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.MAP_TRANSITION)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x0F
        _write_u16(ram, field_spec("time_running").address + _live_base(ram), 6)
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.SLEEP_WAKE_TRANSITION)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        ram[field_spec("ending_scene_index").address + _live_base(ram)] = 0x25
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.ENDING_CREDITS)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0xFE
        scene = classify_scene_from_ram(ram)
        self.assertEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertTrue(scene.is_recoverable)
        self.assertTrue(scene.needs_input_dismiss)

        ram = _ram()
        ram[field_spec("tilemap").address] = 0x00
        _write_u16(ram, field_spec("player_x").address + _live_base(ram), 0)
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 0)
        self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.INVALID_COORDINATES)

    def test_winter_path_is_not_the_overnight_fade(self) -> None:
        """Both load tilemap 0x0F (probe on LN_cowfix_pin_winter_d5).

        The Winter path reads the clock byte 0 (map load, horse event) or 1
        (walking); the fade after the bed reads 6. Treating the path as the
        fade sent the farmer to bed at 09:00 on Winter D5 and D6.
        """
        for clock in (0, 1):
            with self.subTest(clock=clock):
                ram = _ram()
                ram[field_spec("tilemap").address] = 0x0F
                _write_u16(ram, field_spec("time_running").address + _live_base(ram), clock)
                _write_u16(ram, field_spec("player_x").address + _live_base(ram), 191)
                _write_u16(ram, field_spec("player_y").address + _live_base(ram), 128)

                scene = classify_scene_from_ram(ram)

                self.assertEqual(scene.mode, SceneMode.NORMAL)
                self.assertEqual(scene.location, SceneLocation.PATH)
                self.assertFalse(in_sleep_transition(ram))
        _write_u16(ram, field_spec("time_running").address + _live_base(ram), 6)
        self.assertTrue(in_sleep_transition(ram))

    def test_classifies_festival_event_location(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x04
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.FESTIVAL)

    def test_festival_eve_with_stale_dialogue_and_free_player_is_normal(self) -> None:
        """ROM pin (run24 D22): tomorrow-weather 6 marks the festival EVE, and
        text_id/mode were stale while input_lock == 1 and the farmer walked
        freely. Classifying that as a cutscene sent the mash-through recovery
        into 12 x 241f timeouts."""
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x04
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
        _write_u16(ram, field_spec("dialog_text_id").address, 0x0316)
        ram[field_spec("dialog_text_mode").address] = 0x02

        scene = classify_scene_from_ram(ram)

        self.assertNotEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertFalse(scene.needs_input_dismiss)

    def test_classifies_festival_event_player_state_as_cutscene_event(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x04
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
        ram[field_spec("player_state").address + _live_base(ram)] = 0x05

        scene = classify_scene_from_ram(ram)

        self.assertEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertEqual(scene.location, SceneLocation.FESTIVAL)
        self.assertTrue(scene.needs_input_dismiss)

    def test_a_tired_or_carrying_farmer_on_a_festival_eve_is_not_a_cutscene(self) -> None:
        """Y2 Fall D19 (LN_y23_festeve_carry): player_state 0x09 (tired, input free) in town on
        the eve of Fall D20 read as a cutscene; return_home mashed A/B for 8 rescues."""
        for state in (0x03, 0x09, 0x11):
            ram = _ram()
            ram[field_spec("tilemap").address] = 0x04
            ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
            ram[field_spec("player_state").address + _live_base(ram)] = state
            ram[field_spec("input_lock").address] = 1
            scene = classify_scene_from_ram(ram)
            self.assertNotEqual(scene.mode, SceneMode.CUTSCENE_EVENT, hex(state))
            self.assertFalse(scene.needs_input_dismiss, hex(state))
        for state in (0x05, 0x43, 0x45):  # tool animation and event states stay cutscenes
            ram = _ram()
            ram[field_spec("tilemap").address] = 0x04
            ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
            ram[field_spec("player_state").address + _live_base(ram)] = state
            self.assertEqual(classify_scene_from_ram(ram).mode, SceneMode.CUTSCENE_EVENT, hex(state))

    def test_hot_spring_soak_on_a_festival_eve_is_not_a_cutscene(self) -> None:
        """LN_spa_whatif_soak_stall_FAIL: D22 bath, player_state 0x21, festival tomorrow."""
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x10
        ram[field_spec("weather_tomorrow").address + _live_base(ram)] = 6
        ram[field_spec("player_state").address + _live_base(ram)] = 0x21
        ram[field_spec("input_lock").address] = 1
        _write_u16(ram, field_spec("player_x").address + _live_base(ram), 654)
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 201)

        scene = classify_scene_from_ram(ram)

        self.assertNotEqual(scene.mode, SceneMode.CUTSCENE_EVENT)
        self.assertFalse(scene.needs_input_dismiss)

    def test_morning_scene_ready_and_ending_helpers(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x15
        _write_u16(ram, field_spec("player_y").address + _live_base(ram), 120)
        ram[field_spec("hour").address + _live_base(ram)] = 6
        scene = classify_scene_from_ram(ram)
        self.assertTrue(morning_scene_ready(scene, hour=6))
        self.assertFalse(morning_scene_ready(scene, hour=14))

        ram[field_spec("ending_scene_index").address + _live_base(ram)] = 0x04
        ending = classify_scene_from_ram(ram)
        self.assertTrue(scene_indicates_ending(ending))
        self.assertFalse(morning_scene_ready(ending, hour=6))

    def test_classifies_existing_world_snapshot(self) -> None:
        ram = _ram()
        ram[field_spec("tilemap").address] = 0x27
        snapshot = WorldSnapshot.from_ram(ram, bounds=(0, 0, 1, 1))

        scene = classify_scene(snapshot)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.BARN)

    def test_pinned_morning_after_sleep_fixture_is_normal_house_scene(self) -> None:
        state = parse_save_state(resolve_state_path("Y1_D3_Morning"))

        scene = classify_scene_from_ram(state.ram)

        self.assertEqual(scene.mode, SceneMode.NORMAL)
        self.assertEqual(scene.location, SceneLocation.HOUSE)
        self.assertEqual(read_ram_value(state.ram, "season", raw=True), 0)
        self.assertEqual(read_ram_value(state.ram, "day", raw=True), 3)
        self.assertEqual(read_ram_value(state.ram, "hour", raw=True), 6)
        self.assertEqual(read_ram_value(state.ram, "minute", raw=True), 0)




class PeddlerDeclineTests(unittest.TestCase):
    """Year 2+ farm-road peddler: "Buy it" is row 0 and costs 20,000G."""

    def _offer(self, row: int) -> np.ndarray:
        from harvest.core.scene import PEDDLER_OFFER_TEXT_ID

        ram = _ram()
        _write_u16(ram, field_spec("dialog_text_id").address, PEDDLER_OFFER_TEXT_ID)
        ram[field_spec("dialog_choice_result").address] = row
        ram[0x019B] = 0x10
        return ram

    def test_steers_to_not_buy_then_confirms(self) -> None:
        from harvest.core.scene import peddler_decline_action

        on_buy = peddler_decline_action(self._offer(0), step=0)
        self.assertTrue(on_buy[5])   # down
        self.assertFalse(on_buy[8])  # never A on "Buy it"
        on_decline = peddler_decline_action(self._offer(1), step=0)
        self.assertTrue(on_decline[8])
        self.assertFalse(on_decline[5])

    def test_menu_opening_gap_holds_the_pad(self) -> None:
        """ROM trace: bit 0x80 sets ~12 frames before 0x10, and an A there buys.

        LN_y23_whatif_peddler_y2w1: the scene guard's A at 0x81/0x83 closed the
        box on row 0 and the wallet fell 35,000 -> 15,000G.
        """
        from harvest.core.scene import peddler_decline_action

        opening = self._offer(0)
        for flags in (0x81, 0x83):
            opening[0x019B] = flags
            for step in range(12):
                held = peddler_decline_action(opening, step=step)
                self.assertIsNotNone(held)
                self.assertFalse(held.any())

    def test_other_prompts_and_pages_are_left_alone(self) -> None:
        from harvest.core.scene import peddler_decline_action

        paging = self._offer(0)
        paging[0x019B] = 0
        self.assertIsNone(peddler_decline_action(paging, step=0))
        other = self._offer(0)
        _write_u16(other, field_spec("dialog_text_id").address, 0x0303)
        self.assertIsNone(peddler_decline_action(other, step=0))


@pytest.mark.rom
def test_planner_declines_the_native_peddler_offer():
    """What-if pin: natural Y2 Winter D1 morning, wallet poked to 35,000G.

    Before the fix the scene guard's A answered row 0 and the wallet fell to
    15,000G; the guard now waits out the menu's opening and picks row 1.
    """
    from harvest.planner.multi_day_planner import MultiDayPlannerTask
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from retro_harness import AuditedEnv, AuditCapabilities, TaskStatus, WorldState

    env = AuditedEnv(make_harvest_env("LN_y23_whatif_peddler_y2w1", lean_step=True),
                     capabilities=AuditCapabilities.all("harvest.peddler_decline_test"))
    try:
        env.reset()
        assert read_ram_value(env.get_ram(), "money") == 35000
        planner = MultiDayPlannerTask(until_year=99, max_days=4)
        planner.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        texts = set()
        for frame in range(2500):
            result = planner.step(WorldState(frame=frame, ram=env.get_ram(), info={}, obs=None))
            assert result.status == TaskStatus.RUNNING, result.reason
            env.step(result.action.action if result.action else make_action())
            texts.add(int(read_ram_value(env.get_ram(), "dialog_text_id", raw=True)))
        assert {0x014D, 0x014F} <= texts and 0x014E not in texts
        assert read_ram_value(env.get_ram(), "money") == 35000
        assert env.audit().ram_writes == env.audit().mid_run_loads == 0
    finally:
        env.close()


if __name__ == "__main__":
    unittest.main()
