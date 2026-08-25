"""Unit tests for Spring D1 livestock dealer talk (not the buy-cow counter)."""

from __future__ import annotations

import unittest

import numpy as np

from retro_harness import TaskStatus, WorldState
from retro_harness.controls import pressed_snes_buttons

from harvest.core.npc_catalog import (
    ADDR_PLAYER_GOBJ_INDEX,
    GOBJ_STRUCT_BASE,
    GOBJ_STRUCT_STRIDE,
)
from harvest.core.ram_catalog import LIVE_RAM_WRAM_OFFSET, field_spec
from harvest.maps.map_config import ROUTES
from harvest.tasks.town_day1_build import build_day1_handoff_tasks
from harvest.tasks.town_day1_tasks import (
    ANIMAL_SHOP_TILEMAP,
    BIT_ANN,
    BIT_EVE,
    BIT_LIVESTOCK,
    PressAUntilBitOrTimeout,
    TOWN_TILEMAP,
    TrackNpcUntilBitTask,
    WalkUntilCoordTask,
)


def _write_u16(ram: np.ndarray, address: int, value: int) -> None:
    base = LIVE_RAM_WRAM_OFFSET + address
    ram[base] = value & 0xFF
    ram[base + 1] = (value >> 8) & 0xFF


def _write_gobj(ram: np.ndarray, slot: int, sprite: int, x: int, y: int) -> None:
    offset = GOBJ_STRUCT_BASE + slot * GOBJ_STRUCT_STRIDE
    ram[offset] = 0x77
    ram[offset + 1] = 0x77
    ram[offset + 2] = sprite & 0xFF
    ram[offset + 3] = (sprite >> 8) & 0xFF
    ram[offset + 8] = x & 0xFF
    ram[offset + 9] = (x >> 8) & 0xFF
    ram[offset + 10] = y & 0xFF
    ram[offset + 11] = (y >> 8) & 0xFF
    ram[offset + 0x13] = 2
    ram[offset + 0x14] = 4
    ram[offset + 0x15] = 5


def _npc_world(*, player_xy: tuple[int, int], npc_xy: tuple[int, int], bit: int = 0) -> WorldState:
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[field_spec("tilemap").address] = TOWN_TILEMAP
    ram[field_spec("input_lock").address] = 1
    ram[ADDR_PLAYER_GOBJ_INDEX] = 0
    ram[ADDR_PLAYER_GOBJ_INDEX + 1] = 0
    _write_gobj(ram, 0, 0x0005, player_xy[0], player_xy[1])
    _write_gobj(ram, 1, 0x0230, npc_xy[0], npc_xy[1])
    ram[field_spec("d1_town_event_mask").address] = bit
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def _shop_world(*, x: int, y: int, tilemap: int = ANIMAL_SHOP_TILEMAP) -> WorldState:
    ram = np.zeros(LIVE_RAM_WRAM_OFFSET + 0x20000, dtype=np.uint8)
    ram[field_spec("tilemap").address] = tilemap
    ram[field_spec("input_lock").address] = 1
    _write_u16(ram, field_spec("player_x").address, x)
    _write_u16(ram, field_spec("player_y").address, y)
    return WorldState(frame=0, ram=ram, info={}, obs=None)


def _names(task, acc: list[str] | None = None) -> list[str]:
    acc = acc if acc is not None else []
    acc.append(str(getattr(task, "name", "?")))
    child = getattr(task, "child", None)
    if child is not None:
        _names(child, acc)
    for inner in getattr(task, "tasks", ()) or ():
        _names(inner, acc)
    return acc


def _find(task, name: str):
    if getattr(task, "name", None) == name:
        return task
    child = getattr(task, "child", None)
    if child is not None:
        found = _find(child, name)
        if found is not None:
            return found
    for inner in getattr(task, "tasks", ()) or ():
        found = _find(inner, name)
        if found is not None:
            return found
    return None


class WalkUntilCoordTests(unittest.TestCase):
    def test_holds_up_from_town_space_door_coords(self) -> None:
        world = _shop_world(x=598, y=874)
        task = WalkUntilCoordTask(name="livestock_remap_up", max_x=400)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        pressed = {name.lower() for name in pressed_snes_buttons(result.action.action)}
        self.assertIn("up", pressed)
        self.assertIn("b", pressed)
        self.assertNotIn("left", pressed)

    def test_succeeds_once_interior_camera_remaps(self) -> None:
        world = _shop_world(x=128, y=200)
        task = WalkUntilCoordTask(name="livestock_remap_up", max_x=400)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("coord remap", result.reason or "")

    def test_times_out_when_camera_never_remaps(self) -> None:
        world = _shop_world(x=598, y=874)
        task = WalkUntilCoordTask(name="livestock_remap_up", max_x=400, timeout=3)
        task.reset(world)
        statuses = [task.step(world).status for _ in range(4)]
        self.assertEqual(statuses[-1], TaskStatus.FAILURE)
        self.assertTrue(all(status == TaskStatus.RUNNING for status in statuses[:-1]))

    def test_exit_remap_holds_down_until_town_space_x(self) -> None:
        world = _shop_world(x=127, y=212, tilemap=TOWN_TILEMAP)
        task = WalkUntilCoordTask(
            name="livestock_exit_remap",
            direction="down",
            tilemap=TOWN_TILEMAP,
            max_x=None,
            min_x=400,
        )
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        pressed = {name.lower() for name in pressed_snes_buttons(result.action.action)}
        self.assertIn("down", pressed)
        self.assertNotIn("left", pressed)


class LivestockSequenceTests(unittest.TestCase):
    def test_event_stand_is_north_of_buy_cow_counter(self) -> None:
        route = ROUTES["d1_livestock_to_event_stand"]
        self.assertEqual(route[-1].target_px, (230, 139))
        self.assertEqual(route[-1].run_direction, "down")
        self.assertTrue(all(wp.force_run for wp in route))
        self.assertNotIn((201, 157), [wp.target_px for wp in route])
        self.assertNotIn((201, 158), [wp.target_px for wp in route][-1:])

    def test_composed_handoff_talks_face_down_not_npc_track(self) -> None:
        seq = build_day1_handoff_tasks(
            include_sleep=False,
            require_full_mask=True,
            pick_starter_tools=False,
            use_rest_recording=False,
        )
        names = _names(seq)
        self.assertIn("livestock_remap_up", names)
        self.assertIn("nav_livestock_stand", names)
        self.assertIn("talk_livestock", names)
        self.assertIn("livestock_exit_north", names)
        self.assertIn("livestock_exit_to_counter_x", names)
        self.assertIn("livestock_exit_to_counter_row", names)
        self.assertIn("livestock_exit_west", names)
        self.assertIn("livestock_exit_south", names)
        self.assertIn("livestock_exit_remap", names)
        self.assertIn("livestock_off_door", names)
        self.assertNotIn("livestock_remap_left", names)
        self.assertNotIn("exit_livestock", names)

        talk = _find(seq, "talk_livestock")
        self.assertIsInstance(talk, PressAUntilBitOrTimeout)
        self.assertEqual(talk.bit, BIT_LIVESTOCK)
        self.assertEqual(talk.face, "down")
        self.assertNotIsInstance(talk, TrackNpcUntilBitTask)

        stand = _find(seq, "nav_livestock_stand")
        targets = [wp.target_px for wp in stand.waypoints]
        self.assertEqual(targets[-1], (230, 139))

        talk_ann = _find(seq, "talk_ann")
        talk_eve = _find(seq, "talk_eve")
        self.assertIsInstance(talk_ann, TrackNpcUntilBitTask)
        self.assertIsInstance(talk_eve, TrackNpcUntilBitTask)
        self.assertEqual(talk_ann.bit, BIT_ANN)
        self.assertEqual(talk_eve.bit, BIT_EVE)
        self.assertEqual(talk_ann.face_hint, "left")
        self.assertEqual(talk_eve.face_hint, "up")
        self.assertIn("owner_clear_door_remap", names)
        self.assertIn("nina_clear_door_remap", names)


class TrackNpcUntilBitTests(unittest.TestCase):
    def test_walks_toward_nearest_npc_when_out_of_range(self) -> None:
        world = _npc_world(player_xy=(100, 100), npc_xy=(200, 100))
        task = TrackNpcUntilBitTask(name="talk_ann", bit=BIT_ANN, face_hint="left")
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        pressed = {name.lower() for name in pressed_snes_buttons(result.action.action)}
        self.assertIn("right", pressed)
        self.assertNotIn("a", pressed)

    def test_presses_a_with_face_hint_when_in_range(self) -> None:
        world = _npc_world(player_xy=(100, 100), npc_xy=(120, 100))
        task = TrackNpcUntilBitTask(name="talk_ann", bit=BIT_ANN, face_hint="left")
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertIn("talk to moving NPC", result.reason or "")

    def test_succeeds_once_event_bit_is_set(self) -> None:
        world = _npc_world(player_xy=(100, 100), npc_xy=(120, 100), bit=BIT_ANN)
        task = TrackNpcUntilBitTask(name="talk_ann", bit=BIT_ANN, face_hint="left")
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.SUCCESS)
        self.assertIn("bit 0x01 set", result.reason or "")

    def test_idles_when_no_npc_object_is_present(self) -> None:
        world = _npc_world(player_xy=(100, 100), npc_xy=(120, 100))
        # Clear the NPC slot so only the player object remains.
        offset = GOBJ_STRUCT_BASE + GOBJ_STRUCT_STRIDE
        world.ram[offset : offset + 2] = 0
        task = TrackNpcUntilBitTask(name="talk_ann", bit=BIT_ANN)
        task.reset(world)
        result = task.step(world)
        self.assertEqual(result.status, TaskStatus.RUNNING)
        self.assertEqual(result.reason, "waiting for live NPC object")


if __name__ == "__main__":
    unittest.main()
