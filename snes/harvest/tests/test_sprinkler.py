"""The sprinkler replaces the watering can: nothing may ask for a can the owner no longer has."""
from __future__ import annotations

import numpy as np

from harvest.core.tile_catalog import Tool
from harvest.tasks import sprinkler as sp
from harvest.tasks.buy_seeds import TOOL_SHELVES


def test_can_and_sprinkler_are_one_shelf_slot():
    from harvest.planner.tasks.shed_slots import SHED_TOOL_SPECS

    assert SHED_TOOL_SPECS[int(Tool.SPRINKLER)].inside_stand_px == SHED_TOOL_SPECS[int(Tool.WATERING_CAN)].inside_stand_px


def owned_ram() -> np.ndarray:
    ram = np.zeros(0x20000, dtype=np.uint8)
    shelf = TOOL_SHELVES[int(Tool.SPRINKLER)]
    ram[shelf.shed_addr] |= shelf.shed_bit
    return ram


def test_a_request_for_the_can_fetches_the_sprinkler_once_it_is_owned():
    from types import SimpleNamespace

    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask

    task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))
    task.reset(SimpleNamespace(ram=owned_ram(), info={}, obs=None, frame=0))
    assert task.tool_id == int(Tool.SPRINKLER)


def test_without_a_sprinkler_the_can_stays_the_can():
    from types import SimpleNamespace

    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask

    task = EnsureCarryToolTask(tool_id=int(Tool.WATERING_CAN))
    task.reset(SimpleNamespace(ram=np.zeros(0x20000, dtype=np.uint8), info={}, obs=None, frame=0))
    assert task.tool_id == int(Tool.WATERING_CAN)


def test_a_stand_covering_one_dry_crop_is_a_stand(monkeypatch):
    class Finder:
        def find_path(self, _ram, _start, _stand):
            return []

    monkeypatch.setattr(sp, "Pathfinder", Finder)
    stand, covered = sp.sprinkler_stand(None, {(10, 10)}, (20, 20))
    assert covered == {(10, 10)} and max(abs(stand[0] - 10), abs(stand[1] - 10)) <= 1


def test_the_biggest_cluster_goes_first(monkeypatch):
    class Finder:
        def find_path(self, _ram, _start, _stand):
            return []

    monkeypatch.setattr(sp, "Pathfinder", Finder)
    dry = {(10, 10), (11, 10), (12, 10), (30, 30)}
    assert len(sp.sprinkler_stand(None, dry, (0, 0))[1]) == 3
