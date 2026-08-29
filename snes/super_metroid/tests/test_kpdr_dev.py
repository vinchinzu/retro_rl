"""Warehouse entry branch (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from super_metroid.ram import parse_state
from super_metroid.routes.kpdr.warehouse_stack import resolve_warehouse_entry_mode


def test_warehouse_entry_mode() -> None:
    base = parse_state(np.zeros(0x2000, dtype=np.uint8))
    assert resolve_warehouse_entry_mode(replace(base, samus_x=50)) == "left_elevator"
    assert (
        resolve_warehouse_entry_mode(replace(base, samus_x=500))
        == "right_reverse_stack"
    )
    assert (
        resolve_warehouse_entry_mode(
            replace(base, samus_x=500), entry_mode="left_elevator"
        )
        == "left_elevator"
    )
