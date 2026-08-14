from pathlib import Path

import numpy as np
from PIL import Image

from zelda_i.tas.trace_route import (
    AtlasFrame,
    RouteEvent,
    action_array,
    build_contact_sheet,
    event_label,
)


def _event() -> RouteEvent:
    return RouteEvent(
        frame=123,
        kind="room_enter",
        detail="L5:0x76 -> L5:0x66",
        level=5,
        room=0x66,
        x=120,
        y=205,
        health=0x6F,
        keys=1,
        bombs=8,
        triforce=0x0F,
        doors=0x08,
        room_item=0x19,
    )


def test_action_array_pads_and_preserves_opposites() -> None:
    action = action_array([0, 0, 0, 0, 0, 1, 1, 0, 1])
    assert action.dtype == np.int8
    assert action.shape == (9,)
    assert action.tolist()[5:9] == [1, 1, 0, 1]


def test_event_label_contains_ram_pin() -> None:
    label = event_label(_event())
    assert "f123 room_enter L5:0x66" in label
    assert "hp=0x6f" in label
    assert "tf=0x0f" in label


def test_build_contact_sheet_writes_scaled_grid(tmp_path: Path) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 1] = 200
    frames = [
        AtlasFrame(event_index=i, label=f"event {i}", image=image) for i in range(4)
    ]
    output = tmp_path / "sheet.png"
    assert build_contact_sheet(frames, output, columns=2, scale=2) == output
    with Image.open(output) as sheet:
        assert sheet.size == (24, 72)
