"""Level 6 0x3A center push, then NE clip to the proven 0x71 warp.

This retarget freezes the blocked west-around controller.  Room 0x09 proves
the NE 0x68 at (208,96) warps when approached from its south face.  In 0x3A,
stay west of the open east door until a RIGHT+UP clip has cleared y=133.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.level6_stairs3a_ne71 import (
    EAST_DOOR_XMIN,
    EAST_DOOR_Y,
    PASSAGE_MODE,
    PLAY_MODE,
    PUSH_ALIGN_TOL,
    STAIRS_3A_NE71_MAX_FRAMES,
    Level6Stairs3ANE71Controller,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

CLIP_START_X = 176
DOOR_CLEAR_Y = 132
NE_ALIGN_Y = 116


@dataclass
class Level6Stairs3ANEClipController(Level6Stairs3ANE71Controller):
    """Reuse the live push prefix; take the east aisle instead of turning west."""

    spec_id: str = "level6_stairs_0x3a_neclip"

    def _mark_around(self, xy: tuple[int, int], _tile: int) -> None:
        if xy[0] >= CLIP_START_X:
            self.passed_around = True

    def _around_corridor(self, xy: tuple[int, int]) -> bool:
        return not self.passed_around and xy[1] >= 147 and xy[0] < CLIP_START_X

    def _left_around(self, snap: ZeldaSnapshot, *, clip: bool) -> FrameAction:
        del clip
        xy = (int(snap.link_x), int(snap.link_y))
        self.walker.last_dir = None
        if xy[0] >= EAST_DOOR_XMIN and xy[1] > DOOR_CLEAR_Y:
            return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
        self.walker.last_dir = None
        return self._emit(snap, FrameAction(nes_action("UP"), "ne_north_aisle"))

    def _axis_to_ne(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject
    ) -> FrameAction | None:
        tx, ty = int(block.x), int(block.y) + 18
        if self._at_south_face(xy, block):
            return None
        if xy[0] >= EAST_DOOR_XMIN and abs(xy[1] - EAST_DOOR_Y) <= 8:
            return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
        if xy[1] > NE_ALIGN_Y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("UP"), "ne_north_aisle")
            )
        if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
            return self._walk(snap, "RIGHT" if xy[0] < tx else "LEFT", "ne_x")
        return self._walk(snap, "UP" if xy[1] > ty else "DOWN", "ne_y")

    def report(self) -> dict[str, object]:
        report = super().report()
        report["policy"] = (
            "live center push; RIGHT+DOWN through x=160 to x>=176; "
            "UP at x>=176 above east-door band; RIGHT on north band to x=208; "
            "UP onto tile 0x71; halt first new miss"
        )
        return report


def make_stairs_3a_neclip_controller() -> Level6Stairs3ANEClipController:
    return Level6Stairs3ANEClipController()


def level6_stairs3a_neclip_stages():
    stairs = make_stairs_3a_neclip_controller()
    return ((stairs.spec_id, stairs, STAIRS_3A_NE71_MAX_FRAMES),)


def level6_stairs3a_neclip_success(snap: ZeldaSnapshot) -> bool:
    if snap.level != LEVEL6 or snap.triforce != 0x1F or int(snap.rod) == 0:
        return False
    if snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_BLOCK_3A_ROOM
        and snap.screen not in (0x29, 0x09, 0x39, 0x3B)
    )
