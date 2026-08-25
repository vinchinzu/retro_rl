"""Level 6 0x3A stairs via the lane under the NE 0x68.

The blocked neclip reached (178,109), the west face of the live block at
(208,96).  Stop the proven east-aisle climb at y=132, traverse underneath,
then approach the shared south face (208,112) and hold UP onto tile 0x71.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action
from zelda_i.level6_path import PUSH_ALIGN_TOL, south_face_stand
from zelda_i.level6_stairs3a_neclip import (
    CLIP_START_X,
    EAST_DOOR_XMIN,
    EAST_DOOR_Y,
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    PASSAGE_MODE,
    PLAY_MODE,
    STAIRS_3A_NE71_MAX_FRAMES,
    Level6Stairs3ANEClipController,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

# v1: RIGHT at y=125 only moved 2px into the block's west collision face.
# y=132 is the last band above the guarded east-door range (starts at 133).
UNDER_BLOCK_Y = 132
NE_BLOCK_X = 208


@dataclass
class Level6Stairs3ANEUnderController(Level6Stairs3ANEClipController):
    """Climb at x=176, cross y=132, then approach the NE block from below."""

    spec_id: str = "level6_stairs_0x3a_neunder"

    def _under_block_action(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject
    ) -> FrameAction | None:
        tx, ty = south_face_stand(block)
        if self._at_south_face(xy, block):
            return None
        if xy[0] >= EAST_DOOR_XMIN and abs(xy[1] - EAST_DOOR_Y) <= 8:
            return self._fail(snap, f"east_door_{xy[0]}_{xy[1]}")
        if xy[0] < CLIP_START_X:
            return self._emit(
                snap, FrameAction(nes_action("RIGHT", "DOWN"), "ne_around_clip")
            )
        if xy[1] < UNDER_BLOCK_Y and xy[0] < tx - PUSH_ALIGN_TOL:
            return self._walk(snap, "DOWN", "ne_under_y")
        if xy[1] > UNDER_BLOCK_Y and xy[0] < tx - PUSH_ALIGN_TOL:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("UP"), "ne_north_aisle")
            )
        if abs(xy[0] - tx) > PUSH_ALIGN_TOL:
            if xy[0] < tx and xy[1] >= UNDER_BLOCK_Y:
                self.walker.last_dir = None
                return self._emit(
                    snap,
                    FrameAction(nes_action("RIGHT", "DOWN"), "ne_under_clip"),
                )
            return self._walk(snap, "RIGHT" if xy[0] < tx else "LEFT", "ne_under_x")
        return self._walk(snap, "UP" if xy[1] > ty else "DOWN", "ne_y")

    def _left_around(self, snap: ZeldaSnapshot, *, clip: bool) -> FrameAction:
        del clip
        block = self._find_block(snap, ne=True)
        if block is None:
            return self._fail(snap, "no_ne_0x68_under")
        return self._under_block_action(
            snap, (int(snap.link_x), int(snap.link_y)), block
        ) or self._walk(snap, "UP", "push_ne_block")

    def _axis_to_ne(
        self, snap: ZeldaSnapshot, xy: tuple[int, int], block: ZeldaObject
    ) -> FrameAction | None:
        return self._under_block_action(snap, xy, block)

    def report(self) -> dict[str, object]:
        report = super().report()
        report["policy"] = (
            "live center push; east aisle x=176; UP only to y=132; RIGHT "
            "RIGHT+DOWN around NE 0x68 southwest face; shared "
            "south_face_stand then UP to 0x71"
        )
        return report


def make_stairs_3a_neunder_controller() -> Level6Stairs3ANEUnderController:
    return Level6Stairs3ANEUnderController()


def level6_stairs3a_neunder_stages():
    stairs = make_stairs_3a_neunder_controller()
    return ((stairs.spec_id, stairs, STAIRS_3A_NE71_MAX_FRAMES),)


def level6_stairs3a_neunder_success(snap: ZeldaSnapshot) -> bool:
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
