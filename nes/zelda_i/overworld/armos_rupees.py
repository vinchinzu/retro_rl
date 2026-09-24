"""Natural rupee pickup from an Armos cave (0x3D 30R, 0x4E 10R)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from zelda_i.dungeon.hop_controller import room_step
from zelda_i.dungeon.tilemap import tile_at_screen
from zelda_i.overworld.gather_segments import centre_item_walk, idle, push
from zelda_i.overworld.locations import cave_keeper
from zelda_i.ram import CAVE_MODE, PLAY_MODE, ZeldaSnapshot

ARMOS_3D_SCREEN = 0x3D
ARMOS_3D_STAND = (128, 125)
ARMOS_3D_TILE = (144, 128)
ARMOS_3D_CAVE_ID = 0x21
ARMOS_3D_PAYOUT = 30

ARMOS_4E_SCREEN = 0x4E
ARMOS_4E_STAND = (144, 125)
ARMOS_4E_TILE = (160, 128)
ARMOS_4E_CAVE_ID = 0x23
ARMOS_4E_PAYOUT = 10


@dataclass
class ArmosRupeeController:
    """Touch an Armos, enter its revealed stair, and take rupees in play."""

    screen: int = ARMOS_3D_SCREEN
    stand: tuple[int, int] = ARMOS_3D_STAND
    face: str = "RIGHT"
    armos: tuple[int, int] = ARMOS_3D_TILE
    cave_id: int = ARMOS_3D_CAVE_ID
    payout: int = ARMOS_3D_PAYOUT
    max_frames: int = 5000
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    _env: Any = None
    _phase: str = "walk"
    _frames: int = 0
    _cave_rupees: int = -1

    def bind_env(self, env: Any) -> None:
        self._env = env

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self._frames += 1
        if snap.mode == CAVE_MODE and snap.level == 0:
            credited = int(snap.rupees) + int(snap.rupees_to_add)
            if self._cave_rupees < 0:
                self._cave_rupees = credited
            if credited >= self._cave_rupees + self.payout:
                self.success = True
                self.notes.append(f"armos_0x{self.screen:02x}_rupees_{self._cave_rupees}_to_{credited}")
            return centre_item_walk(snap, cave_keeper(self.cave_id), "armos_cave")
        if snap.mode != PLAY_MODE:
            return idle("armos_wait_mode")
        if snap.screen != self.screen:
            self.failed = True
            self.notes.append(f"wrong_screen_0x{snap.screen:02x}")
            return idle("armos_wrong_screen")
        if self._phase == "walk":
            direction = room_step(snap, self.stand, tol=1, env=self._env)
            if direction is not None:
                return push(direction, "armos_stand")
            self._phase = "touch"
        if self._phase == "touch":
            if tile_at_screen(self._env.get_ram(), *self.armos) == 0x70:
                self.notes.append(f"stairs_revealed_f{self._frames}")
                self._phase = "enter"
            elif self._frames > 1500:
                self.failed = True
                self.notes.append("armos_stairs_not_revealed")
                return idle("armos_no_stairs")
            else:
                return push(self.face, "armos_touch")
        direction = room_step(
            snap, (self.armos[0], self.armos[1] - 3), tol=0, env=self._env
        )
        return push(self.face, "armos_enter") if direction is None else push(direction, "armos_stairs")

    def report(self) -> dict[str, Any]:
        return {"success": self.success, "notes": list(self.notes)}
