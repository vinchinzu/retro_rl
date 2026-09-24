"""Scratch walk arms to secret-cave screens (measurement only; stage_replay targets)."""
from zelda_i.overworld.gather_segments import HopWalkController
from zelda_i.overworld.graph import ScreenHop


def to_62():
    return HopWalkController(
        hops=(ScreenHop(0x64, "LEFT", align_y=141), ScreenHop(0x63, "LEFT"), ScreenHop(0x62, "LEFT")),
        waypoints={}, max_frames=6000,
    )


def to_67():
    return HopWalkController(hops=(ScreenHop(0x67, "DOWN"),), waypoints={}, max_frames=4000)


def to_71_from_62():
    return HopWalkController(
        hops=(ScreenHop(0x72, "DOWN", align_x=104), ScreenHop(0x71, "LEFT", align_y=125)),
        waypoints={}, max_frames=8000,
    )


def to_3d_from_l2():
    return HopWalkController(hops=(
        ScreenHop(0x4C, "DOWN", align_x=112),
        ScreenHop(0x4D, "RIGHT", y_band_lo=133, y_band_hi=145),
        ScreenHop(0x3D, "UP", align_x=120),
    ), waypoints={}, max_frames=10000)


def back_3c_from_3d():
    return HopWalkController(hops=(ScreenHop(0x3C, "LEFT", align_y=141),), waypoints={}, max_frames=5000)


def exit_cave():
    from zelda_i.overworld.gather_segments import CaveExitController
    return CaveExitController(clear=0)


def back_4d_from_3d():
    return HopWalkController(hops=(ScreenHop(0x4D, "DOWN", align_x=120),), waypoints={}, max_frames=5000)


def level3_from_4d():
    from zelda_i.level3.overworld import LEVEL3_HOPS_FROM_POST_L2, OverworldPostL2ToLevel3Controller
    targets = [hop.target for hop in LEVEL3_HOPS_FROM_POST_L2]
    return OverworldPostL2ToLevel3Controller(
        hops=LEVEL3_HOPS_FROM_POST_L2[targets.index(0x4D) + 1 :],
        require_dungeon=True,
    )


def select_arrows():
    from zelda_i.dungeon.pause_select import B_SLOT_ARROWS, PauseSelectController
    return PauseSelectController(want=B_SLOT_ARROWS, name="arrows")


def to_67_66_65():
    return HopWalkController(
        hops=(ScreenHop(0x67, "DOWN"), ScreenHop(0x66, "LEFT"), ScreenHop(0x65, "LEFT")),
        waypoints={}, max_frames=6000,
    )


def row6_west():
    return HopWalkController(
        hops=tuple(ScreenHop(t, "LEFT") for t in (0x6A, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64)),
        waypoints={}, max_frames=12000,
    )


def to_6b_from_58():
    return HopWalkController(
        hops=(ScreenHop(0x59, "RIGHT", align_y=133), ScreenHop(0x5A, "RIGHT"), ScreenHop(0x5B, "RIGHT"),
              ScreenHop(0x6B, "DOWN", align_x=48)),
        waypoints={}, max_frames=8000,
    )


def rupees_2d():
    from zelda_i.overworld.gather_segments import RUPEES_2D_HOPS, make_secret_rupee_controller

    return make_secret_rupee_controller(0x2D, RUPEES_2D_HOPS)


def to_0f_from_1e():
    return HopWalkController(
        hops=(ScreenHop(0x1F, "RIGHT", align_y=141), ScreenHop(0x0F, "UP", align_x=128)),
        waypoints={}, max_frames=3000,
    )


def rupees_28():
    from zelda_i.overworld.gather_segments import make_secret_rupee_controller

    return make_secret_rupee_controller(0x28)


# --- Measurement probes (2026-09-23 secret-cave catalog) --------------------
# A probe runs one secret from a pin.  Teleport probes need stage_replay
# ``--set 0x00EB=<start> --set 0x0070=<x> --set 0x0084=141`` (a what-if RAM
# write: Link scrolls onto the target by the ROM's own room load).  Bombs or
# the candle are what-if writes too (``0x0658`` / ``0x065B``, B item
# ``0x0656`` 1 = bombs, 4 = candle).  Never a route result.
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

from zelda_i.dungeon.hop_controller import room_step  # noqa: E402
from zelda_i.dungeon.tilemap import tile_at_screen  # noqa: E402
from zelda_i.overworld.gather_segments import (  # noqa: E402
    BLAST_FRAMES,
    FLAME_WAIT,
    BombWallController,
    centre_item_walk,
    idle,
    push,
)
from zelda_i.overworld.locations import cave_keeper  # noqa: E402

_PAYOUT = {0x21: 30, 0x22: 100, 0x23: 10}


@dataclass
class SecretProbe:
    """Hold ``press`` until play on ``screen``, run ``inner``, log the cave."""

    screen: int
    inner: Any
    press: str | None = None
    max_frames: int = 6000
    notes: list = field(default_factory=list)
    _env: Any = None
    _arrived: bool = False
    _cave_frames: int = 0
    _start_rupees: int = -1
    _done_logged: bool = False

    def bind_env(self, env):
        self._env = env

    @property
    def success(self) -> bool:
        # Hold the stage open until the cave's wares are logged.
        logged = self._cave_frames == 0 or self._cave_frames > 40
        return bool(getattr(self.inner, "success", False)) and logged

    @property
    def failed(self) -> bool:
        return bool(getattr(self.inner, "failed", False))

    def report(self) -> dict:
        inner = list(getattr(self.inner, "notes", []) or [])
        return {"notes": inner[-6:] + self.notes}

    def _ram(self):
        return self._env.get_ram()

    def step(self, snap):
        ram = self._ram()
        if not self._arrived:
            if snap.screen == self.screen and snap.mode == 5:
                self._arrived = True
                o11 = [(hex(o.type_id), o.x, o.y) for o in snap.objects if o.slot == 11]
                self._start_rupees = int(snap.rupees) + int(snap.rupees_to_add)
                self.notes.append(
                    f"arrive 0x{self.screen:02x} ({snap.link_x},{snap.link_y}) "
                    f"flag={int(ram[0x067F + self.screen]):02x} o11={o11} R={self._start_rupees}"
                )
            elif self.press:
                return push(self.press, "teleport")
        if snap.mode == 11 and snap.level == 0:
            self._cave_frames += 1
            if self._cave_frames == 40:
                items = [hex(int(ram[0x0422 + i])) for i in range(3)]
                prices = [int(ram[0x0430 + i]) for i in range(3)]
                keeper = [hex(o.type_id) for o in snap.objects if o.type_id]
                self.notes.append(
                    f"cave 0x{self.screen:02x} items={items} prices={prices} objs={keeper} "
                    f"flag={int(ram[0x067F + self.screen]):02x} entered_at=({snap.link_x},{snap.link_y})"
                )
        if getattr(self.inner, "success", False):
            act = idle("probe_hold")
        else:
            act = self.inner.step(snap)
        if self.success and not self._done_logged:
            self._done_logged = True
            got = int(snap.rupees) + int(snap.rupees_to_add) - self._start_rupees
            self.notes.append(
                f"done credited_delta={got} flag={int(ram[0x067F + self.screen]):02x}"
            )
        return act


def _wall(screen: int, stand, face: str, *, bomb: bool, cave_id: int, hops=()) -> BombWallController:
    pay = _PAYOUT.get(cave_id, 0)
    return BombWallController(
        hops=tuple(hops),
        max_frames=5000,
        screen=screen,
        bomb_x=stand[0],
        bomb_y=stand[1],
        bomb_face=face,
        retreat="DOWN" if bomb else None,
        use_wait=BLAST_FRAMES if bomb else FLAME_WAIT,
        reward="rupees",
        reward_rupees=pay,
        keeper=cave_keeper(cave_id),
        interior_budget=900,
        bomb_fail=f"0x{screen:02x}_did_not_open",
        leave_fail=f"left_0x{screen:02x}",
        consumes_bomb=bomb,
    )


@dataclass
class ArmosTouch:
    """Walk to ``stand``, push ``face`` into the armos, walk onto its stairs, take the cave."""

    screen: int
    stand: tuple[int, int]
    face: str
    armos: tuple[int, int]
    cave_id: int
    max_frames: int = 5000
    success: bool = False
    failed: bool = False
    notes: list = field(default_factory=list)
    _env: Any = None
    _phase: str = "walk"
    _n: int = 0
    _cave_rupees: int = -1

    def bind_env(self, env):
        self._env = env

    def step(self, snap):
        self._n += 1
        ram = self._env.get_ram()
        if snap.mode == 11 and snap.level == 0:
            credited = int(snap.rupees) + int(snap.rupees_to_add)
            if self._cave_rupees < 0:
                self._cave_rupees = credited
            if credited >= self._cave_rupees + _PAYOUT.get(self.cave_id, 0) > self._cave_rupees:
                self.success = True
            return centre_item_walk(snap, cave_keeper(self.cave_id), "armos_cave")
        if snap.mode != 5:
            return idle("mode")
        ax, ay = self.armos
        stairs = tile_at_screen(ram, ax, ay) == 0x70
        if self._phase == "walk":
            step = room_step(snap, self.stand, tol=1, env=self._env)
            if step is not None:
                return push(step, "to_stand")
            self._phase = "touch"
        if self._phase == "touch":
            if stairs:
                self.notes.append(f"stairs at ({ax},{ay}) after touch f{self._n}")
                self._phase = "enter"
            elif self._n > 1500:
                self.failed = True
                return idle("no_stairs")
            else:
                return push(self.face, "touch_armos")
        step = room_step(snap, (ax, ay - 3), tol=0, env=self._env)
        if step is None:
            return push(self.face, "onto_stairs")
        return push(step, "to_stairs")


def probe_67():
    """Real hop 0x77 UP; bomb (112, 80) from (112, 85) UP.  --set 0x0658=1 --set 0x0656=1."""
    return SecretProbe(0x67, _wall(0x67, (112, 85), "UP", bomb=True, cave_id=0x21,
                                   hops=(ScreenHop(0x67, "UP", align_x=112),)))


def probe_71():
    """Teleport 0x72 LEFT; bomb (80, 80)."""
    return SecretProbe(0x71, _wall(0x71, (80, 85), "UP", bomb=True, cave_id=0x21), press="LEFT")


def probe_13():
    """Teleport 0x14 LEFT; bomb (32, 80)."""
    return SecretProbe(0x13, _wall(0x13, (32, 85), "UP", bomb=True, cave_id=0x21), press="LEFT")


def probe_51():
    """Teleport 0x52 LEFT; burn (144, 160) from (144, 141) DOWN."""
    return SecretProbe(0x51, _wall(0x51, (144, 141), "DOWN", bomb=False, cave_id=0x23), press="LEFT")


def probe_3d():
    """BFS_3D pin; armos (144, 128) touched from (128, 125) RIGHT."""
    return SecretProbe(0x3D, ArmosTouch(0x3D, (128, 125), "RIGHT", (144, 128), 0x21))


def armos_3d():
    return ArmosTouch(0x3D, (128, 125), "RIGHT", (144, 128), 0x21)


def probe_4e():
    """Teleport 0x4D RIGHT (x=240); armos (160, 128) touched from (144, 125) RIGHT."""
    return SecretProbe(0x4E, ArmosTouch(0x4E, (144, 125), "RIGHT", (160, 128), 0x23), press="RIGHT")


def _shop(screen: int, stand, face: str, *, bomb: bool, hops=()) -> BombWallController:
    ctl = _wall(screen, stand, face, bomb=bomb, cave_id=0x1A, hops=hops)
    ctl.interior_budget = 120
    return ctl


def probe_78():
    """Real hop 0x77 RIGHT; burn (64, 160) from (64, 141) DOWN."""
    return SecretProbe(0x78, _shop(0x78, (64, 141), "DOWN", bomb=False,
                                   hops=(ScreenHop(0x78, "RIGHT", align_y=141),)))


def probe_4b():
    """At4B pin (0, 141); burn (176, 96) from (152, 93) RIGHT."""
    return SecretProbe(0x4B, _shop(0x4B, (152, 93), "RIGHT", bomb=False))


def probe_0d():
    """Teleport 0x0C RIGHT (x=240); bomb (144, 80)."""
    return SecretProbe(0x0D, _shop(0x0D, (144, 85), "UP", bomb=True), press="RIGHT")


def probe_27():
    """Teleport 0x28 LEFT; bomb (224, 80)."""
    return SecretProbe(0x27, _shop(0x27, (224, 85), "UP", bomb=True), press="LEFT")


def probe_33():
    """Teleport 0x32 RIGHT (x=240); bomb (160, 80)."""
    return SecretProbe(0x33, _shop(0x33, (160, 85), "UP", bomb=True), press="RIGHT")


def probe_64():
    """BFS_65 pin; hop LEFT into 0x64; open doorway (112, 80) entered UP from (112, 93)."""
    return SecretProbe(0x64, _shop(0x64, (112, 93), "UP", bomb=False,
                                   hops=(ScreenHop(0x64, "LEFT", align_y=141),)))


def probe_04():
    """Teleport 0x05 LEFT; open doorway (192, 80) entered UP from (192, 93)."""
    return SecretProbe(0x04, _shop(0x04, (192, 93), "UP", bomb=False), press="LEFT")


def probe_4b_144():
    """At4B pin (0, 141); burn (176, 96) from (144, 93) RIGHT (flame crosses the wall block?)."""
    return SecretProbe(0x4B, _shop(0x4B, (144, 93), "RIGHT", bomb=False))


def probe_4b_east():
    """Teleport 0x3B DOWN (x=200, y=221); burn (176, 96) from (208, 93) LEFT in the east corridor."""
    return SecretProbe(0x4B, _shop(0x4B, (208, 93), "LEFT", bomb=False), press="DOWN")


def rupees_56():
    from zelda_i.overworld.gather_segments import RUPEES_56_HOPS, make_secret_rupee_controller

    return make_secret_rupee_controller(0x56, RUPEES_56_HOPS, 10000)


def from_62_south():
    return HopWalkController(
        hops=(ScreenHop(0x72, "DOWN", align_x=112), ScreenHop(0x73, "RIGHT"), ScreenHop(0x74, "RIGHT"),
              ScreenHop(0x64, "UP")),
        waypoints={}, max_frames=8000,
    )


def from_62_north():
    return HopWalkController(
        hops=(ScreenHop(0x52, "UP", align_x=96), ScreenHop(0x53, "RIGHT"), ScreenHop(0x54, "RIGHT")),
        waypoints={}, max_frames=8000,
    )


def _secret(screen, hops_name=None, frames=3000):
    from zelda_i.overworld import gather_segments as g

    hops = getattr(g, hops_name) if hops_name else ()
    return g.make_secret_rupee_controller(screen, hops, frames)


def rupees_5b():
    return _secret(0x5B, "RUPEES_5B_HOPS", 8000)


def rupees_6b():
    return _secret(0x6B, "RUPEES_6B_HOPS", 4000)


def rupees_62():
    return _secret(0x62, "RUPEES_62_HOPS", 8000)


def potion_from_62():
    from zelda_i.overworld.cave_shop import make_potion_buy_controller
    from zelda_i.overworld.gather_segments import POTION_FROM_62_HOPS

    return make_potion_buy_controller(hops=POTION_FROM_62_HOPS)


def l2_entry():
    from zelda_i.level2.overworld import OverworldToLevel2Controller

    return OverworldToLevel2Controller(door_path=True, require_dungeon=True)


def l4_stepladder():
    from zelda_i.level4.stepladder import make_stepladder_controller

    return make_stepladder_controller(clear_first=False)


def walk_pond():
    from zelda_i.overworld.gather_segments import POND_WALK_HOPS

    return HopWalkController(hops=POND_WALK_HOPS)


def l8_clear_1f():
    from zelda_i.dungeon.engine import DungeonPhase, GenericDungeonRoomController
    from zelda_i.level8.magic_key import _clear_1f_spec

    ctl = GenericDungeonRoomController(_clear_1f_spec())
    ctl.phase = DungeonPhase.FIGHT
    return ctl


def l7_pond_approach():
    from zelda_i.level7.hops import POND_APPROACH_MAX_FRAMES
    from zelda_i.level7.overworld import WARP_JOIN_TO_POND_HOPS, OverworldToLevel7PondController

    return OverworldToLevel7PondController(
        hops=WARP_JOIN_TO_POND_HOPS, max_frames=POND_APPROACH_MAX_FRAMES, resume_on_screen=True
    )
