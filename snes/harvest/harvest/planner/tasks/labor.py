"""The child-birth morning: villagers fill the house and one stands in the doorway.

ROM facts (probed on the e2e pin ``PR_e2ev1n_Y2_Winter_D24``, ``kid1_age`` 60, ``$7F1F6E & 0x10``):
the house holds the wife in her bed at (281,99) and three more people, one of them (sprite 0x2A7)
standing at (128,200) in the middle of the doorway, which spans x=113..143. The farmer is free
(``input_lock`` 1) but nothing fits past her, the bed and every route out are blocked, and the
house clock does not tick indoors, so waiting never clears it: the power-on run ended there with
8 sleep rescues. Talking to the doorway person only repeats "You are going to have a baby
pretty soon" and she stays. What clears it is the wife: from her east side (296,102), facing
left, A opens "I'm OK, honey. Don't worry. Please go to work." with the first choice "Really?"
already under the cursor; the next text is "I'll do alright." and the doorway person walks
away (sprite 0x2A5 at (96,176)). A further A on the wife only repeats her labour text.

Choosing "Really?" sends the farmer to work, and the baby is only born when he comes back; the
crowd (a villager on the bed stand's south side at (296,120), another east of it at (312,104),
the doctor in the doorway again) then stays for good and the house clock does not tick, so he
can neither sleep nor leave: the second e2e wedge. Choosing the second row, "No, no, no", keeps
him in the room: about 3,800 idle frames later the "Waah-waah" cutscene plays, the baby appears
in the crib at (264,96) and the farmer stands at the door (136,184). The bed's east stand
(294,106) is then sealed by a villager at (312,104) and another at (296,120). The east one walks
off after one chat (from tile (20,6), facing left); then the wife answers from the east stand:
"Darling, please name him." opens the name grid (tilemap 0x5F, answered by ``name_entry_press``),
then "Thank you, honey" and the birth announcement, and the room empties: only the wife, up and
standing at (328,120), and the baby remain, with the doorway and the bed free.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

from harvest.core.npc_catalog import iter_game_objects
from harvest.core.ram_catalog import read_ram_value
from harvest.tasks.nav import Navigator, Pathfinder, get_pos_from_ram, make_action

Tile = Tuple[int, int]

HOUSE_TILEMAPS = (0x15, 0x16, 0x17)
# The doorway of every house level: x=113..143 at the south wall, people stand just inside it.
DOOR_BOX = (100, 156, 176, 216)  # x0, x1, y0, y1
WIFE_BED_PX = (281, 99)
BABY_CRIB_PX = (264, 96)
# After the birth a villager stands at (312,104), east of the wife's stand, and another at
# (296,120), south of it: the stand is sealed. Talking to the east one ("Ho-ho, look at this
# baby") makes him walk away east (to (360,104)); he is reached from tile (20,6), facing left.
BLOCKER_PX = (312, 104)
BLOCKER_STAND: Tile = (20, 6)
WAIT_BIRTH_FRAMES = 6500
MENU_LIVE_BIT = 0x10  # $019B
WIFE_RADIUS = 24
# Stand east of the bed, facing left: (296,102) in the probe.
STAND_OFFSET_PX = (15, 3)
ON_STAND_FRAMES = 6
# The house exit is a scripted walk from the wake-up spot (136,120); from the wife's side it
# pushes into the wall (3,000 frames in the probe), so the task walks back there when done.
WAKE_TILE: Tile = (8, 7)
FACE_FRAMES = 8
TALK_PULSE = (2, 12)  # frames of A, frames of rest
TALKS = 3
NORTH_TALKS = 3


def _field(ram: np.ndarray, key: str) -> int:
    return int(read_ram_value(ram, key, raw=True))


def people_in_doorway(ram: np.ndarray) -> bool:
    """A non-player object inside the doorway box of the house."""
    if _field(ram, "tilemap") not in HOUSE_TILEMAPS:
        return False
    x0, x1, y0, y1 = DOOR_BOX
    return any(
        not obj.is_player and x0 <= obj.pixel[0] <= x1 and y0 <= obj.pixel[1] <= y1
        for obj in iter_game_objects(ram)
    )


def wife_in_bed(ram: np.ndarray) -> Optional[Tuple[int, int]]:
    """Pixel of the object lying where the wife's bed is, or None."""
    best = None
    for obj in iter_game_objects(ram):
        if obj.is_player:
            continue
        dist = abs(obj.pixel[0] - WIFE_BED_PX[0]) + abs(obj.pixel[1] - WIFE_BED_PX[1])
        if dist <= WIFE_RADIUS and (best is None or dist < best[0]):
            best = (dist, obj.pixel)
    return None if best is None else best[1]


# Where the villagers stand around the bed stand (294,102): south of it (296,120) and east (312,104).
BED_CROWD_BOX = (286, 320, 100, 132)  # x0, x1, y0, y1; the wife (281,99) is just west, and later stands at (328,120)


def bed_crowded(ram: np.ndarray) -> bool:
    """People stand where the farmer would stand to sleep (the baby-day crowd): bed A is out of reach."""
    if _field(ram, "tilemap") != 0x17:
        return False
    x0, x1, y0, y1 = BED_CROWD_BOX
    return any(
        not obj.is_player and x0 <= obj.pixel[0] <= x1 and y0 <= obj.pixel[1] <= y1
        for obj in iter_game_objects(ram)
    )


def baby_born(ram: np.ndarray) -> bool:
    """The baby lies in the crib: the birth cutscene has played."""
    return any(
        not obj.is_player and abs(obj.pixel[0] - BABY_CRIB_PX[0]) <= 16 and abs(obj.pixel[1] - BABY_CRIB_PX[1]) <= 16
        for obj in iter_game_objects(ram)
    )


def blocker_present(ram: np.ndarray) -> bool:
    """The east villager still stands at (312,104)."""
    return any(
        not obj.is_player and abs(obj.pixel[0] - BLOCKER_PX[0]) <= 10 and abs(obj.pixel[1] - BLOCKER_PX[1]) <= 10
        for obj in iter_game_objects(ram)
    )


def menu_live(ram: np.ndarray) -> bool:
    from harvest.core.ram_catalog import live_wram_base

    idx = 0x019B + live_wram_base(ram)
    return idx < len(ram) and bool(int(ram[idx]) & MENU_LIVE_BIT)


def labor_morning_due(ram: np.ndarray) -> bool:
    """The farmer is in the house and somebody stands in the doorway with the wife in bed."""
    return people_in_doorway(ram) and wife_in_bed(ram) is not None


class LaborMorningTask(Task):
    """See the module docstring: talk the wife through the birth, name the baby, empty the room.

    Stages: ``walk_east`` -> ``face_east`` -> ``press_east`` (the labour talk; the second row of
    its menu) -> ``wait_birth`` (idle until the baby is in the crib) -> ``walk_b`` -> ``face_b``
    -> ``press_b`` (chat with the east villager until he leaves) -> ``walk_wife`` -> ``face_wife``
    -> ``press_wife`` (the naming talk) -> ``return`` (back to the wake spot, where the scripted
    house exit starts). A baby already in the crib starts at ``walk_b``.
    """

    name = "labor_morning"

    def __init__(self, timeout: int = 16000) -> None:
        self.timeout = timeout
        self._pathfinder = Pathfinder(None)
        self._navigator = Navigator(self._pathfinder)
        self._reset()

    def _reset(self) -> None:
        self._step = 0
        self._stage = "walk_east"
        self._on_stand = 0
        self._hold = 0
        self._talks = 0
        self._stall = 0
        self._pulse = 0
        self._menu_frames = 0
        self._chosen = False
        self._extra_a = 0
        self._waited = 0

    def reset(self, world: WorldState) -> None:
        self._reset()
        self._pathfinder = Pathfinder(None)
        self._navigator = Navigator(self._pathfinder)
        if baby_born(world.ram):
            self._stage = "walk_wife" if not blocker_present(world.ram) else "walk_b"

    def can_start(self, world: WorldState) -> bool:
        return labor_morning_due(world.ram)

    @staticmethod
    def _running(action: np.ndarray, reason: str = "") -> TaskResult:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason=reason)

    def step(self, world: WorldState) -> TaskResult:
        ram = world.ram
        self._step += 1
        if self._step > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"labor morning: timeout in {self._stage}")
        self._navigator.update(ram)
        if _field(ram, "tilemap") not in HOUSE_TILEMAPS:
            # The name grid (tilemap 0x5F) or a fade: the caller's name handler owns the buttons.
            return self._running(make_action(), "labor morning: waiting out a scene")
        if _field(ram, "input_lock") != 1:
            return self._dialogue(ram)
        self._menu_frames = 0
        if self._stage == "return":
            return self._walk_back(ram)
        if not people_in_doorway(ram) and not bed_crowded(ram) and self._stage in ("walk_east", "return"):
            return TaskResult(status=TaskStatus.SUCCESS, reason="no_work: nobody in the doorway")
        if self._stage in ("face_wife", "press_wife") and not people_in_doorway(ram) and not bed_crowded(ram):
            self._stage, self._on_stand, self._stall = "return", 0, 0  # named: the room has emptied
            self._navigator.path = []
            return self._walk_back(ram)
        if baby_born(ram) and self._stage in ("walk_east", "face_east", "press_east", "wait_birth"):
            self._stage = "walk_wife" if not blocker_present(ram) else "walk_b"
            self._on_stand, self._hold = 0, 0
            self._navigator.path = []
        if self._stage == "wait_birth":
            self._waited += 1
            if self._waited > WAIT_BIRTH_FRAMES:
                return TaskResult(status=TaskStatus.FAILURE, reason="labor morning: the baby never came")
            return self._running(make_action(), "labor morning: waiting for the birth")
        wife = wife_in_bed(ram)
        if wife is None and self._stage in ("walk_east", "face_east", "press_east"):
            return TaskResult(status=TaskStatus.FAILURE, reason="labor morning: no wife in bed")
        if self._stage == "walk_east":
            return self._walk_to(ram, self._east_stand(wife), "face_east")
        if self._stage == "face_east":
            return self._face("left", "press_east")
        if self._stage == "press_east":
            return self._press("face_east", self._count_talk)
        if self._stage == "walk_b":
            if not blocker_present(ram):
                self._stage, self._on_stand = "walk_wife", 0
                self._navigator.path = []
                return self._running(make_action())
            return self._walk_to(ram, BLOCKER_STAND, "face_b", avoid_objects=True)
        if self._stage == "face_b":
            return self._face("left", "press_b")
        if self._stage == "press_b":
            if not blocker_present(ram):
                self._stage, self._on_stand = "walk_wife", 0
                self._navigator.path = []
                return self._running(make_action())
            if self._talks >= TALKS + NORTH_TALKS:
                return TaskResult(status=TaskStatus.FAILURE, reason="labor morning: the villager would not move")
            return self._press("face_b", self._count_talk, wait=200)
        if self._stage == "walk_wife":
            if wife is None:
                return TaskResult(status=TaskStatus.FAILURE, reason="labor morning: no wife in bed")
            return self._walk_to(ram, self._east_stand(wife), "face_wife", avoid_objects=True)
        if self._stage == "face_wife":
            return self._face("left", "press_wife")
        # press_wife: one A; the naming dialogue then takes over through ``_dialogue``
        if self._talks >= 2 * (TALKS + NORTH_TALKS):
            return TaskResult(status=TaskStatus.FAILURE, reason="labor morning: the wife would not take the name")
        return self._press("face_wife", self._count_talk, wait=240)

    # --- pieces ----------------------------------------------------------------

    def _count_talk(self) -> None:
        self._talks += 1

    def _east_stand(self, wife: Tuple[int, int]) -> Tile:
        return ((wife[0] + STAND_OFFSET_PX[0]) // 16, (wife[1] + STAND_OFFSET_PX[1]) // 16)

    def _dialogue(self, ram: np.ndarray) -> TaskResult:
        """A text box is up. The labour menu wants its second row; everything else wants A pulses."""
        self._pulse += 1
        if self._extra_a > 0:
            self._extra_a -= 1
            return self._running(make_action(a=True), "labor morning: confirm")
        if self._stage in ("press_east", "face_east", "walk_east") and menu_live(ram):
            self._menu_frames += 1
            if not self._chosen:
                if self._menu_frames == 2:
                    return self._running(make_action(down=True), "labor morning: second row")
                if self._menu_frames == 12:
                    # The first A takes the row and the menu goes dead at once, so book it now.
                    self._chosen, self._extra_a = True, 1
                    self._stage = "wait_birth"
                    return self._running(make_action(a=True), "labor morning: confirm")
                return self._running(make_action())
        if self._stage == "press_east" and not self._chosen:
            # Still printing the wife's line: wait for the menu to go live.
            return self._running(make_action(a=True) if self._pulse % 20 < 2 else make_action())
        if self._stage == "press_east":
            self._stage = "wait_birth"
        pulse = make_action(a=True) if self._pulse % sum(TALK_PULSE) < TALK_PULSE[0] else make_action()
        return self._running(pulse, "labor morning: dismiss")

    def _face(self, direction: str, nxt: str) -> TaskResult:
        self._hold += 1
        if self._hold <= FACE_FRAMES:
            return self._running(make_action(**{direction: True}))
        self._stage, self._hold = nxt, 0
        return self._running(make_action())

    def _press(self, back: str, on_press, wait: int = 80) -> TaskResult:
        self._hold += 1
        if self._hold == 1:
            on_press()
        if self._hold <= 3:
            return self._running(make_action(a=True))
        if self._hold > wait:  # nothing opened: face her again
            self._stage, self._hold = back, 0
        return self._running(make_action())

    def _walk_to(self, ram: np.ndarray, stand: Tile, nxt: str, *, avoid_objects: bool = False) -> TaskResult:
        here = self._navigator.current_tile
        self._on_stand = self._on_stand + 1 if here == stand else 0
        if self._on_stand >= ON_STAND_FRAMES:
            self._stage, self._hold = nxt, 0
            return self._running(make_action())
        if avoid_objects:
            # People are sprites, not tiles: keep the path off the ones standing in the way.
            self._pathfinder.temp_blocked = {
                (obj.pixel[0] // 16, obj.pixel[1] // 16) for obj in iter_game_objects(ram) if not obj.is_player
            } - {stand}
        if not self._navigator.path or self._navigator.stasis > 90:
            path = self._pathfinder.find_path(ram, here, stand)
            if path is None:
                self._stall += 1
                if self._stall > 300:
                    return TaskResult(status=TaskStatus.FAILURE, reason=f"labor morning: no path to {stand}")
                return self._running(make_action())
            self._navigator.path = path
            self._navigator.stasis = 0
        action = self._navigator.follow_path(ram)
        if action is None:
            action = self._navigator.center_on_tile(stand, tolerance=1)
        return self._running(action if action is not None else make_action())

    def _walk_back(self, ram: np.ndarray) -> TaskResult:
        here = self._navigator.current_tile
        self._on_stand = self._on_stand + 1 if here == WAKE_TILE else 0
        if self._on_stand >= ON_STAND_FRAMES:
            return TaskResult(status=TaskStatus.SUCCESS, reason="labor morning: the room is clear, back at the wake spot")
        self._pathfinder.temp_blocked = set()
        return self._walk_step(ram, WAKE_TILE)

    def _walk_step(self, ram: np.ndarray, goal: Tile) -> TaskResult:
        here = self._navigator.current_tile
        if not self._navigator.path or self._navigator.stasis > 90:
            path = self._pathfinder.find_path(ram, here, goal)
            if path is None:
                self._stall += 1
                if self._stall > 300:
                    return TaskResult(status=TaskStatus.FAILURE, reason=f"labor morning: no path to {goal}")
                return self._running(make_action())
            self._navigator.path = path
            self._navigator.stasis = 0
        action = self._navigator.follow_path(ram)
        if action is None:
            action = self._navigator.center_on_tile(goal, tolerance=1)
        return self._running(action if action is not None else make_action())


__all__ = [
    "LaborMorningTask", "baby_born", "bed_crowded", "blocker_present", "labor_morning_due", "menu_live",
    "people_in_doorway", "wife_in_bed",
]
