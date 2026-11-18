"""Native power berries: acquire, eat, and require the persistent RAM receipt.

The ground sprite is $0116 (group 0/script $1F, Event_UNK3). Its A
interaction grants item 8. Eating finishes about 200 frames later; neither
the spawn flag nor the empty hand proves that it was eaten.
"""
import os
from dataclasses import dataclass, field

from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState
from harvest.core.animal_status import read_held_item
from harvest.core.npc_catalog import game_objects
from harvest.core.ram_catalog import read_ram_value, read_ram_u16, read_ram_u8
from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot
from harvest.planner.day_plan_status import is_house_tilemap
from harvest.tasks.nav import CENTER_STALL_FRAMES, get_pos_from_ram, get_tile_at, make_action

POWER_BERRY_SPRITE = 0x0116
BERRY_TARGET = 10
FIELD_BITS = {"field_1": 0x0800, "field_2": 0x1000}
# $7F1F64 bit 0x0002 is set the morning weather index 4 applies (bank_82 WeatherTomorrow ``.specialdays``:
# Year 1, Winter, tomorrow == day 8). ReplaceTilesMoutain ($81AC7F) then rewrites the 2x2 boulder at tile
# (23,39). Before that the west pocket holding the special stump (10,38), the goddess pond and the cave
# drop is sealed: hammer, axe and every height under the lip fail (measured 2026-11-02).
BOULDER_OPEN_FLAG = 0x0002
BOULDER_FLAG_ADDR = 0x11F64
# The Harvest and Egg Festival maps close at 15:00 (``$83D304``: the script loads only while the hour is below 0x0F).
# Native camp-to-gate replay takes ~1.2 hours. Keep fishing through 13:xx;
# the entry itself is viable until the ROM's actual 15:00 closing time.
FESTIVAL_GIVE_UP_HOURS = {("harvest_festival", "source"): 14, ("harvest_festival", "festival"): 15,
                          ("egg_festival", "entry"): 14, ("egg_festival", "festival"): 15}
# ``player_action`` while a rod is out (ROM probe, LN_fish_stuck_20261115_FAIL): 17 waits for a bite and ignores every button
# but Y (a Y there only recasts), 18 is the bite, 19 the reel; 16 is the cast and 15 the rest after a reel or the first six
# cast frames, and walking works from 15. A farmer who leaves the task at 17 is pinned for good: the exit walk never presses Y.
FISHING_PINNED_ACTIONS = frozenset({16, 17, 18, 19})
FISHING_BITE_ACTION = 18
FISHING_LEAVE_FRAMES = 4000
FISHING_SOURCES = frozenset({"fish", "harvest_festival"})
# The camp fence around the cast stand (633,371) opens only west, x=585, below y=390 (``FishBerryTask`` "leave_camp"); the
# generic mountain exit route heads north from inside and never finds it (fdet_20261115: 11,001 f swinging between y=342 and 393).
CAMP_GAP_X, CAMP_GAP_ROW_Y, CAMP_FLOOR_Y = 585, 390, 422
CAMP_BOX = (CAMP_GAP_X, 720, 300, CAMP_FLOOR_Y)  # x0 (exclusive), x1, y0, y1


def fishing_exit_pad(ram, age: int = 0):
    """The pad that gets the farmer off a cast rod, or None once he is free to walk.

    Wait through a cast line, reel the bite with one Y, never recast. fdet_20261115 gave up the Harvest Festival fish at
    14:00 on a cast line and ended all eight rescues at (633,371): the run was terminal.
    """
    action = int(read_ram_value(ram, "player_action"))
    if action not in FISHING_PINNED_ACTIONS:
        return None
    return make_action(y=action == FISHING_BITE_ACTION and age % 6 == 1)


def camp_exit_pad(tilemap: int, x: int, y: int, through_gap: bool = True):
    """Down to the gap row, left along it, then down through the gap: the way out of the fishing camp, or None outside it.

    ``through_gap=False`` stops at the gap column (a walker that passes it on its own route must not be steered).
    """
    if int(tilemap) != 0x10:
        return None
    x0, x1, y0, y1 = CAMP_BOX
    if x0 < x < x1 and y0 <= y < y1:
        return make_action(down=True) if y < CAMP_GAP_ROW_Y else make_action(left=True)
    if through_gap and x0 - 16 <= x <= x0 and CAMP_GAP_ROW_Y - 8 <= y < y1:
        return make_action(down=True)
    return None


# The stump stand (tile (10,40)) in the pocket, reached by CaveWalkTask on the mountain.
CAVE_STUMP_STAND = (10, 40)
# The statue berry lands on the statue's own tile; from directly below A examines the statue (text 0x37C) and
# the berry is lost when that text ends. One column right, facing up, A lifts it (ROM search 2026-11-02).
STATUE_PICKUP_STAND = (88, 110)
STATUE_IMPACT_TIMER = 0xB586 + 0x15 * 0x40 + 0x10

def berry_objects(ram):
    return [o for o in game_objects(ram) if not o.is_player and o.sprite_table_idx == POWER_BERRY_SPRITE]

def _berry_in_reach(ram):
    """True when the nearest berry is one tile from the farmer along a row or a column (the pick rule).

    In a house only the row counts: the Egg Festival berry lies on the table's edge, so the column above it is
    blocked by the table and an A from there lifts nothing (ROM probe, ``LN_berries_egg_award``)."""
    objects = berry_objects(ram)
    if not objects:
        return False
    pos = get_pos_from_ram(ram)
    obj = min(objects, key=lambda o: abs(o.pixel[0]-pos.x)+abs(o.pixel[1]-pos.y))
    dx, dy = obj.pixel[0] - pos.x, obj.pixel[1] - pos.y
    if is_house_tilemap(int(read_ram_value(ram, "tilemap"))):
        return abs(dx) <= 22 and abs(dy) <= 8
    return (abs(dx) <= 22 and abs(dy) <= 8) or (abs(dy) <= 24 and abs(dx) <= 8)

@dataclass
class PowerBerryPickupTask(Task):
    name: str = "power_berry_pickup"
    timeout: int = 1800
    # Optional stand and facing (pixels / direction). The chicken statue's berry lands on the statue's own tile:
    # A from directly below examines the statue instead, so it is taken from the next column (88,110) facing up.
    stand: tuple | None = None
    stand_face: str = "up"
    # Face this way before eating: A facing a wall or fence does not eat the held berry.
    eat_face: str = ""
    _steps: int = field(default=0, init=False)
    _before: int = field(default=0, init=False)
    _stamina: int = field(default=0, init=False)
    _held: bool = field(default=False, init=False)
    _child: Task | None = field(default=None, init=False)
    _face: str = field(default="", init=False)
    _wait: int = field(default=0, init=False)
    _walk_failures: int = field(default=0, init=False)
    _gone: int = field(default=0, init=False)
    _tried: list = field(default_factory=list, init=False)
    _house_stand: tuple | None = field(default=None, init=False)

    def reset(self, world):
        self._steps = self._wait = self._walk_failures = self._gone = 0
        self._tried = []
        self._before = int(read_ram_value(world.ram, "power_berry_count"))
        self._stamina = int(read_ram_value(world.ram, "max_stamina"))
        self._held = read_held_item(world.ram) == 8
        self._child = None
        self._face = ""

    def can_start(self, world):
        return self._before < BERRY_TARGET

    def progress_snapshot(self):
        return ProgressSnapshot(self.name, "eat" if self._held else "pick", step_count=self._steps,
                                child=task_progress_snapshot(self._child) if self._child else None)

    def step(self, world):
        from harvest.tasks.skills import NavSkill
        self._steps += 1
        ram = world.ram
        count = int(read_ram_value(ram, "power_berry_count"))
        maximum = int(read_ram_value(ram, "max_stamina"))
        if count == self._before + 1 and maximum == self._stamina + 10 and count <= BERRY_TARGET:
            return TaskResult(TaskStatus.SUCCESS, reason=f"ate power berry {self._before}->{count} max_stamina {self._stamina}->{maximum}")
        if count >= BERRY_TARGET and self._before >= BERRY_TARGET:
            return TaskResult(TaskStatus.BLOCKED, reason="power berries already at ten; refuse another pickup")
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"power berry eat unverified count={count} max_stamina={maximum}")
        held = int(read_held_item(ram))
        if held not in (0, 8):
            return TaskResult(TaskStatus.FAILURE, reason=f"power berry hands full: {held:#x}")
        if int(read_ram_value(ram, "input_lock")) == 2:
            if held == 8 and read_ram_u8(ram, 0x019B) & 0x10 and read_ram_u8(ram, 0x018F) != 0:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(up=self._steps % 8 == 1)),
                                  reason="choose to eat the power berry")
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(a=self._steps % 24 < 6)),
                              reason="advance power berry dialogue")
        if not read_ram_u16(ram, 0x00D2) & 0x4000 and held == 0:
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="wait berry cutscene control")
        if (held == 0 and is_house_tilemap(int(read_ram_value(ram, "tilemap")))
                and (read_ram_u8(ram, 0x25) != 15 or int(ram[0x181]) not in (1, 2))):
            # The prize scene hands control back while the room is still being rebuilt (stride byte 4, farmer at the
            # scene's seat): a walk planned on that grid fails three times in three frames.
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="wait house map settle")
        self._held |= held == 8
        if self._held:
            # Stop pressing when the hand clears: the eat animation has a
            # delayed receipt and a second A could lift something else.
            action = make_action(a=True) if held == 8 and self._steps % 24 < 6 else make_action()
            if held == 8 and self.eat_face:
                wanted = {"down": 0, "up": 1, "left": 3, "right": 2}[self.eat_face]
                if int(read_ram_value(ram, "player_direction")) != wanted:
                    action = make_action(**{self.eat_face: self._steps % 8 == 1})
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(action), reason="wait power berry eat receipt")
        if self._child and self.stand is None and _berry_in_reach(ram):
            # The statue berry spawns at one pixel, lands on another a frame later and is gone 33 frames
            # after that (measured 2026-11-02; its permanent flag is already spent). A walk planned from
            # the spawn frame must not outlive a berry that landed in reach: stop and pick.
            self._child = None
        if self._child:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                # The berry's own body stops the walker a few pixels short of the stand tile
                # centre (cave chamber, measured 2026-11-02), and the pick rule below already
                # reaches from there. Judge the real distance; give up after three short walks.
                self._walk_failures += 1
        objects = berry_objects(ram)
        if not objects and self.stand is None:
            # The statue's own script (text 0x37C, ~365 frames, input locked) hands the berry over and
            # removes the sprite a frame or two before the held item shows; the receipt follows. Wait for it
            # with neutral input (an A now could lift something else) before calling the berry lost.
            self._gone += 1
            if self._gone > 240:
                return TaskResult(TaskStatus.FAILURE, reason="no power berry ground sprite")
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="power berry sprite gone; wait for hand or eat receipt")
        self._gone = 0
        pos = get_pos_from_ram(ram)
        obj = min(objects, key=lambda o: abs(o.pixel[0]-pos.x)+abs(o.pixel[1]-pos.y)) if objects else None
        dx, dy = (obj.pixel[0] - pos.x, obj.pixel[1] - pos.y) if obj else (0,0)
        if self.stand is not None:
            # The statue berry is gone ~33 frames after it lands (its text lock only pauses that clock
            # while the farmer stands under the statue), and NavSkill idles ~24 frames before it moves.
            # The stand is one clear step along the road, so press straight at it.
            sx, sy = self.stand[0] - pos.x, self.stand[1] - pos.y
            if abs(sx) > 3 or abs(sy) > 3:
                step = ("right" if sx > 0 else "left") if abs(sx) > 3 else ("down" if sy > 0 else "up")
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{step: True})),
                                  reason=f"power berry sidestep {step} to stand {self.stand}")
            self._face = self.stand_face
        elif abs(dx) <= 22 and abs(dy) <= 8:
            self._face = "right" if dx > 0 else "left"
        elif abs(dy) <= 24 and abs(dx) <= 8 and not is_house_tilemap(int(read_ram_value(ram, "tilemap"))):
            self._face = "down" if dy > 0 else "up"
            # The hawker's ground sprite needs the farmer within 13px;
            # the walk's soft radius can leave it 23px away.
            if abs(dy) > 13:
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{self._face: True})),
                                  reason="approach power berry within pickup reach")
        else:
            # The berry occupies its cell; approach an adjacent cell.
            if abs(dy) <= 8 and is_house_tilemap(int(read_ram_value(ram, "tilemap"))):
                # On the berry's row but outside the reach zone: the stand tile's centre lies past it. Push toward it.
                step = "right" if dx > 0 else "left"
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{step: True})),
                                  reason=f"house power berry approach {step}")
            stands = [(obj.pixel[0]-16, obj.pixel[1]), (obj.pixel[0]+16, obj.pixel[1]),
                      (obj.pixel[0], obj.pixel[1]+16), (obj.pixel[0], obj.pixel[1]-16)]
            floor = {0xA0,0xD9,0xC5,0xA8}
            house_order = False
            if is_house_tilemap(int(read_ram_value(ram, "tilemap"))):
                # The Egg Festival's house berry lies on the table's edge, off the tile grid. Stand where the pick rule
                # reaches it (24 px above or below, 22 px beside) on open floor; the rug tile 0xFF is walkable.
                stands = [(obj.pixel[0]+20, obj.pixel[1]), (obj.pixel[0]-20, obj.pixel[1]),
                          (obj.pixel[0], obj.pixel[1]-24), (obj.pixel[0], obj.pixel[1]+24)]
                stands = [p for p in stands if p not in self._tried]
                floor = floor | {0xFF, 0xA8}
                house_order = True
            width = {1:16, 2:32}.get(int(ram[0x181]),64)
            if width < 64:
                from harvest.core.tile_catalog import metatile_grid_base
                base = metatile_grid_base(ram)
                stands = [p for p in stands if int(ram[base+(p[1]//16)*width+p[0]//16]) in floor]
            if not stands:
                return TaskResult(TaskStatus.FAILURE, reason="power berry has no accessible adjacent floor")
            if self._walk_failures >= 3:
                return TaskResult(TaskStatus.FAILURE, reason="power berry stand unreachable after 3 walks")
            stand = stands[0] if house_order else min(stands, key=lambda p: abs(p[0]-pos.x)+abs(p[1]-pos.y))
            self._house_stand = stand
            self._child = (CaveWalkTask(target=(stand[0]//16,stand[1]//16),radius=5,mid_tol=3,timeout=900) if width < 64
                           else NavSkill(target_px=stand, radius=4, soft_radius=4, timeout=900))
            self._child.reset(world)
            first = self._child.step(world)
            if first.status == TaskStatus.RUNNING:
                return first
            # A walk that ends on its first step (already on the stand tile) is not a finished pickup.
            self._child = None
            self._walk_failures += first.status != TaskStatus.SUCCESS
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="berry walk already there")
        self._wait += 1
        if self._wait > 120 and self._house_stand is not None and self.stand is None:
            # A stand the script ignores (the table lies between): try the next side of the berry.
            self._tried.append(self._house_stand)
            self._house_stand, self._wait, self._child = None, 0, None
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason="house power berry: next stand")
        facing = int(read_ram_value(ram, "player_direction"))
        wanted = {"down": 0, "up": 1, "left": 3, "right": 2}[self._face]
        action = make_action()
        if obj is not None and self.stand is None and is_house_tilemap(int(read_ram_value(ram, "tilemap"))):
            # The house berry's script sees the farmer only inside a narrow zone: line up across the facing axis
            # (the BFS walk stops up to 5 px off the tile centre, 8 px off the berry's column) before pressing A.
            across = dx if self._face in ("up", "down") else dy
            if abs(across) > 2:
                side = (("right" if across > 0 else "left") if self._face in ("up", "down")
                        else ("down" if across > 0 else "up"))
                return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(**{side: True})),
                                  reason=f"house power berry align {side}")
        if facing != wanted:
            if self._wait % 8 == 1:
                action = make_action(**{self._face: True})
        elif self.stand == STATUE_PICKUP_STAND:
            # CC $15 deletes its impact sprite at timer zero. An earlier lift
            # can reuse that sprite slot and lose its own eat interaction.
            action = make_action(a=read_ram_u16(ram, STATUE_IMPACT_TIMER) <= 1)
        elif self._wait % 24 < 6:
            action = make_action(a=True)
        elif self.stand is None:
            # The ROM refreshes the object's contact flag on a blocked step.
            # Being near and already facing it is insufficient (hawker berry).
            action = make_action(**{self._face: True})
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(action), reason=f"pick power berry facing {self._face}")

def collect_spawned_berry(owner, world):
    """Pause a tool skill for a nearby berry, then resume its work.

    Both crop and pasture hoeing call this before advancing their queues.
    Count ten is a hard stop; all counts use $7F1F36, never transient $0976.
    """
    task = getattr(owner, "_berry_pickup", None)
    if task is None:
        if int(read_ram_value(world.ram, "power_berry_count")) >= BERRY_TARGET:
            return None
        if int(getattr(world, "frame", 0) or 0) < int(getattr(owner, "_berry_giveup_until", -1)):
            return None
        pos = get_pos_from_ram(world.ram)
        radius = 192 if getattr(owner, "source", None) == "chicken_trade" else 64  # hawker walks the farmer away
        nearby = any(abs(o.pixel[0]-pos.x)+abs(o.pixel[1]-pos.y) <= radius for o in berry_objects(world.ram))
        if not nearby and read_held_item(world.ram) != 8:
            return None
        if nearby and getattr(owner, "source", None) == "chicken_trade" and pos.x < 80 and any(o.pixel[0] >= 64 for o in berry_objects(world.ram)):
            # The hawker can leave us west of the fence. Clear its gate before chasing the reward south.
            direction = "up" if pos.x < 64 and pos.y > 426 else "right"
            pad = (make_action(a=world.frame % 24 < 6) if int(read_ram_value(world.ram, "input_lock")) == 2 else make_action(**{direction: True}))
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(pad), reason="clear farm west gate before hawker berry pickup")
        task = PowerBerryPickupTask(stand=getattr(owner, "pickup_stand", None), stand_face=getattr(owner, "pickup_face", "up"),
                                 eat_face=getattr(owner, "eat_face", ""))
        task.reset(world)
        owner._berry_pickup = task
    result = task.step(world)
    if result.status == TaskStatus.SUCCESS:
        owner._berry_pickup = None
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason=result.reason)
    if result.status != TaskStatus.RUNNING:
        # An opportunistic pickup that cannot finish (a cutscene's transient sprite, a berry out of reach) is the
        # owner's side errand: drop it and let the owner carry on, never fail the owner's own work for it.
        # Wait before trying again so one stubborn sprite is not re-chased every frame.
        owner._berry_pickup = None
        owner._berry_giveup_until = int(getattr(world, "frame", 0) or 0) + 1_200
        return None
    return result

@dataclass
class PowerBerryTask(Task):
    """One source through the existing skill API, with a persistent eat receipt."""
    source: str = "field_1"
    name: str = "power_berry"
    timeout: int = 180000
    _steps: int = field(default=0, init=False)
    _before: int = field(default=0, init=False)
    _max: int = field(default=0, init=False)
    _child: Task | None = field(default=None, init=False)
    _used: set = field(default_factory=set, init=False)
    _phase: str = field(default="entry", init=False)
    _phase_start: int = field(default=0, init=False)
    # Read by ``collect_spawned_berry``: this task meets the spawned berry before its child tool task does.
    pickup_stand: tuple | None = field(default=None, init=False)
    pickup_face: str = field(default="up", init=False)
    eat_face: str = field(default="", init=False)
    _leaving: TaskResult | None = field(default=None, init=False)
    _leave_steps: int = field(default=0, init=False)

    def reset(self, world):
        from harvest.planner.tasks.inventory import ExitToFarmTask
        self._leaving, self._leave_steps = None, 0
        self.pickup_stand = (STATUE_PICKUP_STAND if self.source == "statue" else
                             (686, 414) if self.source == "fish" else None)
        self.pickup_face = "down" if self.source == "fish" else "up"
        self.eat_face = ""
        self._steps = 0
        self._before = int(read_ram_value(world.ram, "power_berry_count"))
        self._max = int(read_ram_value(world.ram, "max_stamina"))
        self._used = set()
        self._phase = "entry"
        self._child = ExitToFarmTask()
        self._child.reset(world)
        self._berry_pickup = None

    def _pocket_closed(self, ram):
        return self.source == "cave" and not read_ram_u16(ram, BOULDER_FLAG_ADDR) & BOULDER_OPEN_FLAG

    def _leave_fishing(self, result, world):
        """End the task with ``result`` once the rod is in: a cast line pins the farmer (see ``fishing_exit_pad``)."""
        pad = None
        if self.source in FISHING_SOURCES:
            pos = get_pos_from_ram(world.ram)
            pad = fishing_exit_pad(world.ram, self._leave_steps)
            if pad is None:
                pad = camp_exit_pad(int(read_ram_value(world.ram, "tilemap")), pos.x, pos.y)
        if pad is None or self._leave_steps >= FISHING_LEAVE_FRAMES:
            return result
        self._leaving, self._leave_steps = result, self._leave_steps + 1
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(pad), reason="reel in and leave the fishing camp")

    def can_start(self, world):
        return ((int(read_ram_value(world.ram, "power_berry_count")) < BERRY_TARGET or self._turtle_due(world.ram))
                and not self._pocket_closed(world.ram) and not self._weather_closed(world.ram))

    def _turtle_due(self, ram):
        return self.source == "egg_festival" and not read_ram_u16(ram, 0x11F6C) & 0x1000

    def _weather_closed(self, ram):
        return self.source in {"statue", "fish"} and bool(read_ram_u16(ram, 0x0196) & 0x001A)

    def progress_snapshot(self):
        return ProgressSnapshot(self.name, self._phase, step_count=self._steps,
                                details=(("source", self.source), ("swings", len(self._used))),
                                child=task_progress_snapshot(self._child) if self._child else None)

    def _start(self, task, world, phase):
        self._child, self._phase = task, phase
        self._phase_start = self._steps
        task.reset(world)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()), reason=phase)

    def step(self, world):
        from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
        from harvest.tasks.skills import NavSkill, sequence_skills
        from harvest.tasks.crop_skills import hoe_until_tilled_skill
        from harvest.tasks.hot_spring import HotSpringStaminaTask
        self._steps += 1
        ram = world.ram
        if self._leaving is not None:
            return self._leave_fishing(self._leaving, world)
        count = int(read_ram_value(ram, "power_berry_count"))
        if self.source == "cave" and count > self._before and not (read_ram_u16(ram,0x11F6A)&0x40):
            self._before, self._max = count, int(read_ram_value(ram,"max_stamina"))
        if count == self._before + 1 and int(read_ram_value(ram, "max_stamina")) == self._max + 10 and count <= BERRY_TARGET:
            return TaskResult(TaskStatus.SUCCESS, reason=f"{self.source} eaten {self._before}->{count}")
        if self._before >= BERRY_TARGET and self.source != "egg_festival":
            return TaskResult(TaskStatus.BLOCKED, reason="already ten power berries")
        if self._pocket_closed(ram) and int(read_ram_value(ram, "tilemap")) != 0x29:
            return TaskResult(TaskStatus.BLOCKED, reason="cave berry: boulder pocket opens Year 1 Winter D8 ($7F1F64 & 2)")
        if self._weather_closed(ram):
            return TaskResult(TaskStatus.BLOCKED, reason=f"{self.source} berry requires a clear day")
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"{self.source} acquisition timeout")
        give_up = FESTIVAL_GIVE_UP_HOURS.get((self.source, self._phase))
        if give_up is not None and int(read_ram_value(ram, "hour")) >= give_up and int(read_ram_value(ram, "tilemap")) not in (0x09, 0x0B):
            return self._leave_fishing(
                TaskResult(TaskStatus.FAILURE, reason=f"{self.source}: too late for the festival (it closes at 15:00)"), world)
        if self.source == "snow_flower" and self._phase == "plant" and self._steps-self._phase_start > 900:
            return self._start(PowerBerryPickupTask(), world, "eat")
        # The Kappa's moving berry uses the ground sprite during its cutscene.
        # Let that source finish the handoff before the generic picker takes over.
        pickup = (None if isinstance(self._child, (FishBerryTask, EggHuntTask, PowerBerryPickupTask)) or
                  (self.source == "snow_flower" and self._phase == "plant")
                  else collect_spawned_berry(self, world))
        if pickup is not None:
            return pickup
        if self._child:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            print(f"[POWER_BERRY] {self.source}/{self._phase} step ended {result.status.name}: {result.reason}")
            if result.status != TaskStatus.SUCCESS and (self.source not in FIELD_BITS or self._phase in {"entry", "tool", "spa"}):
                return self._leave_fishing(result, world)
            if self.source == "fish" and self._phase == "source":
                return self._start(PowerBerryPickupTask(), world, "eat")
            if self.source == "statue" and self._phase == "source":
                return self._start(PowerBerryPickupTask(stand=STATUE_PICKUP_STAND), world, "eat")
            if self.source == "egg_festival" and self._phase == "festival":
                return self._start(EggHuntTask(), world, "hunt")
            if self.source == "egg_festival" and self._phase == "hunt":
                if read_ram_u16(ram, 0x11F6C) & 0x6000 != 0x2000:
                    return TaskResult(TaskStatus.SUCCESS, reason="egg festival won: turtle shell, no berry this win")
                return self._start(PowerBerryPickupTask(), world, "eat")
            if self.source == "harvest_festival" and self._phase == "source":
                return self._start(FestivalEntryTask(target_map=0x09), world, "festival")
            if self.source == "snow_flower" and self._phase == "source":
                from harvest.core.tile_catalog import metatile_grid_base
                width = {1:16, 2:32}.get(int(ram[0x181]),64)
                base = metatile_grid_base(ram)
                sites = [(x,y) for y in range(32) for x in range(width) if int(ram[base+y*width+x]) == 0xF6]
                if not sites:
                    return TaskResult(TaskStatus.FAILURE, reason="snow flower summit planting tile missing")
                x,y = sites[0]
                return self._start(sequence_skills("plant_snow_flower",
                    CaveWalkTask(target=(x,y+1), radius=4, mid_tol=4),
                    BerryToolTask(tool_id=0x17, flag_mask=0, timeout=6000, pickup_enabled=False)), world, "plant")
            if self.source in {"stump", "cave"}:
                if self._phase == "entry":
                    return self._start(mountain_axe_recipe(cave=self.source == "cave"), world, "mountain")
                return self._mountain_step(world)
            if self._phase == "entry" and self.source == "statue":
                return self._start(statue_recipe(ram), world, "source")
            if self._phase == "entry" and self.source == "fish":
                return self._start(FishBerryTask(), world, "source")
            if self._phase == "entry" and self.source == "chicken_trade":
                return self._start(chicken_trade_recipe(world), world, "source")
            if self._phase == "entry" and self.source == "harvest_festival":
                return self._start(FishBerryTask(keep_only=True), world, "source")
            if self._phase == "entry" and self.source == "snow_flower":
                return self._start(snow_flower_recipe(world), world, "source")
            if self._phase == "entry" and self.source == "egg_festival":
                return self._start(FestivalEntryTask(target_map=0x0B), world, "festival")
            if self._phase in {"entry", "spa"}:
                return self._start(EnsureCarryToolTask(tool_id=2, exit_when_done=True), world, "tool")
        if self.source in {"stump", "cave"}:
            return self._mountain_step(world)
        if self.source == "statue" and self._phase == "entry":
            return self._start(statue_recipe(ram), world, "source")
        if self.source == "harvest_festival" and self._phase == "festival":
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(
                make_action(a=int(read_ram_value(ram, "input_lock")) == 2 and self._steps % 24 < 6)),
                reason="wait for native Harvest Festival berry eat")
        if self.source not in FIELD_BITS:
            return TaskResult(TaskStatus.BLOCKED, reason=f"native source not yet implemented: {self.source}")
        if read_ram_u16(ram, 0x11F64) & FIELD_BITS[self.source]:
            return TaskResult(TaskStatus.FAILURE, reason=f"{self.source} spawned previously but eat unverified")
        if read_ram_u16(ram, 0x11F5C) & 0x0100:
            return TaskResult(TaskStatus.BLOCKED, reason="field berry daily roll used; retry tomorrow")
        if int(read_ram_value(ram, "stamina")) < 2:
            return self._start(HotSpringStaminaTask(return_to_farm=True), world, "spa")
        pos = get_pos_from_ram(ram)
        start = pos.x//16, pos.y//16
        targets = [(x, y) for y in range(32,59) for x in range(3,60)
                   if (x,y) not in self._used and get_tile_at(ram,x,y) in (1,2)]
        if not targets:
            return TaskResult(TaskStatus.FAILURE, reason="no fresh farm soil for berry roll")
        target = min(targets, key=lambda t:(abs(t[0]-start[0])+abs(t[1]-start[1]),t))
        self._used.add(target)
        stand = target[0]-1,target[1]
        return self._start(sequence_skills("berry_field_hoe",
            NavSkill(target_px=(stand[0]*16+8,stand[1]*16+8),radius=4,soft_radius=4,timeout=1200),
            hoe_until_tilled_skill(target_tile=target,face="right")), world, "hoe")

    def _mountain_step(self, world):
        from harvest.tasks.skills import NavSkill, sequence_skills
        from harvest.tasks.hot_spring import HotSpringStaminaTask
        ram=world.ram
        if self.source == "cave" and int(read_ram_value(ram,"tilemap")) == 0x29:
            return self._start(PowerBerryPickupTask(),world,"source")
        if self.source == "cave" and self._phase == "chop":
            # The eastern summit stump hides a drop into the berry chamber.
            return self._start(cave_drop_skill(),world,"hole")
        if int(read_ram_value(ram,"stamina")) < 16:
            return self._start(HotSpringStaminaTask(return_to_farm=False),world,"mountain_spa")
        pos=get_pos_from_ram(ram)
        targets=([(10,38)] if self.source == "cave" else
                 [(x,y) for y in range(3,45) for x in range(3,60) if get_tile_at(ram,x,y)==9 and (x,y) not in self._used])
        if not targets:return TaskResult(TaskStatus.FAILURE,reason="mountain stumps exhausted; revisit tomorrow")
        target=min(targets,key=lambda t:abs(t[0]*16-pos.x)+abs(t[1]*16-pos.y))
        self._used.add(target)
        stand=target[0],target[1]+2
        return self._start(sequence_skills("mountain_stump",
            NavSkill(target_px=(stand[0]*16+8,stand[1]*16+8),radius=8 if self.source == "cave" else 5,
                     soft_radius=8 if self.source == "cave" else 5,timeout=2400),
            BerryToolTask(tool_id=4,face="up",flag_mask=0 if self.source == "cave" else 0x0400,stump_tile=target,timeout=1800)),world,"chop")

@dataclass
class CaveWalkTask(Task):
    """Cave floor navigation uses the ROM's 32-cell row, not farm stride 64."""
    target: tuple = (14, 18)
    radius: int = 3
    name: str = "cave_walk"
    timeout: int = 6000
    # Pixels from a tile centre that count as "on" an intermediate path tile. The farmer settles
    # 2 px off-centre running a mountain lane (y=694 for row 43), which 1 px never accepted.
    mid_tol: int = 1
    # Off by default. Above 0: before each step along a lane, walk the perpendicular offset to
    # within this many pixels. A mountain corridor sits on a cliff lip (tile 0xA7 over 0xA1):
    # a farmer 6 px below the centre line steps west and drops to the lane below.
    lane_align: int = 0
    walkable: frozenset | None = None
    _path: list = field(default_factory=list, init=False)
    _blocked_edges: set = field(default_factory=set, init=False)
    _leg_steps: int = field(default=0, init=False)
    _steps: int = field(default=0, init=False)

    def reset(self, world):
        self._blocked_edges=set()
        self._steps=self._leg_steps=0
        self._plan(world)

    def _plan(self, world):
        from collections import deque
        from harvest.core.tile_catalog import metatile_grid_base
        ram = world.ram
        pos = get_pos_from_ram(ram)
        start = (pos.x//16, pos.y//16)
        width = {1:16, 2:32}.get(int(ram[0x181]),64)
        base = metatile_grid_base(ram)
        queue, previous = deque([start]), {start:None}
        while queue and self.target not in previous:
            x,y = queue.popleft()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                tile = x+dx,y+dy
                if ((x,y),tile) in self._blocked_edges or tile in previous or not (0 <= tile[0] < width and 0 <= tile[1] < (64 if width == 64 else 32)):continue
                tid=int(ram[base+tile[1]*width+tile[0]])
                if tid not in (self.walkable or {0xA0,0xC5,0xFF,0xD9,0xA8,0xA7}):continue
                previous[tile]=(x,y);queue.append(tile)
        self._path=[]
        if self.target in previous:
            t=self.target
            while t != start:self._path.append(t);t=previous[t]
            self._path.reverse()
        self._leg_steps=0

    def can_start(self, world):return True

    def step(self, world):
        self._steps+=1
        self._leg_steps+=1
        pos=get_pos_from_ram(world.ram)
        if not self._path:
            good=max(abs(pos.x-(self.target[0]*16+8)),abs(pos.y-(self.target[1]*16+8))) <= self.radius
            return TaskResult(TaskStatus.SUCCESS if good else TaskStatus.FAILURE,reason="cave native-width path finished" if good else "no cave floor path")
        if self._steps > self.timeout:return TaskResult(TaskStatus.FAILURE,reason=f"cave floor blocked at {pos}")
        if self._leg_steps > 90:
            start=(pos.x//16,pos.y//16)
            self._blocked_edges.add((start,self._path[0]))
            # If already across the tile boundary, reject entry to that tile
            # from the preceding cell too, so a corner cannot re-arm itself.
            if start==self._path[0]:
                self._blocked_edges.update(((start[0]+dx,start[1]+dy),start) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))
            self._plan(world)
            return TaskResult(TaskStatus.RUNNING,action=ActionResult(make_action()),reason="berry floor replan blocked edge")
        tx,ty=self._path[0][0]*16+8,self._path[0][1]*16+8
        dx,dy=tx-pos.x,ty-pos.y
        tol=self.radius if len(self._path)==1 else self.mid_tol
        if abs(dx)<=tol and abs(dy)<=tol:
            self._path.pop(0)
            self._leg_steps=0
            return TaskResult(TaskStatus.RUNNING,action=ActionResult(make_action()))
        if self.lane_align:
            horizontal=abs(dx)>abs(dy)
            off=dy if horizontal else dx
            if abs(off)>self.lane_align:
                side=("down" if off>0 else "up") if horizontal else ("right" if off>0 else "left")
                return TaskResult(TaskStatus.RUNNING,action=ActionResult(make_action(**{side:True})),reason="cave walk lane align")
        direction=("right" if dx>0 else "left") if abs(dx)>1 else ("down" if dy>0 else "up")
        return TaskResult(TaskStatus.RUNNING,action=ActionResult(make_action(**{direction:True,"b":True})))


@dataclass
class BerryToolTask(Task):
    tool_id: int = 3
    face: str = "up"
    flag_addr: int = 0x11F64
    flag_mask: int = 0x0200
    stump_tile: tuple | None = None
    pickup_stand: tuple | None = None
    pickup_face: str = "up"
    eat_face: str = ""
    name: str = "berry_source_tool"
    timeout: int = 6000
    pickup_enabled: bool = True
    _steps: int = field(default=0, init=False)

    def reset(self, world):self._steps=0;self._berry_pickup=None
    def can_start(self, world):return True
    def step(self, world):
        from harvest.core.carry import selected_tool
        self._steps+=1
        pickup=collect_spawned_berry(self, world) if self.pickup_enabled else None
        if pickup is not None:return pickup
        if read_ram_u16(world.ram,self.flag_addr)&self.flag_mask:
            return TaskResult(TaskStatus.SUCCESS,reason="berry source flag set")
        if self.stump_tile and self._steps > 60 and get_tile_at(world.ram,*self.stump_tile) not in {9,10,11,12,17,18,19,20}:
            return TaskResult(TaskStatus.SUCCESS,reason="stump chopped")
        if self._steps>self.timeout:return TaskResult(TaskStatus.FAILURE,reason="berry source tool timeout")
        if int(read_ram_value(world.ram,"stamina"))<2:return TaskResult(TaskStatus.FAILURE,reason="stamina_low berry source")
        action=make_action()
        if int(read_ram_value(world.ram,"input_lock")) == 2:
            action=make_action(a=self._steps%24<6)
        elif int(selected_tool(world.ram))!=self.tool_id:
            action=make_action(x=self._steps%6==1)
        elif int(read_ram_value(world.ram,"player_direction"))!={"down":0,"up":1,"right":2,"left":3}[self.face]:
            if self._steps%8==1:action=make_action(**{self.face:True})
        elif int(read_ram_value(world.ram,"input_lock"))==1:
            action=make_action(y=8<=self._steps%60<28)
        else:
            action=make_action(a=self._steps%24<6)
        return TaskResult(TaskStatus.RUNNING,action=ActionResult(action))


def statue_recipe(ram=None):
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
    from harvest.planner.tasks.navigation import MultiMapNavTask
    from harvest.maps.map_config import segment_waypoints
    from harvest.tasks.skills import sequence_skills, ButtonSequenceSkill
    from harvest.core.tile_catalog import MOUNTAIN_WALKABLE
    from harvest.core.carry import carry_pair_items

    # A late campaign may own the gold hammer instead. Its shelf is the same
    # stand, but the native pickup gives tool $13, never $03. GoldHammer's
    # statue branch at $82A141 has the same berry roll as the ordinary tool.
    gold = os.environ.get("HARVEST_GOLD_STATUE", "0").strip().lower() in {"1", "true", "yes"}
    hammer = 0x13 if gold and ram is not None and (0x13 in carry_pair_items(ram) or read_ram_u8(ram, 0x11F02) & 4) else 3
    return sequence_skills("statue_berry",EnsureCarryToolTask(tool_id=hammer,exit_when_done=True),
        MultiMapNavTask(waypoints=segment_waypoints("farm_to_path"),timeout=6000),
        CaveWalkTask(target=(4,6), radius=6),
        BerryToolTask(tool_id=hammer, pickup_stand=STATUE_PICKUP_STAND, pickup_face="up"))


def mountain_axe_recipe(*, cave=False):
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
    from harvest.planner.tasks.navigation import MultiMapNavTask
    from harvest.maps.map_config import ROUTES, Waypoint
    from harvest.tasks.skills import sequence_skills
    travel = ROUTES["farm_to_spa"]
    if cave:
        # Land on the mountain (north pull), then follow the tile BFS to the pocket's stump stand. The
        # old (248,688)/(168,688) legs ran into the x=17-18 wall; the way round goes east, up the x=32
        # corridor and back west along rows 39-41 (through the boulder gap once the flag is set).
        travel = ROUTES["farm_to_mountain"] + [
            Waypoint(tilemap=0x10,target_px=(328,688),radius=8,run_direction="up",force_run=True)]
        return sequence_skills("mountain_axe",EnsureCarryToolTask(tool_id=4,exit_when_done=True),
            MultiMapNavTask(waypoints=travel,timeout=18000),
            CaveWalkTask(target=CAVE_STUMP_STAND,radius=6,timeout=6000,mid_tol=6,lane_align=2))
    return sequence_skills("mountain_axe",EnsureCarryToolTask(tool_id=4,exit_when_done=True),
        MultiMapNavTask(waypoints=travel,timeout=18000))


def cave_drop_skill():
    from harvest.tasks.skills import ButtonSequenceSkill
    return ButtonSequenceSkill(name="drop_through_stump_hole",sequence=
        [make_action(up=True,b=True) for _ in range(40)]+[make_action() for _ in range(140)],timeout=240)


def chicken_trade_recipe(world):
    """The first Fall hen sale summons the hawker; the parent catches his berry."""
    from harvest.planner.day_phase_chicken import chicken_sale_cycle_phases
    from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task
    from harvest.tasks.skills import sequence_skills
    return sequence_skills("chicken_trade", *[
        build_phase_task(TaskBuildContext(), spec, world)
        for spec in chicken_sale_cycle_phases(wait_before_return=False, event_hour=0)])


def snow_flower_recipe(world):
    from harvest.maps.map_config import ROUTES, Waypoint
    from harvest.planner.tasks.navigation import MultiMapNavTask
    from harvest.planner.tasks.inventory_shed import EnsureCarryToolTask
    from harvest.tasks.skills import sequence_skills, ButtonSequenceSkill
    from harvest.core.tile_catalog import MOUNTAIN_WALKABLE
    from harvest.planner.day_phase_calendar import horse_event_due, MEET_HORSE_PHASE
    from harvest.planner.day_plan_orchestrator import DayPlanTask
    opening = [DayPlanTask(phase_sequence=[MEET_HORSE_PHASE])] if horse_event_due(world.ram) else []
    return sequence_skills("snow_flower_summit", *opening,
        EnsureCarryToolTask(tool_id=0x17, exit_when_done=True),
        MultiMapNavTask(waypoints=ROUTES["farm_to_spa"][:-4], timeout=12000),
        CaveWalkTask(target=(24,7),radius=6,mid_tol=2,timeout=12000,
                     walkable=MOUNTAIN_WALKABLE-frozenset({0xA1,0xA7,0x35})),
        SummitGateTask())


@dataclass
class SummitGateTask(Task):
    name: str = "summit_fence_lane"
    timeout: int = 1800
    _steps: int = field(default=0, init=False)

    def reset(self, world): self._steps = 0
    def can_start(self, world): return True
    def step(self, world):
        from harvest.core.ram_catalog import read_ram_u8
        self._steps += 1
        tm = int(read_ram_value(world.ram, "tilemap"))
        if tm == 0x34 and read_ram_u8(world.ram, 0x25) == 15:
            return TaskResult(TaskStatus.SUCCESS, reason="winter summit entered")
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason="summit fence lane timeout")
        pad = make_action()
        if tm == 0x10:
            x = get_pos_from_ram(world.ram).x
            pad = make_action(**{"left" if x > 391 else "right": True}) if not 389 <= x <= 391 else make_action(up=True)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(pad))


@dataclass
class FishBerryTask(Task):
    """Borrow the rod, reel on a bite, lift the catch and return it to the pool.

    The fish must be thrown before 17:00; $7F1F66 & $2000 is the one-shot
    receipt for the return event, not proof that its berry has been eaten.
    """
    name: str = "fish_berry"
    timeout: int = 18000
    keep_only: bool = False
    _phase: str = field(default="travel", init=False)
    _steps: int = field(default=0, init=False)
    _age: int = field(default=0, init=False)
    _child: Task | None = field(default=None, init=False)
    _fish: bool = field(default=False, init=False)

    def reset(self, world):
        from harvest.maps.map_config import ROUTES, Waypoint
        from harvest.planner.tasks.navigation import MultiMapNavTask
        self._steps = self._age = 0
        self._phase = "travel"
        self._child = MultiMapNavTask(waypoints=ROUTES["farm_to_mountain"] + [
            Waypoint(0x10, (328, 688), radius=6, run_direction="up", force_run=True),
            Waypoint(0x10, (424, 712), radius=12),
            Waypoint(0x10, (520, 712), radius=8, run_direction="right", force_run=True),
            Waypoint(0x10, (520, 632), radius=8, run_direction="up", force_run=True),
            *[Waypoint(0x10, p, radius=10) for p in
              ((520, 600), (560, 600), (560, 520), (560, 478), (582, 414))]], timeout=12000)
        self._child.reset(world)

    def can_start(self, world):
        return self.keep_only or not read_ram_u16(world.ram, 0x11F66) & 0x2000

    def _next(self, phase):
        self._phase, self._age = phase, 0

    def step(self, world):
        from harvest.tasks.skills import NavSkill
        self._steps += 1
        self._age += 1
        ram = world.ram
        held = int(read_held_item(ram))
        action = int(read_ram_value(ram, "player_action"))
        lock = int(read_ram_value(ram, "input_lock"))
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"fish berry timeout: {self._phase}")
        if held not in (0, 7, 8):
            return TaskResult(TaskStatus.FAILURE, reason=f"fish berry hands full: {held}")
        if self._child:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                return result
            if self._phase == "travel":
                self._child = NavSkill(target_px=(598, 389), radius=3, soft_radius=3, timeout=1800)
                self._child.reset(world)
                self._next("fisherman")
            elif self._phase == "fisherman":
                self._next("borrow")
            elif self._phase == "pool":
                self._next("return")
        pad = make_action()
        pos = get_pos_from_ram(ram)
        direction = int(read_ram_value(ram, "player_direction"))
        if self._phase == "borrow":
            if lock == 2:
                pad = make_action(a=self._age % 24 < 6)
            elif self._age < 90:
                pad = make_action(up=True) if direction != 1 else make_action(a=self._age % 24 < 6)
            else:
                self._next("cast_stand")
        elif self._phase == "cast_stand":
            if action == 15:
                self._next("fishing")
            elif pos.x < 633 or pos.x > 635:
                pad = make_action(**{"right" if pos.x < 633 else "left": True})
            elif pos.y > 371:
                pad = make_action(up=True)
            elif direction != 2:
                pad = make_action(right=True)
        elif self._phase == "fishing":
            if action in (15, 18):
                pad = make_action(y=self._age % 6 == 1)
            elif action == 15 and self._age > 180:
                self._next("catch")
            # A successful reel leaves a ground fish; no need to cast again.
            if self._age > 90 and action == 19:
                self._fish = read_ram_u16(ram, 0x0901) == 0x008B
                self._next("reel")
        elif self._phase == "reel":
            if action == 15:
                # RNGReturn0toA(3) reels a fish only for animation $8B.
                # The other two catches are empty/trash; cast again in place.
                self._next("catch" if self._fish else "fishing")
        elif self._phase == "catch":
            if held == 7:
                self._next("keep")
            elif action not in (0, 15):
                pass
            elif pos.y < 378:
                pad = make_action(down=True)
            elif pos.x > 629:
                pad = make_action(left=True)
            elif direction != 3:
                pad = make_action(left=True)
            else:
                pad = make_action(a=self._age % 24 < 6)
        elif self._phase == "keep":
            if lock == 2:
                if not int(ram[0x019B]) & 0x10:
                    pad = make_action(a=self._age % 24 < 6)
                elif int(read_ram_value(ram, "dialog_choice_result")) == 0:
                    pad = make_action(down=self._age % 8 == 1)
                else:
                    pad = make_action(a=self._age % 24 < 6)
            elif self._age > 60 and action == 0:
                if self.keep_only:
                    self._next("leave_camp")
                    return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action()))
                self._child = NavSkill(target_px=(686, 411), radius=3, soft_radius=3, timeout=2400)
                self._child.reset(world)
                self._next("pool")
        elif self._phase == "leave_camp":
            # The camp fence's gap is west at x=585. Its carried-fish sprite
            # follows the farmer and obstructs generic object-avoiding BFS.
            if pos.x > 585:
                pad = make_action(down=True) if pos.y < 390 else make_action(left=True)
            elif pos.y < 422:
                pad = make_action(down=True)
            else:
                return TaskResult(TaskStatus.SUCCESS, reason="caught fish, kept it and left the camp fence")
        elif self._phase == "return":
            if held == 0 and read_ram_u16(ram, 0x11F66) & 0x2000:
                self._next("reward")
            elif held == 7 and pos.x != 686:
                pad = make_action(**{"right" if pos.x < 686 else "left": True})
            elif held == 7 and pos.y > 406:
                pad = make_action(up=True)
            elif direction != 1:
                pad = make_action(up=True)
            else:
                pad = make_action(a=self._age % 24 < 6)
        elif self._phase == "reward":
            if held == 8:
                return TaskResult(TaskStatus.SUCCESS, reason="caught fish, returned to pool and received berry")
            if lock == 2:
                pad = make_action(a=self._age % 24 < 6)
            elif self._age > 600:
                if pos.y < 411:
                    pad = make_action(down=True)
                elif direction != 0:
                    pad = make_action(down=True)
                else:
                    pad = make_action(a=self._age % 24 < 6)
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(pad), reason=f"fish berry {self._phase}")


@dataclass
class FestivalEntryTask(Task):
    """Reach the dated town west-edge festival from the farm or fish camp."""
    target_map: int = 0x09
    name: str = "berry_festival_entry"
    timeout: int = 12000
    _steps: int = field(default=0, init=False)
    _child: Task | None = field(default=None, init=False)

    def reset(self, world):
        from harvest.maps.map_config import Waypoint, segment_waypoints
        from harvest.planner.tasks.navigation import MultiMapNavTask
        self._steps = 0
        tm = int(read_ram_value(world.ram, "tilemap"))
        route = []
        if tm == 0x10:
            route = [Waypoint(0x10, p, radius=10) for p in
                     ((560,478),(560,600),(520,600),(520,712),(424,712),(328,712))]
            route += [Waypoint(0x10,(312,744),radius=12,is_exit=True,exit_direction="down")]
            route += [Waypoint(0x0C,(132,128),radius=8,run_direction="down",force_run=True)]
        route += segment_waypoints(*(("farm_to_path", "path_to_town") if tm == 0 else ("path_to_town",)))
        # The mountain leg stands 7 px short of a tile centre with a solid below (Y1 Fall D12, fish in hand), and
        # centring toward it never finishes: let a stalled centre give way to the path.
        self._child = MultiMapNavTask(waypoints=route, timeout=self.timeout, center_stall_frames=CENTER_STALL_FRAMES)
        self._child.reset(world)

    def can_start(self, world):
        return int(read_ram_value(world.ram, "hour")) < 15

    def step(self, world):
        self._steps += 1
        tm = int(read_ram_value(world.ram, "tilemap"))
        if tm == self.target_map:
            return TaskResult(TaskStatus.SUCCESS, reason="dated festival entered")
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason="festival entry timeout")
        p = get_pos_from_ram(world.ram)
        if tm == 4 and p.x <= 26 and 120 <= p.y <= 156:
            return TaskResult(TaskStatus.RUNNING, action=ActionResult(make_action(left=True)))
        return self._child.step(world)


@dataclass
class EggHuntTask(Task):
    """Read the shuffled barrel colours, return matching eggs, receive the prize."""
    name: str = "egg_berry_hunt"
    timeout: int = 28000
    _phase: str = field(default="start", init=False)
    _steps: int = field(default=0, init=False)
    _target: int = field(default=0, init=False)
    _score: int = field(default=0, init=False)
    _prizes: int = field(default=0, init=False)
    _goals: list = field(default_factory=list, init=False)
    _child: Task | None = field(default=None, init=False)
    # $81F490..$81F551: the seven searchable barrels, indexed by $09A4+slot.
    SITES = ((0,0),(16,48),(176,144),(448,80),(672,48),(80,704),(528,592),(528,816))

    def reset(self, world):
        self._phase, self._steps, self._target, self._score = "start", 0, 0, 0
        self._prizes = read_ram_u16(world.ram, 0x11F6C) & 0x3000
        self._goals = [(344,536),(344,424),(312,424)]
        self._child = None

    def can_start(self, world):
        return int(read_ram_value(world.ram, "tilemap")) == 0x0B

    def _walk(self, world, target):
        # 0xA6 is the gate of the fenced pen around barrel 7 (tile (33,54) on the festival map); without it
        # that barrel is unreachable and 3 of 12 RNG draws ended the hunt with "no cave floor path".
        self._child = CaveWalkTask(target=target,radius=5,mid_tol=5,timeout=2400,
                                  walkable=frozenset({0xA0,0xA3,0xA6,0xA7,0xA8}))
        self._child.reset(world)

    def step(self, world):
        from harvest.core.ram_catalog import read_ram_u8
        self._steps += 1
        ram = world.ram
        held = int(read_held_item(ram))
        lock = int(read_ram_value(ram, "input_lock"))
        flags = read_ram_u16(ram, 0x11F74)
        tm = int(read_ram_value(ram, "tilemap"))
        score = read_ram_u8(ram, 0x09A2)  # one byte; $09A3 is the opponent's score
        colour = read_ram_u8(ram, 0x09AC)
        if self._steps > self.timeout:
            return TaskResult(TaskStatus.FAILURE, reason=f"egg hunt timeout in {self._phase}, score={score}")
        if self._phase == "award" and tm in (0x15,0x16,0x17):
            new_prize = read_ram_u16(ram, 0x11F6C) & 0x3000 & ~self._prizes
            if new_prize & 0x2000 and read_ram_u8(ram, 0x25) == 15:
                return TaskResult(TaskStatus.SUCCESS, reason=f"egg festival won with {self._score}; house berry armed")
            if new_prize & 0x1000:
                # The prize roll is a coin flip: the turtle shell (+21 ranch points). The next win arms the berry.
                return TaskResult(TaskStatus.SUCCESS, reason=f"egg festival won with {self._score}: turtle shell")
            if read_ram_u8(ram, 0x25) == 15 and not new_prize:
                return TaskResult(TaskStatus.FAILURE, reason=f"egg festival lost with {self._score}: no new prize")
        if self._phase != "award" and flags & 0x10:
            self._phase, self._score = "award", score
            self._goals, self._child = [], None
        if self._phase == "start" and flags & 8:
            self._phase, self._score = "find", score
        # The old colour remains live for a frame after the mayor's next
        # question closes. Do not fetch it again before the new request loads.
        if self._phase == "question_ready":
            self._phase, self._target = "find", 0
        elif self._phase == "question_close" and lock == 1:
            self._phase = "question_ready"
        elif self._phase == "question" and lock == 2:
            self._phase = "question_close"
        if self._phase == "find" and lock == 1 and colour in range(1,6) and colour != self._target:
            matches = [i for i in range(1,8) if read_ram_u8(ram, 0x09A4+i) == colour]
            if matches:
                x,y = self.SITES[matches[0]]
                self._walk(world, (x//16,(y+20)//16))
                self._phase, self._target = "pick", colour
                self._goals = []
        if self._phase == "pick" and held == 0x1D + self._target:
            self._walk(world, (19,26))
            self._phase = "give"
        if self._phase == "give" and score > self._score:
            self._score, self._phase, self._child = score, "question", None
        pad = make_action()
        if lock == 2:
            pad = make_action(a=self._steps % 24 < 6)
        elif self._child:
            result = self._child.step(world)
            if result.status == TaskStatus.RUNNING:
                return result
            self._child = None
            if result.status != TaskStatus.SUCCESS:
                return result
        elif self._goals:
            pos = get_pos_from_ram(ram)
            dx,dy = self._goals[0][0]-pos.x,self._goals[0][1]-pos.y
            if abs(dx)<=2 and abs(dy)<=2:
                self._goals.pop(0)
            else:
                face = ("right" if dx>0 else "left") if abs(dx)>2 else ("down" if dy>0 else "up")
                pad = make_action(**{face:True,"b":True})
        elif self._phase == "pick":
            pad = make_action(up=True,a=self._steps % 24 < 6)
        elif self._phase in {"give", "start"}:
            pad = (make_action(left=True) if int(read_ram_value(ram,"player_direction")) != 3
                   else make_action(a=self._steps % 24 < 6))
        return TaskResult(TaskStatus.RUNNING, action=ActionResult(pad),
                          reason=f"egg hunt {self._phase} score={score} colour={colour}")
