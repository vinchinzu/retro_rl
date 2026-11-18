"""Barn chores: feed the adults, cure the sick, talk to and brush every cow, milk and ship.

The chores are a plan read from RAM. Each step asks what the ROM says is still owed (``_next_job``), walks
there on the measured barn map (``BarnWalker``), presses once and reads the ROM's receipt. Nothing here
remembers a route, so a cow in the way, a lost press or a knock off the lane only costs one retry. Why
the walk and the presses look the way they do is in ``cow_geometry``.

Order: fodder first (an unfed barn adult rolls sickness at night), then the cure, then one pass over the
cows with the tool in hand (talk and brush, or milk and ship) and one with the other: an X swap of the carry
pair is a 69-frame animation, so it is paid once a day, not once a cow. Within a pass the next cow is the
nearest by the walk map (``walk_distances``), not by the straight line. The barn clock does not run, so a
chore pass costs frames, not hours. A job that fails twice is left for this pass (a cow whose job struck out
waits ``RETRY_COOLDOWN_FRAMES`` while others are owed, and a press that missed is not repeated from the same
stand), and the pass then fails with a reason the day plan answers by re-entering the barn, which puts every
cow back in its stall. The pass walks out on its own only when a cow nears the door or it runs long
(``_shift_due``): a reset is ~650 frames, and jobs run ~10% slower per 10,000 frames the herd has scattered.

Where the frames go (12 adult cows, Y3 morning pins, ~22,500 a day): from 11 cows up the ROM runs the barn at
one game tick per two frames (9 cows or fewer: one tick a frame), the farmer covers 2 px a tick, and every
action locks him for a fixed time (fodder 55, trough 43, talk 105, brush 136, milk 154, ship 46 frames).
A day of full care is ~6,500 frames of locks and ~10,400 px of walking (a forkful at a time from the
dispenser, a bottle at a time to the bin in the far corner): a floor of about 17,700 frames before resets.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from harvest.core.animal_probe import BARN_TILEMAP
from harvest.core.animal_status import (
    ADDR_FED_COWS_N,
    ADDR_HELD_ITEM,
    ADDR_NUM_COWS,
    ADDR_STORED_GRASS,
    COW_DAILY_BRUSHED_FLAG,
    COW_DAILY_MILKED_FLAG,
    COW_DAILY_TALKED_FLAG,
    COW_STATUS_ADULT_FLAG,
    COW_STATUS_PREGNANT_FLAG,
    ITEM_FODDER,
    count_cow_slots,
    cow_is_sick,
    cow_needs_milking,
    cows_outside,
    existing_cow_slots,
    read_cow_daily_flags,
    read_cow_happiness,
    read_cow_status,
    read_cow_timer,
    read_fed_cows_flags,
    read_held_item,
    read_num_cows,
    read_stored_grass,
)
from harvest.core.ram_catalog import field_spec, read_animal_slot_field, read_ram_u8, read_ram_value
from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP, Tool
from harvest.tasks.cow_geometry import (
    BARN_DOOR_CENTER_PX,
    BARN_DOOR_DANGER_PX,
    BARN_SHIP_BIN_FACE,
    BARN_SHIFT_FRAMES,
    BARN_SHIFT_REASON,
    BARN_SHIP_BIN_PX,
    BIN_HOLD_PX,
    BLOCKED_REASON,
    COW_EDGE_PX,
    COW_FEED_SPOTS,
    FODDER_FACE,
    FODDER_NORTH_STAND_PX,
    FODDER_STAND_PX,
    PREGNANT_COW_FEED_SPOT,
    BarnWalker,
    CowFeedSpot,
    OPPOSITE_FACE,
    Pixel,
    axis_step,
    barn_cow_pixels,
    cow_stands,
    next_unfed_spot,
    press_reaches,
    walk_cost,
    walk_distances,
)
from harvest.tasks.harvest_task import read_shipping_money
from harvest.tasks.animal_navigation import find_path_around_blockers
from harvest.tasks.nav import Navigator, Pathfinder, get_pos_from_ram, make_action
from harvest.tasks.primitives import press_a_sequence
from retro_harness import ActionResult, Task, TaskResult, TaskStatus, WorldState

ADDR_TOOL_SELECTED = field_spec("tool_selected").address
ADDR_TOOL_BACKPACK = field_spec("tool_backpack").address
ADDR_PLAYER_ACTION = field_spec("player_action").address
BRUSH_TOOL_ID = int(Tool.BRUSH)
MILKER_TOOL_ID = int(Tool.MILKER)
MEDICINE_TOOL_ID = 9

FODDER, TROUGH, SHIP = "fodder", "trough", "ship"
TALK, BRUSH, MILK, MEDICINE = "talk", "brush", "milk", "medicine"
JOB_TOOLS = {BRUSH: BRUSH_TOOL_ID, MILK: MILKER_TOOL_ID, MEDICINE: MEDICINE_TOOL_ID}
# Frames a job may take, walk and retries together. A full crossing of the barn runs about 400.
JOB_FRAMES = {FODDER: 1500, TROUGH: 1500, SHIP: 3000, TALK: 1200, BRUSH: 1200, MILK: 1200, MEDICINE: 1500}
MAX_PRESSES = 4
RETRY_COOLDOWN_FRAMES = 600  # a cow whose job struck out is left for the others this long: she may have moved
TRIED_STAND_PX = 8  # a stand this close to one whose press missed is the same stand
TRIED_STAND_COST = 300
WATCHED_JOBS = (FODDER, TALK, BRUSH, MILK, MEDICINE)  # walks with empty hands: a shift can leave at once
MAX_STRIKES = 2  # a job that failed this often waits for the next pass
SELECT_FRAMES = 90
SETTLE_FRAMES = 6  # pad free and farmer idle this long before a job is judged
STEP_IN_FRAMES = 30  # at most, from the stand to the cow's edge; a turn alone costs a few frames
STEP_BLOCKED_FRAMES = 3  # still this long while stepping: pushing her, which is the contact a tool reads
COW_DRIFT_PX = 5  # she moved this far since the stand was picked: pick again
COW_STILL_FRAMES = 6  # she has stopped walking; a swing lands about 65 frames after the press
VERIFY_FRAMES = 60  # no receipt this long after a settled press: the press missed
MEDICINE_VERIFY_FRAMES = 240  # the bottle leaves the pair first, the sick bit falls after the swing
HAY_NOTE = "hay_short"
# Happiness is a u8 that ``AddCowHappiness`` ($84A5D4) clamps at 255, and the night pass only ever takes it down (unfed, sick, the
# weather), so a talk (+1) and a brush (+3) on a cow at the cap change nothing the ranch score or the milk price can see.
CARE_CAP = 255
SKIP_CAPPED_ENV = "HARVEST_BARN_SKIP_CAPPED"  # =1: a cow at the cap is owed no talk or brush (off: the bench and ledgers expect the flags)
LAST_SHIP_HOUR = 17  # a bottle in the bin from 17:00 is destroyed unpaid (``Dropedonsaleplace``)


@dataclass
class Job:
    kind: str
    slot: Optional[int] = None
    spot: Optional[CowFeedSpot] = None
    stage: str = "walk"
    frames: int = 0
    presses: int = 0
    wait: int = 0
    settled: int = 0
    select_frames: int = 0
    stand: Optional[Tuple[Pixel, str]] = None
    anchor: Pixel = (-1, -1)
    cow_last: Pixel = (-1, -1)
    cow_still: int = 0
    step_frames: int = 0
    step_still: int = 0
    step_last: Pixel = (-1, -1)
    held_before: int = 0
    flags_before: int = 0
    happiness_before: int = 0

    @property
    def key(self) -> Tuple[str, Optional[int]]:
        return self.kind, self.slot


@dataclass
class CowChoresTask(Task):
    """Feed, cure, talk, brush, milk and ship inside the barn."""

    name: str = "cow_chores"
    talk: bool = True
    brush: bool = True
    milk: bool = True
    feed: bool = True
    medicine: bool = True  # cure a sick adult cow when medicine is in the carry pair
    timeout: int = 30000
    shift_frames: int = BARN_SHIFT_FRAMES
    skip_capped: bool = field(default_factory=lambda: os.environ.get(SKIP_CAPPED_ENV) == "1")  # opt in, see ``CARE_CAP``

    _walker: BarnWalker = field(default_factory=BarnWalker, init=False)
    _job: Optional[Job] = field(default=None, init=False)
    _action_queue: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _step_count: int = field(default=0, init=False)
    _strikes: dict = field(default_factory=dict, init=False)
    _tried: dict = field(default_factory=dict, init=False)  # (kind, slot) -> stands whose press got no receipt
    _cooldown: dict = field(default_factory=dict, init=False)  # slot -> step the struck-out job may be tried again
    _ship_money_before: int = field(default=0, init=False)
    talked: bool = field(default=False, init=False)
    brushed: bool = field(default=False, init=False)
    cured_slots: set = field(default_factory=set, init=False)
    milked_count: int = field(default=0, init=False)
    milk_shipped_count: int = field(default=0, init=False)
    fed_count: int = field(default=0, init=False)
    happiness_deltas: list = field(default_factory=list, init=False)

    def reset(self, world: WorldState) -> None:
        ram = world.ram
        self._step_count = 0
        self._job = None
        self._strikes = {}
        self._tried = {}
        self._cooldown = {}
        self._action_queue.clear()
        self._walker.clear()
        self.talked = self.brushed = False
        self.cured_slots = set()
        self.milked_count = self.milk_shipped_count = self.fed_count = 0
        self.happiness_deltas = []
        cows = max(read_num_cows(ram), count_cow_slots(ram))
        print(f"[COW] cows={cows} fed={self._fed_trough_count(ram)}/{len(self._required_spots(ram))} hay={read_stored_grass(ram)}")

    def can_start(self, world: WorldState) -> bool:
        return (int(world.ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(world.ram) else 0) == BARN_TILEMAP

    def resume_after_hotswap(self, world: WorldState) -> None:
        self._action_queue.clear()
        self._walker.clear()
        self._job = None

    def progress_text(self) -> str:
        return (
            f"talk={'Y' if self.talked else 'N'} brush={'Y' if self.brushed else 'N'} "
            f"milk={self.milked_count} fed={self.fed_count}"
        )

    # ── what the ROM says is owed ─────────────────────────────────────

    def _barn_slots(self, ram: np.ndarray) -> list[int]:
        """Cows in the barn. One that walked out of the door keeps the season number as her home map."""
        return [slot for slot in existing_cow_slots(ram) if read_animal_slot_field(ram, "cow", slot, "home_map_raw") == BARN_TILEMAP]

    def _required_spots(self, ram: np.ndarray) -> tuple[CowFeedSpot, ...]:
        """Troughs the night pass checks: one per barn adult, by slot bit; a pregnant cow has her own.

        A child or a calf is never tested for a fed bit (``CowFeedingandStatus`` branches away before the
        test, bank_83 ``.babyonfarm``), so its hay is saved.
        """
        flags = set()
        for slot in self._barn_slots(ram):
            status = read_cow_status(ram, slot)
            if status & COW_STATUS_ADULT_FLAG:
                flags.add(0x1000 if status & COW_STATUS_PREGNANT_FLAG else 1 << slot)
        return tuple(spot for spot in (*COW_FEED_SPOTS, PREGNANT_COW_FEED_SPOT) if spot.flag in flags)

    def _fed_trough_count(self, ram: np.ndarray) -> int:
        flags = read_fed_cows_flags(ram)
        return sum(bool(flags & spot.flag) for spot in self._required_spots(ram))

    def _carried(self, ram: np.ndarray) -> Tuple[int, int]:
        return read_ram_u8(ram, ADDR_TOOL_SELECTED), read_ram_u8(ram, ADDR_TOOL_BACKPACK)

    def _needs(self, ram: np.ndarray, slot: int) -> list[str]:
        """Care still owed to ``slot``, in the order it is done. Respects the phase's switches and the carry pair."""
        pair = self._carried(ram)
        flags = read_cow_daily_flags(ram, slot)
        needs = []
        if self.medicine and MEDICINE_TOOL_ID in pair and cow_is_sick(ram, slot) and read_cow_status(ram, slot) & COW_STATUS_ADULT_FLAG:
            needs.append(MEDICINE)
        capped = self.skip_capped and read_cow_happiness(ram, slot) >= CARE_CAP
        if self.talk and not capped and not flags & COW_DAILY_TALKED_FLAG:
            needs.append(TALK)
        if self.brush and not capped and BRUSH_TOOL_ID in pair and not flags & COW_DAILY_BRUSHED_FLAG and read_ram_value(ram, "stamina") > 0:
            needs.append(BRUSH)  # an empty bar plays the tired animation instead of the swing
        if self.milk and MILKER_TOOL_ID in pair and cow_needs_milking(ram, slot) and read_ram_value(ram, "hour") < LAST_SHIP_HOUR:
            needs.append(MILK)
        return needs

    def care_receipt(self, ram: np.ndarray) -> dict[int, dict[str, bool]]:
        """Per-cow daily receipt: what is done and what is still owed under this phase's switches."""
        receipt: dict[int, dict[str, bool]] = {}
        for slot in self._barn_slots(ram):
            flags = read_cow_daily_flags(ram, slot)
            needs = self._needs(ram, slot)
            capped = self.skip_capped and read_cow_happiness(ram, slot) >= CARE_CAP
            receipt[slot] = {
                "talked": bool(flags & COW_DAILY_TALKED_FLAG) or capped,
                "brushed": bool(flags & COW_DAILY_BRUSHED_FLAG) or capped,
                "milked": bool(flags & COW_DAILY_MILKED_FLAG),
                "cured": slot in self.cured_slots,
                "talk_needed": TALK in needs,
                "brush_needed": BRUSH in needs,
                "milk_needed": MILK in needs,
                "medicine_needed": MEDICINE in needs,
                "complete": not needs,
            }
        return receipt

    def _cow_pixel(self, ram: np.ndarray, slot: int) -> Pixel:
        return (
            int(read_animal_slot_field(ram, "cow", slot, "position_x")),
            int(read_animal_slot_field(ram, "cow", slot, "position_y")),
        )

    def _herd_pixels(self, ram: np.ndarray, *, skip: Optional[int] = None) -> list[Pixel]:
        return [self._cow_pixel(ram, slot) for slot in self._barn_slots(ram) if slot != skip]

    def _open(self, kind: str, slot: Optional[int] = None) -> bool:
        return self._strikes.get((kind, slot), 0) < MAX_STRIKES

    def _cooling(self, slot: int) -> bool:
        return self._cooldown.get(slot, 0) > self._step_count

    def _next_job(self, ram: np.ndarray) -> Optional[Job]:
        held = read_held_item(ram)
        spots = self._required_spots(ram)
        spot = next_unfed_spot(read_fed_cows_flags(ram), spots)
        if held == ITEM_FODDER:
            return Job(TROUGH, spot=spot or spots[0] if spots else COW_FEED_SPOTS[0])
        if held:
            return Job(SHIP)
        if self.feed and spot is not None and read_stored_grass(ram) > 0 and self._open(FODDER):
            return Job(FODDER)
        steps = walk_distances(self._pos(ram))
        for kinds in self._care_groups(ram):
            if MILK in kinds and self._bin_taken(ram):
                continue  # a bottle with nowhere to go would be thrown away: the other pass first, then the reset clears the stand
            owed = [
                (walk_cost(steps, self._cow_pixel(ram, slot)), slot, kind)
                for slot in self._barn_slots(ram)
                for kind in self._needs(ram, slot)
                if kind in kinds and self._open(kind, slot)
            ]
            rested = [row for row in owed if not self._cooling(row[1])]
            if rested or owed:
                _distance, slot, kind = min(rested or owed, key=lambda row: (row[0], row[1], kinds.index(row[2])))
                return Job(kind, slot=slot)
        return None

    def _care_groups(self, ram: np.ndarray) -> tuple[tuple[str, ...], ...]:
        """Care in the order it is walked: the cure, then the pass whose tool is in hand.

        An X swap of the carry pair is a 69-frame animation, so a day pays it once, not once a cow: one pass
        over every cow with the tool in hand, one swap, one pass with the other. The milker's pass comes first
        when it is selected (a day starts with it), otherwise talk and brush do.
        """
        if self._carried(ram)[0] == MILKER_TOOL_ID:
            return ((MEDICINE,), (MILK,), (TALK, BRUSH))
        return ((MEDICINE,), (TALK, BRUSH), (MILK,))

    def _bin_taken(self, ram: np.ndarray) -> bool:
        """A cow on the bin stand, in a corner she cannot be shoved out of."""
        return any(
            abs(cx - BARN_SHIP_BIN_PX[0]) <= COW_EDGE_PX and abs(cy - BARN_SHIP_BIN_PX[1]) <= COW_EDGE_PX
            for cx, cy in self._herd_pixels(ram)
        )

    def _cow_near_door(self, ram: np.ndarray) -> bool:
        return any(
            math.hypot(cx - BARN_DOOR_CENTER_PX[0], cy - BARN_DOOR_CENTER_PX[1]) <= BARN_DOOR_DANGER_PX
            for slot in self._barn_slots(ram)
            if not read_cow_status(ram, slot) & COW_STATUS_PREGNANT_FLAG
            for cx, cy in (self._cow_pixel(ram, slot),)
        )

    def _shift_due(self, ram: np.ndarray) -> bool:
        """Walk out and back in (every cow is put back in her stall) when a cow nears the door or the pass runs long."""
        if read_held_item(ram) or not (self.talk or self.brush or self.milk or self.feed):
            return False  # a bottle or a forkful stays in hand; the cure-only pass is short
        return self._step_count > self.shift_frames or self._cow_near_door(ram)

    # ── the tick ──────────────────────────────────────────────────────

    def _pos(self, ram: np.ndarray) -> Pixel:
        point = get_pos_from_ram(ram)
        return point.x, point.y

    def step(self, world: WorldState) -> TaskResult:
        ram = world.ram
        self._step_count += 1
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{BLOCKED_REASON}: chores over {self.timeout}f")
        tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
        if tilemap != BARN_TILEMAP:
            return TaskResult(status=TaskStatus.BLOCKED, reason=f"not in barn tilemap=0x{tilemap:02X}")
        if self._action_queue:
            return self._running(self._action_queue.popleft())
        from harvest.planner.tasks.transitions import screen_faded_in

        if not screen_faded_in(ram):
            return self._running()  # the pad is dead through the fade: a walk would read that as stuck
        lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
        if lock != 1:
            return self._running(make_action(a=self._step_count % 22 in range(6, 12)))  # taps, a held A does not page text
        if self._job is None:
            self._job = self._next_job(ram)
            if self._job is None:
                return self._finish(ram)
            self._walker.clear()
            if self._shift_due(ram):
                return TaskResult(status=TaskStatus.FAILURE, reason=f"{BARN_SHIFT_REASON} at {self._step_count}f, {self.progress_text()}")
        if self._job.stage == "walk" and self._job.kind in WATCHED_JOBS and self._cow_near_door(ram) and self._shift_due(ram):
            # A cow walked from 57 px to the door tile in 350 frames, inside one job: do not wait for the next boundary.
            self._job = None
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{BARN_SHIFT_REASON} at {self._step_count}f, {self.progress_text()}")
        return self._run_job(ram, self._job)

    def _running(self, action: Optional[np.ndarray] = None) -> TaskResult:
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action if action is not None else make_action()))

    def _finish(self, ram: np.ndarray) -> TaskResult:
        flags = read_fed_cows_flags(ram)
        hay = read_stored_grass(ram)
        spot = next_unfed_spot(flags, self._required_spots(ram))
        if self.feed and spot is not None and hay > 0:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"missing required cow feed flags=0x{flags:04X}")
        receipt = self.care_receipt(ram)
        if receipt:
            self.talked = bool(self.talk) and all(row["talked"] for row in receipt.values())
            self.brushed = bool(self.brush and BRUSH_TOOL_ID in self._carried(ram)) and all(row["brushed"] for row in receipt.values())
        incomplete = {
            slot: [kind for kind in (TALK, BRUSH, MILK, MEDICINE) if row[f"{kind}_needed"]]
            for slot, row in receipt.items()
            if not row["complete"]
        }
        if incomplete:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"incomplete cow care {incomplete}")
        happy = ",".join(f"{slot}:{read_cow_happiness(ram, slot)}" for slot in existing_cow_slots(ram))
        short = f" {HAY_NOTE}=1" if self.feed and spot is not None else ""
        return TaskResult(
            status=TaskStatus.SUCCESS,
            reason=(
                f"talk={self.talked} brush={self.brushed} milk={self.milked_count} ship={self.milk_shipped_count} "
                f"fed={self.fed_count} flags=0x{flags:04X} happy={happy}{short}"
            ),
        )

    # ── one job: walk, press, read the receipt ────────────────────────

    def _run_job(self, ram: np.ndarray, job: Job) -> TaskResult:
        job.frames += 1
        if job.frames > JOB_FRAMES[job.kind] and job.stage != "verify":
            return self._job_failed(ram, job, "over budget")
        if job.stage == "walk":
            if job.kind in JOB_TOOLS and not self._tool_ready(ram, job):
                return self._select_tool(ram, job)
            goal, tolerance = self._job_stand(ram, job)
            action = self._walker.step(self._pos(ram), goal, tolerance=tolerance, cows=barn_cow_pixels(ram))
            if self._walker.blocked:
                return self._job_failed(ram, job, f"walk blocked at {self._pos(ram)}")
            if action is not None:
                return self._running(action)
            if job.kind == SHIP and self._bin_taken(ram):
                return self._running()  # waiting at BIN_HOLD_PX; the budget below still ends it, with the bottle in hand
            if job.presses >= MAX_PRESSES:
                return self._job_failed(ram, job, "presses spent")
            if job.slot is None:
                return self._press(ram, job)
            if job.cow_still < COW_STILL_FRAMES:
                return self._running()
            job.stage, job.step_frames, job.step_still, job.step_last = "step", 0, 0, (-1, -1)
        if job.stage == "step":
            return self._step_in(ram, job)
        return self._verify(ram, job)

    def _step_in(self, ram: np.ndarray, job: Job) -> TaskResult:
        """Walk from the stand into her until she stops the farmer, then press."""
        here = self._pos(ram)
        job.step_frames += 1
        job.step_still = job.step_still + 1 if here == job.step_last else 0
        job.step_last = here
        if job.step_still >= STEP_BLOCKED_FRAMES or job.step_frames >= STEP_IN_FRAMES:
            return self._press(ram, job)
        return self._running(make_action(**{job.stand[1]: True}))

    def _tool_ready(self, ram: np.ndarray, job: Job) -> bool:
        return self._carried(ram)[0] == JOB_TOOLS[job.kind]

    def _select_tool(self, ram: np.ndarray, job: Job) -> TaskResult:
        """X swaps the carry pair. Pulse it with a gap so each press is an edge, never during a swing."""
        if JOB_TOOLS[job.kind] not in self._carried(ram):
            return self._job_done(ram, job)  # the need is gone (a spent bottle); the next plan drops it
        if read_ram_u8(ram, ADDR_PLAYER_ACTION) != 0:
            return self._running()
        job.select_frames += 1
        if job.select_frames > SELECT_FRAMES:
            return self._job_failed(ram, job, "tool would not select")
        return self._running(make_action(x=job.select_frames % 6 == 1))

    def _job_stand(self, ram: np.ndarray, job: Job) -> Tuple[Pixel, int]:
        if job.kind == FODDER:
            # A run stops within 2 px of the stand: at x=214 the press is 2 px short of the dispenser and every
            # third pickup missed (a 60-frame verify and a second press). Within 1 px it always reaches.
            return self._fodder_stand(ram), 1
        if job.kind == TROUGH:
            return job.spot.interact_px, 2
        if job.kind == SHIP:
            if self._bin_taken(ram):
                return BIN_HOLD_PX, 4  # a cow parked on the stand with a bottle in hand: wait for her out of her way
            here = self._pos(ram)
            # The near stand pays without threading the lower-left corner.
            # Its y coordinate matters: a throw at y=337 is destroyed.
            if abs(here[0] - BARN_SHIP_BIN_PX[0]) <= 3 and abs(here[1] - BARN_SHIP_BIN_PX[1]) <= 1:
                return here, 0
            return BARN_SHIP_BIN_PX, 1
        cow = self._cow_pixel(ram, job.slot)
        job.cow_still = job.cow_still + 1 if cow == job.cow_last else 0
        job.cow_last = cow
        if job.stand is None or max(abs(cow[0] - job.anchor[0]), abs(cow[1] - job.anchor[1])) >= COW_DRIFT_PX:
            job.stand, job.anchor = self._pick_cow_stand(ram, job, cow), cow
        return job.stand[0], 2

    def _pick_cow_stand(self, ram: np.ndarray, job: Job, cow: Pixel) -> Tuple[Pixel, str]:
        """The closest stand in line with her that no other cow occupies or would take the press from.

        The ROM gives a press to the first object in its box, so a dose aimed past another cow is wasted.
        """
        here = self._pos(ram)
        others = self._herd_pixels(ram, skip=job.slot)

        def cost(row: Tuple[Pixel, str]) -> int:
            stand, face = row
            occupied = any(abs(ox - stand[0]) <= COW_EDGE_PX and abs(oy - stand[1]) <= COW_EDGE_PX for ox, oy in others)
            stolen = any(press_reaches(stand, face, other) for other in others)
            return abs(stand[0] - here[0]) + abs(stand[1] - here[1]) + 200 * occupied + (600 if job.kind == MEDICINE else 80) * stolen

        tried = self._tried.get(job.key, ())

        def cost_tried(row: Tuple[Pixel, str]) -> int:
            """A stand whose press got no receipt is the last one to try again: a cow pinned to a wall can take a
            tool from one side and not another (a milk bottle came out with no milked flag from her east side)."""
            stand, face = row
            missed = any(face == other_face and abs(stand[0] - sx) <= TRIED_STAND_PX and abs(stand[1] - sy) <= TRIED_STAND_PX
                         for (sx, sy), other_face in tried)
            return cost(row) + TRIED_STAND_COST * missed

        rows = cow_stands(cow)
        return min(rows, key=cost_tried) if rows else ((cow[0] + 17, cow[1]), "left")

    def _fodder_stand(self, ram: np.ndarray) -> Pixel:
        """The dispenser stand no cow is standing on: A would talk to her instead of taking the fodder."""
        cows = self._herd_pixels(ram)

        def crowded(stand: Pixel) -> bool:
            return any(abs(cx - stand[0]) <= COW_EDGE_PX and abs(cy - stand[1]) <= COW_EDGE_PX for cx, cy in cows)

        return FODDER_NORTH_STAND_PX if crowded(FODDER_STAND_PX) and not crowded(FODDER_NORTH_STAND_PX) else FODDER_STAND_PX

    def _press(self, ram: np.ndarray, job: Job) -> TaskResult:
        job.presses += 1
        job.stage = "verify"
        job.wait = job.settled = 0
        job.held_before = read_held_item(ram)
        if job.slot is not None:
            job.flags_before = read_cow_daily_flags(ram, job.slot)
            job.happiness_before = read_cow_happiness(ram, job.slot)
        if job.kind == FODDER:
            queue = press_a_sequence(FODDER_FACE, face_frames=4, pre_press_settle_frames=0, hold_frames=10, settle_frames=4, hold_face_with_a=True)
        elif job.kind == TROUGH:
            queue = press_a_sequence(job.spot.face, face_frames=4, pre_press_settle_frames=0, hold_frames=8, settle_frames=4)
        elif job.kind == SHIP:
            self._ship_money_before = read_shipping_money(ram)
            queue = press_a_sequence(BARN_SHIP_BIN_FACE, face_frames=8, pre_press_settle_frames=0, hold_frames=16, settle_frames=24, hold_face_with_a=True)
        else:
            button = "a" if job.kind == TALK else "y"
            queue = [make_action(), make_action(), *(make_action(**{button: True}) for _ in range(8))]
        self._action_queue.extend(queue)
        return self._running(self._action_queue.popleft())

    def _receipt_landed(self, ram: np.ndarray, job: Job) -> bool:
        if job.kind == FODDER:
            return read_held_item(ram) == ITEM_FODDER
        if job.kind == TROUGH:
            return read_held_item(ram) != ITEM_FODDER and bool(read_fed_cows_flags(ram) & job.spot.flag)
        if job.kind == SHIP:
            return read_held_item(ram) == 0 and read_shipping_money(ram) > self._ship_money_before
        flags = read_cow_daily_flags(ram, job.slot)
        if job.kind == TALK:
            return bool(flags & COW_DAILY_TALKED_FLAG)
        if job.kind == BRUSH:
            return bool(flags & COW_DAILY_BRUSHED_FLAG)
        if job.kind == MILK:
            return bool(flags & COW_DAILY_MILKED_FLAG)
        return not cow_is_sick(ram, job.slot)

    def _verify(self, ram: np.ndarray, job: Job) -> TaskResult:
        busy = read_ram_u8(ram, ADDR_PLAYER_ACTION) != 0
        job.settled = 0 if busy else job.settled + 1
        job.wait += 1
        if job.settled < SETTLE_FRAMES and job.wait < MEDICINE_VERIFY_FRAMES:
            return self._running()
        if self._receipt_landed(ram, job):
            return self._job_done(ram, job)
        limit = MEDICINE_VERIFY_FRAMES if job.kind == MEDICINE else VERIFY_FRAMES
        if job.wait < limit:
            return self._running()
        if job.kind == MEDICINE and MEDICINE_TOOL_ID not in self._carried(ram):
            return self._job_failed(ram, job, "dose spent without a cure")
        where = ""
        if job.slot is not None:
            cow, here = self._cow_pixel(ram, job.slot), self._pos(ram)
            where = f" cow={cow} farmer={here} stand={job.stand} anchor={job.anchor}"
            if job.stand is not None:
                self._tried.setdefault(job.key, []).append(job.stand)
            if job.kind == MILK and read_held_item(ram) and not job.held_before:
                # The bottle is made (a press with the hand full does nothing): ship it, then come back from
                # another side for the flag.
                return self._job_failed(ram, job, "milk made without the milked flag")
        job.stage, job.stand = "walk", None
        print(f"[COW] {job.kind.title()} retry slot={job.slot} presses={job.presses}{where}")
        return self._running()

    def _job_done(self, ram: np.ndarray, job: Job) -> TaskResult:
        self._job = None
        self._walker.clear()
        if job.kind == TROUGH:
            flags = read_fed_cows_flags(ram)
            self.fed_count += 1
            spots = self._required_spots(ram)
            print(f"[COW] Feed OK count={self._fed_trough_count(ram)} remaining={len(spots) - self._fed_trough_count(ram)} flags=0x{flags:04X}")
        elif job.kind == SHIP:
            self.milk_shipped_count += 1
            print(f"[COW] Milk shipped money={read_shipping_money(ram)}")
        elif job.kind in (TALK, BRUSH, MILK) and self._receipt_landed(ram, job):
            after = read_cow_happiness(ram, job.slot)
            self.happiness_deltas.append(
                {"slot": job.slot, "kind": job.kind, "before": job.happiness_before, "after": after, "delta": after - job.happiness_before}
            )
            self.milked_count += job.kind == MILK
            print(f"[COW] {job.kind.title()} OK slot={job.slot} presses={job.presses} frames={job.frames}")
        elif job.kind == MEDICINE and self._receipt_landed(ram, job):
            self.cured_slots.add(job.slot)
            print(f"[COW] Cure OK slot={job.slot} timer={read_cow_timer(ram, job.slot)} presses={job.presses}")
        return self._running()

    def _job_failed(self, ram: np.ndarray, job: Job, why: str) -> TaskResult:
        self._job = None
        self._walker.clear()
        self._strikes[job.key] = self._strikes.get(job.key, 0) + 1
        if job.slot is not None:
            self._cooldown[job.slot] = self._step_count + RETRY_COOLDOWN_FRAMES
        print(f"[COW] {job.kind} failed slot={job.slot} strike={self._strikes[job.key]} frames={job.frames} presses={job.presses} at {self._pos(ram)}: {why}")
        if job.kind in (TROUGH, SHIP) and self._strikes[job.key] >= MAX_STRIKES:
            # Something is in the hand and the walk cannot deliver it: only a fresh barn clears this.
            return TaskResult(status=TaskStatus.FAILURE, reason=f"{BLOCKED_REASON} at {self._pos(ram)} job={job.kind}")
        return self._running()


ROUNDUP_DOOR_PX: Pixel = (328, 360)  # farm tile (20,22), $E1: a cow stepping on it is back in her stall
ROUNDUP_ROW_Y = 376  # the open path south of the barn; she spawns on it
ROUNDUP_ALIGN_PX = 6
ROUNDUP_STAND_PX = 15
ROUNDUP_PUSH_FRAMES = 90  # still this long: pinned. A run stands ~40 frames at her edge, then shoves 2 px in 3


@dataclass
class CowRoundupTask(Task):
    """Shove a cow that left the barn back onto the door tile: along the path to the door column, then north.

    She spawns one tile south of the door ($84A595) and wanders from there. A cow returns the moment she stands
    on tile $E1 ($84A540), so the job is two straight pushes from behind her. The phase is optional: she is
    alive and fed by the pasture out there, a failed roundup only costs her care for the day.
    """

    name: str = "cow_roundup"
    timeout: int = 6000

    _step_count: int = field(default=0, init=False)
    _push_frames: int = field(default=0, init=False)
    _cow_seen: Pixel = field(default=(-1, -1), init=False)
    _flip: bool = field(default=False, init=False)
    _pathfinder: Pathfinder = field(init=False)
    _navigator: Navigator = field(init=False)

    def __post_init__(self) -> None:
        self._pathfinder = Pathfinder()
        self._navigator = Navigator(self._pathfinder)

    def reset(self, world: WorldState) -> None:
        self._step_count = self._push_frames = 0
        self._flip = False
        self._navigator.update(world.ram)
        self._navigator.path = []

    def can_start(self, world: WorldState) -> bool:
        return int(world.ram[ADDR_TILEMAP]) == 0x00

    def _plan(self, cow: Pixel) -> Tuple[Pixel, str]:
        """Where to stand and which way to run: onto the path row, along it to the door column, then north.

        A cow pinned against a fence on one leg (``_flip``) takes the other leg first.
        """
        off_row = abs(cow[1] - ROUNDUP_ROW_Y) > ROUNDUP_ALIGN_PX
        off_column = abs(cow[0] - ROUNDUP_DOOR_PX[0]) > ROUNDUP_ALIGN_PX
        if off_row and off_column and self._flip:
            off_row = False
        if off_row and off_column or (off_row and cow[1] < ROUNDUP_ROW_Y):
            up = cow[1] > ROUNDUP_ROW_Y
            return (cow[0], cow[1] + (ROUNDUP_STAND_PX if up else -ROUNDUP_STAND_PX)), ("up" if up else "down")
        if off_column:
            side = 1 if cow[0] > ROUNDUP_DOOR_PX[0] else -1
            return (cow[0] + side * ROUNDUP_STAND_PX, cow[1]), ("left" if side > 0 else "right")
        return (cow[0], cow[1] + ROUNDUP_STAND_PX), "up"

    def step(self, world: WorldState) -> TaskResult:
        ram = world.ram
        self._step_count += 1
        self._navigator.update(ram)
        outside = cows_outside(ram)
        if not outside:
            return TaskResult(status=TaskStatus.SUCCESS, reason="herd in the barn")
        if self._step_count > self.timeout:
            return TaskResult(status=TaskStatus.FAILURE, reason=f"roundup over {self.timeout}f, outside={outside}")
        if int(ram[ADDR_TILEMAP]) != 0x00:
            return TaskResult(status=TaskStatus.BLOCKED, reason="roundup is a farm job")
        cow = (
            int(read_animal_slot_field(ram, "cow", outside[0], "position_x")),
            int(read_animal_slot_field(ram, "cow", outside[0], "position_y")),
        )
        stand, push = self._plan(cow)
        here = get_pos_from_ram(ram)
        if abs(here.x - stand[0]) <= 8 and abs(here.y - stand[1]) <= 8:
            self._push_frames = 0 if cow != self._cow_seen else self._push_frames + 1
            self._cow_seen = cow
            if self._push_frames > ROUNDUP_PUSH_FRAMES:
                self._push_frames, self._flip = 0, not self._flip
                return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(**{OPPOSITE_FACE[push]: True, "b": True})))
            self._navigator.path = []
            return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(make_action(**{push: True, "b": True})))
        goal = (stand[0] // 16, stand[1] // 16)
        moved = cow != self._cow_seen
        if moved or not self._navigator.path or self._navigator.stasis > 45:
            # Round her, never through her: a run straight at her shoves her the wrong way across the farm.
            ctile = (cow[0] // 16, cow[1] // 16)
            ring = {(ctile[0] + dx, ctile[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
            self._navigator.path = find_path_around_blockers(ram, self._pathfinder, self._navigator.current_tile, goal, ring - {goal}) or []
            self._navigator.stasis = 0
            self._cow_seen = cow
        action = self._navigator.follow_path(ram) if self._navigator.path else None
        if action is None:
            direction, _remaining = axis_step((here.x, here.y), stand)
            action = make_action(**{direction: True}) if direction else make_action()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action))


__all__ = [
    "ADDR_FED_COWS_N",
    "ADDR_HELD_ITEM",
    "ADDR_NUM_COWS",
    "ADDR_PLAYER_ACTION",
    "ADDR_STORED_GRASS",
    "ADDR_TOOL_BACKPACK",
    "ADDR_TOOL_SELECTED",
    "BRUSH_TOOL_ID",
    "COW_DAILY_BRUSHED_FLAG",
    "COW_DAILY_MILKED_FLAG",
    "COW_DAILY_TALKED_FLAG",
    "COW_FEED_SPOTS",
    "CowChoresTask",
    "CowRoundupTask",
    "ITEM_FODDER",
    "MEDICINE_TOOL_ID",
    "MILKER_TOOL_ID",
    "Job",
]
