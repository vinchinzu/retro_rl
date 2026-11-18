"""Randomized barn trials: any herd, any morning, every care action must land.

A chore pass that works on one timeline proves one timeline. Each trial here takes a real morning pin
(farmer in the house), rewrites the herd and the day from a seed, then runs the planner's own cow
section on the ROM and reads the receipts from RAM: every barn adult fed, every cow talked to and
brushed, every ready cow milked, the milk shipped, hands empty, the farmer out on the farm.

What a seed moves (``make_scenario``):

- herd: size, which slots are filled (dead cows leave holes), adults, bought children, calves, a sick
  cow with a bottle on the shelf, a pregnant cow, a cranky cow, any happiness byte
- hay: plenty, exactly enough, or short; chores already done today; the clock hour
- timing: ROM RNG bytes, frames idled outdoors before the walk in (cows are in their stalls) and in the
  barn before the chores start (cows have wandered, up to minutes)
- a cow already out of the barn, somewhere on the path by the door (``outside``)
- knocks (``kick_rate``): a random direction or a pause forced over the task's action at random frames
  in the barn, so every fixed lane and every queued press also runs from the wrong place

    .venv/bin/python -m harvest.tools.barn_fuzz run <tag> --profile twelve --trials 64 --workers 28
    .venv/bin/python -m harvest.tools.barn_fuzz report <tag>
    uv run python -m harvest.tools.barn_fuzz one --seed 17 --profile twelve   # replay one trial

Trials are research what-ifs (RAM is edited before the first frame); a pass proves a component, never a
route. Reports land in ``recordings/barn_fuzz/<tag>/``; a passing trial's log is deleted.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

GAME = Path(__file__).resolve().parents[2]
PINS = GAME / "custom_integrations" / "HarvestMoon-Snes"
OUT = GAME / "recordings" / "barn_fuzz"

STATUS_ADULT = 0x09
STATUS_CHILD = 0x05
STATUS_BABY = 0x03
STATUS_SICK = 0x29
STATUS_PREGNANT = 0x49
STATUS_CRANKY = 0x19
SHED_ROW_2_ADDR = 0x11F01  # bit 0x01 is the medicine bottle (mint_herd_whatif)
RNG_ADDRS = (0x0100, 0x0101, 0x0102)
BARN_TILEMAP = 0x27
FARM_TILEMAP = 0x00

# Morning pins in the house across the calendar, real farms with real tool shelves.
BASE_PINS = (
    "PR_y3k5_s1_Y3_Spring_D12",
    "PR_y3k5_s1_Y3_Spring_D25",
    "PR_y3k5_s1_Y3_Summer_D5",
    "PR_y3k5_s1_Y3_Summer_D21",
    "PR_y3k5_s1_Y3_Summer_D28",
    "PR_e2es4_Y2_Spring_D20",
    "PR_e2es4_Y2_Summer_D9",
    "PR_e2es4_Y2_Fall_D14",
    "PR_e2es4_Y2_Winter_D11",
    "PR_e2es4_Y1_Winter_D20",
)
# Carry pairs a morning can start with, among them both orders of the cow tools and none at all.
CARRY_PAIRS = ((0x0F, 0x0E), (0x0E, 0x0F), (0x01, 0x02), (0x02, 0x10), (0x0F, 0x00), (0x00, 0x0E), (0x00, 0x00))

PROFILES = ("twelve", "wander", "knock", "herd", "morning", "outside", "all")


@dataclass(frozen=True)
class CowSpec:
    slot: int
    status: int
    timer: int = 0
    happiness: int = 0
    flags: int = 0


@dataclass(frozen=True)
class Scenario:
    seed: int
    profile: str
    base_pin: str
    cows: tuple[CowSpec, ...]
    hay: int
    hour: int
    rng: tuple[int, int, int]
    idle_out: int
    idle_in: int
    kick_rate: float
    shed_medicine: bool = False
    stamina: Optional[int] = None
    pair: Optional[tuple[int, int]] = None
    fed_flags: int = 0
    outside: Optional[tuple[int, int, int]] = None  # slot, x, y of a cow that left the barn earlier

    def herd_label(self) -> str:
        names = {STATUS_ADULT: "A", STATUS_CHILD: "c", STATUS_BABY: "b", STATUS_SICK: "S", STATUS_PREGNANT: "P", STATUS_CRANKY: "K"}
        return "".join(names.get(cow.status, "?") for cow in self.cows)


def _healthy_cow(rng: random.Random, slot: int) -> CowSpec:
    return CowSpec(slot, STATUS_ADULT, happiness=rng.randrange(0, 256))


def _mixed_cow(rng: random.Random, slot: int) -> CowSpec:
    roll = rng.random()
    happiness = rng.randrange(0, 256)
    if roll < 0.60:
        return CowSpec(slot, STATUS_ADULT, happiness=happiness)
    if roll < 0.82:
        return CowSpec(slot, STATUS_CHILD, timer=rng.randrange(0, 21), happiness=happiness)
    if roll < 0.90:
        return CowSpec(slot, STATUS_BABY, timer=rng.randrange(0, 14), happiness=happiness)
    if roll < 0.96:
        return CowSpec(slot, STATUS_SICK, timer=rng.randrange(1, 8), happiness=happiness)
    return CowSpec(slot, STATUS_CRANKY, timer=rng.randrange(1, 4), happiness=happiness)


def make_scenario(seed: int, profile: str) -> Scenario:
    """Everything a trial varies, drawn from ``seed`` alone, so a failure replays from its number."""
    if profile not in PROFILES:
        raise ValueError(f"unknown barn profile: {profile}")
    rng = random.Random(f"barn-{profile}-{seed}")
    chosen = rng.choice(PROFILES[:-1]) if profile == "all" else profile
    base_pin = rng.choice(BASE_PINS)
    hay = 900
    hour = rng.randrange(6, 12)
    idle_out = rng.randrange(0, 400)
    idle_in = 0
    kick_rate = 0.0
    shed_medicine = False
    if chosen == "twelve":
        cows = tuple(_healthy_cow(rng, slot) for slot in range(12))
    elif chosen == "wander":
        cows = tuple(_healthy_cow(rng, slot) for slot in range(12))
        idle_in = rng.choice((60, 300, 900, 1500, 2400))
    elif chosen == "knock":
        cows = tuple(_healthy_cow(rng, slot) for slot in range(12))
        idle_in = rng.choice((0, 0, 300, 2400))
        kick_rate = rng.choice((0.001, 0.003, 0.01))
    elif chosen == "outside":
        cows = tuple(_healthy_cow(rng, slot) for slot in range(rng.choice((3, 8, 12))))
    elif chosen == "morning":
        # Twelve healthy cows, but the day around them is not: other tools in hand, a tired farmer, half-done troughs.
        cows = tuple(_healthy_cow(rng, slot) for slot in range(12))
        hour = rng.choice((6, 8, 11, 14, 16, 17))
        idle_in = rng.choice((0, 300, 900))
    else:
        size = rng.randint(1, 12)
        slots = sorted(rng.sample(range(12), size))
        rows = [_mixed_cow(rng, slot) for slot in slots]
        if rng.random() < 0.2:
            pick = rng.randrange(len(rows))
            rows[pick] = CowSpec(rows[pick].slot, STATUS_PREGNANT, timer=rng.randrange(1, 21), happiness=rows[pick].happiness)
        if rng.random() < 0.25:
            rows = [
                dataclasses.replace(row, flags=rng.choice((0, 0x01, 0x02, 0x04, 0x05, 0x07)) if row.status == STATUS_ADULT else 0)
                for row in rows
            ]
        cows = tuple(rows)
        shed_medicine = any(row.status == STATUS_SICK for row in rows) and rng.random() < 0.7
        hay = rng.choice((900, 900, 300, len(rows), max(0, len(rows) - rng.randint(1, 3)), 0))
        idle_in = rng.choice((0, 0, 300, 1500, 2400))
        kick_rate = rng.choice((0.0, 0.0, 0.002))
        hour = rng.choice((6, 7, 9, 12, 15, 16))
    stamina = pair = outside = None
    fed_flags = 0
    if chosen == "outside":
        # She spawns one tile south of the door and wanders along the path from there.
        outside = (rng.choice([cow.slot for cow in cows]), 328 + rng.randrange(-64, 65, 8), 376 + rng.choice((0, 0, 0, 2)))
    if chosen == "morning":
        stamina = rng.choice((None, None, 120, 60, 20, 5, 0))
        pair = rng.choice(CARRY_PAIRS)
        fed_flags = rng.choice((0, 0, 0x0FFF, 0x0F0F, rng.randrange(0x1000)))
    return Scenario(
        seed=seed,
        profile=chosen,
        base_pin=base_pin,
        cows=cows,
        hay=hay,
        hour=hour,
        rng=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
        idle_out=idle_out,
        idle_in=idle_in,
        kick_rate=kick_rate,
        shed_medicine=shed_medicine,
        stamina=stamina,
        pair=pair,
        fed_flags=fed_flags,
        outside=outside,
    )


def mint_trial_pin(scenario: Scenario, name: str) -> Path:
    """Write the scenario into a throwaway copy of its morning pin."""
    from harvest.core.harvest_state import HarvestStateDocument

    doc = HarvestStateDocument.load(scenario.base_pin)
    doc.clear_cows()
    for cow in scenario.cows:
        doc.set_cow_slot(cow.slot, status_raw=cow.status, raw_1=cow.flags, pregnancy_raw=cow.timer, happiness=cow.happiness)
    if scenario.outside is not None:
        slot, x, y = scenario.outside
        doc.set_cow_field(slot, "home_map_raw", int(doc.scalar_value("season")))
        doc.set_cow_field(slot, "position_x", x)
        doc.set_cow_field(slot, "position_y", y)
    doc.set_scalar_value("fed_cows_flags", scenario.fed_flags)
    doc.set_scalar_value("fed_cows_n", bin(scenario.fed_flags).count("1"))
    if scenario.stamina is not None:
        doc.set_scalar_value("stamina", scenario.stamina)
    if scenario.pair is not None:
        # A tool out of the pair lives on the shed shelf; without the bit the game no longer owns it.
        from harvest.planner.livestock_econ import SHED_ROW_2_BRUSH_BIT, SHED_ROW_2_MILKER_BIT

        doc.set_scalar_value("tool_selected", scenario.pair[0])
        doc.set_scalar_value("tool_backpack", scenario.pair[1])
        shelf = int(doc.mutable_state.read_u8(SHED_ROW_2_ADDR))
        for tool, bit in ((0x0F, SHED_ROW_2_BRUSH_BIT), (0x0E, SHED_ROW_2_MILKER_BIT)):
            shelf = shelf & ~bit if tool in scenario.pair else shelf | bit
        doc.mutable_state.write_u8(SHED_ROW_2_ADDR, shelf)
    doc.set_scalar_value("num_cows", len(scenario.cows))
    doc.set_scalar_value("stored_grass", scenario.hay)
    doc.set_clock(hour=scenario.hour, minute=0, second=0)
    for address, value in zip(RNG_ADDRS, scenario.rng):
        doc.mutable_state.write_u8(address, value)
    if scenario.shed_medicine:
        doc.mutable_state.write_u8(SHED_ROW_2_ADDR, int(doc.mutable_state.read_u8(SHED_ROW_2_ADDR)) | 0x01)
    return doc.save_as(PINS / f"{name}.state")


@dataclass
class Receipt:
    """What the ROM says happened, read after the plan ends."""

    ok: bool = False
    plan: str = ""
    plan_reason: str = ""
    frames: int = 0
    barn_frames: int = 0
    resets: int = 0
    recoveries: int = 0
    tilemap: int = -1
    held: int = 0
    missing: list[int] = field(default_factory=list)
    ram_writes: int = 0
    mid_run_loads: int = 0
    shipping_before: Optional[int] = None
    shipping_after: Optional[int] = None
    milk_value: int = 0
    milk_prices: Optional[dict[int, int]] = None
    unfed: list[int] = field(default_factory=list)
    untalked: list[int] = field(default_factory=list)
    unbrushed: list[int] = field(default_factory=list)
    unmilked: list[int] = field(default_factory=list)
    still_sick: list[int] = field(default_factory=list)
    escaped: list[int] = field(default_factory=list)
    planned_cure: bool = False
    failed_phases: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _tools_in_hand(ram) -> set[int]:
    from harvest.tasks.cow_task import ADDR_TOOL_BACKPACK, ADDR_TOOL_SELECTED

    return {int(ram[ADDR_TOOL_SELECTED]), int(ram[ADDR_TOOL_BACKPACK])}


def judge(ram, scenario: Scenario, task, receipt: Receipt) -> Receipt:
    """Fill ``receipt`` from RAM and the plan's phase rows."""
    from harvest.core.animal_status import (
        COW_DAILY_BRUSHED_FLAG,
        COW_DAILY_MILKED_FLAG,
        COW_DAILY_TALKED_FLAG,
        cow_is_sick,
        cow_needs_milking,
        existing_cow_slots,
        read_cow_daily_flags,
        read_cow_happiness,
        read_fed_cows_flags,
        read_held_item,
        unfed_adult_cow_slots,
    )
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.tile_catalog import ADDR_TILEMAP

    slots = existing_cow_slots(ram)
    # Count troughs, including the shared pregnancy trough, and credit the
    # ones already full. Short hay excuses only the unavoidable deficit.
    troughs = {0x1000 if cow.status & 0x40 else 1 << cow.slot
               for cow in scenario.cows if cow.status & 0x08}
    needed = {bit for bit in troughs if not scenario.fed_flags & bit}
    minimum_fed = len(troughs) - max(0, len(needed) - scenario.hay)
    fed = read_fed_cows_flags(ram)
    receipt.missing = sorted({cow.slot for cow in scenario.cows} - set(slots))
    receipt.tilemap = int(ram[ADDR_TILEMAP])
    receipt.held = int(read_held_item(ram))
    receipt.unfed = unfed_adult_cow_slots(ram)
    receipt.untalked = [s for s in slots if not read_cow_daily_flags(ram, s) & COW_DAILY_TALKED_FLAG]
    receipt.unbrushed = [s for s in slots if not read_cow_daily_flags(ram, s) & COW_DAILY_BRUSHED_FLAG]
    receipt.unmilked = [s for s in slots if cow_needs_milking(ram, s)]
    receipt.still_sick = [s for s in slots if cow_is_sick(ram, s)]
    rows = task.phase_results
    receipt.failed_phases = [f"{r.get('phase')}:{str(r.get('status'))}:{str(r.get('reason'))[:90]}" for r in rows if r.get("status") not in ("success",)]
    receipt.resets = sum(1 for r in rows if r.get("phase") == "COW_CHORES" and r.get("status") == "retry")
    problems: list[str] = []
    if receipt.plan != "success":
        problems.append(f"plan:{receipt.plan_reason[:80]}")
    if receipt.tilemap != FARM_TILEMAP:
        problems.append(f"not on farm:0x{receipt.tilemap:02X}")
    if receipt.held:
        problems.append(f"hands full:0x{receipt.held:02X}")
    if receipt.missing:
        problems.append(f"missing:{receipt.missing}")
    if sum(bool(fed & bit) for bit in troughs) < minimum_fed:
        problems.append(f"unfed:{receipt.unfed}")
    if receipt.untalked:
        problems.append(f"untalked:{receipt.untalked}")
    tired = scenario.stamina == 0
    late = read_ram_value(ram, "hour") >= 17
    sick_at_start = sum(1 for cow in scenario.cows if cow.status == STATUS_SICK)
    if receipt.unbrushed and not tired:
        problems.append(f"unbrushed:{receipt.unbrushed}")
    if receipt.unmilked and not late:
        problems.append(f"unmilked:{receipt.unmilked}")
    if receipt.planned_cure and len(receipt.still_sick) >= sick_at_start:
        problems.append(f"sick:{receipt.still_sick}")  # a bottle cures one cow; the rest wait for the next dose
    if receipt.recoveries:
        problems.append(f"recoveries:{receipt.recoveries}")
    if receipt.ram_writes or receipt.mid_run_loads:
        problems.append(f"interventions:writes={receipt.ram_writes},loads={receipt.mid_run_loads}")
    if receipt.shipping_before is not None:
        from harvest.tasks.harvest_task import read_shipping_money

        receipt.shipping_after = read_shipping_money(ram)
        milked = [cow.slot for cow in scenario.cows if cow.slot in slots
                  and not cow.flags & COW_DAILY_MILKED_FLAG
                  and read_cow_daily_flags(ram, cow.slot) & COW_DAILY_MILKED_FLAG]
        prices = receipt.milk_prices
        receipt.milk_value = (sum(prices.values()) if prices is not None else
                              sum(150 if read_cow_happiness(ram, slot) < 0x60 else
                                  250 if read_cow_happiness(ram, slot) < 0xC0 else 350 for slot in milked))
        if receipt.shipping_after - receipt.shipping_before < receipt.milk_value:
            problems.append(f"unpaid milk:{receipt.shipping_after - receipt.shipping_before}/{receipt.milk_value}")
    if receipt.escaped:
        problems.append(f"escaped:{receipt.escaped}")
    still_out = [slot for slot, home in _homes(ram).items() if home != BARN_TILEMAP]
    if still_out:
        problems.append(f"outside:{still_out}")
    receipt.problems = problems
    receipt.ok = not problems
    return receipt


KICKS = ("left", "right", "up", "down", "pause")


def _homes(ram) -> dict[int, int]:
    from harvest.core.animal_status import existing_cow_slots
    from harvest.core.ram_catalog import read_animal_slot_field

    return {slot: int(read_animal_slot_field(ram, "cow", slot, "home_map_raw")) for slot in existing_cow_slots(ram)}


def _px(ram) -> tuple[int, int]:
    from harvest.tasks.nav import get_pos_from_ram

    point = get_pos_from_ram(ram)
    return point.x, point.y


def run_trial(scenario: Scenario, *, max_frames: int = 90_000, pin_name: Optional[str] = None) -> Receipt:
    """Run the planner's cow section for one morning under ``scenario``; see the module docstring."""
    from harvest.core.tile_catalog import ADDR_TILEMAP
    from harvest.planner.cow_cure import cure_day_phases
    from harvest.planner.day_phase_catalog import EXIT_TO_FARM_PHASE, cow_day_phases
    from harvest.planner.day_phase_types import DayPlannerPolicy
    from harvest.planner.day_plan import DayPlanTask
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import make_action
    from harvest.tasks.harvest_task import read_shipping_money
    from harvest.core.animal_status import COW_DAILY_MILKED_FLAG, read_cow_daily_flags, read_cow_happiness
    from retro_harness import AuditCapabilities, AuditedEnv, TaskStatus, WorldState

    name = pin_name or f"ZZ_barnfuzz_{os.getpid()}_{scenario.seed}"
    path = mint_trial_pin(scenario, name)
    kicker = random.Random(f"kick-{scenario.seed}")
    receipt = Receipt(milk_prices={})
    env = None

    def record_milk(ram):
        # Price is fixed when milk is made, before any later care can move
        # happiness across a price boundary.
        for cow in scenario.cows:
            if cow.flags & COW_DAILY_MILKED_FLAG or cow.slot in receipt.milk_prices:
                continue
            if read_cow_daily_flags(ram, cow.slot) & COW_DAILY_MILKED_FLAG:
                happy = read_cow_happiness(ram, cow.slot)
                receipt.milk_prices[cow.slot] = 150 if happy < 0x60 else 250 if happy < 0xC0 else 350

    try:
        env = AuditedEnv(make_harvest_env(name, lean_step=True), capabilities=AuditCapabilities.all("harvest.barn_fuzz"))
        env.reset()
        path.unlink(missing_ok=True)
        ram = env.get_ram()
        receipt.shipping_before = read_shipping_money(ram)
        cure = cure_day_phases(ram)
        receipt.planned_cure = bool(cure)
        phases = [EXIT_TO_FARM_PHASE, *cure, *cow_day_phases(ram)]
        # The trial ends with the barn chores: a late clock must not sleep and clear the receipts.
        task = DayPlanTask(phase_sequence=phases, policy=dataclasses.replace(DayPlannerPolicy(), include_end_day=False))
        task.reset(WorldState(frame=0, ram=env.get_ram(), info={}, obs=None))
        frame = 0
        idled_out = idled_in = False
        result = None
        while frame < max_frames:
            ram = env.get_ram()
            record_milk(ram)
            tilemap = int(ram[ADDR_TILEMAP])
            for slot, home in _homes(ram).items():
                if home != BARN_TILEMAP and slot not in receipt.escaped and (scenario.outside is None or slot != scenario.outside[0]):
                    receipt.escaped.append(slot)
                    print(f"[FUZZ] cow {slot} left the barn frame={frame} home=0x{home:02X} farmer={_px(ram)} tilemap=0x{tilemap:02X} phase={getattr(task, 'phase_text', '?')}", flush=True)
            if tilemap == FARM_TILEMAP and not idled_out:
                idled_out = True
                for _ in range(min(scenario.idle_out, max_frames - frame)):
                    env.step(make_action())
                    frame += 1
                continue
            if tilemap == BARN_TILEMAP:
                if not idled_in:
                    idled_in = True
                    for _ in range(min(scenario.idle_in, max_frames - frame)):
                        env.step(make_action())
                        frame += 1
                        receipt.barn_frames += 1
                    continue
                near_door = _px(ram)[1] >= 340  # a knock through the door is a walk out, nothing to prove
                if scenario.kick_rate and not near_door and kicker.random() < scenario.kick_rate:
                    kick = kicker.choice(KICKS)
                    action = make_action() if kick == "pause" else make_action(**{kick: True})
                    for _ in range(min(kicker.randrange(5, 60), max_frames - frame)):
                        # A knock starting above the hall can reach the door
                        # during its hold. Keep the whole injected hold inside
                        # the barn, as promised by the near-door exclusion.
                        env.step(make_action() if _px(env.get_ram())[1] >= 340 else action)
                        frame += 1
                        receipt.barn_frames += 1
                    continue
            result = task.step(WorldState(frame=frame, ram=ram, info={}, obs=None))
            if result.status != TaskStatus.RUNNING:
                break
            env.step(result.action.action if result.action else make_action())
            frame += 1
            receipt.barn_frames += tilemap == BARN_TILEMAP
        receipt.frames = frame
        record_milk(env.get_ram())
        if result is None or result.status == TaskStatus.RUNNING:
            receipt.plan, receipt.plan_reason = "timeout", f"plan still running at {frame}"
        else:
            receipt.plan, receipt.plan_reason = result.status.value, str(result.reason or "")
        receipt.recoveries = len(getattr(task, "_recovery_attempted_phases", ()) or ())
        audit = env.audit()
        receipt.ram_writes, receipt.mid_run_loads = audit.ram_writes, audit.mid_run_loads
        return judge(env.get_ram(), scenario, task, receipt)
    finally:
        path.unlink(missing_ok=True)
        if env is not None:
            env.close()


# ── driver ────────────────────────────────────────────────────────────


def source_digest() -> str:
    """Identify the controller and harness used by each subprocess replay."""
    digest = hashlib.sha256()
    for path in sorted((GAME / "harvest").rglob("*.py")):
        digest.update(str(path.relative_to(GAME)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _trial_paths(tag: str, seed: int) -> tuple[Path, Path]:
    folder = OUT / tag
    return folder / f"{seed}.json", folder / f"{seed}.log"


def _spawn_trial(args: tuple[str, str, int, int]) -> dict:
    tag, profile, seed, max_frames = args
    out, log = _trial_paths(tag, seed)
    env = dict(os.environ, HEADLESS="1", PYTHONUNBUFFERED="1", QT_QPA_PLATFORM="offscreen", SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    cmd = [sys.executable, "-m", "harvest.tools.barn_fuzz", "one", "--seed", str(seed), "--profile", profile,
           "--max-frames", str(max_frames), "--out", str(out)]
    started = time.time()
    digest = source_digest()
    scenario = make_scenario(seed, profile)
    out.unlink(missing_ok=True)
    with log.open("w") as handle:
        try:
            code = subprocess.run(cmd, cwd=GAME, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=600).returncode
        except subprocess.TimeoutExpired:
            code = "wall timeout (600s)"
    try:
        row = json.loads(out.read_text())
    except (OSError, ValueError):
        row = {"seed": seed, "profile": scenario.profile, "scenario": dataclasses.asdict(scenario),
               "source_digest": digest, "receipt": {"ok": False, "problems": [f"crash:{code}"]}}
    if code != 0 and row["receipt"].get("ok"):
        row["receipt"]["ok"] = False
        row["receipt"].setdefault("problems", []).append(f"exit:{code}")
    row["wall_s"] = round(time.time() - started, 1)
    out.write_text(json.dumps(row))
    if row["receipt"].get("ok"):
        log.unlink(missing_ok=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    folder = OUT / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.json")):
        print(f"refusing to overwrite existing batch: {folder}")
        return 1
    seeds = args.seeds if args.seeds is not None else list(range(args.first_seed, args.first_seed + args.trials))
    if len(seeds) != len(set(seeds)):
        print("duplicate seeds would overwrite trial receipts")
        return 1
    manifest = {"profile": args.profile, "seeds": seeds, "max_frames": args.max_frames, "source_digest": source_digest()}
    (folder / "batch.json").write_text(json.dumps(manifest, indent=2))
    jobs = [(args.tag, args.profile, seed, args.max_frames) for seed in seeds]
    passed = done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_spawn_trial, jobs):
            done += 1
            ok = bool(row["receipt"].get("ok"))
            passed += ok
            print(f"[{done}/{len(jobs)}] seed={row['seed']} {'ok' if ok else 'FAIL'} frames={row['receipt'].get('frames')} "
                  f"resets={row['receipt'].get('resets')} {'; '.join(row['receipt'].get('problems', []))[:150]}", flush=True)
    print(f"{args.tag}: {passed}/{done} clean")
    command_report(argparse.Namespace(tag=args.tag))
    return 0 if passed == done and source_digest() == manifest["source_digest"] else 1


def command_one(args: argparse.Namespace) -> int:
    digest = source_digest()
    scenario = make_scenario(args.seed, args.profile)
    print(f"[FUZZ] {scenario}", flush=True)
    receipt = run_trial(scenario, max_frames=args.max_frames)
    row = {"seed": args.seed, "profile": scenario.profile, "source_digest": digest, "scenario": dataclasses.asdict(scenario),
           "herd": scenario.herd_label(), "receipt": dataclasses.asdict(receipt)}
    print(f"[FUZZ] {json.dumps(row['receipt'])}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(row))
    return 0 if receipt.ok else 1


def _problem_class(problem: str) -> str:
    return problem.split(":")[0]


def command_report(args: argparse.Namespace) -> int:
    folder = OUT / args.tag
    rows = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json")) if path.stem.lstrip("-").isdigit()]
    for row in rows:
        if row.get("profile") == "all":  # older crash rows stored the requested profile
            row["profile"] = make_scenario(row["seed"], "all").profile
    if not rows:
        print("no trials")
        return 1
    ok_rows = [row for row in rows if row["receipt"].get("ok")]
    frames = sorted(row["receipt"].get("frames", 0) for row in ok_rows)
    print(f"{args.tag}: {len(ok_rows)}/{len(rows)} clean")
    barn = [row for row in rows if row.get("profile") != "outside"]
    if barn:
        print(f"barn care: {sum(bool(row['receipt'].get('ok')) for row in barn)}/{len(barn)} clean")
    if frames:
        print(f"clean trial frames: median {frames[len(frames) // 2]:,}, worst {frames[-1]:,}")
    twelve = sorted(row["receipt"].get("barn_frames", 0) for row in ok_rows if row.get("profile") == "twelve")
    if twelve:
        print(f"twelve-cow barn frames: median {twelve[len(twelve) // 2]:,}, mean {sum(twelve) // len(twelve):,} (n={len(twelve)})")
    print(f"resets: {collections.Counter(row['receipt'].get('resets', 0) for row in rows).most_common()}")
    for profile in sorted({row.get("profile", "?") for row in rows}):
        group = [row for row in rows if row.get("profile") == profile]
        print(f"  {profile:8s} {sum(bool(row['receipt'].get('ok')) for row in group):3d}/{len(group)} clean, {sum(bool(row['receipt'].get('escaped')) for row in group)} with an escape")
    classes: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[int]] = collections.defaultdict(list)
    for row in rows:
        for problem in row["receipt"].get("problems", []):
            key = _problem_class(problem)
            classes[key] += 1
            samples[key].append(row["seed"])
    for key, count in classes.most_common():
        print(f"  {key:12s} {count:4d}  seeds {samples[key][:8]}")
    complete = True
    manifest_path = folder / "batch.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        missing = sorted(set(manifest["seeds"]) - {row["seed"] for row in rows})
        mismatched = [row["seed"] for row in rows if row.get("source_digest") is not None
                      and row["source_digest"] != manifest["source_digest"]]
        if missing:
            print(f"incomplete batch: {len(missing)} missing seeds {missing[:8]}")
        if mismatched:
            print(f"controller changed during batch: seeds {mismatched[:8]}")
        complete = not missing and not mismatched
    return 0 if complete and len(ok_rows) == len(rows) else 1


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("tag")
    run.add_argument("--profile", choices=PROFILES, default="all")
    run.add_argument("--trials", type=positive_int, default=64)
    run.add_argument("--first-seed", type=int, default=1)
    run.add_argument("--seeds", type=int, nargs="+", help="replay specific seeds instead of a range")
    run.add_argument("--workers", type=positive_int, default=28)
    run.add_argument("--max-frames", type=positive_int, default=90_000)
    one = sub.add_parser("one")
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--profile", choices=PROFILES, default="all")
    one.add_argument("--max-frames", type=positive_int, default=90_000)
    one.add_argument("--out")
    report = sub.add_parser("report")
    report.add_argument("tag")
    args = parser.parse_args(argv)
    return {"run": command_run, "one": command_one, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
