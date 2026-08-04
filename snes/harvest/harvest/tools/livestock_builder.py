#!/usr/bin/env python3
"""Build compact livestock test states and verify them against the emulator."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harvest.core.harvest_state import HarvestStateDocument
from harvest.runtime.rom_tools import STATES_DIR, SaveStateArchive
from harvest.runtime.retro_setup import make_harvest_env
from harvest.core.tile_catalog import Tool


SCRIPT_DIR = Path(__file__).resolve().parent
SHED_ITEMS_ROW_2_ADDR = 0x7F1F01
SHED_ROW_2_WATERING_CAN_BIT = 0x80
SHED_ROW_2_MILKER_BIT = 0x20
SHED_ROW_2_BRUSH_BIT = 0x40
BARN_TILEMAP = 0x27
FARM_TILEMAP = 0x00

COW_STATUS_MILK_READY = 0x09
COW_STATUS_YOUNG = 0x05
COW_STATUS_BABY = 0x03


@dataclass(frozen=True)
class CowMilkFixtureSpec:
    slot: int
    label: str
    status_raw: int
    raw_1: int
    happiness: int
    milk_profile: str
    name_bytes: tuple[int, int, int, int]
    position_tile: tuple[int, int]


@dataclass(frozen=True)
class LivestockSnapshotSummary:
    state_name: str
    ram_sha1: str
    money: int
    hay: int
    chickens: int
    cows: int
    chicken_slot0: bytes
    cow_slot0: bytes


FOUR_COW_MILK_FIXTURE: tuple[CowMilkFixtureSpec, ...] = (
    CowMilkFixtureSpec(
        slot=0,
        label="adult_small_milk",
        status_raw=COW_STATUS_MILK_READY,
        raw_1=0x00,
        happiness=0x20,
        milk_profile="small",
        name_bytes=(0x00, 0x01, 0x02, 0xB1),
        position_tile=(10, 17),
    ),
    CowMilkFixtureSpec(
        slot=1,
        label="adult_large_milk",
        status_raw=COW_STATUS_MILK_READY,
        raw_1=0x00,
        happiness=0xF0,
        milk_profile="large",
        name_bytes=(0x03, 0x04, 0x05, 0xB1),
        position_tile=(10, 15),
    ),
    CowMilkFixtureSpec(
        slot=2,
        label="young_not_milk_ready",
        status_raw=COW_STATUS_YOUNG,
        raw_1=0x00,
        happiness=0x40,
        milk_profile="not_ready",
        name_bytes=(0x06, 0x07, 0x08, 0xB1),
        position_tile=(10, 13),
    ),
    CowMilkFixtureSpec(
        slot=3,
        label="baby_not_milk_ready",
        status_raw=COW_STATUS_BABY,
        raw_1=0x00,
        happiness=0x10,
        milk_profile="not_ready",
        name_bytes=(0x09, 0x0A, 0x0B, 0xB1),
        position_tile=(10, 9),
    ),
)


def _state_summary_from_bytes(state_name: str, ram: bytes) -> LivestockSnapshotSummary:
    cow_base = 0x7EC1C6 - 0x7E0000
    chicken_base = 0x7EC286 - 0x7E0000
    return LivestockSnapshotSummary(
        state_name=state_name,
        ram_sha1=hashlib.sha1(ram).hexdigest(),
        money=ram[0x11F04] | (ram[0x11F05] << 8) | (ram[0x11F06] << 16),
        hay=ram[0x11F10] | (ram[0x11F11] << 8),
        chickens=ram[0x11F0B],
        cows=ram[0x11F0A],
        chicken_slot0=bytes(ram[chicken_base : chicken_base + 8]),
        cow_slot0=bytes(ram[cow_base : cow_base + 16]),
    )


def summarize_saved_state(state_name: str) -> LivestockSnapshotSummary:
    archive = SaveStateArchive.load(STATES_DIR / f"{state_name}.state")
    return _state_summary_from_bytes(state_name, archive.require_block("RAM"))


def summarize_loaded_state(state_name: str) -> LivestockSnapshotSummary:
    env = make_harvest_env(state_name)
    try:
        env.reset()
        ram = bytes(env.data.memory.blocks[0x7E0000])
    finally:
        env.close()
    return _state_summary_from_bytes(state_name, ram)


def verify_state(state_name: str) -> tuple[LivestockSnapshotSummary, LivestockSnapshotSummary]:
    saved = summarize_saved_state(state_name)
    loaded = summarize_loaded_state(state_name)
    return saved, loaded


def summaries_match(saved: LivestockSnapshotSummary, loaded: LivestockSnapshotSummary) -> bool:
    return (
        saved.money == loaded.money
        and saved.hay == loaded.hay
        and saved.chickens == loaded.chickens
        and saved.cows == loaded.cows
        and saved.chicken_slot0 == loaded.chicken_slot0
        and saved.cow_slot0 == loaded.cow_slot0
    )


def build_compact_livestock_states(
    *,
    base_state: str,
    prefix: str,
    money: int,
    hay: int,
    chicken_feed: int,
    cow_feed: int,
) -> dict[str, Path]:
    document = HarvestStateDocument.load(base_state)
    document.set_purchase_resources(
        money=money,
        hay=hay,
        chicken_feed=chicken_feed,
        cow_feed=cow_feed,
    )

    output_paths: dict[str, Path] = {}

    resources_name = f"{prefix}_resources"
    output_paths["resources"] = document.save_as(STATES_DIR / f"{resources_name}.state")

    document.add_chicken()
    chicken_name = f"{prefix}_chicken"
    output_paths["chicken"] = document.save_as(STATES_DIR / f"{chicken_name}.state")

    document.add_cow()
    cow_name = f"{prefix}_chicken_cow"
    output_paths["chicken_cow"] = document.save_as(STATES_DIR / f"{cow_name}.state")

    return output_paths


def configure_four_cow_milk_fixture(
    document: HarvestStateDocument,
    *,
    hay: int = 99,
    cow_feed: int = 20,
    carry_animal_tools: bool = True,
) -> tuple[CowMilkFixtureSpec, ...]:
    """Mutate a save-state document into a deterministic four-cow milk fixture."""

    document.clear_cows()
    for spec in FOUR_COW_MILK_FIXTURE:
        spawn_x = spec.position_tile[0] * 16 + 8
        spawn_y = spec.position_tile[1] * 16 + 8
        document.set_cow_slot(
            spec.slot,
            status_raw=spec.status_raw,
            raw_1=spec.raw_1,
            home_map_raw=0x27,
            pregnancy_raw=0x00,
            happiness=spec.happiness,
            raw_5=0x00,
            position_x=spawn_x,
            position_y=spawn_y,
            name_bytes=spec.name_bytes,
        )

    document.set_scalar_value("stored_grass", hay)
    document.set_scalar_value("cow_feed", cow_feed)
    document.set_scalar_value("fed_cows_n", 0)
    document.set_scalar_value("fed_cows_flags", 0)

    if carry_animal_tools:
        document.set_scalar_value("tool_selected", int(Tool.BRUSH))
        document.set_scalar_value("tool_backpack", int(Tool.MILKER))
        shed_row_2 = document.mutable_state.read_u8(SHED_ITEMS_ROW_2_ADDR)
        shed_row_2 |= SHED_ROW_2_WATERING_CAN_BIT
        shed_row_2 &= ~(SHED_ROW_2_MILKER_BIT | SHED_ROW_2_BRUSH_BIT)
        document.mutable_state.write_u8(SHED_ITEMS_ROW_2_ADDR, shed_row_2)
    else:
        shed_row_2 = document.mutable_state.read_u8(SHED_ITEMS_ROW_2_ADDR)
        shed_row_2 |= SHED_ROW_2_MILKER_BIT | SHED_ROW_2_BRUSH_BIT
        document.mutable_state.write_u8(SHED_ITEMS_ROW_2_ADDR, shed_row_2)

    return FOUR_COW_MILK_FIXTURE


def _task_world(env, frame: int, obs=None, info=None):
    if str(SCRIPT_DIR.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR.parent))
    from retro_harness import WorldState

    return WorldState(
        frame=frame,
        ram=np.asarray(env.get_ram(), dtype=np.uint8),
        info=info or {},
        obs=obs,
    )


def _run_transition_task(env, task, frame: int, *, max_steps: int) -> int:
    if str(SCRIPT_DIR.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR.parent))
    from retro_harness import TaskStatus

    obs = None
    info = {}
    task.reset(_task_world(env, frame, obs, info))
    for _ in range(max_steps):
        result = task.step(_task_world(env, frame, obs, info))
        if result.status == TaskStatus.SUCCESS:
            return frame
        if result.status in (TaskStatus.FAILURE, TaskStatus.BLOCKED):
            raise RuntimeError(f"{task.name} {result.status.value}: {result.reason}")
        action = (
            result.action.action
            if result.action is not None
            else np.zeros(env.action_space.shape, dtype=np.int8)
        )
        obs, _reward, _terminated, _truncated, info = env.step(action)
        frame += 1
    raise RuntimeError(f"{task.name} exceeded {max_steps} steps")


def refresh_barn_runtime_state(state_name: str, *, output_name: str | None = None) -> Path:
    """Exit and re-enter a barn save so runtime cow actors match cow slots."""

    from harvest.planner.day_plan_tasks import DirectionalTransitionTask

    output_name = output_name or state_name
    env = make_harvest_env(state_name)
    try:
        env.reset()
        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        if int(ram[0x22]) != BARN_TILEMAP:
            return STATES_DIR / f"{state_name}.state"

        frame = 0
        frame = _run_transition_task(
            env,
            DirectionalTransitionTask(
                name="refresh_exit_barn",
                direction="down",
                origin_tilemap=BARN_TILEMAP,
                target_tilemap=FARM_TILEMAP,
                timeout=1200,
                min_frames_before_success=15,
                stand_tile=(8, 22),
                stand_tolerance=1,
                settle_frames=5,
            ),
            frame,
            max_steps=1500,
        )
        frame = _run_transition_task(
            env,
            DirectionalTransitionTask(
                name="refresh_enter_barn",
                direction="up",
                origin_tilemap=FARM_TILEMAP,
                target_tilemap=BARN_TILEMAP,
                timeout=1200,
                min_frames_before_success=10,
                stand_tile=(20, 22),
                stand_tolerance=0,
                target_stand_tile=(8, 22),
                target_stand_tolerance=1,
                settle_frames=45,
                door_align_px=20 * 16 + 8,
                overshoot_limit_px=330,
                require_empty_hands=True,
            ),
            frame,
            max_steps=1500,
        )

        output_path = STATES_DIR / f"{output_name}.state"
        with gzip.open(output_path, "wb") as handle:
            handle.write(env.em.get_state())
        return output_path
    finally:
        env.close()


def build_four_cow_milk_fixture_state(
    *,
    base_state: str,
    output_name: str,
    money: int = 5000,
    hay: int = 99,
    cow_feed: int = 20,
    carry_animal_tools: bool = True,
    refresh_barn_runtime: bool = False,
) -> Path:
    document = HarvestStateDocument.load(base_state)
    document.set_scalar_value("money", money)
    configure_four_cow_milk_fixture(
        document,
        hay=hay,
        cow_feed=cow_feed,
        carry_animal_tools=carry_animal_tools,
    )
    path = document.save_as(STATES_DIR / f"{output_name}.state")
    if refresh_barn_runtime:
        path = refresh_barn_runtime_state(output_name)
    return path


def _print_summary(label: str, summary: LivestockSnapshotSummary) -> None:
    print(
        f"{label}: ram_sha1={summary.ram_sha1[:12]} "
        f"money={summary.money} hay={summary.hay} "
        f"chickens={summary.chickens} cows={summary.cows} "
        f"chicken0={summary.chicken_slot0.hex()} cow0={summary.cow_slot0.hex()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify compact livestock save states")
    parser.add_argument("--base", default="Y1_After_Buy_Potato", help="Base save-state name")
    parser.add_argument("--prefix", default="Y1_Livestock_Compact", help="Output state name prefix")
    parser.add_argument("--money", type=int, default=5000, help="Raw money value to inject")
    parser.add_argument("--hay", type=int, default=99, help="Hay/stored grass total to inject")
    parser.add_argument("--chicken-feed", type=int, default=20, help="Chicken feed inventory")
    parser.add_argument("--cow-feed", type=int, default=20, help="Cow feed inventory")
    parser.add_argument("--four-cow-fixture", action="store_true", help="Write one four-cow milk test fixture")
    parser.add_argument("--four-cow-output", default=None, help="Output state name for --four-cow-fixture")
    parser.add_argument("--no-carry-animal-tools", action="store_true", help="Do not preload brush/milker in the fixture")
    parser.add_argument(
        "--refresh-barn-runtime",
        action="store_true",
        help="When the output starts in the barn, exit and re-enter so active cow actors are rebuilt",
    )
    parser.add_argument("--verify", action="store_true", help="Load written states in the emulator and compare live RAM")
    parser.add_argument("--sheep", action="store_true", help="Rejected explicitly; this ROM has no sheep support")
    args = parser.parse_args()

    if args.sheep:
        raise SystemExit("Sheep are not supported in this Harvest Moon SNES build; use chicken/cow states instead.")

    if args.four_cow_fixture:
        state_name = args.four_cow_output or f"{args.prefix}_four_cow_milk_fixture"
        path = build_four_cow_milk_fixture_state(
            base_state=args.base,
            output_name=state_name,
            money=args.money,
            hay=args.hay,
            cow_feed=args.cow_feed,
            carry_animal_tools=not args.no_carry_animal_tools,
            refresh_barn_runtime=args.refresh_barn_runtime,
        )
        print(f"Wrote four-cow milk fixture: {path}")
        if args.verify:
            saved, live = verify_state(path.stem)
            _print_summary("  saved", saved)
            _print_summary("  live ", live)
            if not summaries_match(saved, live):
                raise SystemExit(f"Verification mismatch for {path.stem}")
        return

    outputs = build_compact_livestock_states(
        base_state=args.base,
        prefix=args.prefix,
        money=args.money,
        hay=args.hay,
        chicken_feed=args.chicken_feed,
        cow_feed=args.cow_feed,
    )

    print("Wrote states:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    if args.verify:
        print("\nVerification:")
        for key, path in outputs.items():
            state_name = path.stem
            saved, live = verify_state(state_name)
            _print_summary("  saved", saved)
            _print_summary("  live ", live)
            if not summaries_match(saved, live):
                raise SystemExit(f"Verification mismatch for {state_name}")
            if saved.ram_sha1 != live.ram_sha1:
                print("  note: live WRAM SHA differs after reset, but validated fields match")
            print(f"  ok: {state_name}")


if __name__ == "__main__":
    main()
