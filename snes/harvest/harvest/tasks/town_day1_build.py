"""Spring D1 town handoff — sequence builders.

Extracted from ``town_day1_handoff`` (LOC budget).
"""

from __future__ import annotations

import os
from typing import List

from retro_harness import Task

from harvest.core.carry import seed_item_id
from harvest.core.tile_catalog import Tool
from harvest.maps.map_config import ROUTES, Waypoint
from harvest.planner.day_plan_status import TASKS_DIR
from harvest.planner.tasks.home import GoToSleepTask
from harvest.planner.tasks.inventory import (
    CompleteOutdoorMorningIntroTask,
    EnsureCarryToolTask,
    RecordingSliceSpec,
    SHED_SEED_SPECS,
    ShedFetchItemTask,
    load_recording_slice,
)
from harvest.planner.tasks.navigation import MultiMapNavTask
from harvest.tasks.town_day1_tasks import (
    BIT_ANN,
    BIT_EVE,
    BIT_FLOWER_OWNER,
    BIT_LIVESTOCK,
    BIT_MARIA,
    BIT_NINA,
    TARGET_MASK,
    PressAUntilBitOrTimeout,
    TrackNpcUntilBitTask,
    ScriptedWalkTask,
    TOWN_TILEMAP,
    WalkUntilCoordTask,
    SequenceTask,
    SkipIfBitSet,
    _AssertCarryToolsTask,
    _AssertMaskTask,
    _HoldButtonsTask,
    _SoftOptionalTask,
    _TruckLeaveTask,
)


def _clone_route(name: str) -> List[Waypoint]:
    route = ROUTES.get(name)
    if not route:
        raise KeyError(f"missing route {name!r}")
    return list(route)


def _nav(name: str, waypoints: List[Waypoint], *, timeout: int = 6000, settle: int = 20) -> MultiMapNavTask:
    return MultiMapNavTask(
        name=name,
        waypoints=waypoints,
        timeout=timeout,
        initial_settle_frames=settle,
    )


def _clear_flower_shop_door(name: str) -> SequenceTask:
    """Town-space flower-shop door: idle until tiles load, then walk south.

    Rest tape: leaked interior (136,468) → snap (600,232) 0xFF → idle ~70f →
    Down to y~280 path.  Immediate Down+Right or a (600,280) nav waypoint
    re-enters the shop.
    """
    return SequenceTask(
        name=name,
        tasks=(
            WalkUntilCoordTask(
                name=f"{name}_remap",
                direction="down",
                tilemap=TOWN_TILEMAP,
                min_x=400,
                timeout=240,
            ),
            _HoldButtonsTask(name=f"{name}_idle", buttons=(), frames=80),
            WalkUntilCoordTask(
                name=f"{name}_south",
                direction="down",
                tilemap=TOWN_TILEMAP,
                min_y=275,
                timeout=120,
            ),
        ),
    )


def _talk_route(
    name: str,
    route_name: str,
    bit: int,
    *,
    face: str,
    timeout: int = 6000,
    exit_route: str | None = None,
    required: bool = True,
) -> Task:
    """Nav to a neighborhood, then talk until bit; optional exit back to town.

    Required outdoor talks (Ann/Eve) track the nearest live NPC because those
    sprites wander off a 6px stand. Optional talks stay on a fixed face+A.
    """
    steps: list[Task] = [
        _nav(f"nav_{name}", _clone_route(route_name), timeout=timeout),
        TrackNpcUntilBitTask(
            name=f"talk_{name}", bit=bit, timeout=2400, face_hint=face
        )
        if required
        else PressAUntilBitOrTimeout(
            name=f"talk_{name}", bit=bit, face=face, required=False
        ),
    ]
    if exit_route:
        steps.append(_nav(f"exit_{name}", _clone_route(exit_route), timeout=timeout, settle=15))
    return SkipIfBitSet(
        name=f"skip_{name}",
        bit=bit,
        child=SequenceTask(name=name, tasks=tuple(steps)),
    )


def _shed_starter_tools(*, exit_when_done: bool = True, required: bool = True) -> Task:
    """Pick free D1 grass seeds + watering can from the tool shed.

    New-game init puts both on shed_items_row_2 (``0x88`` = can 0x80 | grass
    0x08). Stock ``grass_seeds`` is already 1; equipping requires a shelf A.
    Carry only holds two slots — order is grass then can so both stay ready.

    Gate B pure path: after truck→D2 bed, first house→farm fires outdoor dog
    intro (``CODE_83CEAE``). ``CompleteOutdoorMorningIntroTask`` names the dog
    (``AAAA``) so ``event_flags_1f68`` reaches ``0x00B1`` and free-move returns
    before shed nav. Y1 fixtures already have intro flags — task no-ops.

    Verified from ``house_size=0`` morning house. Some D1 fixtures incorrectly
    have ``house_size=2`` (AnnEve / rest_end); set ``required=False`` to
    soft-continue the handoff when shed is optional.
    """
    grass_shelf = SHED_SEED_SPECS["grass"]
    seq = SequenceTask(
        name="shed_starter_tools",
        tasks=(
            CompleteOutdoorMorningIntroTask(name="outdoor_morning_intro"),
            ShedFetchItemTask(
                name="pick_grass_seeds",
                item_id=seed_item_id("grass"),
                shelf=grass_shelf,
                exit_when_done=False,
            ),
            EnsureCarryToolTask(
                name="pick_watering_can",
                tool_id=int(Tool.WATERING_CAN),
                exit_when_done=exit_when_done,
            ),
            _AssertCarryToolsTask(
                name="assert_starter_tools",
                required_ids=(seed_item_id("grass"), int(Tool.WATERING_CAN)),
            ),
        ),
    )
    if required:
        return seq
    return _SoftOptionalTask(name="shed_starter_tools_optional", child=seq)


def build_day1_handoff_tasks(
    *,
    include_sleep: bool = True,
    require_full_mask: bool = True,
    pick_starter_tools: bool = True,
    require_starter_tools: bool = False,
    use_rest_recording: bool = True,
) -> SequenceTask:
    """Build the full D1 town → truck → shed pickups → optional sleep sequence.

    Talk stands verified from ``tasks/town_day1_rest.json`` (Ann|Eve start,
    full mask 0x3F, truck cutscene → house, sleep → D2). Outdoor talks first.

    ``require_starter_tools`` forces shed grass+can into carry (Gate B /
    ``house_size=0``). Soft-optional when false so AnnEve ``house_size=2``
    fixtures can still finish the truck/sleep path.

    ``use_rest_recording``: when True and the rest capture exists, replay it
    for the remaining four talks + truck + sleep (AnnEve oracle path). When
    False (clean power-on / Town_Gate with Ann|Eve still open), use composed
    pure routes — the rest recording desyncs if Ann|Eve were just run pure
    (input_lock / path drift).
    """
    # Flower owner: enter shop, remap, push to counter stand ~(34,347), face down A.
    # town_day1_rest bit 0x08 at (34,347) Down+A.
    flower_owner = SkipIfBitSet(
        name="skip_flower_owner",
        bit=BIT_FLOWER_OWNER,
        child=SequenceTask(
            name="flower_owner",
            tasks=(
                _nav("to_flower_shop", _clone_route("d1_town_to_flower_shop"), timeout=5000),
                # Remap interior coords (probe: left then up settles ~144,456).
                ScriptedWalkTask(name="shop_remap_left", direction="left", frames=40, run=True),
                ScriptedWalkTask(name="shop_remap_up", direction="up", frames=20, run=False),
                # Push past counter lip toward owner object ~(40,360) / stand y~347.
                _HoldButtonsTask(name="counter_push", buttons=("up", "a"), frames=140),
                ScriptedWalkTask(name="to_owner_x", direction="left", frames=55, run=True),
                ScriptedWalkTask(name="to_owner_y", direction="up", frames=20, run=False),
                PressAUntilBitOrTimeout(
                    name="owner_a",
                    bit=BIT_FLOWER_OWNER,
                    face="down",
                    attempts=10,
                    attempt_timeout=180,
                    required=True,
                ),
                # Exit front room → town (same door as nina's shop exit).
                ScriptedWalkTask(name="owner_to_door_x", direction="right", frames=40, run=True),
                ScriptedWalkTask(name="owner_exit_down", direction="down", frames=100, run=True),
                _nav(
                    "owner_exit_shop",
                    [
                        Waypoint(tilemap=0x1C, target_px=(144, 456), radius=18),
                        Waypoint(
                            tilemap=0x1C,
                            target_px=(144, 480),
                            radius=12,
                            is_exit=True,
                            exit_direction="down",
                        ),
                    ],
                    timeout=4000,
                    settle=10,
                ),
                _clear_flower_shop_door("owner_clear_door"),
            ),
        ),
    )

    # Nina: play town_day1_rest from flower-door spawn through talk.
    # Slice assumes shop entry coords ~(598,218) → remap → back room → (101,102) A.
    nina_rest = load_recording_slice(
        RecordingSliceSpec("town_day1_rest", start_frame=4564, end_frame=5300),
        TASKS_DIR,
    )
    nina_rest.name = "nina_rest_talk"
    nina = SkipIfBitSet(
        name="skip_nina",
        bit=BIT_NINA,
        child=SequenceTask(
            name="nina",
            tasks=(
                # Stop at door spawn so the recording slice lines up.
                _nav(
                    "to_flower_door",
                    [
                        Waypoint(tilemap=0x04, target_px=(688, 280), radius=16),
                        Waypoint(tilemap=0x04, target_px=(600, 280), radius=14),
                        Waypoint(
                            tilemap=0x04,
                            target_px=(600, 262),
                            radius=10,
                            is_exit=True,
                            exit_direction="up",
                        ),
                        Waypoint(tilemap=0x1C, target_px=(598, 218), radius=18),
                    ],
                    timeout=5000,
                    settle=25,
                ),
                nina_rest,
                # Slice can desync on talk; force stand + A if bit not yet set.
                _nav("nina_stand", _clone_route("d1_flower_back_to_nina"), timeout=4000, settle=15),
                PressAUntilBitOrTimeout(
                    name="talk_nina",
                    bit=BIT_NINA,
                    face="left",
                    attempts=10,
                    attempt_timeout=180,
                    required=True,
                ),
                _nav("nina_exit", _clone_route("d1_flower_back_exit_to_town"), timeout=5000, settle=15),
                _clear_flower_shop_door("nina_clear_door"),
            ),
        ),
    )

    # Truck leave often cutscenes into the farmhouse (town_day1_rest). Soft-nav
    # to farm is optional; _TruckLeaveTask succeeds on non-town tilemap.
    #
    # Gate B (require_starter_tools): rest leave-only slice f9200→~9800 (path
    # 0x0C), then GoToSleep owns D2 morning. Full f9200→end includes bed but
    # post-truck house→farm still clears free-move (0x4000) — shed open.
    # AnnEve oracle (soft shed): keep full rest truck+sleep slice.
    rest_path = os.path.join(TASKS_DIR, "town_day1_rest.json")
    truck_includes_sleep = False
    if os.path.isfile(rest_path):
        if require_starter_tools:
            truck_slice = load_recording_slice(
                RecordingSliceSpec("town_day1_rest", start_frame=9200, end_frame=9800),
                TASKS_DIR,
            )
            truck_slice.name = "truck_leave_rest_slice_no_sleep"
            truck_includes_sleep = False
        else:
            truck_slice = load_recording_slice(
                RecordingSliceSpec("town_day1_rest", start_frame=9200, end_frame=None),
                TASKS_DIR,
            )
            truck_slice.name = "truck_leave_sleep_rest_slice"
            truck_includes_sleep = True
        truck = SequenceTask(
            name="truck_leave",
            tasks=(
                _nav(
                    "to_truck_slice_start",
                    _clone_route("d1_town_to_truck"),
                    timeout=7000,
                    settle=20,
                ),
                truck_slice,
            ),
        )
    else:
        truck = SequenceTask(
            name="truck_leave",
            tasks=(
                _nav(
                    "to_truck_stand",
                    _clone_route("d1_town_to_truck_stand"),
                    timeout=8000,
                    settle=20,
                ),
                _TruckLeaveTask(timeout=6000),
            ),
        )

    # Outdoor first (Ann + Eve ROM-verified on clean D1 entry).
    parts: List[Task] = [
        _talk_route("ann", "d1_town_to_ann", BIT_ANN, face="left", timeout=5000),
        _talk_route("eve", "d1_town_to_eve", BIT_EVE, face="up", timeout=5000),
    ]
    if require_full_mask:
        # Prefer the verified human capture only when the run already matches
        # its AnnEve entry (mask 0x03). Clean power-on/Town_Gate must use
        # composed pure routes — rest desyncs after pure Ann|Eve (rr-bhr).
        rest_path = os.path.join(TASKS_DIR, "town_day1_rest.json")
        if use_rest_recording and os.path.isfile(rest_path):
            rest = load_recording_slice(
                RecordingSliceSpec("town_day1_rest", start_frame=0, end_frame=None),
                TASKS_DIR,
            )
            rest.name = "town_day1_rest_recording"
            parts.append(rest)
            # Do not re-sleep: recording already ends D2 morning house.
            include_sleep = False
        else:
            parts.extend(
                [
                    SkipIfBitSet(
                        name="skip_livestock",
                        bit=BIT_LIVESTOCK,
                        child=SequenceTask(
                            name="livestock",
                            tasks=(
                                _nav(
                                    "nav_livestock",
                                    _clone_route("d1_town_to_livestock"),
                                    timeout=7000,
                                ),
                                # Door entry keeps town-space pixels (~598,874)
                                # until the player walks UP off the trigger.
                                # Do not X-align toward the lobby — that walks
                                # left and misses the remap.
                                WalkUntilCoordTask(
                                    name="livestock_remap_up",
                                    direction="up",
                                    max_x=400,
                                    timeout=240,
                                ),
                                # Around the north of the counter to the D1
                                # gift stand (230,139).  (201,157) face-right
                                # is buy-cow and does not set bit 0x10.
                                _nav(
                                    "nav_livestock_stand",
                                    _clone_route("d1_livestock_to_event_stand"),
                                    timeout=4000,
                                    settle=8,
                                ),
                                PressAUntilBitOrTimeout(
                                    name="talk_livestock",
                                    bit=BIT_LIVESTOCK,
                                    face="down",
                                    attempts=10,
                                    attempt_timeout=160,
                                    required=True,
                                ),
                                _AssertMaskTask(name="assert_livestock", expected=BIT_LIVESTOCK),
                                # Leave the gift stand the way rest did: north
                                # of the counter, west, then south through the
                                # door.  MultiNav y-align uses safe-walk and
                                # idles on 0xA1 structure.
                                WalkUntilCoordTask(
                                    name="livestock_exit_north",
                                    direction="up",
                                    min_y=118,
                                    max_y=123,
                                    timeout=80,
                                ),
                                WalkUntilCoordTask(
                                    name="livestock_exit_to_counter_x",
                                    direction="left",
                                    max_x=205,
                                    timeout=80,
                                ),
                                WalkUntilCoordTask(
                                    name="livestock_exit_to_counter_row",
                                    direction="down",
                                    min_y=155,
                                    timeout=80,
                                ),
                                WalkUntilCoordTask(
                                    name="livestock_exit_west",
                                    direction="left",
                                    max_x=129,
                                    timeout=180,
                                ),
                                WalkUntilCoordTask(
                                    name="livestock_exit_south",
                                    direction="down",
                                    min_y=200,
                                    timeout=180,
                                ),
                                WalkUntilCoordTask(
                                    name="livestock_exit_remap",
                                    direction="down",
                                    tilemap=TOWN_TILEMAP,
                                    min_x=400,
                                    timeout=280,
                                ),
                                _HoldButtonsTask(
                                    name="livestock_off_door",
                                    buttons=("right", "b"),
                                    frames=28,
                                ),
                            ),
                        ),
                    ),
                    nina,
                    flower_owner,
                    SkipIfBitSet(
                        name="skip_maria",
                        bit=BIT_MARIA,
                        child=SequenceTask(
                            name="maria",
                            tasks=(
                                _nav(
                                    "nav_maria_door",
                                    _clone_route("d1_town_to_maria"),
                                    timeout=7000,
                                ),
                                ScriptedWalkTask(
                                    name="church_door_up",
                                    direction="up",
                                    frames=80,
                                    run=True,
                                ),
                                _nav(
                                    "nav_maria_stand",
                                    _clone_route("d1_church_to_maria"),
                                    timeout=4000,
                                    settle=20,
                                ),
                                PressAUntilBitOrTimeout(
                                    name="talk_maria",
                                    bit=BIT_MARIA,
                                    face="up",
                                    attempts=10,
                                    attempt_timeout=180,
                                    required=True,
                                ),
                                _nav(
                                    "exit_maria",
                                    _clone_route("d1_maria_to_town"),
                                    timeout=5000,
                                    settle=15,
                                ),
                            ),
                        ),
                    ),
                    _AssertMaskTask(name="assert_mask", expected=TARGET_MASK),
                    truck,
                ]
            )
        # Sleep before shed when truck slice does not already overnight.
        # Rest truck+sleep / leave+GoToSleep ends D2 morning at bed (136,120).
        # 2026-08-09 (rr-bhr): ExitToFarm from truck D2 bed fires ROM morning
        # intro when event_flags_1f68 lacks 0x00A1 (Y1 has 0x00B1). Free-move
        # clears until dog name entry completes → flags 0x00B1 + free-move.
        # house_size not causal. CompleteOutdoorMorningIntroTask (in shed seq)
        # pure-completes dog name AAAA then shed grass+can.
        if include_sleep and not truck_includes_sleep:
            parts.append(GoToSleepTask(name="sleep_to_d2", timeout=12000))
        if pick_starter_tools:
            # Free grass bag + watering can into carry after D2 morning settle.
            # Required when house_size_at_start==0 (power-on / Gate B). Soft-
            # optional otherwise (AnnEve fixtures). Outdoor intro runs first.
            parts.append(
                _shed_starter_tools(
                    exit_when_done=True,
                    required=bool(require_starter_tools),
                )
            )
    else:
        # Baseline progress run: only the proven outdoor pair.
        parts.append(_AssertMaskTask(name="assert_ann_eve", expected=BIT_ANN | BIT_EVE))
    return SequenceTask(name="town_day1_handoff", tasks=tuple(parts))
