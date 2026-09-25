"""Survival-spine L3 hops through the natural Raft boundary.

The continuous runner attaches these rows, then the carried-bomb boss suffix.
"""

from __future__ import annotations

from zelda_i.route.chain import PredicateStopController
from zelda_i.door_graph import (
    L3_DARKNUTS,
    L3_ENTRY,
    L3_NORTH_ZOLS,
    L3_WEST_KEY,
    LEVEL_3_DOOR_GRAPH,
    InventoryCaps,
    RoomExit,
)
from zelda_i.level3.dungeon import (
    LEVEL3,
    ROOM_5B_SPEC,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_RAFT_PASSAGE,
)
from zelda_i.level3.overworld import (
    LEVEL3_HOPS_FROM_POST_L2,
    POST_L2_PATH_MAX_FRAMES,
    OverworldPostL2ToLevel3Controller,
)
from zelda_i.overworld.armos_rupees import (
    ARMOS_4E_CAVE_ID,
    ARMOS_4E_PAYOUT,
    ARMOS_4E_SCREEN,
    ARMOS_4E_STAND,
    ARMOS_4E_TILE,
    ArmosRupeeController,
)
from zelda_i.overworld.gather_segments import (
    L1_FROM_POND_HOPS,
    L1_POND_HOPS,
    CaveExitController,
    HopWalkController,
)
from zelda_i.overworld.cave_shop import potion_restock_stages
from zelda_i.overworld.heart_farm import PondFairyController
from zelda_i.overworld.graph import ScreenHop
from zelda_i.overworld.settle import PostL2TriforceSettleController
from zelda_i.overworld.settle import POST_L2_SETTLE_MAX_FRAMES
from zelda_i.level3.boss_path import level3_boss_suffix_stages
from zelda_i.level3.path import Level3NorthChainController, Level3WestKeyController
from zelda_i.level3.raft_path import (
    CLEAR_59_MAX_FRAMES,
    CLEAR_69_MAX_FRAMES,
    DOWN_69_MAX_FRAMES,
    LEFT_5B_MAX_FRAMES,
    SPAWN_SETTLE_FRAMES,
    STAIRS_69_MAX_FRAMES,
    PASSAGE_RAFT_MAX_FRAMES,
    Level3RaftPathController,
)
from zelda_i.ram import PASSAGE_MODE
from zelda_i.spine.hops import SpineHop, ready

WEST_KEY_SPINE_MAX_FRAMES = 8000
NORTH_CHAIN_SPINE_MAX_FRAMES = 32000  # 0x6b zols + occupancy north + 0x5b Darknuts
COMPASS_SPINE_MAX_FRAMES = LEFT_5B_MAX_FRAMES + 100
WEST_DARKNUTS_SPINE_MAX_FRAMES = 3000
SOUTH_DARKNUTS_SPINE_MAX_FRAMES = (
    CLEAR_59_MAX_FRAMES + DOWN_69_MAX_FRAMES + SPAWN_SETTLE_FRAMES + 500
)
RAFT_SPINE_MAX_FRAMES = (
    CLEAR_69_MAX_FRAMES
    + STAIRS_69_MAX_FRAMES
    + PASSAGE_RAFT_MAX_FRAMES
    + SPAWN_SETTLE_FRAMES
    + 500
)
_DEST_6B_ROOMS = (L3_ENTRY, L3_WEST_KEY, L3_NORTH_ZOLS, L3_DARKNUTS)

__all__ = [
    "dest_6b_room_plan",
    "level3_dest_6b_stages",
    "level3_dest_6b_success",
    "level3_entrance_tf_stages",
    "l3_hops",
    "run_level3_entrance_tf",
]


def dest_6b_room_plan() -> tuple[RoomExit, ...]:
    """Offline room sequence 0x7c → 0x5b (kill-clear on 0x6b north)."""
    path = LEVEL_3_DOOR_GRAPH.bfs_path(
        L3_ENTRY,
        L3_DARKNUTS,
        InventoryCaps(can_clear=True),
    )
    if path is None:
        raise RuntimeError("L3 door graph has no 0x7c → 0x5b path")
    rooms = (L3_ENTRY, *[edge.target_room for edge in path])
    if rooms != _DEST_6B_ROOMS:
        raise RuntimeError(
            f"L3 dest 0x5b rooms {[hex(r) for r in rooms]} "
            f"!= {[hex(r) for r in _DEST_6B_ROOMS]}"
        )
    return path


def _dest_6b_stages():
    dest_6b_room_plan()
    return (
        ("west_key", Level3WestKeyController(), WEST_KEY_SPINE_MAX_FRAMES),
        (
            "north_chain",
            Level3NorthChainController(),
            NORTH_CHAIN_SPINE_MAX_FRAMES,
        ),
    )


def level3_dest_6b_stages():
    """Public alias for the L3 dest-0x5b stage list (0x7c west key → 0x6b north).

    The ``l3-dest-6b`` :class:`SpineHop` calls :func:`_dest_6b_stages` directly;
    this wrapper is the stable public entry point used by tooling and tests.
    """
    return _dest_6b_stages()


def level3_entrance_tf_stages():
    """Clean fixture-live: Level3Entrance 0x7c → dest 0x5b → bomb shortcut → TF.

    ``route_eligible=False``. No bomb/key poke. Integrator promotes.
    """
    dest_6b_room_plan()
    return (*_dest_6b_stages(), *level3_boss_suffix_stages())


def run_level3_entrance_tf(env, *, assist=None, on_frame=None, frame_base: int = 0):
    """Drive dest-hop stages on an open env. Leave is snapshot leftover."""
    from zelda_i.ram import read_snapshot
    from zelda_i.route.chain import run_controller_stage
    from zelda_i.screen_glance import leftover_from_snapshot

    reports = []
    obs = getattr(env, "last_observation", None)
    frame = frame_base
    failed = None
    for name, controller, max_frames in level3_entrance_tf_stages():
        obs, stage = run_controller_stage(
            env,
            obs,
            name=name,
            controller=controller,
            max_frames=max_frames,
            assist=assist,
            on_frame=on_frame,
            frame_base=frame,
        )
        reports.append(stage.report())
        frame = stage.end_frame
        if not stage.success:
            failed = name
            break
    snap = read_snapshot(env.get_ram())
    leftover = leftover_from_snapshot(snap)
    tf04 = bool(int(snap.triforce) & 0x04)
    deaths = 1 if int(snap.mode) == 17 else 0
    hearts0 = int(snap.filled_hearts) == 0
    return {
        "ok": bool(tf04 and failed is None and deaths == 0 and not hearts0),
        "tf04": tf04,
        "failed_stage": failed,
        "stages": reports,
        "leftover": leftover,
        "deaths": deaths,
        "hearts_lo": int(snap.filled_hearts),
        "hearts_hi": int(snap.heart_containers) - 1,
        "frames": frame,
        "route_eligible": False,
        "natural_entry": False,
        "intervention_class": "clean",
        "obs": obs,
    }


# L3 dest-0x5b spine stop: play mode in a cleared 0x5b Darknuts room. The
# ``l3-dest-6b`` hop encodes this inline via ``ready(..., spec=ROOM_5B_SPEC)``;
# this is the same predicate exposed as a named callable.
level3_dest_6b_success = ready(
    level=LEVEL3, screen=ROOM_L3_DARKNUTS, spec=ROOM_5B_SPEC
)


def _raft_hop(through: str, stop: str, pred, name: str, max_frames: int) -> SpineHop:
    def stages():
        return (
            (
                stop,
                PredicateStopController(Level3RaftPathController(), pred, name),
                max_frames,
            ),
        )

    return SpineHop(through, stop, stages, pred)


# Index of the 0x59 hop on the post-L2 walk (the pond detour's branch).
_L3_WALK_59 = [hop.target for hop in LEVEL3_HOPS_FROM_POST_L2].index(0x59)
_L3_FROM_POND_HOPS = L1_FROM_POND_HOPS[:2] + LEVEL3_HOPS_FROM_POST_L2[_L3_WALK_59 + 1 :]
# Rupees kept past a pre-L3 potion. The pond leaves ~88R; L3 and L4 pay ~40
# more before the 80R arrows at 0x4A on the L4 -> L5 walk, which a bomb pack
# at 0x44 can make short (open: the arrow buy should not block L5).
L3_POTION_RESERVE = 40


def l3_hops(*, after_entry=None) -> tuple[SpineHop, ...]:
    """Entry → dest 0x5b → compass → west/south Darknuts → Raft."""
    compass = ready(level=LEVEL3, screen=0x5A)
    west = ready(level=LEVEL3, screen=0x59)
    south = ready(level=LEVEL3, screen=0x69)
    raft = ready(
        level=LEVEL3, screen=ROOM_L3_RAFT_PASSAGE, mode=PASSAGE_MODE, item="raft"
    )
    return (
        SpineHop(
            "l3-entry",
            "enter_level3",
            (
                (
                    "settle_l2_tf",
                    PostL2TriforceSettleController(),
                    POST_L2_SETTLE_MAX_FRAMES,
                ),
                (
                    "walk_armos_3d",
                    HopWalkController(
                        hops=LEVEL3_HOPS_FROM_POST_L2[:2]
                        + (ScreenHop(0x3D, "UP", align_x=120),),
                        waypoints={},
                        max_frames=10000,
                    ),
                    10000,
                ),
                ("armos_rupees_3d", ArmosRupeeController(), 5000),
                ("exit_armos_3d", CaveExitController(clear=0), 600),
                (
                    "return_4d_from_armos",
                    HopWalkController(
                        hops=(ScreenHop(0x4D, "DOWN", align_x=120),),
                        waypoints={},
                        max_frames=5000,
                    ),
                    5000,
                ),
                (
                    "walk_armos_4e",
                    HopWalkController(
                        hops=(ScreenHop(0x4E, "RIGHT"),),
                        waypoints={},
                        max_frames=5000,
                    ),
                    5000,
                ),
                (
                    "armos_rupees_4e",
                    ArmosRupeeController(
                        screen=ARMOS_4E_SCREEN,
                        stand=ARMOS_4E_STAND,
                        face="RIGHT",
                        armos=ARMOS_4E_TILE,
                        cave_id=ARMOS_4E_CAVE_ID,
                        payout=ARMOS_4E_PAYOUT,
                    ),
                    5000,
                ),
                ("exit_armos_4e", CaveExitController(clear=0), 600),
                (
                    "return_4d_from_4e",
                    HopWalkController(
                        hops=(ScreenHop(0x4D, "LEFT"),),
                        waypoints={},
                        max_frames=5000,
                    ),
                    5000,
                ),
                # The 0x39 pond is two screens up from 0x59 on this walk: L3
                # is entered full (Clean L3 from 4.5/7 died 8/8 offsets).
                (
                    "walk_pond_l3",
                    OverworldPostL2ToLevel3Controller(
                        hops=LEVEL3_HOPS_FROM_POST_L2[2:_L3_WALK_59 + 1] + L1_POND_HOPS[3:],
                        require_dungeon=False,
                    ),
                    POST_L2_PATH_MAX_FRAMES,
                ),
                ("pond_39_l3", PondFairyController(), 3000),
                # And a potion from 0x64 on the way in when the wallet can
                # spare it past the L5 arrows: Manhandla killed 2.2-heart
                # Links (Clean power-on 66/67/68).
                *potion_restock_stages(_L3_FROM_POND_HOPS, "l2", reserve=L3_POTION_RESERVE),
                (
                    "enter_level3",
                    OverworldPostL2ToLevel3Controller(
                        hops=_L3_FROM_POND_HOPS,
                        require_dungeon=True,
                        resume_on_screen=True,
                    ),
                    POST_L2_PATH_MAX_FRAMES,
                ),
            ),
            ready(level=LEVEL3, screen=ROOM_L3_ENTRY),
            after=after_entry,
        ),
        SpineHop(
            "l3-dest-6b",
            "north_chain",
            _dest_6b_stages,
            ready(level=LEVEL3, screen=ROOM_L3_DARKNUTS, spec=ROOM_5B_SPEC),
        ),
        _raft_hop(
            "l3-compass",
            "compass_0x5a",
            compass,
            "level3_compass_0x5a",
            COMPASS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-west-darknuts",
            "west_darknuts_0x59",
            west,
            "level3_west_darknuts_0x59",
            WEST_DARKNUTS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-south-darknuts",
            "south_darknuts_0x69",
            south,
            "level3_south_darknuts_0x69",
            SOUTH_DARKNUTS_SPINE_MAX_FRAMES,
        ),
        _raft_hop(
            "l3-raft",
            "raft_0x0f",
            raft,
            "level3_raft",
            RAFT_SPINE_MAX_FRAMES,
        ),
    )
