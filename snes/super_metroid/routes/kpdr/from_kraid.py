"""Kraid return reverse hops (Eye → Baby → Kihunter → Zeela → Warehouse).

Several hops on this spine are locked ``continuous`` on the Business /
Frog Save tip (see ``SPEED_GRAPH``). Controllers stay pure — no
env ownership, door-warps, or progression writes.

Climb / door / morph-bomb open-loops live as named skills under
:mod:`super_metroid.routes.skills.kraid_return` and
:mod:`super_metroid.routes.skills.door_exit` /
:mod:`super_metroid.routes.skills.morph_bomb`. Product hops here only
compose phases and keep hop signatures stable.
"""

from __future__ import annotations

from super_metroid.policy import StateRequirement
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    require_state,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ROOM_BABY_KRAID,
    ROOM_KRAID_EYE,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.door_exit import (
    beam_open_door,
    jump_enter_exit,
    lip_stage,
)
from super_metroid.routes.skills.kraid_return import (
    eye_mid_room_approach,
    kihunter_bomb_hole,
    kihunter_mid_ledge,
    kihunter_upper_to_zeela_window,
    kihunter_wall_plant,
    zeela_below_platform_lip,
    zeela_bottom_roll,
    zeela_mid_platform,
    zeela_shotblock_wall_climb,
    zeela_wall_plant,
    zeela_warehouse_door_exit,
)


def _play_return(
    session: ControllerSession,
    *,
    source_room: int,
    target_room: int,
    direction: str,
    label: str,
    spin: bool = False,
) -> SuperMetroidState:
    require_state(
        session,
        StateRequirement(room_id=source_room, game_states=frozenset({8})),
        label,
    )
    require_room(session, source_room, label)
    select_weapon(session, 0)

    buttons = (direction, "B", "A") if spin else (direction, "A")
    for _ in range(900):
        state = hold(session, 1, *buttons, reason=f"{label}_exit")
        if state.room_id == target_room:
            break
    else:
        raise TimeoutError(f"{label}: exit timed out: {session.state}")

    return wait_ordinary_room(
        session,
        target_room,
        settle_frames=320,
        label=label,
    )


def play_eye_to_baby_return(session: ControllerSession) -> SuperMetroidState:
    """Left return from Kraid Eye Door Room to Baby Kraid (post pure K3.3).

    Controller-dev reverse hop. Not continuous evidence.

    Geometry (SM-K4-R-01B): walk-only / floor spin pins mid-room (~x=373,
    pose 138) on a ledge. Clear with sustained jump-left, open the left blue
    hatch with standing beams near the lip, then jump-enter. Source:
    ``scratch/post_kraid_to_eye_return.state`` (0xA56B).
    """
    require_state(
        session,
        StateRequirement(room_id=ROOM_KRAID_EYE, game_states=frozenset({8})),
        "eye_to_baby_return",
    )
    require_room(session, ROOM_KRAID_EYE, "eye_to_baby_return")
    select_weapon(session, 0)

    # Jump-left across the mid-room ledge that floor-walk pins at x≈373.
    eye_mid_room_approach(session, "eye_to_baby_return")

    # Stage just right of the left lip; open blue hatch with X-only beams.
    lip_stage(
        session,
        label="eye_to_baby",
        backoff="RIGHT",
        face="LEFT",
        backoff_frames=8,
        face_frames=8,
        release_frames=6,
        settle_frames=8,
    )
    beam_open_door(
        session,
        label="eye_to_baby",
        shots=6,
        shot_frames=4,
        fuse_frames=14,
    )
    jump_enter_exit(
        session,
        ROOM_BABY_KRAID,
        label="eye_to_baby",
        direction="LEFT",
        max_frames=700,
        transition_drain=80,
    )

    return wait_ordinary_room(
        session,
        ROOM_BABY_KRAID,
        settle_frames=320,
        label="eye_to_baby_return",
    )


def play_baby_to_kihunter_return(session: ControllerSession) -> SuperMetroidState:
    """Left return Baby Kraid → Warehouse Kihunter (post pure eye→baby).

    Controller-dev reverse hop. Not continuous evidence.

    The left hatch is **gray** with ``clear_room_enemies`` (graph node Baby
    Kraid Left Door). Floor spin alone pins at the shell (x≈37–69, pose 138,
    door_transition=0). Clear pirates/Mini-Kraid first, then beam-open and
    exit left. Source: ``scratch/post_eye_to_baby_return.state``.
    """
    from super_metroid.routes.kpdr.to_kraid import _baby_kraid_sweep

    require_state(
        session,
        StateRequirement(room_id=ROOM_BABY_KRAID, game_states=frozenset({8})),
        "baby_to_kihunter_return",
    )
    require_room(session, ROOM_BABY_KRAID, "baby_to_kihunter_return")
    # Supers match play_baby_kraid_to_eye — beams alone leave Mini-Kraid alive
    # and the gray left hatch stays locked (clear_room_enemies).
    select_weapon(session, 2)

    # Land from elevated eye-door entry if needed, then clear the room lock.
    for _ in range(30):
        state = hold(session, 1, reason="baby_return_land")
        if state.velocity_y == 0 and state.pose in (1, 2, 5, 6, 9, 10, 137, 138):
            break
    _baby_kraid_sweep(session, "LEFT", 80, limit=1700, label="baby_return_clear_left")
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(
            session, "RIGHT", 1490, limit=1900, label="baby_return_clear_right"
        )
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(
            session, "LEFT", 80, limit=1900, label="baby_return_clear_left2"
        )

    # Stage near left gray door; beams open the unlocked shell.
    select_weapon(session, 0)
    for _ in range(200):
        state = hold(session, 1, "LEFT", "B", reason="baby_return_door_approach")
        if state.samus_x <= 120:
            break
    lip_stage(
        session,
        label="baby_return",
        backoff="RIGHT",
        face="LEFT",
        backoff_frames=10,
        face_frames=8,
        release_frames=6,
    )
    beam_open_door(
        session,
        label="baby_return",
        shots=6,
        shot_frames=4,
        fuse_frames=14,
    )
    jump_enter_exit(
        session,
        ROOM_WAREHOUSE_KIHUNTER,
        label="baby_return",
        direction="LEFT",
        max_frames=700,
        transition_drain=80,
    )

    return wait_ordinary_room(
        session,
        ROOM_WAREHOUSE_KIHUNTER,
        settle_frames=320,
        label="baby_to_kihunter_return",
    )


def play_kihunter_to_zeela_return(session: ControllerSession) -> SuperMetroidState:
    """Climb out of the lower Kihunter alcove and drop through Zeela's door.

    This is not continuous evidence; it requires a natural source state.

    Redesign (Wave 9 / SM-K4-R-CLIMB-REDESIGN) — room geometry:

    * Floor hard wall at **x≈357** blocks walking under the hole.
    * Mid ledge at **y≈299, x≈367–379** is reached by wall-plant Hi-Jump then
      RIGHT drift at apex.
    * Forward bomb hole is exactly **x≈376** (upper floor y≈171). From the mid
      ledge, **morph bomb-jump** through that hole, then left to the Zeela
      down-door window **x∈[96,160]**.
    """
    label = "kihunter_to_zeela_return"
    require_state(
        session,
        StateRequirement(
            room_id=ROOM_WAREHOUSE_KIHUNTER,
            game_states=frozenset({8}),
        ),
        label,
    )
    require_room(session, ROOM_WAREHOUSE_KIHUNTER, label)
    select_weapon(session, 0)
    unmorph(session)

    best_min_y = [session.state.samus_y]
    kihunter_wall_plant(session, label)
    kihunter_mid_ledge(session, label, best_min_y=best_min_y)
    kihunter_bomb_hole(session, label, best_min_y=best_min_y)
    kihunter_upper_to_zeela_window(session, label)

    return wait_ordinary_room(
        session,
        ROOM_ZEELA,
        settle_frames=320,
        label=label,
        # The down-door transition becomes ordinary while Samus is still
        # falling.  Do not hand its successor the airborne x≈357/y≈362 frame:
        # Zeela's reverse controller needs the real floor handoff near y=395.
        y_range=(385, 410),
    )


def play_zeela_to_warehouse_return(session: ControllerSession) -> SuperMetroidState:
    """Climb Zeela reverse to the upper-left Warehouse door.

    Continuous on the Business / Frog Save tip (graph verification
    ``continuous``). Pure probes still need a continuous-like source state.

    Phases (named; one-knob geometry edits stay inside a single phase):

    1. ``bottom_roll`` — morph reverse-roll + align second-drop lane
    2. ``mid_platform`` — reverse-shot climb onto middle platform
    3. ``below_platform_lip`` — crouch-load from mid right-edge to lip
    4. ``wall_plant`` — hop-left to wall plant band (x≈37, y≈219)
    5. ``shotblock_clear`` — stand + UP+X clear reverse-entry shot blocks
    6. ``wall_replant`` — LEFT+B re-plant on wall after clear
    7. ``wall_spin_climb`` — wall-spin cadence to top door band (y≤150)
    8. ``warehouse_door_exit`` — standing LEFT beams into Warehouse 0xA6A1

    Steps 5–7 are composed by ``zeela_shotblock_wall_climb`` (retry wrapper).

    Geometry facts (SM-K4-R-ZEELA-REDESIGN):

    * Historical fixtures begin bottom-right after the Kihunter drop (~x=403)
      without Hi-Jump (``items=0x1005``); the natural Varia lineage has
      Hi-Jump and needs a farther-right mid-platform landing gate.
    * Floor-left is the Energy Tank door ``0xA4B1`` — fail-loud if ``y>250``.
    * Middle platform is narrow (~x=90–107, y≈331). Reach it with reverse-shot
      **RIGHT bias in the hole** (left-only peaks the shaft at x≈52 and falls).
    * Below-platform lip (~x=40–70, y≈219–235) is reached by crouch-load from
      mid right-edge (x≈107), then LEFT drift — not by floor-left spam.
    * Lip crouch-load lands ~x=69 (LEFT walk blocked); hop-left to plant
      x≈37 y≈219.
    * Reverse re-entry leaves shot blocks solid above the left wall — clear
      with ~40 UP+X before wall-spin climb. Hard cap without clear: y≈188.
    * Top door band is **y≈112–139**. Stop only on grounded standing/crouch
      at y≤150 (pose 1/2/137/138), not walljump pose 26.
    * Warehouse door opens with standing LEFT beams from that top band.
    """
    label = "zeela_to_warehouse_return"
    require_state(
        session,
        StateRequirement(room_id=ROOM_ZEELA, game_states=frozenset({8})),
        label,
    )
    require_room(session, ROOM_ZEELA, label)
    select_weapon(session, 0)

    zeela_bottom_roll(session, label)
    zeela_mid_platform(session, label)
    zeela_below_platform_lip(session, label)
    zeela_wall_plant(session, label)
    early = zeela_shotblock_wall_climb(session, label)
    if early is not None:
        return early
    zeela_warehouse_door_exit(session, label)

    return wait_ordinary_room(
        session,
        ROOM_WAREHOUSE,
        settle_frames=320,
        label=label,
    )
